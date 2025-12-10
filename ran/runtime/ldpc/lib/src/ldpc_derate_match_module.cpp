/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <any>
#include <cstddef>
#include <cstdint>
#include <format>
#include <iterator>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <cuphy.h>
#include <cuphy.hpp>
#include <driver_types.h>
#include <quill/LogMacros.h>
#include <tensor_desc.hpp>

#include <gsl-lite/gsl-lite.hpp>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "ldpc/ldpc_derate_match_module.hpp"
#include "ldpc/ldpc_log.hpp"
#include "ldpc/ldpc_params.hpp"
#include "ldpc/outer_rx_params.hpp"
#include "log/rt_log_macros.hpp"
#include "pipeline/igraph.hpp"
#include "pipeline/kernel_descriptor_accessor.hpp"
#include "pipeline/kernel_launch_config.hpp"
#include "pipeline/types.hpp"
#include "tensor/data_types.hpp"
#include "tensor/tensor_info.hpp"
#include "utils/error_macros.hpp"

namespace ran::ldpc {

LdpcDerateMatchModule::LdpcDerateMatchModule(std::string instance_id, const std::any &init_params)
        : instance_id_(std::move(instance_id)) {

    RT_LOGC_DEBUG(LdpcComponent::DerateMatch, "Constructing module {}", instance_id_);
    try {
        config_ = std::any_cast<StaticParams>(init_params);
    } catch (const std::bad_any_cast &e) {
        const std::string error_message = "Invalid initialization parameters!";
        RT_LOGC_ERROR(LdpcComponent::DerateMatch, "{}", error_message);
        throw std::invalid_argument(error_message);
    }

    constexpr int FP_CONFIG = 3; // FP16 in, FP16 out
    const cuphyStatus_t status = cuphyCreatePuschRxRateMatch(
            &pusch_rm_hndl_, FP_CONFIG, static_cast<int>(config_.enable_scrambling));
    if (status != CUPHY_STATUS_SUCCESS) {
        const std::string error_message = "Failed to create LDPC derate match object!";
        RT_LOGC_ERROR(LdpcComponent::DerateMatch, "{}", error_message);
        throw std::runtime_error(error_message);
    }

    FRAMEWORK_CUDA_RUNTIME_CHECK_THROW(cudaHostAlloc(
            &de_rm_output_,
            config_.max_num_tbs * sizeof(std::byte *),
            cudaHostAllocPortable | cudaHostAllocMapped));

    t_prm_llr_vec_.reserve(config_.max_num_ue_grps);
    t_prm_llr_cdm1_vec_.reserve(config_.max_num_ue_grps);
    t_desc_llr_vec_.reserve(config_.max_num_ue_grps);
    t_desc_llr_cdm1_vec_.reserve(config_.max_num_ue_grps);
}

LdpcDerateMatchModule::~LdpcDerateMatchModule() {
    if (const cuphyStatus_t status = cuphyDestroyPuschRxRateMatch(pusch_rm_hndl_);
        status != CUPHY_STATUS_SUCCESS) {
        const std::string error_message = "Failed to destroy LDPC derate match object!";
        RT_LOGC_ERROR(LdpcComponent::DerateMatch, "{}", error_message);
    }
    if (de_rm_output_ != nullptr) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        cudaFreeHost(reinterpret_cast<void *>(de_rm_output_));
    }
}

std::string_view LdpcDerateMatchModule::get_instance_id() const { return instance_id_; }

void LdpcDerateMatchModule::setup_memory(
        const framework::pipeline::ModuleMemorySlice &memory_slice) {
    memory_slice_ = memory_slice;

    // Setup output port info (tensors are set later in configure_io).
    framework::pipeline::PortInfo output_port;
    output_port.name = "derate_matched_llrs";
    output_port.tensors.reserve(config_.max_num_tbs);
    outputs_.push_back(output_port);

    kernel_desc_mgr_ =
            std::make_unique<framework::pipeline::KernelDescriptorAccessor>(memory_slice);
}

std::vector<framework::tensor::TensorInfo>
LdpcDerateMatchModule::get_input_tensor_info(std::string_view port_name) const {

    if (const auto it = std::find_if(
                inputs_.begin(),
                inputs_.end(),
                [&port_name](const framework::pipeline::PortInfo &port) {
                    return port.name == port_name;
                });
        it != inputs_.end()) {
        std::vector<framework::tensor::TensorInfo> result;
        result.reserve(it->tensors.size());
        std::transform(
                it->tensors.begin(),
                it->tensors.end(),
                std::back_inserter(result),
                [](const auto &tensor) { return tensor.tensor_info; });
        return result;
    }
    const std::string error_message =
            std::format("Port info not found for input port {}", port_name);
    RT_LOGC_ERROR(LdpcComponent::DerateMatch, "{}", error_message);
    throw std::invalid_argument(error_message);
}

std::vector<framework::tensor::TensorInfo>
LdpcDerateMatchModule::get_output_tensor_info(std::string_view port_name) const {

    if (!setup_complete_) {
        const std::string error_message =
                "Called get_output_tensor_info() before setup is complete";
        RT_LOGC_ERROR(LdpcComponent::DerateMatch, "{}", error_message);
        throw std::runtime_error(error_message);
    }

    if (const auto it = std::find_if(
                outputs_.begin(),
                outputs_.end(),
                [&port_name](const framework::pipeline::PortInfo &port) {
                    return port.name == port_name;
                });
        it != outputs_.end()) {
        std::vector<framework::tensor::TensorInfo> result;
        result.reserve(it->tensors.size());
        std::transform(
                it->tensors.begin(),
                it->tensors.end(),
                std::back_inserter(result),
                [](const auto &tensor) { return tensor.tensor_info; });
        return result;
    }

    const std::string error_message =
            std::format("Port info not found for output port {}", port_name);
    RT_LOGC_ERROR(LdpcComponent::DerateMatch, "{}", error_message);
    throw std::invalid_argument(error_message);
}

std::vector<std::string> LdpcDerateMatchModule::get_input_port_names() const {
    return {"llrs", "llrs_cdm1"};
}

std::vector<std::string> LdpcDerateMatchModule::get_output_port_names() const {
    return {"derate_matched_llrs"};
}

void LdpcDerateMatchModule::set_inputs(std::span<const framework::pipeline::PortInfo> inputs) {
    // Validate inputs
    if (inputs.size() != 2) {
        const std::string error_message = "Expected 2 inputs";
        RT_LOGC_ERROR(LdpcComponent::DerateMatch, "{}", error_message);
        throw std::invalid_argument(error_message);
    }

    // Find llrs and llrsCdm1 inputs
    const framework::pipeline::PortInfo *llrs_input{};
    const framework::pipeline::PortInfo *llrs_cdm1_input{};

    for (const auto &input : inputs) {
        if (input.name == "llrs") {
            llrs_input = &input;
        } else if (input.name == "llrs_cdm1") {
            llrs_cdm1_input = &input;
        } else {
            const std::string error_message = std::format("Unknown port {}", input.name);
            RT_LOGC_ERROR(LdpcComponent::DerateMatch, "{}", error_message);
            throw std::invalid_argument(error_message);
        }
    }

    // Validate that both required inputs are present
    if (llrs_input == nullptr) {
        const std::string error_message = "Missing required input port 'llrs'";
        RT_LOGC_ERROR(LdpcComponent::DerateMatch, "{}", error_message);
        throw std::invalid_argument(error_message);
    }
    if (llrs_cdm1_input == nullptr) {
        const std::string error_message = "Missing required input port 'llrs_cdm1'";
        RT_LOGC_ERROR(LdpcComponent::DerateMatch, "{}", error_message);
        throw std::invalid_argument(error_message);
    }

    // Validate that input LLRs have been set
    for (const auto &tensor : llrs_input->tensors) {

        if (tensor.device_ptr == nullptr) {
            const std::string error_message = "Input tensor not set for port 'llrs'";
            RT_LOGC_ERROR(LdpcComponent::DerateMatch, "{}", error_message);
            throw std::invalid_argument(error_message);
        }
        if (tensor.tensor_info.get_type() != framework::tensor::TensorInfo::DataType::TensorR16F) {
            const std::string error_message = "Input LLRs must be TensorR16F for port 'llrs'";
            RT_LOGC_ERROR(LdpcComponent::DerateMatch, "{}", error_message);
            throw std::invalid_argument(error_message);
        }
    }

    for (const auto &tensor : llrs_cdm1_input->tensors) {

        if (tensor.device_ptr == nullptr) {
            const std::string error_message = "Input tensor not set for port 'llrs_cdm1'";
            RT_LOGC_ERROR(LdpcComponent::DerateMatch, "{}", error_message);
            throw std::invalid_argument(error_message);
        }

        if (tensor.tensor_info.get_type() != framework::tensor::TensorInfo::DataType::TensorR16F) {
            const std::string error_message = "Input LLRs must be TensorR16F for port 'llrs_cdm1'";
            RT_LOGC_ERROR(LdpcComponent::DerateMatch, "{}", error_message);
            throw std::invalid_argument(error_message);
        }
    }

    inputs_.assign({*llrs_input, *llrs_cdm1_input});
}

std::vector<framework::pipeline::PortInfo> LdpcDerateMatchModule::get_outputs() const {
    if (!setup_complete_) {
        const std::string error_message = "Called before setup completion";
        RT_LOGC_ERROR(LdpcComponent::DerateMatch, "{}", error_message);
        throw std::runtime_error(error_message);
    }
    return outputs_;
}

void LdpcDerateMatchModule::configure_io(
        const framework::pipeline::DynamicParams &params, cudaStream_t stream) {

    try {
        pusch_outer_rx_params_ = std::any_cast<PuschOuterRxParams>(params.module_specific_params);
    } catch (const std::bad_any_cast &e) {
        const std::string error_message = "Invalid dynamic parameters!";
        RT_LOGC_ERROR(LdpcComponent::DerateMatch, "{}", error_message);
        throw std::invalid_argument(error_message);
    }

    // Update output tensor info and device pointers.
    const std::size_t num_tbs = pusch_outer_rx_params_.value().num_tbs();
    if (num_tbs > config_.max_num_tbs) {
        const std::string error_message =
                "Number of TBs in dynamic parameters exceeds maximum number of TBs";
        RT_LOGC_ERROR(LdpcComponent::DerateMatch, "{}", error_message);
        throw std::invalid_argument(error_message);
    }

    outputs_[0].tensors.resize(num_tbs);
    for (std::size_t tb_idx = 0; tb_idx < num_tbs; tb_idx++) {
        const auto offset = tb_idx * config_.max_num_rm_llrs_per_cb * config_.max_num_cbs_per_tb;

        void *device_ptr =
                std::next(memory_slice_.device_tensor_ptr, static_cast<std::ptrdiff_t>(offset));

        const auto circular_buffer_size_padded =
                pusch_outer_rx_params_.value()[tb_idx].ldpc_params().circular_buffer_size_padded();
        const auto num_code_blocks =
                pusch_outer_rx_params_.value()[tb_idx].ldpc_params().num_code_blocks();

        outputs_[0].tensors[tb_idx] = framework::pipeline::DeviceTensor{
                .device_ptr = device_ptr,
                .tensor_info = framework::tensor::TensorInfo(
                        framework::tensor::TensorInfo::DataType::TensorR16F,
                        {static_cast<std::size_t>(circular_buffer_size_padded),
                         static_cast<std::size_t>(num_code_blocks)})};

        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        de_rm_output_[tb_idx] = device_ptr;
    }

    PerTbParams *p_tb_prms_cpu = pusch_outer_rx_params_.value().get_per_tb_params_cpu_ptr();
    PerTbParams *p_tb_prms_gpu = pusch_outer_rx_params_.value().get_per_tb_params_gpu_ptr();
    const auto sch_user_idxs_span = pusch_outer_rx_params_.value().get_sch_user_idxs();
    std::vector<std::uint16_t> sch_user_idxs(sch_user_idxs_span.begin(), sch_user_idxs_span.end());
    const auto n_sch_ues = static_cast<std::uint16_t>(sch_user_idxs.size());

    const std::size_t num_ue_grps = inputs_[0].tensors.size();
    t_desc_llr_vec_.resize(num_ue_grps);
    t_desc_llr_cdm1_vec_.resize(num_ue_grps);
    t_prm_llr_vec_.resize(num_ue_grps);
    t_prm_llr_cdm1_vec_.resize(num_ue_grps);

    const auto llrs_tensor_infos = get_input_tensor_info("llrs");
    const auto llrs_cdm1_tensor_infos = get_input_tensor_info("llrs_cdm1");

    for (std::size_t ue_grp_idx = 0; ue_grp_idx < num_ue_grps; ue_grp_idx++) {

        set_tensor_descriptor(t_desc_llr_vec_[ue_grp_idx], llrs_tensor_infos[ue_grp_idx]);
        set_tensor_descriptor(t_desc_llr_cdm1_vec_[ue_grp_idx], llrs_cdm1_tensor_infos[ue_grp_idx]);

        t_prm_llr_vec_[ue_grp_idx].desc = t_desc_llr_vec_[ue_grp_idx].handle();
        t_prm_llr_vec_[ue_grp_idx].pAddr = inputs_[0].tensors[ue_grp_idx].device_ptr;

        t_prm_llr_cdm1_vec_[ue_grp_idx].desc = t_desc_llr_cdm1_vec_[ue_grp_idx].handle();
        t_prm_llr_cdm1_vec_[ue_grp_idx].pAddr = inputs_[1].tensors[ue_grp_idx].device_ptr;
    }

    constexpr std::uint8_t ENABLE_CPU_TO_GPU_DESCR_ASYNC_CPY = 0;
    const cuphyStatus_t status = cuphySetupPuschRxRateMatch(
            pusch_rm_hndl_,
            n_sch_ues,
            sch_user_idxs.data(),
            p_tb_prms_cpu,
            p_tb_prms_gpu,
            t_prm_llr_vec_.data(),
            t_prm_llr_cdm1_vec_.data(),
            de_rm_output_,
            memory_slice_.dynamic_kernel_descriptor_cpu_ptr,
            memory_slice_.dynamic_kernel_descriptor_gpu_ptr,
            ENABLE_CPU_TO_GPU_DESCR_ASYNC_CPY,
            &kernel_launch_cfg_,
            stream);

    if (status != CUPHY_STATUS_SUCCESS) {
        const std::string error_message = "Failed to setup LDPC derate match";
        RT_LOGC_ERROR(LdpcComponent::DerateMatch, "{}", error_message);
        throw std::runtime_error(error_message);
    }

    setup_complete_ = true;
}

void LdpcDerateMatchModule::set_tensor_descriptor(
        cuphy::tensor_desc &desc, const framework::tensor::TensorInfo &tensor_info) {

    const std::vector<std::size_t> &llrs_dims = tensor_info.get_dimensions();

    if (llrs_dims.size() != 4) {
        const std::string error_message = "Input LLRs must have 4 dimensions";
        RT_LOGC_ERROR(LdpcComponent::DerateMatch, "{}", error_message);
        throw std::invalid_argument(error_message);
    }

    const auto qam_stride = static_cast<int>(llrs_dims[0]);
    const auto num_layers = static_cast<int>(llrs_dims[1]);
    const auto num_subc = static_cast<int>(llrs_dims[2]);
    const auto num_data_syms = static_cast<int>(llrs_dims[3]);
    desc.set(
            CUPHY_R_16F,
            qam_stride,
            num_layers,
            num_subc,
            num_data_syms,
            cuphy::tensor_flags::align_tight);
}

void LdpcDerateMatchModule::execute(cudaStream_t stream) {

    if (!setup_complete_) {
        const std::string error_message = "execute() called before setup completion";
        RT_LOGC_ERROR(LdpcComponent::DerateMatch, "{}", error_message);
        throw std::runtime_error(error_message);
    }

    RT_LOGC_DEBUG(
            LdpcComponent::DerateMatch,
            "Executing module {} on stream {}",
            instance_id_,
            static_cast<void *>(stream));

    const CUDA_KERNEL_NODE_PARAMS &kernel_node_params_driver =
            kernel_launch_cfg_.kernelNodeParamsDriver;
    FRAMEWORK_CUDA_DRIVER_CHECK_THROW(
            framework::pipeline::launch_kernel(kernel_node_params_driver, stream));
}

framework::pipeline::ModuleMemoryRequirements LdpcDerateMatchModule::get_requirements() const {
    std::size_t dyn_descr_size_bytes{};
    std::size_t dyn_descr_align_bytes{};
    if (const cuphyStatus_t status =
                cuphyPuschRxRateMatchGetDescrInfo(&dyn_descr_size_bytes, &dyn_descr_align_bytes);
        status != CUPHY_STATUS_SUCCESS) {
        const std::string error_message = "Failed to get workspace size for LDPC derate match";
        RT_LOGC_ERROR(LdpcComponent::DerateMatch, "{}", error_message);
        throw std::runtime_error(error_message);
    }

    static constexpr std::size_t NUM_BYTES_PER_LLR = sizeof(__half);
    const std::size_t n_bytes = NUM_BYTES_PER_LLR * config_.max_num_rm_llrs_per_cb *
                                config_.max_num_cbs_per_tb * config_.max_num_tbs;

    framework::pipeline::ModuleMemoryRequirements req;
    req.static_kernel_descriptor_bytes = 0;
    req.dynamic_kernel_descriptor_bytes = dyn_descr_size_bytes;
    req.device_tensor_bytes = n_bytes;
    req.alignment = dyn_descr_align_bytes;
    return req;
}

std::span<const CUgraphNode> LdpcDerateMatchModule::add_node_to_graph(
        gsl_lite::not_null<framework::pipeline::IGraph *> graph,
        const std::span<const CUgraphNode> deps) {

    RT_LOGC_DEBUG(
            LdpcComponent::DerateMatch,
            "Adding kernel node to graph with {} dependencies",
            deps.size());

    // Add kernel node using kernel params from kernel_config_
    graph_node_ = graph->add_kernel_node(deps, kernel_launch_cfg_.kernelNodeParamsDriver);
    if (graph_node_ == nullptr) {
        const std::string error_message = "Failed to add kernel node to graph";
        RT_LOGC_ERROR(LdpcComponent::DerateMatch, "{}", error_message);
        throw std::runtime_error(error_message);
    }

    RT_LOGC_DEBUG(
            LdpcComponent::DerateMatch, "Kernel node added: {}", static_cast<void *>(graph_node_));

    return {&graph_node_, 1};
}

void LdpcDerateMatchModule::update_graph_node_params(
        CUgraphExec exec, [[maybe_unused]] const framework::pipeline::DynamicParams &params) {
    const auto &kernel_params = kernel_launch_cfg_.kernelNodeParamsDriver;
    FRAMEWORK_CUDA_DRIVER_CHECK_THROW(
            cuGraphExecKernelNodeSetParams(exec, graph_node_, &kernel_params));

    RT_LOGC_DEBUG(LdpcComponent::DerateMatch, "Graph node params updated");
}

} // namespace ran::ldpc
