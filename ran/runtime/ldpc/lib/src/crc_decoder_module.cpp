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
#include <array>
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
#include <driver_types.h>
#include <quill/LogMacros.h>

#include <gsl-lite/gsl-lite.hpp>

#include <cuda.h>

#include "ldpc/crc_decoder_module.hpp"
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

static constexpr int BITS_PER_BYTE = 8;

CrcDecoderModule::CrcDecoderModule(std::string instance_id, const std::any &init_params)
        : instance_id_(std::move(instance_id)), crc_launch_cfgs_{} {

    RT_LOGC_DEBUG(LdpcComponent::CrcDecoder, "Constructing module {}", instance_id_);

    try {
        static_params_ = std::any_cast<StaticParams>(init_params);
    } catch (const std::bad_any_cast &) {
        const std::string error_message = "Invalid initialization parameters for CrcDecoderModule";
        RT_LOGC_ERROR(LdpcComponent::CrcDecoder, "{}", error_message);
        throw std::invalid_argument(error_message);
    }

    if (const cuphyStatus_t status = cuphyCreatePuschRxCrcDecode(
                &crc_decode_hndl_, static_cast<int>(static_params_.reverse_bytes));
        status != CUPHY_STATUS_SUCCESS) {
        const std::string error_message = "Failed to create CRC decode object!";
        RT_LOGC_ERROR(LdpcComponent::CrcDecoder, "{}", error_message);
        throw std::runtime_error(error_message);
    }

    tb_payload_offsets_.reserve(static_params_.max_num_tbs);
    cb_crc_offsets_.reserve(static_params_.max_num_tbs);
}

CrcDecoderModule::~CrcDecoderModule() {
    if (const cuphyStatus_t status = cuphyDestroyPuschRxCrcDecode(crc_decode_hndl_);
        status != CUPHY_STATUS_SUCCESS) {
        const std::string error_message = "Failed to destroy CRC decode object!";
        RT_LOGC_ERROR(LdpcComponent::CrcDecoder, "{}", error_message);
    }
}

std::string_view CrcDecoderModule::get_instance_id() const { return instance_id_; }

void CrcDecoderModule::setup_memory(const framework::pipeline::ModuleMemorySlice &memory_slice) {
    memory_slice_ = memory_slice;

    // Separate outputs per TB.
    // The memory for each tensor is dynamically set in configure_io
    // from the memory slice allocated by the PipelineMemoryManager.

    framework::pipeline::PortInfo cb_crcs_port;
    cb_crcs_port.name = "cb_crcs";
    cb_crcs_port.tensors.reserve(static_params_.max_num_tbs);
    outputs_.push_back(cb_crcs_port);

    framework::pipeline::PortInfo tb_crcs_port;
    tb_crcs_port.name = "tb_crcs";
    tb_crcs_port.tensors.reserve(static_params_.max_num_tbs);
    outputs_.push_back(tb_crcs_port);

    framework::pipeline::PortInfo tb_payloads_port;
    tb_payloads_port.name = "tb_payloads";
    tb_payloads_port.tensors.reserve(static_params_.max_num_tbs);
    outputs_.push_back(tb_payloads_port);

    kernel_desc_mgr_ =
            std::make_unique<framework::pipeline::KernelDescriptorAccessor>(memory_slice);
}

std::vector<framework::tensor::TensorInfo>
CrcDecoderModule::get_input_tensor_info(std::string_view port_name) const {

    const auto it = std::find_if(
            inputs_.begin(), inputs_.end(), [&port_name](const framework::pipeline::PortInfo &p) {
                return p.name == port_name;
            });

    if (it != inputs_.end()) {
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
    RT_LOGC_ERROR(LdpcComponent::CrcDecoder, "{}", error_message);
    throw std::invalid_argument(error_message);
}

std::vector<framework::tensor::TensorInfo>
CrcDecoderModule::get_output_tensor_info(std::string_view port_name) const {

    if (!setup_complete_) {
        const std::string error_message = "get_output_tensor_info called "
                                          "before setup completion";
        RT_LOGC_ERROR(LdpcComponent::CrcDecoder, "{}", error_message);
        throw std::runtime_error(error_message);
    }

    const auto it = std::find_if(
            outputs_.begin(), outputs_.end(), [&port_name](const framework::pipeline::PortInfo &p) {
                return p.name == port_name;
            });

    if (it != outputs_.end()) {
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
    RT_LOGC_ERROR(LdpcComponent::CrcDecoder, "{}", error_message);
    throw std::invalid_argument(error_message);
}

std::vector<std::string> CrcDecoderModule::get_input_port_names() const { return {"decoded_bits"}; }

std::vector<std::string> CrcDecoderModule::get_output_port_names() const {
    return {"cb_crcs", "tb_crcs", "tb_payloads"};
}

void CrcDecoderModule::set_inputs(std::span<const framework::pipeline::PortInfo> inputs) {

    // Validate inputs
    if (inputs.size() != 1) {
        const std::string error_message = "Expected 1 input";
        RT_LOGC_ERROR(LdpcComponent::CrcDecoder, "{}", error_message);
        throw std::invalid_argument(error_message);
    }

    const auto &input = inputs[0];
    if (input.name != "decoded_bits") {
        const std::string error_message = std::format("Unknown port {}", input.name);
        RT_LOGC_ERROR(LdpcComponent::CrcDecoder, "{}", error_message);
        throw std::invalid_argument(error_message);
    }

    if (input.tensors.empty() || input.tensors[0].device_ptr == nullptr) {
        const std::string error_message =
                std::format("Input device pointer not set for port {}", input.name);
        RT_LOGC_ERROR(LdpcComponent::CrcDecoder, "{}", error_message);
        throw std::invalid_argument(error_message);
    }

    if (input.tensors[0].tensor_info.get_type() !=
        framework::tensor::TensorInfo::DataType::TensorBit) {
        const std::string error_message =
                std::format("Input must be TensorBit for port {}", input.name);
        RT_LOGC_ERROR(LdpcComponent::CrcDecoder, "{}", error_message);
        throw std::invalid_argument(error_message);
    }

    inputs_.assign(inputs.begin(), inputs.end());
}

std::vector<framework::pipeline::PortInfo> CrcDecoderModule::get_outputs() const {
    if (!setup_complete_) {
        const std::string error_message = "get_outputs called before setup completion";
        RT_LOGC_ERROR(LdpcComponent::CrcDecoder, "{}", error_message);
        throw std::runtime_error(error_message);
    }
    return outputs_;
}

void CrcDecoderModule::configure_io(
        const framework::pipeline::DynamicParams &params, cudaStream_t stream) {

    try {
        pusch_outer_rx_params_ = std::any_cast<PuschOuterRxParams>(params.module_specific_params);
    } catch (const std::bad_any_cast &) {
        const std::string error_message = "Invalid dynamic parameters!";
        RT_LOGC_ERROR(LdpcComponent::CrcDecoder, "{}", error_message);
        throw std::invalid_argument(error_message);
    }

    // Update output tensor info and device pointers.
    // Memory slice usage:
    // | Code block CRCs for all TBs |
    // | Transport block CRCs for all TBs |
    // | Transport block payloads for all TBs |
    const std::size_t tb_crcs_offset =
            sizeof(std::uint32_t) * static_params_.max_num_cbs_per_tb * static_params_.max_num_tbs;
    const std::size_t tb_payload_offset = sizeof(std::uint32_t) * static_params_.max_num_tbs;
    // cppcheck-suppress constVariablePointer
    std::byte *cb_crcs_ptr = memory_slice_.device_tensor_ptr;
    // cppcheck-suppress constVariablePointer
    std::byte *tb_crcs_ptr = std::next(cb_crcs_ptr, static_cast<std::ptrdiff_t>(tb_crcs_offset));
    // cppcheck-suppress constVariablePointer
    std::byte *tb_payloads_ptr =
            std::next(tb_crcs_ptr, static_cast<std::ptrdiff_t>(tb_payload_offset));

    const std::size_t num_tbs = pusch_outer_rx_params_.value().num_tbs();
    outputs_[0].tensors.resize(num_tbs);
    outputs_[1].tensors.resize(num_tbs);
    outputs_[2].tensors.resize(num_tbs);

    std::size_t total_num_tb_payload_bytes = 0;
    std::size_t total_num_code_blocks = 0;
    for (std::size_t tb_idx = 0; tb_idx < num_tbs; tb_idx++) {

        const auto ldpc_params = pusch_outer_rx_params_.value()[tb_idx].ldpc_params();

        const auto tb_size_bytes = ldpc_params.transport_block_size() / BITS_PER_BYTE;
        const auto num_code_blocks = ldpc_params.num_code_blocks();
        const auto tb_size_bytes_with_crc =
                ldpc_params.get_tb_size_with_crc(ldpc_params.transport_block_size()) /
                BITS_PER_BYTE;

        // Set output tensor information and device pointers.

        // Code block CRCs.
        outputs_[0].tensors[tb_idx].device_ptr = std::next(
                cb_crcs_ptr,
                static_cast<std::ptrdiff_t>(sizeof(std::uint32_t) * total_num_code_blocks));
        outputs_[0].tensors[tb_idx].tensor_info = framework::tensor::TensorInfo(
                framework::tensor::TensorInfo::DataType::TensorR32U,
                {static_cast<std::size_t>(num_code_blocks)});

        // Transport block CRCs.
        outputs_[1].tensors[tb_idx].device_ptr =
                std::next(tb_crcs_ptr, static_cast<std::ptrdiff_t>(sizeof(std::uint32_t) * tb_idx));
        outputs_[1].tensors[tb_idx].tensor_info = framework::tensor::TensorInfo(
                framework::tensor::TensorInfo::DataType::TensorR32U, {1});

        // Transport block payloads.
        outputs_[2].tensors[tb_idx].device_ptr =
                std::next(tb_payloads_ptr, static_cast<std::ptrdiff_t>(total_num_tb_payload_bytes));
        outputs_[2].tensors[tb_idx].tensor_info = framework::tensor::TensorInfo(
                framework::tensor::TensorInfo::DataType::TensorR8U,
                {static_cast<std::size_t>(tb_size_bytes), 1} // Do not include CRC
        );

        total_num_code_blocks += num_code_blocks;
        total_num_tb_payload_bytes += tb_size_bytes_with_crc;

        // Add alignment padding.
        const std::size_t tb_word_align_padding_bytes =
                (sizeof(std::uint32_t) - (tb_size_bytes_with_crc % sizeof(std::uint32_t))) %
                sizeof(std::uint32_t);
        total_num_tb_payload_bytes += tb_word_align_padding_bytes;
    }

    // Validate input LLR tensor info
    if (inputs_[0].tensors[0].tensor_info.get_dimensions()[0] !=
        LdpcParams::MAX_CODE_BLOCK_SIZE_BG1) {
        const std::string error_message = "Wrong number of input bits per code block!";
        RT_LOGC_ERROR(LdpcComponent::CrcDecoder, "{}", error_message);
        throw std::invalid_argument(error_message);
    }
    if (inputs_[0].tensors[0].tensor_info.get_dimensions()[1] != total_num_code_blocks) {
        const std::string error_message = "Wrong number of code blocks!";
        RT_LOGC_ERROR(LdpcComponent::CrcDecoder, "{}", error_message);
        throw std::invalid_argument(error_message);
    }

    const auto sch_user_idxs_span = pusch_outer_rx_params_.value().get_sch_user_idxs();
    std::vector<std::uint16_t> sch_user_idxs(sch_user_idxs_span.begin(), sch_user_idxs_span.end());
    const auto n_sch_ues = static_cast<std::uint16_t>(sch_user_idxs.size());

    const PerTbParams *p_tb_prms_cpu = pusch_outer_rx_params_.value().get_per_tb_params_cpu_ptr();
    const PerTbParams *p_tb_prms_gpu = pusch_outer_rx_params_.value().get_per_tb_params_gpu_ptr();

    // Not even used by the setup!! No copy done.
    constexpr uint8_t ENABLE_CPU_TO_GPU_DESCR_ASYNC_CPY = 0;

    const cuphyStatus_t setup_crc_decode_status = cuphySetupPuschRxCrcDecode(
            crc_decode_hndl_,
            n_sch_ues,
            sch_user_idxs.data(),
            // Output addresses: The address corresponding to the first TB.
            static_cast<uint32_t *>(outputs_[0].tensors[0].device_ptr), // CB CRCs
            static_cast<uint8_t *>(outputs_[2].tensors[0].device_ptr),  // TBs
            static_cast<uint32_t *>(inputs_[0].tensors[0].device_ptr),
            static_cast<uint32_t *>(outputs_[1].tensors[0].device_ptr), // TB CRCs
            p_tb_prms_cpu,
            p_tb_prms_gpu,
            memory_slice_.dynamic_kernel_descriptor_cpu_ptr,
            memory_slice_.dynamic_kernel_descriptor_gpu_ptr,
            ENABLE_CPU_TO_GPU_DESCR_ASYNC_CPY,
            crc_launch_cfgs_.data(),
            std::next(crc_launch_cfgs_.data(), 1),
            stream);
    if (setup_crc_decode_status != CUPHY_STATUS_SUCCESS) {
        const std::string error_message = "Failed to setup CRC check module!";
        RT_LOGC_ERROR(LdpcComponent::CrcDecoder, "{}", error_message);
        throw std::runtime_error(error_message);
    }

    setup_complete_ = true;
}

void CrcDecoderModule::execute(cudaStream_t stream) {

    RT_LOGC_DEBUG(
            LdpcComponent::CrcDecoder,
            "Executing module {} on stream {}",
            instance_id_,
            static_cast<void *>(stream));

    if (!setup_complete_) {
        const std::string error_message = "execute called before setup completion!";
        RT_LOGC_ERROR(LdpcComponent::CrcDecoder, "{}", error_message);
        throw std::runtime_error(error_message);
    }

    // Run code block CRC decoding.
    const CUDA_KERNEL_NODE_PARAMS &kernel_node_params_driver1 =
            crc_launch_cfgs_[0].kernelNodeParamsDriver;
    FRAMEWORK_CUDA_DRIVER_CHECK_THROW(
            framework::pipeline::launch_kernel(kernel_node_params_driver1, stream));

    // Run transport block CRC decoding.
    const CUDA_KERNEL_NODE_PARAMS &kernel_node_params_driver2 =
            crc_launch_cfgs_[1].kernelNodeParamsDriver;
    FRAMEWORK_CUDA_DRIVER_CHECK_THROW(
            framework::pipeline::launch_kernel(kernel_node_params_driver2, stream));
}

framework::pipeline::ModuleMemoryRequirements CrcDecoderModule::get_requirements() const {

    std::size_t dyn_descr_size_bytes{};
    std::size_t dyn_descr_align_bytes{};
    if (const cuphyStatus_t status =
                cuphyPuschRxCrcDecodeGetDescrInfo(&dyn_descr_size_bytes, &dyn_descr_align_bytes);
        status != CUPHY_STATUS_SUCCESS) {
        const std::string error_message = "Failed to get workspace size for CRC check";
        RT_LOGC_ERROR(LdpcComponent::CrcDecoder, "{}", error_message);
        throw std::runtime_error(error_message);
    }

    const std::size_t max_bytes_cb_crc =
            sizeof(std::uint32_t) * static_params_.max_num_cbs_per_tb * static_params_.max_num_tbs;
    const std::size_t max_bytes_tb_crc = sizeof(std::uint32_t) * static_params_.max_num_tbs;
    const std::size_t max_bytes_tb =
            static_params_.max_num_tbs * static_params_.max_num_cbs_per_tb *
            static_cast<std::size_t>(LdpcParams::MAX_CODE_BLOCK_SIZE_BG1 / BITS_PER_BYTE);
    const std::size_t n_bytes = max_bytes_cb_crc + max_bytes_tb_crc + max_bytes_tb;

    framework::pipeline::ModuleMemoryRequirements req{};
    req.static_kernel_descriptor_bytes = 0;
    req.dynamic_kernel_descriptor_bytes = dyn_descr_size_bytes;
    req.device_tensor_bytes = n_bytes;
    req.alignment = dyn_descr_align_bytes;
    return req;
}

std::span<const CUgraphNode> CrcDecoderModule::add_node_to_graph(
        gsl_lite::not_null<framework::pipeline::IGraph *> graph,
        const std::span<const CUgraphNode> deps) {

    RT_LOGC_DEBUG(
            LdpcComponent::CrcDecoder,
            "Adding kernel nodes to graph with {} dependencies",
            deps.size());

    std::vector<CUgraphNode> crc_deps(deps.begin(), deps.end());

    for (std::size_t idx = 0; idx < graph_nodes_.size(); ++idx) {
        auto &graph_node = graph_nodes_.at(idx);
        const auto updated_deps = std::span<const CUgraphNode>(crc_deps);
        graph_node = graph->add_kernel_node(
                updated_deps, crc_launch_cfgs_.at(idx).kernelNodeParamsDriver);
        if (graph_node == nullptr) {
            const std::string error_message = "Failed to add kernel node to graph";
            RT_LOGC_ERROR(LdpcComponent::CrcDecoder, "{}", error_message);
            throw std::runtime_error(error_message);
        }
        RT_LOGC_DEBUG(
                LdpcComponent::CrcDecoder,
                "Kernel node {} added: {}",
                idx,
                static_cast<void *>(graph_node));

        crc_deps.emplace_back(graph_node);
    }

    return graph_nodes_;
}

void CrcDecoderModule::update_graph_node_params(
        CUgraphExec exec, [[maybe_unused]] const framework::pipeline::DynamicParams &params) {
    for (std::size_t idx = 0; idx < graph_nodes_.size(); ++idx) {
        auto &graph_node = graph_nodes_.at(idx);
        const auto &kernel_params = crc_launch_cfgs_.at(idx).kernelNodeParamsDriver;
        FRAMEWORK_CUDA_DRIVER_CHECK_THROW(
                cuGraphExecKernelNodeSetParams(exec, graph_node, &kernel_params));

        RT_LOGC_DEBUG(LdpcComponent::CrcDecoder, "Graph node {} params updated", idx);
    }
}

} // namespace ran::ldpc
