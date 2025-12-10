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
#include <climits>
#include <cstddef>
#include <cstdint>
#include <format>
#include <functional>
#include <iterator>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <cuphy.h>
#include <driver_types.h>
#include <ldpc/ldpc_api.hpp>
#include <quill/LogMacros.h>

#include <gsl-lite/gsl-lite.hpp>

#include <cuda.h>

#include "ldpc/derate_match_params.hpp"
#include "ldpc/ldpc_decoder_module.hpp"
#include "ldpc/ldpc_log.hpp"
#include "ldpc/ldpc_params.hpp"
#include "ldpc/outer_rx_params.hpp"
#include "log/rt_log_macros.hpp"
#include "pipeline/igraph.hpp"
#include "pipeline/kernel_launch_config.hpp"
#include "pipeline/types.hpp"
#include "tensor/data_types.hpp"
#include "tensor/tensor_info.hpp"
#include "utils/error_macros.hpp"

namespace ran::ldpc {

LdpcDecoderModule::LdpcDecoderModule(std::string instance_id, const std::any &init_params)
        : instance_id_(std::move(instance_id)), decoder_(ctx_) {
    try {
        static_params_ = std::any_cast<StaticParams>(init_params);
    } catch (const std::bad_any_cast &) {
        const std::string error_message = "Invalid initialization parameters";
        RT_LOGC_ERROR(LdpcComponent::LdpcDecoder, "{}", error_message);
        throw std::invalid_argument(error_message);
    }

    // Preallocate descriptor set and launch config vector
    ldpc_desc_set_.resize(static_params_.max_num_ldpc_het_configs);
    ldpc_launch_cfgs_.reserve(static_params_.max_num_ldpc_het_configs);
}

std::string_view LdpcDecoderModule::get_instance_id() const { return instance_id_; }

void LdpcDecoderModule::setup_memory(const framework::pipeline::ModuleMemorySlice &memory_slice) {

    memory_slice_ = memory_slice;

    // Setup output port info
    // Only one output tensor that contains all the decoded bits for all TBs.
    constexpr std::size_t NUM_LDPC_DECODER_OUTPUT_TENSORS = 1;
    framework::pipeline::PortInfo output_port;
    output_port.name = "decoded_bits";
    output_port.tensors.resize(NUM_LDPC_DECODER_OUTPUT_TENSORS);
    outputs_.push_back(output_port);
}

std::vector<framework::tensor::TensorInfo>
LdpcDecoderModule::get_input_tensor_info(std::string_view port_name) const {
    if (port_name != "llrs") {
        const std::string error_message = std::format("Unknown port {}", port_name);
        RT_LOGC_ERROR(LdpcComponent::LdpcDecoder, "{}", error_message);
        throw std::invalid_argument(error_message);
    }

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
    RT_LOGC_ERROR(LdpcComponent::LdpcDecoder, "{}", error_message);
    throw std::invalid_argument(error_message);
}

std::vector<framework::tensor::TensorInfo>
LdpcDecoderModule::get_output_tensor_info(std::string_view port_name) const {

    if (!setup_complete_) {
        const std::string error_message = "get_output_tensor_info called before setup completion";
        RT_LOGC_ERROR(LdpcComponent::LdpcDecoder, "{}", error_message);
        throw std::runtime_error(error_message);
    }

    if (port_name != "decoded_bits") {
        const std::string error_message = std::format("Unknown port {}", port_name);
        RT_LOGC_ERROR(LdpcComponent::LdpcDecoder, "{}", error_message);
        throw std::invalid_argument(error_message);
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
    RT_LOGC_ERROR(LdpcComponent::LdpcDecoder, "{}", error_message);
    throw std::invalid_argument(error_message);
}

std::vector<std::string> LdpcDecoderModule::get_input_port_names() const { return {"llrs"}; }

std::vector<std::string> LdpcDecoderModule::get_output_port_names() const {
    return {"decoded_bits"};
}

void LdpcDecoderModule::set_inputs(std::span<const framework::pipeline::PortInfo> inputs) {

    // Validate inputs
    if (inputs.size() != 1) {
        const std::string error_message = "Expected 1 input";
        RT_LOGC_ERROR(LdpcComponent::LdpcDecoder, "{}", error_message);
        throw std::invalid_argument(error_message);
    }

    // Validate input port name
    const auto &input = inputs[0];
    if (input.name != "llrs") {
        const std::string error_message = std::format("Unknown port {}", input.name);
        RT_LOGC_ERROR(LdpcComponent::LdpcDecoder, "{}", error_message);
        throw std::invalid_argument(error_message);
    }

    // Validate that input tensors have been set
    if (input.tensors.empty()) {
        const std::string error_message = "Input tensors not set";
        RT_LOGC_ERROR(LdpcComponent::LdpcDecoder, "{}", error_message);
        throw std::invalid_argument(error_message);
    }

    // Validate that input LLRs have been set
    for (const auto &tensor : input.tensors) {
        if (tensor.device_ptr == nullptr) {
            const std::string error_message = "Input tensor address not set";
            RT_LOGC_ERROR(LdpcComponent::LdpcDecoder, "{}", error_message);
            throw std::invalid_argument(error_message);
        }
        if (tensor.tensor_info.get_type() != framework::tensor::TensorInfo::DataType::TensorR16F) {
            const std::string error_message = "Input tensor must be TensorR16F";
            RT_LOGC_ERROR(LdpcComponent::LdpcDecoder, "{}", error_message);
            throw std::invalid_argument(error_message);
        }
    }

    inputs_.assign(inputs.begin(), inputs.end());
}

std::vector<framework::pipeline::PortInfo> LdpcDecoderModule::get_outputs() const {
    if (!setup_complete_) {
        const std::string error_message = "get_outputs called before setup completion";
        RT_LOGC_ERROR(LdpcComponent::LdpcDecoder, "{}", error_message);
        throw std::runtime_error(error_message);
    }
    return outputs_;
}

void LdpcDecoderModule::configure_io(
        const framework::pipeline::DynamicParams &params, [[maybe_unused]] cudaStream_t stream) {

    try {
        pusch_outer_rx_params_ = std::any_cast<PuschOuterRxParams>(params.module_specific_params);
    } catch (const std::bad_any_cast &e) {
        const std::string error_message = "Invalid dynamic parameters!";
        RT_LOGC_ERROR(LdpcComponent::LdpcDecoder, "{}", error_message);
        throw std::invalid_argument(error_message);
    }

    // Validate input LLR tensor info
    const std::size_t num_tbs = pusch_outer_rx_params_.value().num_tbs();
    if (num_tbs > static_params_.max_num_tbs) {
        const std::string error_message =
                "Number of TBs in dynamic parameters exceeds maximum number of TBs";
        RT_LOGC_ERROR(LdpcComponent::LdpcDecoder, "{}", error_message);
        throw std::invalid_argument(error_message);
    }

    if (num_tbs != inputs_[0].tensors.size()) {
        const std::string error_message =
                "Number of TBs in dynamic parameters does not match number of input tensors";
        RT_LOGC_ERROR(LdpcComponent::LdpcDecoder, "{}", error_message);
        throw std::invalid_argument(error_message);
    }

    std::size_t num_total_code_blocks = 0;
    for (std::size_t tb_idx = 0; tb_idx < num_tbs; tb_idx++) {
        // Compute total number of code blocks
        const auto num_code_blocks =
                pusch_outer_rx_params_.value()[tb_idx].ldpc_params().num_code_blocks();
        num_total_code_blocks += num_code_blocks;

        // Validate input LLR tensor info
        const auto circular_buffer_size_padded =
                pusch_outer_rx_params_.value()[tb_idx].ldpc_params().circular_buffer_size_padded();
        if (inputs_[0].tensors[tb_idx].tensor_info.get_dimensions()[0] !=
            static_cast<std::size_t>(circular_buffer_size_padded)) {
            const std::string error_message =
                    "Input LLRs must have the same number of elements as the circular buffer size";
            RT_LOGC_ERROR(LdpcComponent::LdpcDecoder, "{}", error_message);
            throw std::invalid_argument(error_message);
        }
        if (inputs_[0].tensors[tb_idx].tensor_info.get_dimensions()[1] != num_code_blocks) {
            const std::string error_message =
                    "Input LLRs must have the same number of code blocks as the dynamic parameters";
            RT_LOGC_ERROR(LdpcComponent::LdpcDecoder, "{}", error_message);
            throw std::invalid_argument(error_message);
        }
    }

    outputs_[0].tensors[0].device_ptr = memory_slice_.device_tensor_ptr;
    outputs_[0].tensors[0].tensor_info = framework::tensor::TensorInfo(
            framework::tensor::TensorInfo::DataType::TensorBit,
            {static_cast<std::size_t>(LdpcParams::MAX_CODE_BLOCK_SIZE_BG1), num_total_code_blocks});

    setup_complete_ = true;

    // Prepare descriptors (groups TBs by configuration)
    prepare_ldpc_descriptors();

    // Get launch configurations for each descriptor
    const auto desc_count = ldpc_desc_set_.count();
    ldpc_launch_cfgs_.resize(desc_count);
    for (std::size_t i = 0; i < desc_count; ++i) {
        ldpc_launch_cfgs_[i].decode_desc = ldpc_desc_set_[i];

        decoder_.get_launch_config(ldpc_launch_cfgs_[i]);
    }
}

std::size_t LdpcDecoderModule::calculate_max_iterations_from_se(
        const LdpcParams &ldpc_params, const ModulationOrder mod_order) {

    // Calculate spectral efficiency: modulation order * code rate
    // LdpcParams already provides normalized code rate (0.0 to 1.0)
    const float spectral_efficiency =
            static_cast<float>(static_cast<std::uint32_t>(mod_order)) * ldpc_params.code_rate();

    // Lookup table based on spectral efficiency (from cuPHY pusch_rx.cpp lines 5640-5664)
    // Entries ordered from highest to lowest threshold
    struct SEThreshold {
        float max_se;
        std::size_t iterations;
    };

    static constexpr std::array SE_LUT = {
            SEThreshold{7.2F, 7},  // High spectral efficiency
            SEThreshold{0.4F, 10}, // Medium spectral efficiency
            SEThreshold{0.0F, 20}  // Low spectral efficiency (always matches)
    };

    // Find first entry where spectral_efficiency >= max_se
    const auto *const it =
            std::ranges::find_if(SE_LUT, [spectral_efficiency](const SEThreshold &entry) {
                return spectral_efficiency >= entry.max_se;
            });

    const std::size_t max_iterations = it->iterations;

    RT_LOGC_DEBUG(
            LdpcComponent::LdpcDecoder,
            "Spectral efficiency: {:.4f} (mod_order={}, code_rate={:.4f}) -> {} iterations",
            spectral_efficiency,
            static_cast<std::uint32_t>(mod_order),
            ldpc_params.code_rate(),
            max_iterations);

    return max_iterations;
}

void LdpcDecoderModule::prepare_ldpc_descriptors() {
    // Reset descriptor set to start fresh
    ldpc_desc_set_.reset();

    static constexpr ::cuphyDataType_t INPUT_DATA_TYPE = ::cuphyDataType_t::CUPHY_R_16F;
    static constexpr std::size_t WORD_SIZE_BITS = sizeof(std::uint32_t) * CHAR_BIT;
    static constexpr std::size_t OUTPUT_STRIDE =
            (LdpcParams::MAX_CODE_BLOCK_SIZE_BG1 + WORD_SIZE_BITS - 1) / WORD_SIZE_BITS;
    static constexpr uint32_t LDPC_FLAGS = CUPHY_LDPC_DECODE_CHOOSE_THROUGHPUT;
    static constexpr uint32_t ALGO_INDEX = 0;

    if (!pusch_outer_rx_params_.has_value()) {
        const std::string error_message =
                "LDPC decoder module prepare_ldpc_descriptors: outer_rx parameters not set";
        RT_LOGC_ERROR(LdpcComponent::LdpcDecoder, "{}", error_message);
        throw std::runtime_error(error_message);
    }

    const auto &outer_rx_params = pusch_outer_rx_params_.value();
    const auto num_tbs = outer_rx_params.num_tbs();

    std::size_t output_offset = 0;

    // Group TBs by (base_graph, lifting_size, num_parity_nodes)
    for (std::size_t tb_idx = 0; tb_idx < num_tbs; tb_idx++) {
        const auto &ldpc_params = outer_rx_params[tb_idx].ldpc_params();

        // Find or allocate descriptor for this configuration
        const auto base_graph = ldpc_params.base_graph();
        const auto lifting_size = gsl_lite::narrow_cast<std::int16_t>(ldpc_params.lifting_size());
        const auto num_parity = gsl_lite::narrow_cast<std::int16_t>(ldpc_params.num_parity_nodes());
        const auto num_code_blocks = ldpc_params.num_code_blocks();

        ::cuphy::LDPC_decode_desc &desc = ldpc_desc_set_.find(base_graph, lifting_size, num_parity);

        const auto tb_idx_in_desc = desc.num_tbs;

        // First TB in descriptor: populate full configuration
        if (tb_idx_in_desc == 0) {
            desc.config.llr_type = INPUT_DATA_TYPE;

            // Determine max iterations based on method
            std::size_t max_iterations{};
            if (static_params_.max_iterations_method == LdpcMaxIterationsMethod::Lut) {
                // Get modulation order from derate match params
                const auto mod_order = outer_rx_params[tb_idx].de_rm_params().mod_order;
                max_iterations = calculate_max_iterations_from_se(ldpc_params, mod_order);
            } else {
                // Use fixed value from static params
                max_iterations = static_params_.max_num_iterations;
            }

            desc.config.max_iterations = gsl_lite::narrow_cast<std::int16_t>(max_iterations);
            desc.config.clamp_value = static_params_.clamp_value;
            desc.config.Kb = gsl_lite::narrow_cast<std::int16_t>(ldpc_params.num_info_nodes());
            desc.config.BG = base_graph;
            desc.config.Z = lifting_size;
            desc.config.num_parity_nodes = num_parity;
            desc.config.flags = LDPC_FLAGS;
            desc.config.algo = ALGO_INDEX;
            desc.config.workspace = nullptr;
            // Set normalization constant based on code rate
            decoder_.set_normalization(desc.config);
        }

        // Set input LLR address and stride for this TB
        auto &llr_input = gsl_lite::at(desc.llr_input, static_cast<std::size_t>(tb_idx_in_desc));
        llr_input.addr = inputs_[0].tensors[tb_idx].device_ptr;
        llr_input.stride_elements =
                gsl_lite::narrow_cast<std::int32_t>(ldpc_params.circular_buffer_size_padded());
        llr_input.num_codewords = gsl_lite::narrow_cast<std::int32_t>(num_code_blocks);

        // Set output address and stride for this TB
        auto &tb_output = gsl_lite::at(desc.tb_output, static_cast<std::size_t>(tb_idx_in_desc));
        void *output_ptr = std::next(
                static_cast<std::byte *>(outputs_[0].tensors[0].device_ptr),
                gsl_lite::narrow_cast<std::ptrdiff_t>(output_offset));
        tb_output.addr = static_cast<std::uint32_t *>(output_ptr);
        tb_output.stride_words = gsl_lite::narrow_cast<std::int32_t>(OUTPUT_STRIDE);
        tb_output.num_codewords = gsl_lite::narrow_cast<std::int32_t>(num_code_blocks);

        // Increment TB count in this descriptor
        ++desc.num_tbs;

        output_offset += OUTPUT_STRIDE * num_code_blocks * sizeof(std::uint32_t);
    }

    RT_LOGC_DEBUG(
            LdpcComponent::LdpcDecoder,
            "Prepared {} LDPC descriptor(s) for {} TB(s)",
            ldpc_desc_set_.count(),
            num_tbs);
}

void LdpcDecoderModule::execute(cudaStream_t stream) {

    RT_LOGC_DEBUG(
            LdpcComponent::LdpcDecoder,
            "Executing module {} on stream {}",
            instance_id_,
            static_cast<void *>(stream));

    if (!setup_complete_) {
        const std::string error_message =
                "LDPC decoder module execute called before setup completion";
        RT_LOGC_ERROR(LdpcComponent::LdpcDecoder, "{}", error_message);
        throw std::runtime_error(error_message);
    }

    // Launch all descriptors sequentially
    const auto desc_count = ldpc_desc_set_.count();
    for (std::size_t i = 0; i < desc_count; ++i) {
        FRAMEWORK_CUDA_DRIVER_CHECK_THROW(framework::pipeline::launch_kernel(
                ldpc_launch_cfgs_[i].kernel_node_params_driver, stream));
    }

    RT_LOGC_DEBUG(
            LdpcComponent::LdpcDecoder,
            "Launched {} LDPC decoder kernel(s) on stream {}",
            desc_count,
            static_cast<void *>(stream));
}

std::span<const CUgraphNode> LdpcDecoderModule::add_node_to_graph(
        gsl_lite::not_null<framework::pipeline::IGraph *> graph,
        const std::span<const CUgraphNode> deps) {

    RT_LOGC_DEBUG(
            LdpcComponent::LdpcDecoder,
            "Adding LDPC decoder node(s) to graph for module {}",
            instance_id_);

    if (!setup_complete_) {
        const std::string error_message = std::format(
                "LDPC decoder module '{}': configure_io() must be called before "
                "add_node_to_graph()",
                instance_id_);
        RT_LOGC_ERROR(LdpcComponent::LdpcDecoder, "{}", error_message);
        throw std::runtime_error(error_message);
    }

    // Add kernel nodes using the launch configs prepared in configure_io()
    const auto desc_count = ldpc_desc_set_.count();

    ldpc_graph_nodes_.clear();
    ldpc_graph_nodes_.reserve(desc_count);

    for (std::size_t i = 0; i < desc_count; ++i) {
        CUgraphNode node =
                graph->add_kernel_node(deps, ldpc_launch_cfgs_[i].kernel_node_params_driver);
        if (node == nullptr) {
            const std::string error_message = "Failed to add LDPC decoder kernel node to graph";
            RT_LOGC_ERROR(LdpcComponent::LdpcDecoder, "{}", error_message);
            throw std::runtime_error(error_message);
        }
        ldpc_graph_nodes_.push_back(node);

        RT_LOGC_DEBUG(
                LdpcComponent::LdpcDecoder,
                "Added LDPC decoder kernel node {} (descriptor {}/{}) to graph",
                // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
                reinterpret_cast<std::uintptr_t>(node),
                i + 1,
                desc_count);
    }

    RT_LOGC_INFO(
            LdpcComponent::LdpcDecoder,
            "Module '{}': Added {} LDPC decoder kernel node(s) to graph",
            instance_id_,
            desc_count);

    // Return span of all created nodes
    return ldpc_graph_nodes_;
}

void LdpcDecoderModule::update_graph_node_params(
        CUgraphExec exec, [[maybe_unused]] const framework::pipeline::DynamicParams &params) {

    RT_LOGC_DEBUG(
            LdpcComponent::LdpcDecoder,
            "Updating graph node parameters for module {}",
            instance_id_);

    // Descriptors and launch configs have already been prepared in configure_io()
    const auto desc_count = static_cast<std::size_t>(ldpc_desc_set_.count());

    // Update parameters
    for (std::size_t i = 0; i < desc_count; ++i) {
        FRAMEWORK_CUDA_DRIVER_CHECK_THROW(cuGraphExecKernelNodeSetParams(
                exec, ldpc_graph_nodes_[i], &ldpc_launch_cfgs_[i].kernel_node_params_driver));

        RT_LOGC_DEBUG(
                LdpcComponent::LdpcDecoder,
                "Updated graph node {} (descriptor {}/{})",
                // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
                reinterpret_cast<std::uintptr_t>(ldpc_graph_nodes_[i]),
                i + 1,
                desc_count);
    }

    RT_LOGC_DEBUG(
            LdpcComponent::LdpcDecoder,
            "Updated {} LDPC decoder graph node(s) for module {}",
            desc_count,
            instance_id_);
}

framework::pipeline::ModuleMemoryRequirements LdpcDecoderModule::get_requirements() const {
    framework::pipeline::ModuleMemoryRequirements req;

    static constexpr std::size_t WORD_SIZE_BITS = sizeof(std::uint32_t) * CHAR_BIT;
    static constexpr std::size_t NUM_WORDS_OUT =
            (LdpcParams::MAX_CODE_BLOCK_SIZE_BG1 + WORD_SIZE_BITS - 1) / WORD_SIZE_BITS;
    const std::size_t n_bytes = sizeof(std::uint32_t) * NUM_WORDS_OUT *
                                static_cast<std::size_t>(static_params_.max_num_cbs_per_tb) *
                                static_cast<std::size_t>(static_params_.max_num_tbs);

    req.static_kernel_descriptor_bytes = 0;
    req.dynamic_kernel_descriptor_bytes = 0;
    req.device_tensor_bytes = n_bytes;
    static constexpr std::size_t ALIGNMENT_BYTES = 128;
    req.alignment = ALIGNMENT_BYTES;
    return req;
}

} // namespace ran::ldpc
