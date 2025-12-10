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

#include <cstddef>
#include <cstdint> // for int64_t, int32_t, uint16_t
#include <format>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <driver_types.h>
#include <quill/LogMacros.h>

#include <gsl-lite/gsl-lite.hpp>
#include <wise_enum.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include "log/rt_log_macros.hpp"
#include "pipeline/igraph.hpp"
#include "pipeline/igraph_node_provider.hpp"
#include "pipeline/istream_executor.hpp"
#include "pipeline/pipeline_memory_manager.hpp"
#include "pipeline/types.hpp"
#include "pusch/inner_rx_module.hpp"
#include "pusch/pusch_log.hpp"
#include "pusch/pusch_trt_utils.hpp"
#include "ran_common.hpp"
#include "tensor/data_types.hpp"
#include "tensor/tensor_info.hpp"
#include "tensorrt/mlir_trt_engine.hpp"
#include "tensorrt/trt_engine.hpp"
#include "tensorrt/trt_engine_interfaces.hpp"
#include "tensorrt/trt_engine_logger.hpp"
#include "tensorrt/trt_engine_params.hpp"
#include "tensorrt/trt_null_pre_post_enqueue.hpp"
#include "tensorrt/trt_pre_post_enqueue_stream_cap.hpp"
#include "utils/error_macros.hpp"
#include "utils/errors.hpp"

namespace ran::pusch {

namespace pipeline = framework::pipeline;
namespace tensor = framework::tensor;
namespace tensorrt = framework::tensorrt;
namespace utils = framework::utils;

// ============================================================================
// Construction
// ============================================================================

InnerRxModule::InnerRxModule(std::string instance_id, const StaticParams &params)
        : static_params_(params), instance_id_(std::move(instance_id)),
          execution_mode_(params.execution_mode),
          trt_logger_(std::make_unique<tensorrt::TrtLogger>()) {

    RT_LOGC_INFO(
            PuschComponent::InnerRxModule,
            "InnerRxModule: Constructing instance '{}', execution_mode={}",
            instance_id_,
            ::wise_enum::to_string(execution_mode_));

    // Initialize TensorRT plugins
    if (!init_ran_trt_plugins()) {
        throw std::runtime_error("Failed to initialize RAN TensorRT plugins");
    }

    // Get TRT engine path from environment variable
    const std::string trt_engine_path = get_trt_engine_path();
    RT_LOGC_INFO(PuschComponent::InnerRxModule, "Using TensorRT engine: {}", trt_engine_path);

    const std::uint16_t num_rx_ant = params.phy_params.num_rx_ant;
    const std::uint32_t num_prb = params.phy_params.num_prb;

    // Configure TensorRT engine parameters
    const auto num_subcarriers =
            static_cast<std::size_t>(num_prb) * ran::common::NUM_SUBCARRIERS_PER_PRB;
    const std::size_t num_ofdm_symbols = ran::common::OFDM_SYMBOLS_NORMAL_CYCLIC_PREFIX;
    const auto num_real_imag_interleaved =
            static_cast<std::size_t>(ran::common::REAL_IMAG_INTERLEAVED);

    const tensorrt::MLIRTensorParams xtf{
            .name = "arg0", // Input XTF
            .data_type = tensor::TensorR16F,
            .rank = 4,
            .dims = {num_rx_ant, num_ofdm_symbols, num_subcarriers, num_real_imag_interleaved}};

    const tensorrt::MLIRTensorParams post_eq_noise_var_db{
            .name = "result0", // Output post-equalizer noise var dB
            .data_type = tensor::TensorR32F,
            .rank = 1,
            .dims = {ran::common::MAX_UES_PER_SLOT}};

    const tensorrt::MLIRTensorParams post_eq_sinr_db{
            .name = "result1", // Output post-equalizer SINR dB
            .data_type = tensor::TensorR32F,
            .rank = 1,
            .dims = {ran::common::MAX_UES_PER_SLOT}};

    const tensorrt::MLIRTensorParams llr{
            .name = "result2", // Output LLRs
            .data_type = tensor::TensorR16F,
            .rank = 4,
            .dims = {
                    ran::common::OFDM_SYMBOLS_NORMAL_CYCLIC_PREFIX -
                            ran::common::MAX_DMRS_OFDM_SYMBOLS,
                    static_cast<std::size_t>(num_prb) * ran::common::NUM_SUBCARRIERS_PER_PRB,
                    ran::common::MAX_UL_LAYERS,
                    ran::common::MAX_QAM_BITS}};

    const std::vector<tensorrt::MLIRTensorParams> inputs = {xtf};
    const std::vector<tensorrt::MLIRTensorParams> outputs = {
            post_eq_noise_var_db, post_eq_sinr_db, llr};

    // Create TensorRT engine with engine file path
    auto tensorrt_runtime = std::make_unique<tensorrt::TrtEngine>(trt_engine_path, *trt_logger_);

    // Create graph capturer based on execution mode
    auto create_graph_capturer = [this]() -> std::unique_ptr<tensorrt::IPrePostTrtEngEnqueue> {
        if (execution_mode_ == pipeline::ExecutionMode::Graph) {
            auto capturer = std::make_unique<tensorrt::CaptureStreamPrePostTrtEngEnqueue>();
            graph_capturer_ = capturer.get();
            return capturer;
        }
        auto capturer = std::make_unique<tensorrt::NullPrePostTrtEngEnqueue>();
        graph_capturer_ = capturer.get();
        return capturer;
    };

    // Create MLIR TRT engine with the runtime and capture helper (takes ownership
    // of both)
    trt_engine_ = std::make_unique<tensorrt::MLIRTrtEngine>(
            inputs, outputs, std::move(tensorrt_runtime), create_graph_capturer());

    RT_LOGC_INFO(
            PuschComponent::InnerRxModule,
            "InnerRxModule '{}': Constructor complete",
            instance_id_);
}

// ============================================================================
// Interface Access
// ============================================================================

framework::pipeline::IStreamExecutor *InnerRxModule::as_stream_executor() { return this; }

framework::pipeline::IGraphNodeProvider *InnerRxModule::as_graph_node_provider() { return this; }

// ============================================================================
// Port Introspection
// ============================================================================

std::vector<std::string> InnerRxModule::get_input_port_names() const { return {"xtf"}; }

std::vector<std::string> InnerRxModule::get_output_port_names() const {
    return {"llrs", "post_eq_noise_var_db", "post_eq_sinr_db"};
}

std::vector<tensor::TensorInfo>
InnerRxModule::get_input_tensor_info(std::string_view port_name) const {

    if (port_name == "xtf") {
        const std::uint16_t num_rx_ant = static_params_.phy_params.num_rx_ant;
        const std::uint32_t num_prb = static_params_.phy_params.num_prb;
        return {tensor::TensorInfo(
                tensor::TensorInfo::DataType::TensorR16F,
                {num_rx_ant,
                 ran::common::OFDM_SYMBOLS_NORMAL_CYCLIC_PREFIX,
                 static_cast<std::size_t>(num_prb) * ran::common::NUM_SUBCARRIERS_PER_PRB,
                 ran::common::REAL_IMAG_INTERLEAVED})};
    }

    const std::string error_msg =
            std::format("InnerRxModule '{}': Unknown input port '{}'", instance_id_, port_name);
    RT_LOGC_ERROR(PuschComponent::InnerRxModule, "{}", error_msg);
    throw std::invalid_argument(error_msg);
}

std::vector<tensor::TensorInfo>
InnerRxModule::get_output_tensor_info(std::string_view port_name) const {
    const std::uint32_t num_prb = static_params_.phy_params.num_prb;

    if (port_name == "llrs") {
        return {tensor::TensorInfo(
                tensor::TensorInfo::DataType::TensorR16F,
                {ran::common::OFDM_SYMBOLS_NORMAL_CYCLIC_PREFIX -
                         ran::common::MAX_DMRS_OFDM_SYMBOLS,
                 static_cast<std::size_t>(num_prb) * ran::common::NUM_SUBCARRIERS_PER_PRB,
                 ran::common::MAX_UL_LAYERS,
                 ran::common::MAX_QAM_BITS})};
    } else if (port_name == "post_eq_noise_var_db" || port_name == "post_eq_sinr_db") {
        return {tensor::TensorInfo(
                tensor::TensorInfo::DataType::TensorR32F, {ran::common::MAX_UES_PER_SLOT})};
    }

    const std::string error_msg =
            std::format("InnerRxModule '{}': Unknown output port '{}'", instance_id_, port_name);
    RT_LOGC_ERROR(PuschComponent::InnerRxModule, "{}", error_msg);
    throw std::invalid_argument(error_msg);
}

// ============================================================================
// Memory Configuration
// ============================================================================

pipeline::InputPortMemoryCharacteristics
InnerRxModule::get_input_memory_characteristics(std::string_view port_name) const {
    if (port_name == "xtf") {
        const bool requires_fixed = (execution_mode_ == pipeline::ExecutionMode::Graph);
        return pipeline::InputPortMemoryCharacteristics{
                .requires_fixed_address_for_zero_copy = requires_fixed};
    }

    const std::string error_msg =
            std::format("InnerRxModule '{}': Unknown input port '{}'", instance_id_, port_name);
    RT_LOGC_ERROR(PuschComponent::InnerRxModule, "{}", error_msg);
    throw std::invalid_argument(error_msg);
}

pipeline::OutputPortMemoryCharacteristics
InnerRxModule::get_output_memory_characteristics(std::string_view port_name) const {
    if (port_name == "llrs" || port_name == "post_eq_noise_var_db" ||
        port_name == "post_eq_sinr_db") {
        return pipeline::OutputPortMemoryCharacteristics{
                .provides_fixed_address_for_zero_copy = true};
    }

    const std::string error_msg =
            std::format("InnerRxModule '{}': Unknown output port '{}'", instance_id_, port_name);
    RT_LOGC_ERROR(PuschComponent::InnerRxModule, "{}", error_msg);
    throw std::invalid_argument(error_msg);
}

void InnerRxModule::set_connection_copy_mode(
        std::string_view port_name, pipeline::ConnectionCopyMode mode) {
    const bool zero_copy = (mode == pipeline::ConnectionCopyMode::ZeroCopy);

    if (port_name == "xtf") {
        xtf_upstream_is_fixed_ = zero_copy;
        RT_LOGC_DEBUG(
                PuschComponent::InnerRxModule,
                "'{}': input 'xtf' connection copy mode = {}",
                instance_id_,
                ::wise_enum::to_string(mode));
    } else {
        const std::string error_msg =
                std::format("'{}': Unknown input port '{}'", instance_id_, port_name);
        RT_LOGC_ERROR(PuschComponent::InnerRxModule, "{}", error_msg);
        throw std::invalid_argument(error_msg);
    }
}

pipeline::ModuleMemoryRequirements InnerRxModule::get_requirements() const {
    pipeline::ModuleMemoryRequirements reqs{};

    const std::uint16_t num_rx_ant = static_params_.phy_params.num_rx_ant;
    const std::uint32_t num_prb = static_params_.phy_params.num_prb;

    // Calculate required memory with proper alignment between tensors
    // Must match the allocation logic in setup_memory()
    std::size_t total_bytes = 0;
    std::size_t num_tensors = 0;

    // Input tensors
    if (!xtf_upstream_is_fixed_) {
        total_bytes += pipeline::align_memory_offset(
                get_nv_type_storage_element_size(tensor::TensorInfo::DataType::TensorR16F) *
                        num_rx_ant * ran::common::OFDM_SYMBOLS_NORMAL_CYCLIC_PREFIX *
                        static_cast<std::size_t>(num_prb) * ran::common::NUM_SUBCARRIERS_PER_PRB *
                        ran::common::REAL_IMAG_INTERLEAVED,
                MEMORY_ALIGNMENT);
        num_tensors++;
    }
    // Output tensors - always allocate outputs
    // LLR tensor
    total_bytes += pipeline::align_memory_offset(
            get_nv_type_storage_element_size(tensor::TensorInfo::DataType::TensorR16F) *
                    (ran::common::OFDM_SYMBOLS_NORMAL_CYCLIC_PREFIX -
                     ran::common::MAX_DMRS_OFDM_SYMBOLS) *
                    num_prb * ran::common::NUM_SUBCARRIERS_PER_PRB * ran::common::MAX_UL_LAYERS *
                    ran::common::MAX_QAM_BITS,
            MEMORY_ALIGNMENT);
    num_tensors++;
    // Post eq noise var db
    total_bytes += pipeline::align_memory_offset(
            get_nv_type_storage_element_size(tensor::TensorInfo::DataType::TensorR32F) *
                    ran::common::MAX_UES_PER_SLOT,
            MEMORY_ALIGNMENT);
    num_tensors++;
    // Post eq sinr db
    total_bytes += pipeline::align_memory_offset(
            get_nv_type_storage_element_size(tensor::TensorInfo::DataType::TensorR32F) *
                    ran::common::MAX_UES_PER_SLOT,
            MEMORY_ALIGNMENT);
    num_tensors++;
    reqs.device_tensor_bytes = total_bytes;
    reqs.alignment = MEMORY_ALIGNMENT;
    RT_LOGC_DEBUG(
            PuschComponent::InnerRxModule,
            "InnerRxModule '{}': Memory requirements - device={} tensors={}",
            instance_id_,
            reqs.device_tensor_bytes,
            num_tensors);
    return reqs;
}

// ============================================================================
// Setup Phase
// ============================================================================

void InnerRxModule::setup_memory(const pipeline::ModuleMemorySlice &memory_slice) {
    RT_LOGC_INFO(
            PuschComponent::InnerRxModule,
            "InnerRxModule '{}': setup_memory() called",
            instance_id_);

    const std::uint16_t num_rx_ant = static_params_.phy_params.num_rx_ant;
    const std::uint32_t num_prb = static_params_.phy_params.num_prb;

    mem_slice_ = memory_slice;
    std::byte *base_ptr = mem_slice_.device_tensor_ptr;
    std::size_t offset = 0;

    // Input tensors
    if (!xtf_upstream_is_fixed_) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast,cppcoreguidelines-pro-bounds-pointer-arithmetic)
        d_xtf_ = reinterpret_cast<__half *>(base_ptr + offset);
        offset += get_nv_type_storage_element_size(tensor::TensorInfo::DataType::TensorR16F) *
                  static_cast<std::size_t>(static_params_.phy_params.num_prb) *
                  ran::common::NUM_SUBCARRIERS_PER_PRB *
                  ran::common::OFDM_SYMBOLS_NORMAL_CYCLIC_PREFIX * num_rx_ant *
                  ran::common::REAL_IMAG_INTERLEAVED;
        offset = pipeline::align_memory_offset(offset, MEMORY_ALIGNMENT);
        RT_LOGC_INFO(
                PuschComponent::InnerRxModule,
                "InnerRxModule '{}': Allocated xtf={} (copy mode)",
                instance_id_,
                static_cast<void *>(d_xtf_));
    } else {
        d_xtf_ = nullptr;
        RT_LOGC_INFO(
                PuschComponent::InnerRxModule,
                "InnerRxModule '{}': xtf in zero-copy mode (no allocation)",
                instance_id_);
    }

    // Output tensors - always allocate output (no alignment needed after last buffer)
    // LLR tensor
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast,cppcoreguidelines-pro-bounds-pointer-arithmetic)
    d_llr_ = reinterpret_cast<__half *>(base_ptr + offset);
    offset +=
            get_nv_type_storage_element_size(tensor::TensorInfo::DataType::TensorR16F) *
            (ran::common::OFDM_SYMBOLS_NORMAL_CYCLIC_PREFIX - ran::common::MAX_DMRS_OFDM_SYMBOLS) *
            num_prb * ran::common::NUM_SUBCARRIERS_PER_PRB * ran::common::MAX_UL_LAYERS *
            ran::common::MAX_QAM_BITS;
    offset = pipeline::align_memory_offset(offset, MEMORY_ALIGNMENT);
    RT_LOGC_INFO(
            PuschComponent::InnerRxModule,
            "InnerRxModule '{}': Allocated llrs={}",
            instance_id_,
            static_cast<void *>(d_llr_));

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast,cppcoreguidelines-pro-bounds-pointer-arithmetic)
    d_post_eq_noise_var_db_ = reinterpret_cast<float *>(base_ptr + offset);
    offset += get_nv_type_storage_element_size(tensor::TensorInfo::DataType::TensorR32F) *
              ran::common::MAX_UES_PER_SLOT;
    offset = pipeline::align_memory_offset(offset, MEMORY_ALIGNMENT);
    RT_LOGC_INFO(
            PuschComponent::InnerRxModule,
            "InnerRxModule '{}': Allocated post_eq_noise_var_db={}",
            instance_id_,
            static_cast<void *>(d_post_eq_noise_var_db_));

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast,cppcoreguidelines-pro-bounds-pointer-arithmetic)
    d_post_eq_sinr_db_ = reinterpret_cast<float *>(base_ptr + offset);

    RT_LOGC_INFO(
            PuschComponent::InnerRxModule,
            "InnerRxModule '{}': Allocated post_eq_sinr_db={}",
            instance_id_,
            static_cast<void *>(d_post_eq_sinr_db_));
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void InnerRxModule::set_inputs(std::span<const pipeline::PortInfo> inputs) {
    RT_LOGC_DEBUG(
            PuschComponent::InnerRxModule,
            "InnerRxModule '{}': set_inputs() called with {} ports",
            instance_id_,
            inputs.size());

    for (const auto &port : inputs) {
        if (port.tensors.empty()) {
            const std::string error_msg = std::format(
                    "InnerRxModule '{}': Port '{}' has no tensors", instance_id_, port.name);
            RT_LOGEC_ERROR(
                    PuschComponent::InnerRxModule,
                    PuschPipelineEvent::ModuleSetInputs,
                    "{}",
                    error_msg);
            throw std::invalid_argument(error_msg);
        }

        if (port.name == "xtf") {
            xtf_data_ = port.tensors[0].device_ptr;

            // Zero-copy: Use upstream address directly
            if (xtf_upstream_is_fixed_) {
                // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
                d_xtf_ = const_cast<__half *>(static_cast<const __half *>(xtf_data_));
                RT_LOGC_DEBUG(
                        PuschComponent::InnerRxModule,
                        "InnerRxModule '{}': xtf={} (zero-copy mode)",
                        instance_id_,
                        static_cast<void *>(d_xtf_));
            } else {
                RT_LOGC_DEBUG(
                        PuschComponent::InnerRxModule,
                        "InnerRxModule '{}': input={} (will copy in configure_io)",
                        instance_id_,
                        xtf_data_);
            }
        } else {
            const std::string error_msg = std::format(
                    "InnerRxModule '{}': Unknown input port '{}'", instance_id_, port.name);
            RT_LOGC_ERROR(PuschComponent::InnerRxModule, "{}", error_msg);
            throw std::invalid_argument(error_msg);
        }
    }

    RT_LOGC_INFO(
            PuschComponent::InnerRxModule,
            "InnerRxModule '{}': Input connections established ",
            instance_id_);
}

void InnerRxModule::warmup(cudaStream_t stream) {
    RT_LOGC_INFO(
            PuschComponent::InnerRxModule,
            "'{}': warmup(stream={}) called",
            instance_id_,
            static_cast<void *>(stream));

    if (is_warmed_up_) {
        RT_LOGC_DEBUG(
                PuschComponent::InnerRxModule, "'{}': Already warmed up, skipping", instance_id_);
        return;
    }

    // Validate that all tensor pointers are set
    // - Inputs (copy mode): Allocated in setup_memory()
    // - Inputs (zero-copy mode): Assigned in set_inputs()
    // - Outputs: Always allocated in setup_memory()
    if (d_xtf_ == nullptr || d_llr_ == nullptr || d_post_eq_noise_var_db_ == nullptr ||
        d_post_eq_sinr_db_ == nullptr) {
        const std::string error_msg = std::format(
                "InnerRxModule '{}': warmup() called before tensor pointers established. "
                "Ensure setup_memory() and set_inputs() have been called.",
                instance_id_);
        RT_LOGC_ERROR(PuschComponent::InnerRxModule, "{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    // Validate that inputs have been set
    if (xtf_data_ == nullptr) {
        const std::string error_msg = std::format(
                "InnerRxModule '{}': warmup() called before set_inputs(). "
                "Input connections must be established before warmup.",
                instance_id_);
        RT_LOGC_ERROR(PuschComponent::InnerRxModule, "{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    // Configure TRT engine with FIXED tensor addresses (one-time setup)
    // Use internal fixed buffers for all tensors
    const std::vector<void *> input_buffers = {d_xtf_};
    const std::vector<void *> output_buffers = {
            d_post_eq_noise_var_db_, d_post_eq_sinr_db_, d_llr_};

    RT_LOGC_INFO(
            PuschComponent::InnerRxModule,
            "'{}': Performing one-time warmup "
            "(loads engine to device, captures CUDA graph for graph mode). "
            "Fixed tensor addresses: stream={}",
            instance_id_,
            static_cast<void *>(stream));

    RT_LOGC_DEBUG(
            PuschComponent::InnerRxModule,
            "'{}': Input tensor addresses: xtf={}",
            instance_id_,
            static_cast<void *>(d_xtf_));

    RT_LOGC_DEBUG(
            PuschComponent::InnerRxModule,
            "'{}': Output tensor addresses: "
            "llrs={}, post_eq_noise_var_db={}, post_eq_sinr_db={}",
            instance_id_,
            static_cast<void *>(d_llr_),
            static_cast<void *>(d_post_eq_noise_var_db_),
            static_cast<void *>(d_post_eq_sinr_db_));

    RT_LOGC_INFO(PuschComponent::InnerRxModule, "'{}': Calling TRT engine setup()", instance_id_);
    const utils::NvErrc setup_result = trt_engine_->setup(input_buffers, output_buffers);
    if (setup_result != utils::NvErrc::Success) {
        const std::string error_msg =
                std::format("InnerRxModule '{}': TRT engine setup() failed", instance_id_);
        RT_LOGC_ERROR(PuschComponent::InnerRxModule, "{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    // Use provided stream for warmup/graph capture
    // Note: TensorRT graph capture requires a non-default stream (cannot use cudaStreamDefault)
    RT_LOGC_INFO(PuschComponent::InnerRxModule, "'{}': Calling TRT engine warmup()", instance_id_);
    const utils::NvErrc warmup_result = trt_engine_->warmup(stream);
    if (warmup_result != utils::NvErrc::Success) {
        const std::string error_msg =
                std::format("InnerRxModule '{}': TRT engine warmup() failed", instance_id_);
        RT_LOGC_ERROR(PuschComponent::InnerRxModule, "{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    // Synchronize stream to ensure graph capture is complete
    FRAMEWORK_CUDA_RUNTIME_CHECK_THROW(cudaStreamSynchronize(stream));

    is_warmed_up_ = true;
    RT_LOGC_INFO(
            PuschComponent::InnerRxModule,
            "'{}': Warmup complete - TRT engine ready for "
            "execution (graph captured for graph mode, stream mode ready)",
            instance_id_);
}

// ============================================================================
// Per-Iteration Configuration
// ============================================================================

void InnerRxModule::configure_io(const pipeline::DynamicParams &params, cudaStream_t stream) {
    RT_LOGC_INFO(PuschComponent::InnerRxModule, "InnerRxModule '{}': configure_io()", instance_id_);

    // Validate external input ports have been set
    if (xtf_data_ == nullptr) {
        const std::string error_msg = std::format(
                "InnerRxModule '{}': Input ports not set before configure_io()", instance_id_);
        RT_LOGC_ERROR(PuschComponent::InnerRxModule, "{}", error_msg);
        throw std::runtime_error(error_msg);
    }
    // Populate internal buffers with external input data
    (void)params;

    const std::uint16_t num_rx_ant = static_params_.phy_params.num_rx_ant;
    const std::uint32_t num_prb = static_params_.phy_params.num_prb;

    const std::size_t xtf_tensor_size =
            static_cast<std::size_t>(num_prb) * ran::common::NUM_SUBCARRIERS_PER_PRB *
            ran::common::OFDM_SYMBOLS_NORMAL_CYCLIC_PREFIX * num_rx_ant *
            ran::common::REAL_IMAG_INTERLEAVED; // 4D tensor (n_sc=3276, n_sym=14, n_rxant=4, ri=2)
                                                // - column-major, real/imag interleaved
    if (!xtf_upstream_is_fixed_) {
        FRAMEWORK_CUDA_RUNTIME_CHECK_THROW(cudaMemcpyAsync(
                d_xtf_,
                xtf_data_,
                xtf_tensor_size * sizeof(__half),
                cudaMemcpyDeviceToDevice,
                stream));
        RT_LOGC_DEBUG(
                PuschComponent::InnerRxModule,
                "InnerRxModule '{}': Copied xtf {} -> {} (async)",
                instance_id_,
                xtf_data_,
                static_cast<void *>(d_xtf_));
    } else {
        RT_LOGC_DEBUG(
                PuschComponent::InnerRxModule,
                "InnerRxModule '{}': xtf={} (zero-copy, no copy needed)",
                instance_id_,
                static_cast<void *>(d_xtf_));
    }

    // Note: Stream synchronization is handled by the pipeline after all module configure_io calls
    RT_LOGC_INFO(
            PuschComponent::InnerRxModule,
            "InnerRxModule '{}': configure_io() complete (async copies queued on stream)",
            instance_id_);
}

std::vector<pipeline::PortInfo> InnerRxModule::get_outputs() const {

    const std::uint32_t num_prb = static_params_.phy_params.num_prb;

    std::vector<tensor::TensorInfo> output_tensor_info = {
            // LLR tensor
            tensor::TensorInfo(
                    tensor::TensorInfo::DataType::TensorR16F,
                    {ran::common::OFDM_SYMBOLS_NORMAL_CYCLIC_PREFIX -
                             ran::common::MAX_DMRS_OFDM_SYMBOLS,
                     static_cast<std::size_t>(num_prb) * ran::common::NUM_SUBCARRIERS_PER_PRB,
                     ran::common::MAX_UL_LAYERS,
                     ran::common::MAX_QAM_BITS}),
            // Post-EQ noise var db
            tensor::TensorInfo(
                    tensor::TensorInfo::DataType::TensorR32F, {ran::common::MAX_UES_PER_SLOT}),
            // Post-EQ SINR db
            tensor::TensorInfo(
                    tensor::TensorInfo::DataType::TensorR32F, {ran::common::MAX_UES_PER_SLOT})};

    const std::vector<pipeline::DeviceTensor> output_tensors = {
            pipeline::DeviceTensor{.device_ptr = d_llr_, .tensor_info = output_tensor_info[0]},
            pipeline::DeviceTensor{
                    .device_ptr = d_post_eq_noise_var_db_, .tensor_info = output_tensor_info[1]},
            pipeline::DeviceTensor{
                    .device_ptr = d_post_eq_sinr_db_, .tensor_info = output_tensor_info[2]}};

    std::vector<pipeline::PortInfo> output_port_info = {
            {.name = "llrs", .tensors = {output_tensors[0]}},
            {.name = "post_eq_noise_var_db", .tensors = {output_tensors[1]}},
            {.name = "post_eq_sinr_db", .tensors = {output_tensors[2]}}};

    return output_port_info;
}

// ============================================================================
// Execution - Stream Mode
// ============================================================================

void InnerRxModule::execute(cudaStream_t stream) {
    RT_LOGC_DEBUG(
            PuschComponent::InnerRxModule,
            "'{}': execute() on stream {}",
            instance_id_,
            static_cast<void *>(stream));

    const utils::NvErrc execute_result = trt_engine_->run(stream);
    if (execute_result != utils::NvErrc::Success) {
        const std::string error_msg =
                std::format("InnerRxModule '{}': TRT engine run() failed", instance_id_);
        RT_LOGC_ERROR(PuschComponent::InnerRxModule, "{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    RT_LOGC_DEBUG(PuschComponent::InnerRxModule, "'{}': Execution complete", instance_id_);
}

// ============================================================================
// Execution - Graph Mode
// ============================================================================

std::span<const CUgraphNode> InnerRxModule::add_node_to_graph(
        gsl_lite::not_null<pipeline::IGraph *> graph, std::span<const CUgraphNode> deps) {
    RT_LOGC_DEBUG(
            PuschComponent::InnerRxModule, "'{}': Adding node to pipeline graph", instance_id_);

    if (!is_warmed_up_) {
        const std::string error_msg = std::format(
                "InnerRxModule '{}': warmup() must be called before add_node_to_graph()",
                instance_id_);
        RT_LOGC_ERROR(PuschComponent::InnerRxModule, "{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    // Get the captured TensorRT graph from the capture helper
    // graph_capturer_ is guaranteed to be a CaptureStreamPrePostTrtEngEnqueue
    // when execution_mode_ is Graph (validated in constructor)
    const auto *capturer =
            dynamic_cast<const tensorrt::CaptureStreamPrePostTrtEngEnqueue *>(graph_capturer_);
    if (capturer == nullptr) {
        const std::string error_msg = std::format(
                "InnerRxModule '{}': Graph capturer is not in capture mode - "
                "this should not happen",
                instance_id_);
        RT_LOGC_ERROR(PuschComponent::InnerRxModule, "{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    CUgraph trt_graph = capturer->get_graph();
    if (trt_graph == nullptr) {
        const std::string error_msg = std::format(
                "InnerRxModule '{}': No captured TensorRT graph available - warmup() "
                "must be called first",
                instance_id_);
        RT_LOGC_ERROR(PuschComponent::InnerRxModule, "{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    // Add TensorRT subgraph as child graph node and store the handle
    trt_node_ = graph->add_child_graph_node(deps, trt_graph);

    RT_LOGC_INFO(
            PuschComponent::InnerRxModule,
            "'{}': TensorRT subgraph added to pipeline graph "
            "as node {}",
            instance_id_,
            static_cast<void *>(trt_node_));
    return {&trt_node_, 1};
}

void InnerRxModule::update_graph_node_params(
        [[maybe_unused]] CUgraphExec exec, [[maybe_unused]] const pipeline::DynamicParams &params) {
    RT_LOGC_DEBUG(PuschComponent::InnerRxModule, "'{}': update_graph_node_params()", instance_id_);

    // For TensorRT, dynamic parameters would be updated via the runtime
    RT_LOGC_DEBUG(
            PuschComponent::InnerRxModule,
            "'{}': update_graph_node_params() - no-op",
            instance_id_);
}

} // namespace ran::pusch
