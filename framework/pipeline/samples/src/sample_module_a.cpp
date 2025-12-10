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

// Execution flow: See
// framework/pipeline/samples/docs/tensorrt-relu-pipeline-architecture.md Functions
// below are ordered by execution sequence for readability

#include <algorithm>
#include <format>
#include <stdexcept>
#include <utility>

#include <quill/LogMacros.h>

#include <wise_enum.h>

#include <cuda_runtime_api.h>

#include "log/rt_log_macros.hpp"
#include "pipeline/igraph.hpp"
#include "pipeline/pipeline_memory_manager.hpp"
#include "sample_module_a.hpp"
#include "tensor/data_types.hpp"
#include "tensorrt/trt_engine.hpp"
#include "tensorrt/trt_engine_logger.hpp"
#include "tensorrt/trt_engine_params.hpp"
#include "tensorrt/trt_pre_post_enqueue_stream_cap.hpp"
#include "utils/error_macros.hpp"
#include "utils/errors.hpp"

namespace framework::pipelines::samples {

// Namespace alias for compatibility with framework reorganization
namespace pipeline = ::framework::pipeline;
namespace tensor = ::framework::tensor;
namespace tensorrt = ::framework::tensorrt;

// ============================================================================
// Construction
// ============================================================================

SampleModuleA::SampleModuleA(std::string instance_id, const StaticParams &params)
        : instance_id_(std::move(instance_id)), tensor_size_(params.tensor_size),
          tensor_bytes_(params.tensor_size * sizeof(float)), execution_mode_(params.execution_mode),
          trt_logger_(std::make_unique<tensorrt::TrtLogger>()) {

    RT_LOG_INFO(
            "SampleModuleA: Constructing instance '{}', tensor_size={}, "
            "engine='{}', execution_mode={}",
            instance_id_,
            tensor_size_,
            params.trt_engine_path,
            ::wise_enum::to_string(execution_mode_));

    if (tensor_size_ == 0) {
        const std::string error_msg =
                std::format("SampleModuleA '{}': tensor_size cannot be zero", instance_id_);
        RT_LOG_ERROR("{}", error_msg);
        throw std::invalid_argument(error_msg);
    }

    if (params.trt_engine_path.empty()) {
        const std::string error_msg =
                std::format("SampleModuleA '{}': trt_engine_path cannot be empty", instance_id_);
        RT_LOG_ERROR("{}", error_msg);
        throw std::invalid_argument(error_msg);
    }

    // Configure TensorRT engine parameters
    // Simple add operation: result0 = arg0 + arg1
    // Note: The TRT engine uses "arg0", "arg1", "result0" as tensor names
    const tensorrt::MLIRTensorParams input0_params{
            .name = "arg0", .data_type = tensor::TensorR32F, .rank = 1, .dims = {tensor_size_}};

    const tensorrt::MLIRTensorParams input1_params{
            .name = "arg1", .data_type = tensor::TensorR32F, .rank = 1, .dims = {tensor_size_}};

    const tensorrt::MLIRTensorParams output_params{
            .name = "result0", .data_type = tensor::TensorR32F, .rank = 1, .dims = {tensor_size_}};

    const std::vector<tensorrt::MLIRTensorParams> inputs = {input0_params, input1_params};
    const std::vector<tensorrt::MLIRTensorParams> outputs = {output_params};

    // Create TensorRT engine with engine file path
    auto tensorrt_runtime =
            std::make_unique<tensorrt::TrtEngine>(params.trt_engine_path, *trt_logger_);

    // Create graph capture helper for CUDA graph integration
    auto graph_capturer = std::make_unique<tensorrt::CaptureStreamPrePostTrtEngEnqueue>();
    graph_capturer_ = graph_capturer.get(); // Keep non-owning pointer

    // Create MLIR TRT engine with the runtime and capture helper (takes ownership
    // of both)
    trt_engine_ = std::make_unique<tensorrt::MLIRTrtEngine>(
            inputs, outputs, std::move(tensorrt_runtime), std::move(graph_capturer));

    RT_LOG_DEBUG("SampleModuleA '{}': Constructor complete", instance_id_);
}

// ============================================================================
// Interface Access
// ============================================================================

pipeline::IStreamExecutor *SampleModuleA::as_stream_executor() { return this; }

pipeline::IGraphNodeProvider *SampleModuleA::as_graph_node_provider() { return this; }

// ============================================================================
// Port Introspection (called during pipeline construction)
// ============================================================================

std::vector<std::string> SampleModuleA::get_input_port_names() const {
    return {"input0", "input1"};
}

std::vector<std::string> SampleModuleA::get_output_port_names() const { return {"output"}; }

std::vector<tensor::TensorInfo>
SampleModuleA::get_input_tensor_info(std::string_view port_name) const {
    if (port_name == "input0" || port_name == "input1") {
        return {tensor::TensorInfo(tensor::TensorInfo::DataType::TensorR32F, {tensor_size_})};
    }

    const std::string error_msg =
            std::format("SampleModuleA '{}': Unknown input port '{}'", instance_id_, port_name);
    RT_LOG_ERROR("{}", error_msg);
    throw std::invalid_argument(error_msg);
}

std::vector<tensor::TensorInfo>
SampleModuleA::get_output_tensor_info(std::string_view port_name) const {
    if (port_name != "output") {
        const std::string error_msg = std::format(
                "SampleModuleA '{}': Unknown output port '{}'", instance_id_, port_name);
        RT_LOG_ERROR("{}", error_msg);
        throw std::invalid_argument(error_msg);
    }

    return {tensor::TensorInfo(tensor::TensorInfo::DataType::TensorR32F, {tensor_size_})};
}

// ============================================================================
// Memory Configuration (called before setup())
// ============================================================================

pipeline::InputPortMemoryCharacteristics
SampleModuleA::get_input_memory_characteristics(std::string_view port_name) const {
    if (port_name == "input0" || port_name == "input1") {
        // Graph mode: requires fixed addresses for CUDA graph capture
        // Stream mode: flexible, uses set_tensor_address() per tick
        const bool requires_fixed = (execution_mode_ == pipeline::ExecutionMode::Graph);
        return pipeline::InputPortMemoryCharacteristics{
                .requires_fixed_address_for_zero_copy = requires_fixed};
    }

    const std::string error_msg =
            std::format("SampleModuleA '{}': Unknown input port '{}'", instance_id_, port_name);
    RT_LOG_ERROR("{}", error_msg);
    throw std::invalid_argument(error_msg);
}

pipeline::OutputPortMemoryCharacteristics
SampleModuleA::get_output_memory_characteristics(std::string_view port_name) const {
    if (port_name == "output") {
        // Output (d_output_) is allocated once in setup_memory(), never changes
        return pipeline::OutputPortMemoryCharacteristics{
                .provides_fixed_address_for_zero_copy = true};
    }

    const std::string error_msg =
            std::format("SampleModuleA '{}': Unknown output port '{}'", instance_id_, port_name);
    RT_LOG_ERROR("{}", error_msg);
    throw std::invalid_argument(error_msg);
}

void SampleModuleA::set_connection_copy_mode(
        std::string_view port_name, pipeline::ConnectionCopyMode mode) {
    const bool zero_copy = (mode == pipeline::ConnectionCopyMode::ZeroCopy);

    if (port_name == "input0") {
        input0_upstream_is_fixed_ = zero_copy;
        RT_LOG_INFO(
                "SampleModuleA '{}': input0 connection copy mode = {}",
                instance_id_,
                ::wise_enum::to_string(mode));
    } else if (port_name == "input1") {
        input1_upstream_is_fixed_ = zero_copy;
        RT_LOG_INFO(
                "SampleModuleA '{}': input1 connection copy mode = {}",
                instance_id_,
                ::wise_enum::to_string(mode));
    } else {
        const std::string error_msg =
                std::format("SampleModuleA '{}': Unknown input port '{}'", instance_id_, port_name);
        RT_LOG_ERROR("{}", error_msg);
        throw std::invalid_argument(error_msg);
    }
}

pipeline::ModuleMemoryRequirements SampleModuleA::get_requirements() const {
    pipeline::ModuleMemoryRequirements reqs{};

    // Calculate required memory with proper alignment between tensors
    // Must match the allocation logic in setup_memory()
    std::size_t total_bytes = 0;
    std::size_t num_tensors = 0;

    // Input0: allocate only if NOT using zero-copy
    if (!input0_upstream_is_fixed_) {
        total_bytes = pipeline::align_memory_offset(total_bytes + tensor_bytes_, MEMORY_ALIGNMENT);
        num_tensors++;
    }

    // Input1: allocate only if NOT using zero-copy
    if (!input1_upstream_is_fixed_) {
        total_bytes = pipeline::align_memory_offset(total_bytes + tensor_bytes_, MEMORY_ALIGNMENT);
        num_tensors++;
    }

    // Always allocate output (no alignment needed after last buffer)
    total_bytes += tensor_bytes_;
    num_tensors++;

    reqs.device_tensor_bytes = total_bytes;
    reqs.alignment = MEMORY_ALIGNMENT;

    RT_LOG_DEBUG(
            "SampleModuleA '{}': Memory requirements - device={} bytes ({} "
            "tensor{}, input0_zero_copy={}, input1_zero_copy={})",
            instance_id_,
            reqs.device_tensor_bytes,
            num_tensors,
            num_tensors == 1 ? "" : "s",
            input0_upstream_is_fixed_,
            input1_upstream_is_fixed_);

    return reqs;
}

// ============================================================================
// Setup Phase (one-time initialization)
// ============================================================================

void SampleModuleA::setup_memory(const pipeline::ModuleMemorySlice &memory_slice) {
    RT_LOG_INFO("SampleModuleA '{}': setup_memory() called", instance_id_);

    mem_slice_ = memory_slice;
    std::byte *base_ptr = mem_slice_.device_tensor_ptr;
    std::size_t offset = 0;

    // Conditional allocation based on zero-copy mode
    // Layout (copy mode):     [input0][padding][input1][padding][output]
    // Layout (zero-copy):     [output]  (inputs use upstream addresses)
    // Each buffer starts at a 256-byte aligned offset

    // Input0: allocate only if NOT using zero-copy
    if (!input0_upstream_is_fixed_) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast,cppcoreguidelines-pro-bounds-pointer-arithmetic)
        d_input0_ = reinterpret_cast<float *>(base_ptr + offset);
        offset += tensor_bytes_;
        offset = pipeline::align_memory_offset(offset, MEMORY_ALIGNMENT);
        RT_LOG_INFO(
                "SampleModuleA '{}': Allocated input0={} (copy mode)",
                instance_id_,
                static_cast<void *>(d_input0_));
    } else {
        d_input0_ = nullptr; // Will use upstream address directly
        RT_LOG_INFO("SampleModuleA '{}': input0 in zero-copy mode (no allocation)", instance_id_);
    }

    // Input1: allocate only if NOT using zero-copy
    if (!input1_upstream_is_fixed_) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast,cppcoreguidelines-pro-bounds-pointer-arithmetic)
        d_input1_ = reinterpret_cast<float *>(base_ptr + offset);
        offset += tensor_bytes_;
        offset = pipeline::align_memory_offset(offset, MEMORY_ALIGNMENT);
        RT_LOG_INFO(
                "SampleModuleA '{}': Allocated input1={} (copy mode)",
                instance_id_,
                static_cast<void *>(d_input1_));
    } else {
        d_input1_ = nullptr; // Will use upstream address directly
        RT_LOG_INFO("SampleModuleA '{}': input1 in zero-copy mode (no allocation)", instance_id_);
    }

    // Always allocate output (no alignment needed after last buffer)
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast,cppcoreguidelines-pro-bounds-pointer-arithmetic)
    d_output_ = reinterpret_cast<float *>(base_ptr + offset);
    RT_LOG_INFO(
            "SampleModuleA '{}': Allocated output={}",
            instance_id_,
            static_cast<void *>(d_output_));
}

void SampleModuleA::set_inputs(std::span<const pipeline::PortInfo> inputs) {
    RT_LOG_DEBUG(
            "SampleModuleA '{}': set_inputs() called with {} ports", instance_id_, inputs.size());

    // Extract external device pointers from PortInfo
    // Zero-copy mode: Use upstream pointers directly as TRT input addresses
    // Copy mode: Store for later copy in setup_tick()
    for (const auto &port : inputs) {
        if (port.tensors.empty()) {
            const std::string error_msg = std::format(
                    "SampleModuleA '{}': Port '{}' has no tensors", instance_id_, port.name);
            RT_LOG_ERROR("{}", error_msg);
            throw std::invalid_argument(error_msg);
        }

        if (port.name == "input0") {
            external_input0_data_ = port.tensors[0].device_ptr;

            // Zero-copy: Use upstream address directly
            if (input0_upstream_is_fixed_) {
                // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
                d_input0_ = const_cast<float *>(static_cast<const float *>(external_input0_data_));
                RT_LOG_INFO(
                        "SampleModuleA '{}': input0={} (zero-copy mode)",
                        instance_id_,
                        static_cast<void *>(d_input0_));
            } else {
                RT_LOG_DEBUG(
                        "SampleModuleA '{}': input0={} (will copy in configure_io)",
                        instance_id_,
                        external_input0_data_);
            }
        } else if (port.name == "input1") {
            external_input1_data_ = port.tensors[0].device_ptr;

            // Zero-copy: Use upstream address directly
            if (input1_upstream_is_fixed_) {
                // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
                d_input1_ = const_cast<float *>(static_cast<const float *>(external_input1_data_));
                RT_LOG_INFO(
                        "SampleModuleA '{}': input1={} (zero-copy mode)",
                        instance_id_,
                        static_cast<void *>(d_input1_));
            } else {
                RT_LOG_DEBUG(
                        "SampleModuleA '{}': input1={} (will copy in configure_io)",
                        instance_id_,
                        external_input1_data_);
            }
        } else {
            const std::string error_msg = std::format(
                    "SampleModuleA '{}': Unknown input port '{}'", instance_id_, port.name);
            RT_LOG_ERROR("{}", error_msg);
            throw std::invalid_argument(error_msg);
        }
    }

    RT_LOG_INFO(
            "SampleModuleA '{}': Input connections established - input0={} "
            "(zero_copy={}), input1={} (zero_copy={}). Call warmup() to load "
            "TRT engine.",
            instance_id_,
            static_cast<void *>(d_input0_),
            input0_upstream_is_fixed_,
            static_cast<void *>(d_input1_),
            input1_upstream_is_fixed_);
}

void SampleModuleA::warmup(cudaStream_t stream) {
    RT_LOG_INFO(
            "SampleModuleA '{}': warmup(stream={}) called",
            instance_id_,
            static_cast<void *>(stream));

    // Skip if already warmed up
    if (is_warmed_up_) {
        RT_LOG_DEBUG("SampleModuleA '{}': Already warmed up, skipping", instance_id_);
        return;
    }

    // Validate that all tensor pointers are set
    // - Output: Always allocated in setup_memory()
    // - Inputs (copy mode): Allocated in setup_memory()
    // - Inputs (zero-copy mode): Assigned in set_inputs()
    if (d_input0_ == nullptr || d_input1_ == nullptr || d_output_ == nullptr) {
        const std::string error_msg = std::format(
                "SampleModuleA '{}': warmup() called before tensor "
                "pointers established. "
                "Ensure setup_memory() and set_inputs() have been called.",
                instance_id_);
        RT_LOG_ERROR("{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    // Validate that inputs have been set
    if (external_input0_data_ == nullptr || external_input1_data_ == nullptr) {
        const std::string error_msg = std::format(
                "SampleModuleA '{}': warmup() called before set_inputs(). "
                "Input connections must be established before warmup.",
                instance_id_);
        RT_LOG_ERROR("{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    // Configure TRT engine with FIXED tensor addresses (one-time setup)
    // Use d_input0_/d_input1_ (our internal fixed buffers), not external pointers
    const std::vector<void *> input_buffers = {d_input0_, d_input1_};
    const std::vector<void *> output_buffers = {d_output_};

    RT_LOG_INFO(
            "SampleModuleA '{}': Performing one-time warmup "
            "(loads engine to device, captures CUDA graph for graph mode). "
            "Fixed tensor addresses: input0={}, input1={}, output={}, stream={}",
            instance_id_,
            static_cast<void *>(d_input0_),
            static_cast<void *>(d_input1_),
            static_cast<void *>(d_output_),
            static_cast<void *>(stream));

    RT_LOG_DEBUG("SampleModuleA '{}': Calling TRT engine setup()", instance_id_);
    const ::framework::utils::NvErrc setup_result =
            trt_engine_->setup(input_buffers, output_buffers);
    if (setup_result != ::framework::utils::NvErrc::Success) {
        const std::string error_msg =
                std::format("SampleModuleA '{}': TRT engine setup() failed", instance_id_);
        RT_LOG_ERROR("{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    // Use provided stream for warmup/graph capture
    // Note: TensorRT graph capture requires a non-default stream (cannot use
    // cudaStreamDefault)
    const ::framework::utils::NvErrc warmup_result = trt_engine_->warmup(stream);
    if (warmup_result != ::framework::utils::NvErrc::Success) {
        const std::string error_msg =
                std::format("SampleModuleA '{}': TRT engine warmup() failed", instance_id_);
        RT_LOG_ERROR("{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    // Synchronize stream to ensure graph capture is complete
    FRAMEWORK_CUDA_RUNTIME_CHECK_THROW(cudaStreamSynchronize(stream));

    is_warmed_up_ = true;
    RT_LOG_INFO(
            "SampleModuleA '{}': Warmup complete - TRT engine ready for "
            "execution (graph captured for graph mode, stream mode ready)",
            instance_id_);
}

// ============================================================================
// Per-Iteration Configuration
// ============================================================================

void SampleModuleA::configure_io(const pipeline::DynamicParams &params, cudaStream_t stream) {
    RT_LOG_DEBUG("SampleModuleA '{}': configure_io()", instance_id_);

    // Suppress unused parameter warning
    (void)params;

    // Copy external input data to fixed internal buffers using stream-aware operations
    // Zero-copy mode: Skip copy, already using upstream addresses directly

    // Validate external input ports have been set
    if (external_input0_data_ == nullptr || external_input1_data_ == nullptr) {
        const std::string error_msg = std::format(
                "SampleModuleA '{}': Input ports not set before configure_io()", instance_id_);
        RT_LOG_ERROR("{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    // Input0: Copy only if NOT in zero-copy mode
    if (!input0_upstream_is_fixed_) {
        FRAMEWORK_CUDA_RUNTIME_CHECK_THROW(cudaMemcpyAsync(
                d_input0_, external_input0_data_, tensor_bytes_, cudaMemcpyDeviceToDevice, stream));
        RT_LOG_DEBUG(
                "SampleModuleA '{}': Copied input0 {} -> {}",
                instance_id_,
                external_input0_data_,
                static_cast<void *>(d_input0_));
    } else {
        RT_LOG_DEBUG(
                "SampleModuleA '{}': input0={} (zero-copy, no copy needed)",
                instance_id_,
                static_cast<void *>(d_input0_));
    }

    // Input1: Copy only if NOT in zero-copy mode
    if (!input1_upstream_is_fixed_) {
        FRAMEWORK_CUDA_RUNTIME_CHECK_THROW(cudaMemcpyAsync(
                d_input1_, external_input1_data_, tensor_bytes_, cudaMemcpyDeviceToDevice, stream));
        RT_LOG_DEBUG(
                "SampleModuleA '{}': Copied input1 {} -> {}",
                instance_id_,
                external_input1_data_,
                static_cast<void *>(d_input1_));
    } else {
        RT_LOG_DEBUG(
                "SampleModuleA '{}': input1={} (zero-copy, no copy needed)",
                instance_id_,
                static_cast<void *>(d_input1_));
    }

    // Synchronize stream to ensure input copies complete before execution
    FRAMEWORK_CUDA_RUNTIME_CHECK_THROW(cudaStreamSynchronize(stream));

    RT_LOG_INFO(
            "SampleModuleA '{}': configure_io() complete - input0={} "
            "(zero_copy={}), input1={} (zero_copy={})",
            instance_id_,
            static_cast<void *>(d_input0_),
            input0_upstream_is_fixed_,
            static_cast<void *>(d_input1_),
            input1_upstream_is_fixed_);
}

std::vector<pipeline::PortInfo> SampleModuleA::get_outputs() const {
    RT_LOG_INFO(
            "SampleModuleA '{}': get_outputs() - returning d_output_={}",
            instance_id_,
            static_cast<void *>(d_output_));

    // Create TensorInfo for output
    const tensor::TensorInfo output_info(tensor::TensorInfo::DataType::TensorR32F, {tensor_size_});

    // Create DeviceTensor
    pipeline::DeviceTensor output_tensor{.device_ptr = d_output_, .tensor_info = output_info};

    // Return PortInfo (characteristics use defaults)
    return {pipeline::PortInfo{.name = "output", .tensors = {output_tensor}}};
}

// ============================================================================
// Execution - Stream Mode
// ============================================================================

void SampleModuleA::execute(cudaStream_t stream) {
    RT_LOG_DEBUG(
            "SampleModuleA '{}': execute() on stream {}",
            instance_id_,
            static_cast<void *>(stream));

    // Execute TensorRT inference using run()
    const ::framework::utils::NvErrc run_result = trt_engine_->run(stream);
    if (run_result != ::framework::utils::NvErrc::Success) {
        const std::string error_msg =
                std::format("SampleModuleA '{}': TRT engine run() failed", instance_id_);
        RT_LOG_ERROR("{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    RT_LOG_DEBUG("SampleModuleA '{}': TensorRT execution complete", instance_id_);
}

// ============================================================================
// Execution - Graph Mode
// ============================================================================

std::span<const CUgraphNode> SampleModuleA::add_node_to_graph(
        gsl_lite::not_null<pipeline::IGraph *> graph, std::span<const CUgraphNode> deps) {
    RT_LOG_DEBUG("SampleModuleA '{}': Adding TensorRT subgraph to pipeline graph", instance_id_);

    // Ensure warmup has been called (requires real tensor addresses)
    if (!is_warmed_up_) {
        const std::string error_msg = std::format(
                "SampleModuleA '{}': set_inputs() must be called before "
                "add_node_to_graph(). Warmup with real tensor addresses is required "
                "before graph capture can be retrieved.",
                instance_id_);
        RT_LOG_ERROR("{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    // Get the captured TensorRT graph from the capture helper
    CUgraph trt_graph = graph_capturer_->get_graph();
    if (trt_graph == nullptr) {
        const std::string error_msg = std::format(
                "SampleModuleA '{}': No captured TensorRT graph available - warmup() "
                "must be called first",
                instance_id_);
        RT_LOG_ERROR("{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    // Add TensorRT subgraph as child graph node and store the handle
    trt_node_ = graph->add_child_graph_node(deps, trt_graph);

    RT_LOG_INFO(
            "SampleModuleA '{}': TensorRT subgraph added to pipeline graph "
            "as node {}",
            instance_id_,
            static_cast<void *>(trt_node_));

    return {&trt_node_, 1};
}

void SampleModuleA::update_graph_node_params(
        [[maybe_unused]] CUgraphExec exec, [[maybe_unused]] const pipeline::DynamicParams &params) {
    // For TensorRT, dynamic parameters would be updated via the runtime
    // For this demo, we don't have dynamic parameters
    RT_LOG_DEBUG("SampleModuleA '{}': update_graph_node_params() - no-op", instance_id_);
}

} // namespace framework::pipelines::samples
