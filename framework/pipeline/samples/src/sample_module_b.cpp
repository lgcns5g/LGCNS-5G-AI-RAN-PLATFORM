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
#include <vector_types.h>

#include "log/rt_log_macros.hpp"
#include "pipeline/igraph.hpp"
#include "sample_module_b.hpp"
#include "sample_module_b_kernel.cuh"
#include "tensor/data_types.hpp"
#include "utils/error_macros.hpp"

namespace framework::pipelines::samples {

// Namespace alias for compatibility with framework reorganization
namespace pipeline = ::framework::pipeline;
namespace tensor = ::framework::tensor;

// ============================================================================
// Construction
// ============================================================================

SampleModuleB::SampleModuleB(std::string instance_id, const StaticParams &params)
        : instance_id_(std::move(instance_id)), tensor_size_(params.tensor_size),
          tensor_bytes_(params.tensor_size * sizeof(float)),
          execution_mode_(params.execution_mode) {

    RT_LOG_INFO(
            "SampleModuleB: Constructing instance '{}', tensor_size={}, "
            "execution_mode={}",
            instance_id_,
            tensor_size_,
            execution_mode_ == pipeline::ExecutionMode::Graph ? "Graph" : "Stream");

    if (tensor_size_ == 0) {
        const std::string error_msg =
                std::format("SampleModuleB '{}': tensor_size cannot be zero", instance_id_);
        RT_LOG_ERROR("{}", error_msg);
        throw std::invalid_argument(error_msg);
    }

    // Calculate grid size for kernel launches
    grid_size_ = (tensor_size_ + BLOCK_SIZE - 1) / BLOCK_SIZE;

    RT_LOG_DEBUG("SampleModuleB '{}': Constructor complete", instance_id_);
}

// ============================================================================
// Interface Access
// ============================================================================

pipeline::IStreamExecutor *SampleModuleB::as_stream_executor() { return this; }

pipeline::IGraphNodeProvider *SampleModuleB::as_graph_node_provider() { return this; }

// ============================================================================
// Port Introspection (called during pipeline construction)
// ============================================================================

std::vector<std::string> SampleModuleB::get_input_port_names() const { return {"input"}; }

std::vector<std::string> SampleModuleB::get_output_port_names() const { return {"output"}; }

std::vector<tensor::TensorInfo>
SampleModuleB::get_input_tensor_info(std::string_view port_name) const {
    if (port_name != "input") {
        const std::string error_msg =
                std::format("SampleModuleB '{}': Unknown input port '{}'", instance_id_, port_name);
        RT_LOG_ERROR("{}", error_msg);
        throw std::invalid_argument(error_msg);
    }

    return {tensor::TensorInfo(tensor::TensorInfo::DataType::TensorR32F, {tensor_size_})};
}

std::vector<tensor::TensorInfo>
SampleModuleB::get_output_tensor_info(std::string_view port_name) const {
    if (port_name != "output") {
        const std::string error_msg = std::format(
                "SampleModuleB '{}': Unknown output port '{}'", instance_id_, port_name);
        RT_LOG_ERROR("{}", error_msg);
        throw std::invalid_argument(error_msg);
    }

    return {tensor::TensorInfo(tensor::TensorInfo::DataType::TensorR32F, {tensor_size_})};
}

// ============================================================================
// Memory Configuration (called before setup())
// ============================================================================

pipeline::InputPortMemoryCharacteristics
SampleModuleB::get_input_memory_characteristics(std::string_view port_name) const {
    if (port_name == "input") {
        // ModuleB is flexible: uses dynamic kernel descriptors updated per tick
        // Can accept any address (fixed or changing) - doesn't require fixed
        return pipeline::InputPortMemoryCharacteristics{
                .requires_fixed_address_for_zero_copy = false};
    }

    const std::string error_msg =
            std::format("SampleModuleB '{}': Unknown input port '{}'", instance_id_, port_name);
    RT_LOG_ERROR("{}", error_msg);
    throw std::invalid_argument(error_msg);
}

pipeline::OutputPortMemoryCharacteristics
SampleModuleB::get_output_memory_characteristics(std::string_view port_name) const {
    if (port_name == "output") {
        // Output (d_output_) is allocated once in setup_memory(), never changes
        return pipeline::OutputPortMemoryCharacteristics{
                .provides_fixed_address_for_zero_copy = true};
    }

    const std::string error_msg =
            std::format("SampleModuleB '{}': Unknown output port '{}'", instance_id_, port_name);
    RT_LOG_ERROR("{}", error_msg);
    throw std::invalid_argument(error_msg);
}

pipeline::ModuleMemoryRequirements SampleModuleB::get_requirements() const {
    pipeline::ModuleMemoryRequirements reqs{};

    // Kernel descriptor requirements
    reqs.static_kernel_descriptor_bytes = sizeof(SampleModuleBStaticKernelParams);
    reqs.dynamic_kernel_descriptor_bytes = sizeof(SampleModuleBDynamicKernelParams);

    // Device tensor requirements
    reqs.device_tensor_bytes = tensor_bytes_;
    reqs.alignment = MEMORY_ALIGNMENT;

    RT_LOG_DEBUG(
            "SampleModuleB '{}': Memory requirements - static_desc={}, "
            "dynamic_desc={}, device={} bytes",
            instance_id_,
            reqs.static_kernel_descriptor_bytes,
            reqs.dynamic_kernel_descriptor_bytes,
            reqs.device_tensor_bytes);

    return reqs;
}

// ============================================================================
// Setup Phase (one-time initialization)
// ============================================================================

void SampleModuleB::setup_memory(const pipeline::ModuleMemorySlice &memory_slice) {
    RT_LOG_INFO(
            "SampleModuleB '{}': setup_memory() called - static_gpu={}, "
            "dynamic_gpu={}, tensor={}",
            instance_id_,
            static_cast<void *>(memory_slice.static_kernel_descriptor_gpu_ptr),
            static_cast<void *>(memory_slice.dynamic_kernel_descriptor_gpu_ptr),
            static_cast<void *>(memory_slice.device_tensor_ptr));

    mem_slice_ = memory_slice;

    // Allocate output tensor from device memory slice
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    d_output_ = reinterpret_cast<float *>(mem_slice_.device_tensor_ptr);

    RT_LOG_DEBUG(
            "SampleModuleB '{}': Allocated output tensor at {}",
            instance_id_,
            static_cast<void *>(d_output_));

    // Create kernel descriptor accessor
    kernel_desc_mgr_ = std::make_unique<pipeline::KernelDescriptorAccessor>(memory_slice);

    // Create static kernel parameters in pinned memory using placement new
    static_params_cpu_ptr_ =
            &kernel_desc_mgr_->create_static_param<SampleModuleBStaticKernelParams>(0);

    // Initialize static kernel parameters
    static_params_cpu_ptr_->output = d_output_;
    static_params_cpu_ptr_->size = tensor_size_;

    // Create dynamic kernel parameters in pinned memory
    dynamic_params_cpu_ptr_ =
            &kernel_desc_mgr_->create_dynamic_param<SampleModuleBDynamicKernelParams>(0);

    // Initialize dynamic parameters (will be updated in configure_io)
    dynamic_params_cpu_ptr_->input = nullptr;

    // Get device pointers for kernel parameters
    static_params_gpu_ptr_ =
            kernel_desc_mgr_->get_static_device_ptr<SampleModuleBStaticKernelParams>(0);
    dynamic_params_gpu_ptr_ =
            kernel_desc_mgr_->get_dynamic_device_ptr<SampleModuleBDynamicKernelParams>(0);

    RT_LOG_INFO(
            "SampleModuleB '{}': Kernel descriptors initialized - "
            "static_cpu={}, static_gpu={}, dynamic_cpu={}, dynamic_gpu={}",
            instance_id_,
            static_cast<void *>(static_params_cpu_ptr_),
            static_cast<void *>(static_params_gpu_ptr_),
            static_cast<void *>(dynamic_params_cpu_ptr_),
            static_cast<void *>(dynamic_params_gpu_ptr_));

    // Configure kernel launch parameters using framework helper functions
    RT_LOG_INFO("SampleModuleB '{}': Configuring kernel launch parameters", instance_id_);

    // Setup kernel function pointer
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    pipeline::setup_kernel_function(
            kernel_config_,
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
            reinterpret_cast<const void *>(sample_module_b_kernel));

    // Setup kernel dimensions
    pipeline::setup_kernel_dimensions(
            kernel_config_,
            dim3(static_cast<unsigned int>(grid_size_), 1, 1),
            dim3(BLOCK_SIZE, 1, 1));

    // Setup kernel arguments (indirection pattern)
    pipeline::setup_kernel_arguments(
            kernel_config_, static_params_gpu_ptr_, dynamic_params_gpu_ptr_);

    RT_LOG_INFO(
            "SampleModuleB '{}': Kernel launch configuration complete - "
            "grid={}, block={}",
            instance_id_,
            grid_size_,
            BLOCK_SIZE);
}

void SampleModuleB::set_inputs(std::span<const pipeline::PortInfo> inputs) {
    RT_LOG_DEBUG(
            "SampleModuleB '{}': set_inputs() called with {} ports", instance_id_, inputs.size());

    // Extract device pointer from PortInfo
    for (const auto &port : inputs) {
        if (port.tensors.empty()) {
            const std::string error_msg = std::format(
                    "SampleModuleB '{}': Port '{}' has no tensors", instance_id_, port.name);
            RT_LOG_ERROR("{}", error_msg);
            throw std::invalid_argument(error_msg);
        }

        if (port.name == "input") {
            d_input_ = port.tensors[0].device_ptr;
            RT_LOG_INFO(
                    "SampleModuleB '{}': set_inputs() - Set d_input_={}", instance_id_, d_input_);
        } else {
            const std::string error_msg = std::format(
                    "SampleModuleB '{}': Unknown input port '{}'", instance_id_, port.name);
            RT_LOG_ERROR("{}", error_msg);
            throw std::invalid_argument(error_msg);
        }
    }
}

void SampleModuleB::warmup([[maybe_unused]] cudaStream_t stream) {
    RT_LOG_INFO(
            "SampleModuleB '{}': warmup(stream={}) called - no warmup "
            "required for simple CUDA kernel",
            instance_id_,
            static_cast<void *>(stream));
    // No-op: Simple CUDA kernels don't need warmup
}

// ============================================================================
// Per-Iteration Configuration
// ============================================================================

void SampleModuleB::configure_io(
        const pipeline::DynamicParams &params, [[maybe_unused]] cudaStream_t stream) {
    RT_LOG_INFO(
            "SampleModuleB '{}': configure_io() - d_input_={}, "
            "updating dynamic_params_cpu_ptr_->input",
            instance_id_,
            d_input_);

    // Suppress unused parameter warning
    (void)params;

    // Update dynamic kernel parameters with current input pointer
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    dynamic_params_cpu_ptr_->input = reinterpret_cast<const float *>(d_input_);

    RT_LOG_INFO(
            "SampleModuleB '{}': Dynamic params CPU: input={} (should match "
            "d_input_={}). Pipeline will bulk-copy to device.",
            instance_id_,
            static_cast<const void *>(dynamic_params_cpu_ptr_->input),
            d_input_);
}

std::vector<pipeline::PortInfo> SampleModuleB::get_outputs() const {
    // Create TensorInfo for output
    const tensor::TensorInfo output_info(tensor::TensorInfo::DataType::TensorR32F, {tensor_size_});

    // Create DeviceTensor
    const pipeline::DeviceTensor output_tensor{.device_ptr = d_output_, .tensor_info = output_info};

    // Return PortInfo (characteristics use defaults)
    return {pipeline::PortInfo{.name = "output", .tensors = {output_tensor}}};
}

// ============================================================================
// Execution - Stream Mode
// ============================================================================

void SampleModuleB::execute(cudaStream_t stream) {
    RT_LOG_DEBUG(
            "SampleModuleB '{}': execute() on stream {}",
            instance_id_,
            static_cast<void *>(stream));

    launch_relu_kernel(stream);
}

// ============================================================================
// Execution - Graph Mode
// ============================================================================

std::span<const CUgraphNode> SampleModuleB::add_node_to_graph(
        gsl_lite::not_null<pipeline::IGraph *> graph, std::span<const CUgraphNode> deps) {
    // Validate input has been set
    if (d_input_ == nullptr) {
        const std::string error_msg =
                std::format("SampleModuleB '{}': Input not set before graph capture", instance_id_);
        RT_LOG_ERROR("{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    RT_LOG_DEBUG(
            "SampleModuleB '{}': Adding kernel node to graph with {} dependencies",
            instance_id_,
            deps.size());

    // Add kernel node using kernel params from kernel_config_
    kernel_node_ = graph->add_kernel_node(deps, kernel_config_.get_kernel_params());

    RT_LOG_DEBUG(
            "SampleModuleB '{}': Kernel node added: {}",
            instance_id_,
            static_cast<void *>(kernel_node_));

    return {&kernel_node_, 1};
}

void SampleModuleB::update_graph_node_params(
        CUgraphExec exec, [[maybe_unused]] const pipeline::DynamicParams &params) {
    const auto &kernel_params = kernel_config_.get_kernel_params();
    FRAMEWORK_CUDA_DRIVER_CHECK_THROW(
            cuGraphExecKernelNodeSetParams(exec, kernel_node_, &kernel_params));

    RT_LOG_DEBUG("SampleModuleB '{}': Graph node params updated", instance_id_);
}

// ============================================================================
// Private Helpers
// ============================================================================

void SampleModuleB::launch_relu_kernel(cudaStream_t stream) {
    // Validate input has been set
    if (d_input_ == nullptr) {
        const std::string error_msg =
                std::format("SampleModuleB '{}': Input not set before execution", instance_id_);
        RT_LOG_ERROR("{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    RT_LOG_DEBUG(
            "SampleModuleB '{}': Launching ReLU kernel on stream {}",
            instance_id_,
            static_cast<void *>(stream));

    // Launch kernel using framework's launch() method
    const CUresult launch_err = kernel_config_.launch(stream);

    if (launch_err != CUDA_SUCCESS) {
        const char *error_str = nullptr;
        cuGetErrorString(launch_err, &error_str);
        const std::string error_msg = std::format(
                "SampleModuleB '{}': Kernel launch failed: {} ({})",
                instance_id_,
                static_cast<int>(launch_err),
                error_str != nullptr ? error_str : "unknown");
        RT_LOG_ERROR("{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    RT_LOG_DEBUG("SampleModuleB '{}': ReLU kernel launched", instance_id_);
}

} // namespace framework::pipelines::samples
