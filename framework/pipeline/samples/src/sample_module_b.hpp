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

#ifndef FRAMEWORK_PIPELINES_SAMPLE_MODULE_B_HPP
#define FRAMEWORK_PIPELINES_SAMPLE_MODULE_B_HPP

// clang-format off
#include <driver_types.h>
#include <gsl-lite/gsl-lite.hpp>
#include <cuda.h>
#include <algorithm>
#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "pipeline/iallocation_info_provider.hpp"
#include "pipeline/igraph.hpp"
#include "pipeline/igraph_node_provider.hpp"
#include "pipeline/imodule.hpp"
#include "pipeline/istream_executor.hpp"
#include "pipeline/kernel_descriptor_accessor.hpp"
#include "pipeline/kernel_launch_config.hpp"
#include "tensor/tensor_info.hpp"
#include "pipeline/types.hpp"
#include "sample_module_b_kernel.cuh"
// clang-format on

namespace framework::pipelines::samples {

// Namespace alias for compatibility with framework reorganization
namespace pipeline = ::framework::pipeline;
namespace tensor = ::framework::tensor;

/**
 * Sample Module B - CUDA ReLU Activation
 *
 * Demonstrates custom CUDA kernel integration with IModule interface.
 * Performs element-wise ReLU: output[i] = max(0, input[i])
 *
 * Configuration:
 * - 1 input port: "input" (float32, configurable size)
 * - 1 output port: "output" (float32, same size as input)
 * - Custom CUDA kernel for GPU execution
 */
class SampleModuleB final : public pipeline::IModule,
                            public pipeline::IAllocationInfoProvider,
                            public pipeline::IGraphNodeProvider,
                            public pipeline::IStreamExecutor {
public:
    /**
     * Static parameters for module construction
     */
    struct StaticParams {
        std::size_t tensor_size{0}; //!< Number of elements per tensor
        pipeline::ExecutionMode execution_mode{
                pipeline::ExecutionMode::Graph}; //!< Pipeline execution mode (default:
                                                 //!< Graph for production)
    };

    /**
     * Construct module with instance ID and parameters
     *
     * @param[in] instance_id Unique identifier for this module instance
     * @param[in] params Static configuration parameters
     * @throws std::invalid_argument if tensor_size is zero
     */
    SampleModuleB(std::string instance_id, const StaticParams &params);

    ~SampleModuleB() override = default;

    // Non-copyable, non-movable
    SampleModuleB(const SampleModuleB &) = delete;
    SampleModuleB &operator=(const SampleModuleB &) = delete;
    SampleModuleB(SampleModuleB &&) = delete;
    SampleModuleB &operator=(SampleModuleB &&) = delete;

    // ========================================================================
    // IModule Interface - Identification
    // ========================================================================

    [[nodiscard]] std::string_view get_type_id() const override { return "sample_module_b"; }

    [[nodiscard]] std::string_view get_instance_id() const override { return instance_id_; }

    [[nodiscard]] pipeline::IStreamExecutor *as_stream_executor() override;

    [[nodiscard]] pipeline::IGraphNodeProvider *as_graph_node_provider() override;

    // ========================================================================
    // IModule Interface - Port Introspection
    // ========================================================================

    [[nodiscard]] std::vector<std::string> get_input_port_names() const override;

    [[nodiscard]] std::vector<std::string> get_output_port_names() const override;

    [[nodiscard]] std::vector<tensor::TensorInfo>
    get_input_tensor_info(std::string_view port_name) const override;

    [[nodiscard]] std::vector<tensor::TensorInfo>
    get_output_tensor_info(std::string_view port_name) const override;

    // ========================================================================
    // IModule & IAllocationInfoProvider - Memory Configuration
    // ========================================================================

    [[nodiscard]] pipeline::InputPortMemoryCharacteristics
    get_input_memory_characteristics(std::string_view port_name) const override;

    [[nodiscard]] pipeline::OutputPortMemoryCharacteristics
    get_output_memory_characteristics(std::string_view port_name) const override;

    [[nodiscard]] pipeline::ModuleMemoryRequirements get_requirements() const override;

    // ========================================================================
    // IModule Interface - Setup Phase
    // ========================================================================

    void setup_memory(const pipeline::ModuleMemorySlice &memory_slice) override;

    /**
     * Configure input port connections.
     *
     * @param[in] inputs Input port information containing data pointers
     */
    void set_inputs(std::span<const pipeline::PortInfo> inputs) override;

    void warmup(cudaStream_t stream) override;

    // ========================================================================
    // IModule Interface - Per-Iteration Configuration
    // ========================================================================

    void configure_io(const pipeline::DynamicParams &params, cudaStream_t stream) override;

    [[nodiscard]] std::vector<pipeline::PortInfo> get_outputs() const override;

    // ========================================================================
    // IStreamExecutor Interface - Stream Mode Execution
    // ========================================================================

    void execute(cudaStream_t stream) override;

    // ========================================================================
    // IGraphNodeProvider Interface - Graph Mode Execution
    // ========================================================================

    /**
     * Add ReLU kernel node to CUDA graph.
     *
     * @param[in] graph Graph interface for node creation
     * @param[in] deps Dependency nodes that must complete before this node
     * @return Span of created graph node handle (single kernel node)
     */
    [[nodiscard]] std::span<const CUgraphNode> add_node_to_graph(
            gsl_lite::not_null<pipeline::IGraph *> graph,
            std::span<const CUgraphNode> deps) override;

    void update_graph_node_params(CUgraphExec exec, const pipeline::DynamicParams &params) override;

private:
    /**
     * Launch ReLU kernel
     * @param[in] stream CUDA stream for kernel launch
     */
    void launch_relu_kernel(cudaStream_t stream);

    std::string instance_id_;                //!< Module instance identifier
    std::size_t tensor_size_;                //!< Number of elements per tensor
    std::size_t tensor_bytes_;               //!< Bytes per tensor
    pipeline::ExecutionMode execution_mode_; //!< Pipeline execution mode (Stream or Graph)
    pipeline::ModuleMemorySlice mem_slice_;  //!< Assigned memory slice

    // Kernel descriptor management
    std::unique_ptr<pipeline::KernelDescriptorAccessor>
            kernel_desc_mgr_; //!< Kernel descriptor accessor
    SampleModuleBStaticKernelParams *static_params_cpu_ptr_{
            nullptr}; //!< Static params in pinned memory (CPU)
    SampleModuleBDynamicKernelParams *dynamic_params_cpu_ptr_{
            nullptr}; //!< Dynamic params in pinned memory (CPU)
    SampleModuleBStaticKernelParams *static_params_gpu_ptr_{
            nullptr}; //!< Static params on device (GPU)
    SampleModuleBDynamicKernelParams *dynamic_params_gpu_ptr_{
            nullptr}; //!< Dynamic params on device (GPU)

    // Device memory pointers (assigned during setup_memory)
    float *d_output_{nullptr}; //!< Output tensor device memory

    // Input data pointer (set by set_inputs)
    const void *d_input_{nullptr}; //!< Input pointer (from upstream module)

    // Graph node handle (set during add_node_to_graph)
    CUgraphNode kernel_node_{nullptr}; //!< CUDA kernel graph node handle

    // Kernel launch configuration using framework's DualKernelLaunchConfig
    // (2 parameters: static params, dynamic params)
    pipeline::DualKernelLaunchConfig kernel_config_; //!< Kernel configuration (set once in setup)

    // Kernel launch parameters
    static constexpr unsigned int BLOCK_SIZE = 256;
    static constexpr std::size_t MEMORY_ALIGNMENT = 256; //!< GPU memory alignment
    std::size_t grid_size_{0};                           //!< Grid size for kernel launch
};

} // namespace framework::pipelines::samples

#endif // FRAMEWORK_PIPELINES_SAMPLE_MODULE_B_HPP
