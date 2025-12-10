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

#ifndef FRAMEWORK_PIPELINES_SAMPLE_MODULE_A_HPP
#define FRAMEWORK_PIPELINES_SAMPLE_MODULE_A_HPP

// clang-format off
#include <driver_types.h>
#include <gsl-lite/gsl-lite.hpp>
#include <cuda.h>
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
#include "tensorrt/mlir_trt_engine.hpp"
#include "tensor/tensor_info.hpp"
#include "tensorrt/trt_engine_logger.hpp"
#include "tensorrt/trt_pre_post_enqueue_stream_cap.hpp"
#include "pipeline/types.hpp"
// clang-format on

namespace framework::pipelines::samples {

// Namespace alias for compatibility with framework reorganization
namespace pipeline = ::framework::pipeline;
namespace tensor = ::framework::tensor;
namespace tensorrt = ::framework::tensorrt;

/**
 * Sample Module A - TensorRT Element-wise Addition
 *
 * Demonstrates TensorRT integration with IModule interface.
 * Performs element-wise addition: output = input0 + input1
 *
 * Configuration:
 * - 2 input ports: "input0", "input1" (float32, configurable size)
 * - 1 output port: "output" (float32, same size as inputs)
 * - TensorRT engine for GPU execution
 */
class SampleModuleA final : public pipeline::IModule,
                            public pipeline::IAllocationInfoProvider,
                            public pipeline::IGraphNodeProvider,
                            public pipeline::IStreamExecutor {
public:
    /**
     * Static parameters for module construction
     */
    struct StaticParams {
        std::size_t tensor_size{0};  //!< Number of elements per tensor
        std::string trt_engine_path; //!< Path to serialized TensorRT engine
        pipeline::ExecutionMode execution_mode{
                pipeline::ExecutionMode::Graph}; //!< Pipeline execution mode (default:
                                                 //!< Graph for production)
    };

    /**
     * Construct module with instance ID and parameters
     *
     * @param[in] instance_id Unique identifier for this module instance
     * @param[in] params Static configuration parameters
     * @throws std::invalid_argument if tensor_size is zero or engine path empty
     */
    SampleModuleA(std::string instance_id, const StaticParams &params);

    /**
     * Destructor
     */
    ~SampleModuleA() override = default;

    // Non-copyable, non-movable
    SampleModuleA(const SampleModuleA &) = delete;
    SampleModuleA &operator=(const SampleModuleA &) = delete;
    SampleModuleA(SampleModuleA &&) = delete;
    SampleModuleA &operator=(SampleModuleA &&) = delete;

    // ========================================================================
    // IModule Interface - Identification
    // ========================================================================

    [[nodiscard]] std::string_view get_type_id() const override { return "sample_module_a"; }

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

    /**
     * Configure connection copy mode for an input port (BEFORE setup_memory)
     *
     * @param[in] port_name Input port name
     * @param[in] mode Connection copy mode (Copy or ZeroCopy)
     */
    void set_connection_copy_mode(
            std::string_view port_name, pipeline::ConnectionCopyMode mode) override;

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
     * Add TensorRT execution node to CUDA graph.
     *
     * @param[in] graph Graph interface for node creation
     * @param[in] deps Dependency nodes that must complete before this node
     * @return Span of created graph node handle (single TensorRT node)
     */
    [[nodiscard]] std::span<const CUgraphNode> add_node_to_graph(
            gsl_lite::not_null<pipeline::IGraph *> graph,
            std::span<const CUgraphNode> deps) override;

    void update_graph_node_params(CUgraphExec exec, const pipeline::DynamicParams &params) override;

private:
    std::string instance_id_;                //!< Module instance identifier
    std::size_t tensor_size_;                //!< Number of elements per tensor (for the
                                             //!< dimensions of the tensor)
    std::size_t tensor_bytes_;               //!< Bytes per tensor (size * sizeof(float))
    pipeline::ExecutionMode execution_mode_; //!< Pipeline execution mode (Stream or Graph)
    pipeline::ModuleMemorySlice mem_slice_;  //!< Assigned memory slice

    // TensorRT components
    std::unique_ptr<tensorrt::TrtLogger> trt_logger_; //!< TRT logger (Requirement for logging)
    std::unique_ptr<tensorrt::MLIRTrtEngine>
            trt_engine_; //!< TRT engine (for the TensorRT execution)
    tensorrt::CaptureStreamPrePostTrtEngEnqueue *graph_capturer_{
            nullptr}; //!< Non-owning pointer to graph capture helper (for the CUDA
                      //!< graph capture)

    // Device memory pointers (assigned during setup_memory)
    // Note: May be nullptr if zero-copy mode enabled (using upstream addresses)
    float *d_input0_{nullptr}; //!< Input 0 tensor device memory (nullptr in zero-copy mode)
    float *d_input1_{nullptr}; //!< Input 1 tensor device memory (nullptr in zero-copy mode)
    float *d_output_{nullptr}; //!< Output tensor device memory (always allocated)

    // External input data pointers (set by set_inputs)
    // In zero-copy mode: d_input0/1 point directly to these
    // In copy mode: data copied from these to d_input0/1 in setup_tick
    const void *external_input0_data_{nullptr}; //!< External input 0 pointer
    const void *external_input1_data_{nullptr}; //!< External input 1 pointer

    // Zero-copy state (set before setup_memory via set_connection_info)
    bool input0_upstream_is_fixed_{false}; //!< If true, use zero-copy for input0
    bool input1_upstream_is_fixed_{false}; //!< If true, use zero-copy for input1

    // Graph node handle (set during add_node_to_graph)
    CUgraphNode trt_node_{nullptr}; //!< TRT child graph node handle

    // State tracking
    bool is_warmed_up_{false}; //!< Track one-time warmup completion

    // Constants
    static constexpr std::size_t MEMORY_ALIGNMENT = 256; //!< GPU memory alignment
};

} // namespace framework::pipelines::samples

#endif // FRAMEWORK_PIPELINES_SAMPLE_MODULE_A_HPP
