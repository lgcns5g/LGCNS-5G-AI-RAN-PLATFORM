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

#ifndef RAN_PUSCH_INNER_RX_MODULE_HPP
#define RAN_PUSCH_INNER_RX_MODULE_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <driver_types.h>

#include <gsl-lite/gsl-lite.hpp>

#include <cuda.h>

#include "pipeline/iallocation_info_provider.hpp"
#include "pipeline/igraph.hpp"
#include "pipeline/igraph_node_provider.hpp"
#include "pipeline/imodule.hpp"
#include "pipeline/istream_executor.hpp"
#include "pipeline/types.hpp"
#include "ran_common.hpp"
#include "tensor/tensor_info.hpp"
#include "tensorrt/mlir_trt_engine.hpp"
#include "tensorrt/trt_engine_logger.hpp"
#include "tensorrt/trt_pre_post_enqueue_stream_cap.hpp"

namespace ran::pusch {

/**
 * Inner Rx Module
 *
 * Generic front-end processing module template.
 * Can be customized for specific front-end processing tasks.
 *
 * Configuration:
 * - Configurable input/output ports
 * - Support for both stream and graph execution modes
 */
class InnerRxModule final : public framework::pipeline::IModule,
                            public framework::pipeline::IAllocationInfoProvider,
                            public framework::pipeline::IGraphNodeProvider,
                            public framework::pipeline::IStreamExecutor {
public:
    /**
     * Static parameters for module construction
     */
    struct StaticParams {
        ran::common::PhyParams phy_params{}; //!< Physical layer parameters
        framework::pipeline::ExecutionMode execution_mode{
                framework::pipeline::ExecutionMode::Graph}; //!< Module execution mode
    };

    /**
     * Construct module with instance ID and parameters
     *
     * @param[in] instance_id Unique identifier for this module instance
     * @param[in] params Static configuration parameters
     */
    InnerRxModule(std::string instance_id, const StaticParams &params);

    /**
     * Destructor
     */
    ~InnerRxModule() override = default;

    // Non-copyable, non-movable
    InnerRxModule(const InnerRxModule &) = delete;
    InnerRxModule &operator=(const InnerRxModule &) = delete;
    InnerRxModule(InnerRxModule &&) = delete;
    InnerRxModule &operator=(InnerRxModule &&) = delete;

    // ========================================================================
    // IModule Interface - Identification
    // ========================================================================

    /**
     * Get module type identifier
     *
     * @return Module type ID
     */
    [[nodiscard]] std::string_view get_type_id() const override { return "inner_rx_module"; }

    /**
     * Get module instance identifier
     *
     * @return Module instance ID
     */
    [[nodiscard]] std::string_view get_instance_id() const override { return instance_id_; }

    /**
     * Get stream executor interface
     *
     * @return Pointer to stream executor interface
     */
    [[nodiscard]] framework::pipeline::IStreamExecutor *as_stream_executor() override;

    /**
     * Get graph node provider interface
     *
     * @return Pointer to graph node provider interface
     */
    [[nodiscard]] framework::pipeline::IGraphNodeProvider *as_graph_node_provider() override;

    // ========================================================================
    // IModule Interface - Port Introspection
    // ========================================================================

    /**
     * Get list of input port names
     *
     * @return Vector of input port names
     */
    [[nodiscard]] std::vector<std::string> get_input_port_names() const override;

    /**
     * Get list of output port names
     *
     * @return Vector of output port names
     */
    [[nodiscard]] std::vector<std::string> get_output_port_names() const override;

    /**
     * Get tensor information for input port
     *
     * @param[in] port_name Input port name
     * @return Vector of tensor information
     * @throws std::invalid_argument if port name is unknown
     */
    [[nodiscard]] std::vector<framework::tensor::TensorInfo>
    get_input_tensor_info(std::string_view port_name) const override;

    /**
     * Get tensor information for output port
     *
     * @param[in] port_name Output port name
     * @return Vector of tensor information
     * @throws std::invalid_argument if port name is unknown
     */
    [[nodiscard]] std::vector<framework::tensor::TensorInfo>
    get_output_tensor_info(std::string_view port_name) const override;

    // ========================================================================
    // IModule & IAllocationInfoProvider - Memory Configuration
    // ========================================================================

    /**
     * Get input port memory characteristics
     *
     * @param[in] port_name Input port name
     * @return Memory characteristics for the port
     * @throws std::invalid_argument if port name is unknown
     */
    [[nodiscard]] framework::pipeline::InputPortMemoryCharacteristics
    get_input_memory_characteristics(std::string_view port_name) const override;

    /**
     * Get output port memory characteristics
     *
     * @param[in] port_name Output port name
     * @return Memory characteristics for the port
     * @throws std::invalid_argument if port name is unknown
     */
    [[nodiscard]] framework::pipeline::OutputPortMemoryCharacteristics
    get_output_memory_characteristics(std::string_view port_name) const override;

    /**
     * Configure connection copy mode for an input port
     *
     * @param[in] port_name Input port name
     * @param[in] mode Connection copy mode (Copy or ZeroCopy)
     * @throws std::invalid_argument if port name is unknown
     */
    void set_connection_copy_mode(
            std::string_view port_name, framework::pipeline::ConnectionCopyMode mode) override;

    /**
     * Get module memory requirements
     *
     * @return Memory requirements for this module
     */
    [[nodiscard]] framework::pipeline::ModuleMemoryRequirements get_requirements() const override;

    // ========================================================================
    // IModule Interface - Setup Phase
    // ========================================================================

    /**
     * Setup module memory
     *
     * @param[in] memory_slice Memory slice allocated for this module
     */
    void setup_memory(const framework::pipeline::ModuleMemorySlice &memory_slice) override;

    /**
     * Configure input port connections
     *
     * @param[in] inputs Input port information containing data pointers
     * @throws std::invalid_argument if required inputs are missing or port names don't match with
     * expected inputs
     */
    void set_inputs(std::span<const framework::pipeline::PortInfo> inputs) override;

    /**
     * Warmup execution
     *
     * @param[in] stream CUDA stream for warmup
     * @throws std::runtime_error if warmup is called before setup_memory() and set_inputs() have
     * been called, or if the TRT engine setup() or warmup() fails
     */
    void warmup(cudaStream_t stream) override;

    // ========================================================================
    // IModule Interface - Per-Iteration Configuration
    // ========================================================================

    /**
     * Configure I/O for current iteration
     *
     * @param[in] params Dynamic parameters for this iteration
     * @param[in] stream CUDA stream for async operations during configuration
     * @throws std::runtime_error if input ports are not set before configure_io()
     */
    void
    configure_io(const framework::pipeline::DynamicParams &params, cudaStream_t stream) override;

    /**
     * Get output port information
     *
     * @return Vector of output port information
     */
    [[nodiscard]] std::vector<framework::pipeline::PortInfo> get_outputs() const override;

    // ========================================================================
    // IStreamExecutor Interface - Stream Mode Execution
    // ========================================================================

    /**
     * Execute module on stream
     *
     * @param[in] stream CUDA stream for execution
     * @throws std::runtime_error if the TRT engine run() fails
     */
    void execute(cudaStream_t stream) override;

    // ========================================================================
    // IGraphNodeProvider Interface - Graph Mode Execution
    // ========================================================================

    /**
     * Add processing node to CUDA graph
     *
     * @param[in] graph Graph interface for node creation
     * @param[in] deps Dependency nodes that must complete before this node
     * @return Span of created graph node handle (single TensorRT node)
     * @throws std::runtime_error if warmup() is not called before add_node_to_graph(), or if the
     * graph capturer is not in capture mode, or if the captured TensorRT graph is not available
     */
    [[nodiscard]] std::span<const CUgraphNode> add_node_to_graph(
            gsl_lite::not_null<framework::pipeline::IGraph *> graph,
            std::span<const CUgraphNode> deps) override;

    /**
     * Update graph node parameters
     *
     * @param[in] exec Graph executable to update
     * @param[in] params Dynamic parameters for the update
     */
    void update_graph_node_params(
            CUgraphExec exec, const framework::pipeline::DynamicParams &params) override;

private:
    StaticParams static_params_;                        //!< Static configuration parameters
    std::string instance_id_;                           //!< Module instance identifier
    framework::pipeline::ExecutionMode execution_mode_; //!< Module execution mode
    framework::pipeline::ModuleMemorySlice mem_slice_;  //!< Assigned memory slice

    // TensorRT components
    std::unique_ptr<framework::tensorrt::TrtLogger>
            trt_logger_; //!< TRT logger (required for logging)
    std::unique_ptr<framework::tensorrt::MLIRTrtEngine>
            trt_engine_; //!< TRT engine (for the TensorRT execution)
    framework::tensorrt::IPrePostTrtEngEnqueue *graph_capturer_{
            nullptr}; /*!< Non-owning observer pointer to graph capture helper.
                       *   Owned by trt_engine_. Lifetime guaranteed by member
                       *   destruction order (graph_capturer_ destroyed before trt_engine_). */

    // Device memory pointers
    __half *d_xtf_{};                 //!< XTF tensor device memory (FP16)
    __half *d_llr_{};                 //!< LLR tensor device memory (FP16)
    float *d_post_eq_noise_var_db_{}; //!< Post eq noise var db tensor device memory
    float *d_post_eq_sinr_db_{};      //!< Post eq sinr db tensor device memory

    // Input data pointers
    const void *xtf_data_{}; //!< Input tensor data pointer for XTF

    // Zero-copy state
    bool xtf_upstream_is_fixed_{true}; //!< If true, use zero-copy for XTF

    // Graph node handle
    CUgraphNode trt_node_{}; //!< TRT child graph node handle

    // State tracking
    bool is_warmed_up_{}; //!< Track one-time warmup completion

    // Constants
    static constexpr std::size_t MEMORY_ALIGNMENT = 256; //!< GPU memory alignment
};

} // namespace ran::pusch

#endif // RAN_PUSCH_INNER_RX_MODULE_HPP
