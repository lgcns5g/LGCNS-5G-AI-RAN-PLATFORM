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

#ifndef RAN_PUSCH_PIPELINE_HPP
#define RAN_PUSCH_PIPELINE_HPP

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <driver_types.h>

#include <gsl-lite/gsl-lite.hpp>

#include "pipeline/igraph_manager.hpp"
#include "pipeline/imodule.hpp"
#include "pipeline/imodule_factory.hpp"
#include "pipeline/ipipeline.hpp"
#include "pipeline/module_router.hpp"
#include "pipeline/pipeline_memory_manager.hpp"
#include "pipeline/types.hpp"
#include "pusch/pusch_defines.hpp"
#include "tensor/data_types.hpp"

namespace ran::pusch {

/**
 * PUSCH Pipeline
 *
 * Pipeline for PUSCH processing:
 * - External Inputs
 *  ─→ InnerRx module (TensorRT)
 *  ─→ LDPC derate match module (CUDA)
 *  ─→ LDPC decoder module
 *  ─→ CRC decoder module
 *  ─→ External Outputs
 *
 * Features:
 * - The full pipeline for PUSCH processing
 * - InnerRx module: TensorRT-based processing
 * - Channel decoding chain modules: CUDA-based LDPC decoding
 * - Stream and graph execution modes
 * - External input/output handling
 * - Memory management
 */
class PuschPipeline final : public framework::pipeline::IPipeline {
public:
    /**
     * Construct PUSCH pipeline using factory pattern
     *
     * Creates all modules via the provided factory and configures
     * the pipeline according to the PipelineSpec.
     *
     * @param[in] pipeline_id Unique identifier for pipeline instance
     * @param[in] module_factory Factory for creating modules (non-owning pointer)
     * @param[in] spec Pipeline specification with module configurations
     * @throws std::invalid_argument if spec doesn't have exactly 4 modules
     * @throws std::runtime_error if module creation fails
     */
    explicit PuschPipeline(
            std::string pipeline_id,
            gsl_lite::not_null<framework::pipeline::IModuleFactory *> module_factory,
            const framework::pipeline::PipelineSpec &spec);

    ~PuschPipeline() override = default;

    // Non-copyable, non-movable
    PuschPipeline(const PuschPipeline &) = delete;
    PuschPipeline &operator=(const PuschPipeline &) = delete;
    PuschPipeline(PuschPipeline &&) = delete;
    PuschPipeline &operator=(PuschPipeline &&) = delete;

    // ========================================================================
    // IPipeline Interface - Identification
    // ========================================================================

    /**
     * Get pipeline identifier
     *
     * @return Pipeline ID string
     */
    [[nodiscard]] std::string_view get_pipeline_id() const override { return pipeline_id_; }

    /**
     * Get number of external inputs
     *
     * @return Number of external input ports
     */
    [[nodiscard]] std::size_t get_num_external_inputs() const override {
        return NUM_EXTERNAL_INPUTS;
    }

    /**
     * Get number of external outputs
     *
     * @return Number of external output ports
     */
    [[nodiscard]] std::size_t get_num_external_outputs() const override {
        return NUM_EXTERNAL_OUTPUTS;
    }

    /**
     * Get pipeline execution mode
     *
     * @return Execution mode (Stream or Graph)
     */
    [[nodiscard]] framework::pipeline::ExecutionMode get_execution_mode() const {
        return execution_mode_;
    }

    // ========================================================================
    // IPipeline Interface - Setup Phase
    // ========================================================================

    /** Setup pipeline and modules
     *
     * Allocates memory and initializes all modules
     */
    void setup() override;

    /**
     * Warmup pipeline execution
     *
     * @param[in] stream CUDA stream for warmup
     */
    void warmup(cudaStream_t stream) override;

    // ========================================================================
    // IPipeline Interface - Per-Iteration Configuration
    // ========================================================================

    /**
     * Configure pipeline I/O for the current iteration.
     *
     * Routes external inputs to modules and calls configure_io() on each module.
     * Copies dynamic kernel descriptors to device and synchronizes stream.
     * Routes module outputs to external outputs.
     *
     * @param[in] params Dynamic parameters for this iteration
     * @param[in] external_inputs External input port information
     * @param[out] external_outputs External output port information to populate
     * @param[in] stream CUDA stream for any necessary operations
     * @throws std::invalid_argument if dynamic parameters are invalid
     */
    void configure_io(
            const framework::pipeline::DynamicParams &params,
            std::span<const framework::pipeline::PortInfo> external_inputs,
            std::span<framework::pipeline::PortInfo> external_outputs,
            cudaStream_t stream) override;

    // ========================================================================
    // IPipeline Interface - Execution
    // ========================================================================

    /**
     * Execute pipeline in stream mode
     *
     * @param[in] stream CUDA stream for execution
     */
    void execute_stream(cudaStream_t stream) override;

    /**
     * Execute pipeline in graph mode
     *
     * @param[in] stream CUDA stream for execution
     * @throws std::runtime_error if not in graph mode or if graph build fails
     * @throws std::runtime_error if graph execution fails
     */
    void execute_graph(cudaStream_t stream) override;

private:
    /**
     * Build the CUDA graph for graph-based execution
     * Called automatically by execute_graph() on first execution
     *
     * @param[in] stream CUDA stream for graph upload
     * @throws std::runtime_error if graph build fails
     */
    void build_graph(cudaStream_t stream);
    /**
     * Create modules from PipelineSpec using the factory
     *
     * @param[in] spec Pipeline specification with module configurations
     * @throws std::runtime_error if module creation or casting fails
     */
    void create_modules_from_spec(const framework::pipeline::PipelineSpec &spec);

    std::string pipeline_id_;                             //!< Pipeline instance ID
    framework::pipeline::IModuleFactory *module_factory_; //!< Module factory (non-owning)

    std::optional<PuschDynamicParams> pusch_input_; //!< PUSCH input parameters

    // Modules (order matters for execution)
    std::unique_ptr<framework::pipeline::IModule> inner_rx_module_; //!< Inner Rx module (TensorRT)
    std::unique_ptr<framework::pipeline::IModule>
            ldpc_derate_match_module_; //!< LDPC derate match module
    std::unique_ptr<framework::pipeline::IModule> ldpc_decoder_module_; //!< LDPC decoder module
    std::unique_ptr<framework::pipeline::IModule> crc_decoder_module_;  //!< CRC decoder module
    std::vector<framework::pipeline::IModule *> modules_;               //!< Module execution order

    // Infrastructure
    framework::pipeline::ModuleRouter router_;                               //!< Connection routing
    std::unique_ptr<framework::pipeline::PipelineMemoryManager> memory_mgr_; //!< Memory management
    std::unique_ptr<framework::pipeline::IGraphManager> graph_manager_;      //!< Graph management

    // State tracking
    bool graph_built_{false}; //!< Track one-time graph build completion
    framework::pipeline::ExecutionMode execution_mode_{
            framework::pipeline::ExecutionMode::Graph}; //!< Execution mode (Graph or Stream)
};

} // namespace ran::pusch

#endif // RAN_PUSCH_PIPELINE_HPP
