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

#ifndef FRAMEWORK_PIPELINES_SAMPLE_PIPELINE_HPP
#define FRAMEWORK_PIPELINES_SAMPLE_PIPELINE_HPP

#include <cstddef>
#include <memory>
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

namespace framework::pipelines::samples {

// Namespace alias for compatibility with framework reorganization
namespace pipeline = ::framework::pipeline;

/**
 * Sample Pipeline
 *
 * Demonstrates basic pipeline pattern with two modules:
 * - External Input 0 ──┐
 *                      ├─→ ModuleA (Add) ─→ ModuleB (ReLU) ─→ External Output
 * - External Input 1 ──┘
 *
 * Features:
 * - Two-module pipeline with routing
 * - Stream and graph execution modes
 * - External input/output handling
 * - Memory management demonstration
 */
class SamplePipeline final : public pipeline::IPipeline {
public:
    /**
     * Construct sample pipeline using factory pattern
     *
     * Creates modules via the provided factory and configures the pipeline
     * according to the PipelineSpec.
     *
     * @param[in] pipeline_id Unique identifier for pipeline instance
     * @param[in] module_factory Factory for creating modules (non-owning pointer)
     * @param[in] spec Pipeline specification with module configurations
     * @throws std::invalid_argument if spec doesn't have exactly 2 modules
     * @throws std::runtime_error if module creation fails
     */
    SamplePipeline(
            std::string pipeline_id,
            gsl_lite::not_null<pipeline::IModuleFactory *> module_factory,
            const pipeline::PipelineSpec &spec);

    ~SamplePipeline() override = default;

    // Non-copyable, non-movable
    SamplePipeline(const SamplePipeline &) = delete;
    SamplePipeline &operator=(const SamplePipeline &) = delete;
    SamplePipeline(SamplePipeline &&) = delete;
    SamplePipeline &operator=(SamplePipeline &&) = delete;

    // ========================================================================
    // IPipeline Interface - Identification
    // ========================================================================

    [[nodiscard]] std::string_view get_pipeline_id() const override { return pipeline_id_; }

    [[nodiscard]] std::size_t get_num_external_inputs() const override {
        return 2; // Two inputs for ModuleA
    }

    [[nodiscard]] std::size_t get_num_external_outputs() const override {
        return 1; // One output from ModuleB
    }

    // ========================================================================
    // IPipeline Interface - Setup Phase
    // ========================================================================

    void setup() override;

    void warmup(cudaStream_t stream) override;

    // ========================================================================
    // IPipeline Interface - Per-Iteration Configuration
    // ========================================================================

    /**
     * Configure pipeline I/O for the current iteration.
     *
     * @param[in] params Dynamic parameters for this iteration
     * @param[in] external_inputs External input port information
     * @param[out] external_outputs External output port information to populate
     * @param[in] stream CUDA stream for any necessary operations
     */
    void configure_io(
            const pipeline::DynamicParams &params,
            std::span<const pipeline::PortInfo> external_inputs,
            std::span<pipeline::PortInfo> external_outputs,
            cudaStream_t stream) override;

    // ========================================================================
    // IPipeline Interface - Execution
    // ========================================================================

    void execute_stream(cudaStream_t stream) override;

    void execute_graph(cudaStream_t stream) override;

private:
    /**
     * Build the CUDA graph for graph-based execution
     * Called automatically by execute_graph() on first execution
     */
    void build_graph();
    /**
     * Create modules from PipelineSpec using the factory
     *
     * @param[in] spec Pipeline specification with module configurations
     * @throws std::runtime_error if module creation or casting fails
     */
    void create_modules_from_spec(const pipeline::PipelineSpec &spec);

    std::string pipeline_id_;                  //!< Pipeline instance ID
    pipeline::IModuleFactory *module_factory_; //!< Module factory (non-owning)

    // Modules (order matters for execution)
    std::unique_ptr<pipeline::IModule> module_a_; //!< TensorRT addition module
    std::unique_ptr<pipeline::IModule> module_b_; //!< CUDA ReLU module
    std::vector<pipeline::IModule *> modules_;    //!< Module execution order

    // Infrastructure
    pipeline::ModuleRouter router_;                               //!< Connection routing
    std::unique_ptr<pipeline::PipelineMemoryManager> memory_mgr_; //!< Memory management
    std::unique_ptr<pipeline::IGraphManager> graph_manager_;      //!< Graph management

    // State tracking
    bool graph_built_{false}; //!< Track one-time graph build completion
    pipeline::ExecutionMode execution_mode_{
            pipeline::ExecutionMode::Graph}; //!< Execution mode (Graph or Stream)
};

} // namespace framework::pipelines::samples

#endif // FRAMEWORK_PIPELINES_SAMPLE_PIPELINE_HPP
