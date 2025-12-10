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

// OrderKernelPipeline implementation for ORAN UL Receiver

#ifndef RAN_FRONTHAUL_ORDER_KERNEL_PIPELINE_HPP
#define RAN_FRONTHAUL_ORDER_KERNEL_PIPELINE_HPP

// clang-format off
#include <driver_types.h>
#include <gsl-lite/gsl-lite.hpp>
#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "fronthaul/order_kernel_module.hpp"
#include "memory/gdrcopy_buffer.hpp"  // for UniqueGdrHandle, gdr_t
#include "pipeline/igraph_manager.hpp"
#include "pipeline/imodule.hpp"
#include "pipeline/imodule_factory.hpp"
#include "pipeline/ipipeline.hpp"
#include "pipeline/pipeline_memory_manager.hpp"
#include "pipeline/types.hpp"

#include "net/doca_types.hpp"  // for DocaRxQParams

// clang-format on

namespace ran::fronthaul {

/**
 * OrderKernelPipeline - ORAN UL Receiver Pipeline
 *
 * Single-module pipeline for ORAN packet reception and processing:
 * - External Input (DOCA objects) ─→ OrderKernelModule ─→ External Output (PUSCH data)
 *
 * Features:
 * - Single OrderKernelModule wrapping order_kernel_doca_single_subSlot_pingpong
 * - Stream and graph execution modes
 * - GDRCopy memory management for NIC↔GPU communication
 * - External DOCA input handling
 */
class OrderKernelPipeline final : public framework::pipeline::IPipeline {
public:
    /**
     * Construct OrderKernelPipeline using factory pattern
     *
     * Creates the OrderKernelModule via the provided factory and configures
     * the pipeline according to the PipelineSpec.
     *
     * @param[in] pipeline_id Unique identifier for pipeline instance
     * @param[in] module_factory Factory for creating modules (takes ownership)
     * @param[in] spec Pipeline specification with module configuration
     * @param[in] doca_rxq_params DOCA RX queue parameters (non-owning pointer, must outlive
     * pipeline)
     * @throws std::invalid_argument if spec doesn't have exactly 1 module or doca_rxq_params is
     * null
     * @throws std::runtime_error if module creation fails or GDRCopy init fails
     */
    OrderKernelPipeline(
            std::string pipeline_id,
            std::unique_ptr<framework::pipeline::IModuleFactory> module_factory,
            const framework::pipeline::PipelineSpec &spec,
            const framework::net::DocaRxQParams *doca_rxq_params);

    ~OrderKernelPipeline() override = default;

    // Non-copyable, non-movable
    OrderKernelPipeline(const OrderKernelPipeline &) = delete;
    OrderKernelPipeline &operator=(const OrderKernelPipeline &) = delete;
    OrderKernelPipeline(OrderKernelPipeline &&) = delete;
    OrderKernelPipeline &operator=(OrderKernelPipeline &&) = delete;

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
     * @return Number of external inputs (1 for DOCA objects)
     */
    [[nodiscard]] std::size_t get_num_external_inputs() const override {
        return 1; // One input: DOCA objects
    }

    /**
     * Get number of external outputs
     *
     * @return Number of external outputs (1 for PUSCH buffer)
     */
    [[nodiscard]] std::size_t get_num_external_outputs() const override {
        return 1; // One output: PUSCH buffer
    }

    /**
     * Get pipeline output port information
     *
     * Provides access to Order Kernel's output buffer addresses. These addresses
     * are stable after warmup() and can be used for zero-copy data passing to
     * downstream pipelines (e.g., PUSCH).
     *
     * @return Vector containing one PortInfo with PUSCH buffer information
     * @throws std::runtime_error if module not initialized
     */
    [[nodiscard]] std::vector<framework::pipeline::PortInfo> get_outputs() const override;

    // ========================================================================
    // IPipeline Interface - Setup
    // ========================================================================

    /**
     * Perform pipeline setup
     *
     * Allocates memory and initializes all modules
     */
    void setup() override;

    /**
     * Perform warmup operations
     *
     * @param[in] stream CUDA stream for warmup operations
     */
    void warmup(cudaStream_t stream) override;

    // ========================================================================
    // IPipeline Interface - Per-Iteration Configuration
    // ========================================================================

    /**
     * Configure pipeline I/O for the current iteration.
     *
     * @param[in] params Dynamic parameters for this iteration
     * @param[in] external_inputs External input port information (DOCA objects)
     * @param[out] external_outputs External output port information to populate
     * @param[in] stream CUDA stream for any necessary operations
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
     * @param[in] stream CUDA stream for graph execution
     */
    void execute_graph(cudaStream_t stream) override;

    // ========================================================================
    // OrderKernelPipeline Specific Interface - Kernel Results Access
    // ========================================================================

    /**
     * Read kernel execution results from the OrderKernelModule
     *
     * This method provides access to kernel results including exit condition
     * and PRB counts. Should be called after kernel execution completes.
     *
     * @return OrderKernelModule::OrderKernelResults structure with current values
     * @throws std::runtime_error if module is not available or not an OrderKernelModule
     */
    [[nodiscard]] OrderKernelModule::OrderKernelResults read_kernel_results() const;

private:
    /**
     * Build the CUDA graph for graph-based execution
     * Called automatically by execute_graph() on first execution
     */
    void build_graph();

    /**
     * Create module from PipelineSpec using the factory
     *
     * @param[in] spec Pipeline specification with module configuration
     * @throws std::runtime_error if module creation or casting fails
     */
    void create_module_from_spec(const framework::pipeline::PipelineSpec &spec);

    std::string pipeline_id_; //!< Pipeline instance ID
    std::unique_ptr<framework::pipeline::IModuleFactory>
            module_factory_; //!< Module factory (owned)

    // Infrastructure handles
    framework::memory::UniqueGdrHandle
            gdr_handle_; //!< GDRCopy handle for NIC↔GPU memory (RAII-managed)
    const framework::net::DocaRxQParams *doca_rxq_params_{
            nullptr}; //!< DOCA RX queue parameters (non-owning)

    // Single module
    std::unique_ptr<OrderKernelModule>
            order_kernel_module_; //!< OrderKernelModule instance (concrete type)
    std::vector<framework::pipeline::IModule *> modules_; //!< Module execution order (size=1)

    // Infrastructure
    std::unique_ptr<framework::pipeline::PipelineMemoryManager> memory_mgr_; //!< Memory management
    std::unique_ptr<framework::pipeline::IGraphManager> graph_manager_;      //!< Graph management

    // State tracking
    bool graph_built_{false}; //!< Track one-time graph build completion
    framework::pipeline::ExecutionMode execution_mode_{
            framework::pipeline::ExecutionMode::Graph}; //!< Execution mode (Graph or Stream)
};

} // namespace ran::fronthaul

#endif // RAN_FRONTHAUL_ORDER_KERNEL_PIPELINE_HPP
