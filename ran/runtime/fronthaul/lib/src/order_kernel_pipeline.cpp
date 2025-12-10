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

#include <algorithm>
#include <any>
#include <cstddef>
#include <format>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <NamedType/named_type_impl.hpp>
#include <driver_types.h>
#include <gdrapi.h>
#include <quill/LogMacros.h>

#include <gsl-lite/gsl-lite.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include "fronthaul/fronthaul_log.hpp"
#include "fronthaul/order_kernel_module.hpp"
#include "fronthaul/order_kernel_pipeline.hpp"
#include "log/rt_log_macros.hpp"
#include "memory/gdrcopy_buffer.hpp"
#include "net/doca_types.hpp"
#include "pipeline/graph_manager.hpp"
#include "pipeline/igraph_manager.hpp"
#include "pipeline/igraph_node_provider.hpp"
#include "pipeline/imodule_factory.hpp"
#include "pipeline/istream_executor.hpp"
#include "pipeline/pipeline_memory_manager.hpp"
#include "pipeline/types.hpp"
#include "utils/error_macros.hpp"

namespace ran::fronthaul {

// Namespace aliases for cleaner code
namespace pipeline = framework::pipeline;
namespace memory = framework::memory;

// ============================================================================
// Construction
// ============================================================================

OrderKernelPipeline::OrderKernelPipeline(
        std::string pipeline_id,
        std::unique_ptr<framework::pipeline::IModuleFactory> module_factory,
        const framework::pipeline::PipelineSpec &spec,
        const framework::net::DocaRxQParams *doca_rxq_params)
        : pipeline_id_(std::move(pipeline_id)), module_factory_(std::move(module_factory)),
          doca_rxq_params_(doca_rxq_params), execution_mode_(spec.execution_mode) {

    RT_LOGC_INFO(
            FronthaulKernels::OrderPipeline,
            "OrderKernelPipeline: Constructing pipeline '{}' using factory pattern (mode={})",
            pipeline_id_,
            execution_mode_ == pipeline::ExecutionMode::Graph ? "Graph" : "Stream");

    // Validate spec has exactly 1 module
    if (spec.modules.size() != 1) {
        const std::string error_msg = std::format(
                "OrderKernelPipeline '{}': requires exactly 1 module, got {}",
                pipeline_id_,
                spec.modules.size());
        RT_LOGC_ERROR(FronthaulKernels::OrderPipeline, "{}", error_msg);
        throw std::invalid_argument(error_msg);
    }

    // Validate DOCA RX queue parameters
    if (doca_rxq_params_ == nullptr) {
        const std::string error_msg = std::format(
                "OrderKernelPipeline '{}': doca_rxq_params cannot be null", pipeline_id_);
        RT_LOGC_ERROR(FronthaulKernels::OrderPipeline, "{}", error_msg);
        throw std::invalid_argument(error_msg);
    }

    // Initialize GDRCopy handle (RAII-managed, automatically closed on destruction)
    gdr_handle_ = memory::make_unique_gdr_handle();
    RT_LOGC_INFO(
            FronthaulKernels::OrderPipeline,
            "OrderKernelPipeline '{}': GDRCopy handle opened successfully",
            pipeline_id_);

    // Create module using factory
    create_module_from_spec(spec);

    // Build module execution order (single module)
    modules_ = {order_kernel_module_.get()};

    RT_LOGC_DEBUG(
            FronthaulKernels::OrderPipeline,
            "OrderKernelPipeline '{}': Constructor complete",
            pipeline_id_);
}

void OrderKernelPipeline::create_module_from_spec(const pipeline::PipelineSpec &spec) {
    const auto &module_info = spec.modules[0].get();
    RT_LOGC_INFO(
            FronthaulKernels::OrderFactory,
            "OrderKernelPipeline: Creating module '{}' of type '{}'",
            module_info.instance_id,
            module_info.module_type);

    // Extract StaticParams from module spec and inject pipeline-managed handles
    OrderKernelModule::StaticParams module_params;

    try {
        // Try to extract existing params from spec (may contain execution_mode)
        module_params = std::any_cast<OrderKernelModule::StaticParams>(module_info.init_params);
    } catch (const std::bad_any_cast &) {
        // If no params provided in spec, use default constructed params
        RT_LOGC_DEBUG(
                FronthaulKernels::OrderFactory,
                "OrderKernelPipeline: No StaticParams in spec, using defaults");
    }

    // Inject pipeline-managed infrastructure handles
    module_params.gdr_handle = gdr_handle_.get(); // Get raw pointer from UniqueGdrHandle
    module_params.doca_rxq_params = doca_rxq_params_;

    RT_LOGC_DEBUG(
            FronthaulKernels::OrderFactory,
            "OrderKernelPipeline: Injecting gdr_handle={}, doca_rxq_params={}",
            static_cast<const void *>(gdr_handle_.get()),
            static_cast<const void *>(doca_rxq_params_));

    // Create module with modified params
    const std::any modified_params = module_params;
    auto module = module_factory_->create_module(
            module_info.module_type, module_info.instance_id, modified_params);

    if (!module) {
        const std::string error_msg = std::format(
                "OrderKernelPipeline '{}': Failed to create module '{}'",
                pipeline_id_,
                module_info.instance_id);
        RT_LOGC_ERROR(FronthaulKernels::OrderFactory, "{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    // Safe static_cast - factory guarantees OrderKernelModule type
    order_kernel_module_ = std::unique_ptr<OrderKernelModule>(
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
            static_cast<OrderKernelModule *>(module.release()));

    RT_LOGC_INFO(
            FronthaulKernels::OrderFactory,
            "OrderKernelPipeline '{}': Successfully created module '{}' via factory",
            pipeline_id_,
            module_info.instance_id);
}

// ============================================================================
// Setup (one-time initialization)
// ============================================================================

void OrderKernelPipeline::setup() {
    RT_LOGC_INFO(
            FronthaulKernels::OrderPipeline,
            "OrderKernelPipeline '{}': setup() called (mode={})",
            pipeline_id_,
            execution_mode_ == pipeline::ExecutionMode::Graph ? "Graph" : "Stream");

    // Configure zero-copy for external DOCA input
    // DOCA objects are external (stable pointers), so use zero-copy
    order_kernel_module_->set_connection_copy_mode(
            "doca_objects", pipeline::ConnectionCopyMode::ZeroCopy);
    RT_LOGC_INFO(
            FronthaulKernels::OrderDoca,
            "OrderKernelPipeline '{}': External DOCA input → module zero-copy enabled",
            pipeline_id_);

    // Allocate memory
    memory_mgr_ = pipeline::PipelineMemoryManager::create_for_modules(modules_);
    RT_LOGC_DEBUG(
            FronthaulKernels::OrderMemory,
            "OrderKernelPipeline '{}': Memory manager created",
            pipeline_id_);

    // Allocate memory slices for the module
    memory_mgr_->allocate_all_module_slices(modules_);

    // Call setup_memory() on the module with its memory slice
    const auto slice = memory_mgr_->get_module_slice(order_kernel_module_->get_instance_id());
    order_kernel_module_->setup_memory(slice);

    RT_LOGC_INFO(
            FronthaulKernels::OrderPipeline,
            "OrderKernelPipeline '{}': setup() complete",
            pipeline_id_);
}

void OrderKernelPipeline::warmup(cudaStream_t stream) {
    RT_LOGC_INFO(
            FronthaulKernels::OrderPipeline,
            "OrderKernelPipeline '{}': warmup(stream={}) called",
            pipeline_id_,
            static_cast<void *>(stream));

    // Copy all static kernel descriptors to device
    memory_mgr_->copy_all_static_descriptors_to_device(stream);

    // Call warmup() on the module
    RT_LOGC_DEBUG(
            FronthaulKernels::OrderPipeline,
            "OrderKernelPipeline '{}': Calling warmup(stream) on module '{}'",
            pipeline_id_,
            order_kernel_module_->get_instance_id());
    order_kernel_module_->warmup(stream);

    RT_LOGC_INFO(
            FronthaulKernels::OrderPipeline,
            "OrderKernelPipeline '{}': warmup() complete",
            pipeline_id_);
}

std::vector<pipeline::PortInfo> OrderKernelPipeline::get_outputs() const {
    if (!order_kernel_module_) {
        const std::string error_msg =
                std::format("OrderKernelPipeline '{}': module not initialized", pipeline_id_);
        RT_LOGC_ERROR(FronthaulKernels::OrderPipeline, "{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    auto outputs = order_kernel_module_->get_outputs();
    RT_LOGC_DEBUG(
            FronthaulKernels::OrderPipeline,
            "OrderKernelPipeline '{}': get_outputs() returning {} output ports",
            pipeline_id_,
            outputs.size());

    return outputs;
}

// ============================================================================
// Per-Iteration Configuration
// ============================================================================

void OrderKernelPipeline::configure_io(
        const pipeline::DynamicParams &params,
        std::span<const pipeline::PortInfo> external_inputs,
        std::span<pipeline::PortInfo> external_outputs,
        cudaStream_t stream) {

    RT_LOGC_DEBUG(
            FronthaulKernels::OrderPipeline,
            "OrderKernelPipeline '{}': configure_io(stream={})",
            pipeline_id_,
            static_cast<void *>(stream));

    // Set external inputs to module
    order_kernel_module_->set_inputs(external_inputs);
    order_kernel_module_->configure_io(params, stream);

    // Copy all dynamic kernel descriptors to device
    memory_mgr_->copy_all_dynamic_descriptors_to_device(stream);

    // Synchronize to ensure descriptors are copied
    FRAMEWORK_CUDA_RUNTIME_CHECK_THROW(cudaStreamSynchronize(stream));

    // Route module outputs to external outputs
    const auto module_outputs = order_kernel_module_->get_outputs();
    for (std::size_t i = 0; i < external_outputs.size() && i < module_outputs.size(); ++i) {
        external_outputs[i] = module_outputs[i];
    }

    RT_LOGC_DEBUG(
            FronthaulKernels::OrderPipeline,
            "OrderKernelPipeline '{}': configure_io complete",
            pipeline_id_);
}

// ============================================================================
// Execution - Stream Mode
// ============================================================================

void OrderKernelPipeline::execute_stream(cudaStream_t stream) {
    // Validate execution mode
    if (execution_mode_ != pipeline::ExecutionMode::Stream) {
        const std::string error_msg = std::format(
                "OrderKernelPipeline '{}': execute_stream() called but execution_mode is Graph. "
                "Use execute_graph() for Graph mode.",
                pipeline_id_);
        RT_LOGC_ERROR(FronthaulKernels::OrderPipeline, "{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    RT_LOGC_DEBUG(
            FronthaulKernels::OrderPipeline,
            "OrderKernelPipeline '{}': execute_stream() on stream {}",
            pipeline_id_,
            static_cast<void *>(stream));

    // Execute module using IStreamExecutor
    auto *stream_executor = order_kernel_module_->as_stream_executor();
    if (stream_executor != nullptr) {
        stream_executor->execute(stream);
    }

    RT_LOGC_DEBUG(
            FronthaulKernels::OrderPipeline,
            "OrderKernelPipeline '{}': execute_stream complete",
            pipeline_id_);
}

// ============================================================================
// Execution - Graph Mode
// ============================================================================

void OrderKernelPipeline::build_graph() {
    if (graph_built_) {
        RT_LOGC_DEBUG(
                FronthaulKernels::OrderPipeline,
                "OrderKernelPipeline '{}': Graph already built, skipping build_graph()",
                pipeline_id_);
        return;
    }

    RT_LOGC_INFO(
            FronthaulKernels::OrderPipeline,
            "OrderKernelPipeline '{}': build_graph() - building graph",
            pipeline_id_);

    // Create graph manager
    graph_manager_ = std::make_unique<pipeline::GraphManager>();

    // Add module to graph
    std::vector<CUgraphNode> prev_nodes;

    auto *graph_provider = order_kernel_module_->as_graph_node_provider();
    if (graph_provider == nullptr) {
        const std::string error_msg = std::format(
                "OrderKernelPipeline '{}': Module '{}' does not implement "
                "IGraphNodeProvider",
                pipeline_id_,
                std::string(order_kernel_module_->get_instance_id()));
        RT_LOGC_ERROR(FronthaulKernels::OrderPipeline, "{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    // Add module's node(s) to graph
    const auto nodes = graph_manager_->add_kernel_node(
            gsl_lite::not_null<pipeline::IGraphNodeProvider *>(graph_provider), prev_nodes);

    std::ignore = nodes; // Suppress unused warning

    // Instantiate and upload graph
    graph_manager_->instantiate_graph();
    graph_manager_->upload_graph(nullptr);

    graph_built_ = true;
    RT_LOGC_INFO(
            FronthaulKernels::OrderPipeline,
            "OrderKernelPipeline '{}': build_graph() complete",
            pipeline_id_);
}

void OrderKernelPipeline::execute_graph(cudaStream_t stream) {
    // Build graph on first execution (idempotent)
    build_graph();

    // Validate execution mode
    if (execution_mode_ != pipeline::ExecutionMode::Graph) {
        const std::string error_msg = std::format(
                "OrderKernelPipeline '{}': execute_graph() called but execution_mode is Stream. "
                "Use execute_stream() for Stream mode.",
                pipeline_id_);
        RT_LOGC_ERROR(FronthaulKernels::OrderPipeline, "{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    RT_LOGC_DEBUG(
            FronthaulKernels::OrderPipeline,
            "OrderKernelPipeline '{}': execute_graph() on stream {}",
            pipeline_id_,
            static_cast<void *>(stream));

    if (!graph_manager_) {
        const std::string error_msg = std::format(
                "OrderKernelPipeline '{}': build_graph() must be called before execute_graph()",
                pipeline_id_);
        RT_LOGC_ERROR(FronthaulKernels::OrderPipeline, "{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    // Update graph node parameters before execution
    auto *const exec = graph_manager_->get_exec();
    const pipeline::DynamicParams dummy_params{};
    auto *graph_node_provider = order_kernel_module_->as_graph_node_provider();
    if (graph_node_provider != nullptr) {
        graph_node_provider->update_graph_node_params(exec, dummy_params);
    }

    graph_manager_->launch_graph(stream);

    RT_LOGC_DEBUG(
            FronthaulKernels::OrderPipeline,
            "OrderKernelPipeline '{}': execute_graph complete",
            pipeline_id_);
}

// ============================================================================
// Kernel Results Access
// ============================================================================

OrderKernelModule::OrderKernelResults OrderKernelPipeline::read_kernel_results() const {
    // Direct access - order_kernel_module_ is already the concrete type
    return order_kernel_module_->read_kernel_results();
}

} // namespace ran::fronthaul
