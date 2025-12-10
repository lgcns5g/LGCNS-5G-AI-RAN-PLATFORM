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

#include <NamedType/named_type_impl.hpp>
#include <quill/LogMacros.h>

#include <wise_enum.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "log/rt_log_macros.hpp"
#include "pipeline/graph_manager.hpp"
#include "pipeline/igraph_node_provider.hpp"
#include "pipeline/istream_executor.hpp"
#include "sample_pipeline.hpp"
#include "utils/error_macros.hpp"

namespace framework::pipelines::samples {

// Namespace alias for compatibility with framework reorganization
namespace pipeline = ::framework::pipeline;

// ============================================================================
// Construction
// ============================================================================

SamplePipeline::SamplePipeline(
        std::string pipeline_id,
        gsl_lite::not_null<pipeline::IModuleFactory *> module_factory,
        const pipeline::PipelineSpec &spec)
        : pipeline_id_(std::move(pipeline_id)), module_factory_(module_factory),
          execution_mode_(spec.execution_mode) {

    RT_LOG_INFO(
            "SamplePipeline: Constructing pipeline '{}' using factory "
            "pattern (mode={})",
            pipeline_id_,
            ::wise_enum::to_string(execution_mode_));

    // Validate spec has exactly 2 modules
    if (spec.modules.size() != 2) {
        const std::string error_msg = std::format(
                "SamplePipeline '{}': requires exactly 2 modules, got {}",
                pipeline_id_,
                spec.modules.size());
        RT_LOG_ERROR("{}", error_msg);
        throw std::invalid_argument(error_msg);
    }

    // Create modules using factory
    create_modules_from_spec(spec);

    // Build module execution order
    modules_ = {module_a_.get(), module_b_.get()};

    // Setup routing from spec connections
    for (const auto &connection : spec.connections) {
        router_.add_connection(connection);
    }
    router_.validate();

    RT_LOG_DEBUG("SamplePipeline '{}': Constructor complete", pipeline_id_);
}

void SamplePipeline::create_modules_from_spec(const pipeline::PipelineSpec &spec) {
    // Create module A
    const auto &module_a_info = spec.modules[0].get();
    RT_LOG_INFO(
            "SamplePipeline: Creating module '{}' of type '{}'",
            module_a_info.instance_id,
            module_a_info.module_type);

    auto module_a = module_factory_->create_module(
            module_a_info.module_type, module_a_info.instance_id, module_a_info.init_params);

    if (!module_a) {
        const std::string error_msg = std::format(
                "SamplePipeline '{}': Failed to create module '{}'",
                pipeline_id_,
                module_a_info.instance_id);
        RT_LOG_ERROR("{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    module_a_ = std::move(module_a);

    // Create module B
    const auto &module_b_info = spec.modules[1].get();
    RT_LOG_INFO(
            "SamplePipeline: Creating module '{}' of type '{}'",
            module_b_info.instance_id,
            module_b_info.module_type);

    auto module_b = module_factory_->create_module(
            module_b_info.module_type, module_b_info.instance_id, module_b_info.init_params);

    if (!module_b) {
        const std::string error_msg = std::format(
                "SamplePipeline '{}': Failed to create module '{}'",
                pipeline_id_,
                module_b_info.instance_id);
        RT_LOG_ERROR("{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    module_b_ = std::move(module_b);

    RT_LOG_INFO(
            "SamplePipeline '{}': Successfully created modules '{}' and "
            "'{}' via factory",
            pipeline_id_,
            module_a_info.instance_id,
            module_b_info.instance_id);
}

// ============================================================================
// Setup Phase (one-time initialization)
// ============================================================================

void SamplePipeline::setup() {
    RT_LOG_INFO(
            "SamplePipeline '{}': setup() called (mode={})",
            pipeline_id_,
            ::wise_enum::to_string(execution_mode_));

    // Configure zero-copy based on execution mode and memory characteristics
    // Topology: External → ModuleA (TRT) → ModuleB (ReLU) → External

    // 1. External → ModuleA connections
    // External inputs don't have characteristics API, so we use execution mode
    if (execution_mode_ == pipeline::ExecutionMode::Stream) {
        // Stream mode: ModuleA doesn't require stable addresses (uses
        // set_tensor_address())
        module_a_->set_connection_copy_mode("input0", pipeline::ConnectionCopyMode::ZeroCopy);
        module_a_->set_connection_copy_mode("input1", pipeline::ConnectionCopyMode::ZeroCopy);
        RT_LOG_INFO(
                "SamplePipeline '{}': Stream mode - external→A zero-copy enabled", pipeline_id_);
    } else {
        // Graph mode: ModuleA requires fixed addresses for CUDA graph capture
        // External inputs are dynamic in this example (can change per iteration),
        // so must copy. In a different example, the external inputs could be fixed,
        // in which case we would enable zero-copy.
        //
        // ```cpp
        // module_a_->set_connection_copy_mode("input0",
        //                                     pipeline::ConnectionCopyMode::ZeroCopy);
        // module_a_->set_connection_copy_mode("input1",
        //                                     pipeline::ConnectionCopyMode::ZeroCopy);
        // ```
        //
        // ```cpp
        // RT_LOG_INFO("SamplePipeline '{}': Graph mode - external→A zero-copy
        //              enabled", pipeline_id_);
        // ```
        module_a_->set_connection_copy_mode("input0", pipeline::ConnectionCopyMode::Copy);
        module_a_->set_connection_copy_mode("input1", pipeline::ConnectionCopyMode::Copy);
        RT_LOG_INFO("SamplePipeline '{}': Graph mode - external→A requires copy", pipeline_id_);
    }

    // 2. ModuleA → ModuleB connection (module-to-module negotiation)
    auto module_a_output_chars = module_a_->get_output_memory_characteristics("output");
    auto module_b_input_chars = module_b_->get_input_memory_characteristics("input");

    if (pipeline::can_zero_copy(module_a_output_chars, module_b_input_chars)) {
        module_b_->set_connection_copy_mode("input", pipeline::ConnectionCopyMode::ZeroCopy);
        RT_LOG_INFO(
                "SamplePipeline '{}': A→B zero-copy enabled (A provides_fixed={}, "
                "B requires_fixed={})",
                pipeline_id_,
                module_a_output_chars.provides_fixed_address_for_zero_copy,
                module_b_input_chars.requires_fixed_address_for_zero_copy);
    } else {
        module_b_->set_connection_copy_mode("input", pipeline::ConnectionCopyMode::Copy);
        RT_LOG_INFO(
                "SamplePipeline '{}': A→B requires copy (A provides_fixed={}, "
                "B requires_fixed={})",
                pipeline_id_,
                module_a_output_chars.provides_fixed_address_for_zero_copy,
                module_b_input_chars.requires_fixed_address_for_zero_copy);
    }

    // NOW allocate memory (get_requirements() will see the configured flags)
    memory_mgr_ = pipeline::PipelineMemoryManager::create_for_modules(modules_);

    RT_LOG_DEBUG("SamplePipeline '{}': Memory manager created", pipeline_id_);

    // Allocate memory slices for all modules
    memory_mgr_->allocate_all_module_slices(modules_);

    // Call setup_memory() on each module with their memory slice
    for (auto *module : modules_) {
        const auto slice = memory_mgr_->get_module_slice(module->get_instance_id());
        module->setup_memory(slice);
    }

    RT_LOG_INFO("SamplePipeline '{}': setup() complete", pipeline_id_);
}

void SamplePipeline::warmup(cudaStream_t stream) {
    RT_LOG_INFO(
            "SamplePipeline '{}': warmup(stream={}) called - performing "
            "one-time module initialization",
            pipeline_id_,
            static_cast<void *>(stream));

    // Copy all static kernel descriptors to device in one bulk operation
    // This should be done before warmup so modules can use their static params
    memory_mgr_->copy_all_static_descriptors_to_device(stream);

    // Call warmup() on all modules with the provided stream
    for (auto *module : modules_) {
        RT_LOG_DEBUG(
                "SamplePipeline '{}': Calling warmup(stream) on module '{}'",
                pipeline_id_,
                module->get_instance_id());
        module->warmup(stream);
    }

    RT_LOG_INFO(
            "SamplePipeline '{}': warmup() complete - all modules ready "
            "for execution",
            pipeline_id_);
}

// ============================================================================
// Per-Iteration Configuration
// ============================================================================

void SamplePipeline::configure_io(
        const pipeline::DynamicParams &params,
        std::span<const pipeline::PortInfo> external_inputs,
        std::span<pipeline::PortInfo> external_outputs,
        cudaStream_t stream) {

    RT_LOG_DEBUG(
            "SamplePipeline '{}': configure_io(stream={})",
            pipeline_id_,
            static_cast<void *>(stream));

    // Step 1: Route external inputs to ModuleA and call configure_io
    module_a_->set_inputs(external_inputs);
    module_a_->configure_io(
            params, stream); // Called before get_outputs() to support dynamic topology

    // Step 2: Get ModuleA outputs (AFTER configure_io, supports dynamic
    // determination)
    auto module_a_outputs = module_a_->get_outputs();
    if (!module_a_outputs.empty()) {
        module_a_outputs[0].name = "input"; // ModuleB expects port named "input"
    }

    // Step 3: Route ModuleA outputs to ModuleB and call configure_io
    module_b_->set_inputs(module_a_outputs);
    module_b_->configure_io(params, stream); // ModuleB can now use inputs from ModuleA

    // Copy all dynamic kernel descriptors to device in one bulk operation
    // This should be done after all modules have updated their dynamic params
    memory_mgr_->copy_all_dynamic_descriptors_to_device(stream);

    // Synchronize to ensure descriptors are copied before graph capture
    // Graph capture needs descriptor data to be present in device memory
    FRAMEWORK_CUDA_RUNTIME_CHECK_THROW(cudaStreamSynchronize(stream));

    // Route module outputs to external outputs
    const auto module_b_outputs = module_b_->get_outputs();
    for (std::size_t i = 0; i < external_outputs.size() && i < module_b_outputs.size(); ++i) {
        external_outputs[i] = module_b_outputs[i];
    }

    RT_LOG_DEBUG("SamplePipeline '{}': configure_io complete", pipeline_id_);
}

// ============================================================================
// Execution - Stream Mode
// ============================================================================

void SamplePipeline::execute_stream(cudaStream_t stream) {
    // Validate execution mode
    if (execution_mode_ != pipeline::ExecutionMode::Stream) {
        const std::string error_msg = std::format(
                "SamplePipeline '{}': execute_stream() called but "
                "execution_mode is Graph. "
                "Use execute_graph() for Graph mode.",
                pipeline_id_);
        RT_LOG_ERROR("{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    RT_LOG_DEBUG(
            "SamplePipeline '{}': execute_stream() on stream {}",
            pipeline_id_,
            static_cast<void *>(stream));

    // Execute modules in order using IStreamExecutor
    for (auto *module : modules_) {
        auto *stream_executor = module->as_stream_executor();
        if (stream_executor != nullptr) {
            stream_executor->execute(stream); // Uses parameters from configure_io()
        }
    }

    RT_LOG_DEBUG("SamplePipeline '{}': execute_stream complete", pipeline_id_);
}

// ============================================================================
// Execution - Graph Mode
// ============================================================================

void SamplePipeline::build_graph() {
    if (graph_built_) {
        RT_LOG_DEBUG(
                "SamplePipeline '{}': Graph already built, skipping build_graph()", pipeline_id_);
        return;
    }

    RT_LOG_INFO("SamplePipeline '{}': build_graph() - building graph for first time", pipeline_id_);

    // Create graph manager (constructor implicitly creates empty CUDA graph)
    graph_manager_ = std::make_unique<pipeline::GraphManager>();

    // Add modules to graph in execution order
    std::vector<CUgraphNode> prev_nodes;

    for (auto *module : modules_) {
        auto *graph_provider = module->as_graph_node_provider();
        if (graph_provider == nullptr) {
            const std::string error_msg = std::format(
                    "SamplePipeline '{}': Module '{}' does not implement "
                    "IGraphNodeProvider",
                    pipeline_id_,
                    std::string(module->get_instance_id()));
            RT_LOG_ERROR("{}", error_msg);
            throw std::runtime_error(error_msg);
        }

        // Add module's node(s) to graph with dependencies on previous nodes
        const auto nodes = graph_manager_->add_kernel_node(
                gsl_lite::not_null<pipeline::IGraphNodeProvider *>(graph_provider), prev_nodes);

        // Current node(s) become dependencies for next module
        prev_nodes.assign(nodes.begin(), nodes.end());
    }

    // Instantiate and upload graph
    graph_manager_->instantiate_graph();
    graph_manager_->upload_graph(nullptr);

    graph_built_ = true;
    RT_LOG_INFO(
            "SamplePipeline '{}': build_graph() complete - graph ready "
            "for execution",
            pipeline_id_);
}

void SamplePipeline::execute_graph(cudaStream_t stream) {
    // Build graph on first execution (idempotent)
    build_graph();

    // Validate execution mode
    if (execution_mode_ != pipeline::ExecutionMode::Graph) {
        const std::string error_msg = std::format(
                "SamplePipeline '{}': execute_graph() called but "
                "execution_mode is Stream. "
                "Use execute_stream() for Stream mode.",
                pipeline_id_);
        RT_LOG_ERROR("{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    RT_LOG_DEBUG(
            "SamplePipeline '{}': execute_graph() on stream {}",
            pipeline_id_,
            static_cast<void *>(stream));

    if (!graph_manager_) {
        const std::string error_msg = std::format(
                "SamplePipeline '{}': build_graph() must be called before "
                "execute_graph()",
                pipeline_id_);
        RT_LOG_ERROR("{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    // Update graph node parameters before execution
    auto *const exec = graph_manager_->get_exec();
    const pipeline::DynamicParams dummy_params{}; // We don't use params for parameter updates
    for (auto *module : modules_) {
        auto *graph_node_provider = module->as_graph_node_provider();
        if (graph_node_provider != nullptr) {
            graph_node_provider->update_graph_node_params(exec, dummy_params);
        }
    }

    graph_manager_->launch_graph(stream);

    RT_LOG_DEBUG("SamplePipeline '{}': execute_graph complete", pipeline_id_);
}

} // namespace framework::pipelines::samples
