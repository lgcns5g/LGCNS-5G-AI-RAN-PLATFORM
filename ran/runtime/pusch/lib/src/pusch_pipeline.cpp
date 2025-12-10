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
#include <any> // for any_cast, any
#include <cstddef>
#include <format>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map> // for unordered_map
#include <utility>
#include <vector>

#include <NamedType/named_type_impl.hpp>
#include <driver_types.h>
#include <quill/LogMacros.h>

#include <gsl-lite/gsl-lite.hpp>
#include <wise_enum.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "ldpc/outer_rx_params.hpp"
#include "log/rt_log_macros.hpp"
#include "pipeline/graph_manager.hpp"
#include "pipeline/igraph_manager.hpp"
#include "pipeline/igraph_node_provider.hpp"
#include "pipeline/imodule.hpp"
#include "pipeline/imodule_factory.hpp"
#include "pipeline/istream_executor.hpp"
#include "pipeline/pipeline_memory_manager.hpp"
#include "pipeline/types.hpp"
#include "pusch/pusch_defines.hpp"
#include "pusch/pusch_log.hpp"
#include "pusch/pusch_pipeline.hpp"
#include "utils/error_macros.hpp"

namespace ran::pusch {

namespace pipeline = framework::pipeline;

// ============================================================================
// Construction
// ============================================================================

PuschPipeline::PuschPipeline(
        std::string pipeline_id,
        gsl_lite::not_null<framework::pipeline::IModuleFactory *> module_factory,
        const framework::pipeline::PipelineSpec &spec)
        : pipeline_id_(std::move(pipeline_id)), module_factory_(module_factory),
          execution_mode_(spec.execution_mode) {

    RT_LOGC_INFO(
            PuschComponent::PuschPipeline,
            "Constructing pipeline {} with execution mode={}",
            pipeline_id_,
            ::wise_enum::to_string(execution_mode_));

    if (spec.modules.size() != NUM_PUSCH_MODULES) {
        const std::string error_msg = std::format(
                "PuschPipeline '{}': requires exactly {} modules, got {}",
                pipeline_id_,
                NUM_PUSCH_MODULES,
                spec.modules.size());
        RT_LOGC_ERROR(PuschComponent::PuschPipeline, "{}", error_msg);
        throw std::invalid_argument(error_msg);
    }

    // Create modules using factory
    create_modules_from_spec(spec);

    // Build module execution order
    modules_ = {
            inner_rx_module_.get(),
            ldpc_derate_match_module_.get(),
            ldpc_decoder_module_.get(),
            crc_decoder_module_.get()};

    RT_LOGC_DEBUG(PuschComponent::PuschPipeline, "Constructor complete");
}

void PuschPipeline::create_modules_from_spec(const pipeline::PipelineSpec &spec) {

    // Map module types to their corresponding member unique_ptrs using reference_wrapper
    std::unordered_map<std::string_view, std::reference_wrapper<std::unique_ptr<pipeline::IModule>>>
            module_map = {
                    {"inner_rx_module", std::ref(inner_rx_module_)},
                    {"ldpc_derate_match_module", std::ref(ldpc_derate_match_module_)},
                    {"ldpc_decoder_module", std::ref(ldpc_decoder_module_)},
                    {"crc_decoder_module", std::ref(crc_decoder_module_)}};

    for (const auto &module_spec : spec.modules) {
        const auto &module_info = module_spec.get();
        auto module = module_factory_->create_module(
                module_info.module_type, module_info.instance_id, module_info.init_params);
        if (!module) {
            const std::string error_msg =
                    std::format("Failed to create module '{}'", module_info.instance_id);
            RT_LOGEC_ERROR(
                    PuschComponent::PuschPipeline,
                    PuschPipelineEvent::CreateModules,
                    "{}",
                    error_msg);
            throw std::runtime_error(error_msg);
        }

        try {
            module_map.at(module_info.module_type).get() = std::move(module);
        } catch (const std::out_of_range &) {
            const std::string error_msg =
                    std::format("Unknown module type '{}'", module_info.module_type);
            RT_LOGC_ERROR(PuschComponent::PuschPipeline, "{}", error_msg);
            throw std::runtime_error(error_msg);
        }
        RT_LOGC_INFO(
                PuschComponent::PuschPipeline,
                "Successfully created module '{}'",
                module_info.instance_id);
    }
    RT_LOGC_INFO(PuschComponent::PuschPipeline, "Successfully created all modules via factory");
}

// ============================================================================
// Setup Phase (one-time initialization)
// ============================================================================

void PuschPipeline::setup() {
    RT_LOGC_INFO(
            PuschComponent::PuschPipeline,
            "Pipeline '{}': setup() called (mode={})",
            pipeline_id_,
            ::wise_enum::to_string(execution_mode_));

    // Configure zero-copy based on execution mode and memory characteristics

    // 1. External → Inner Rx connections
    // External inputs don't have characteristics API, so we use execution mode
    if (execution_mode_ == pipeline::ExecutionMode::Stream) {
        // Stream mode: Inner Rx module doesn't require stable addresses (uses
        // set_tensor_address())
        inner_rx_module_->set_connection_copy_mode("xtf", pipeline::ConnectionCopyMode::ZeroCopy);
        RT_LOGC_INFO(
                PuschComponent::PuschPipeline,
                "Pipeline '{}': Stream mode - external -> Inner Rx zero-copy enabled",
                pipeline_id_);
    } else {
        // Graph mode: Inner Rx module requires fixed addresses for CUDA graph capture
        // Order Kernel provides fixed address, so zero-copy is safe!
        inner_rx_module_->set_connection_copy_mode("xtf", pipeline::ConnectionCopyMode::ZeroCopy);
        RT_LOGC_INFO(
                PuschComponent::PuschPipeline,
                "Pipeline '{}': Graph mode - external (Order Kernel) -> Inner Rx zero-copy "
                "enabled "
                "(Order Kernel provides fixed address)",
                pipeline_id_);
    }

    // 2. Inner Rx → LDPC connection (module-to-module negotiation)
    auto front_end_output_chars = inner_rx_module_->get_output_memory_characteristics("llrs");
    auto ldpc_input_chars = ldpc_derate_match_module_->get_input_memory_characteristics("llrs");

    if (pipeline::can_zero_copy(front_end_output_chars, ldpc_input_chars)) {
        ldpc_derate_match_module_->set_connection_copy_mode(
                "llrs", pipeline::ConnectionCopyMode::ZeroCopy);
        RT_LOGC_INFO(
                PuschComponent::PuschPipeline,
                "Pipeline '{}': InnerRx to LDPC zero-copy enabled (InnerRx provides_fixed={}, "
                "LDPC requires_fixed={})",
                pipeline_id_,
                front_end_output_chars.provides_fixed_address_for_zero_copy ? "true" : "false",
                ldpc_input_chars.requires_fixed_address_for_zero_copy ? "true" : "false");
    } else {
        ldpc_derate_match_module_->set_connection_copy_mode(
                "llrs", pipeline::ConnectionCopyMode::Copy);
        RT_LOGC_INFO(
                PuschComponent::PuschPipeline,
                "Pipeline '{}': Inner Rx→LDPC requires copy (InnerRx provides_fixed={}, "
                "LDPC requires_fixed={})",
                pipeline_id_,
                front_end_output_chars.provides_fixed_address_for_zero_copy ? "true" : "false",
                ldpc_input_chars.requires_fixed_address_for_zero_copy ? "true" : "false");
    }

    // NOW allocate memory (get_requirements() will see the configured flags)
    memory_mgr_ = pipeline::PipelineMemoryManager::create_for_modules(modules_);

    RT_LOGC_DEBUG(
            PuschComponent::PuschPipeline, "Pipeline '{}': Memory manager created", pipeline_id_);

    // Allocate memory slices for all modules
    memory_mgr_->allocate_all_module_slices(modules_);

    // Call setup_memory() on each module with their memory slice
    for (auto *module : modules_) {
        const auto slice = memory_mgr_->get_module_slice(module->get_instance_id());
        module->setup_memory(slice);
    }

    RT_LOGC_INFO(PuschComponent::PuschPipeline, "Pipeline '{}': setup() complete", pipeline_id_);
}

void PuschPipeline::warmup(cudaStream_t stream) {
    RT_LOGC_INFO(
            PuschComponent::PuschPipeline,
            "Pipeline '{}': warmup(stream={}) called - performing "
            "one-time module initialization",
            pipeline_id_,
            static_cast<void *>(stream));

    // Copy all static kernel descriptors to device in one bulk operation
    // This should be done before warmup so modules can use their static params
    memory_mgr_->copy_all_static_descriptors_to_device(stream);

    // Call warmup() on all modules with the provided stream
    for (auto *module : modules_) {
        RT_LOGC_DEBUG(
                PuschComponent::PuschPipeline,
                "Pipeline '{}': Calling warmup(stream) on module '{}'",
                pipeline_id_,
                module->get_instance_id());
        module->warmup(stream);
    }

    RT_LOGC_INFO(
            PuschComponent::PuschPipeline,
            "Pipeline '{}': warmup() complete - all modules ready "
            "for execution",
            pipeline_id_);
}

// ============================================================================
// Per-Iteration Configuration
// ============================================================================

void PuschPipeline::configure_io(
        const pipeline::DynamicParams &params,
        std::span<const pipeline::PortInfo> external_inputs,
        std::span<pipeline::PortInfo> external_outputs,
        cudaStream_t stream) {

    RT_LOGC_INFO(
            PuschComponent::PuschPipeline,
            "Pipeline '{}': configure_io(stream={})",
            pipeline_id_,
            static_cast<void *>(stream));

    // Extract PuschDynamicParams from module_specific_params and update pointers
    try {
        pusch_input_ = std::any_cast<const PuschDynamicParams &>(params.module_specific_params);
    } catch (const std::bad_any_cast &) {
        const std::string error_msg = "Invalid dynamic parameters!";
        RT_LOGC_ERROR(PuschComponent::PuschPipeline, "{}", error_msg);
        throw std::invalid_argument(error_msg);
    }

    // Step 1: Route external inputs to Inner Rx module and call configure_io
    inner_rx_module_->set_inputs(external_inputs);

    // Called before get_outputs() to support dynamic topology
    pipeline::DynamicParams inner_rx_dynamic_params{};
    inner_rx_dynamic_params.module_specific_params = pusch_input_.value().inner_rx_params;
    inner_rx_module_->configure_io(inner_rx_dynamic_params, stream);

    // Step 2: Get Inner Rx outputs (AFTER configure_io, supports dynamic determination)
    const auto inner_rx_module_outputs = inner_rx_module_->get_outputs();
    if (inner_rx_module_outputs.empty()) {
        const std::string error_msg = std::format(
                "PuschPipeline '{}': Inner Rx module did not provide any outputs", pipeline_id_);
        RT_LOGC_ERROR(PuschComponent::PuschPipeline, "{}", error_msg);
        throw std::runtime_error(error_msg);
    }
    std::size_t external_output_index = 0;
    for (const auto &output : inner_rx_module_outputs) {
        if (output.name == "post_eq_noise_var_db" || output.name == "post_eq_sinr_db") {
            if (external_output_index >= external_outputs.size()) {
                const std::string error_msg = std::format(
                        "PuschPipeline '{}': Too many outputs, external_outputs buffer overflow",
                        pipeline_id_);
                RT_LOGC_ERROR(PuschComponent::PuschPipeline, "{}", error_msg);
                throw std::runtime_error(error_msg);
            }
            external_outputs[external_output_index++] = output;
        }
    }

    // Step 3: Route Inner Rx outputs to LDPC derate match module and call configure_io
    pipeline::DynamicParams outer_rx_dynamic_params{};
    auto pusch_outer_rx_params = pusch_input_.value().outer_rx_params;
    pusch_outer_rx_params.copy_tb_params_to_gpu(stream);
    outer_rx_dynamic_params.module_specific_params = pusch_outer_rx_params;

    // LDPC derate match module expects two ports, "llrs" and "llrs_cdm1"
    // For now we just use the same input for both ports.
    std::vector<pipeline::PortInfo> de_rm_inputs{
            inner_rx_module_outputs[0], inner_rx_module_outputs[0]};
    de_rm_inputs[0].name = "llrs";
    de_rm_inputs[1].name = "llrs_cdm1";

    ldpc_derate_match_module_->set_inputs(de_rm_inputs);
    ldpc_derate_match_module_->configure_io(outer_rx_dynamic_params, stream);
    auto de_rm_outputs = ldpc_derate_match_module_->get_outputs();
    if (de_rm_outputs.empty()) {
        const std::string error_msg = std::format(
                "PuschPipeline '{}': LDPC derate match module did not provide any outputs",
                pipeline_id_);
        RT_LOGC_ERROR(PuschComponent::PuschPipeline, "{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    // Step 4: Route LDPC derate match outputs to LDPC decoder module and call configure_io
    de_rm_outputs[0].name = "llrs";
    ldpc_decoder_module_->set_inputs(de_rm_outputs);
    ldpc_decoder_module_->configure_io(outer_rx_dynamic_params, stream);
    const auto ldpc_decoder_outputs = ldpc_decoder_module_->get_outputs();
    if (ldpc_decoder_outputs.empty()) {
        const std::string error_msg = std::format(
                "PuschPipeline '{}': LDPC decoder module did not provide any outputs",
                pipeline_id_);
        RT_LOGC_ERROR(PuschComponent::PuschPipeline, "{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    // Step 5: Route LDPC decoder outputs to CRC decoder module and call configure_io
    crc_decoder_module_->set_inputs(ldpc_decoder_outputs);
    crc_decoder_module_->configure_io(outer_rx_dynamic_params, stream);
    const auto crc_decoder_outputs = crc_decoder_module_->get_outputs();
    if (crc_decoder_outputs.empty()) {
        const std::string error_msg = std::format(
                "PuschPipeline '{}': CRC decoder module did not provide any outputs", pipeline_id_);
        RT_LOGC_ERROR(PuschComponent::PuschPipeline, "{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    // Copy all dynamic kernel descriptors to device in one bulk operation
    // This should be done after all modules have updated their dynamic params
    memory_mgr_->copy_all_dynamic_descriptors_to_device(stream);

    // Synchronize to ensure descriptors are copied before graph capture
    // Graph capture needs descriptor data to be present in device memory
    FRAMEWORK_CUDA_RUNTIME_CHECK_THROW(cudaStreamSynchronize(stream));

    // Route LDPC module outputs to external outputs
    for (const auto &output : crc_decoder_outputs) {
        if (output.name == "tb_crcs" || output.name == "tb_payloads") {
            if (external_output_index >= external_outputs.size()) {
                const std::string error_msg = std::format(
                        "PuschPipeline '{}': Too many outputs, external_outputs buffer overflow",
                        pipeline_id_);
                RT_LOGC_ERROR(PuschComponent::PuschPipeline, "{}", error_msg);
                throw std::runtime_error(error_msg);
            }
            external_outputs[external_output_index++] = output;
        }
    }

    RT_LOGC_INFO(
            PuschComponent::PuschPipeline, "Pipeline '{}': configure_io complete", pipeline_id_);
}

// ============================================================================
// Execution - Stream Mode
// ============================================================================

void PuschPipeline::execute_stream(cudaStream_t stream) {
    // Validate execution mode
    if (execution_mode_ != pipeline::ExecutionMode::Stream) {
        const std::string error_msg = std::format(
                "PuschPipeline '{}': execute_stream() called but "
                "execution_mode is Graph. "
                "Use execute_graph() for Graph mode.",
                pipeline_id_);
        RT_LOGC_ERROR(PuschComponent::PuschPipeline, "{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    RT_LOGC_DEBUG(
            PuschComponent::PuschPipeline,
            "Pipeline '{}': execute_stream() on stream {}",
            pipeline_id_,
            static_cast<void *>(stream));

    // Execute modules in order using IStreamExecutor
    for (auto *module : modules_) {
        auto *stream_executor = module->as_stream_executor();
        if (stream_executor != nullptr) {
            stream_executor->execute(stream); // Uses parameters from configure_io()
        }
    }

    RT_LOGC_DEBUG(
            PuschComponent::PuschPipeline, "Pipeline '{}': execute_stream complete", pipeline_id_);
}

// ============================================================================
// Execution - Graph Mode
// ============================================================================

void PuschPipeline::build_graph(cudaStream_t stream) {
    if (graph_built_) {
        RT_LOGC_DEBUG(
                PuschComponent::PuschPipeline,
                "Pipeline '{}': Graph already built, skipping build_graph()",
                pipeline_id_);
        return;
    }

    RT_LOGC_INFO(
            PuschComponent::PuschPipeline,
            "Pipeline '{}': build_graph() - building graph for first time",
            pipeline_id_);

    // Create graph manager (constructor implicitly creates empty CUDA graph)
    graph_manager_ = std::make_unique<pipeline::GraphManager>();

    // Add modules to graph in execution order
    std::vector<CUgraphNode> prev_nodes;

    for (auto *module : modules_) {
        auto *graph_provider = module->as_graph_node_provider();
        if (graph_provider == nullptr) {
            const std::string error_msg = std::format(
                    "PuschPipeline '{}': Module '{}' does not implement "
                    "IGraphNodeProvider",
                    pipeline_id_,
                    std::string(module->get_instance_id()));
            RT_LOGC_ERROR(PuschComponent::PuschPipeline, "{}", error_msg);
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
    graph_manager_->upload_graph(stream);

    graph_built_ = true;
    RT_LOGC_INFO(
            PuschComponent::PuschPipeline,
            "Pipeline '{}': build_graph() complete - graph ready "
            "for execution",
            pipeline_id_);
}

void PuschPipeline::execute_graph(cudaStream_t stream) {
    // Build graph on first execution (idempotent)
    build_graph(stream);

    // Validate execution mode
    if (execution_mode_ != pipeline::ExecutionMode::Graph) {
        const std::string error_msg = std::format(
                "PuschPipeline '{}': execute_graph() called but "
                "execution_mode is Stream. "
                "Use execute_stream() for Stream mode.",
                pipeline_id_);
        RT_LOGC_ERROR(PuschComponent::PuschPipeline, "{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    RT_LOGC_DEBUG(
            PuschComponent::PuschPipeline,
            "PuschPipeline '{}': execute_graph() on stream {}",
            pipeline_id_,
            static_cast<void *>(stream));

    if (!graph_manager_) {
        const std::string error_msg = std::format(
                "PuschPipeline '{}': build_graph() must be called before "
                "execute_graph()",
                pipeline_id_);
        RT_LOGC_ERROR(PuschComponent::PuschPipeline, "{}", error_msg);
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

    RT_LOGC_DEBUG(
            PuschComponent::PuschPipeline, "Pipeline '{}': execute_graph complete", pipeline_id_);
}

} // namespace ran::pusch
