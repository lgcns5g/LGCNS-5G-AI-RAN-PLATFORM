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

#include <algorithm> // for copy
#include <any>
#include <array>     // for array
#include <atomic>    // for atomic, memory_order_relaxed
#include <cstdint>   // for int32_t, uint32_t, uint8_t
#include <cstdlib>   // for rand, RAND_MAX
#include <format>    // for format, format_string
#include <iosfwd>    // for ptrdiff_t
#include <memory>    // for make_unique, unique_ptr
#include <optional>  // for optional, nullopt
#include <span>      // for span
#include <stdexcept> // for out_of_range
#include <string>
#include <utility> // for pair, make_pair
#include <vector>  // for vector

#include <driver_types.h>    // for CUstream_st, cudaMemcpyKind
#include <quill/LogMacros.h> // for QUILL_LOG_ERROR, QUILL_LOG_INFO

#include <gsl-lite/gsl-lite.hpp> // for not_null

#include <cuda_runtime_api.h> // for cudaMalloc, cudaMemcpy

#include "driver/driver_log.hpp"
#include "driver/pusch_pipeline_context.hpp"
#include "ldpc/crc_decoder_module.hpp"
#include "ldpc/derate_match_params.hpp" // for ModulationOrder, NewDataIndicator
#include "ldpc/ldpc_decoder_module.hpp"
#include "ldpc/ldpc_derate_match_module.hpp"
#include "ldpc/ldpc_params.hpp" // for LdpcParams
#include "ldpc/outer_rx_params.hpp"
#include "log/rt_log_macros.hpp"
#include "pipeline/imodule_factory.hpp" // for IModuleFactory
#include "pipeline/types.hpp"
#include "pusch/inner_rx_module.hpp"
#include "pusch/pusch_defines.hpp" // for PuschUeParams, PuschInput
#include "pusch/pusch_module_factories.hpp"
#include "pusch/pusch_pipeline.hpp"
#include "ran_common.hpp" // for MAX_PUSCH_PIPELINES
#include "scf_5g_fapi.h"
#include "tensor/data_types.hpp"
#include "tensor/tensor_info.hpp" // for TensorInfo
#include "utils/error_macros.hpp" // for FRAMEWORK_CUDA_RUNTIME_CHECK_THROW

namespace ran::driver {

namespace pipeline = framework::pipeline;
namespace common = ran::common;
namespace pusch = ran::pusch;
namespace ldpc = ran::ldpc;

// ============================================================================
// PuschPipelineContext Implementation
// ============================================================================

PuschPipelineContext::PuschPipelineContext() {
    // pipeline_resources_ and host_buffers_ arrays are automatically initialized
    // Each PuschPipelineResources will create its own stream in its constructor

    // Initialize all pipeline resources as free (not allocated)
    for (auto &allocated : pipeline_allocated_) {
        allocated.store(false, std::memory_order_relaxed);
    }

    // Initialize all host input resources as free (not allocated)
    for (auto &allocated : host_buffers_allocated_) {
        allocated.store(false, std::memory_order_relaxed);
    }
}

PuschPipelineResources::~PuschPipelineResources() noexcept {
    // Note: external_inputs contains borrowed pointers (owned by Order Kernel)
    // Do NOT free - zero-copy protocol means inputs are non-owning views
    // Clear borrowed pointers to nullptr for safety (defense-in-depth: prevents future double-free
    // bugs)
    for (auto &port : external_inputs) {
        for (auto &tensor : port.tensors) {
            tensor.device_ptr = nullptr;
        }
    }

    // Destroy CUDA stream (safe to call even if stream is null)
    const cudaError_t err = cudaStreamDestroy(stream);
    if (err != cudaSuccess) {
        RT_LOG_ERROR("Failed to destroy CUDA stream: {}", cudaGetErrorString(err));
    }

    // Pipeline unique_ptr will automatically clean up
    // external_inputs vector will automatically clean up
}

pipeline::PipelineSpec PuschPipelineContext::create_pusch_pipeline_spec(
        const common::PhyParams &phy_params,
        pipeline::ExecutionMode execution_mode,
        const std::string &instance_id) {

    pipeline::PipelineSpec spec;
    spec.pipeline_name = "PuschPipeline";

    // Execution mode (set first, will be used in module params)
    spec.execution_mode = execution_mode;

    // Inner Rx Module configuration
    const pusch::InnerRxModule::StaticParams inner_rx_params{
            .phy_params = phy_params, .execution_mode = spec.execution_mode};

    const pipeline::ModuleCreationInfo inner_rx_info{
            .module_type = "inner_rx_module",
            .instance_id = instance_id,
            .init_params = std::any(inner_rx_params)};

    spec.modules.emplace_back(inner_rx_info);

    // LDPC Derate Match Module configuration
    const ldpc::LdpcDerateMatchModule::StaticParams ldpc_derate_match_params{
            .max_num_tbs = ran::common::MAX_NUM_TBS,
            .max_num_cbs_per_tb = ran::common::MAX_NUM_CBS_PER_TB,
            .max_num_rm_llrs_per_cb = ran::ldpc::MAX_NUM_RM_LLRS_PER_CB,
            .max_num_ue_grps = ran::common::MAX_NUM_UE_GRPS};

    const pipeline::ModuleSpec ldpc_derate_match_module_spec{pipeline::ModuleCreationInfo{
            .module_type = "ldpc_derate_match_module",
            .instance_id = std::format("ldpc_derate_match_{}", instance_id),
            .init_params = std::any(ldpc_derate_match_params)}};

    spec.modules.emplace_back(ldpc_derate_match_module_spec);

    // LDPC Decoder Module configuration
    const ldpc::LdpcDecoderModule::StaticParams ldpc_decoder_params{
            .clamp_value = ran::ldpc::LDPC_CLAMP_VALUE,
            .max_num_iterations = ran::ldpc::LDPC_MAX_ITERATIONS,
            .max_num_cbs_per_tb = ran::common::MAX_NUM_CBS_PER_TB,
            .max_num_tbs = ran::common::MAX_NUM_TBS,
            .normalization_factor = ran::ldpc::LDPC_NORMALIZATION_FACTOR,
            .max_iterations_method = ldpc::LdpcMaxIterationsMethod::Fixed,
            .max_num_ldpc_het_configs = ran::ldpc::LDPC_MAX_HET_CONFIGS};

    const pipeline::ModuleSpec ldpc_decoder_module_spec{pipeline::ModuleCreationInfo{
            .module_type = "ldpc_decoder_module",
            .instance_id = std::format("ldpc_decoder_{}", instance_id),
            .init_params = std::any(ldpc_decoder_params)}};

    spec.modules.emplace_back(ldpc_decoder_module_spec);

    // CRC Decoder Module configuration
    const ldpc::CrcDecoderModule::StaticParams crc_decoder_params{
            .max_num_cbs_per_tb = ran::common::MAX_NUM_CBS_PER_TB,
            .max_num_tbs = ran::common::MAX_NUM_TBS};

    const pipeline::ModuleSpec crc_decoder_module_spec{pipeline::ModuleCreationInfo{
            .module_type = "crc_decoder_module",
            .instance_id = std::format("crc_decoder_{}", instance_id),
            .init_params = std::any(crc_decoder_params)}};

    spec.modules.emplace_back(crc_decoder_module_spec);

    // External I/O
    spec.external_inputs = {"xtf"};
    spec.external_outputs = {"post_eq_noise_var_db", "post_eq_sinr_db", "llr"};

    return spec;
}

/* This function is called to warmup the trt engine. Warm up of trt engine is done
at cell configuration before the slot indications start. However, during cell configuration
PUSCH values are not available. As long as we provide valid values for PUSCH configuration,
the trt engine warmup is fine. This function prepares some mock PUSCH configuration for
warmup. The values are taken from TV 7204*/
static inline pusch::PuschDynamicParams prepare_mock_pusch_dynamic_params() {
    std::vector<ldpc::SingleTbPuschOuterRxParams> pusch_outer_rx_params;
    constexpr std::size_t NUM_UES = 1;
    pusch_outer_rx_params.reserve(NUM_UES);
    std::vector<std::uint16_t> sch_user_idxs;
    sch_user_idxs.reserve(NUM_UES);
    for (std::size_t i = 0; i < NUM_UES; ++i) {
        sch_user_idxs.push_back(static_cast<std::uint16_t>(i));
        const std::uint32_t rate_matching_length = ldpc::get_rate_matching_length(
                ran::common::NUM_PRBS_SUPPORTED,
                ran::common::MAX_UL_LAYERS,
                static_cast<ldpc::ModulationOrder>(8),
                ran::common::OFDM_SYMBOLS_NORMAL_CYCLIC_PREFIX,
                2,                   // n_dmrs_cdm_grps_no_data
                0b0000000000001000); // dmrs_sym_loc_bmsk
        // Scrambling initialization
        const auto rnti = 0;
        const auto data_scram_id = 0;
        const auto scrambling_init =
                ldpc::get_scrambling_init(static_cast<std::uint32_t>(rnti), data_scram_id);
        // Create DerateMatchParams object
        std::vector<std::uint32_t> layer_map(1);
        layer_map[0] = 0;
        const ldpc::DerateMatchParams de_rm_params{
                .mod_order = static_cast<ran::ldpc::ModulationOrder>(8),
                .n_dmrs_cdm_grps_no_data = 2,
                .ndi = static_cast<ran::ldpc::NewDataIndicator>(1),
                .num_layers = 1,
                .user_group_idx = 0,
                .num_ue_grp_layers = 1,
                .layer_map = layer_map,
                .scrambling_init = scrambling_init};

        const auto code_rate = static_cast<float>(9480) / 10240.0F;
        const ldpc::LdpcParams ldpc_params(39973 * 8, code_rate, rate_matching_length, 0);
        pusch_outer_rx_params.emplace_back(ldpc_params, de_rm_params);
    }

    pusch::PuschDynamicParams pusch_dynamic_params{
            .inner_rx_params = {},
            .outer_rx_params = ldpc::PuschOuterRxParams(pusch_outer_rx_params, sch_user_idxs)};

    return pusch_dynamic_params;
}

void PuschPipelineContext::create_pusch_pipeline(
        const common::PhyParams &phy_params,
        const pipeline::ExecutionMode execution_mode,
        std::span<const pipeline::PortInfo> order_kernel_outputs) {
    auto module_factory = std::make_unique<pusch::PuschModuleFactory>();

    // Order Kernel is REQUIRED - must be provided by Fronthaul
    if (order_kernel_outputs.empty()) {
        RT_LOGEC_ERROR(
                DriverComponent::PuschPipelineContext,
                DriverEvent::CreatePuschPipeline,
                "Order Kernel outputs are required but not provided. Ensure Fronthaul is "
                "initialized "
                "and OrderKernelPipeline reference is passed to Driver.");
        throw std::runtime_error(
                "Order Kernel outputs required for PUSCH pipeline - missing Fronthaul integration");
    }

    RT_LOGC_INFO(
            DriverComponent::PuschPipelineContext,
            "Using Order Kernel outputs for PUSCH input (zero-copy mode)");

    // Create and initialize each pipeline with its resources
    for (std::size_t i = 0; i < common::MAX_PUSCH_PIPELINES; i++) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
        auto &resources = pipeline_resources_[i];

        // Create pipeline
        RT_LOGEC_INFO(
                DriverComponent::PuschPipelineContext,
                DriverEvent::CreatePuschPipeline,
                "Creating pusch_pipeline_{} instance",
                i);
        const auto pusch_pipeline_spec =
                create_pusch_pipeline_spec(phy_params, execution_mode, std::to_string(i));
        resources.pipeline = std::make_unique<pusch::PuschPipeline>(
                "pusch_pipeline_" + std::to_string(i),
                gsl_lite::not_null<pipeline::IModuleFactory *>(module_factory.get()),
                pusch_pipeline_spec);

        // Setup pipeline
        resources.pipeline->setup();

        // Configure external inputs from Order Kernel (zero-copy)
        // Map Order Kernel output port "pusch" to PUSCH input port "xtf"
        auto xtf_port = order_kernel_outputs[0]; // "pusch" port from Order Kernel

        // Validate that the port has tensors
        if (xtf_port.tensors.empty()) {
            RT_LOGEC_ERROR(
                    DriverComponent::PuschPipelineContext,
                    DriverEvent::CreatePuschPipeline,
                    "Order Kernel output port has no tensors");
            throw std::runtime_error("Order Kernel output port has no tensors");
        }

        xtf_port.name = "xtf";                         // Rename for PUSCH consumption
        resources.external_inputs.push_back(xtf_port); // Zero-copy: borrowed from Order Kernel

        RT_LOGEC_INFO(
                DriverComponent::PuschPipelineContext,
                DriverEvent::CreatePuschPipeline,
                "'{}': Using Order Kernel output buffer (zero-copy) - ptr={}, size={} bytes [{}]",
                i,
                xtf_port.tensors[0].device_ptr,
                xtf_port.tensors[0].tensor_info.get_total_elements() *
                        framework::tensor::get_nv_type_storage_element_size(
                                xtf_port.tensors[0].tensor_info.get_type()),
                framework::tensor::nv_get_data_type_string(
                        xtf_port.tensors[0].tensor_info.get_type()));

        // Step 1: configure_io (establishes connections)
        const pusch::PuschDynamicParams pusch_dynamic_params = prepare_mock_pusch_dynamic_params();
        const pipeline::DynamicParams params{.module_specific_params = pusch_dynamic_params};
        resources.pipeline->configure_io(
                params, resources.external_inputs, resources.external_outputs, resources.stream);

        // Step 2: warmup (loads TRT engine, captures graph)
        RT_LOGEC_INFO(
                DriverComponent::PuschPipelineContext,
                DriverEvent::CreatePuschPipeline,
                "'{}': Calling warmup() - loads engine, captures graph",
                i);
        resources.pipeline->warmup(resources.stream);

        // Step 3: execute (build_graph() called automatically on first execution)
        RT_LOGEC_INFO(
                DriverComponent::PuschPipelineContext,
                DriverEvent::CreatePuschPipeline,
                "'{}': Calling execute_stream() - first time",
                i);
        resources.pipeline->execute_stream(resources.stream);
        FRAMEWORK_CUDA_RUNTIME_CHECK_THROW(cudaStreamSynchronize(resources.stream));

        if (resources.external_outputs.empty() || resources.external_outputs[0].tensors.empty()) {
            RT_LOGEC_ERROR(
                    DriverComponent::PuschPipelineContext,
                    DriverEvent::CreatePuschPipeline,
                    "Pipeline initialization failed: No output tensors generated");
        } else {
            RT_LOGEC_INFO(
                    DriverComponent::PuschPipelineContext,
                    DriverEvent::CreatePuschPipeline,
                    "'{}': Pipeline initialized successfully",
                    i);
        }
    }
}

std::optional<std::pair<std::size_t, PuschPipelineResources *>>
PuschPipelineContext::get_pipeline_resource() {
    // Find first available (unallocated) pipeline resource and allocate it
    // Uses lock-free atomic operations for thread-safe allocation
    for (std::size_t i = 0; i < common::MAX_PUSCH_PIPELINES; ++i) {
        bool expected = false;
        // Atomically check if free and mark as allocated
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
        if (pipeline_allocated_[i].compare_exchange_strong(
                    expected, true, std::memory_order_acquire)) {
            RT_LOGEC_DEBUG(
                    DriverComponent::PuschPipelineContext,
                    DriverEvent::GetPipelineResource,
                    "'{}': Allocated and returning pipeline resource at index {}",
                    i);
            // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
            return std::make_pair(i, &pipeline_resources_[i]);
        }
    }

    // No free resources available
    RT_LOGEC_ERROR(
            DriverComponent::PuschPipelineContext,
            DriverEvent::GetPipelineResource,
            "No free PUSCH pipeline resources available (max: {})",
            common::MAX_PUSCH_PIPELINES);
    return std::nullopt;
}

bool PuschPipelineContext::release_pipeline_resource(const std::size_t resource_index) {
    if (resource_index >= common::MAX_PUSCH_PIPELINES) {
        RT_LOGEC_ERROR(
                DriverComponent::PuschPipelineContext,
                DriverEvent::ReleasePipelineResource,
                "Invalid pipeline resource index {} (max: {})",
                resource_index,
                common::MAX_PUSCH_PIPELINES);
        return false;
    }

    // Atomically check if allocated and mark as free (lock-free operation)
    bool expected = true;
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
    if (pipeline_allocated_[resource_index].compare_exchange_strong(
                expected, false, std::memory_order_release)) {
        RT_LOGEC_DEBUG(
                DriverComponent::PuschPipelineContext,
                DriverEvent::ReleasePipelineResource,
                "'{}': Released pipeline resource at index {}",
                resource_index);
        return true;
    }

    // Resource was not allocated (double-release attempt)
    RT_LOGEC_ERROR(
            DriverComponent::PuschPipelineContext,
            DriverEvent::ReleasePipelineResource,
            "Attempted to release pipeline resource at index {} that was not allocated",
            resource_index);
    return false;
}

std::optional<std::pair<std::size_t, PuschHostBuffers *>> PuschPipelineContext::get_host_buffers() {
    // Find first available (unallocated) host buffers and allocate it
    // Uses lock-free atomic operations for thread-safe allocation
    for (std::size_t i = 0; i < common::MAX_PUSCH_PIPELINES; ++i) {
        bool expected = false;
        // Atomically check if free and mark as allocated
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
        if (host_buffers_allocated_[i].compare_exchange_strong(
                    expected, true, std::memory_order_acquire)) {
            RT_LOGEC_DEBUG(
                    DriverComponent::PuschPipelineContext,
                    DriverEvent::GetHostBuffers,
                    "'{}': Allocated and returning host input at index {}",
                    i);
            // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
            return std::make_pair(i, &host_buffers_[i]);
        }
    }

    // No free resources available
    RT_LOGEC_ERROR(
            DriverComponent::PuschPipelineContext,
            DriverEvent::GetHostBuffers,
            "No free PUSCH host buffers available (max: {})",
            common::MAX_PUSCH_PIPELINES);
    return std::nullopt;
}

bool PuschPipelineContext::release_host_buffers(const std::size_t resource_index) {
    if (resource_index >= common::MAX_PUSCH_PIPELINES) {
        RT_LOGEC_ERROR(
                DriverComponent::PuschPipelineContext,
                DriverEvent::ReleaseHostBuffers,
                "Invalid host input index {} (max: {})",
                resource_index,
                common::MAX_PUSCH_PIPELINES);
        return false;
    }

    // Atomically check if allocated and mark as free (lock-free operation)
    bool expected = true;
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
    if (host_buffers_allocated_[resource_index].compare_exchange_strong(
                expected, false, std::memory_order_release)) {
        // Reset all counters to 0 for clean state
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
        host_buffers_[resource_index].reset();
        RT_LOGEC_DEBUG(
                DriverComponent::PuschPipelineContext,
                DriverEvent::ReleaseHostBuffers,
                "'{}': Released host input at index {}",
                resource_index);
        return true;
    }

    // Resource was not allocated (double-release attempt)
    RT_LOGEC_ERROR(
            DriverComponent::PuschPipelineContext,
            DriverEvent::ReleaseHostBuffers,
            "Attempted to release host input at index {} that was not allocated",
            resource_index);
    return false;
}

PuschHostInput &PuschPipelineContext::get_host_input_by_index(const std::size_t resource_index) {
    if (resource_index >= common::MAX_PUSCH_PIPELINES) {
        throw std::out_of_range(std::format(
                "Host input index {} out of range (max: {})",
                resource_index,
                common::MAX_PUSCH_PIPELINES));
    }
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
    return host_buffers_[resource_index].inputs;
}

PuschHostOutput &PuschPipelineContext::get_host_output_by_index(const std::size_t resource_index) {
    if (resource_index >= common::MAX_PUSCH_PIPELINES) {
        throw std::out_of_range(std::format(
                "Host output index {} out of range (max: {})",
                resource_index,
                common::MAX_PUSCH_PIPELINES));
    }
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
    return host_buffers_[resource_index].outputs;
}

cudaStream_t PuschPipelineContext::get_stream_by_index(const std::size_t pipeline_index) const {
    if (pipeline_index >= common::MAX_PUSCH_PIPELINES) {
        throw std::out_of_range(std::format(
                "Pipeline index {} out of range (max: {})",
                pipeline_index,
                common::MAX_PUSCH_PIPELINES));
    }
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
    return pipeline_resources_[pipeline_index].stream;
}

const std::vector<pipeline::PortInfo> &
PuschPipelineContext::get_external_outputs_by_index(const std::size_t pipeline_index) const {
    if (pipeline_index >= common::MAX_PUSCH_PIPELINES) {
        throw std::out_of_range(std::format(
                "Pipeline index {} out of range (max: {})",
                pipeline_index,
                common::MAX_PUSCH_PIPELINES));
    }
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
    return pipeline_resources_[pipeline_index].external_outputs;
}

std::size_t PuschPipelineContext::get_available_pipeline_count() const {
    // Lock-free counting using atomic loads
    std::size_t count = 0;
    for (const auto &allocated : pipeline_allocated_) {
        if (!allocated.load(std::memory_order_relaxed)) {
            ++count; // cppcheck-suppress useStlAlgorithm
        }
    }

    return count;
}

std::size_t PuschPipelineContext::get_available_host_buffers_count() const {
    // Lock-free counting using atomic loads
    std::size_t count = 0;
    for (const auto &allocated : host_buffers_allocated_) {
        if (!allocated.load(std::memory_order_relaxed)) {
            ++count; // cppcheck-suppress useStlAlgorithm
        }
    }

    return count;
}

pusch::PuschDynamicParams
PuschPipelineContext::prepare_pusch_dynamic_params(const pusch::PuschInput &pusch_input) {
    std::vector<ldpc::SingleTbPuschOuterRxParams> pusch_outer_rx_params;
    pusch_outer_rx_params.reserve(pusch_input.ue_params_index);
    std::vector<std::uint16_t> sch_user_idxs;
    sch_user_idxs.reserve(pusch_input.ue_params_index);

    for (std::size_t i = 0; i < pusch_input.ue_params_index; ++i) {
        const std::uint32_t rate_matching_length = ldpc::get_rate_matching_length(
                pusch_input.ue_params[i].num_prb,
                pusch_input.ue_params[i].num_layers,
                static_cast<ldpc::ModulationOrder>(pusch_input.ue_params[i].qam_mod_order),
                pusch_input.ue_params[i].num_symbols,
                pusch_input.ue_params[i].num_dmrs_cdm_grps_no_data,
                pusch_input.ue_params[i].dmrs_sym_pos_bmsk);

        // Scrambling initialization
        const auto rnti = pusch_input.ue_params[i].rnti;
        const auto data_scram_id = pusch_input.ue_params[i].data_scrambling_id;
        const auto scrambling_init =
                ldpc::get_scrambling_init(static_cast<std::uint32_t>(rnti), data_scram_id);

        // Create DerateMatchParams object
        std::vector<std::uint32_t> layer_map(pusch_input.ue_params[i].num_layers);
        for (std::size_t j = 0; j < pusch_input.ue_params[i].num_layers; ++j) {
            layer_map[j] = static_cast<std::uint32_t>(j);
        }

        const ldpc::DerateMatchParams de_rm_params{
                .mod_order = static_cast<ran::ldpc::ModulationOrder>(
                        pusch_input.ue_params[i].qam_mod_order),
                .n_dmrs_cdm_grps_no_data = pusch_input.ue_params[i].num_dmrs_cdm_grps_no_data,
                .ndi = static_cast<ran::ldpc::NewDataIndicator>(pusch_input.ue_params[i].ndi),
                .num_layers = pusch_input.ue_params[i].num_layers,
                .user_group_idx = 0,
                .num_ue_grp_layers = pusch_input.ue_params[i].num_layers,
                .layer_map = layer_map,
                .scrambling_init = scrambling_init};

        const auto code_rate =
                static_cast<float>(pusch_input.ue_params[i].target_code_rate) / 10240.0F;
        const ldpc::LdpcParams ldpc_params(
                pusch_input.ue_params[i].tb_size * 8,
                code_rate,
                rate_matching_length,
                pusch_input.ue_params[i].rv_index);
        pusch_outer_rx_params.emplace_back(ldpc_params, de_rm_params);
        sch_user_idxs.push_back(pusch_input.ue_params[i].rnti);
    }

    pusch::PuschDynamicParams pusch_dynamic_params{
            .inner_rx_params = {},
            .outer_rx_params = ldpc::PuschOuterRxParams(pusch_outer_rx_params, sch_user_idxs)};

    return pusch_dynamic_params;
}

bool PuschPipelineContext::prepare_input_data(
        const uint16_t sfn,
        const uint16_t cell_id,
        const scf_fapi_pusch_pdu_t &pusch_pdu,
        pusch::PuschInput &pusch_input,
        const common::PhyParams &phy_params) {
    // Check input parameters
    if ((pusch_pdu.bwp.bwp_size > phy_params.num_prb) ||
        (pusch_pdu.bwp.cyclic_prefix !=
         common::CYCLIC_PREFIX_NORMAL) || // Only normal cyclic prefix is supported
        (pusch_pdu.bwp.scs !=
         common::SUBCARRIER_SPACING_MU_1)) { // Only mu-1 subcarrier spacing is supported
        RT_LOGEC_ERROR(
                DriverComponent::PuschPipelineContext,
                DriverEvent::PrepareInputData,
                "'{}': Invalid input parameters");
        return false;
    }

    if (!pusch_input.check_bounds()) {
        RT_LOGEC_ERROR(
                DriverComponent::PuschPipelineContext,
                DriverEvent::PrepareInputData,
                "'{}': input parameters array overruns detected");
        return false;
    }
    // Populate PuschInput from FAPI PDU
    const uint32_t ue_params_index = pusch_input.ue_params_index;
    auto &ue_params = pusch_input.ue_params[ue_params_index];

    ue_params.sfn = sfn;
    ue_params.cell_id = cell_id;
    ue_params.rnti = pusch_pdu.rnti;
    ue_params.handle = pusch_pdu.handle;
    ue_params.target_code_rate = pusch_pdu.target_code_rate;
    ue_params.qam_mod_order = pusch_pdu.qam_mod_order;
    ue_params.mcs_index = pusch_pdu.mcs_index;
    ue_params.mcs_table = pusch_pdu.mcs_table;
    ue_params.transform_precoding = pusch_pdu.transform_precoding;
    ue_params.data_scrambling_id = pusch_pdu.data_scrambling_id;
    ue_params.num_layers = pusch_pdu.num_of_layers;
    ue_params.dmrs_sym_pos_bmsk = pusch_pdu.ul_dmrs_sym_pos;
    ue_params.num_dmrs_cdm_grps_no_data = pusch_pdu.num_dmrs_cdm_groups_no_data;
    ue_params.start_prb = pusch_pdu.rb_start;
    ue_params.num_prb = pusch_pdu.rb_size;
    ue_params.num_symbols = pusch_pdu.num_of_symbols;
    ue_params.start_symbol_index = pusch_pdu.start_symbol_index;

    // NOLINTNEXTLINE(readability-implicit-bool-conversion,hicpp-signed-bitwise)
    if (pusch_pdu.pdu_bitmap & 0x1U) {
        const uint8_t *next = &pusch_pdu.payload[0];
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast,hicpp-use-auto,modernize-use-auto)
        const scf_fapi_pusch_data_t *data = reinterpret_cast<const scf_fapi_pusch_data_t *>(next);
        ue_params.rv_index = data->rv_index;
        ue_params.harq_process_id = data->harq_process_id;
        ue_params.ndi = data->new_data_indicator;
        ue_params.tb_size = data->tb_size;
    }

    pusch_input.ue_params_index++;
    return true;
}

void PuschPipelineContext::save_host_buffers_index(
        const std::size_t slot, const std::size_t host_buffers_index) {
    if (slot >= common::NUM_SLOTS_PER_SF) {
        RT_LOGC_ERROR(
                DriverComponent::PuschPipelineContext,
                "Invalid slot {} (max: {})",
                slot,
                common::NUM_SLOTS_PER_SF);
        return;
    }

    // Check if host input index is already set for this slot (lock-free read)
    const std::ptrdiff_t current_value =
            // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
            slot_resources_[slot].host_buffers_index.load(std::memory_order_relaxed);
    if (current_value != -1) {
        RT_LOGEC_ERROR(
                DriverComponent::PuschPipelineContext,
                DriverEvent::SaveHostBuffersIndex,
                "Slot {} already has host_buffers_index={} allocated, overwriting with new "
                "value={}",
                slot,
                current_value,
                host_buffers_index);
    }

    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
    slot_resources_[slot].host_buffers_index.store(
            static_cast<std::ptrdiff_t>(host_buffers_index), std::memory_order_relaxed);

    RT_LOGC_DEBUG(
            DriverComponent::PuschPipelineContext,
            "Saved host_buffers_index={} for slot {}",
            host_buffers_index,
            slot);
}

void PuschPipelineContext::save_pipeline_index(
        const std::size_t slot, const std::size_t pipeline_index) {
    if (slot >= common::NUM_SLOTS_PER_SF) {
        RT_LOGEC_ERROR(
                DriverComponent::PuschPipelineContext,
                DriverEvent::SavePipelineIndex,
                "Invalid slot {} (max: {})",
                slot,
                common::NUM_SLOTS_PER_SF);
        return;
    }

    // Check if pipeline index is already set for this slot (lock-free read)
    const std::ptrdiff_t current_value =
            // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
            slot_resources_[slot].pipeline_index.load(std::memory_order_relaxed);
    if (current_value != -1) {
        RT_LOGEC_ERROR(
                DriverComponent::PuschPipelineContext,
                DriverEvent::SavePipelineIndex,
                "Slot {} already has pipeline_index={} allocated, overwriting with new value={}",
                slot,
                current_value,
                pipeline_index);
    }

    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
    slot_resources_[slot].pipeline_index.store(
            static_cast<std::ptrdiff_t>(pipeline_index), std::memory_order_relaxed);

    RT_LOGEC_DEBUG(
            DriverComponent::PuschPipelineContext,
            DriverEvent::SavePipelineIndex,
            "'{}': Saved pipeline_index={} for slot {}",
            pipeline_index,
            slot);
}

const PuschSlotResources &PuschPipelineContext::get_slot_resources(const std::size_t slot) const {
    if (slot >= common::NUM_SLOTS_PER_SF) {
        RT_LOGEC_ERROR(
                DriverComponent::PuschPipelineContext,
                DriverEvent::GetSlotResources,
                "Invalid slot {} (max: {})",
                slot,
                common::NUM_SLOTS_PER_SF);
        // Default-constructed PuschSlotResources has both indices set to -1
        static const PuschSlotResources empty_resources{};
        return empty_resources;
    }

    // Lock-free access to slot resources
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
    return slot_resources_[slot];
}

void PuschPipelineContext::clear_slot_resources(const std::size_t slot) {
    if (slot >= common::NUM_SLOTS_PER_SF) {
        RT_LOGEC_ERROR(
                DriverComponent::PuschPipelineContext,
                DriverEvent::ClearSlotResources,
                "Invalid slot {} (max: {})",
                slot,
                common::NUM_SLOTS_PER_SF);
        return;
    }

    // Lock-free clearing of slot resources
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
    slot_resources_[slot].clear();

    RT_LOGEC_DEBUG(
            DriverComponent::PuschPipelineContext,
            DriverEvent::ClearSlotResources,
            "'{}': Cleared resources for slot {}",
            slot);
}

} // namespace ran::driver
