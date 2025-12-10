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

#ifndef RAN_DRIVER_PUSCH_PIPELINE_CONTEXT_HPP
#define RAN_DRIVER_PUSCH_PIPELINE_CONTEXT_HPP

#include <array>
#include <atomic>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <driver_types.h> // for CUstream_st, cudaStream_t
#include <quill/LogMacros.h>
#include <scf_5g_fapi.h>

#include <cuda_runtime_api.h> // for cudaStreamCreate, cudaStreamDestroy

#include "pipeline/types.hpp"
#include "pusch/pusch_defines.hpp"
#include "pusch/pusch_pipeline.hpp"
#include "ran_common.hpp"
#include "utils/error_macros.hpp"

namespace ran::driver {

/**
 * PUSCH Resource Indices per Slot
 *
 * Tracks which pipeline and host input resources are allocated for a specific slot.
 * Uses -1 as sentinel value for "not set".
 * Uses atomic operations for lock-free access.
 */
struct PuschSlotResources {
    std::atomic<std::ptrdiff_t> pipeline_index{-1};     //!< Pipeline resource index (lock-free)
    std::atomic<std::ptrdiff_t> host_buffers_index{-1}; //!< Host buffers resource index (lock-free)

    /**
     * Check if both resources are allocated
     *
     * @return true if both pipeline and host input are allocated (not -1)
     */
    [[nodiscard]] bool has_resources() const {
        return (pipeline_index.load(std::memory_order_relaxed) >= 0) &&
               (host_buffers_index.load(std::memory_order_relaxed) >= 0);
    }

    /**
     * Clear all resource indices
     */
    void clear() {
        pipeline_index.store(-1, std::memory_order_relaxed);
        host_buffers_index.store(-1, std::memory_order_relaxed);
    }
};

/**
 * PUSCH Host Input Wrapper
 *
 * Encapsulates host-side input data with RAII cleanup.
 */
struct PuschHostInput {
    pusch::PuschInput pusch_inputs{}; //!< Host-side input data

    /**
     * Default constructor - initializes vectors with proper sizes
     */
    PuschHostInput() {
        pusch_inputs.ue_params.resize(ran::common::MAX_UES_PER_SLOT);
        pusch_inputs.ue_group_idx_map.resize(ran::common::MAX_NUM_UE_GRPS);
        for (std::size_t i = 0; i < ran::common::MAX_NUM_UE_GRPS; ++i) {
            pusch_inputs.ue_group_idx_map[i].resize(ran::common::MAX_NUM_UES_PER_UE_GRP);
        }
        pusch_inputs.xtf.resize(ran::common::NUM_CELLS_SUPPORTED);
        reset();
    }

    /**
     * Reset all index counters to 0
     */
    void reset() {
        pusch_inputs.ue_params_index = 0;
        pusch_inputs.ue_group_idx_index = 0;
        for (std::size_t i = 0; i < ran::common::MAX_NUM_UE_GRPS; ++i) {
            for (std::size_t j = 0; j < ran::common::MAX_NUM_UES_PER_UE_GRP; ++j) {
                pusch_inputs.ue_group_idx_map[i][j] = -1;
            }
        }
    }

    // Non-copyable, non-movable
    PuschHostInput(const PuschHostInput &) = delete;            //!< Copy constructor (deleted)
    PuschHostInput &operator=(const PuschHostInput &) = delete; //!< Copy assignment (deleted)
    PuschHostInput(PuschHostInput &&) = delete;                 //!< Move constructor (deleted)
    PuschHostInput &operator=(PuschHostInput &&) = delete;      //!< Move assignment (deleted)
    ~PuschHostInput() = default;                                //!< Destructor (default)
};

/**
 * PUSCH Host Output Wrapper
 *
 * Encapsulates host-side output data with RAII cleanup for pinned memory.
 */
struct PuschHostOutput {
    pusch::PuschOutput pusch_outputs{}; //!< Host-side output data

    //! Maximum buffer size for transport block payloads
    static constexpr std::uint32_t TB_PAYLOAD_BUFFER_SIZE =
            (ran::common::MAX_NUM_UE_GRPS * ran::common::MAX_NUM_UES_PER_UE_GRP *
             ran::common::PUSCH_MAX_TB_SIZE_BYTES * ran::common::NUM_CELLS_SUPPORTED);

    /**
     * Constructor - allocates pinned memory for transport block payloads
     */
    PuschHostOutput() {
        pusch_outputs.tb_crcs.resize(ran::common::MAX_UES_PER_SLOT);
        pusch_outputs.post_eq_noise_var_db.resize(ran::common::MAX_UES_PER_SLOT);
        pusch_outputs.post_eq_sinr_db.resize(ran::common::MAX_UES_PER_SLOT);
        pusch_outputs.tb_payloads.resize(ran::common::NUM_CELLS_SUPPORTED);

        for (std::size_t i = 0; i < ran::common::NUM_CELLS_SUPPORTED; ++i) {
            FRAMEWORK_CUDA_RUNTIME_CHECK_THROW(
                    cudaHostAlloc(&pusch_outputs.tb_payloads[i], TB_PAYLOAD_BUFFER_SIZE, 0));
        }
    }

    ~PuschHostOutput() noexcept {
        for (std::size_t i = 0; i < ran::common::NUM_CELLS_SUPPORTED; ++i) {
            if (pusch_outputs.tb_payloads[i] != nullptr) {
                const cudaError_t err = cudaFreeHost(pusch_outputs.tb_payloads[i]);
                if (err != cudaSuccess) {
                    RT_LOG_ERROR(
                            "Failed to free host memory for tb_payloads[{}]: {}",
                            i,
                            cudaGetErrorString(err));
                }
                pusch_outputs.tb_payloads[i] = nullptr;
            }
        }
    }

    /**
     * Reset output buffer state
     */
    void reset() { std::fill(pusch_outputs.tb_crcs.begin(), pusch_outputs.tb_crcs.end(), 1); }

    // Non-copyable, non-movable
    PuschHostOutput(const PuschHostOutput &) = delete;            //!< Copy constructor (deleted)
    PuschHostOutput &operator=(const PuschHostOutput &) = delete; //!< Copy assignment (deleted)
    PuschHostOutput(PuschHostOutput &&) = delete;                 //!< Move constructor (deleted)
    PuschHostOutput &operator=(PuschHostOutput &&) = delete;      //!< Move assignment (deleted)
};

/**
 * PUSCH Host Buffers
 *
 * Encapsulates both input and output host-side buffers for PUSCH processing.
 * Provides unified management of related input/output data structures.
 */
struct PuschHostBuffers {
    PuschHostInput inputs;   //!< Host input buffers
    PuschHostOutput outputs; //!< Host output buffers

    /**
     * Default constructor
     */
    PuschHostBuffers() = default;

    /**
     * Reset both input and output buffers
     */
    void reset() {
        inputs.reset();
        outputs.reset();
    }

    // Non-copyable, non-movable
    PuschHostBuffers(const PuschHostBuffers &) = delete;            //!< Copy constructor (deleted)
    PuschHostBuffers &operator=(const PuschHostBuffers &) = delete; //!< Copy assignment (deleted)
    PuschHostBuffers(PuschHostBuffers &&) = delete;                 //!< Move constructor (deleted)
    PuschHostBuffers &operator=(PuschHostBuffers &&) = delete;      //!< Move assignment (deleted)
    ~PuschHostBuffers() = default;                                  //!< Destructor (default)
};

/**
 * PUSCH Pipeline Resource Bundle
 *
 * Encapsulates pipeline, external inputs, and CUDA stream.
 * Ensures proper RAII cleanup.
 */
struct PuschPipelineResources {
    std::unique_ptr<pusch::PuschPipeline> pipeline; //!< PUSCH pipeline instance
    std::vector<framework::pipeline::PortInfo>
            external_inputs; //!< External input ports (borrowed, non-owning)
    cudaStream_t stream{};   //!< CUDA stream for this pipeline
    std::vector<framework::pipeline::PortInfo> external_outputs; //!< External output ports

    /**
     * Constructor - creates and initializes the CUDA stream
     */
    PuschPipelineResources() : external_outputs(ran::pusch::NUM_EXTERNAL_OUTPUTS) {
        FRAMEWORK_CUDA_RUNTIME_CHECK_THROW(cudaStreamCreate(&stream));
    }

    /**
     * Destructor - cleans up all resources
     */
    ~PuschPipelineResources() noexcept;

    // Non-copyable, non-movable
    PuschPipelineResources(const PuschPipelineResources &) = delete; //!< Copy constructor (deleted)
    PuschPipelineResources &
    operator=(const PuschPipelineResources &) = delete;         //!< Copy assignment (deleted)
    PuschPipelineResources(PuschPipelineResources &&) = delete; //!< Move constructor (deleted)
    PuschPipelineResources &
    operator=(PuschPipelineResources &&) = delete; //!< Move assignment (deleted)
};

/**
 * PUSCH Pipeline Context
 *
 * Manages multiple PUSCH pipeline instances with their associated resources.
 */
class PuschPipelineContext {
public:
    /**
     * Constructor
     */
    PuschPipelineContext();

    /**
     * Destructor
     */
    ~PuschPipelineContext() = default;

    // Non-copyable, non-movable
    PuschPipelineContext(const PuschPipelineContext &) = delete;
    PuschPipelineContext &operator=(const PuschPipelineContext &) = delete;
    PuschPipelineContext(PuschPipelineContext &&) = delete;
    PuschPipelineContext &operator=(PuschPipelineContext &&) = delete;

    /**
     * Create PUSCH pipeline
     *
     * @param[in] phy_params Physical layer parameters
     * @param[in] execution_mode Execution mode (Graph or Stream)
     * @param[in] order_kernel_outputs Output buffers from Order Kernel pipeline (PUSCH IQ data)
     */
    void create_pusch_pipeline(
            const ran::common::PhyParams &phy_params,
            framework::pipeline::ExecutionMode execution_mode,
            std::span<const framework::pipeline::PortInfo> order_kernel_outputs);

    /**
     * Atomically allocate and get an available pipeline resource from the pool
     *
     * @return Optional pair of (resource_index, resource_pointer) if available, std::nullopt if all
     * resources are in use
     */
    [[nodiscard]] std::optional<std::pair<std::size_t, PuschPipelineResources *>>
    get_pipeline_resource();

    /**
     * Release a previously allocated pipeline resource
     *
     * @param[in] resource_index Index of the resource to release
     * @return true if successfully released, false if index was invalid or already free
     */
    bool release_pipeline_resource(std::size_t resource_index);

    /**
     * Atomically allocate and get an available host input from the pool
     *
     * @return Optional pair of (resource_index, resource_pointer) if available, std::nullopt if all
     * resources are in use
     */
    [[nodiscard]] std::optional<std::pair<std::size_t, PuschHostBuffers *>> get_host_buffers();

    /**
     * Release a previously allocated host buffers
     *
     * @param[in] resource_index Index of the resource to release
     * @return true if successfully released, false if index was invalid or already free
     */
    bool release_host_buffers(std::size_t resource_index);

    /**
     * Get host input by index
     *
     * @param[in] resource_index Index of the host input (0 to ran::common::MAX_PUSCH_PIPELINES-1)
     * @return Reference to PuschHostInput
     * @throws std::out_of_range if index is invalid
     */
    [[nodiscard]] PuschHostInput &get_host_input_by_index(std::size_t resource_index);

    /**
     * Get host output by index
     *
     * @param[in] resource_index Index of the host output (0 to ran::common::MAX_PUSCH_PIPELINES-1)
     * @return Reference to PuschHostOutput
     * @throws std::out_of_range if index is invalid
     */
    [[nodiscard]] PuschHostOutput &get_host_output_by_index(std::size_t resource_index);

    /**
     * Get CUDA stream by pipeline index
     *
     * @param[in] pipeline_index Index of the pipeline (0 to ran::common::MAX_PUSCH_PIPELINES-1)
     * @return CUDA stream associated with the pipeline
     * @throws std::out_of_range if index is invalid
     */
    [[nodiscard]] cudaStream_t get_stream_by_index(std::size_t pipeline_index) const;

    /**
     * Get external outputs by pipeline index
     *
     * @param[in] pipeline_index Index of the pipeline (0 to ran::common::MAX_PUSCH_PIPELINES-1)
     * @return Reference to external outputs vector
     * @throws std::out_of_range if index is invalid
     */
    [[nodiscard]] const std::vector<framework::pipeline::PortInfo> &
    get_external_outputs_by_index(std::size_t pipeline_index) const;

    /**
     * Get number of available (free) pipeline resources
     *
     * @return Number of free pipeline resources
     */
    [[nodiscard]] std::size_t get_available_pipeline_count() const;

    /**
     * Get number of available (free) host buffers
     *
     * @return Number of free host buffers
     */
    [[nodiscard]] std::size_t get_available_host_buffers_count() const;

    /**
     * Prepare PUSCH input data from FAPI PDU
     *
     * Converts FAPI PUSCH PDU to PuschInput format and validates parameters.
     *
     * @param[in] sfn System Frame Number
     * @param[in] cell_id Cell ID
     * @param[in] pusch_pdu FAPI PUSCH PDU containing input parameters
     * @param[in,out] pusch_input PuschInput structure to populate
     * @param[in] phy_params Physical layer parameters for validation
     * @return true if successful, false if validation fails
     */
    static bool prepare_input_data(
            uint16_t sfn,
            uint16_t cell_id,
            const scf_fapi_pusch_pdu_t &pusch_pdu,
            pusch::PuschInput &pusch_input,
            const ran::common::PhyParams &phy_params);

    /**
     * Prepare PUSCH dynamic parameters from input data
     *
     * @param[in] pusch_input PUSCH input data containing UE parameters
     * @return Prepared dynamic parameters for pipeline execution
     */
    [[nodiscard]] static pusch::PuschDynamicParams
    prepare_pusch_dynamic_params(const pusch::PuschInput &pusch_input);

    /**
     * Save host buffers index for a specific slot
     *
     * @param[in] slot Slot number (0 to ran::common::NUM_SLOTS_PER_SF-1)
     * @param[in] host_buffers_index Host buffers resource index
     */
    void save_host_buffers_index(std::size_t slot, std::size_t host_buffers_index);

    /**
     * Save pipeline index for a specific slot
     *
     * @param[in] slot Slot number (0 to ran::common::NUM_SLOTS_PER_SF-1)
     * @param[in] pipeline_index Pipeline resource index
     */
    void save_pipeline_index(std::size_t slot, std::size_t pipeline_index);

    /**
     * Get resource indices for a specific slot
     *
     * @param[in] slot Slot number (0 to ran::common::NUM_SLOTS_PER_SF-1)
     * @return Reference to slot resources
     */
    [[nodiscard]] const PuschSlotResources &get_slot_resources(std::size_t slot) const;

    /**
     * Clear resource indices for a specific slot
     *
     * @param[in] slot Slot number (0 to ran::common::NUM_SLOTS_PER_SF-1)
     */
    void clear_slot_resources(std::size_t slot);

private:
    /**
     * Create PUSCH pipeline specification
     *
     * @param[in] phy_params Physical layer parameters
     * @param[in] execution_mode Execution mode (Graph or Stream)
     * @param[in] instance_id Instance identifier
     * @return Pipeline specification
     */
    static framework::pipeline::PipelineSpec create_pusch_pipeline_spec(
            const ran::common::PhyParams &phy_params,
            framework::pipeline::ExecutionMode execution_mode,
            const std::string &instance_id);

    /**
     * Allocate PUSCH external outputs
     *
     * @param[in] phy_params Physical layer parameters for determining output dimensions
     * @param[in,out] external_outputs Vector to populate with allocated outputs
     */
    static void allocate_pusch_external_outputs(
            const ran::common::PhyParams &phy_params,
            std::vector<framework::pipeline::PortInfo> &external_outputs);

    std::array<PuschPipelineResources, ran::common::MAX_PUSCH_PIPELINES>
            pipeline_resources_{}; //!< Pipeline resources
    std::array<std::atomic_bool, ran::common::MAX_PUSCH_PIPELINES>
            pipeline_allocated_{}; //!< Track which pipeline resources are allocated (lock-free)

    std::array<PuschHostBuffers, ran::common::MAX_PUSCH_PIPELINES>
            host_buffers_{}; //!< Host buffers resources
    std::array<std::atomic_bool, ran::common::MAX_PUSCH_PIPELINES>
            host_buffers_allocated_{}; //!< Track which host buffers resources are allocated
                                       //!< (lock-free)

    std::array<PuschSlotResources, ran::common::NUM_SLOTS_PER_SF>
            slot_resources_{}; //!< Resource indices per slot (lock-free atomic members)
};

} // namespace ran::driver

#endif // RAN_DRIVER_PUSCH_PIPELINE_CONTEXT_HPP
