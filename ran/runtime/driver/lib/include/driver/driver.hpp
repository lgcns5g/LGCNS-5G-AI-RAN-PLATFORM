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

#ifndef RAN_DRIVER_HPP
#define RAN_DRIVER_HPP

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "pipeline/types.hpp"
#include "pusch_pipeline_context.hpp"
#include "ran_common.hpp"

namespace ran::driver {

/**
 * UL Indication Callback
 *
 * Callback function type for sending UL indications from Driver to the pipeline owner.
 * Called after pipeline execution completes.
 *
 * @param[in] slot Slot number that was processed
 */
using UlIndicationCallback = std::function<void(std::size_t slot)>;

/**
 * Slot Readiness Tracking
 *
 * Tracks which cells have sent slot responses for a given slot.
 * Uses atomic operations for lock-free access.
 */
struct SlotReadyStatus {
    std::atomic<std::uint32_t> slot_rsp_rcvd{
            0}; //!< Bitmap of cells that have responded (lock-free)
    std::atomic<bool> is_completed{
            false}; //!< True when all active cells have responded (lock-free)
};

/**
 * Pipeline Driver
 *
 * Manages device memory allocation for external inputs to PUSCH pipelines.
 * Handles allocation, deallocation, and data transfer for pipeline processing.
 *
 * Features:
 * - CUDA device memory management
 * - RAII-based resource handling
 * - Support for multiple pipelines
 * - External input/output buffer management
 * - Slot response tracking across multiple cells
 */
class Driver {
public:
    Driver();

    /**
     * Destructor
     */
    ~Driver() = default;

    // Non-copyable, non-movable
    Driver(const Driver &) = delete;
    Driver &operator=(const Driver &) = delete;
    Driver(Driver &&) = delete;
    Driver &operator=(Driver &&) = delete;

    /**
     * Create PUSCH pipeline based on physical layer parameters
     *
     * @param[in] phy_params Physical layer parameters
     * @param[in] execution_mode Pipeline execution mode ("Stream" or "Graph")
     * @param[in] ul_indication_callback Callback function for sending UL indications
     * @param[in] order_kernel_outputs Order Kernel output addresses from Fronthaul (captured after
     * warmup)
     */
    void create_pusch_pipeline(
            const ran::common::PhyParams &phy_params,
            const std::string &execution_mode,
            UlIndicationCallback ul_indication_callback,
            std::span<const framework::pipeline::PortInfo> order_kernel_outputs);

    /**
     * Process slot response from a cell
     *
     * Updates the slot_ready array with the cell's response.
     * If all active cells have responded for this slot, marks the slot as completed.
     *
     * @param[in] slot Slot number (0 to ran::common::NUM_SLOTS_PER_SF-1)
     * @param[in] cell_id Cell ID that sent the response
     * @param[in] active_cell_bitmap Bitmap of currently active cells
     * @return true if all active cells have responded for this slot, false otherwise
     */
    bool process_slot_response(
            std::size_t slot, std::uint16_t cell_id, std::uint32_t active_cell_bitmap);

    /**
     * Reset slot ready status for a specific slot
     *
     * @param[in] slot Slot number (0 to ran::common::NUM_SLOTS_PER_SF-1)
     */
    void reset_slot_status(std::size_t slot);

    /**
     * Launch pipeline processing for a completed slot
     *
     * @param[in] slot Slot number (0 to ran::common::NUM_SLOTS_PER_SF-1)
     */
    void launch_pipelines(std::size_t slot);

    PuschPipelineContext pusch_pipeline_context; //!< PUSCH pipeline context
    std::array<SlotReadyStatus, ran::common::NUM_SLOTS_PER_SF>
            slot_ready{}; //!< Slot readiness tracking

private:
    /**
     * Set UL indication callback
     *
     * @param[in] callback Callback function to be invoked after pipeline execution
     */
    void set_ul_indication_callback(UlIndicationCallback callback);

    /**
     * Send UL indication
     *
     * Internal method to invoke the registered UL indication callback.
     * Retrieves external outputs from slot resources.
     *
     * @param[in] slot Slot number that was processed
     */
    void send_ul_indication(std::size_t slot);

    UlIndicationCallback ul_indication_callback_; //!< Callback for sending UL indications
};

} // namespace ran::driver

#endif // RAN_DRIVER_HPP
