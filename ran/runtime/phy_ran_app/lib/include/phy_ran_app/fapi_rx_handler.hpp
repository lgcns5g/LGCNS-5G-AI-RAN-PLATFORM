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

/**
 * @file fapi_rx_handler.hpp
 * @brief FAPI message reception and processing handler
 *
 * Provides FapiRxHandler class that owns the nvIPC endpoint and manages
 * FAPI message processing through a Sample5GPipeline instance. Replaces
 * FapiState in the phy-ran-app architecture.
 */

#ifndef RAN_PHY_RAN_APP_FAPI_RX_HANDLER_HPP
#define RAN_PHY_RAN_APP_FAPI_RX_HANDLER_HPP

#include <atomic>
#include <functional>
#include <memory>

#include <nv_ipc.hpp>

#include <gsl-lite/gsl-lite.hpp>

#include "fapi/fapi_state.hpp"
#include "ifapi_slot_info_provider.hpp"
#include "ipipeline_executor.hpp"
#include "islot_indication_sender.hpp"
#include "message_adapter/phy_stats.hpp"

// Forward declarations
namespace ran::message_adapter {
class Sample5GPipeline;
} // namespace ran::message_adapter

namespace framework::pipeline {
class IPipelineOutputProvider;
}

namespace ran::phy_ran_app {

/**
 * Callback invoked when uplink graph should be scheduled
 *
 * Called by Sample5GPipeline after all cells have sent SLOT_RESPONSE for a slot.
 * Passes the slot number for which the graph should execute.
 * Used in Phase 1 for graph scheduling.
 *
 * @param[in] slot Slot number (0-19) for which to schedule the graph
 */
using GraphScheduleCallback = std::function<void(ran::fapi::SlotInfo slot)>;

/**
 * FAPI RX message handler
 *
 * Owns the nvIPC endpoint and manages a single Sample5GPipeline instance
 * for processing FAPI messages. Replaces FapiState in the phy_ran_app architecture.
 *
 * Responsibilities:
 * - Own nvIPC endpoint lifecycle
 * - Receive messages from nvIPC in polling loop
 * - Forward messages to Sample5GPipeline for processing
 * - Provide interface access for slot indication and message capture
 *
 * Thread-safety: Not thread-safe. Should be called from single FAPI RX thread.
 * However, interface accessors (get_slot_indication_sender, get_slot_info_provider,
 * get_pipeline_executor) return pointers that may be used from different threads if
 * the underlying implementation is thread-safe.
 */
class FapiRxHandler {
public:
    /**
     * Initialization parameters type
     *
     * Reuses FapiState::InitParams for compatibility with existing nvIPC
     * configuration infrastructure.
     */
    using InitParams = ran::fapi::FapiState::InitParams;

    /**
     * Construct FapiRxHandler
     *
     * Creates nvIPC endpoint and Sample5GPipeline with graph schedule callback.
     *
     * @param[in] params Initialization parameters
     * @param[in] on_graph_schedule Callback for scheduling uplink graph
     * @param[in] output_provider Reference to pipeline output provider interface (non-owning)
     *
     * @throw std::runtime_error if nvIPC creation fails
     */
    explicit FapiRxHandler(
            const InitParams &params,
            GraphScheduleCallback on_graph_schedule,
            framework::pipeline::IPipelineOutputProvider &output_provider);

    /**
     * Destructor
     *
     * Cleans up Sample5GPipeline and closes nvIPC endpoint.
     */
    ~FapiRxHandler();

    /**
     * Copy constructor (deleted - owns nvIPC endpoint)
     */
    FapiRxHandler(const FapiRxHandler &) = delete;

    /**
     * Assignment operator (deleted - owns nvIPC endpoint)
     */
    FapiRxHandler &operator=(const FapiRxHandler &) = delete;

    /**
     * Move constructor (deleted - owns nvIPC endpoint)
     */
    FapiRxHandler(FapiRxHandler &&) = delete;

    /**
     * Move assignment operator (deleted - owns nvIPC endpoint)
     */
    FapiRxHandler &operator=(FapiRxHandler &&) = delete;

    /**
     * Receive and process FAPI messages
     *
     * Polling loop with non-blocking receive. Sleeps 100µs when no message
     * is available. Runs until running flag is cleared.
     *
     * This method should be called from the FAPI RX task in the task graph.
     *
     * @param[in] running Atomic flag to control loop execution
     */
    void receive_and_process_messages(const std::atomic<bool> &running);

    /**
     * Get slot indication sender interface
     *
     * Returns pointer to ISlotIndicationSender interface for use by the
     * slot indication timer. This design avoids the code smell of having
     * FapiRxHandler (RX concern) directly exposing send methods (TX concern).
     *
     * @return Pointer to ISlotIndicationSender, or nullptr if not available
     */
    [[nodiscard]] ran::message_adapter::ISlotIndicationSender *get_slot_indication_sender();

    /**
     * Get slot info provider for C-plane task (Phase 1)
     *
     * Returns pointer to IFapiSlotInfoProvider interface for use by the
     * C-plane processing task to access current slot info and accumulated UL-TTI messages.
     *
     * @return Pointer to IFapiSlotInfoProvider interface, or nullptr if not available
     */
    [[nodiscard]] ran::message_adapter::IFapiSlotInfoProvider *get_slot_info_provider();

    /**
     * Get pipeline executor for PUSCH task (Phase 1)
     *
     * Returns pointer to IPipelineExecutor interface for use by the
     * PUSCH RX processing task to launch pipeline execution for a given slot.
     *
     * @return Pointer to IPipelineExecutor interface, or nullptr if not available
     */
    [[nodiscard]] ran::message_adapter::IPipelineExecutor *get_pipeline_executor();

    /**
     * Get number of currently running cells
     *
     * Thread-safe: Delegates to Sample5GPipeline which uses atomic operations.
     *
     * @return Count of cells in RUNNING state, or 0 if pipeline not initialized
     */
    [[nodiscard]] std::size_t get_num_cells_running() const;

    /**
     * @brief Get snapshot of pipeline PHY statistics
     *
     * Thread-safe method to retrieve current statistics from the underlying pipeline.
     *
     * @return Snapshot of current pipeline statistics
     */
    [[nodiscard]] const ran::message_adapter::PhyStats &get_stats() const;

private:
    nv_ipc_t *ipc_{nullptr};                                           //!< Owned nvIPC endpoint
    std::unique_ptr<ran::message_adapter::Sample5GPipeline> pipeline_; //!< Pipeline instance
};

} // namespace ran::phy_ran_app

#endif // RAN_PHY_RAN_APP_FAPI_RX_HANDLER_HPP
