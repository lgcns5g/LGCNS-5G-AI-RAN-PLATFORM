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

#ifndef RAN_SAMPLE_5G_PIPELINE_HPP
#define RAN_SAMPLE_5G_PIPELINE_HPP

#include <array>      // for array
#include <atomic>     // for atomic
#include <cstddef>    // for size_t
#include <cstdint>    // for uint16_t, uint32_t, uint64_t
#include <functional> // for function
#include <memory>     // for unique_ptr
#include <span>       // for span
#include <string>     // for string
#include <vector>     // for vector

#include <nv_ipc.h>      // for nv_ipc_t, nv_ipc_msg_t
#include <scf_5g_fapi.h> // for scf_fapi_error_codes_t

#include "cell.hpp"                          // for Cell
#include "driver/driver.hpp"                 // for Driver
#include "driver/pusch_pipeline_context.hpp" // for PuschHostInput, PuschHostOutput
#include "fapi/fapi_file_writer.hpp"         // for CapturedFapiMessage
#include "fapi/fapi_state.hpp"
#include "ifapi_message_processor.hpp"            // for IFapiMessageProcessor
#include "ifapi_slot_info_provider.hpp"           // for IFapiSlotInfoProvider
#include "ipipeline_executor.hpp"                 // for IPipelineExecutor
#include "islot_indication_sender.hpp"            // for ISlotIndicationSender
#include "message_adapter/phy_stats.hpp"          // for PhyStats
#include "pipeline/ipipeline_output_provider.hpp" // for IPipelineOutputProvider
#include "ran_common.hpp"                         // for common defines

namespace ran::fapi_5g {
namespace pipeline = framework::pipeline;
namespace driver = ran::driver;
namespace common = ran::common;
namespace msg_adapter = ran::message_adapter;
} // namespace ran::fapi_5g

// Provide Sample5GPipeline in ran::message_adapter namespace
namespace ran::message_adapter {

using ran::fapi_5g::Cell;
using ran::fapi_5g::FapiStateT;

// Namespace aliases (cannot use 'using' for namespace aliases)
namespace common = ran::common;
namespace driver = ran::driver;
namespace pipeline = framework::pipeline;

/**
 * Callback invoked when uplink graph should be scheduled
 *
 * @param[in] slot Slot information (sfn and slot) for which to schedule the graph
 */
using GraphScheduleCallback = std::function<void(ran::fapi::SlotInfo slot)>;

/**
 * Sample 5G pipeline implementation with thread-safe operation
 *
 * This class implements FAPI message processing for 5G networks with
 * atomic operations for thread safety. Supports multiple cells with
 * independent state machines.
 *
 * Thread-safety: Uses lock-free atomics for sfn/slot counter, cell bitmap,
 * and cell states. Safe for concurrent access from FAPI RX thread and
 * slot indication timer thread.
 *
 * Implements:
 * - IFapiMessageProcessor: Process incoming FAPI messages
 * - ISlotIndicationSender: Send periodic slot indications
 * - IFapiSlotInfoProvider: Provide slot info and accumulated messages
 * - IPipelineExecutor: Execute PUSCH pipelines for a given slot
 */
class Sample5GPipeline : public IFapiMessageProcessor,
                         public ISlotIndicationSender,
                         public IFapiSlotInfoProvider,
                         public IPipelineExecutor {
public:
    /**
     * Initialization parameters for Sample5GPipeline
     */
    struct InitParams final {
        nv_ipc_t *ipc{nullptr};                             //!< nvIPC endpoint (non-owning)
        std::size_t max_cells{common::NUM_CELLS_SUPPORTED}; //!< Maximum supported cells
        GraphScheduleCallback on_graph_schedule;            //!< Callback to schedule graph
        framework::pipeline::IPipelineOutputProvider *output_provider{
                nullptr}; //!< Pipeline output provider (non-owning)
    };

    /**
     * Constructor for the sample 5G pipeline
     *
     * @param[in] params Initialization parameters
     * @throw std::invalid_argument if ipc is nullptr
     */
    explicit Sample5GPipeline(const InitParams &params);

    /**
     * Destructor
     */
    ~Sample5GPipeline() override = default;

    // Non-copyable, non-movable
    Sample5GPipeline(const Sample5GPipeline &) = delete;
    Sample5GPipeline &operator=(const Sample5GPipeline &) = delete;
    Sample5GPipeline(Sample5GPipeline &&) = delete;
    Sample5GPipeline &operator=(Sample5GPipeline &&) = delete;

    /**
     * Get 5G pipeline status information
     *
     * @return Status string containing pipeline state and statistics
     */
    [[nodiscard]] std::string get_status() const;

    /**
     * Get number of currently running cells
     *
     * Thread-safe: Uses atomic operations to read active_cell_bitmap.
     *
     * @return Count of cells in RUNNING state
     */
    [[nodiscard]] std::size_t get_num_cells_running() const;

    /**
     * @brief Get PHY phy statistics
     *
     * Safe to call after all worker threads have joined.
     * Returns const reference to atomic statistics - caller uses .load() to read values.
     *
     * @return Const reference to statistics
     */
    [[nodiscard]] const PhyStats &get_stats() const noexcept { return phy_stats_; }

    // ========== IFapiMessageProcessor Interface ==========

    /**
     * Process incoming FAPI message
     *
     * Handles ConfigRequest, StartRequest, StopRequest, UlTtiRequest,
     * DlTtiRequest, and SlotResponse messages. Updates internal state
     * atomically for thread-safe operation.
     *
     * @param[in,out] msg FAPI message to process
     */
    void process_msg(nv_ipc_msg_t &msg) override;

    // ========== ISlotIndicationSender Interface ==========

    /**
     * Send slot indications to all active cells
     *
     * Called periodically (typically every 500µs) to advance the slot counter
     * atomically and send slot indication messages via nvIPC to all running cells.
     *
     * Thread-safety: Safe to call concurrently with process_msg(). Uses atomic
     * operations for sfn/slot counter and active cell bitmap.
     */
    void send_slot_indications() override;

    // ========== IFapiSlotInfoProvider Interface ==========

    /**
     * Get current slot information (SFN and slot number)
     *
     * Returns the current SFN and slot being processed. Uses atomic operations
     * to safely read the packed sfn/slot counter.
     *
     * Thread-safe: Uses atomic load operation.
     *
     * @return ran::fapi::SlotInfo with sfn in [0, 1023] and slot in [0, 19] for 30kHz SCS
     */
    [[nodiscard]] ran::fapi::SlotInfo get_current_slot() const override;

    /**
     * Get accumulated UL-TTI messages for current slot
     *
     * Returns non-owning view of accumulated UL-TTI-REQUEST messages.
     * Messages remain valid until next slot's accumulation begins.
     *
     * @param[in] slot Slot number (0-19 for 30kHz SCS) to get accumulated messages for
     * @return Span of captured FAPI messages
     */
    [[nodiscard]] std::span<const ran::fapi::CapturedFapiMessage>
    get_accumulated_ul_tti_msgs(std::uint16_t slot) const override;

    /**
     * Get absolute slot number for given slot info
     *
     * Calculates absolute slot number accounting for SFN wrap-arounds.
     * The absolute slot is a monotonic counter that never wraps.
     * Thread-safe: Uses atomic load with acquire semantics.
     *
     * @param[in] slot_info Slot information containing SFN and slot
     * @return Absolute slot number since initialization
     */
    [[nodiscard]] std::uint64_t
    get_current_absolute_slot(ran::fapi::SlotInfo slot_info) const noexcept override;

    // ========== IPipelineExecutor Interface ==========

    /**
     * Launch PUSCH pipelines for the given slot
     *
     * Triggers execution of the PUSCH pipeline(s) for the specified slot.
     * This should be called by the PUSCH RX task after U-Plane processing
     * has prepared the I/Q data.
     *
     * Thread-safe: Delegates to Driver which uses appropriate synchronization.
     *
     * @param[in] slot Slot number to process (0-19 for 30kHz SCS)
     */
    void launch_pipelines(std::size_t slot) override;

private:
    /**
     * Send UL indications
     *
     * Callback invoked by Driver after pipeline execution completes.
     * Sends UL indication messages via IPC.
     *
     * @param[in] slot Slot number that was processed
     */
    void send_ul_indications(std::size_t slot);

    /**
     * Send RX data indication
     *
     * @param[in] slot Slot number
     * @param[in] host_input Host input containing UE parameters
     * @param[in] host_output Host output containing decoded transport blocks
     */
    void send_rx_data_indication(
            std::size_t slot,
            const driver::PuschHostInput &host_input,
            const driver::PuschHostOutput &host_output);

    /**
     * Send UL noise variance indication
     *
     * @param[in] slot Slot number
     * @param[in] host_input Host input containing UE parameters
     * @param[in] host_output Host output containing noise variance measurements
     */
    void send_ul_noise_var_indication(
            std::size_t slot,
            const driver::PuschHostInput &host_input,
            const driver::PuschHostOutput &host_output);

    // Message processing handlers
    /**
     * Process CONFIG_REQUEST message
     *
     * @param[in] config_request CONFIG_REQUEST message
     * @param[in] cell_id Cell identifier
     * @param[in] body_len Body length
     * @return Error code indicating success (MSG_OK) or failure reason
     */
    [[nodiscard]]
    scf_fapi_error_codes_t process_config_request(
            scf_fapi_config_request_msg_t &config_request, uint16_t cell_id, uint32_t body_len);

    /**
     * Process UL_TTI_REQUEST message
     *
     * @param[in] ul_tti_request UL_TTI_REQUEST message
     * @param[in] cell_id Cell identifier
     * @return true if message processed successfully
     */
    [[nodiscard]]
    bool process_ul_tti_request(const scf_fapi_ul_tti_req_t &ul_tti_request, uint16_t cell_id);

    /**
     * Process DL_TTI_REQUEST message
     *
     * @param[in] dl_tti_request DL_TTI_REQUEST message
     * @param[in] cell_id Cell identifier
     * @return true if message processed successfully
     */
    [[nodiscard]]
    // cppcheck-suppress functionStatic
    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    bool process_dl_tti_request(const scf_fapi_dl_tti_req_t &dl_tti_request, uint16_t cell_id);

    /**
     * Send CELL_CONFIG_RESPONSE message
     *
     * @param[in] cell_id Cell identifier
     * @param[in] error_code Error code to report
     * @return true if message sent successfully
     */
    [[nodiscard]]
    bool send_cell_config_response(uint16_t cell_id, scf_fapi_error_codes_t error_code);

    /**
     * Process START.request message
     *
     * @param[in] cell_id Cell identifier
     * @return Error code indicating success (MSG_OK) or failure reason
     */
    [[nodiscard]]
    scf_fapi_error_codes_t process_start_request(uint16_t cell_id);

    /**
     * Process STOP.request message
     *
     * @param[in] cell_id Cell identifier
     * @return Error code indicating success (MSG_OK) or failure reason
     */
    [[nodiscard]]
    scf_fapi_error_codes_t process_stop_request(uint16_t cell_id);

    /**
     * Send ERROR.indication message
     *
     * @param[in] cell_id Cell identifier
     * @param[in] msg_id Message ID that caused error
     * @param[in] error_code Error code to report
     * @return true if message sent successfully
     */
    [[nodiscard]]
    bool send_error_indication(
            uint16_t cell_id,
            scf_fapi_message_id_e msg_id,
            scf_fapi_error_codes_t error_code) const;

    /**
     * Process SLOT_RESPONSE message
     *
     * @param[in] slot_response SLOT_RESPONSE message
     * @param[in] cell_id Cell identifier
     */
    void process_slot_response(const scf_fapi_slot_rsp_t &slot_response, uint16_t cell_id);

    // ========== Helper Functions for Atomic SFN/Slot ==========

    static constexpr auto NUM_BITS_SHIFT_FOR_PACKED_SLOT = 16U;
    static constexpr std::uint16_t L2_TIMING_ADVANCE = 3; //!< L2 timing advance in slots

    /**
     * Pack SFN and slot into single uint32_t for atomic operations
     *
     * @param[in] sfn SFN value (0-1023)
     * @param[in] slot Slot value (0-19)
     * @return Packed value (sfn in upper 16 bits, slot in lower 16 bits)
     */
    [[nodiscard]]
    static constexpr uint32_t pack_sfn_slot(const uint16_t sfn, const uint16_t slot) noexcept {
        return (static_cast<uint32_t>(sfn) << NUM_BITS_SHIFT_FOR_PACKED_SLOT) | slot;
    }

    /**
     * Extract SFN from packed uint32_t
     *
     * @param[in] packed Packed sfn/slot value
     * @return SFN value (upper 16 bits)
     */
    [[nodiscard]]
    static constexpr uint16_t unpack_sfn(const uint32_t packed) noexcept {
        return static_cast<uint16_t>(packed >> NUM_BITS_SHIFT_FOR_PACKED_SLOT);
    }

    /**
     * Extract slot from packed uint32_t
     *
     * @param[in] packed Packed sfn/slot value
     * @return Slot value (lower 16 bits)
     */
    [[nodiscard]]
    static constexpr uint16_t unpack_slot(const uint32_t packed) noexcept {
        return static_cast<uint16_t>(packed & ((1U << NUM_BITS_SHIFT_FOR_PACKED_SLOT) - 1));
    }

    // ========== Member Variables ==========

    // Statistics (non-atomic, only updated from FAPI RX thread)
    unsigned int processed_messages_{0}; //!< Counter for processed messages
    unsigned int total_errors_{0};       //!< Counter for processing errors

    // Thread-safe timing and state (atomic for concurrent access)
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers, readability-magic-numbers)
    std::atomic_uint32_t sfn_slot_packed_{pack_sfn_slot(
            common::NUM_SFNS_PER_FRAME - 1,
            common::NUM_SLOTS_PER_SF - 1)}; //!< Packed SFN/slot (atomic for thread safety)
    std::atomic_uint64_t sfn_wrap_counter_{
            0}; //!< Counts SFN wrap-arounds (for absolute slot calculation)
    std::atomic_uint64_t active_cell_bitmap_{
            0}; //!< Bitmap tracking running cells (atomic for thread safety)

    // Configuration (immutable after construction, safe without atomics)
    uint32_t num_cells_configured_{0}; //!< Number of configured cells
    std::size_t max_cells_{};          //!< Maximum supported cells
    std::array<uint32_t, common::NUM_CELLS_SUPPORTED> cell_id_map_{}; //!< Map cell_id to index

    // Per-cell data (cell state is atomic, others immutable after config)
    std::array<std::unique_ptr<Cell>, common::NUM_CELLS_SUPPORTED> cells_{}; //!< Cell instances

    // Driver and IPC (set once, used from multiple threads - safe)
    driver::Driver driver_;  //!< PUSCH pipeline driver
    nv_ipc_t *ipc_{nullptr}; //!< IPC context (non-owning pointer)
    framework::pipeline::IPipelineOutputProvider *output_provider_{
            nullptr};    //!< Pipeline output provider (non-owning)
    PhyStats phy_stats_; //!< PHY statistics collector

    // Callbacks (set once in constructor, invoked from FAPI RX thread only)
    GraphScheduleCallback on_graph_schedule_; //!< Callback to schedule uplink graph

    // Message accumulation per slot
    struct SlotAccumulation {
        std::vector<ran::fapi::CapturedFapiMessage> messages; //!< Accumulated UL-TTI messages
        std::atomic_bool done{
                false}; //!< True when SLOT_RESPONSE received and accumulation complete
    };
    std::array<SlotAccumulation, common::NUM_SLOTS_PER_SF>
            slot_accumulation_{}; //!< Per-slot message accumulation state
};

} // namespace ran::message_adapter

#endif // RAN_SAMPLE_5G_PIPELINE_HPP
