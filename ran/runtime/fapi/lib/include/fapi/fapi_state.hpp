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
 * @file fapi_state.hpp
 * @brief FAPI state machine and NVIPC message management
 *
 * Provides FapiState class for managing 5G NR FAPI (PHY-MAC interface) message
 * processing via NVIPC transport, including per-cell state machines, message routing,
 * and callback-based event forwarding.
 */

#ifndef RAN_FAPI_STATE_HPP
#define RAN_FAPI_STATE_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <nv_ipc.h>
#include <scf_5g_fapi.h>

#include <wise_enum.h>

namespace ran::fapi {

/**
 * FAPI state machine states for each cell
 */
enum class FapiStateT : std::uint8_t {
    FapiStateIdle = 0,   //!< Cell not configured
    FapiStateConfigured, //!< Cell configured but not running
    FapiStateRunning,    //!< Cell running and processing slots
    FapiStateStopped     //!< Cell stopped after running
};

} // namespace ran::fapi

// WISE_ENUM_ADAPT must be called at global namespace scope for ADL to work
WISE_ENUM_ADAPT(
        ran::fapi::FapiStateT,
        FapiStateIdle,
        FapiStateConfigured,
        FapiStateRunning,
        FapiStateStopped)

namespace ran::fapi {

/**
 * Per-cell configuration and state
 */
struct CellConfig {
    std::atomic<FapiStateT> state{
            FapiStateT::FapiStateIdle}; //!< Current FAPI state (atomic for thread-safe access)
    uint16_t cell_id{};                 //!< Cell index
    uint16_t phy_cell_id{};             //!< Physical cell ID from CONFIG.request
    uint16_t num_rx_ant{0};             //!< Number of RX antennas from CONFIG.request
};

/**
 * 5G NR slot timing information
 */
struct SlotInfo {
    uint16_t sfn{};  //!< System Frame Number (0-1023)
    uint16_t slot{}; //!< Slot number (0-19 for 30kHz SCS)

    /**
     * Three-way comparison operator
     *
     * @return std::strong_ordering result of comparison
     */
    auto operator<=>(const SlotInfo &) const = default;
};

/**
 * Message data for capture/replay
 *
 * Contains raw FAPI message data without transport-specific dependencies.
 * Uses std::span for zero-copy view of message buffers.
 */
struct FapiMessageData {
    uint16_t cell_id{};                //!< Cell identifier
    scf_fapi_message_id_e msg_id{};    //!< FAPI message type
    std::span<const uint8_t> msg_buf;  //!< FAPI body (body_hdr + body)
    std::span<const uint8_t> data_buf; //!< Optional data buffer (empty if not present)
};

/**
 * FAPI state machine manager for 5G NR PHY-MAC interface
 *
 * Provides comprehensive management of FAPI message processing, including:
 * - NVIPC transport lifecycle (initialization, message send/receive, cleanup via RAII)
 * - Per-cell state machines (Idle → Configured → Running → Stopped transitions)
 * - Message routing and validation for CONFIG/START/STOP.request messages
 * - User-configurable callbacks for UL_TTI_REQUEST, DL_TTI_REQUEST, SLOT_RESPONSE forwarding
 * - Automatic SLOT.indication generation with SFN/slot tracking and wraparound
 * - ERROR.indication and STOP.indication transmission
 *
 * Message flow:
 * 1. Receive messages via NVIPC (receive_message)
 * 2. Process and route via process_message (handles state transitions, invokes callbacks)
 * 3. Send indications via NVIPC (send_slot_indication, send_stop_indication, etc.)
 *
 * State machine: Each cell independently transitions through states based on
 * CONFIG.request → START.request → STOP.request message sequence.
 *
 * Thread safety:
 * - Thread-safe methods (can be called concurrently from multiple threads):
 *   - get_cell_state(), get_num_cells_configured(), get_num_cells_running()
 *   - allocate_message(), receive_message(), send_message(), release_message()
 *   - process_message() (cell state transitions use atomics; nvipc provides buffer synchronization)
 *   - increment_slot(), reset_slot(), get_current_slot() (use packed atomic for slot/sfn)
 *   - send_slot_indication(), send_config_response(), send_stop_indication(),
 * send_error_indication()
 * - NOT thread-safe methods (require external synchronization if used from multiple threads):
 *   - All callback setters (set_on_*)
 * - Note: nvipc operations (allocate/send/receive/release) are internally thread-safe; memory
 * fences ensure proper buffer visibility between TX and RX threads. Slot management uses lock-free
 * atomics with packed uint32_t representation (sfn in upper 16 bits, slot in lower 16 bits).
 *
 * @note Destructor automatically frees NVIPC resources and unlinks /dev/shm files
 *       via RAII through NvIpcDeleter.
 */
class FapiState {
public:
    /**
     * Configuration parameters for FapiState
     */
    struct InitParams {
        static constexpr std::size_t DEFAULT_MAX_CELLS = 20; //!< Default maximum cells
        static constexpr uint16_t DEFAULT_MAX_SFN = 1024;    //!< Default maximum SFN
        static constexpr uint16_t DEFAULT_MAX_SLOT = 20;     //!< Default maximum slot

        std::string nvipc_config_file;   //!< Full path to NVIPC config YAML
        std::string nvipc_config_string; //!< NVIPC config YAML as string (used if nvipc_config_file
                                         //!< is empty)
        std::size_t max_cells{DEFAULT_MAX_CELLS}; //!< Maximum supported cells
        uint16_t max_sfn{DEFAULT_MAX_SFN};        //!< Maximum SFN value (wraps to 0)
        uint16_t max_slot{DEFAULT_MAX_SLOT};      //!< Maximum slot value (wraps to 0)
    };

    /**
     * Construct and initialize FAPI state machine
     *
     * Initializes NVIPC from config file or string in constructor.
     * If nvipc_config_file is provided, uses file; otherwise uses nvipc_config_string.
     * Throws std::runtime_error if NVIPC initialization fails.
     *
     * @param[in] params Configuration parameters
     */
    explicit FapiState(const InitParams &params);

    /**
     * Allocate a TX message buffer with proper synchronization
     *
     * Allocates a buffer from the NVIPC free pool with acquire fence to ensure
     * the buffer is clean after allocation.
     *
     * @param[out] msg Message structure to populate
     * @return 0 on success, negative on failure
     */
    [[nodiscard]] int allocate_message(nv_ipc_msg_t &msg);

    /**
     * Receive a message from NVIPC
     *
     * Non-blocking receive. Returns negative value if no message available.
     * Forwards the return code from NVIPC rx_recv_msg().
     *
     * @param[out] msg Message structure to populate
     * @return >= 0 on success, < 0 if no message available (NVIPC returns -1)
     */
    [[nodiscard]] int receive_message(nv_ipc_msg_t &msg);

    /**
     * Release a message buffer back to NVIPC
     *
     * Must be called after processing each received message.
     *
     * @param[in,out] msg Message to release
     */
    void release_message(nv_ipc_msg_t &msg);

    /**
     * Process incoming FAPI message (main entry point)
     *
     * Routes message to appropriate handler based on message type.
     * Handles CONFIG.request, START.request, STOP.request, and callbacks
     * for UL_TTI_REQUEST, DL_TTI_REQUEST, SLOT_RESPONSE.
     *
     * @param[in,out] msg NVIPC message to process
     * @return Error code indicating success (MSG_OK) or failure reason
     */
    [[nodiscard]] scf_fapi_error_codes_t process_message(nv_ipc_msg_t &msg);

    /**
     * Send SLOT.indication to all running cells (thread-safe)
     *
     * Atomically captures current slot at start to ensure consistency
     * across all cells in a single invocation.
     *
     * @return true if all messages sent successfully
     */
    [[nodiscard]] bool send_slot_indication();

    /**
     * Send STOP.indication message for a cell
     *
     * @param[in] cell_id Cell identifier
     * @return true if message sent successfully
     */
    [[nodiscard]] bool send_stop_indication(uint16_t cell_id);

    /**
     * Send ERROR.indication message (thread-safe)
     *
     * Atomically captures current slot to include in error indication.
     *
     * @param[in] cell_id Cell identifier
     * @param[in] msg_id Message ID that caused error
     * @param[in] error_code Error code to report
     * @return true if message sent successfully
     */
    [[nodiscard]] bool send_error_indication(
            uint16_t cell_id, scf_fapi_message_id_e msg_id, scf_fapi_error_codes_t error_code);

    /**
     * Increment slot counter with wraparound (thread-safe)
     *
     * Advances to next slot, wrapping SFN and slot as needed.
     * Uses lock-free compare-and-swap to handle concurrent access.
     */
    void increment_slot();

    /**
     * Reset slot counter to (0, 0) (thread-safe)
     *
     * Atomically resets both SFN and slot to zero.
     */
    void reset_slot();

    /**
     * Get current slot information (thread-safe)
     *
     * Atomically reads current SFN and slot without tearing.
     *
     * @return Current SFN and slot
     */
    [[nodiscard]] SlotInfo get_current_slot() const noexcept;

    /**
     * Get state of a specific cell
     *
     * @param[in] cell_id Cell identifier
     * @return Cell state, or idle if invalid cell_id
     */
    [[nodiscard]] FapiStateT get_cell_state(uint16_t cell_id) const noexcept;

    /**
     * Get number of configured cells
     *
     * @return Count of cells in configured or later states
     */
    [[nodiscard]] std::size_t get_num_cells_configured() const noexcept;

    /**
     * Get number of running cells
     *
     * @return Count of cells in running state
     */
    [[nodiscard]] std::size_t get_num_cells_running() const noexcept;

    /**
     * Callback function type for UL TTI request events
     *
     * @param[in] cell_id Cell identifier
     * @param[in] body_hdr Message body header
     * @param[in] body_len Length of message body
     */
    using OnUlTtiRequestCallback = std::function<void(
            uint16_t cell_id, const scf_fapi_body_header_t &body_hdr, uint32_t body_len)>;

    /**
     * Callback function type for DL TTI request events
     *
     * @param[in] cell_id Cell identifier
     * @param[in] body_hdr Message body header
     * @param[in] body_len Length of message body
     */
    using OnDlTtiRequestCallback = std::function<void(
            uint16_t cell_id, const scf_fapi_body_header_t &body_hdr, uint32_t body_len)>;

    /**
     * Callback function type for slot response events
     *
     * @param[in] cell_id Cell identifier
     * @param[in] body_hdr Message body header
     * @param[in] body_len Length of message body
     */
    using OnSlotResponseCallback = std::function<void(
            uint16_t cell_id, const scf_fapi_body_header_t &body_hdr, uint32_t body_len)>;

    /**
     * Callback function type for config request events
     *
     * Called when a CONFIG.request is received. Return error code to indicate success or failure.
     *
     * @param[in] cell_id Cell identifier
     * @param[in] body_hdr Message body header
     * @param[in] body_len Length of message body
     * @return Error code indicating validation result
     */
    using OnConfigRequestCallback = std::function<scf_fapi_error_codes_t(
            uint16_t cell_id, const scf_fapi_body_header_t &body_hdr, uint32_t body_len)>;

    /**
     * Callback function type for start request events
     *
     * Called when a START.request is received. Return error code to indicate success or failure.
     *
     * @param[in] cell_id Cell identifier
     * @param[in] body_hdr Message body header
     * @param[in] body_len Length of message body
     * @return Error code indicating validation result
     */
    using OnStartRequestCallback = std::function<scf_fapi_error_codes_t(
            uint16_t cell_id, const scf_fapi_body_header_t &body_hdr, uint32_t body_len)>;

    /**
     * Callback function type for stop request events
     *
     * Called when a STOP.request is received. Return error code to indicate success or failure.
     *
     * @param[in] cell_id Cell identifier
     * @param[in] body_hdr Message body header
     * @param[in] body_len Length of message body
     * @return Error code indicating validation result
     */
    using OnStopRequestCallback = std::function<scf_fapi_error_codes_t(
            uint16_t cell_id, const scf_fapi_body_header_t &body_hdr, uint32_t body_len)>;

    /**
     * Callback function type for capturing all messages
     *
     * Called for every message before routing to specific handlers.
     * Provides raw message data for recording/replay purposes without NVIPC dependencies.
     *
     * @param[in] msg Message data including buffers and metadata
     */
    using OnMessageCallback = std::function<void(const FapiMessageData &msg)>;

    /**
     * Set callback for UL TTI request events
     *
     * @param[in] callback Callback function to invoke on UL TTI request
     */
    void set_on_ul_tti_request(OnUlTtiRequestCallback callback);

    /**
     * Set callback for DL TTI request events
     *
     * @param[in] callback Callback function to invoke on DL TTI request
     */
    void set_on_dl_tti_request(OnDlTtiRequestCallback callback);

    /**
     * Set callback for slot response events
     *
     * @param[in] callback Callback function to invoke on slot response
     */
    void set_on_slot_response(OnSlotResponseCallback callback);

    /**
     * Set callback for config request events
     *
     * @param[in] callback Callback function to invoke on CONFIG.request
     */
    void set_on_config_request(OnConfigRequestCallback callback);

    /**
     * Set callback for start request events
     *
     * @param[in] callback Callback function to invoke on START.request
     */
    void set_on_start_request(OnStartRequestCallback callback);

    /**
     * Set callback for stop request events
     *
     * @param[in] callback Callback function to invoke on STOP.request
     */
    void set_on_stop_request(OnStopRequestCallback callback);

    /**
     * Set callback for capturing all messages
     *
     * This callback is invoked before message-specific callbacks, allowing
     * capture of all messages for recording/replay purposes.
     *
     * @param[in] callback Callback function to invoke for every message
     */
    void set_on_message(OnMessageCallback callback);

private:
    /**
     * Custom deleter for NVIPC resources
     *
     * Cleans up NVIPC connection and associated /dev/shm files
     */
    struct NvIpcDeleter {
        std::string prefix; //!< NVIPC prefix for /dev/shm cleanup

        void operator()(nv_ipc_t *ipc) const noexcept;
    };

    /**
     * Process CONFIG.request message for a cell
     */
    [[nodiscard]] scf_fapi_error_codes_t process_config_request(
            uint16_t cell_id, scf_fapi_config_request_msg_t &config_msg, uint32_t body_len);

    /**
     * Process START.request for a cell
     */
    [[nodiscard]] scf_fapi_error_codes_t process_start_request(
            uint16_t cell_id, const scf_fapi_body_header_t &body_hdr, uint32_t body_len);

    /**
     * Process STOP.request for a cell
     */
    [[nodiscard]] scf_fapi_error_codes_t process_stop_request(
            uint16_t cell_id, const scf_fapi_body_header_t &body_hdr, uint32_t body_len);

    /**
     * Process SLOT_RESPONSE from MAC
     */
    void process_slot_response(
            uint16_t cell_id, const scf_fapi_body_header_t &body_hdr, uint32_t body_len) const;

    /**
     * Invoke message capture callback
     *
     * Constructs FapiMessageData from message components and invokes on_message_ callback.
     */
    void invoke_message_callback(
            uint16_t cell_id,
            uint16_t type_id,
            const scf_fapi_body_header_t &body_hdr,
            uint32_t body_len,
            const nv_ipc_msg_t &msg) const;

    /**
     * Handle CONFIG, START, and STOP request messages
     *
     * Routes and processes CONFIG.request, START.request, and STOP.request messages,
     * sending appropriate responses or error indications.
     *
     * @return Error code indicating processing result
     */
    [[nodiscard]] scf_fapi_error_codes_t handle_config_start_stop_request(
            uint16_t cell_id,
            uint16_t type_id,
            scf_fapi_body_header_t &body_hdr,
            uint32_t body_len);

    /**
     * Send CONFIG.response message
     *
     * @param[in] cell_id Cell identifier
     * @param[in] error_code Error code to include in response
     * @return true if message sent successfully, false otherwise
     */
    [[nodiscard]] bool send_config_response(uint16_t cell_id, scf_fapi_error_codes_t error_code);

    /**
     * Extract NVIPC prefix from YAML config
     *
     * @param[in] yaml_content YAML configuration content string
     * @return NVIPC prefix string, or empty string if extraction fails
     */
    [[nodiscard]] static std::string extract_nvipc_prefix(const std::string &yaml_content);

    /**
     * Send message with proper memory synchronization
     *
     * Ensures all buffer writes are visible before sending via memory fence.
     * Automatically releases buffer and logs error on failure.
     * Message type is derived from msg.msg_id for logging.
     *
     * @param[in,out] msg Message to send
     * @return 0 on success, -1 on failure
     */
    [[nodiscard]] int send_message(nv_ipc_msg_t &msg);

    InitParams params_;
    std::string nvipc_prefix_; //!< NVIPC prefix for /dev/shm cleanup
    std::unique_ptr<nv_ipc_t, NvIpcDeleter> ipc_;
    std::vector<CellConfig> cells_; //!< Per-cell configuration and state (indexed by cell_id)
    // NOLINTBEGIN(readability-redundant-member-init) - {} is required for std::atomic zero-init
    std::atomic<std::uint32_t>
            num_cells_configured_{}; //!< Number of cells in configured or later state
    std::atomic<std::uint32_t> num_cells_running_{};   //!< Number of cells in running state
    std::atomic<std::uint32_t> current_slot_packed_{}; //!< Packed SFN/slot (sfn in upper 16, slot
                                                       //!< in lower 16) for lock-free access
    // NOLINTEND(readability-redundant-member-init)

    OnUlTtiRequestCallback on_ul_tti_request_;  //!< Callback for UL TTI request events
    OnDlTtiRequestCallback on_dl_tti_request_;  //!< Callback for DL TTI request events
    OnSlotResponseCallback on_slot_response_;   //!< Callback for slot response events
    OnConfigRequestCallback on_config_request_; //!< Callback for CONFIG.request events
    OnStartRequestCallback on_start_request_;   //!< Callback for START.request events
    OnStopRequestCallback on_stop_request_;     //!< Callback for STOP.request events
    OnMessageCallback on_message_;              //!< Callback for capturing all messages
};

} // namespace ran::fapi

#endif // RAN_FAPI_STATE_HPP
