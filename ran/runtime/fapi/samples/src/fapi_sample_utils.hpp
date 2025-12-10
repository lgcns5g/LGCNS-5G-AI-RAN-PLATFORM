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
 * @file fapi_sample_utils.hpp
 * @brief Utility functions for FAPI sample application
 */

#ifndef RAN_FAPI_FAPI_SAMPLE_UTILS_HPP
#define RAN_FAPI_FAPI_SAMPLE_UTILS_HPP

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "fapi/fapi_state.hpp"
#include "log/rt_log_macros.hpp"

namespace fapi_sample {

/**
 * Application configuration
 */
struct AppConfig {
    static constexpr int DEFAULT_SLOT_INTERVAL_MS = 10; //!< Default slot interval in milliseconds

    std::chrono::microseconds slot_interval{
            std::chrono::milliseconds{DEFAULT_SLOT_INTERVAL_MS}}; //!< Slot timing interval
    bool validate_message_counts{};   //!< Enable message count validation
    std::string launch_pattern_file;  //!< Path to launch pattern YAML file
    std::string test_mac_config_file; //!< Path to test MAC config YAML file
    std::size_t expected_cells{};     //!< Number of cells to wait for before starting slot loop
    std::string capture_file_path;    //!< Path to capture file (empty = no capture)
    std::optional<std::size_t> test_slots; //!< Number of slots to send (nullopt = unlimited)
};

/**
 * Result of parsing command line arguments
 */
struct ParseResult {
    std::optional<AppConfig> config; //!< Config if parsing succeeded
    int exit_code{};                 //!< Exit code (0 for success, non-zero for error)

    /**
     * Check if parsing was successful
     *
     * @return true if config is available, false otherwise
     */
    [[nodiscard]] bool has_config() const { return config.has_value(); }
};

/**
 * Message counts for a cell (used for both expected and actual counts)
 */
struct MessageCounts {
    std::uint64_t ul_tti_requests{}; //!< Number of UL_TTI_REQUEST messages
    std::uint64_t dl_tti_requests{}; //!< Number of DL_TTI_REQUEST messages
    std::uint64_t slot_responses{};  //!< Number of SLOT.indication responses
};

/**
 * Per-cell message counters for tracking received messages
 */
struct MessageCounters {
    std::atomic<std::uint64_t> ul_tti_request_count; //!< UL_TTI_REQUEST count
    std::atomic<std::uint64_t> dl_tti_request_count; //!< DL_TTI_REQUEST count
    std::atomic<std::uint64_t> slot_response_count;  //!< SLOT_RESPONSE count

    explicit MessageCounters()
            : ul_tti_request_count(0), dl_tti_request_count(0), slot_response_count(0) {}
};

/**
 * Convert MessageCounters vector to MessageCounts vector for logging
 *
 * Snapshots atomic counter values into a copyable POD struct.
 *
 * @param[in] cell_counters Vector of MessageCounters with atomic values
 * @param[in] num_cells Number of cells to convert
 * @return Vector of MessageCounts with loaded atomic values
 */
[[nodiscard]] std::vector<MessageCounts>
snapshot_counters(const std::vector<MessageCounters> &cell_counters, std::size_t num_cells);

/**
 * Parse launch pattern YAML to get number of cells
 *
 * @param[in] launch_pattern_file Path to launch pattern YAML file
 * @return Number of cells in Cell_Configs array, or 0 on error
 */
[[nodiscard]] std::size_t parse_cell_count(const std::filesystem::path &launch_pattern_file);

/**
 * Parse test_mac config YAML to get test_slots
 *
 * @param[in] test_mac_config_file Path to test_mac config YAML file
 * @return Test slots value (nullopt = unlimited), or nullopt on error
 */
[[nodiscard]] std::optional<std::size_t>
parse_test_slots(const std::filesystem::path &test_mac_config_file);

/**
 * Parse launch pattern YAML to extract expected counts per cell
 *
 * @param[in] launch_pattern_file Path to launch pattern YAML file
 * @param[in] test_mac_config_file Path to test_mac config YAML file
 * @return Vector of expected message counts (one per cell), or std::nullopt if parsing fails
 */
[[nodiscard]] std::optional<std::vector<MessageCounts>> parse_expected_counts(
        const std::filesystem::path &launch_pattern_file,
        const std::filesystem::path &test_mac_config_file);

/**
 * Validate message counts against expected values
 *
 * @param[in] expected Expected message counts
 * @param[in] actual Actual message counts
 * @return true if validation passes, false otherwise
 */
[[nodiscard]] bool
validate_message_counts(const MessageCounts &expected, const MessageCounters &actual);

/**
 * Perform validation if requested
 *
 * @param[in] config Application configuration
 * @param[in] cell_counters Vector of per-cell actual message counts
 * @return true if validation passes or is not requested, false if validation fails
 */
[[nodiscard]] bool
perform_validation(const AppConfig &config, const std::vector<MessageCounters> &cell_counters);

/**
 * Setup logging for FAPI sample
 */
void setup_logging();

/**
 * Setup message counting callbacks on FAPI state
 *
 * @param[in,out] fapi_state FAPI state machine to attach callbacks to
 * @param[in,out] cell_counters Vector of per-cell message counters to increment (indexed by cell ID
 * from callbacks)
 * @param[in] max_cells Maximum number of cells supported
 */
void setup_message_callbacks(
        ran::fapi::FapiState &fapi_state,
        std::vector<MessageCounters> &cell_counters,
        std::size_t max_cells);

/**
 * Parse command line arguments
 *
 * @param[in] argc Number of command line arguments
 * @param[in] argv Array of command line argument strings
 * @return Parse result with config (if successful) and exit code
 */
[[nodiscard]] ParseResult parse_arguments(int argc, const char **argv);

/**
 * Process incoming messages from MAC (runs on worker thread)
 *
 * @param[in,out] fapi_state FAPI state machine
 * @param[in,out] running Flag to signal shutdown
 */
void process_rx_messages(ran::fapi::FapiState &fapi_state, std::atomic<bool> &running);

/**
 * Generate NVIPC YAML configuration for primary (PHY side)
 *
 * @return NVIPC configuration as YAML string
 */
[[nodiscard]] std::string create_nvipc_config();

/**
 * Create slot trigger function for timed slot indications
 *
 * @param[in,out] fapi_state FAPI state machine
 * @param[in,out] running Flag to signal shutdown
 * @param[in,out] slots_sent Counter for slots sent
 * @param[in] test_slots Number of slots to send (nullopt = unlimited)
 * @return Function to be called on each trigger
 */
[[nodiscard]] std::function<void()> make_slot_trigger_func(
        ran::fapi::FapiState &fapi_state,
        std::atomic<bool> &running,
        std::atomic<std::uint64_t> &slots_sent,
        std::optional<std::size_t> test_slots);

} // namespace fapi_sample

/// @cond HIDE_FROM_DOXYGEN
/**
 * Enable logging support for MessageCounts struct
 *
 * Registers MessageCounts with the logging framework to enable
 * direct logging of message count objects.
 *
 * Note: This must be in the global namespace so Quill's formatter can detect it.
 */
// cppcheck-suppress functionStatic
RT_LOGGABLE_DEFERRED_FORMAT(
        fapi_sample::MessageCounts,
        "UL_TTI={}, DL_TTI={}, SLOT_RESP={}",
        obj.ul_tti_requests,
        obj.dl_tti_requests,
        obj.slot_responses)

/**
 * Enable logging support for AppConfig struct
 *
 * Registers AppConfig with the logging framework to enable
 * direct logging of config objects.
 *
 * Note: This must be in the global namespace so Quill's formatter can detect it.
 */
// cppcheck-suppress functionStatic
RT_LOGGABLE_DEFERRED_FORMAT(
        fapi_sample::AppConfig,
        "interval={}us, validation={}, cells={}, test_slots={}, capture={}",
        obj.slot_interval.count(),
        obj.validate_message_counts ? "enabled" : "disabled",
        obj.expected_cells,
        obj.test_slots.has_value() ? std::to_string(obj.test_slots.value()) : "unlimited",
        obj.capture_file_path.empty() ? "disabled" : obj.capture_file_path)
/// @endcond

#endif // RAN_FAPI_FAPI_SAMPLE_UTILS_HPP
