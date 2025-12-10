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
 * @file fronthaul_app_utils.hpp
 * @brief Utility functions for fronthaul application
 */

#ifndef RAN_FRONTHAUL_FRONTHAUL_APP_UTILS_HPP
#define RAN_FRONTHAUL_FRONTHAUL_APP_UTILS_HPP

// clang-format off
#include <tl/expected.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "fapi/fapi_file_replay.hpp"
#include "log/rt_log_macros.hpp"
#include "fronthaul/fronthaul.hpp"
#include "net/dpdk_types.hpp"
#include "task/task_scheduler.hpp"
#include "task/task_utils.hpp"
#include "task/timed_trigger.hpp"
// clang-format on

namespace ran::fronthaul::samples {

/// Default worker CPU core
constexpr std::uint32_t DEFAULT_WORKER_CORE = 8;
/// Default trigger CPU core
constexpr std::uint32_t DEFAULT_TRIGGER_CORE = 7;
/// Default slot ahead value
constexpr std::uint32_t DEFAULT_SLOT_AHEAD = 1;

/**
 * Application command-line arguments
 */
struct AppArguments final {
    std::string nic_pcie_addr;                        //!< DU NIC PCIe address
    std::string config_file_path;                     //!< Path to ru_emulator_config.yaml
    std::string fapi_file_path;                       //!< FAPI capture file path
    std::uint32_t worker_core{DEFAULT_WORKER_CORE};   //!< Worker CPU core
    std::uint32_t trigger_core{DEFAULT_TRIGGER_CORE}; //!< Trigger CPU core
    std::optional<std::size_t> num_slots;             //!< Number of slots (unlimited if nullopt)
    std::uint32_t slot_ahead{DEFAULT_SLOT_AHEAD};     //!< Slots to process ahead
    std::uint32_t gpu_device_id{0};                   //!< GPU device ID for U-Plane processing
    bool validate_uplane_prbs{false};                 //!< Validate U-Plane PRB counts
};

/**
 * Setup logging for all components
 */
void setup_logging();

/**
 * Get FAPI capture file path
 *
 * Uses CLI argument if provided, otherwise searches executable directory for .fapi file.
 *
 * @param[in] cli_path Path from CLI argument (may be empty)
 * @param[in] argv0 First argument from main (executable path)
 * @return Path to FAPI capture file
 * @throw std::runtime_error if no file found
 */
[[nodiscard]] std::string get_fapi_file_path(const std::string &cli_path, const char *argv0);

/**
 * Parse command line arguments
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Argument strings
 * @return Parsed arguments on success, empty string if --help or --version shown, error message on
 * failure
 */
[[nodiscard]] tl::expected<AppArguments, std::string> parse_arguments(int argc, const char **argv);

/**
 * Convert MAC address strings to MacAddress objects
 *
 * @param[in] mac_strings Vector of MAC address strings
 * @return Vector of MacAddress objects
 * @throw std::runtime_error if any MAC address is invalid
 */
[[nodiscard]] std::vector<framework::net::MacAddress>
convert_mac_addresses(const std::vector<std::string> &mac_strings);

/**
 * Create fronthaul configuration from YAML config file
 *
 * Parses the YAML configuration, extracts cell configurations, timing parameters,
 * and network settings, then creates a complete FronthaulConfig ready for use.
 *
 * @param[in] yaml_config_path Path to ru_emulator_config.yaml file
 * @param[in] nic_pcie_addr DU-side NIC PCIe address
 * @param[in] gpu_device_id GPU device ID for U-Plane processing
 * @param[in] dpdk_core DPDK CPU core for network operations
 * @param[in] slot_ahead Number of slots to process ahead
 * @return FronthaulConfig if successful, std::nullopt on error
 */
[[nodiscard]] std::optional<ran::fronthaul::FronthaulConfig> create_fronthaul_config_from_yaml(
        const std::string &yaml_config_path,
        const std::string &nic_pcie_addr,
        std::uint32_t gpu_device_id,
        std::uint32_t dpdk_core,
        std::uint32_t slot_ahead);

/**
 * Timing parameters result
 */
struct TimingResult final {
    std::uint64_t start_time_ns{};         //!< Start time in nanoseconds
    std::chrono::nanoseconds t0{};         //!< Time for SFN 0, subframe 0, slot 0
    std::chrono::nanoseconds tai_offset{}; //!< TAI offset
};

/**
 * Calculate timing parameters for fronthaul operation
 *
 * @param[in] start_params Start time parameters
 * @param[in] slot_ahead Number of slots to process ahead
 * @param[in] slot_period_ns Slot period in nanoseconds
 * @return Timing parameters including start time, t0, and TAI offset
 */
[[nodiscard]] TimingResult calculate_timing_parameters(
        const framework::task::StartTimeParams &start_params,
        std::uint32_t slot_ahead,
        std::uint64_t slot_period_ns);

/**
 * Create process C-Plane function for fronthaul processing
 *
 * @warning LIFETIME SAFETY: The returned function captures fronthaul, fapi_replay,
 * and is_first_slot by reference. The caller MUST ensure that all captured objects
 * remain valid for the entire lifetime of the returned function and any copies of it.
 * Using the returned function after any object is destroyed results in undefined
 * behavior.
 *
 * @note This function uses reference capture for performance. Ensure the returned
 * function is used only within the scope where all captured references are valid.
 *
 * @param[in] fronthaul Fronthaul instance reference (must outlive returned function)
 * @param[in] fapi_replay FAPI file replay instance reference (must outlive returned function)
 * @param[in,out] is_first_slot Flag tracking first slot processing (must outlive returned function)
 * @param[in] t0 Time for SFN 0, subframe 0, slot 0
 * @param[in] tai_offset TAI offset
 * @return Function to process C-Plane for a slot (captures by reference)
 */
[[nodiscard]] std::function<void()> make_process_cplane_func(
        ran::fronthaul::Fronthaul &fronthaul,
        ran::fapi::FapiFileReplay &fapi_replay,
        bool &is_first_slot,
        std::chrono::nanoseconds t0,
        std::chrono::nanoseconds tai_offset);

/**
 * Create process U-Plane function for Order Kernel pipeline execution
 *
 * Creates a function that executes the Fronthaul U-Plane processing per slot.
 * The function first checks if the current slot has uplink data by querying
 * the FAPI replay. If no uplink data is present (DL-only slot), the function
 * returns early without processing. Otherwise, it calls Fronthaul::process_uplane()
 * which internally manages the Order Kernel pipeline execution.
 *
 * @warning LIFETIME SAFETY: The returned function captures fronthaul and fapi_replay
 * by reference. The caller MUST ensure that both objects remain valid for the entire
 * lifetime of the returned function and any copies of it. Using the returned
 * function after either object is destroyed results in undefined behavior.
 *
 * @param[in] fronthaul Fronthaul instance reference (must outlive returned function)
 * @param[in] fapi_replay FAPI file replay instance reference (must outlive returned function)
 * @return Function to process U-Plane per slot (captures by reference)
 */
[[nodiscard]] std::function<void()> make_process_uplane_func(
        ran::fronthaul::Fronthaul &fronthaul, ran::fapi::FapiFileReplay &fapi_replay);

/**
 * Print all statistics
 *
 * @param[in] fronthaul Fronthaul instance
 * @param[in] scheduler Task scheduler instance
 * @param[in] trigger Timed trigger instance
 * @param[in] num_slots Number of slots processed (used to determine if detailed stats should print)
 */
void print_statistics(
        const ran::fronthaul::Fronthaul &fronthaul,
        framework::task::TaskScheduler &scheduler,
        framework::task::TimedTrigger &trigger,
        std::optional<std::size_t> num_slots);

/**
 * Validate kernel results from U-Plane processing
 *
 * Reads kernel results and validates that PRB counts match expected values.
 *
 * @param[in] fronthaul Fronthaul instance
 * @return true if validation passed, false otherwise
 */
[[nodiscard]] bool validate_kernel_results(const ran::fronthaul::Fronthaul &fronthaul);

} // namespace ran::fronthaul::samples

/**
 * Enable logging support for AppArguments struct
 *
 * Registers AppArguments with the logging framework to enable
 * direct logging of application configuration.
 *
 * Note: This must be in the global namespace so Quill's formatter can detect it.
 */
// cppcheck-suppress functionStatic
RT_LOGGABLE_DEFERRED_FORMAT(
        ran::fronthaul::samples::AppArguments,
        "nic={}, config={}, fapi={}, worker_core={}, trigger_core={}, num_slots={}, slot_ahead={}, "
        "gpu_device={}, validate_uplane_prbs={}",
        obj.nic_pcie_addr,
        obj.config_file_path,
        obj.fapi_file_path,
        obj.worker_core,
        obj.trigger_core,
        obj.num_slots.has_value() ? std::to_string(obj.num_slots.value()) : "unlimited",
        obj.slot_ahead,
        obj.gpu_device_id,
        obj.validate_uplane_prbs)

#endif // RAN_FRONTHAUL_FRONTHAUL_APP_UTILS_HPP
