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
 * @file phy_ran_app_utils.hpp
 * @brief Utility functions for PHY RAN App
 *
 * Provides helper functions for application setup and configuration.
 */

#ifndef RAN_PHY_RAN_APP_UTILS_HPP
#define RAN_PHY_RAN_APP_UTILS_HPP

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include <NamedType/named_type_impl.hpp>
#include <tl/expected.hpp>

#include "fronthaul/fronthaul.hpp"
#include "log/rt_log_macros.hpp"
#include "message_adapter/phy_stats.hpp"
#include "task/task_utils.hpp"

namespace ran::phy_ran_app {

/// Default core for FAPI RX thread (polls nvIPC)
inline constexpr std::uint32_t DEFAULT_RX_CORE = 6;
/// Default core for slot indication trigger thread
inline constexpr std::uint32_t DEFAULT_SLOT_INDICATION_CORE = 7;
/// Default core for C-Plane processing task
inline constexpr std::uint32_t DEFAULT_CPLANE_CORE = 8;
/// Default core for U-Plane processing task
inline constexpr std::uint32_t DEFAULT_UPLANE_CORE = 9;
/// Default core for PUSCH RX processing task
inline constexpr std::uint32_t DEFAULT_PUSCH_CORE = 10;
/// Default slots to process ahead for C-Plane transmission (PHY timing chain)
inline constexpr std::uint32_t DEFAULT_SLOT_AHEAD = 1;
/// Default expected number of cells (limited by TensorRT engine)
inline constexpr std::uint32_t DEFAULT_EXPECTED_CELLS = 1;

/// Strong type for validation flag to avoid ambiguous boolean parameters
using PrintAndValidate = fluent::NamedType<bool, struct PrintAndValidateTag>;

/**
 * @brief Print runtime statistics and optionally validate (fail on errors)
 *
 * Always prints per-cell and total CRC failure counts using RT_LOGC macros.
 * When validation is enabled, returns false if any CRC failures detected.
 *
 * @param[in] stats The runtime statistics to print and validate
 * @param[in] fail_on_errors Whether to fail (return false) on detection of errors
 * @return true if validation passes or is disabled, false if validation enabled and errors detected
 */
[[nodiscard]] bool print_and_validate_stats(
        const ran::message_adapter::PhyStats &stats, const PrintAndValidate fail_on_errors);

/**
 * @brief Command-line arguments for phy_ran_app (based on fronthaul_app pattern + FAPI extensions)
 */
struct AppArguments final {
    std::string nic_pcie_addr;              //!< DU NIC PCIe address
    std::string config_file_path;           //!< Path to YAML configuration file
    std::uint32_t rx_core{DEFAULT_RX_CORE}; //!< FAPI RX thread core (polls nvIPC)
    std::uint32_t slot_indication_core{
            DEFAULT_SLOT_INDICATION_CORE};          //!< Slot indication trigger core
    std::uint32_t cplane_core{DEFAULT_CPLANE_CORE}; //!< C-Plane processing core
    std::uint32_t uplane_core{DEFAULT_UPLANE_CORE}; //!< U-Plane processing core
    std::uint32_t pusch_core{DEFAULT_PUSCH_CORE};   //!< PUSCH RX processing core
    std::optional<std::size_t> num_slots;           //!< Number of slots (unlimited if nullopt)
    std::uint32_t slot_ahead{DEFAULT_SLOT_AHEAD};   //!< Slots to process ahead
    std::uint32_t gpu_device_id{0};                 //!< GPU device ID
    std::uint32_t expected_cells{DEFAULT_EXPECTED_CELLS}; //!< Expected cells (FAPI-specific)
    bool validate{false}; //!< Validate runtime statistics and fail on errors
};

/**
 * @brief Setup logging for the application
 *
 * Configures quill logging and registers all component categories.
 * Adapted from fronthaul_app_utils.cpp::setup_logging().
 *
 */
void setup_logging();

/**
 * @brief Parse command-line arguments
 *
 * Uses CLI11 to parse arguments with validation.
 * Adapted from fronthaul_app_utils.cpp::parse_arguments().
 *
 * @param[in] argc Argument count
 * @param[in] argv Argument vector
 * @return Parsed arguments or error message
 */
tl::expected<AppArguments, std::string> parse_arguments(int argc, char **argv);

/**
 * @brief Create NVIPC configuration YAML string
 *
 * Generates NVIPC configuration for PHY side (primary).
 * Adapted from fapi_sample_utils.cpp::create_nvipc_config().
 *
 * @return NVIPC configuration as YAML string
 */
[[nodiscard]] std::string create_nvipc_config();

} // namespace ran::phy_ran_app

namespace ran::phy_ran_app {

/**
 * @brief Timing parameters result
 */
struct TimingResult final {
    std::uint64_t start_time_ns{};         //!< Start time in nanoseconds
    std::chrono::nanoseconds t0{};         //!< Time for SFN 0, subframe 0, slot 0
    std::chrono::nanoseconds tai_offset{}; //!< TAI offset
};

/**
 * @brief Calculate timing parameters for C-Plane transmission and trigger start
 *
 * Computes GPS-based timing for slot indication trigger and C-Plane transmission.
 * Duplicated from fronthaul_app_utils.cpp::calculate_timing_parameters().
 *
 * @param[in] start_params Start time parameters (current time, period, GPS alpha/beta)
 * @param[in] slot_ahead Number of slots to process ahead for C-Plane
 * @param[in] slot_period_ns Slot period in nanoseconds
 * @return Timing parameters including start time, t0, and TAI offset
 */
[[nodiscard]] TimingResult calculate_timing_parameters(
        const framework::task::StartTimeParams &start_params,
        std::uint32_t slot_ahead,
        std::uint64_t slot_period_ns);

/**
 * @brief Create Fronthaul configuration from YAML file
 *
 * Parses Fronthaul YAML configuration and creates FronthaulConfig structure.
 * Adapted from fronthaul_app_utils.cpp::create_fronthaul_config_from_yaml().
 *
 * @param[in] yaml_config_path Path to Fronthaul YAML configuration file
 * @param[in] nic_pcie_addr NIC PCIe address for DPDK
 * @param[in] gpu_device_id GPU device ID
 * @param[in] dpdk_core DPDK lcore for C-Plane processing
 * @param[in] slot_ahead Number of slots ahead for C-Plane transmission
 * @return FronthaulConfig if successful, error message on failure
 */
tl::expected<ran::fronthaul::FronthaulConfig, std::string> create_fronthaul_config(
        const std::string &yaml_config_path,
        const std::string &nic_pcie_addr,
        std::uint32_t gpu_device_id,
        std::uint32_t dpdk_core,
        std::uint32_t slot_ahead);

} // namespace ran::phy_ran_app

/// @cond HIDE_FROM_DOXYGEN
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
        ran::phy_ran_app::AppArguments,
        "nic={}, config={}, rx_core={}, slot_indication_core={}, cplane_core={}, uplane_core={}, "
        "pusch_core={}, num_slots={}, slot_ahead={}, gpu_device={}, expected_cells={}, validate={}",
        obj.nic_pcie_addr,
        obj.config_file_path,
        obj.rx_core,
        obj.slot_indication_core,
        obj.cplane_core,
        obj.uplane_core,
        obj.pusch_core,
        obj.num_slots.has_value() ? std::to_string(obj.num_slots.value()) : "unlimited",
        obj.slot_ahead,
        obj.gpu_device_id,
        obj.expected_cells,
        obj.validate)
/// @endcond

#endif // RAN_PHY_RAN_APP_UTILS_HPP
