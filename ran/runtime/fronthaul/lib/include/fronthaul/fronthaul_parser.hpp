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
 * @file fronthaul_parser.hpp
 * @brief Parser for fronthaul configuration from RU emulator YAML files
 */

#ifndef RAN_FRONTHAUL_FRONTHAUL_PARSER_HPP
#define RAN_FRONTHAUL_FRONTHAUL_PARSER_HPP

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include <tl/expected.hpp>

#include "log/rt_log_macros.hpp"

namespace ran::fronthaul {

/**
 * Per-cell configuration from YAML
 */
struct CellConfig final {
    std::string mac_address;            //!< RU MAC address (eth field)
    std::uint16_t vlan_tci{};           //!< VLAN TCI (includes PCP in upper bits)
    std::vector<std::uint16_t> eaxc_ul; //!< UL eAxC IDs for antenna ports
};

/**
 * O-RAN timing parameters from YAML
 *
 * Note: Timing values in the YAML file are specified in microseconds (µs)
 * and are automatically converted to nanoseconds during parsing.
 */
struct OranTimingConfig final {
    std::uint64_t t1a_max_ns{}; //!< T1a max CP UL in nanoseconds
    std::uint64_t t1a_min_ns{}; //!< T1a min CP UL in nanoseconds
};

/**
 * Fronthaul configuration parsed from RU emulator YAML
 */
struct FronthaulYamlConfig final {
    std::vector<CellConfig> cells; //!< Per-cell configuration
    OranTimingConfig timing{};     //!< O-RAN timing parameters
    std::uint32_t mtu_size{};      //!< MTU size for network config
};

/**
 * Parse fronthaul configuration from RU emulator YAML file
 *
 * Extracts:
 * - Cell configurations (MAC addresses, VLANs)
 * - O-RAN timing parameters (t1a_max, t1a_min in µs, converted to ns)
 * - MTU size
 *
 * @param[in] config_file_path Path to ru_emulator_config.yaml
 * @return Parsed configuration on success, error message on failure
 */
[[nodiscard]] tl::expected<FronthaulYamlConfig, std::string>
parse_fronthaul_config(const std::filesystem::path &config_file_path);

/**
 * Parse fronthaul configuration from YAML string
 *
 * @param[in] yaml_content YAML content as string
 * @return Parsed configuration on success, error message on failure
 */
[[nodiscard]] tl::expected<FronthaulYamlConfig, std::string>
parse_fronthaul_config_from_string(std::string_view yaml_content);

} // namespace ran::fronthaul

/// @cond HIDE_FROM_DOXYGEN
/**
 * Enable logging support for CellConfig struct
 *
 * Registers CellConfig with the logging framework to enable direct logging
 * of cell configuration objects.
 *
 * Note: This must be in the global namespace so Quill's formatter can detect it.
 */
// cppcheck-suppress functionStatic
RT_LOGGABLE_DEFERRED_FORMAT(
        ran::fronthaul::CellConfig,
        "{{mac={}, vlan_tci=0x{:04X}, eaxc_ul={}}}",
        obj.mac_address,
        obj.vlan_tci,
        obj.eaxc_ul)

/**
 * Enable logging support for FronthaulYamlConfig struct
 *
 * Registers FronthaulYamlConfig with the logging framework to enable
 * direct logging of configuration objects.
 *
 * Note: This must be in the global namespace so Quill's formatter can detect it.
 */
// cppcheck-suppress functionStatic
RT_LOGGABLE_DEFERRED_FORMAT(
        ran::fronthaul::FronthaulYamlConfig,
        "cells={}, MTU={}, t1a_max={}ns, t1a_min={}ns",
        obj.cells,
        obj.mtu_size,
        obj.timing.t1a_max_ns,
        obj.timing.t1a_min_ns)
/// @endcond

#endif // RAN_FRONTHAUL_FRONTHAUL_PARSER_HPP
