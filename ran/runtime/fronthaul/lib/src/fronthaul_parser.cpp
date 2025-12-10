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
 * @file fronthaul_parser.cpp
 * @brief Implementation of fronthaul configuration parser
 */

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <quill/LogMacros.h>
#include <tl/expected.hpp>
#include <unistd.h>
#include <yaml.hpp>

#include <gsl-lite/gsl-lite.hpp>

#include "fronthaul/fronthaul_log.hpp"
#include "fronthaul/fronthaul_parser.hpp"
#include "log/rt_log_macros.hpp"

namespace ran::fronthaul {

namespace {

/**
 * Convert microseconds to nanoseconds
 *
 * @param[in] microseconds Time in microseconds
 * @return Time in nanoseconds
 */
[[nodiscard]] constexpr std::uint64_t
microseconds_to_nanoseconds(const std::uint64_t microseconds) {
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
    return microseconds * 1000ULL;
}

/**
 * Create unique temporary YAML file path
 *
 * @param[in] prefix File name prefix
 * @return Unique temporary file path using PID and timestamp
 */
[[nodiscard]] std::filesystem::path create_unique_temp_yaml_file(const std::string_view prefix) {
    const std::filesystem::path temp_dir = std::filesystem::temp_directory_path();
    const auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    return temp_dir / std::format("{}_{}_{}_.yaml", prefix, ::getpid(), now);
}

/**
 * Parse cell configurations from YAML node
 *
 * @param[in] cell_configs_node YAML node containing cell_configs array
 * @return Vector of cell configurations
 */
[[nodiscard]] std::vector<CellConfig> parse_cell_configs(yaml::node &cell_configs_node) {
    std::vector<CellConfig> cells{};
    const std::size_t num_cells = cell_configs_node.length();

    if (num_cells == 0) {
        RT_LOGC_ERROR(FronthaulLog::FronthaulParser, "No cells found in cell_configs");
        return cells;
    }

    cells.reserve(num_cells);

    for (std::size_t i{}; i < num_cells; ++i) {
        const yaml::node cell_node = cell_configs_node[i];

        CellConfig cell{};

        // Parse MAC address, VLAN, and PCP (will throw if missing)
        cell.mac_address = cell_node["eth"].as<std::string>();
        const std::uint16_t vlan = cell_node["vlan"].as<std::uint16_t>();
        const std::uint16_t pcp = cell_node["pcp"].as<std::uint16_t>();

        // Validate PCP value (3 bits: 0-7)
        static constexpr std::uint16_t MAX_PCP_VALUE = 7;
        if (pcp > MAX_PCP_VALUE) {
            RT_LOGC_ERROR(
                    FronthaulLog::FronthaulParser,
                    "Invalid PCP value {} for cell {}: must be between 0 and {} (3 bits)",
                    pcp,
                    i,
                    MAX_PCP_VALUE);
            throw std::invalid_argument(
                    std::format("PCP value {} exceeds maximum of {}", pcp, MAX_PCP_VALUE));
        }

        // Validate VLAN value (12 bits: 0-4095)
        static constexpr std::uint16_t MAX_VLAN_VALUE = 4095;
        if (vlan > MAX_VLAN_VALUE) {
            RT_LOGC_ERROR(
                    FronthaulLog::FronthaulParser,
                    "Invalid VLAN value {} for cell {}: must be between 0 and {} (12 bits)",
                    vlan,
                    i,
                    MAX_VLAN_VALUE);
            throw std::invalid_argument(
                    std::format("VLAN value {} exceeds maximum of {}", vlan, MAX_VLAN_VALUE));
        }

        // Combine PCP and VLAN into TCI
        static constexpr std::uint16_t PCP_SHIFT = 13;
        cell.vlan_tci = static_cast<std::uint16_t>(pcp << PCP_SHIFT) | vlan;

        // Parse eAxC_UL array (will throw if missing)
        yaml::node eaxc_ul_node = cell_node["eAxC_UL"];
        const std::size_t num_eaxc = eaxc_ul_node.length();
        cell.eaxc_ul.reserve(num_eaxc);
        for (std::size_t j{}; j < num_eaxc; ++j) {
            cell.eaxc_ul.push_back(eaxc_ul_node[j].as<std::uint16_t>());
        }

        cells.push_back(cell);
    }

    return cells;
}

/**
 * Parse timing configuration from YAML node
 *
 * Timing values in the YAML file are in microseconds and need to be converted to nanoseconds.
 *
 * @param[in] timing_node YAML node containing oran_timing_info
 * @return O-RAN timing configuration
 */
[[nodiscard]] OranTimingConfig parse_timing_config(yaml::node &timing_node) {
    OranTimingConfig timing{};

    // Parse timing values (will throw if missing) - values are in microseconds
    const std::uint64_t ul_timing_delay_us =
            timing_node["ul_c_plane_timing_delay"].as<std::uint64_t>();
    const std::uint64_t window_size_us = timing_node["ul_c_plane_window_size"].as<std::uint64_t>();

    // Convert to nanoseconds
    timing.t1a_max_ns = microseconds_to_nanoseconds(ul_timing_delay_us);
    const std::uint64_t window_size_ns = microseconds_to_nanoseconds(window_size_us);

    // Calculate T1a min: t1a_max - window_size
    if (timing.t1a_max_ns >= window_size_ns) {
        timing.t1a_min_ns = timing.t1a_max_ns - window_size_ns;
    } else {
        RT_LOGC_WARN(
                FronthaulLog::FronthaulParser,
                "Window size ({} µs = {} ns) exceeds t1a_max ({} µs = {} ns), setting t1a_min to 0",
                window_size_us,
                window_size_ns,
                ul_timing_delay_us,
                timing.t1a_max_ns);
        timing.t1a_min_ns = 0;
    }

    return timing;
}

/**
 * Parse YAML document and extract fronthaul configuration
 *
 * @param[in] doc YAML document
 * @return Parsed configuration on success, error message on failure
 */
[[nodiscard]] tl::expected<FronthaulYamlConfig, std::string>
parse_yaml_document(yaml::document &doc) {
    try {
        const yaml::node root = doc.root();
        const yaml::node ru_emulator = root["ru_emulator"];

        FronthaulYamlConfig config{};

        // Parse cell configurations (will throw if missing)
        yaml::node cell_configs_node = ru_emulator["cell_configs"];
        config.cells = parse_cell_configs(cell_configs_node);

        if (config.cells.empty()) {
            const std::string error_msg = "No cells found in cell_configs";
            RT_LOGC_ERROR(FronthaulLog::FronthaulParser, "{}", error_msg);
            return tl::unexpected(error_msg);
        }

        // Parse timing configuration (will throw if missing)
        yaml::node timing_node = ru_emulator["oran_timing_info"];
        config.timing = parse_timing_config(timing_node);

        // Parse MTU size (will throw if missing)
        config.mtu_size = ru_emulator["aerial_fh_mtu"].as<std::uint32_t>();

        RT_LOGC_DEBUG(FronthaulLog::FronthaulParser, "Parsed YAML config: {}", config);

        return config;

    } catch (const std::exception &e) {
        const std::string error_msg =
                std::format("Exception while parsing YAML document: {}", e.what());
        RT_LOGC_ERROR(FronthaulLog::FronthaulParser, "{}", error_msg);
        return tl::unexpected(error_msg);
    }
}

} // namespace

tl::expected<FronthaulYamlConfig, std::string>
parse_fronthaul_config(const std::filesystem::path &config_file_path) {
    if (!std::filesystem::exists(config_file_path)) {
        const std::string error_msg =
                std::format("Config file not found: {}", config_file_path.string());
        RT_LOGC_ERROR(FronthaulLog::FronthaulParser, "{}", error_msg);
        return tl::unexpected(error_msg);
    }

    try {
        yaml::file_parser parser(config_file_path.c_str());
        yaml::document doc = parser.next_document();
        return parse_yaml_document(doc);
    } catch (const std::exception &e) {
        const std::string error_msg = std::format(
                "Failed to parse config file {}: {}", config_file_path.string(), e.what());
        RT_LOGC_ERROR(FronthaulLog::FronthaulParser, "{}", error_msg);
        return tl::unexpected(error_msg);
    }
}

tl::expected<FronthaulYamlConfig, std::string>
parse_fronthaul_config_from_string(const std::string_view yaml_content) {
    if (yaml_content.empty()) {
        const std::string error_msg = "YAML content string is empty";
        RT_LOGC_ERROR(FronthaulLog::FronthaulParser, "{}", error_msg);
        return tl::unexpected(error_msg);
    }

    // YAML library requires file parsing, so write to temp file
    // Use unique file name to avoid collisions between concurrent processes
    const std::filesystem::path temp_file = create_unique_temp_yaml_file("fronthaul_config");

    // Setup cleanup guard to ensure temp file is removed
    const auto cleanup = gsl_lite::finally([&temp_file] { std::filesystem::remove(temp_file); });

    try {
        // Write string to temp file
        std::ofstream ofs(temp_file);
        if (!ofs) {
            const std::string error_msg = std::format(
                    "Failed to create temp file for YAML string: {}", temp_file.string());
            RT_LOGC_ERROR(FronthaulLog::FronthaulParser, "{}", error_msg);
            return tl::unexpected(error_msg);
        }
        ofs << yaml_content;
        ofs.close();

        // Parse from temp file
        yaml::file_parser parser(temp_file.c_str());
        yaml::document doc = parser.next_document();
        return parse_yaml_document(doc);

    } catch (const std::exception &e) {
        const std::string error_msg = std::format("Failed to parse YAML string: {}", e.what());
        RT_LOGC_ERROR(FronthaulLog::FronthaulParser, "{}", error_msg);
        return tl::unexpected(error_msg);
    }
}

} // namespace ran::fronthaul
