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
 * @file fronthaul_parser_tests.cpp
 * @brief Unit tests for fronthaul YAML configuration parser
 */

#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <tl/expected.hpp>
#include <unistd.h>

#include <gtest/gtest.h>

#include "fronthaul/fronthaul_parser.hpp"
#include "log/rt_log_macros.hpp"

namespace ran::fronthaul::tests {

/**
 * RAII helper for temporary YAML files
 */
class TempYamlFile final {
public:
    explicit TempYamlFile(const std::string_view content) : path_(create_temp_file_path()) {
        std::ofstream file(path_);
        if (!file) {
            throw std::runtime_error("Failed to create temp file: " + path_.string());
        }
        file << content;
    }

    ~TempYamlFile() {
        try {
            std::filesystem::remove(path_);
        } catch (const std::exception &e) {
            RT_LOG_ERROR("Failed to remove temp file: {}", e.what());
        }
    }

    TempYamlFile(const TempYamlFile &) = delete;
    TempYamlFile &operator=(const TempYamlFile &) = delete;
    TempYamlFile(TempYamlFile &&) = delete;
    TempYamlFile &operator=(TempYamlFile &&) = delete;

    [[nodiscard]] const std::filesystem::path &path() const { return path_; }

private:
    [[nodiscard]] static std::filesystem::path create_temp_file_path() {
        const std::filesystem::path temp_dir = std::filesystem::temp_directory_path();
        const auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        return temp_dir / std::format("fronthaul_parser_test_{}_{}_.yaml", ::getpid(), now);
    }

    std::filesystem::path path_;
};

/**
 * Test: Parse valid minimal single-cell configuration
 */
TEST(FronthaulConfigParser, ParseValidMinimalConfig) {
    const std::string yaml_content = R"(
ru_emulator:
  cell_configs:
    - eth: "aa:bb:cc:dd:ee:ff"
      vlan: 2
      pcp: 7
      eAxC_UL: [0,1,2,3]
  oran_timing_info:
    ul_c_plane_timing_delay: 336
    ul_c_plane_window_size: 51
  aerial_fh_mtu: 1514
)";

    const TempYamlFile temp_file(yaml_content);
    const auto config_opt = parse_fronthaul_config(temp_file.path());

    ASSERT_TRUE(config_opt.has_value());
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    const auto &config = config_opt.value();

    EXPECT_EQ(config.cells.size(), 1);
    EXPECT_EQ(config.cells[0].mac_address, "aa:bb:cc:dd:ee:ff");
    EXPECT_EQ(config.cells[0].vlan_tci, 0xE002); // (7 << 13) | 2
    EXPECT_EQ(config.mtu_size, 1514);
    EXPECT_GT(config.timing.t1a_max_ns, 0);
    EXPECT_GT(config.timing.t1a_min_ns, 0);
    EXPECT_LT(config.timing.t1a_min_ns, config.timing.t1a_max_ns);
}

/**
 * Test: Parse valid multi-cell configuration
 */
TEST(FronthaulConfigParser, ParseMultipleCells) {
    const std::string yaml_content = R"(
ru_emulator:
  cell_configs:
    - eth: "aa:bb:cc:dd:ee:ff"
      vlan: 2
      pcp: 7
      eAxC_UL: [0,1,2,3]
    - eth: "11:22:33:44:55:66"
      vlan: 3
      pcp: 7
      eAxC_UL: [0,1,2,3]
  oran_timing_info:
    ul_c_plane_timing_delay: 336
    ul_c_plane_window_size: 51
  aerial_fh_mtu: 1514
)";

    const TempYamlFile temp_file(yaml_content);
    const auto config_opt = parse_fronthaul_config(temp_file.path());

    ASSERT_TRUE(config_opt.has_value());
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    const auto &config = config_opt.value();

    EXPECT_EQ(config.cells.size(), 2);

    // Cell 0
    EXPECT_EQ(config.cells[0].mac_address, "aa:bb:cc:dd:ee:ff");
    EXPECT_EQ(config.cells[0].vlan_tci, 0xE002); // (7 << 13) | 2

    // Cell 1
    EXPECT_EQ(config.cells[1].mac_address, "11:22:33:44:55:66");
    EXPECT_EQ(config.cells[1].vlan_tci, 0xE003); // (7 << 13) | 3
}

/**
 * Test: Parse timing configuration correctly
 */
TEST(FronthaulConfigParser, ParseTimingConfiguration) {
    const std::string yaml_content = R"(
ru_emulator:
  cell_configs:
    - eth: "aa:bb:cc:dd:ee:ff"
      vlan: 2
      pcp: 7
      eAxC_UL: [0,1,2,3]
  oran_timing_info:
    ul_c_plane_timing_delay: 336
    ul_c_plane_window_size: 51
  aerial_fh_mtu: 1514
)";

    const TempYamlFile temp_file(yaml_content);
    const auto config_opt = parse_fronthaul_config(temp_file.path());

    ASSERT_TRUE(config_opt.has_value());
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    const auto &config = config_opt.value();

    // Timing values in YAML are in microseconds:
    // ul_c_plane_timing_delay: 336 µs = 336,000 ns
    // ul_c_plane_window_size: 51 µs = 51,000 ns
    // T1a min = T1a max - window_size = 336,000 - 51,000 = 285,000 ns

    static constexpr std::uint64_t EXPECTED_T1A_MAX_NS = 336'000;    // 336 µs
    static constexpr std::uint64_t EXPECTED_WINDOW_SIZE_NS = 51'000; // 51 µs
    static constexpr std::uint64_t EXPECTED_T1A_MIN_NS = 285'000;    // 285 µs

    EXPECT_EQ(config.timing.t1a_max_ns, EXPECTED_T1A_MAX_NS);
    EXPECT_EQ(config.timing.t1a_min_ns, EXPECTED_T1A_MIN_NS);
    EXPECT_EQ(config.timing.t1a_max_ns - config.timing.t1a_min_ns, EXPECTED_WINDOW_SIZE_NS);
}

/**
 * Test: Parse MTU size correctly
 */
TEST(FronthaulConfigParser, ParseMtuSize) {
    const std::string yaml_content = R"(
ru_emulator:
  cell_configs:
    - eth: "aa:bb:cc:dd:ee:ff"
      vlan: 2
      pcp: 7
      eAxC_UL: [0,1,2,3]
  oran_timing_info:
    ul_c_plane_timing_delay: 336
    ul_c_plane_window_size: 51
  aerial_fh_mtu: 9000
)";

    const TempYamlFile temp_file(yaml_content);
    const auto config_opt = parse_fronthaul_config(temp_file.path());

    ASSERT_TRUE(config_opt.has_value());
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    const auto &config = config_opt.value();

    EXPECT_EQ(config.mtu_size, 9000);
}

/**
 * Test: Parse VLAN TCI with different PCP values
 */
TEST(FronthaulConfigParser, ParseVlanTciWithDifferentPcp) {
    const std::string yaml_content = R"(
ru_emulator:
  cell_configs:
    - eth: "aa:bb:cc:dd:ee:ff"
      vlan: 100
      pcp: 0
      eAxC_UL: [0,1,2,3]
    - eth: "11:22:33:44:55:66"
      vlan: 200
      pcp: 3
      eAxC_UL: [0,1,2,3]
    - eth: "77:88:99:aa:bb:cc"
      vlan: 300
      pcp: 7
      eAxC_UL: [0,1,2,3]
  oran_timing_info:
    ul_c_plane_timing_delay: 336
    ul_c_plane_window_size: 51
  aerial_fh_mtu: 1514
)";

    const TempYamlFile temp_file(yaml_content);
    const auto config_opt = parse_fronthaul_config(temp_file.path());

    ASSERT_TRUE(config_opt.has_value());
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    const auto &config = config_opt.value();

    EXPECT_EQ(config.cells.size(), 3);

    // Cell 0: PCP=0, VLAN=100 -> TCI = (0 << 13) | 100 = 100
    EXPECT_EQ(config.cells[0].vlan_tci, 100);

    // Cell 1: PCP=3, VLAN=200 -> TCI = (3 << 13) | 200 = 24776
    EXPECT_EQ(config.cells[1].vlan_tci, 24776);

    // Cell 2: PCP=7, VLAN=300 -> TCI = (7 << 13) | 300 = 57644
    EXPECT_EQ(config.cells[2].vlan_tci, 57644);
}

/**
 * Test: Missing ru_emulator node
 */
TEST(FronthaulConfigParser, MissingRuEmulatorNode) {
    const std::string yaml_content = R"(
some_other_config:
  value: 42
)";

    const TempYamlFile temp_file(yaml_content);
    const auto config = parse_fronthaul_config(temp_file.path());

    EXPECT_FALSE(config.has_value());
}

/**
 * Test: Missing cell_configs field
 */
TEST(FronthaulConfigParser, MissingCellConfigs) {
    const std::string yaml_content = R"(
ru_emulator:
  oran_timing_info:
    ul_c_plane_timing_delay: 336
    ul_c_plane_window_size: 51
  aerial_fh_mtu: 1514
)";

    const TempYamlFile temp_file(yaml_content);
    const auto config = parse_fronthaul_config(temp_file.path());

    EXPECT_FALSE(config.has_value());
}

/**
 * Test: Missing timing info
 */
TEST(FronthaulConfigParser, MissingTimingInfo) {
    const std::string yaml_content = R"(
ru_emulator:
  cell_configs:
    - eth: "aa:bb:cc:dd:ee:ff"
      vlan: 2
      pcp: 7
  aerial_fh_mtu: 1514
)";

    const TempYamlFile temp_file(yaml_content);
    const auto config = parse_fronthaul_config(temp_file.path());

    EXPECT_FALSE(config.has_value());
}

/**
 * Test: Missing MTU
 */
TEST(FronthaulConfigParser, MissingMtu) {
    const std::string yaml_content = R"(
ru_emulator:
  cell_configs:
    - eth: "aa:bb:cc:dd:ee:ff"
      vlan: 2
      pcp: 7
  oran_timing_info:
    ul_c_plane_timing_delay: 336
    ul_c_plane_window_size: 51
)";

    const TempYamlFile temp_file(yaml_content);
    const auto config = parse_fronthaul_config(temp_file.path());

    EXPECT_FALSE(config.has_value());
}

/**
 * Test: Missing MAC address in cell config
 */
TEST(FronthaulConfigParser, MissingMacAddress) {
    const std::string yaml_content = R"(
ru_emulator:
  cell_configs:
    - vlan: 2
      pcp: 7
  oran_timing_info:
    ul_c_plane_timing_delay: 336
    ul_c_plane_window_size: 51
  aerial_fh_mtu: 1514
)";

    const TempYamlFile temp_file(yaml_content);
    const auto config = parse_fronthaul_config(temp_file.path());

    EXPECT_FALSE(config.has_value());
}

/**
 * Test: Invalid YAML syntax
 */
TEST(FronthaulConfigParser, InvalidYamlSyntax) {
    const std::string yaml_content = R"(
ru_emulator:
  cell_configs:
    - eth: "aa:bb:cc:dd:ee:ff
      vlan: 2
)";

    const TempYamlFile temp_file(yaml_content);
    const auto config = parse_fronthaul_config(temp_file.path());

    EXPECT_FALSE(config.has_value());
}

/**
 * Test: File not found
 */
TEST(FronthaulConfigParser, FileNotFound) {
    const auto config = parse_fronthaul_config("/nonexistent/path/config.yaml");
    EXPECT_FALSE(config.has_value());
}

/**
 * Test: Parse from string
 */
TEST(FronthaulConfigParser, ParseFromString) {
    const std::string yaml_content = R"(
ru_emulator:
  cell_configs:
    - eth: "aa:bb:cc:dd:ee:ff"
      vlan: 2
      pcp: 7
      eAxC_UL: [0,1,2,3]
  oran_timing_info:
    ul_c_plane_timing_delay: 336
    ul_c_plane_window_size: 51
  aerial_fh_mtu: 1514
)";

    const auto config_opt = parse_fronthaul_config_from_string(yaml_content);

    ASSERT_TRUE(config_opt.has_value());
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    const auto &config = config_opt.value();

    EXPECT_EQ(config.cells.size(), 1);
    EXPECT_EQ(config.cells[0].mac_address, "aa:bb:cc:dd:ee:ff");
    EXPECT_EQ(config.mtu_size, 1514);
}

/**
 * Test: Parse from empty string
 */
TEST(FronthaulConfigParser, ParseFromEmptyString) {
    const auto config = parse_fronthaul_config_from_string("");
    EXPECT_FALSE(config.has_value());
}

} // namespace ran::fronthaul::tests
