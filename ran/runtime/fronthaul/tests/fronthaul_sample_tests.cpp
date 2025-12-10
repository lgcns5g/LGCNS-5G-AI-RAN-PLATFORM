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
 * @file fronthaul_sample_tests.cpp
 * @brief Sample tests for fronthaul library documentation
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <aerial-fh-driver/oran.hpp>
#include <tl/expected.hpp>

#include <gtest/gtest.h>

#include "fronthaul/fronthaul.hpp"
#include "fronthaul/fronthaul_parser.hpp"
#include "fronthaul/uplane_config.hpp"
#include "net/dpdk_types.hpp"
#include "net/env.hpp"
#include "net/nic.hpp"
#include "oran/cplane_types.hpp"
#include "oran/cplane_utils.hpp"
#include "oran/numerology.hpp"

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

namespace {

using ran::fronthaul::calculate_packet_send_time;
using ran::fronthaul::create_packet_header_template;
using ran::fronthaul::FronthaulConfig;
using ran::fronthaul::FronthaulStats;
using ran::fronthaul::PacketSendTimeParams;
using ran::fronthaul::parse_fronthaul_config;
using ran::fronthaul::parse_fronthaul_config_from_string;
using ran::fronthaul::UPlaneConfig;
using namespace std::chrono_literals;

TEST(FronthaulSampleTests, ParseConfiguration) {
    // example-begin parse-config-1
    // Parse fronthaul configuration from YAML file
    const std::filesystem::path config_path = "ru_emulator_config.yaml";
    const auto yaml_config = parse_fronthaul_config(config_path);

    // Check if parsing succeeded
    const auto success = yaml_config.has_value();
    // example-end parse-config-1

    // Note: File may not exist in test environment, so we don't assert on success
    // Just demonstrate the API usage
    if (success) {
        const auto &config = yaml_config.value();
        EXPECT_GE(config.cells.size(), 0);
    }
}

TEST(FronthaulSampleTests, ParseConfigFromString) {
    // example-begin parse-string-1
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

    // Parse configuration from string
    const auto yaml_config = parse_fronthaul_config_from_string(yaml_content);

    // Access parsed values
    const auto &config = yaml_config.value();
    const auto num_cells = config.cells.size();
    const auto mac_address = config.cells[0].mac_address;
    const auto vlan_tci = config.cells[0].vlan_tci;
    const auto mtu = config.mtu_size;
    const auto t1a_max = config.timing.t1a_max_ns;
    // example-end parse-string-1

    EXPECT_TRUE(yaml_config.has_value());
    EXPECT_EQ(num_cells, 1);
    EXPECT_EQ(mac_address, "aa:bb:cc:dd:ee:ff");
    EXPECT_EQ(vlan_tci, 0xE002);
    EXPECT_EQ(mtu, 1514);
    EXPECT_GT(t1a_max, 0);
}

TEST(FronthaulSampleTests, UPlaneConfig) {
    // example-begin uplane-config-1
    // Create U-Plane configuration with default settings
    UPlaneConfig uplane_config{};

    // Customize timing windows for 30kHz SCS
    uplane_config.ta4_min_ns = 50'000;        // 50us early window
    uplane_config.ta4_max_ns = 450'000;       // 450us late window
    uplane_config.slot_duration_ns = 500'000; // 500us slot

    // Configure packet reception
    uplane_config.num_packets = 16384;    // 16K packet buffers
    uplane_config.max_packet_size = 8192; // 8KB max packet size

    // Access configured values
    const auto ta4_min = uplane_config.ta4_min_ns;
    const auto ta4_max = uplane_config.ta4_max_ns;
    const auto num_pkts = uplane_config.num_packets;
    // example-end uplane-config-1

    EXPECT_EQ(ta4_min, 50'000);
    EXPECT_EQ(ta4_max, 450'000);
    EXPECT_EQ(num_pkts, 16384);
}

TEST(FronthaulSampleTests, PacketSendTimeCalculation) {
    // example-begin packet-timing-1
    // Calculate packet send time for a slot
    const PacketSendTimeParams params{
            .t0 = 0ns,
            .tai_offset = 0ns,
            .absolute_slot = 100,
            .slot_period = 500us,
            .slot_ahead = 1,
            .t1a_max_cp_ul = 285us,
            .actual_start = 50ms};

    const auto result = calculate_packet_send_time(params);

    // Access timing results
    const auto actual_start = result.actual_start;
    const auto start_tx = result.start_tx;
    // example-end packet-timing-1

    EXPECT_EQ(actual_start, 50ms);
    EXPECT_GT(start_tx.count(), 0);
}

TEST(FronthaulSampleTests, CreatePacketHeader) {
    // example-begin packet-header-1
    // Create packet header template for ORAN C-Plane
    const framework::net::MacAddress src_mac{{0x11, 0x22, 0x33, 0x44, 0x55, 0x66}};
    const framework::net::MacAddress dest_mac{{0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff}};
    const std::uint16_t vlan_tci = 0xE002; // VLAN 2, PCP 7
    const std::uint16_t eac_id = 0;        // Enhanced antenna carrier ID

    const auto header = create_packet_header_template(src_mac, dest_mac, vlan_tci, eac_id);

    // Header contains ethernet, VLAN, and eCPRI fields
    const auto header_size = sizeof(header);
    // example-end packet-header-1

    EXPECT_GT(header_size, 0);
    EXPECT_EQ(header.vlan.vlan_tci, ran::oran::cpu_to_be_16(vlan_tci));
}

TEST(FronthaulSampleTests, FronthaulConfigBuilder) {
    // example-begin fronthaul-config-1
    // Build fronthaul configuration
    FronthaulConfig config{};

    // Network configuration - NIC address
    config.net_config.nic_config.nic_pcie_addr = "0000:17:00.0";

    // Cell configuration - destination MACs and VLANs
    config.cell_dest_macs.push_back({{0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff}});
    config.cell_vlan_tcis.push_back(0xE002); // VLAN 2, PCP 7

    // ORAN parameters
    config.numerology = ran::oran::from_scs(ran::oran::SubcarrierSpacing::Scs30Khz);
    config.num_antenna_ports = 4;
    config.mtu = 1514;

    // Timing parameters
    config.slot_ahead = 1;
    config.t1a_max_cp_ul_ns = 285'000; // 285us
    config.t1a_min_cp_ul_ns = 234'000; // 234us

    // Access configuration
    const auto num_cells = config.cell_dest_macs.size();
    const auto num_ports = config.num_antenna_ports;
    const auto mtu_size = config.mtu;
    // example-end fronthaul-config-1

    EXPECT_EQ(num_cells, 1);
    EXPECT_EQ(num_ports, 4);
    EXPECT_EQ(mtu_size, 1514);
}

TEST(FronthaulSampleTests, FronthaulStatistics) {
    // example-begin fronthaul-stats-1
    // Create statistics structure
    FronthaulStats stats{};
    stats.requests_sent = 1000;
    stats.packets_sent = 4000;
    stats.send_errors = 0;
    stats.avg_packets_per_request = 4.0;

    // Access statistics
    const auto total_requests = stats.requests_sent;
    const auto total_packets = stats.packets_sent;
    const auto avg_packets = stats.avg_packets_per_request;
    const auto errors = stats.send_errors;
    // example-end fronthaul-stats-1

    EXPECT_EQ(total_requests, 1000);
    EXPECT_EQ(total_packets, 4000);
    EXPECT_EQ(avg_packets, 4.0);
    EXPECT_EQ(errors, 0);
}

} // namespace

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
