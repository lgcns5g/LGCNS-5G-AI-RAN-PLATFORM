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

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

#include <aerial-fh-driver/oran.hpp>
#include <rte_ether.h>

#include <gtest/gtest.h>

#include "fronthaul/fronthaul.hpp"
#include "net/dpdk_types.hpp"
#include "oran/cplane_types.hpp"
#include "oran/cplane_utils.hpp"

namespace {

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

namespace fh = ran::fronthaul;
namespace oran = ran::oran;

/**
 * Test basic packet header template creation with default parameters
 */
TEST(PacketHeaderTemplateTest, BasicHeaderCreation) {
    static constexpr framework::net::MacAddress SRC_MAC{0x11, 0x22, 0x33, 0x44, 0x55, 0x66};
    static constexpr framework::net::MacAddress DEST_MAC{0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF};
    static constexpr std::uint16_t TEST_VLAN_TCI = 0xe002;
    static constexpr std::uint16_t TEST_EAC_ID = 0x0304; // cell=3, antenna=4

    const auto header =
            fh::create_packet_header_template(SRC_MAC, DEST_MAC, TEST_VLAN_TCI, TEST_EAC_ID);

    // Verify Ethernet header - source MAC
    static constexpr std::size_t MAC_ADDR_SIZE = 6;
    for (std::size_t i = 0; i < MAC_ADDR_SIZE; ++i) {
        EXPECT_EQ(std::span{header.eth.src_addr.addr_bytes}[i], std::span{SRC_MAC.bytes}[i]);
    }

    // Verify Ethernet header - destination MAC
    for (std::size_t i = 0; i < MAC_ADDR_SIZE; ++i) {
        EXPECT_EQ(std::span{header.eth.dst_addr.addr_bytes}[i], std::span{DEST_MAC.bytes}[i]);
    }

    // Verify Ethernet type is VLAN
    EXPECT_EQ(header.eth.ether_type, oran::cpu_to_be_16(RTE_ETHER_TYPE_VLAN));
}

/**
 * Test VLAN header fields
 */
TEST(PacketHeaderTemplateTest, VlanHeaderFields) {
    static constexpr framework::net::MacAddress SRC_MAC{0x11, 0x22, 0x33, 0x44, 0x55, 0x66};
    static constexpr framework::net::MacAddress DEST_MAC{0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF};
    static constexpr std::uint16_t TEST_VLAN_TCI = 0xe002;
    static constexpr std::uint16_t TEST_EAC_ID = 0x0304;

    const auto header =
            fh::create_packet_header_template(SRC_MAC, DEST_MAC, TEST_VLAN_TCI, TEST_EAC_ID);

    // Verify VLAN TCI is in network byte order
    EXPECT_EQ(header.vlan.vlan_tci, oran::cpu_to_be_16(TEST_VLAN_TCI));

    // Verify VLAN ethertype is eCPRI
    EXPECT_EQ(header.vlan.eth_proto, oran::cpu_to_be_16(ETHER_TYPE_ECPRI));
}

/**
 * Test eCPRI header fixed fields
 */
TEST(PacketHeaderTemplateTest, EcpriHeaderFixedFields) {
    static constexpr framework::net::MacAddress SRC_MAC{0x11, 0x22, 0x33, 0x44, 0x55, 0x66};
    static constexpr framework::net::MacAddress DEST_MAC{0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF};
    static constexpr std::uint16_t TEST_VLAN_TCI = 0xe002;
    static constexpr std::uint16_t TEST_EAC_ID = 0x0304;

    auto header = fh::create_packet_header_template(SRC_MAC, DEST_MAC, TEST_VLAN_TCI, TEST_EAC_ID);

    // Extract bitfield values (bitfield conversion operator is not const)
    // NOLINTBEGIN(cppcoreguidelines-pro-type-union-access)
    const std::uint8_t version = header.ecpri.ecpriVersion;
    const std::uint8_t reserved = header.ecpri.ecpriReserved;
    const std::uint8_t concatenation = header.ecpri.ecpriConcatenation;
    const std::uint8_t message = header.ecpri.ecpriMessage;
    const std::uint8_t ebit = header.ecpri.ecpriEbit;
    const std::uint8_t subseqid = header.ecpri.ecpriSubSeqid;
    // NOLINTEND(cppcoreguidelines-pro-type-union-access)

    EXPECT_EQ(version, ORAN_DEF_ECPRI_VERSION);
    EXPECT_EQ(reserved, ORAN_DEF_ECPRI_RESERVED);
    EXPECT_EQ(concatenation, ORAN_ECPRI_CONCATENATION_NO);
    EXPECT_EQ(message, ECPRI_MSG_TYPE_RTC); // C-Plane
    EXPECT_EQ(ebit, 1);                     // End bit set
    EXPECT_EQ(subseqid, 0);                 // No sub-sequencing
}

/**
 * Test enhanced antenna carrier ID encoding in PC ID field
 */
TEST(PacketHeaderTemplateTest, EnhancedAntennaCarrierEncoding) {
    static constexpr framework::net::MacAddress SRC_MAC{0x11, 0x22, 0x33, 0x44, 0x55, 0x66};
    static constexpr framework::net::MacAddress DEST_MAC{0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF};
    static constexpr std::uint16_t TEST_VLAN_TCI = 0xe002;
    static constexpr std::uint16_t TEST_EAC_ID = 0x0304;

    const auto header =
            fh::create_packet_header_template(SRC_MAC, DEST_MAC, TEST_VLAN_TCI, TEST_EAC_ID);

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-union-access)
    const auto pcid = oran::cpu_to_be_16(header.ecpri.ecpriPcid); // cpu_to_be is bidirectional
    EXPECT_EQ(pcid, TEST_EAC_ID);
}

/**
 * Test cell and antenna port encoding in enhanced antenna carrier ID
 */
TEST(PacketHeaderTemplateTest, CellAndAntennaPortEncoding) {
    static constexpr framework::net::MacAddress SRC_MAC{0x11, 0x22, 0x33, 0x44, 0x55, 0x66};
    static constexpr framework::net::MacAddress DEST_MAC{0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF};
    static constexpr std::uint16_t TEST_VLAN_TCI = 0xe002;

    // Test encoding: (cell_idx << 8) | antenna_port_idx
    static constexpr std::uint16_t CELL_IDX = 5U;
    static constexpr std::uint16_t ANTENNA_PORT = 7U;
    // NOLINTNEXTLINE(hicpp-signed-bitwise)
    static constexpr auto EAC_ID = static_cast<std::uint16_t>((CELL_IDX << 8U) | ANTENNA_PORT);

    const auto header = fh::create_packet_header_template(SRC_MAC, DEST_MAC, TEST_VLAN_TCI, EAC_ID);

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-union-access)
    const auto pcid = oran::cpu_to_be_16(header.ecpri.ecpriPcid);
    EXPECT_EQ(pcid, EAC_ID);
    // NOLINTNEXTLINE(hicpp-signed-bitwise)
    static constexpr std::uint16_t EXPECTED_EAC_FINAL = (CELL_IDX << 8U) | ANTENNA_PORT;
    EXPECT_EQ(pcid, EXPECTED_EAC_FINAL);
}

/**
 * Test different VLAN TCI values
 */
TEST(PacketHeaderTemplateTest, DifferentVlanTciValues) {
    static constexpr framework::net::MacAddress SRC_MAC{0x11, 0x22, 0x33, 0x44, 0x55, 0x66};
    static constexpr framework::net::MacAddress DEST_MAC{0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF};
    static constexpr std::uint16_t TEST_EAC_ID = 0x0304;

    static constexpr std::array<std::uint16_t, 3> TEST_VLAN_VALUES = {
            0x0000, // Zero
            0xe002, // Default
            0xFFFF  // Max value
    };

    for (const auto vlan_tci : TEST_VLAN_VALUES) {
        SCOPED_TRACE(vlan_tci);
        const auto header =
                fh::create_packet_header_template(SRC_MAC, DEST_MAC, vlan_tci, TEST_EAC_ID);
        EXPECT_EQ(header.vlan.vlan_tci, oran::cpu_to_be_16(vlan_tci));
    }
}

/**
 * Test different MAC address combinations
 */
TEST(PacketHeaderTemplateTest, DifferentMacAddresses) {
    static constexpr framework::net::MacAddress ZERO_MAC{0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
    static constexpr framework::net::MacAddress BROADCAST_MAC{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
    static constexpr std::uint16_t TEST_VLAN_TCI = 0xe002;
    static constexpr std::uint16_t TEST_EAC_ID = 0x0304;

    const auto header1 =
            fh::create_packet_header_template(ZERO_MAC, BROADCAST_MAC, TEST_VLAN_TCI, TEST_EAC_ID);
    static constexpr std::size_t MAC_SIZE = 6;
    for (std::size_t i = 0; i < MAC_SIZE; ++i) {
        EXPECT_EQ(std::span{header1.eth.src_addr.addr_bytes}[i], 0x00);
        EXPECT_EQ(std::span{header1.eth.dst_addr.addr_bytes}[i], 0xFF);
    }

    const auto header2 =
            fh::create_packet_header_template(BROADCAST_MAC, ZERO_MAC, TEST_VLAN_TCI, TEST_EAC_ID);
    for (std::size_t i = 0; i < MAC_SIZE; ++i) {
        EXPECT_EQ(std::span{header2.eth.src_addr.addr_bytes}[i], 0xFF);
        EXPECT_EQ(std::span{header2.eth.dst_addr.addr_bytes}[i], 0x00);
    }
}

/**
 * Test boundary values for cell and antenna port encoding
 */
TEST(PacketHeaderTemplateTest, BoundaryValuesForEncodingTest) {
    static constexpr framework::net::MacAddress SRC_MAC{0x11, 0x22, 0x33, 0x44, 0x55, 0x66};
    static constexpr framework::net::MacAddress DEST_MAC{0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF};
    static constexpr std::uint16_t TEST_VLAN_TCI = 0xe002;

    // Test max values: cell=255, antenna=255
    static constexpr std::uint16_t MAX_EAC = 0xFFFF;
    const auto header_max =
            fh::create_packet_header_template(SRC_MAC, DEST_MAC, TEST_VLAN_TCI, MAX_EAC);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-union-access)
    EXPECT_EQ(oran::cpu_to_be_16(header_max.ecpri.ecpriPcid), MAX_EAC);

    // Test min values: cell=0, antenna=0
    static constexpr std::uint16_t MIN_EAC = 0x0000;
    const auto header_min =
            fh::create_packet_header_template(SRC_MAC, DEST_MAC, TEST_VLAN_TCI, MIN_EAC);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-union-access)
    EXPECT_EQ(oran::cpu_to_be_16(header_min.ecpri.ecpriPcid), MIN_EAC);
}

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

} // namespace
