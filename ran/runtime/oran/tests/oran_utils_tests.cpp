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

#include <cstdint>
#include <cstring>
#include <format>
#include <span>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <aerial-fh-driver/oran.hpp>

#include <gtest/gtest.h>

#include "fapi/fapi_buffer.hpp"
#include "oran/cplane_utils.hpp"

namespace {

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers,cppcoreguidelines-pro-type-union-access)

/**
 * Parameters for creating basic ORAN message buffer
 */
struct OranMessageParams {
    std::uint8_t frame_id{};
    std::uint8_t subframe_id{};
    std::uint8_t slot_id{};
    std::uint8_t symbol_id{};
};

/**
 * Helper function to create a minimal valid ORAN message buffer
 */
std::vector<std::uint8_t> create_oran_message_buffer(const OranMessageParams &params) {
    std::vector<std::uint8_t> buffer(1024, 0);

    // Ethernet header (14 bytes) + VLAN header (4 bytes) = 18 bytes
    constexpr std::size_t ETH_HEADER_SIZE = 18;

    // eCPRI header starts at offset 18
    auto buffer_span = ran::fapi::make_buffer_span(buffer.data(), buffer.size());
    auto ecpri_span = buffer_span.subspan(ETH_HEADER_SIZE);
    auto *ecpri_hdr = ran::fapi::assume_cast<oran_ecpri_hdr>(ecpri_span.data());
    ecpri_hdr->ecpriVersion = ORAN_DEF_ECPRI_VERSION;
    ecpri_hdr->ecpriReserved = ORAN_DEF_ECPRI_RESERVED;
    ecpri_hdr->ecpriConcatenation = ORAN_ECPRI_CONCATENATION_NO;
    ecpri_hdr->ecpriMessage = ECPRI_MSG_TYPE_IQ;

    // U-plane message header starts after eCPRI header
    constexpr std::size_t U_MSG_OFFSET = ETH_HEADER_SIZE + sizeof(oran_ecpri_hdr);
    auto u_msg_span = buffer_span.subspan(U_MSG_OFFSET);
    auto *u_msg_hdr = ran::fapi::assume_cast<oran_umsg_iq_hdr>(u_msg_span.data());

    // Set message fields
    u_msg_hdr->frameId = params.frame_id;
    u_msg_hdr->subframeId = params.subframe_id;
    u_msg_hdr->slotId = params.slot_id;
    u_msg_hdr->symbolId = params.symbol_id;

    return buffer;
}

/**
 * Test data structure for ORAN message field tests
 */
struct OranMessageTestData {
    std::uint8_t frame_id{};
    std::uint8_t subframe_id{};
    std::uint8_t slot_id{};
    std::uint8_t symbol_id{};
    const char *description{};
};

/**
 * Test data structure for eCPRI message type tests
 */
struct EcpriMsgTypeTestData {
    int msg_type{};
    const char *expected_string{};
    const char *description{};
};

/**
 * Table-based test for ecpri_msgtype_to_string utility function
 */
TEST(oran_utility_test, ecpri_msg_type_to_string) {
    const std::vector<EcpriMsgTypeTestData> test_cases = {
            // Valid message types
            {ECPRI_MSG_TYPE_IQ, "Type #0: IQ Data", "IQ Data message type"},
            {ECPRI_MSG_TYPE_RTC, "Type #2: Real-Time Control", "Real-Time Control message type"},
            {ECPRI_MSG_TYPE_ND, "Type #5: Network Delay", "Network Delay message type"},

            // Invalid/unknown message types
            {-1, "Unknown", "Negative message type"},
            {1, "Unknown", "Undefined message type 1"},
            {3, "Unknown", "Undefined message type 3"},
            {4, "Unknown", "Undefined message type 4"},
            {6, "Unknown", "Undefined message type 6"},
            {99, "Unknown", "Large invalid message type"},
            {255, "Unknown", "Maximum uint8_t value"},
            {1000, "Unknown", "Very large invalid message type"}};

    for (const auto &test_case : test_cases) {
        SCOPED_TRACE(test_case.description);
        EXPECT_STREQ(ecpri_msgtype_to_string(test_case.msg_type), test_case.expected_string);
    }
}

/**
 * Test data structure for maximum transmission bandwidth tests
 */
struct MaxTransmissionBandwidthTestData {
    int scs{};
    int channel_bandwidth{};
    int expected_result{};
    const char *description{};
};

/**
 * Table-based test for getMaxTransmissionBandwidth function
 */
TEST(oran_utility_test, get_max_transmission_bandwidth) {
    const std::vector<MaxTransmissionBandwidthTestData> test_cases = {
            // Valid inputs for 15 kHz SCS (scs = 0)
            {0, 5, 25, "15 kHz SCS, 5 MHz bandwidth"},
            {0, 10, 52, "15 kHz SCS, 10 MHz bandwidth"},
            {0, 15, 79, "15 kHz SCS, 15 MHz bandwidth"},
            {0, 20, 106, "15 kHz SCS, 20 MHz bandwidth"},
            {0, 25, 133, "15 kHz SCS, 25 MHz bandwidth"},
            {0, 30, 160, "15 kHz SCS, 30 MHz bandwidth"},
            {0, 40, 216, "15 kHz SCS, 40 MHz bandwidth"},
            {0, 50, 270, "15 kHz SCS, 50 MHz bandwidth"},

            // Valid inputs for 30 kHz SCS (scs = 1)
            {1, 5, 11, "30 kHz SCS, 5 MHz bandwidth"},
            {1, 10, 24, "30 kHz SCS, 10 MHz bandwidth"},
            {1, 15, 38, "30 kHz SCS, 15 MHz bandwidth"},
            {1, 20, 51, "30 kHz SCS, 20 MHz bandwidth"},
            {1, 25, 65, "30 kHz SCS, 25 MHz bandwidth"},
            {1, 30, 78, "30 kHz SCS, 30 MHz bandwidth"},
            {1, 40, 106, "30 kHz SCS, 40 MHz bandwidth"},
            {1, 50, 133, "30 kHz SCS, 50 MHz bandwidth"},
            {1, 60, 162, "30 kHz SCS, 60 MHz bandwidth"},
            {1, 80, 217, "30 kHz SCS, 80 MHz bandwidth"},
            {1, 100, 273, "30 kHz SCS, 100 MHz bandwidth"},

            // Valid inputs for 60 kHz SCS (scs = 2)
            {2, 10, 11, "60 kHz SCS, 10 MHz bandwidth"},
            {2, 15, 18, "60 kHz SCS, 15 MHz bandwidth"},
            {2, 20, 24, "60 kHz SCS, 20 MHz bandwidth"},
            {2, 25, 31, "60 kHz SCS, 25 MHz bandwidth"},
            {2, 30, 38, "60 kHz SCS, 30 MHz bandwidth"},
            {2, 40, 51, "60 kHz SCS, 40 MHz bandwidth"},
            {2, 50, 65, "60 kHz SCS, 50 MHz bandwidth"},
            {2, 60, 79, "60 kHz SCS, 60 MHz bandwidth"},
            {2, 80, 107, "60 kHz SCS, 80 MHz bandwidth"},
            {2, 90, 121, "60 kHz SCS, 90 MHz bandwidth"},
            {2, 100, 135, "60 kHz SCS, 100 MHz bandwidth"},

            // Invalid SCS values
            {-1, 5, -1, "Negative SCS value"},
            {3, 5, -1, "SCS out of range (too high)"},
            {10, 20, -1, "Very high SCS value"},

            // Invalid bandwidth values
            {0, 0, -1, "Zero bandwidth"},
            {0, -5, -1, "Negative bandwidth"},
            {0, 101, -1, "Bandwidth too large"},
            {1, 200, -1, "Very large bandwidth"},

            // Unsupported combinations (should return 0 based on lookup table)
            {0, 6, 0, "15 kHz SCS, unsupported 6 MHz bandwidth"},
            {0, 7, 0, "15 kHz SCS, unsupported 7 MHz bandwidth"},
            {1, 6, 0, "30 kHz SCS, unsupported 6 MHz bandwidth"},
            {2, 5, 0, "60 kHz SCS, unsupported 5 MHz bandwidth"}};

    for (const auto &test_case : test_cases) {
        SCOPED_TRACE(test_case.description);
        EXPECT_EQ(
                getMaxTransmissionBandwidth(test_case.scs, test_case.channel_bandwidth),
                test_case.expected_result);
    }
}

/**
 * Test data structure for bitfield tests
 */
template <typename T, int OFFSET, int BITS> struct BitfieldTestData {
    T input_value{};
    T expected_output{};
    const char *description{};
};

/**
 * Test for Bitfield template class basic operations
 */
TEST(BitfieldTest, basic_operations_uint8) {
    const std::vector<BitfieldTestData<std::uint8_t, 0, 4>> test_cases = {
            {0x0, 0x0, "Zero value"},
            {0x5, 0x5, "Mid-range value"},
            {0xF, 0xF, "Maximum 4-bit value"},
            {0x1F, 0xF, "Overflow - should mask to 4 bits"},
            {0xFF, 0xF, "Full byte - should mask to 4 bits"}};

    for (const auto &test_case : test_cases) {
        SCOPED_TRACE(test_case.description);

        Bitfield<std::uint8_t, 0, 4> bf{};
        bf = test_case.input_value;

        EXPECT_EQ(static_cast<std::uint8_t>(bf), test_case.expected_output);
        EXPECT_EQ(bf.get(), test_case.expected_output);
    }
}

/**
 * Test for Bitfield template class with different offsets
 */
TEST(BitfieldTest, offset_operations_uint8) {
    const std::vector<BitfieldTestData<std::uint8_t, 4, 4>> test_cases = {
            {0x0, 0x0, "Zero value at offset 4"},
            {0x3, 0x3, "Value 3 at offset 4"},
            {0xF, 0xF, "Maximum 4-bit value at offset 4"},
            {0x1F, 0xF, "Overflow at offset 4"}};

    for (const auto &test_case : test_cases) {
        SCOPED_TRACE(test_case.description);

        Bitfield<std::uint8_t, 4, 4> bf{};
        bf = test_case.input_value;

        EXPECT_EQ(static_cast<std::uint8_t>(bf), test_case.expected_output);
        EXPECT_EQ(bf.get(), test_case.expected_output);
    }
}

/**
 * Test for Bitfield template class with uint16_t
 */
TEST(BitfieldTest, basic_operations_uint16) {
    const std::vector<BitfieldTestData<std::uint16_t, 0, 12>> test_cases = {
            {0x000, 0x000, "Zero value"},
            {0x123, 0x123, "Mid-range value"},
            {0xFFF, 0xFFF, "Maximum 12-bit value"},
            {0x1FFF, 0xFFF, "Overflow - should mask to 12 bits"},
            {0xFFFF, 0xFFF, "Full 16-bit - should mask to 12 bits"}};

    for (const auto &test_case : test_cases) {
        SCOPED_TRACE(test_case.description);

        Bitfield<std::uint16_t, 0, 12> bf{};
        bf = test_case.input_value;

        EXPECT_EQ(static_cast<std::uint16_t>(bf), test_case.expected_output);
        EXPECT_EQ(bf.get(), test_case.expected_output);
    }
}

/**
 * Test for Bitfield template class with uint32_t
 */
TEST(BitfieldTest, basic_operations_uint32) {
    const std::vector<BitfieldTestData<std::uint32_t, 8, 16>> test_cases = {
            {0x0000, 0x0000, "Zero value"},
            {0x1234, 0x1234, "Mid-range value"},
            {0xFFFF, 0xFFFF, "Maximum 16-bit value"},
            {0x1FFFF, 0xFFFF, "Overflow - should mask to 16 bits"},
            {0xFFFFFFFF, 0xFFFF, "Full 32-bit - should mask to 16 bits"}};

    for (const auto &test_case : test_cases) {
        SCOPED_TRACE(test_case.description);

        Bitfield<std::uint32_t, 8, 16> bf{};
        bf = test_case.input_value;

        EXPECT_EQ(static_cast<std::uint32_t>(bf), test_case.expected_output);
        EXPECT_EQ(bf.get(), test_case.expected_output);
    }
}

/**
 * Test for Bitfield boundary conditions
 */
TEST(BitfieldTest, boundary_conditions) {
    // Test single bit field
    {
        Bitfield<std::uint8_t, 7, 1> single_bit{};
        single_bit = 0;
        EXPECT_EQ(static_cast<std::uint8_t>(single_bit), 0);

        single_bit = 1;
        EXPECT_EQ(static_cast<std::uint8_t>(single_bit), 1);

        single_bit = 2; // Should mask to 1 bit
        EXPECT_EQ(static_cast<std::uint8_t>(single_bit), 0);
    }

    // Test maximum width field (7 bits for uint8_t)
    {
        Bitfield<std::uint8_t, 0, 7> max_width{};
        max_width = 0x7F; // Maximum 7-bit value
        EXPECT_EQ(static_cast<std::uint8_t>(max_width), 0x7F);

        max_width = 0xFF; // Should mask to 7 bits
        EXPECT_EQ(static_cast<std::uint8_t>(max_width), 0x7F);
    }
}

/**
 * Test data structure for structure layout tests
 */
struct StructureSizeTestData {
    std::size_t actual_size{};
    std::size_t expected_size{};
    const char *struct_name{};
};

/**
 * Test for structure sizes and alignment
 */
TEST(StructureLayoutTest, struct_sizes_and_alignment) {
    const std::vector<StructureSizeTestData> test_cases = {
            // Ethernet structures
            {sizeof(oran_ether_addr), 6, "oran_ether_addr"},
            {sizeof(oran_ether_hdr), 14, "oran_ether_hdr"},
            {sizeof(oran_vlan_hdr), 4, "oran_vlan_hdr"},
            {sizeof(oran_eth_hdr), 18, "oran_eth_hdr"},

            // eCPRI structures
            {sizeof(oran_ecpri_hdr), 8, "oran_ecpri_hdr"},

            // U-plane structures
            {sizeof(oran_umsg_iq_hdr), 4, "oran_umsg_iq_hdr"},
            {sizeof(oran_u_section_uncompressed), 4, "oran_u_section_uncompressed"},
            {sizeof(oran_u_section_compression_hdr), 2, "oran_u_section_compression_hdr"},

            // Resource element structures
            {sizeof(oran_re_16b), 4, "oran_re_16b"},
            {sizeof(oran_prb_16b_uncompressed), 48, "oran_prb_16b_uncompressed"},
            {sizeof(oran_prb_14b_compressed), 43, "oran_prb_14b_compressed"},
            {sizeof(oran_prb_9b_compressed), 28, "oran_prb_9b_compressed"},

            // C-plane structures (updated with actual sizes)
            {sizeof(oran_cmsg_radio_app_hdr), 6, "oran_cmsg_radio_app_hdr"},
            {sizeof(oran_cmsg_sect0_common_hdr), 12, "oran_cmsg_sect0_common_hdr"},
            {sizeof(oran_cmsg_sect0), 8, "oran_cmsg_sect0"},
            {sizeof(oran_cmsg_sect1_common_hdr), 8, "oran_cmsg_sect1_common_hdr"},
            {sizeof(oran_cmsg_sect1), 8, "oran_cmsg_sect1"},
            {sizeof(oran_cmsg_sect3_common_hdr), 12, "oran_cmsg_sect3_common_hdr"},
            {sizeof(oran_cmsg_sect3), 12, "oran_cmsg_sect3"},
            {sizeof(oran_cmsg_sect5_common_hdr), 8, "oran_cmsg_sect5_common_hdr"},
            {sizeof(oran_cmsg_sect5), 8, "oran_cmsg_sect5"},
            {sizeof(oran_cmsg_sect6_common_hdr), 8, "oran_cmsg_sect6_common_hdr"},
            {sizeof(oran_cmsg_sect6), 11, "oran_cmsg_sect6"},

            // Extension structures
            {sizeof(oran_cmsg_ext_hdr), 1, "oran_cmsg_ext_hdr"},
            {sizeof(oran_cmsg_sect_ext_type_4), 3, "oran_cmsg_sect_ext_type_4"},
            {sizeof(oran_cmsg_sect_ext_type_5), 11, "oran_cmsg_sect_ext_type_5"},
            {sizeof(oran_cmsg_sect_ext_type_11), 4, "oran_cmsg_sect_ext_type_11"},
            {sizeof(oran_cmsg_sect_ext_type_11_disableBFWs_0_bfwCompHdr),
             1,
             "oran_cmsg_sect_ext_type_11_disableBFWs_0_bfwCompHdr"},
            {sizeof(oran_cmsg_sect_ext_type_11_disableBFWs_0_bfwCompParam),
             1,
             "oran_cmsg_sect_ext_type_11_disableBFWs_0_bfwCompParam"},
            {sizeof(oran_cmsg_sect_ext_type_11_disableBFWs_0_beamId),
             2,
             "oran_cmsg_sect_ext_type_11_disableBFWs_0_beamId"},
            {sizeof(oran_cmsg_sect_ext_type_11_disableBFWs_0_bundle_uncompressed),
             2,
             "oran_cmsg_sect_ext_type_11_disableBFWs_0_bundle_uncompressed"},
            {sizeof(oran_cmsg_sect_ext_type_11_disableBFWs_0_bfp_compressed_bundle_hdr),
             3,
             "oran_cmsg_sect_ext_type_11_disableBFWs_0_bfp_compressed_bundle_hdr"},
            {sizeof(oran_cmsg_sect_ext_type_11_disableBFWs_1_bundle),
             2,
             "oran_cmsg_sect_ext_type_11_disableBFWs_1_bundle"}};

    for (const auto &test_case : test_cases) {
        SCOPED_TRACE(test_case.struct_name);
        EXPECT_EQ(test_case.actual_size, test_case.expected_size)
                << "Structure " << test_case.struct_name << " has unexpected size";
    }
}

/**
 * Test for structure packing verification
 */
TEST(StructureLayoutTest, structure_packing) {
    // Verify that structures are properly packed (no padding)

    // Test oran_ether_addr alignment
    EXPECT_EQ(alignof(oran_ether_addr), 2); // Should be 2-byte aligned

    // Test oran_ether_hdr alignment
    EXPECT_EQ(alignof(oran_ether_hdr), 2); // Should be 2-byte aligned

    // Test packed structures have no padding
    EXPECT_EQ(sizeof(oran_vlan_hdr),
              4); // Should be exactly 4 bytes with no padding
    EXPECT_EQ(sizeof(oran_ecpri_hdr),
              8); // Should be exactly 8 bytes with no padding

    // Verify bitfield structures are properly packed
    EXPECT_EQ(sizeof(oran_umsg_iq_hdr), 4);
    EXPECT_EQ(sizeof(oran_u_section_uncompressed), 4);
}

/**
 * Test data structure for endianness tests
 */
struct EndiannessTestData {
    std::uint32_t test_value{};
    const char *description{};
};

/**
 * Test for endianness consistency across different bitfield operations
 */
TEST(EndiannessTest, big_endian_little_endian_consistency) {
    const std::vector<EndiannessTestData> test_cases = {
            {0x12345678, "Standard test pattern"},
            {0x00000000, "All zeros"},
            {0xFFFFFFFF, "All ones"},
            {0xAAAAAAAA, "Alternating pattern 1"},
            {0x55555555, "Alternating pattern 2"},
            {0x01234567, "Sequential pattern"},
            {0xFEDCBA98, "Reverse sequential pattern"}};

    for (const auto &test_case : test_cases) {
        SCOPED_TRACE(test_case.description);

        // Test that bitfield extraction is consistent regardless of endianness
        // Create a test message with known values
        std::vector<std::uint8_t> buffer(1024, 0);
        constexpr std::size_t ETH_HEADER_SIZE = 18;
        constexpr std::size_t U_MSG_OFFSET = ETH_HEADER_SIZE + sizeof(oran_ecpri_hdr);

        auto buffer_span = ran::fapi::make_buffer_span(buffer.data(), buffer.size());
        auto ecpri_span = buffer_span.subspan(ETH_HEADER_SIZE);
        auto *ecpri_hdr = ran::fapi::assume_cast<oran_ecpri_hdr>(ecpri_span.data());
        ecpri_hdr->ecpriVersion = ORAN_DEF_ECPRI_VERSION;
        ecpri_hdr->ecpriReserved = ORAN_DEF_ECPRI_RESERVED;
        ecpri_hdr->ecpriConcatenation = ORAN_ECPRI_CONCATENATION_NO;
        ecpri_hdr->ecpriMessage = ECPRI_MSG_TYPE_IQ;

        auto u_msg_span = buffer_span.subspan(U_MSG_OFFSET);
        auto *u_msg_hdr = ran::fapi::assume_cast<oran_umsg_iq_hdr>(u_msg_span.data());
        u_msg_hdr->frameId = static_cast<std::uint8_t>((test_case.test_value >> 24U) & 0xFFU);
        u_msg_hdr->subframeId = static_cast<std::uint8_t>((test_case.test_value >> 20U) & 0x0FU);
        u_msg_hdr->slotId = static_cast<std::uint8_t>((test_case.test_value >> 18U) & 0x03U);
        u_msg_hdr->symbolId = static_cast<std::uint8_t>((test_case.test_value >> 12U) & 0x3FU);

        // Extract values and verify they match what we put in
        const std::uint8_t extracted_frame_id = oran_umsg_get_frame_id(buffer.data());
        const std::uint8_t extracted_subframe_id = oran_umsg_get_subframe_id(buffer.data());
        const std::uint8_t extracted_slot_id = oran_umsg_get_slot_id(buffer.data());
        const std::uint8_t extracted_symbol_id = oran_umsg_get_symbol_id(buffer.data());

        EXPECT_EQ(
                extracted_frame_id,
                static_cast<std::uint8_t>((test_case.test_value >> 24U) & 0xFFU));
        EXPECT_EQ(
                extracted_subframe_id,
                static_cast<std::uint8_t>((test_case.test_value >> 20U) & 0x0FU));
        EXPECT_EQ(
                extracted_slot_id,
                static_cast<std::uint8_t>((test_case.test_value >> 18U) & 0x03U));
        EXPECT_EQ(
                extracted_symbol_id,
                static_cast<std::uint8_t>((test_case.test_value >> 12U) & 0x3FU));
    }
}

/**
 * Test for bitfield endianness handling in actual ORAN structures
 */
TEST(EndiannessTest, oran_structure_endianness) {
    // Test eCPRI header bitfields
    {
        std::vector<std::uint8_t> buffer(1024, 0);
        constexpr std::size_t ETH_HEADER_SIZE = 18;

        auto buffer_span = ran::fapi::make_buffer_span(buffer.data(), buffer.size());
        auto ecpri_span = buffer_span.subspan(ETH_HEADER_SIZE);
        auto *ecpri_hdr = ran::fapi::assume_cast<oran_ecpri_hdr>(ecpri_span.data());

        // Set known values
        ecpri_hdr->ecpriVersion = 1;
        ecpri_hdr->ecpriReserved = 0;
        ecpri_hdr->ecpriConcatenation = 0;
        ecpri_hdr->ecpriMessage = ECPRI_MSG_TYPE_IQ;
        ecpri_hdr->ecpriSubSeqid = 0;
        ecpri_hdr->ecpriEbit = 1;

        // Verify extraction works correctly
        EXPECT_EQ(static_cast<std::uint8_t>(ecpri_hdr->ecpriVersion), 1);
        EXPECT_EQ(static_cast<std::uint8_t>(ecpri_hdr->ecpriReserved), 0);
        EXPECT_EQ(static_cast<std::uint8_t>(ecpri_hdr->ecpriConcatenation), 0);
        EXPECT_EQ(ecpri_hdr->ecpriMessage, ECPRI_MSG_TYPE_IQ);
        EXPECT_EQ(static_cast<std::uint8_t>(ecpri_hdr->ecpriSubSeqid), 0);
        EXPECT_EQ(static_cast<std::uint8_t>(ecpri_hdr->ecpriEbit), 1);
    }

    // Test U-plane message header bitfields
    {
        std::vector<std::uint8_t> buffer(1024, 0);
        constexpr std::size_t U_MSG_OFFSET = 18 + sizeof(oran_ecpri_hdr);

        auto buffer_span = ran::fapi::make_buffer_span(buffer.data(), buffer.size());
        auto u_msg_span = buffer_span.subspan(U_MSG_OFFSET);
        auto *u_msg_hdr = ran::fapi::assume_cast<oran_umsg_iq_hdr>(u_msg_span.data());

        // Set known values
        u_msg_hdr->dataDirection = DIRECTION_DOWNLINK;
        u_msg_hdr->payloadVersion = ORAN_DEF_PAYLOAD_VERSION;
        u_msg_hdr->filterIndex = ORAN_DEF_FILTER_INDEX;
        u_msg_hdr->frameId = 42;
        u_msg_hdr->subframeId = 5;
        u_msg_hdr->slotId = 1;
        u_msg_hdr->symbolId = 7;

        // Verify extraction works correctly
        EXPECT_EQ(static_cast<std::uint8_t>(u_msg_hdr->dataDirection), DIRECTION_DOWNLINK);
        EXPECT_EQ(static_cast<std::uint8_t>(u_msg_hdr->payloadVersion), ORAN_DEF_PAYLOAD_VERSION);
        EXPECT_EQ(static_cast<std::uint8_t>(u_msg_hdr->filterIndex), ORAN_DEF_FILTER_INDEX);
        EXPECT_EQ(u_msg_hdr->frameId, 42);
        EXPECT_EQ(static_cast<std::uint8_t>(u_msg_hdr->subframeId), 5);
        EXPECT_EQ(static_cast<std::uint8_t>(u_msg_hdr->slotId), 1);
        EXPECT_EQ(static_cast<std::uint8_t>(u_msg_hdr->symbolId), 7);
    }
}

/**
 * Test for byte order consistency in payload and flow ID handling
 */
TEST(EndiannessTest, payload_and_flow_id_byte_order) {
    const std::vector<std::pair<std::uint16_t, const char *>> test_cases = {
            {0x0100, "Standard payload size"},
            {0x0080, "Small payload size"},
            {0x1234, "Test pattern 1"},
            {0xABCD, "Test pattern 2"},
            {0x0001, "Minimum payload"},
            {0xFFFF, "Maximum payload"}};

    for (const auto &[test_value, description] : test_cases) {
        SCOPED_TRACE(description);

        // Test U-plane payload and flow ID
        {
            std::vector<std::uint8_t> buffer(1024, 0);
            constexpr std::size_t ETH_HEADER_SIZE = 18;

            auto buffer_span = ran::fapi::make_buffer_span(buffer.data(), buffer.size());
            auto ecpri_span = buffer_span.subspan(ETH_HEADER_SIZE);
            auto *ecpri_hdr = ran::fapi::assume_cast<oran_ecpri_hdr>(ecpri_span.data());
            ecpri_hdr->ecpriVersion = ORAN_DEF_ECPRI_VERSION;
            ecpri_hdr->ecpriReserved = ORAN_DEF_ECPRI_RESERVED;
            ecpri_hdr->ecpriConcatenation = ORAN_ECPRI_CONCATENATION_NO;
            ecpri_hdr->ecpriMessage = ECPRI_MSG_TYPE_IQ;

            // Set test values
            ecpri_hdr->ecpriPayload = test_value;
            ecpri_hdr->ecpriPcid = test_value;

            // Test extraction with byte swapping
            const std::uint16_t extracted_payload = oran_umsg_get_ecpri_payload(buffer.data());
            const std::uint16_t extracted_flow_id = oran_umsg_get_flowid(buffer.data());

            // Values should be byte-swapped
            const auto expected_swapped = static_cast<std::uint16_t>(
                    ((test_value & 0xFFU) << 8U) | ((test_value & 0xFF00U) >> 8U));
            EXPECT_EQ(extracted_payload, expected_swapped);
            EXPECT_EQ(extracted_flow_id, expected_swapped);
        }

        // Test C-plane payload and flow ID
        {
            std::vector<std::uint8_t> buffer(1024, 0);
            constexpr std::size_t ETH_HEADER_SIZE = 18;
            constexpr std::size_t C_MSG_OFFSET = ETH_HEADER_SIZE + sizeof(oran_ecpri_hdr);

            auto buffer_span = ran::fapi::make_buffer_span(buffer.data(), buffer.size());
            auto ecpri_span = buffer_span.subspan(ETH_HEADER_SIZE);
            auto *ecpri_hdr = ran::fapi::assume_cast<oran_ecpri_hdr>(ecpri_span.data());
            ecpri_hdr->ecpriVersion = ORAN_DEF_ECPRI_VERSION;
            ecpri_hdr->ecpriReserved = ORAN_DEF_ECPRI_RESERVED;
            ecpri_hdr->ecpriConcatenation = ORAN_ECPRI_CONCATENATION_NO;
            ecpri_hdr->ecpriMessage = ECPRI_MSG_TYPE_RTC;

            auto c_msg_span = buffer_span.subspan(C_MSG_OFFSET);
            auto *c_msg_hdr = ran::fapi::assume_cast<oran_cmsg_radio_app_hdr>(c_msg_span.data());
            c_msg_hdr->dataDirection = DIRECTION_DOWNLINK;
            c_msg_hdr->payloadVersion = ORAN_DEF_PAYLOAD_VERSION;
            c_msg_hdr->filterIndex = ORAN_DEF_FILTER_INDEX;

            // Set test values
            ecpri_hdr->ecpriPayload = test_value;
            ecpri_hdr->ecpriRtcid = test_value;

            // Test extraction with byte swapping
            const std::uint16_t extracted_payload = oran_cmsg_get_ecpri_payload(buffer.data());
            const std::uint16_t extracted_flow_id = oran_cmsg_get_flowid(buffer.data());

            // Values should be byte-swapped
            const auto expected_swapped = static_cast<std::uint16_t>(
                    ((test_value & 0xFFU) << 8U) | ((test_value & 0xFF00U) >> 8U));
            EXPECT_EQ(extracted_payload, expected_swapped);
            EXPECT_EQ(extracted_flow_id, expected_swapped);
        }
    }
}

/**
 * Test for bitfield boundary conditions
 */
TEST(oran_message_bitfield_test, bitfield_boundaries) {
    // Test maximum values for each bitfield
    const std::vector<OranMessageTestData> boundary_test_cases = {
            // Test maximum values for each field based on bitfield sizes
            {255,
             15,
             63,
             63,
             "Maximum values for all bitfields"}, // Note: actual max may be smaller
                                                  // due to protocol constraints
            {0, 0, 0, 0, "Minimum values for all bitfields"},

            // Test protocol-specific maximum values
            {255, 9, 1, 13, "Protocol maximum values"}, // Frame: 0-255, Subframe: 0-9, Slot: 0-1
                                                        // (TTI=500), Symbol: 0-13

            // Test mid-range values
            {127, 4, 0, 6, "Mid-range values"},
            {128, 5, 1, 7, "Mid-range values + 1"}};

    for (const auto &test_case : boundary_test_cases) {
        SCOPED_TRACE(test_case.description);

        auto buffer = create_oran_message_buffer(
                {.frame_id = test_case.frame_id,
                 .subframe_id = test_case.subframe_id,
                 .slot_id = test_case.slot_id,
                 .symbol_id = test_case.symbol_id});

        // Verify all fields are extracted correctly
        EXPECT_EQ(oran_umsg_get_frame_id(buffer.data()), test_case.frame_id);
        EXPECT_EQ(oran_umsg_get_subframe_id(buffer.data()), test_case.subframe_id);
        EXPECT_EQ(oran_umsg_get_slot_id(buffer.data()), test_case.slot_id);
        EXPECT_EQ(oran_umsg_get_symbol_id(buffer.data()), test_case.symbol_id);
    }
}

/**
 * Test for endianness handling
 */
TEST(oran_message_endianness_test, endianness_consistency) {
    // Test that the same values are extracted regardless of the specific bit
    // patterns
    const std::vector<OranMessageTestData> endianness_test_cases = {
            {0x00, 0x0, 0x0, 0x00, "All zeros"},
            {0xFF, 0xF, 0x1, 0x0F, "Mixed bit patterns"},
            {0xAA, 0x5, 0x0, 0x0A, "Alternating bit pattern 1"},
            {0x55, 0xA, 0x1, 0x05, "Alternating bit pattern 2"},
            {0x0F, 0x0, 0x0, 0x0F, "Low nibble set"},
            {0xF0, 0xF, 0x1, 0x00, "High nibble set"}};

    for (const auto &test_case : endianness_test_cases) {
        SCOPED_TRACE(test_case.description);

        auto buffer = create_oran_message_buffer(
                {.frame_id = test_case.frame_id,
                 .subframe_id = test_case.subframe_id,
                 .slot_id = test_case.slot_id,
                 .symbol_id = test_case.symbol_id});

        // Extract values and verify consistency
        const std::uint8_t extracted_frame_id = oran_umsg_get_frame_id(buffer.data());
        const std::uint8_t extracted_subframe_id = oran_umsg_get_subframe_id(buffer.data());
        const std::uint8_t extracted_slot_id = oran_umsg_get_slot_id(buffer.data());
        const std::uint8_t extracted_symbol_id = oran_umsg_get_symbol_id(buffer.data());

        EXPECT_EQ(extracted_frame_id, test_case.frame_id);
        EXPECT_EQ(extracted_subframe_id, test_case.subframe_id);
        EXPECT_EQ(extracted_slot_id, test_case.slot_id);
        EXPECT_EQ(extracted_symbol_id, test_case.symbol_id);
    }
}

/**
 * Test data structure for additional bandwidth function tests
 */
struct AdditionalBandwidthTestData {
    int channel_bandwidth{};
    int scs{};
    int expected_nrb{};
    int expected_guardband{};
    const char *description{};
};

/**
 * Table-based test for additional bandwidth functions
 */
TEST(oran_additional_bandwidth_test, additional_bandwidth_functions) {
    const std::vector<AdditionalBandwidthTestData> test_cases = {
            // Valid combinations that exist in the bandwidthTable (note: table maps
            // NRB->channelBandwidth, not channelBandwidth->NRB)
            // So most channelBandwidth lookups will return -1
            {5, 15, 25, 242, "5 MHz, 15 kHz SCS - should find NRB=25"},
            {10, 15, -1, 312, "10 MHz, 15 kHz SCS - no direct mapping"},
            {20, 15, -1, 452, "20 MHz, 15 kHz SCS - no direct mapping"},
            {10, 30, 24, 665, "10 MHz, 30 kHz SCS - should find NRB=24"},
            {20, 30, 51, 805, "20 MHz, 30 kHz SCS - should find NRB=51"},
            {100, 30, 273, 845, "100 MHz, 30 kHz SCS - should find NRB=273"},
            {10, 60, 11, 1010, "10 MHz, 60 kHz SCS - should find NRB=11"},
            {40, 60, 51, 1610, "40 MHz, 60 kHz SCS - should find NRB=51"},
            {100, 60, 135, 1370, "100 MHz, 60 kHz SCS - should find NRB=135"}};

    for (const auto &test_case : test_cases) {
        SCOPED_TRACE(test_case.description);

        // Test getMaxTransmissionBWNRB function
        EXPECT_EQ(
                getMaxTransmissionBWNRB(test_case.channel_bandwidth, test_case.scs),
                test_case.expected_nrb);

        // Test getGuardband function
        EXPECT_EQ(
                getGuardband(test_case.scs, test_case.channel_bandwidth),
                test_case.expected_guardband);
    }
}

/**
 * Test for invalid bandwidth function inputs
 */
TEST(oran_additional_bandwidth_test, invalid_bandwidth_inputs) {
    // Test getMaxTransmissionBWNRB with invalid inputs
    EXPECT_EQ(getMaxTransmissionBWNRB(999, 15), -1); // Invalid bandwidth
    EXPECT_EQ(getMaxTransmissionBWNRB(20, 45), -1);  // Invalid SCS
    EXPECT_EQ(getMaxTransmissionBWNRB(7, 15), -1);   // Unsupported combination

    // Test getGuardband with invalid inputs
    EXPECT_EQ(getGuardband(45, 20), -1);  // Invalid SCS
    EXPECT_EQ(getGuardband(15, 999), -1); // Invalid bandwidth
    EXPECT_EQ(getGuardband(15, 7), -1);   // Unsupported combination
}

/**
 * Test data for section size validation
 */
struct SectionSizeTestData {
    std::uint8_t section_type{};
    std::uint16_t expected_common_hdr_size{};
    std::uint16_t expected_section_size{};
    const char *description{};
};

/**
 * Test for get_cmsg_common_hdr_size and get_cmsg_section_size with valid types
 */
TEST(CPlaneUtils, section_sizes_valid_types) {
    const std::vector<SectionSizeTestData> test_cases = {
            {ORAN_CMSG_SECTION_TYPE_0,
             sizeof(oran_cmsg_sect0_common_hdr),
             sizeof(oran_cmsg_sect0),
             "Section Type 0"},
            {ORAN_CMSG_SECTION_TYPE_1,
             sizeof(oran_cmsg_sect1_common_hdr),
             sizeof(oran_cmsg_sect1),
             "Section Type 1"},
            {ORAN_CMSG_SECTION_TYPE_3,
             sizeof(oran_cmsg_sect3_common_hdr),
             sizeof(oran_cmsg_sect3),
             "Section Type 3"},
            {ORAN_CMSG_SECTION_TYPE_5,
             sizeof(oran_cmsg_sect5_common_hdr),
             sizeof(oran_cmsg_sect5),
             "Section Type 5"}};

    for (const auto &test_case : test_cases) {
        SCOPED_TRACE(test_case.description);
        EXPECT_EQ(
                ran::oran::get_cmsg_common_hdr_size(test_case.section_type),
                test_case.expected_common_hdr_size);
        EXPECT_EQ(
                ran::oran::get_cmsg_section_size(test_case.section_type),
                test_case.expected_section_size);
    }
}

/**
 * Test for get_cmsg_common_hdr_size and get_cmsg_section_size with invalid types
 */
TEST(CPlaneUtils, section_sizes_invalid_types) {
    const std::vector<std::uint8_t> invalid_types = {
            2,  // Section Type 2 (unsupported)
            4,  // Section Type 4 (unsupported)
            6,  // Out of bounds
            255 // Out of bounds
    };

    for (const auto section_type : invalid_types) {
        SCOPED_TRACE(std::format("Section type {}", section_type));
        EXPECT_THROW(
                { std::ignore = ran::oran::get_cmsg_common_hdr_size(section_type); },
                std::invalid_argument);
        EXPECT_THROW(
                { std::ignore = ran::oran::get_cmsg_section_size(section_type); },
                std::invalid_argument);
    }
}

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers,cppcoreguidelines-pro-type-union-access)

} // namespace
