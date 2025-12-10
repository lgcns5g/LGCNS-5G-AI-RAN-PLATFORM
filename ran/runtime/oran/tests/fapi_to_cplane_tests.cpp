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

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <span>
#include <system_error>
#include <utility>
#include <vector>

#include <aerial-fh-driver/oran.hpp>
#include <scf_5g_fapi.h>

#include <gtest/gtest.h>

#include "fapi/fapi_buffer.hpp"
#include "oran/cplane_types.hpp"
#include "oran/fapi_to_cplane.hpp"
#include "oran/numerology.hpp"
#include "oran/oran_errors.hpp"

namespace {

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers,cppcoreguidelines-pro-type-union-access)

/**
 * Helper to create a basic PUSCH PDU for testing
 */
scf_fapi_pusch_pdu_t create_basic_pusch_pdu() {
    scf_fapi_pusch_pdu_t pdu{};
    std::memset(&pdu, 0, sizeof(pdu));

    // Basic parameters
    pdu.rnti = 0x1234;
    pdu.handle = 0x5678;

    // BWP configuration
    pdu.bwp.bwp_size = 100;
    pdu.bwp.bwp_start = 0;
    pdu.bwp.scs = 1;           // 30 kHz
    pdu.bwp.cyclic_prefix = 0; // Normal CP

    // DMRS configuration
    pdu.dmrs_ports = 0x0001; // Port 0
    pdu.scid = 0;
    pdu.ul_dmrs_sym_pos = 0x08; // Symbol 3

    // Frequency allocation
    pdu.resource_alloc = 1; // Type 1
    pdu.rb_start = 0;
    pdu.rb_size = 50;

    // Time allocation
    pdu.start_symbol_index = 0;
    pdu.num_of_symbols = 14;

    return pdu;
}

/**
 * Helper to create PUSCH PDU matching test vector configuration
 *
 * Simplified version of pusch_test_vector.txt:
 * - Frame 200, Subframe 2, Slot 3
 * - 2 PRB groups: [0-24] and [25-49]
 * - 14 symbols (0-13)
 * - 4 antenna ports (portMask: 0x000F)
 */
scf_fapi_pusch_pdu_t create_pusch_test_vector_pdu() {
    scf_fapi_pusch_pdu_t pdu{};
    std::memset(&pdu, 0, sizeof(pdu));

    // Basic parameters
    pdu.rnti = 0x0100;
    pdu.handle = 0x000001;

    // BWP configuration
    pdu.bwp.bwp_size = 273;
    pdu.bwp.bwp_start = 0;
    pdu.bwp.scs = 1;           // 30 kHz
    pdu.bwp.cyclic_prefix = 0; // Normal CP

    // DMRS configuration - 4 ports (0,1,2,3)
    pdu.dmrs_ports = 0x000F; // Ports 0,1,2,3
    pdu.scid = 0;
    pdu.ul_dmrs_sym_pos = 0x2008; // Symbols 3 and 13

    // Frequency allocation - 50 PRBs total
    pdu.resource_alloc = 1; // Type 1
    pdu.rb_start = 0;
    pdu.rb_size = 50;

    // Time allocation - full slot
    pdu.start_symbol_index = 0;
    pdu.num_of_symbols = 14;

    return pdu;
}

/**
 * Helper to create PUSCH PDU with multiple antenna ports
 */
scf_fapi_pusch_pdu_t
create_multi_port_pusch_pdu(const std::uint16_t dmrs_ports, const std::uint8_t scid) {
    auto pdu = create_basic_pusch_pdu();
    pdu.dmrs_ports = dmrs_ports;
    pdu.scid = scid;
    return pdu;
}

/**
 * Default slot timing for tests
 */
constexpr ran::oran::OranSlotTiming DEFAULT_TEST_SLOT_TIMING{
        .frame_id = 100, .subframe_id = 5, .slot_id = 3};

/**
 * Helper to create default TX windows for tests
 * Symbol duration for 30 kHz SCS: ~35.714 µs = 35714 ns
 */
ran::oran::OranTxWindows create_test_tx_windows(const std::uint64_t base_time = 1000000000ULL) {
    ran::oran::OranTxWindows windows{};
    windows.tx_window_start = base_time;
    windows.tx_window_bfw_start = 0;
    windows.tx_window_end = base_time + 35714; // One symbol duration
    return windows;
}

/**
 * Symbol duration for 30 kHz SCS in nanoseconds
 */
constexpr std::uint64_t SYMBOL_DURATION_30_KHZ_NS = 35714;

/**
 * Create 30 kHz numerology for tests
 */
ran::oran::OranNumerology create_30khz_numerology() {
    return ran::oran::from_scs(ran::oran::SubcarrierSpacing::Scs30Khz);
}

/**
 * Helper to convert a single PUSCH PDU to PRB chunks to C-plane messages
 *
 * This helper wraps the new chunk-based conversion for backward compatibility
 * with tests that used to call convert_pusch_pdu_to_cplane directly.
 */
std::error_code convert_single_pusch_pdu_to_cplane(
        const scf_fapi_pusch_pdu_t &pdu,
        const ran::oran::OranSlotTiming &slot_timing,
        const std::uint16_t num_antenna_ports,
        const ran::oran::OranTxWindows &tx_windows,
        const ran::oran::OranNumerology &numerology,
        std::vector<ran::oran::OranCPlaneMsgInfo> &msg_infos) {
    // Create a single-chunk from the PDU
    ran::oran::PrbChunks chunks{};
    ran::oran::PrbChunk chunk{};
    chunk.rb_start = pdu.rb_start;
    chunk.rb_size = pdu.rb_size;
    chunk.start_symbol_index = pdu.start_symbol_index;
    chunk.num_of_symbols = pdu.num_of_symbols;
    chunks.chunks.push_back(chunk);

    // Convert chunks to C-plane messages
    return ran::oran::convert_prb_chunks_to_cplane(
            chunks, slot_timing, num_antenna_ports, tx_windows, numerology, msg_infos);
}

/**
 * Helper to create a UL_TTI.request with multiple PUSCH PDUs
 *
 * @param[in] pusch_pdus Vector of PUSCH PDUs to include
 * @return Buffer containing the UL_TTI.request
 */
std::vector<std::uint8_t>
create_ul_tti_request_with_multiple_pusch(const std::vector<scf_fapi_pusch_pdu_t> &pusch_pdus) {
    std::vector<std::uint8_t> buffer{};

    // Calculate sizes
    constexpr std::size_t HEADER_SIZE = sizeof(scf_fapi_ul_tti_req_t);
    constexpr std::size_t PDU_HEADER_SIZE = 2 * sizeof(std::uint16_t); // type + size
    constexpr std::size_t PDU_SIZE = sizeof(scf_fapi_pusch_pdu_t);
    const std::size_t total_payload_size = pusch_pdus.size() * (PDU_HEADER_SIZE + PDU_SIZE);
    const std::size_t total_size = HEADER_SIZE + total_payload_size;

    buffer.resize(total_size, 0);

    // Fill UL_TTI.request header
    auto *request = ran::fapi::assume_cast<scf_fapi_ul_tti_req_t>(buffer.data());
    request->sfn = 200;
    request->slot = 23;
    request->num_pdus = static_cast<std::uint8_t>(pusch_pdus.size());
    request->rach_present = 0;
    request->num_ulsch = static_cast<std::uint8_t>(pusch_pdus.size());
    request->num_ulcch = 0;

    // Fill PDUs
    auto payload_span = ran::fapi::make_buffer_span(&request->payload[0], total_payload_size);
    std::size_t offset = 0;
    for (const auto &pusch_pdu : pusch_pdus) {
        // PDU header - use memcpy since payload may not be aligned for uint16_t
        const auto pdu_type = UL_TTI_PDU_TYPE_PUSCH;
        std::memcpy(payload_span.subspan(offset).data(), &pdu_type, sizeof(std::uint16_t));
        offset += sizeof(std::uint16_t);

        const auto pdu_size = static_cast<std::uint16_t>(PDU_SIZE);
        std::memcpy(payload_span.subspan(offset).data(), &pdu_size, sizeof(std::uint16_t));
        offset += sizeof(std::uint16_t);

        // PDU data
        std::memcpy(payload_span.subspan(offset).data(), &pusch_pdu, PDU_SIZE);
        offset += PDU_SIZE;
    }

    return buffer;
}

/**
 * Helper to create a UL_TTI.request with a single PUSCH PDU
 */
std::vector<std::uint8_t> create_ul_tti_request_with_pusch() {
    std::vector<std::uint8_t> buffer{};

    // Calculate sizes
    constexpr std::size_t HEADER_SIZE = sizeof(scf_fapi_ul_tti_req_t);
    constexpr std::size_t PDU_HEADER_SIZE = 2 * sizeof(std::uint16_t); // type + size
    constexpr std::size_t PDU_SIZE = sizeof(scf_fapi_pusch_pdu_t);
    const std::size_t total_size = HEADER_SIZE + PDU_HEADER_SIZE + PDU_SIZE;

    buffer.resize(total_size, 0);

    // Fill UL_TTI.request header
    auto *request = ran::fapi::assume_cast<scf_fapi_ul_tti_req_t>(buffer.data());
    request->sfn = 200;
    request->slot = 23; // Subframe 2, Slot 3 (for 30kHz SCS: slot = subframe*10 + slot_id)
    request->num_pdus = 1;
    request->rach_present = 0;
    request->num_ulsch = 1;
    request->num_ulcch = 0;

    // Fill PDU header - use memcpy since payload may not be aligned for uint16_t
    const std::size_t payload_size = PDU_HEADER_SIZE + PDU_SIZE;
    auto payload_span = ran::fapi::make_buffer_span(&request->payload[0], payload_size);
    std::size_t offset = 0;

    const std::uint16_t pdu_type = UL_TTI_PDU_TYPE_PUSCH;
    std::memcpy(payload_span.subspan(offset).data(), &pdu_type, sizeof(std::uint16_t));
    offset += sizeof(std::uint16_t);

    const auto pdu_size = static_cast<std::uint16_t>(PDU_SIZE);
    std::memcpy(payload_span.subspan(offset).data(), &pdu_size, sizeof(std::uint16_t));
    offset += sizeof(std::uint16_t);

    // Fill PUSCH PDU
    const auto pusch_pdu = create_pusch_test_vector_pdu();
    std::memcpy(payload_span.subspan(offset).data(), &pusch_pdu, PDU_SIZE);

    return buffer;
}

// Tests for PRB chunk aggregation

TEST(FindContiguousPrbChunks, EmptyRequest) {
    // Test with no PUSCH PDUs
    std::vector<std::uint8_t> buffer{};
    constexpr std::size_t HEADER_SIZE = sizeof(scf_fapi_ul_tti_req_t);
    buffer.resize(HEADER_SIZE, 0);

    auto *request = ran::fapi::assume_cast<scf_fapi_ul_tti_req_t>(buffer.data());
    request->num_pdus = 0;
    request->num_ulsch = 0;

    ran::oran::PrbChunks output{};
    const std::size_t body_len = buffer.size() - sizeof(scf_fapi_body_header_t);
    const std::error_code ec = ran::oran::find_contiguous_prb_chunks(*request, body_len, output);

    ASSERT_FALSE(ec) << "Error: " << ec.message();
    EXPECT_EQ(output.chunks.size(), 0U);
}

TEST(FindContiguousPrbChunks, SinglePdu) {
    // Test with single PUSCH PDU
    auto pdu = create_basic_pusch_pdu();
    pdu.rb_start = 10;
    pdu.rb_size = 20;
    pdu.start_symbol_index = 2;
    pdu.num_of_symbols = 12;

    const std::vector<scf_fapi_pusch_pdu_t> pdus = {pdu};
    const auto buffer = create_ul_tti_request_with_multiple_pusch(pdus);
    const auto *request = ran::fapi::assume_cast<const scf_fapi_ul_tti_req_t>(buffer.data());

    ran::oran::PrbChunks output{};
    const std::size_t body_len = buffer.size() - sizeof(scf_fapi_body_header_t);
    const std::error_code ec = ran::oran::find_contiguous_prb_chunks(*request, body_len, output);

    ASSERT_FALSE(ec) << "Error: " << ec.message();
    ASSERT_EQ(output.chunks.size(), 1U);

    // Verify chunk matches the PDU
    EXPECT_EQ(output.chunks.at(0).rb_start, 10);
    EXPECT_EQ(output.chunks.at(0).rb_size, 20);
    EXPECT_EQ(output.chunks.at(0).start_symbol_index, 2);
    EXPECT_EQ(output.chunks.at(0).num_of_symbols, 12);
}

TEST(FindContiguousPrbChunks, TwoContiguousPdus) {
    // Test two PDUs that are contiguous in frequency with same symbol range
    auto pdu1 = create_basic_pusch_pdu();
    pdu1.rb_start = 10;
    pdu1.rb_size = 5;
    pdu1.start_symbol_index = 2;
    pdu1.num_of_symbols = 12;

    auto pdu2 = create_basic_pusch_pdu();
    pdu2.rb_start = 15; // Contiguous with pdu1 (10 + 5 = 15)
    pdu2.rb_size = 10;
    pdu2.start_symbol_index = 2;
    pdu2.num_of_symbols = 12;

    const std::vector<scf_fapi_pusch_pdu_t> pdus = {pdu1, pdu2};
    const auto buffer = create_ul_tti_request_with_multiple_pusch(pdus);
    const auto *request = ran::fapi::assume_cast<const scf_fapi_ul_tti_req_t>(buffer.data());

    ran::oran::PrbChunks output{};
    const std::size_t body_len = buffer.size() - sizeof(scf_fapi_body_header_t);
    const std::error_code ec = ran::oran::find_contiguous_prb_chunks(*request, body_len, output);

    ASSERT_FALSE(ec) << "Error: " << ec.message();
    ASSERT_EQ(output.chunks.size(), 1U); // Merged into one chunk

    // Verify merged chunk
    EXPECT_EQ(output.chunks.at(0).rb_start, 10);
    EXPECT_EQ(output.chunks.at(0).rb_size, 15); // 5 + 10
    EXPECT_EQ(output.chunks.at(0).start_symbol_index, 2);
    EXPECT_EQ(output.chunks.at(0).num_of_symbols, 12);
}

TEST(FindContiguousPrbChunks, TwoNonContiguousPdus) {
    // Test two PDUs with gap in frequency
    auto pdu1 = create_basic_pusch_pdu();
    pdu1.rb_start = 10;
    pdu1.rb_size = 5;
    pdu1.start_symbol_index = 2;
    pdu1.num_of_symbols = 12;

    auto pdu2 = create_basic_pusch_pdu();
    pdu2.rb_start = 20; // Gap of 5 PRBs (10+5=15, gap 15-20)
    pdu2.rb_size = 10;
    pdu2.start_symbol_index = 2;
    pdu2.num_of_symbols = 12;

    const std::vector<scf_fapi_pusch_pdu_t> pdus = {pdu1, pdu2};
    const auto buffer = create_ul_tti_request_with_multiple_pusch(pdus);
    const auto *request = ran::fapi::assume_cast<const scf_fapi_ul_tti_req_t>(buffer.data());

    ran::oran::PrbChunks output{};
    const std::size_t body_len = buffer.size() - sizeof(scf_fapi_body_header_t);
    const std::error_code ec = ran::oran::find_contiguous_prb_chunks(*request, body_len, output);

    ASSERT_FALSE(ec) << "Error: " << ec.message();
    ASSERT_EQ(output.chunks.size(), 2U); // Two separate chunks

    // Verify first chunk
    EXPECT_EQ(output.chunks.at(0).rb_start, 10);
    EXPECT_EQ(output.chunks.at(0).rb_size, 5);
    EXPECT_EQ(output.chunks.at(0).start_symbol_index, 2);
    EXPECT_EQ(output.chunks.at(0).num_of_symbols, 12);

    // Verify second chunk
    EXPECT_EQ(output.chunks.at(1).rb_start, 20);
    EXPECT_EQ(output.chunks.at(1).rb_size, 10);
    EXPECT_EQ(output.chunks.at(1).start_symbol_index, 2);
    EXPECT_EQ(output.chunks.at(1).num_of_symbols, 12);
}

TEST(FindContiguousPrbChunks, DifferentSymbolRanges) {
    // Test PDUs with different symbol allocations (not contiguous)
    auto pdu1 = create_basic_pusch_pdu();
    pdu1.rb_start = 10;
    pdu1.rb_size = 8;
    pdu1.start_symbol_index = 0;
    pdu1.num_of_symbols = 2;

    auto pdu2 = create_basic_pusch_pdu();
    pdu2.rb_start = 10;
    pdu2.rb_size = 15;
    pdu2.start_symbol_index = 2;
    pdu2.num_of_symbols = 12;

    const std::vector<scf_fapi_pusch_pdu_t> pdus = {pdu1, pdu2};
    const auto buffer = create_ul_tti_request_with_multiple_pusch(pdus);
    const auto *request = ran::fapi::assume_cast<const scf_fapi_ul_tti_req_t>(buffer.data());

    ran::oran::PrbChunks output{};
    const std::size_t body_len = buffer.size() - sizeof(scf_fapi_body_header_t);
    const std::error_code ec = ran::oran::find_contiguous_prb_chunks(*request, body_len, output);

    ASSERT_FALSE(ec) << "Error: " << ec.message();
    ASSERT_EQ(output.chunks.size(), 2U); // Cannot merge different symbol ranges

    // Chunks are sorted by symbol range, so pdu1 should come first
    EXPECT_EQ(output.chunks.at(0).rb_start, 10);
    EXPECT_EQ(output.chunks.at(0).rb_size, 8);
    EXPECT_EQ(output.chunks.at(0).start_symbol_index, 0);
    EXPECT_EQ(output.chunks.at(0).num_of_symbols, 2);

    EXPECT_EQ(output.chunks.at(1).rb_start, 10);
    EXPECT_EQ(output.chunks.at(1).rb_size, 15);
    EXPECT_EQ(output.chunks.at(1).start_symbol_index, 2);
    EXPECT_EQ(output.chunks.at(1).num_of_symbols, 12);
}

TEST(FindContiguousPrbChunks, ThreeContiguousPdus) {
    // Test three PDUs all contiguous
    auto pdu1 = create_basic_pusch_pdu();
    pdu1.rb_start = 10;
    pdu1.rb_size = 5;
    pdu1.start_symbol_index = 2;
    pdu1.num_of_symbols = 12;

    auto pdu2 = create_basic_pusch_pdu();
    pdu2.rb_start = 15;
    pdu2.rb_size = 10;
    pdu2.start_symbol_index = 2;
    pdu2.num_of_symbols = 12;

    auto pdu3 = create_basic_pusch_pdu();
    pdu3.rb_start = 25;
    pdu3.rb_size = 8;
    pdu3.start_symbol_index = 2;
    pdu3.num_of_symbols = 12;

    const std::vector<scf_fapi_pusch_pdu_t> pdus = {pdu1, pdu2, pdu3};
    const auto buffer = create_ul_tti_request_with_multiple_pusch(pdus);
    const auto *request = ran::fapi::assume_cast<const scf_fapi_ul_tti_req_t>(buffer.data());

    ran::oran::PrbChunks output{};
    const std::size_t body_len = buffer.size() - sizeof(scf_fapi_body_header_t);
    const std::error_code ec = ran::oran::find_contiguous_prb_chunks(*request, body_len, output);

    ASSERT_FALSE(ec) << "Error: " << ec.message();
    ASSERT_EQ(output.chunks.size(), 1U); // All merged

    // Verify merged chunk
    EXPECT_EQ(output.chunks.at(0).rb_start, 10);
    EXPECT_EQ(output.chunks.at(0).rb_size, 23); // 5 + 10 + 8
    EXPECT_EQ(output.chunks.at(0).start_symbol_index, 2);
    EXPECT_EQ(output.chunks.at(0).num_of_symbols, 12);
}

TEST(FindContiguousPrbChunks, ComplexMixedScenario) {
    // Complex scenario with multiple symbol ranges and mixed contiguity
    auto pdu1 = create_basic_pusch_pdu();
    pdu1.rb_start = 10;
    pdu1.rb_size = 5;
    pdu1.start_symbol_index = 2;
    pdu1.num_of_symbols = 12;

    auto pdu2 = create_basic_pusch_pdu();
    pdu2.rb_start = 15;
    pdu2.rb_size = 10;
    pdu2.start_symbol_index = 2;
    pdu2.num_of_symbols = 12;

    auto pdu3 = create_basic_pusch_pdu();
    pdu3.rb_start = 30;
    pdu3.rb_size = 5;
    pdu3.start_symbol_index = 2;
    pdu3.num_of_symbols = 12;

    auto pdu4 = create_basic_pusch_pdu();
    pdu4.rb_start = 10;
    pdu4.rb_size = 8;
    pdu4.start_symbol_index = 0;
    pdu4.num_of_symbols = 2;

    const std::vector<scf_fapi_pusch_pdu_t> pdus = {pdu1, pdu2, pdu3, pdu4};
    const auto buffer = create_ul_tti_request_with_multiple_pusch(pdus);
    const auto *request = ran::fapi::assume_cast<const scf_fapi_ul_tti_req_t>(buffer.data());

    ran::oran::PrbChunks output{};
    const std::size_t body_len = buffer.size() - sizeof(scf_fapi_body_header_t);
    const std::error_code ec = ran::oran::find_contiguous_prb_chunks(*request, body_len, output);

    ASSERT_FALSE(ec) << "Error: " << ec.message();
    ASSERT_EQ(output.chunks.size(), 3U);

    // Chunk 1: pdu4 (symbols 0-2)
    EXPECT_EQ(output.chunks.at(0).rb_start, 10);
    EXPECT_EQ(output.chunks.at(0).rb_size, 8);
    EXPECT_EQ(output.chunks.at(0).start_symbol_index, 0);
    EXPECT_EQ(output.chunks.at(0).num_of_symbols, 2);

    // Chunk 2: pdu1+pdu2 merged (symbols 2-14)
    EXPECT_EQ(output.chunks.at(1).rb_start, 10);
    EXPECT_EQ(output.chunks.at(1).rb_size, 15);
    EXPECT_EQ(output.chunks.at(1).start_symbol_index, 2);
    EXPECT_EQ(output.chunks.at(1).num_of_symbols, 12);

    // Chunk 3: pdu3 alone (symbols 2-14, different frequency)
    EXPECT_EQ(output.chunks.at(2).rb_start, 30);
    EXPECT_EQ(output.chunks.at(2).rb_size, 5);
    EXPECT_EQ(output.chunks.at(2).start_symbol_index, 2);
    EXPECT_EQ(output.chunks.at(2).num_of_symbols, 12);
}

TEST(FindContiguousPrbChunks, UnsortedInput) {
    // Test that PDUs are properly sorted before merging
    // pdu1: 25-29, pdu2: 10-14, pdu3: 15-24
    // After sort: 10-14, 15-24, 25-29 (all contiguous!)
    auto pdu1 = create_basic_pusch_pdu();
    pdu1.rb_start = 25;
    pdu1.rb_size = 5;
    pdu1.start_symbol_index = 2;
    pdu1.num_of_symbols = 12;

    auto pdu2 = create_basic_pusch_pdu();
    pdu2.rb_start = 10; // Lower start, should be sorted first
    pdu2.rb_size = 5;
    pdu2.start_symbol_index = 2;
    pdu2.num_of_symbols = 12;

    auto pdu3 = create_basic_pusch_pdu();
    pdu3.rb_start = 15; // Middle, contiguous with both pdu2 and pdu1
    pdu3.rb_size = 10;
    pdu3.start_symbol_index = 2;
    pdu3.num_of_symbols = 12;

    const std::vector<scf_fapi_pusch_pdu_t> pdus = {pdu1, pdu2, pdu3};
    const auto buffer = create_ul_tti_request_with_multiple_pusch(pdus);
    const auto *request = ran::fapi::assume_cast<const scf_fapi_ul_tti_req_t>(buffer.data());

    ran::oran::PrbChunks output{};
    const std::size_t body_len = buffer.size() - sizeof(scf_fapi_body_header_t);
    const std::error_code ec = ran::oran::find_contiguous_prb_chunks(*request, body_len, output);

    ASSERT_FALSE(ec) << "Error: " << ec.message();
    ASSERT_EQ(output.chunks.size(), 1U); // All 3 PDUs merge into one chunk

    // Single chunk: all merged (10+5=15, 15+10=25, 25+5=30)
    EXPECT_EQ(output.chunks.at(0).rb_start, 10);
    EXPECT_EQ(output.chunks.at(0).rb_size, 20); // 5 + 10 + 5
    EXPECT_EQ(output.chunks.at(0).start_symbol_index, 2);
    EXPECT_EQ(output.chunks.at(0).num_of_symbols, 12);
}

TEST(FindContiguousPrbChunks, ClearsPreviousOutput) {
    // Test that output is cleared before processing
    auto pdu = create_basic_pusch_pdu();
    pdu.rb_start = 10;
    pdu.rb_size = 20;
    pdu.start_symbol_index = 2;
    pdu.num_of_symbols = 12;

    const std::vector<scf_fapi_pusch_pdu_t> pdus = {pdu};
    const auto buffer = create_ul_tti_request_with_multiple_pusch(pdus);
    const auto *request = ran::fapi::assume_cast<const scf_fapi_ul_tti_req_t>(buffer.data());

    ran::oran::PrbChunks output{};

    // Pre-populate output with dummy data
    ran::oran::PrbChunk dummy{};
    dummy.rb_start = 999;
    output.chunks.push_back(dummy);
    output.chunks.push_back(dummy);

    EXPECT_EQ(output.chunks.size(), 2U); // Verify pre-population

    const std::size_t body_len = buffer.size() - sizeof(scf_fapi_body_header_t);
    const std::error_code ec = ran::oran::find_contiguous_prb_chunks(*request, body_len, output);

    ASSERT_FALSE(ec) << "Error: " << ec.message();
    ASSERT_EQ(output.chunks.size(), 1U); // Should only have new data

    EXPECT_EQ(output.chunks.at(0).rb_start, 10);
    EXPECT_NE(output.chunks.at(0).rb_start, 999);
}

TEST(FindContiguousPrbChunks, EdgeCaseMinimalAllocation) {
    // Test minimum allocation: 1 PRB, 1 symbol
    auto pdu = create_basic_pusch_pdu();
    pdu.rb_start = 50;
    pdu.rb_size = 1;
    pdu.start_symbol_index = 7;
    pdu.num_of_symbols = 1;

    const std::vector<scf_fapi_pusch_pdu_t> pdus = {pdu};
    const auto buffer = create_ul_tti_request_with_multiple_pusch(pdus);
    const auto *request = ran::fapi::assume_cast<const scf_fapi_ul_tti_req_t>(buffer.data());

    ran::oran::PrbChunks output{};
    const std::size_t body_len = buffer.size() - sizeof(scf_fapi_body_header_t);
    const std::error_code ec = ran::oran::find_contiguous_prb_chunks(*request, body_len, output);

    ASSERT_FALSE(ec) << "Error: " << ec.message();
    ASSERT_EQ(output.chunks.size(), 1U);

    EXPECT_EQ(output.chunks.at(0).rb_start, 50);
    EXPECT_EQ(output.chunks.at(0).rb_size, 1);
    EXPECT_EQ(output.chunks.at(0).start_symbol_index, 7);
    EXPECT_EQ(output.chunks.at(0).num_of_symbols, 1);
}

TEST(FindContiguousPrbChunks, EdgeCaseMaximalAllocation) {
    // Test large allocation: 273 PRBs (max for 100 MHz @ 30 kHz), 14 symbols
    auto pdu = create_basic_pusch_pdu();
    pdu.bwp.bwp_size = 273;
    pdu.rb_start = 0;
    pdu.rb_size = 273;
    pdu.start_symbol_index = 0;
    pdu.num_of_symbols = 14;

    const std::vector<scf_fapi_pusch_pdu_t> pdus = {pdu};
    const auto buffer = create_ul_tti_request_with_multiple_pusch(pdus);
    const auto *request = ran::fapi::assume_cast<const scf_fapi_ul_tti_req_t>(buffer.data());

    ran::oran::PrbChunks output{};
    const std::size_t body_len = buffer.size() - sizeof(scf_fapi_body_header_t);
    const std::error_code ec = ran::oran::find_contiguous_prb_chunks(*request, body_len, output);

    ASSERT_FALSE(ec) << "Error: " << ec.message();
    ASSERT_EQ(output.chunks.size(), 1U);

    EXPECT_EQ(output.chunks.at(0).rb_start, 0);
    EXPECT_EQ(output.chunks.at(0).rb_size, 273);
    EXPECT_EQ(output.chunks.at(0).start_symbol_index, 0);
    EXPECT_EQ(output.chunks.at(0).num_of_symbols, 14);
}

TEST(FindContiguousPrbChunks, SameStartDifferentSymbols) {
    // Test PDUs starting at same PRB but different symbols
    auto pdu1 = create_basic_pusch_pdu();
    pdu1.rb_start = 10;
    pdu1.rb_size = 20;
    pdu1.start_symbol_index = 0;
    pdu1.num_of_symbols = 7;

    auto pdu2 = create_basic_pusch_pdu();
    pdu2.rb_start = 10;          // Same start
    pdu2.rb_size = 20;           // Same size
    pdu2.start_symbol_index = 7; // Different symbols
    pdu2.num_of_symbols = 7;

    const std::vector<scf_fapi_pusch_pdu_t> pdus = {pdu1, pdu2};
    const auto buffer = create_ul_tti_request_with_multiple_pusch(pdus);
    const auto *request = ran::fapi::assume_cast<const scf_fapi_ul_tti_req_t>(buffer.data());

    ran::oran::PrbChunks output{};
    const std::size_t body_len = buffer.size() - sizeof(scf_fapi_body_header_t);
    const std::error_code ec = ran::oran::find_contiguous_prb_chunks(*request, body_len, output);

    ASSERT_FALSE(ec) << "Error: " << ec.message();
    ASSERT_EQ(output.chunks.size(), 2U); // Cannot merge different symbol ranges

    EXPECT_EQ(output.chunks.at(0).start_symbol_index, 0);
    EXPECT_EQ(output.chunks.at(1).start_symbol_index, 7);
}

TEST(FindContiguousPrbChunks, ManyPdus) {
    // Test with many PDUs (stress test for stack allocation)
    std::vector<scf_fapi_pusch_pdu_t> pdus{};

    // Create 50 PDUs, every 2 are contiguous
    for (std::size_t i = 0; i < 50; ++i) {
        auto pdu = create_basic_pusch_pdu();
        pdu.rb_start = static_cast<std::uint16_t>(i * 5);
        pdu.rb_size = 5;
        pdu.start_symbol_index = 2;
        pdu.num_of_symbols = 12;
        pdus.push_back(pdu);
    }

    const auto buffer = create_ul_tti_request_with_multiple_pusch(pdus);
    const auto *request = ran::fapi::assume_cast<const scf_fapi_ul_tti_req_t>(buffer.data());

    ran::oran::PrbChunks output{};
    const std::size_t body_len = buffer.size() - sizeof(scf_fapi_body_header_t);
    const std::error_code ec = ran::oran::find_contiguous_prb_chunks(*request, body_len, output);

    ASSERT_FALSE(ec) << "Error: " << ec.message();
    ASSERT_EQ(output.chunks.size(), 1U); // All merged (contiguous 0-245)

    EXPECT_EQ(output.chunks.at(0).rb_start, 0);
    EXPECT_EQ(output.chunks.at(0).rb_size, 250); // 50 * 5
    EXPECT_EQ(output.chunks.at(0).start_symbol_index, 2);
    EXPECT_EQ(output.chunks.at(0).num_of_symbols, 12);
}

TEST(FindContiguousPrbChunks, PayloadOutOfBounds) {
    // Test with payload_size smaller than actual data
    auto pdu = create_basic_pusch_pdu();
    pdu.rb_start = 10;
    pdu.rb_size = 20;
    pdu.start_symbol_index = 2;
    pdu.num_of_symbols = 12;

    const std::vector<scf_fapi_pusch_pdu_t> pdus = {pdu};
    const auto buffer = create_ul_tti_request_with_multiple_pusch(pdus);
    const auto *request = ran::fapi::assume_cast<const scf_fapi_ul_tti_req_t>(buffer.data());

    ran::oran::PrbChunks output{};
    // Intentionally provide a payload_size that is too small
    const std::size_t invalid_payload_size = 2; // Only enough for PDU header type field
    const std::error_code ec =
            ran::oran::find_contiguous_prb_chunks(*request, invalid_payload_size, output);

    ASSERT_TRUE(ec) << "Expected error for out-of-bounds payload";
    EXPECT_EQ(ec, ran::oran::OranErrc::PduPayloadOutOfBounds);
}

TEST(FindContiguousPrbChunks, PayloadBoundsExactSize) {
    // Test with exact payload size (should succeed)
    auto pdu = create_basic_pusch_pdu();
    pdu.rb_start = 10;
    pdu.rb_size = 20;

    const std::vector<scf_fapi_pusch_pdu_t> pdus = {pdu};
    const auto buffer = create_ul_tti_request_with_multiple_pusch(pdus);
    const auto *request = ran::fapi::assume_cast<const scf_fapi_ul_tti_req_t>(buffer.data());

    ran::oran::PrbChunks output{};
    const std::size_t body_len = buffer.size() - sizeof(scf_fapi_body_header_t);
    const std::error_code ec = ran::oran::find_contiguous_prb_chunks(*request, body_len, output);

    ASSERT_FALSE(ec) << "Error: " << ec.message();
    EXPECT_EQ(output.chunks.size(), 1U);
}

// Tests for individual PDU conversion

TEST(FapiToCPlane, ConvertPuschBasic) {
    const auto pdu = create_basic_pusch_pdu(); // 14 symbols, dmrs_ports=0x0001 (1 port)
    std::vector<ran::oran::OranCPlaneMsgInfo> msg_infos{};
    const auto slot_timing =
            ran::oran::OranSlotTiming{.frame_id = 42, .subframe_id = 5, .slot_id = 3};
    const auto tx_windows = create_test_tx_windows();

    const std::error_code result = convert_single_pusch_pdu_to_cplane(
            pdu, slot_timing, 4, tx_windows, create_30khz_numerology(), msg_infos);

    ASSERT_FALSE(result) << "Error: " << result.message();

    // Should create 4 messages (1 per antenna port, each covering 14 symbols)
    ASSERT_EQ(msg_infos.size(), 4U);

    // Verify first message (antenna port 0)
    const auto &msg_info = msg_infos.at(0);

    // Verify radio application header
    const auto &radio_hdr = msg_info.section_common_hdr.sect_1_common_hdr.radioAppHdr;
    EXPECT_EQ(radio_hdr.frameId, 42);
    EXPECT_EQ(radio_hdr.subframeId.get(), 5U);
    EXPECT_EQ(radio_hdr.slotId.get(), 3U);
    EXPECT_EQ(radio_hdr.startSymbolId.get(), 0U);
    EXPECT_EQ(radio_hdr.sectionType, ORAN_CMSG_SECTION_TYPE_1);
    EXPECT_EQ(radio_hdr.dataDirection.get(), static_cast<std::uint8_t>(DIRECTION_UPLINK));

    // Verify message properties
    EXPECT_EQ(msg_info.data_direction, DIRECTION_UPLINK);
    EXPECT_FALSE(msg_info.has_section_ext);
    EXPECT_EQ(msg_info.num_sections, 1);
    EXPECT_EQ(msg_info.ap_idx, 0U);

    // Verify section content
    const auto &section = msg_info.sections.at(0).sect_1;
    EXPECT_EQ(section.sectionId.get(), 0U);
    EXPECT_EQ(section.startPrbc.get(), 0U);
    EXPECT_EQ(section.numPrbc.get(), 50U);
    EXPECT_EQ(section.reMask.get(), 0x0FFFU);
    EXPECT_EQ(section.numSymbol.get(), 14U); // All 14 symbols in one message

    // Verify last message (antenna port 3)
    const auto &last_msg = msg_infos.at(3);
    const auto &last_radio_hdr = last_msg.section_common_hdr.sect_1_common_hdr.radioAppHdr;
    EXPECT_EQ(last_radio_hdr.startSymbolId.get(), 0U); // Same start symbol
    EXPECT_EQ(last_msg.ap_idx, 3U);
    EXPECT_EQ(last_msg.sections.at(0).sect_1.numSymbol.get(), 14U);
}

TEST(FapiToCPlane, PuschMultipleAntennaPorts) {
    // Test with different num_antenna_ports configurations
    // Note: dmrs_ports is no longer used for port filtering

    // 4 antenna ports, 14 symbols
    {
        const auto pdu = create_multi_port_pusch_pdu(0x000F, 0); // 14 symbols
        std::vector<ran::oran::OranCPlaneMsgInfo> msg_infos{};
        const auto slot_timing =
                ran::oran::OranSlotTiming{.frame_id = 0, .subframe_id = 0, .slot_id = 0};
        const auto tx_windows = create_test_tx_windows();

        const std::error_code result = convert_single_pusch_pdu_to_cplane(
                pdu, slot_timing, 4, tx_windows, create_30khz_numerology(), msg_infos);

        ASSERT_FALSE(result) << "Error: " << result.message();
        // Should create 4 messages (1 per antenna port, each covering 14 symbols)
        EXPECT_EQ(msg_infos.size(), 4U);
    }

    // 16 antenna ports, 14 symbols
    {
        const auto pdu = create_multi_port_pusch_pdu(0x000F, 1); // 14 symbols
        std::vector<ran::oran::OranCPlaneMsgInfo> msg_infos{};
        const auto slot_timing =
                ran::oran::OranSlotTiming{.frame_id = 0, .subframe_id = 0, .slot_id = 0};
        const auto tx_windows = create_test_tx_windows();

        const std::error_code result = convert_single_pusch_pdu_to_cplane(
                pdu, slot_timing, 16, tx_windows, create_30khz_numerology(), msg_infos);

        ASSERT_FALSE(result) << "Error: " << result.message();
        // Should create 16 messages (1 per antenna port, each covering 14 symbols)
        EXPECT_EQ(msg_infos.size(), 16U);
    }

    // 8 antenna ports, 14 symbols
    {
        const auto pdu = create_multi_port_pusch_pdu(0x1001, 0); // 14 symbols
        std::vector<ran::oran::OranCPlaneMsgInfo> msg_infos{};
        const auto slot_timing =
                ran::oran::OranSlotTiming{.frame_id = 0, .subframe_id = 0, .slot_id = 0};
        const auto tx_windows = create_test_tx_windows();

        const std::error_code result = convert_single_pusch_pdu_to_cplane(
                pdu, slot_timing, 8, tx_windows, create_30khz_numerology(), msg_infos);

        ASSERT_FALSE(result) << "Error: " << result.message();
        // Should create 8 messages (1 per antenna port, each covering 14 symbols)
        EXPECT_EQ(msg_infos.size(), 8U);
    }
}

TEST(FapiToCPlane, PuschTestVectorMatch) {
    // Test conversion matches expected output from test vector
    // Frame 200, Subframe 2, Slot 3, 14 symbols, 4 antenna ports
    // Test vector shows: dmrs_ports=0x000F, scid=0 → 56 messages total
    const auto pdu = create_pusch_test_vector_pdu(); // 14 symbols
    std::vector<ran::oran::OranCPlaneMsgInfo> msg_infos{};
    const auto slot_timing =
            ran::oran::OranSlotTiming{.frame_id = 200, .subframe_id = 2, .slot_id = 3};
    const auto tx_windows = create_test_tx_windows();

    const std::error_code result = convert_single_pusch_pdu_to_cplane(
            pdu, slot_timing, 4, tx_windows, create_30khz_numerology(), msg_infos);

    ASSERT_FALSE(result) << "Error: " << result.message();

    // Test vector: dmrs_ports=0x000F, scid=0 → portMask=0x000F (4 ports: 0,1,2,3)
    // Should create 4 messages (1 per antenna port, each covering 14 symbols)
    ASSERT_EQ(msg_infos.size(), 4U);

    // Verify first message (antenna port 0)
    const auto &msg_info = msg_infos.at(0);

    // Verify timing matches test vector
    const auto &radio_hdr = msg_info.section_common_hdr.sect_1_common_hdr.radioAppHdr;
    EXPECT_EQ(radio_hdr.frameId, 200);
    EXPECT_EQ(radio_hdr.subframeId.get(), 2U);
    EXPECT_EQ(radio_hdr.slotId.get(), 3U);
    EXPECT_EQ(radio_hdr.startSymbolId.get(), 0U);

    // Verify PRB allocation
    const auto &section = msg_infos.at(0).sections.at(0).sect_1;
    EXPECT_EQ(section.startPrbc.get(), 0U);
    EXPECT_EQ(section.numPrbc.get(), 50U);

    // Verify all 14 symbols covered by this message
    EXPECT_EQ(section.numSymbol.get(), 14U);

    // Verify section type and direction
    EXPECT_EQ(radio_hdr.sectionType, ORAN_CMSG_SECTION_TYPE_1);
    EXPECT_EQ(msg_info.data_direction, DIRECTION_UPLINK);
}

TEST(FapiToCPlane, PuschVariousPrbAllocations) {
    // Test different PRB allocations
    const std::array<std::pair<std::uint16_t, std::uint16_t>, 4> test_cases = {{
            {0, 25},   // Start at 0, 25 PRBs
            {25, 25},  // Start at 25, 25 PRBs
            {50, 50},  // Start at 50, 50 PRBs
            {100, 73}, // Start at 100, 73 PRBs
    }};

    for (const auto &[rb_start, rb_size] : test_cases) {
        auto pdu = create_basic_pusch_pdu();
        pdu.bwp.bwp_size = 273; // Ensure BWP is large enough
        pdu.rb_start = rb_start;
        pdu.rb_size = rb_size;

        std::vector<ran::oran::OranCPlaneMsgInfo> msg_infos{};
        const auto slot_timing =
                ran::oran::OranSlotTiming{.frame_id = 0, .subframe_id = 0, .slot_id = 0};
        const auto tx_windows = create_test_tx_windows();
        const std::error_code result = convert_single_pusch_pdu_to_cplane(
                pdu, slot_timing, 4, tx_windows, create_30khz_numerology(), msg_infos);

        ASSERT_FALSE(result) << "Error: " << result.message();
        ASSERT_GE(msg_infos.size(), 1U);
        EXPECT_EQ(msg_infos.at(0).sections.at(0).sect_1.startPrbc.get(), rb_start);
        EXPECT_EQ(msg_infos.at(0).sections.at(0).sect_1.numPrbc.get(), rb_size);
    }
}

TEST(FapiToCPlane, PuschVariousSymbolAllocations) {
    // Test different symbol allocations
    const std::array<std::pair<std::uint8_t, std::uint8_t>, 4> test_cases = {{
            {0, 14}, // Full slot
            {0, 7},  // First half
            {7, 7},  // Second half
            {2, 10}, // Middle symbols
    }};

    for (const auto &[start_symbol, num_symbols] : test_cases) {
        auto pdu = create_basic_pusch_pdu();
        pdu.start_symbol_index = start_symbol;
        pdu.num_of_symbols = num_symbols;

        std::vector<ran::oran::OranCPlaneMsgInfo> msg_infos{};
        const auto slot_timing =
                ran::oran::OranSlotTiming{.frame_id = 0, .subframe_id = 0, .slot_id = 0};
        const auto tx_windows = create_test_tx_windows();
        const std::error_code result = convert_single_pusch_pdu_to_cplane(
                pdu, slot_timing, 4, tx_windows, create_30khz_numerology(), msg_infos);

        ASSERT_FALSE(result) << "Error: " << result.message();

        // 4 antenna ports (each message covers all symbols)
        ASSERT_EQ(msg_infos.size(), 4U);

        // First message should start at start_symbol and cover all symbols
        EXPECT_EQ(
                msg_infos.at(0)
                        .section_common_hdr.sect_1_common_hdr.radioAppHdr.startSymbolId.get(),
                start_symbol);

        // Each message covers all symbols in the chunk
        EXPECT_EQ(msg_infos.at(0).sections.at(0).sect_1.numSymbol.get(), num_symbols);

        // All messages should have the same start symbol (they differ by antenna
        // port)
        const auto &last_radio_hdr =
                msg_infos.at(3).section_common_hdr.sect_1_common_hdr.radioAppHdr;
        EXPECT_EQ(last_radio_hdr.startSymbolId.get(), start_symbol);
        EXPECT_EQ(msg_infos.at(3).sections.at(0).sect_1.numSymbol.get(), num_symbols);
        EXPECT_EQ(msg_infos.at(3).ap_idx, 3U);
    }
}

// Tests for UL_TTI.request conversion

TEST(FapiToCPlane, ConvertUlTtiRequestBasic) {
    const auto request_buffer = create_ul_tti_request_with_pusch();
    const auto *request =
            ran::fapi::assume_cast<const scf_fapi_ul_tti_req_t>(request_buffer.data());

    std::vector<ran::oran::OranCPlaneMsgInfo> msg_infos{};
    const auto numerology = create_30khz_numerology();
    const auto tx_windows = create_test_tx_windows();
    const std::size_t body_len = request_buffer.size() - sizeof(scf_fapi_body_header_t);
    const std::error_code ec = ran::oran::convert_ul_tti_request_to_cplane(
            *request, body_len, 4, numerology, tx_windows, msg_infos);

    ASSERT_FALSE(ec) << "Error: " << ec.message();

    // UL_TTI.request contains 1 PUSCH PDU with dmrs_ports=0x000F (4 active
    // ports), 14 symbols. Should create 4 messages (1 per antenna port)
    EXPECT_EQ(msg_infos.size(), 4U);

    // Verify the first converted message
    const auto &msg_info = msg_infos.at(0);
    EXPECT_EQ(msg_info.data_direction, DIRECTION_UPLINK);
    EXPECT_EQ(msg_info.section_common_hdr.sect_1_common_hdr.radioAppHdr.frameId, 200);
}

TEST(FapiToCPlane, ConvertUlTtiRequestMultiplePdus) {
    // Create request with multiple PDUs (when other converters are implemented)
    // For now, just test with single PUSCH
    const auto request_buffer = create_ul_tti_request_with_pusch();
    const auto *request =
            ran::fapi::assume_cast<const scf_fapi_ul_tti_req_t>(request_buffer.data());

    std::vector<ran::oran::OranCPlaneMsgInfo> msg_infos{};
    const auto numerology = create_30khz_numerology();
    const auto tx_windows = create_test_tx_windows();
    const std::size_t body_len = request_buffer.size() - sizeof(scf_fapi_body_header_t);
    const std::error_code ec = ran::oran::convert_ul_tti_request_to_cplane(
            *request, body_len, 4, numerology, tx_windows, msg_infos);

    ASSERT_FALSE(ec) << "Error: " << ec.message();
    // With 1 PUSCH PDU: 4 messages (1 per antenna port, each covers 14 symbols)
    EXPECT_EQ(msg_infos.size(), 4U);
}

TEST(FapiToCPlane, ConvertUlTtiRequestExtractsTimingCorrectly) {
    // Test that frame/subframe/slot are extracted correctly
    // For 30kHz SCS: 2 slots per subframe, so slot = subframe*2 + slot_id
    // FAPI slot 23 = subframe 11 * 2 + slot 1 = 23
    const auto request_buffer = create_ul_tti_request_with_pusch();
    const auto *request =
            ran::fapi::assume_cast<const scf_fapi_ul_tti_req_t>(request_buffer.data());

    std::vector<ran::oran::OranCPlaneMsgInfo> msg_infos{};
    const auto numerology = create_30khz_numerology();
    const auto tx_windows = create_test_tx_windows();
    const std::size_t body_len = request_buffer.size() - sizeof(scf_fapi_body_header_t);
    const std::error_code ec = ran::oran::convert_ul_tti_request_to_cplane(
            *request, body_len, 4, numerology, tx_windows, msg_infos);

    ASSERT_FALSE(ec) << "Error: " << ec.message();
    // With 4 messages (1 per antenna port, each covers 14 symbols)
    ASSERT_EQ(msg_infos.size(), 4U);

    // All messages should have the same frame/subframe/slot timing
    const auto &radio_hdr = msg_infos.at(0).section_common_hdr.sect_1_common_hdr.radioAppHdr;
    EXPECT_EQ(radio_hdr.frameId, 200);
    EXPECT_EQ(radio_hdr.subframeId.get(), 11U); // 23 / 2 = 11
    EXPECT_EQ(radio_hdr.slotId.get(), 1U);      // 23 % 2 = 1
}

TEST(FapiToCPlane, PuschAppendsToVector) {
    // Test that convert_pusch_pdu_to_cplane appends to the vector
    std::vector<ran::oran::OranCPlaneMsgInfo> msg_infos{};
    const auto slot_timing =
            ran::oran::OranSlotTiming{.frame_id = 0, .subframe_id = 0, .slot_id = 0};
    const auto tx_windows = create_test_tx_windows();

    // First conversion: 4 antenna ports = 4 messages (each covers all symbols)
    const auto pdu1 = create_basic_pusch_pdu();
    const std::error_code result1 = convert_single_pusch_pdu_to_cplane(
            pdu1, slot_timing, 4, tx_windows, create_30khz_numerology(), msg_infos);
    ASSERT_FALSE(result1) << "Error: " << result1.message();
    EXPECT_EQ(msg_infos.size(), 4U);

    // Second conversion: should append 4 more messages
    const auto pdu2 = create_basic_pusch_pdu();
    const std::error_code result2 = convert_single_pusch_pdu_to_cplane(
            pdu2, slot_timing, 4, tx_windows, create_30khz_numerology(), msg_infos);
    ASSERT_FALSE(result2) << "Error: " << result2.message();
    EXPECT_EQ(msg_infos.size(), 8U);

    // Verify both sets of messages are present
    // First message of first PDU (port 0)
    EXPECT_EQ(
            msg_infos.at(0).section_common_hdr.sect_1_common_hdr.radioAppHdr.startSymbolId.get(),
            0U);
    EXPECT_EQ(msg_infos.at(0).ap_idx, 0U);
    // First message of second PDU (port 0) at index 4
    EXPECT_EQ(
            msg_infos.at(4).section_common_hdr.sect_1_common_hdr.radioAppHdr.startSymbolId.get(),
            0U);
    EXPECT_EQ(msg_infos.at(4).ap_idx, 0U);
}

TEST(FapiToCPlane, UlTtiRequestClearsVector) {
    // Test that convert_ul_tti_request_to_cplane clears the vector before use
    const auto request_buffer = create_ul_tti_request_with_pusch();
    const auto *request =
            ran::fapi::assume_cast<const scf_fapi_ul_tti_req_t>(request_buffer.data());
    const auto numerology = create_30khz_numerology();
    const auto tx_windows = create_test_tx_windows();

    std::vector<ran::oran::OranCPlaneMsgInfo> msg_infos{};

    // Pre-populate vector with some data
    msg_infos.resize(100);

    // Conversion should clear and repopulate
    const std::size_t body_len = request_buffer.size() - sizeof(scf_fapi_body_header_t);
    const std::error_code ec = ran::oran::convert_ul_tti_request_to_cplane(
            *request, body_len, 4, numerology, tx_windows, msg_infos);

    ASSERT_FALSE(ec) << "Error: " << ec.message();
    // Should have 4 messages (1 per antenna port), not 104 (100 + 4)
    EXPECT_EQ(msg_infos.size(), 4U);
}

// Extended test coverage based on test vectors

TEST(FapiToCPlane, PuschTripleLoopVerification) {
    // Comprehensive test to verify message structure: one message per antenna
    // port
    auto pdu = create_basic_pusch_pdu();
    pdu.start_symbol_index = 0;
    pdu.num_of_symbols = 14;
    pdu.dmrs_ports = 0x000F; // 4 ports
    pdu.scid = 0;

    std::vector<ran::oran::OranCPlaneMsgInfo> msg_infos{};
    const auto slot_timing = DEFAULT_TEST_SLOT_TIMING;
    const auto tx_windows = create_test_tx_windows();

    const std::error_code result = convert_single_pusch_pdu_to_cplane(
            pdu, slot_timing, 4, tx_windows, create_30khz_numerology(), msg_infos);

    ASSERT_FALSE(result) << "Error: " << result.message();
    // Should create 4 messages (1 per antenna port)
    ASSERT_EQ(msg_infos.size(), 4U);

    // Verify ordering: messages are organized by antenna port
    // Message 0: Port 0, covers all 14 symbols
    // Message 1: Port 1, covers all 14 symbols
    // Message 2: Port 2, covers all 14 symbols
    // Message 3: Port 3, covers all 14 symbols

    for (std::uint8_t ap = 0; ap < 4; ++ap) {
        const auto &msg = msg_infos.at(ap);
        const auto &radio_hdr = msg.section_common_hdr.sect_1_common_hdr.radioAppHdr;

        // All messages start at symbol 0
        EXPECT_EQ(radio_hdr.startSymbolId.get(), 0U);

        // Each message covers all 14 symbols
        EXPECT_EQ(msg.sections.at(0).sect_1.numSymbol.get(), 14U);

        // Verify antenna port index
        EXPECT_EQ(msg.ap_idx, ap);

        // Verify common fields
        EXPECT_EQ(msg.data_direction, DIRECTION_UPLINK);
        EXPECT_EQ(msg.num_sections, 1);
    }
}

TEST(FapiToCPlane, PuschSingleAntennaPort) {
    // Test with single antenna port configured
    auto pdu = create_basic_pusch_pdu(); // 14 symbols
    pdu.dmrs_ports = 0x0001;             // DMRS configuration (not used for filtering)
    pdu.scid = 0;

    std::vector<ran::oran::OranCPlaneMsgInfo> msg_infos{};
    const auto slot_timing =
            ran::oran::OranSlotTiming{.frame_id = 100, .subframe_id = 5, .slot_id = 7};
    const auto tx_windows = create_test_tx_windows();
    const std::error_code result = convert_single_pusch_pdu_to_cplane(
            pdu, slot_timing, 1, tx_windows, create_30khz_numerology(), msg_infos);

    ASSERT_FALSE(result) << "Error: " << result.message();
    // Should create 1 message (1 antenna port covering 14 symbols)
    ASSERT_EQ(msg_infos.size(), 1U);

    const auto &msg_info = msg_infos.at(0);
    EXPECT_EQ(msg_info.data_direction, DIRECTION_UPLINK);
    EXPECT_EQ(msg_info.num_sections, 1);
}

TEST(FapiToCPlane, PuschTwoAntennaPorts) {
    // Test with 2 antenna ports configured
    auto pdu = create_basic_pusch_pdu(); // 14 symbols
    pdu.dmrs_ports = 0x0003;             // DMRS configuration (not used for filtering)
    pdu.scid = 0;

    std::vector<ran::oran::OranCPlaneMsgInfo> msg_infos{};
    const auto slot_timing =
            ran::oran::OranSlotTiming{.frame_id = 100, .subframe_id = 5, .slot_id = 7};
    const auto tx_windows = create_test_tx_windows();
    const std::error_code result = convert_single_pusch_pdu_to_cplane(
            pdu, slot_timing, 2, tx_windows, create_30khz_numerology(), msg_infos);

    ASSERT_FALSE(result) << "Error: " << result.message();
    // Should create 2 messages (1 per antenna port, each covering 14 symbols)
    ASSERT_EQ(msg_infos.size(), 2U);

    // Verify all messages have correct configuration
    for (std::size_t i = 0; i < msg_infos.size(); ++i) {
        const auto &msg = msg_infos.at(i);
        EXPECT_EQ(msg.data_direction, DIRECTION_UPLINK);
        EXPECT_EQ(msg.num_sections, 1);
        EXPECT_EQ(msg.sections.at(0).sect_1.startPrbc.get(), 0U);
        EXPECT_EQ(msg.sections.at(0).sect_1.numPrbc.get(), 50U);
        EXPECT_EQ(msg.ap_idx, i % 2); // Port 0, 1, 0, 1, ...
    }
}

TEST(FapiToCPlane, PuschEightAntennaPorts) {
    // Test with 8 antenna ports configured
    auto pdu = create_basic_pusch_pdu(); // 14 symbols
    pdu.dmrs_ports = 0x00FF;             // DMRS configuration (not used for filtering)
    pdu.scid = 0;

    std::vector<ran::oran::OranCPlaneMsgInfo> msg_infos{};
    const auto slot_timing =
            ran::oran::OranSlotTiming{.frame_id = 100, .subframe_id = 5, .slot_id = 7};
    const auto tx_windows = create_test_tx_windows();
    const std::error_code result = convert_single_pusch_pdu_to_cplane(
            pdu, slot_timing, 8, tx_windows, create_30khz_numerology(), msg_infos);

    ASSERT_FALSE(result) << "Error: " << result.message();
    // Should create 8 messages (1 per antenna port, each covering 14 symbols)
    EXPECT_EQ(msg_infos.size(), 8U);
}

TEST(FapiToCPlane, PuschWithScid1) {
    // Test with SCID=1 configuration
    auto pdu = create_basic_pusch_pdu(); // 14 symbols
    pdu.dmrs_ports = 0x000F;
    pdu.scid = 1;

    std::vector<ran::oran::OranCPlaneMsgInfo> msg_infos{};
    const auto slot_timing =
            ran::oran::OranSlotTiming{.frame_id = 100, .subframe_id = 5, .slot_id = 7};
    const auto tx_windows = create_test_tx_windows();
    const std::error_code result = convert_single_pusch_pdu_to_cplane(
            pdu, slot_timing, 4, tx_windows, create_30khz_numerology(), msg_infos);

    ASSERT_FALSE(result) << "Error: " << result.message();
    // Should create 4 messages (1 per antenna port, each covering 14 symbols)
    EXPECT_EQ(msg_infos.size(), 4U);
}

TEST(FapiToCPlane, PuschVariousAntennaPortCounts) {
    // Test with different antenna port counts
    auto pdu = create_basic_pusch_pdu(); // 14 symbols
    pdu.dmrs_ports = 0x0055;             // DMRS configuration (not used for filtering)
    pdu.scid = 0;

    std::vector<ran::oran::OranCPlaneMsgInfo> msg_infos{};
    const auto slot_timing =
            ran::oran::OranSlotTiming{.frame_id = 100, .subframe_id = 5, .slot_id = 7};
    const auto tx_windows = create_test_tx_windows();
    const std::error_code result = convert_single_pusch_pdu_to_cplane(
            pdu, slot_timing, 4, tx_windows, create_30khz_numerology(), msg_infos);

    ASSERT_FALSE(result) << "Error: " << result.message();
    // Should create 4 messages (1 per antenna port, each covering 14 symbols)
    EXPECT_EQ(msg_infos.size(), 4U);
}

TEST(FapiToCPlane, PuschMinimalPrbAllocation) {
    // Test minimum PRB allocation (1 PRB)
    auto pdu = create_basic_pusch_pdu();
    pdu.rb_start = 50;
    pdu.rb_size = 1;

    std::vector<ran::oran::OranCPlaneMsgInfo> msg_infos{};
    const auto slot_timing = DEFAULT_TEST_SLOT_TIMING;
    const auto tx_windows = create_test_tx_windows();

    const std::error_code result = convert_single_pusch_pdu_to_cplane(
            pdu, slot_timing, 4, tx_windows, create_30khz_numerology(), msg_infos);

    ASSERT_FALSE(result) << "Error: " << result.message();
    ASSERT_GE(msg_infos.size(), 1U);
    EXPECT_EQ(msg_infos.at(0).sections.at(0).sect_1.startPrbc.get(), 50U);
    EXPECT_EQ(msg_infos.at(0).sections.at(0).sect_1.numPrbc.get(), 1U);
}

TEST(FapiToCPlane, PuschMaxPrbAllocation) {
    // Test large PRB allocation requiring multiple sections (273 PRBs)
    // numPrbc is 8-bit field (max 255), so this will split into 2 sections
    auto pdu = create_basic_pusch_pdu();
    pdu.bwp.bwp_size = 273; // Max PRBs for 100 MHz @ 30 kHz SCS
    pdu.rb_start = 0;
    pdu.rb_size = 273; // Max PRBs for 100 MHz @ 30 kHz SCS

    std::vector<ran::oran::OranCPlaneMsgInfo> msg_infos{};
    const auto slot_timing = DEFAULT_TEST_SLOT_TIMING;
    const auto tx_windows = create_test_tx_windows();

    const std::error_code result = convert_single_pusch_pdu_to_cplane(
            pdu, slot_timing, 4, tx_windows, create_30khz_numerology(), msg_infos);

    ASSERT_FALSE(result) << "Error: " << result.message();
    // 4 messages (1 per antenna port, each covering 14 symbols)
    ASSERT_EQ(msg_infos.size(), 4U);

    // Each message should have 2 sections (255 + 18 PRBs)
    EXPECT_EQ(msg_infos.at(0).num_sections, 2);

    // Section 0: PRBs 0-254 (255 PRBs)
    EXPECT_EQ(msg_infos.at(0).sections.at(0).sect_1.sectionId.get(), 0U);
    EXPECT_EQ(msg_infos.at(0).sections.at(0).sect_1.startPrbc.get(), 0U);
    EXPECT_EQ(msg_infos.at(0).sections.at(0).sect_1.numPrbc.get(), 255U);

    // Section 1: PRBs 255-272 (18 PRBs)
    EXPECT_EQ(msg_infos.at(0).sections.at(1).sect_1.sectionId.get(), 1U);
    EXPECT_EQ(msg_infos.at(0).sections.at(1).sect_1.startPrbc.get(), 255U);
    EXPECT_EQ(msg_infos.at(0).sections.at(1).sect_1.numPrbc.get(), 18U);
}

TEST(FapiToCPlane, PuschPartialSlotStartSymbol) {
    // Test PUSCH starting at non-zero symbol
    auto pdu = create_basic_pusch_pdu();
    pdu.start_symbol_index = 2;
    pdu.num_of_symbols = 12;

    std::vector<ran::oran::OranCPlaneMsgInfo> msg_infos{};
    const auto slot_timing = DEFAULT_TEST_SLOT_TIMING;
    const auto tx_windows = create_test_tx_windows();

    const std::error_code result = convert_single_pusch_pdu_to_cplane(
            pdu, slot_timing, 4, tx_windows, create_30khz_numerology(), msg_infos);

    ASSERT_FALSE(result) << "Error: " << result.message();

    // 4 messages (1 per antenna port, each covering 12 symbols)
    ASSERT_EQ(msg_infos.size(), 4U);

    // Messages should start at symbol 2
    const auto &radio_hdr = msg_infos.at(0).section_common_hdr.sect_1_common_hdr.radioAppHdr;
    EXPECT_EQ(radio_hdr.startSymbolId.get(), 2U);

    // Each message contains 12 symbols
    EXPECT_EQ(msg_infos.at(0).sections.at(0).sect_1.numSymbol.get(), 12U);

    // All messages should start at symbol 2 (differ by antenna port)
    const auto &last_radio_hdr = msg_infos.at(3).section_common_hdr.sect_1_common_hdr.radioAppHdr;
    EXPECT_EQ(last_radio_hdr.startSymbolId.get(), 2U);
    EXPECT_EQ(msg_infos.at(3).ap_idx, 3U);
}

TEST(FapiToCPlane, PuschSingleSymbol) {
    // Test PUSCH with single symbol
    auto pdu = create_basic_pusch_pdu();
    pdu.start_symbol_index = 7;
    pdu.num_of_symbols = 1;

    std::vector<ran::oran::OranCPlaneMsgInfo> msg_infos{};
    const auto slot_timing = DEFAULT_TEST_SLOT_TIMING;
    const auto tx_windows = create_test_tx_windows();

    const std::error_code result = convert_single_pusch_pdu_to_cplane(
            pdu, slot_timing, 4, tx_windows, create_30khz_numerology(), msg_infos);

    ASSERT_FALSE(result) << "Error: " << result.message();
    // 1 symbol × 4 antenna ports = 4 messages
    ASSERT_EQ(msg_infos.size(), 4U);

    const auto &radio_hdr = msg_infos.at(0).section_common_hdr.sect_1_common_hdr.radioAppHdr;
    EXPECT_EQ(radio_hdr.startSymbolId.get(), 7U);
    EXPECT_EQ(msg_infos.at(0).sections.at(0).sect_1.numSymbol.get(), 1U);
}

TEST(FapiToCPlane, PuschLastSymbol) {
    // Test PUSCH using last symbol of slot
    auto pdu = create_basic_pusch_pdu();
    pdu.start_symbol_index = 13;
    pdu.num_of_symbols = 1;

    std::vector<ran::oran::OranCPlaneMsgInfo> msg_infos{};
    const auto slot_timing = DEFAULT_TEST_SLOT_TIMING;
    const auto tx_windows = create_test_tx_windows();

    const std::error_code result = convert_single_pusch_pdu_to_cplane(
            pdu, slot_timing, 4, tx_windows, create_30khz_numerology(), msg_infos);

    ASSERT_FALSE(result) << "Error: " << result.message();
    // 4 messages (1 per antenna port, each covering 1 symbol)
    ASSERT_EQ(msg_infos.size(), 4U);

    const auto &radio_hdr = msg_infos.at(0).section_common_hdr.sect_1_common_hdr.radioAppHdr;
    EXPECT_EQ(radio_hdr.startSymbolId.get(), 13U);
    EXPECT_EQ(msg_infos.at(0).sections.at(0).sect_1.numSymbol.get(), 1U);
}

TEST(FapiToCPlane, PuschAllFieldsValidation) {
    // Comprehensive validation of all fields in generated message
    auto pdu = create_basic_pusch_pdu();
    pdu.rb_start = 10;
    pdu.rb_size = 20;
    pdu.start_symbol_index = 2;
    pdu.num_of_symbols = 10;
    pdu.dmrs_ports = 0x0003; // DMRS configuration (not used for filtering)
    pdu.scid = 0;

    std::vector<ran::oran::OranCPlaneMsgInfo> msg_infos{};
    const auto slot_timing =
            ran::oran::OranSlotTiming{.frame_id = 42, .subframe_id = 5, .slot_id = 3};
    const auto tx_windows = create_test_tx_windows();
    const std::error_code result = convert_single_pusch_pdu_to_cplane(
            pdu, slot_timing, 8, tx_windows, create_30khz_numerology(), msg_infos);

    ASSERT_FALSE(result) << "Error: " << result.message();
    // Should create 8 messages (1 per antenna port, each covering 10 symbols)
    ASSERT_EQ(msg_infos.size(), 8U);

    // Validate all messages have correct common fields
    for (const auto &msg_info : msg_infos) {
        // Validate radio app header
        const auto &radio_hdr = msg_info.section_common_hdr.sect_1_common_hdr.radioAppHdr;
        EXPECT_EQ(radio_hdr.payloadVersion.get(), ORAN_DEF_PAYLOAD_VERSION);
        EXPECT_EQ(radio_hdr.filterIndex.get(), ORAN_DEF_FILTER_INDEX);
        EXPECT_EQ(radio_hdr.frameId, 42);
        EXPECT_EQ(radio_hdr.subframeId.get(), 5U);
        EXPECT_EQ(radio_hdr.slotId.get(), 3U);
        EXPECT_EQ(radio_hdr.sectionType, ORAN_CMSG_SECTION_TYPE_1);
        EXPECT_EQ(radio_hdr.dataDirection.get(), static_cast<std::uint8_t>(DIRECTION_UPLINK));
        EXPECT_EQ(radio_hdr.numberOfSections, 1);

        // Validate message properties
        EXPECT_EQ(msg_info.data_direction, DIRECTION_UPLINK);
        EXPECT_FALSE(msg_info.has_section_ext);
        EXPECT_EQ(msg_info.num_sections, 1);

        // Validate section content
        const auto &section = msg_info.sections.at(0).sect_1;
        EXPECT_EQ(section.sectionId.get(), 0U);
        EXPECT_EQ(section.rb.get(), 0U);
        EXPECT_EQ(section.symInc.get(), 0U);
        EXPECT_EQ(section.startPrbc.get(), 10U);
        EXPECT_EQ(section.numPrbc.get(), 20U);
        EXPECT_EQ(section.reMask.get(), 0x0FFFU);
        EXPECT_EQ(section.numSymbol.get(), 10U); // All 10 symbols in one message
        EXPECT_EQ(section.ef.get(), 0U);
        EXPECT_EQ(section.beamId.get(), 0U);
    }

    // Verify all messages: one per antenna port, all start at symbol 2
    // Message 0: Port 0, symbols 2-11
    // Message 1: Port 1, symbols 2-11
    // ...
    // Message 7: Port 7, symbols 2-11
    for (std::uint8_t port = 0; port < 8; ++port) {
        const auto &radio_hdr = msg_infos.at(port).section_common_hdr.sect_1_common_hdr.radioAppHdr;
        EXPECT_EQ(radio_hdr.startSymbolId.get(), 2U);
        EXPECT_EQ(msg_infos.at(port).ap_idx, port);
    }
}
// Manual UL_TTI.request construction doesn't currently work with
// chunk-based converter
TEST(FapiToCPlane, DISABLED_PuschDifferentNumerologies) {
    // Test timing extraction with different numerologies
    const std::array<std::pair<std::uint16_t, ran::oran::SubcarrierSpacing>, 4> test_cases = {{
            {23, ran::oran::SubcarrierSpacing::Scs30Khz},  // 30 kHz: 2 slots/subframe, slot 23 →
                                                           // subframe 2, slot 3
            {5, ran::oran::SubcarrierSpacing::Scs15Khz},   // 15 kHz: 1 slot/subframe, slot 5 →
                                                           // subframe 5, slot 0
            {47, ran::oran::SubcarrierSpacing::Scs60Khz},  // 60 kHz: 4 slots/subframe, slot 47 →
                                                           // subframe 2, slot 7
            {95, ran::oran::SubcarrierSpacing::Scs120Khz}, // 120 kHz: 8 slots/subframe, slot 95 →
                                                           // subframe 2, slot 15
    }};

    for (const auto &tc : test_cases) {
        const auto [fapi_slot, scs] = tc;
        const auto numerology = ran::oran::from_scs(scs);

        // Create UL_TTI.request
        std::vector<std::uint8_t> buffer{};
        constexpr std::size_t HEADER_SIZE = sizeof(scf_fapi_ul_tti_req_t);
        constexpr std::size_t PDU_HEADER_SIZE = 2 * sizeof(std::uint16_t);
        constexpr std::size_t PDU_SIZE = sizeof(scf_fapi_pusch_pdu_t);
        const std::size_t total_size = HEADER_SIZE + PDU_HEADER_SIZE + PDU_SIZE;
        buffer.resize(total_size, 0);

        auto *request = ran::fapi::assume_cast<scf_fapi_ul_tti_req_t>(buffer.data());
        request->sfn = 100;
        request->slot = fapi_slot;
        request->num_pdus = 1;

        const std::size_t payload_size = PDU_HEADER_SIZE + PDU_SIZE;
        auto payload_span = ran::fapi::make_buffer_span(&request->payload[0], payload_size);
        std::size_t offset = 0;

        const std::uint16_t pdu_type = UL_TTI_PDU_TYPE_PUSCH;
        std::memcpy(payload_span.subspan(offset).data(), &pdu_type, sizeof(std::uint16_t));
        offset += sizeof(std::uint16_t);

        const auto pdu_size = static_cast<std::uint16_t>(PDU_SIZE);
        std::memcpy(payload_span.subspan(offset).data(), &pdu_size, sizeof(std::uint16_t));
        offset += sizeof(std::uint16_t);

        const auto pusch_pdu = create_basic_pusch_pdu();
        std::memcpy(payload_span.subspan(offset).data(), &pusch_pdu, PDU_SIZE);

        std::vector<ran::oran::OranCPlaneMsgInfo> msg_infos{};
        const auto tx_windows = create_test_tx_windows();
        const std::size_t body_len = buffer.size() - sizeof(scf_fapi_body_header_t);
        const std::error_code ec = ran::oran::convert_ul_tti_request_to_cplane(
                *request, body_len, 4, numerology, tx_windows, msg_infos);

        ASSERT_FALSE(ec) << "Error: " << ec.message();
        ASSERT_GE(msg_infos.size(), 1U);

        const auto expected_subframe = fapi_slot / numerology.slots_per_subframe;
        const auto expected_slot_in_sf = fapi_slot % numerology.slots_per_subframe;

        const auto &radio_hdr = msg_infos.at(0).section_common_hdr.sect_1_common_hdr.radioAppHdr;
        EXPECT_EQ(radio_hdr.subframeId.get(), expected_subframe);
        EXPECT_EQ(radio_hdr.slotId.get(), expected_slot_in_sf);
    }
}

TEST(FapiToCPlane, PuschMultipleSections) {
    // Test that PRB allocations > 255 are split into multiple sections
    const std::array<std::pair<std::uint16_t, std::uint8_t>, 3> test_cases = {{
            {50, 1},  // 50 PRBs → 1 section
            {255, 1}, // 255 PRBs → 1 section (boundary)
            {256, 2}, // 256 PRBs → 2 sections (255 + 1)
    }};

    for (const auto &[rb_size, expected_sections] : test_cases) {
        auto pdu = create_basic_pusch_pdu();
        pdu.bwp.bwp_size = 510; // Ensure BWP is large enough for all test cases
        pdu.rb_start = 0;
        pdu.rb_size = rb_size;
        pdu.num_of_symbols = 1; // Single symbol for simpler testing

        std::vector<ran::oran::OranCPlaneMsgInfo> msg_infos{};
        const auto slot_timing =
                ran::oran::OranSlotTiming{.frame_id = 100, .subframe_id = 5, .slot_id = 7};
        const auto tx_windows = create_test_tx_windows();
        const std::error_code result = convert_single_pusch_pdu_to_cplane(
                pdu, slot_timing, 4, tx_windows, create_30khz_numerology(), msg_infos);

        ASSERT_FALSE(result) << "Error: " << result.message();
        ASSERT_GE(msg_infos.size(), 1U);

        // Verify number of sections
        EXPECT_EQ(msg_infos.at(0).num_sections, expected_sections);

        // Verify total PRBs across all sections
        std::uint16_t total_prbs = 0;
        for (std::uint8_t i = 0; i < msg_infos.at(0).num_sections; ++i) {
            total_prbs +=
                    static_cast<std::uint16_t>(msg_infos.at(0).sections.at(i).sect_1.numPrbc.get());
        }
        EXPECT_EQ(total_prbs, rb_size);
    }
}

TEST(FapiToCPlane, PuschBoundaryFrameNumbers) {
    // Test boundary frame numbers
    const std::array<std::uint16_t, 3> frame_ids = {0, 512, 1023};

    for (const auto frame_id : frame_ids) {
        auto pdu = create_basic_pusch_pdu();
        std::vector<ran::oran::OranCPlaneMsgInfo> msg_infos{};

        const auto slot_timing = ran::oran::OranSlotTiming{
                .frame_id = static_cast<std::uint8_t>(frame_id & 0xFFU),
                .subframe_id = 5,
                .slot_id = 3};
        const auto tx_windows = create_test_tx_windows();
        const std::error_code result = convert_single_pusch_pdu_to_cplane(
                pdu, slot_timing, 4, tx_windows, create_30khz_numerology(), msg_infos);

        ASSERT_FALSE(result) << "Error: " << result.message();
        ASSERT_GE(msg_infos.size(), 1U);

        const auto &radio_hdr = msg_infos.at(0).section_common_hdr.sect_1_common_hdr.radioAppHdr;
        EXPECT_EQ(radio_hdr.frameId, static_cast<std::uint8_t>(frame_id & 0xFFU));
    }
}

TEST(FapiToCPlane, PuschUnusedFieldsAreZeroed) {
    // Test that unused/optional fields are properly initialized
    auto pdu = create_basic_pusch_pdu();
    pdu.dmrs_ports = 0x0003; // DMRS configuration (not used for filtering)
    pdu.scid = 0;
    pdu.num_of_symbols = 3; // Only 3 symbols for simpler testing

    std::vector<ran::oran::OranCPlaneMsgInfo> msg_infos{};
    const auto slot_timing =
            ran::oran::OranSlotTiming{.frame_id = 100, .subframe_id = 5, .slot_id = 3};
    const auto tx_windows = create_test_tx_windows();
    const std::error_code result = convert_single_pusch_pdu_to_cplane(
            pdu, slot_timing, 16, tx_windows, create_30khz_numerology(), msg_infos);

    ASSERT_FALSE(result) << "Error: " << result.message();
    // 16 messages (1 per antenna port, each covering 3 symbols)
    ASSERT_EQ(msg_infos.size(), 16U);

    for (std::size_t msg_idx = 0; msg_idx < msg_infos.size(); ++msg_idx) {
        const auto &msg = msg_infos.at(msg_idx);

        // Verify section extensions are disabled
        EXPECT_FALSE(msg.has_section_ext);

        // Verify ap_idx corresponds to the antenna port (0-15)
        const std::uint16_t expected_port = msg_idx % 16;
        EXPECT_EQ(msg.ap_idx, expected_port);

        // Verify timing windows are calculated per symbol
        // tx_window_start and tx_window_end increment per symbol
        // tx_window_bfw_start is slot-relative and does NOT increment
        const std::uint64_t expected_base_time = 1000000000ULL;
        const std::size_t symbol_idx = msg_idx / 16; // 16 ports per symbol
        const std::uint64_t expected_offset = symbol_idx * SYMBOL_DURATION_30_KHZ_NS;
        EXPECT_EQ(msg.tx_window_start, expected_base_time + expected_offset);
        EXPECT_EQ(msg.tx_window_bfw_start,
                  0U); // BFW window is slot-relative, always 0 in test
        EXPECT_EQ(
                msg.tx_window_end,
                expected_base_time + SYMBOL_DURATION_30_KHZ_NS + expected_offset);

        // Verify unused sections beyond num_sections are zeroed
        ASSERT_GT(msg.num_sections, 0U);
        for (std::uint8_t sect_idx = msg.num_sections; sect_idx < ran::oran::MAX_CPLANE_SECTIONS;
             ++sect_idx) {
            const auto &section = msg.sections.at(sect_idx).sect_1;

            // All fields should be zero for unused sections
            EXPECT_EQ(section.sectionId.get(), 0U);
            EXPECT_EQ(section.rb.get(), 0U);
            EXPECT_EQ(section.symInc.get(), 0U);
            EXPECT_EQ(section.startPrbc.get(), 0U);
            EXPECT_EQ(section.numPrbc.get(), 0U);
            EXPECT_EQ(section.reMask.get(), 0U);
            EXPECT_EQ(section.numSymbol.get(), 0U);
            EXPECT_EQ(section.ef.get(), 0U);
            EXPECT_EQ(section.beamId.get(), 0U);
        }

        // Verify section extension flag is 0 in all sections (no extensions)
        for (std::uint8_t sect_idx = 0; sect_idx < msg.num_sections; ++sect_idx) {
            const auto &section = msg.sections.at(sect_idx).sect_1;
            EXPECT_EQ(section.ef.get(), 0U);
        }
    }
}

TEST(FapiToCPlane, PuschAntennaPortIndexProgression) {
    // Verify ap_idx progresses through antenna ports and timing per symbol
    auto pdu = create_basic_pusch_pdu();
    pdu.dmrs_ports = 0x000F; // DMRS configuration (not used for filtering)
    pdu.scid = 0;
    pdu.num_of_symbols = 2; // 2 symbols for simpler verification

    std::vector<ran::oran::OranCPlaneMsgInfo> msg_infos{};
    const auto slot_timing =
            ran::oran::OranSlotTiming{.frame_id = 100, .subframe_id = 5, .slot_id = 3};
    const auto tx_windows = create_test_tx_windows();
    const std::error_code result = convert_single_pusch_pdu_to_cplane(
            pdu, slot_timing, 16, tx_windows, create_30khz_numerology(), msg_infos);

    ASSERT_FALSE(result) << "Error: " << result.message();
    // 16 messages (1 per antenna port, each covering 2 symbols)
    ASSERT_EQ(msg_infos.size(), 16U);

    const std::uint64_t base_time = 1000000000ULL;
    for (std::size_t msg_idx = 0; msg_idx < msg_infos.size(); ++msg_idx) {
        const auto &msg = msg_infos.at(msg_idx);
        const auto &radio_hdr = msg.section_common_hdr.sect_1_common_hdr.radioAppHdr;

        // Expected: symbol0/port0-15, symbol1/port0-15
        const std::size_t symbol_idx = msg_idx / 16; // 16 ports per symbol
        const std::uint16_t expected_port = msg_idx % 16;

        // Verify ap_idx progresses through antenna ports
        EXPECT_EQ(msg.ap_idx, expected_port) << "Message " << msg_idx << " has incorrect ap_idx";
        EXPECT_EQ(radio_hdr.startSymbolId.get(), static_cast<std::uint8_t>(symbol_idx))
                << "Message " << msg_idx << " has incorrect symbol";

        // Verify timing windows progress per symbol
        const std::uint64_t expected_offset = symbol_idx * SYMBOL_DURATION_30_KHZ_NS;
        EXPECT_EQ(msg.tx_window_start, base_time + expected_offset)
                << "Message " << msg_idx << " has incorrect tx_window_start";
        EXPECT_EQ(msg.tx_window_bfw_start, 0U)
                << "Message " << msg_idx
                << " has incorrect tx_window_bfw_start (should be slot-relative)";
    }
}

// Edge Case Tests

TEST(FapiToCPlane, PuschEdgeCaseZeroPrbs) {
    // Test rejection of zero PRB allocation
    auto pdu = create_basic_pusch_pdu();
    pdu.rb_size = 0; // Invalid: zero PRBs

    std::vector<ran::oran::OranCPlaneMsgInfo> msg_infos{};
    const auto slot_timing = DEFAULT_TEST_SLOT_TIMING;
    const auto tx_windows = create_test_tx_windows();

    const std::error_code result = convert_single_pusch_pdu_to_cplane(
            pdu, slot_timing, 4, tx_windows, create_30khz_numerology(), msg_infos);

    EXPECT_TRUE(result);             // Should fail
    EXPECT_EQ(msg_infos.size(), 0U); // No messages created
}

TEST(FapiToCPlane, PuschEdgeCaseZeroSymbols) {
    // Test rejection of zero symbol allocation
    auto pdu = create_basic_pusch_pdu();
    pdu.num_of_symbols = 0; // Invalid: zero symbols

    std::vector<ran::oran::OranCPlaneMsgInfo> msg_infos{};
    const auto slot_timing = DEFAULT_TEST_SLOT_TIMING;
    const auto tx_windows = create_test_tx_windows();

    const std::error_code result = convert_single_pusch_pdu_to_cplane(
            pdu, slot_timing, 4, tx_windows, create_30khz_numerology(), msg_infos);

    EXPECT_TRUE(result);             // Should fail
    EXPECT_EQ(msg_infos.size(), 0U); // No messages created
}

TEST(FapiToCPlane, PuschEdgeCaseSymbolOverflow) {
    // Test rejection when symbols exceed slot boundary
    auto pdu = create_basic_pusch_pdu();
    pdu.start_symbol_index = 10;
    pdu.num_of_symbols = 5; // 10 + 5 = 15 > 14 (slot boundary)

    std::vector<ran::oran::OranCPlaneMsgInfo> msg_infos{};
    const auto slot_timing = DEFAULT_TEST_SLOT_TIMING;
    const auto tx_windows = create_test_tx_windows();

    const std::error_code result = convert_single_pusch_pdu_to_cplane(
            pdu, slot_timing, 4, tx_windows, create_30khz_numerology(), msg_infos);

    EXPECT_TRUE(result);             // Should fail
    EXPECT_EQ(msg_infos.size(), 0U); // No messages created
}

// PRB overflow validation not currently implemented in chunk-based converter
TEST(FapiToCPlane, DISABLED_PuschEdgeCasePrbOverflow) {
    // Test rejection when PRBs exceed BWP boundary
    auto pdu = create_basic_pusch_pdu();
    pdu.bwp.bwp_size = 100;
    pdu.rb_start = 90;
    pdu.rb_size = 20; // 90 + 20 = 110 > 100 (BWP size)

    std::vector<ran::oran::OranCPlaneMsgInfo> msg_infos{};
    const auto slot_timing = DEFAULT_TEST_SLOT_TIMING;
    const auto tx_windows = create_test_tx_windows();

    const std::error_code result = convert_single_pusch_pdu_to_cplane(
            pdu, slot_timing, 4, tx_windows, create_30khz_numerology(), msg_infos);

    EXPECT_TRUE(result);             // Should fail
    EXPECT_EQ(msg_infos.size(), 0U); // No messages created
}

TEST(FapiToCPlane, PuschEdgeCaseNoActivePorts) {
    // Test behavior with dmrs_ports set to 0 (dmrs_ports no longer used for
    // filtering)
    auto pdu = create_basic_pusch_pdu();
    pdu.dmrs_ports = 0x0000; // DMRS configuration

    std::vector<ran::oran::OranCPlaneMsgInfo> msg_infos{};
    const auto slot_timing = DEFAULT_TEST_SLOT_TIMING;
    const auto tx_windows = create_test_tx_windows();

    const std::error_code result = convert_single_pusch_pdu_to_cplane(
            pdu, slot_timing, 4, tx_windows, create_30khz_numerology(), msg_infos);

    ASSERT_FALSE(result) << "Error: " << result.message(); // Should succeed
    EXPECT_EQ(msg_infos.size(), 4U); // 4 antenna ports (each covers 14 symbols)
}

TEST(FapiToCPlane, PuschEdgeCaseSingleSymbol) {
    // Test minimum valid allocation: 1 symbol, 1 PRB, 1 antenna port
    auto pdu = create_basic_pusch_pdu();
    pdu.start_symbol_index = 0;
    pdu.num_of_symbols = 1;
    pdu.rb_start = 0;
    pdu.rb_size = 1;
    pdu.dmrs_ports = 0x0001;
    pdu.scid = 0;

    std::vector<ran::oran::OranCPlaneMsgInfo> msg_infos{};
    const auto slot_timing = DEFAULT_TEST_SLOT_TIMING;
    const auto tx_windows = create_test_tx_windows();

    const std::error_code result = convert_single_pusch_pdu_to_cplane(
            pdu, slot_timing, 1, tx_windows, create_30khz_numerology(), msg_infos);

    ASSERT_FALSE(result) << "Error: " << result.message();
    ASSERT_EQ(msg_infos.size(), 1U); // 1 antenna port (covering 1 symbol)

    const auto &msg = msg_infos.at(0);
    EXPECT_EQ(msg.num_sections, 1U);
    EXPECT_EQ(msg.sections.at(0).sect_1.startPrbc.get(), 0U);
    EXPECT_EQ(msg.sections.at(0).sect_1.numPrbc.get(), 1U);
    EXPECT_EQ(msg.sections.at(0).sect_1.numSymbol.get(), 1U);
}

TEST(FapiToCPlane, PuschEdgeCaseFullSlot) {
    // Test maximum symbol allocation: all 14 symbols
    auto pdu = create_basic_pusch_pdu();
    pdu.start_symbol_index = 0;
    pdu.num_of_symbols = 14; // Full slot
    pdu.dmrs_ports = 0x0001;
    pdu.scid = 0;

    std::vector<ran::oran::OranCPlaneMsgInfo> msg_infos{};
    const auto slot_timing = DEFAULT_TEST_SLOT_TIMING;
    const auto tx_windows = create_test_tx_windows();

    const std::error_code result = convert_single_pusch_pdu_to_cplane(
            pdu, slot_timing, 1, tx_windows, create_30khz_numerology(), msg_infos);

    ASSERT_FALSE(result) << "Error: " << result.message();
    ASSERT_EQ(msg_infos.size(), 1U); // 1 antenna port (covering 14 symbols)

    // Verify the message covers all 14 symbols
    const auto &radio_hdr = msg_infos.at(0).section_common_hdr.sect_1_common_hdr.radioAppHdr;
    EXPECT_EQ(radio_hdr.startSymbolId.get(), 0U);
    EXPECT_EQ(msg_infos.at(0).sections.at(0).sect_1.numSymbol.get(), 14U);

    // Original test checking multiple messages - now we have 1 message
    for (std::size_t i = 0; i < 1; ++i) {
        EXPECT_EQ(radio_hdr.startSymbolId.get(), static_cast<std::uint8_t>(i));
    }
}

TEST(FapiToCPlane, PuschEdgeCaseSymbolBoundary) {
    // Test valid allocation at the end of slot
    auto pdu = create_basic_pusch_pdu();
    pdu.start_symbol_index = 13;
    pdu.num_of_symbols = 1; // Symbol 13 (last symbol)
    pdu.dmrs_ports = 0x0001;
    pdu.scid = 0;

    std::vector<ran::oran::OranCPlaneMsgInfo> msg_infos{};
    const auto slot_timing = DEFAULT_TEST_SLOT_TIMING;
    const auto tx_windows = create_test_tx_windows();

    const std::error_code result = convert_single_pusch_pdu_to_cplane(
            pdu, slot_timing, 1, tx_windows, create_30khz_numerology(), msg_infos);

    ASSERT_FALSE(result) << "Error: " << result.message();
    ASSERT_EQ(msg_infos.size(), 1U); // 1 antenna port (covering 1 symbol)

    const auto &radio_hdr = msg_infos.at(0).section_common_hdr.sect_1_common_hdr.radioAppHdr;
    EXPECT_EQ(radio_hdr.startSymbolId.get(), 13U);
}

TEST(FapiToCPlane, PuschBfwWindowIsSlotRelative) {
    // Verify that tx_window_bfw_start does NOT change per symbol
    auto pdu = create_basic_pusch_pdu();
    pdu.num_of_symbols = 3;
    pdu.dmrs_ports = 0x0001;
    pdu.scid = 0;

    std::vector<ran::oran::OranCPlaneMsgInfo> msg_infos{};
    const auto slot_timing = DEFAULT_TEST_SLOT_TIMING;

    // Set a non-zero bfw_start to ensure it's preserved
    ran::oran::OranTxWindows tx_windows{};
    tx_windows.tx_window_start = 1000000000ULL;
    tx_windows.tx_window_bfw_start = 5000ULL; // Non-zero
    tx_windows.tx_window_end = 1000000000ULL + SYMBOL_DURATION_30_KHZ_NS;

    const std::error_code result = convert_single_pusch_pdu_to_cplane(
            pdu, slot_timing, 1, tx_windows, create_30khz_numerology(), msg_infos);

    ASSERT_FALSE(result) << "Error: " << result.message();
    ASSERT_EQ(msg_infos.size(), 1U); // 1 antenna port (covering 3 symbols)

    // The message should have tx_window_bfw_start preserved (slot-relative)
    EXPECT_EQ(msg_infos.at(0).tx_window_bfw_start, 5000ULL)
            << "BFW window should be slot-relative and not change per symbol";

    // tx_window_start should be set to the start of the first symbol
    EXPECT_EQ(msg_infos.at(0).tx_window_start, 1000000000ULL);
}

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers,cppcoreguidelines-pro-type-union-access)

} // namespace
