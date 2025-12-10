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
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <aerial-fh-driver/oran.hpp>
#include <quill/LogMacros.h>
#include <scf_5g_fapi.h>
#include <tl/expected.hpp>

#include <gtest/gtest.h>

#include "fapi/fapi_buffer.hpp"
#include "fapi/fapi_file_reader.hpp"
#include "fapi/fapi_file_writer.hpp"
#include "fapi_test_utils.hpp"
#include "log/rt_log_macros.hpp"
#include "oran/cplane_message.hpp"
#include "oran/cplane_types.hpp"
#include "oran/fapi_to_cplane.hpp"
#include "oran/numerology.hpp"
#include "oran/oran_log.hpp"
#include "oran/vec_buf.hpp"

namespace {

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

/**
 * Load UL_TTI_REQUEST messages from FAPI capture file
 *
 * @param[in] fapi_file_path Path to .fapi capture file
 * @param[out] cell_request_buffers Map of cell_id to vector of message buffers
 */
void load_ul_tti_requests_from_fapi_file(
        const std::string &fapi_file_path,
        std::map<std::uint16_t, std::vector<std::vector<std::uint8_t>>> &cell_request_buffers) {

    ran::fapi::FapiFileReader reader(fapi_file_path);

    while (auto msg = reader.read_next()) {
        // Filter for UL_TTI_REQUEST messages only
        if (msg->msg_id != SCF_FAPI_UL_TTI_REQUEST) {
            continue;
        }

        // Skip empty messages
        if (msg->msg_data.empty()) {
            continue;
        }

        const std::uint16_t cell_id = msg->cell_id;

        // Copy message data to buffer
        std::vector<std::uint8_t> msg_buffer(msg->msg_data.begin(), msg->msg_data.end());

        // Add to cell's request buffer vector
        cell_request_buffers[cell_id].push_back(std::move(msg_buffer));
    }
}

/**
 * Base fixture for FAPI integration tests
 *
 * Loads FAPI capture file once per test suite and provides shared
 * access to parsed UL_TTI_REQUEST messages for all tests.
 */
class FapiIntegrationTestBase : public ::testing::Test {
protected:
    /**
     * Set up test suite - runs once for all tests
     *
     * Reads FAPI_CAPTURE_DIR and TEST_CELLS environment variables,
     * constructs capture filename dynamically, validates file exists,
     * and loads all UL_TTI_REQUEST messages from the capture file.
     */
    static void SetUpTestSuite() {
        const auto result = ran::fapi::get_fapi_capture_file_path();
        ASSERT_TRUE(result.has_value()) << result.error();
        fapi_capture_file_path = result.value().string();

        RT_LOGC_INFO(
                ran::oran::Oran::OranFapi, "Loading FAPI capture file: {}", fapi_capture_file_path);

        // Read all UL_TTI_REQUEST messages from the capture file
        try {
            load_ul_tti_requests_from_fapi_file(fapi_capture_file_path, cell_request_buffers);

            ASSERT_GT(cell_request_buffers.size(), 0)
                    << "No UL_TTI_REQUEST messages found in FAPI capture file";

            RT_LOGC_INFO(
                    ran::oran::Oran::OranFapi,
                    "Loaded {} cells with FAPI messages from capture file",
                    cell_request_buffers.size());

        } catch (const std::exception &e) {
            FAIL() << "Failed to read FAPI capture file: " << e.what();
        }
    }

    /// Path to FAPI capture file
    inline static std::string fapi_capture_file_path{};

    /// Map of cell_id to vector of UL_TTI_REQUEST message buffers
    inline static std::map<std::uint16_t, std::vector<std::vector<std::uint8_t>>>
            cell_request_buffers{};
};

/// Standard Ethernet MTU size
constexpr std::uint16_t TEST_MTU = 1500;

/// Number of antenna ports for testing
constexpr std::uint16_t NUM_ANTENNA_PORTS = 4;

/**
 * Create 30 kHz numerology for tests
 */
ran::oran::OranNumerology create_30khz_numerology() {
    return ran::oran::from_scs(ran::oran::SubcarrierSpacing::Scs30Khz);
}

/**
 * Create test transmission windows
 *
 * @param[in] base_time Base timestamp in nanoseconds
 * @return Transmission windows structure
 */
ran::oran::OranTxWindows create_test_tx_windows(const std::uint64_t base_time = 1000000000ULL) {
    ran::oran::OranTxWindows windows{};
    windows.tx_window_start = base_time;
    windows.tx_window_bfw_start = 0;
    /// Symbol duration for 30 kHz SCS in nanoseconds
    static constexpr std::uint64_t SYMBOL_DURATION_30KHZ_NS = 35714;
    windows.tx_window_end = base_time + SYMBOL_DURATION_30KHZ_NS;
    return windows;
}

/**
 * Create test packet header template
 *
 * @return Packet header template with example Ethernet/VLAN/eCPRI headers
 */
ran::oran::PacketHeaderTemplate create_test_packet_header() {
    ran::oran::PacketHeaderTemplate hdr{};

    // Example Ethernet addresses
    hdr.eth.src_addr.addr_bytes[0] = 0xAA;
    hdr.eth.src_addr.addr_bytes[1] = 0xBB;
    hdr.eth.src_addr.addr_bytes[2] = 0xCC;
    hdr.eth.src_addr.addr_bytes[3] = 0xDD;
    hdr.eth.src_addr.addr_bytes[4] = 0xEE;
    hdr.eth.src_addr.addr_bytes[5] = 0xFF;

    hdr.eth.dst_addr.addr_bytes[0] = 0x11;
    hdr.eth.dst_addr.addr_bytes[1] = 0x22;
    hdr.eth.dst_addr.addr_bytes[2] = 0x33;
    hdr.eth.dst_addr.addr_bytes[3] = 0x44;
    hdr.eth.dst_addr.addr_bytes[4] = 0x55;
    hdr.eth.dst_addr.addr_bytes[5] = 0x66;

    // VLAN tag
    hdr.vlan.vlan_tci = 0x0100;

    // eCPRI header
    hdr.ecpri.ecpriVersion = 1; // NOLINT(cppcoreguidelines-pro-type-union-access)
    hdr.ecpri.ecpriMessage = ECPRI_MSG_TYPE_IQ;

    return hdr;
}

/**
 * End-to-end integration test: FAPI file -> C-plane -> ORAN packets
 *
 * Tests the complete pipeline from captured FAPI messages to network packets
 */
TEST_F(FapiIntegrationTestBase, json_to_fapi_to_cplane_to_packets) {
    RT_LOGC_DEBUG(ran::oran::Oran::OranFapi, "Starting json_to_fapi_to_cplane_to_packets test");

    // Get first cell's requests from shared fixture data
    const auto &[first_cell_id, request_buffers] = *cell_request_buffers.begin();
    ASSERT_GT(request_buffers.size(), 0) << "No UL TTI requests parsed for cell " << first_cell_id;

    RT_LOGC_DEBUG(
            ran::oran::Oran::OranFapi,
            "Read {} FAPI requests from file for cell {}",
            request_buffers.size(),
            first_cell_id);

    // Step 2: Convert first FAPI request to C-plane messages
    const auto *first_request =
            ran::fapi::assume_cast<const scf_fapi_ul_tti_req_t>(request_buffers[0].data());

    RT_LOGC_DEBUG(
            ran::oran::Oran::OranFapi,
            "Converting FAPI UL_TTI.request: SFN={}, Slot={}, PDUs={}",
            first_request->sfn,
            first_request->slot,
            first_request->num_pdus);

    const auto tx_windows = create_test_tx_windows();
    std::vector<ran::oran::OranCPlaneMsgInfo> cplane_msg_infos{};

    const std::size_t body_len = request_buffers[0].size() - sizeof(scf_fapi_body_header_t);
    const std::error_code ec = ran::oran::convert_ul_tti_request_to_cplane(
            *first_request,
            body_len,
            NUM_ANTENNA_PORTS,
            create_30khz_numerology(),
            tx_windows,
            cplane_msg_infos);

    ASSERT_FALSE(ec) << "Failed to convert FAPI to C-plane: " << ec.message();
    ASSERT_GT(cplane_msg_infos.size(), 0) << "No C-plane messages generated";

    RT_LOGC_DEBUG(
            ran::oran::Oran::OranFapi, "Generated {} C-plane messages", cplane_msg_infos.size());

    // Step 3: Create OranFlow and OranPeer for packet generation
    const auto packet_header = create_test_packet_header();
    ran::oran::SimpleOranFlow flow{packet_header};
    ran::oran::SimpleOranPeer peer{};

    // Step 4: Prepare buffers for packet generation
    // Count how many packets we'll need
    const std::size_t predicted_packet_count =
            ran::oran::count_cplane_packets(cplane_msg_infos, TEST_MTU);

    ASSERT_GT(predicted_packet_count, 0) << "Expected at least one packet";

    RT_LOGC_DEBUG(
            ran::oran::Oran::OranFapi,
            "Predicted {} ORAN packets needed (MTU={})",
            predicted_packet_count,
            TEST_MTU);

    // Create buffers
    std::vector<ran::oran::VecBuf> test_buffers{};
    test_buffers.reserve(predicted_packet_count);

    for (std::size_t i = 0; i < predicted_packet_count; ++i) {
        test_buffers.emplace_back(TEST_MTU);
    }

    // Step 5: Generate packets for first C-plane message
    const std::uint16_t packet_count = ran::oran::prepare_cplane_message(
            cplane_msg_infos[0], flow, peer, std::span<ran::oran::VecBuf>{test_buffers}, TEST_MTU);

    // Verify packets were created
    ASSERT_GT(packet_count, 0) << "No packets created";
    EXPECT_LE(packet_count, predicted_packet_count) << "More packets than predicted";

    RT_LOGC_DEBUG(
            ran::oran::Oran::OranFapi,
            "Generated {} ORAN packets (first packet size={} bytes)",
            packet_count,
            test_buffers[0].size());

    // Verify first packet has data
    EXPECT_GT(test_buffers[0].size(), 0) << "First packet is empty";

    // Verify packet structure
    const auto *first_packet_data = test_buffers[0].data();
    ASSERT_NE(first_packet_data, nullptr) << "First packet data is null";

    // Verify packet has expected headers (Ethernet + VLAN + eCPRI)
    constexpr std::size_t MIN_HEADER_SIZE = sizeof(ran::oran::PacketHeaderTemplate);
    EXPECT_GE(test_buffers[0].size(), MIN_HEADER_SIZE) << "Packet too small for headers";

    RT_LOGC_DEBUG(
            ran::oran::Oran::OranFapi,
            "Test complete: Generated {} packets from {} C-plane messages",
            packet_count,
            cplane_msg_infos.size());
}

/**
 * Test processing multiple requests from FAPI file
 */
TEST_F(FapiIntegrationTestBase, process_multiple_json_requests) {
    RT_LOGC_DEBUG(ran::oran::Oran::OranFapi, "Starting process_multiple_json_requests test");

    // Get first cell's requests from shared fixture data
    const auto &[first_cell_id, request_buffers] = *cell_request_buffers.begin();
    ASSERT_GT(request_buffers.size(), 0);

    RT_LOGC_DEBUG(
            ran::oran::Oran::OranFapi,
            "Read {} requests for cell {}, will process first 3",
            request_buffers.size(),
            first_cell_id);

    const auto tx_windows = create_test_tx_windows();
    const auto packet_header = create_test_packet_header();
    ran::oran::SimpleOranFlow flow{packet_header};
    ran::oran::SimpleOranPeer peer{};

    std::size_t total_cplane_messages = 0;
    std::size_t total_packets = 0;

    // Process first 3 requests (or all if fewer)
    const std::size_t requests_to_process =
            std::min(request_buffers.size(), static_cast<std::size_t>(3));

    for (std::size_t req_idx = 0; req_idx < requests_to_process; ++req_idx) {
        const auto *request = ran::fapi::assume_cast<const scf_fapi_ul_tti_req_t>(
                request_buffers[req_idx].data());

        // Convert to C-plane messages
        std::vector<ran::oran::OranCPlaneMsgInfo> cplane_msg_infos{};
        const std::size_t body_len =
                request_buffers[req_idx].size() - sizeof(scf_fapi_body_header_t);
        const std::error_code ec = ran::oran::convert_ul_tti_request_to_cplane(
                *request,
                body_len,
                NUM_ANTENNA_PORTS,
                create_30khz_numerology(),
                tx_windows,
                cplane_msg_infos);

        ASSERT_FALSE(ec) << "Conversion failed for request " << req_idx << ": " << ec.message();
        ASSERT_GT(cplane_msg_infos.size(), 0) << "No C-plane messages for request " << req_idx;

        total_cplane_messages += cplane_msg_infos.size();

        // Actually generate packets for first C-plane message of each request
        const std::size_t predicted_count =
                ran::oran::count_cplane_packets({cplane_msg_infos.data(), 1}, TEST_MTU);

        EXPECT_GT(predicted_count, 0) << "No packets predicted for request " << req_idx;

        // Create buffers and generate packets
        std::vector<ran::oran::VecBuf> test_buffers{};
        test_buffers.reserve(predicted_count);

        for (std::size_t i = 0; i < predicted_count; ++i) {
            test_buffers.emplace_back(TEST_MTU);
        }

        const std::uint16_t actual_count = ran::oran::prepare_cplane_message(
                cplane_msg_infos[0],
                flow,
                peer,
                std::span<ran::oran::VecBuf>{test_buffers},
                TEST_MTU);

        EXPECT_GT(actual_count, 0) << "No packets generated for request " << req_idx;
        EXPECT_LE(actual_count, predicted_count)
                << "More packets than predicted for request " << req_idx;

        // Verify first packet has data
        EXPECT_GT(test_buffers[0].size(), 0) << "First packet empty for request " << req_idx;

        total_packets += actual_count;
    }

    // Verify we processed something meaningful
    EXPECT_GT(total_cplane_messages, 0) << "No C-plane messages generated";
    EXPECT_GT(total_packets, 0) << "No packets generated";

    RT_LOGC_DEBUG(
            ran::oran::Oran::OranFapi,
            "Processed {} requests, generated {} C-plane messages, created "
            "{} packets",
            requests_to_process,
            total_cplane_messages,
            total_packets);
}

/**
 * Test statistics collection across pipeline with actual packet generation
 */
TEST_F(FapiIntegrationTestBase, pipeline_statistics_with_packet_generation) {
    RT_LOGC_DEBUG(
            ran::oran::Oran::OranFapi, "Starting pipeline_statistics_with_packet_generation test");

    // Get first cell's requests from shared fixture data
    const auto &[first_cell_id, request_buffers] = *cell_request_buffers.begin();
    ASSERT_GT(request_buffers.size(), 0);

    struct PipelineStats {
        std::size_t fapi_requests{};
        std::size_t total_pdus{};
        std::size_t cplane_messages{};
        std::size_t total_packets_generated{};
        std::size_t total_packet_bytes{};
    };

    PipelineStats stats{};
    stats.fapi_requests = request_buffers.size();

    const auto tx_windows = create_test_tx_windows();
    const auto packet_header = create_test_packet_header();
    ran::oran::SimpleOranFlow flow{packet_header};
    ran::oran::SimpleOranPeer peer{};

    // Process first 5 requests (or all if fewer)
    const std::size_t requests_to_analyze =
            std::min(request_buffers.size(), static_cast<std::size_t>(5));

    for (std::size_t req_idx = 0; req_idx < requests_to_analyze; ++req_idx) {
        const auto *request = ran::fapi::assume_cast<const scf_fapi_ul_tti_req_t>(
                request_buffers[req_idx].data());

        stats.total_pdus += request->num_pdus;

        std::vector<ran::oran::OranCPlaneMsgInfo> cplane_msg_infos{};
        const std::size_t body_len =
                request_buffers[req_idx].size() - sizeof(scf_fapi_body_header_t);
        const std::error_code ec = ran::oran::convert_ul_tti_request_to_cplane(
                *request,
                body_len,
                NUM_ANTENNA_PORTS,
                create_30khz_numerology(),
                tx_windows,
                cplane_msg_infos);

        if (!ec && !cplane_msg_infos.empty()) {
            stats.cplane_messages += cplane_msg_infos.size();

            // Actually generate packets for first C-plane message
            const std::size_t predicted_count =
                    ran::oran::count_cplane_packets({cplane_msg_infos.data(), 1}, TEST_MTU);

            if (predicted_count > 0) {
                std::vector<ran::oran::VecBuf> test_buffers{};
                test_buffers.reserve(predicted_count);

                for (std::size_t i = 0; i < predicted_count; ++i) {
                    test_buffers.emplace_back(TEST_MTU);
                }

                const std::uint16_t packet_count = ran::oran::prepare_cplane_message(
                        cplane_msg_infos[0],
                        flow,
                        peer,
                        std::span<ran::oran::VecBuf>{test_buffers},
                        TEST_MTU);

                stats.total_packets_generated += packet_count;

                // Collect packet sizes
                for (std::size_t i = 0; i < packet_count; ++i) {
                    stats.total_packet_bytes += test_buffers[i].size();
                }
            }
        }
    }

    // Verify we have meaningful statistics
    EXPECT_GT(stats.fapi_requests, 0);
    EXPECT_GT(stats.total_pdus, 0);
    EXPECT_GT(stats.cplane_messages, 0);
    EXPECT_GT(stats.total_packets_generated, 0);
    EXPECT_GT(stats.total_packet_bytes, 0);

    // Calculate average packet size
    const std::size_t avg_packet_size =
            stats.total_packets_generated > 0
                    ? stats.total_packet_bytes / stats.total_packets_generated
                    : 0;

    EXPECT_GT(avg_packet_size, 0);
    EXPECT_LE(avg_packet_size, TEST_MTU) << "Average packet size exceeds MTU";

    RT_LOGC_DEBUG(
            ran::oran::Oran::OranFapi,
            "Pipeline statistics: {} FAPI requests, {} PDUs, {} C-plane messages, "
            "{} packets generated, {} total bytes, {} avg packet size",
            stats.fapi_requests,
            stats.total_pdus,
            stats.cplane_messages,
            stats.total_packets_generated,
            stats.total_packet_bytes,
            avg_packet_size);
}

/**
 * Test packet fragmentation with standard MTU
 */
TEST_F(FapiIntegrationTestBase, packet_fragmentation_standard_mtu) {
    RT_LOGC_DEBUG(
            ran::oran::Oran::OranFapi,
            "Starting packet_fragmentation_standard_mtu test (MTU={})",
            TEST_MTU);

    // Get first cell's requests from shared fixture data
    const auto &[first_cell_id, request_buffers] = *cell_request_buffers.begin();
    ASSERT_GT(request_buffers.size(), 0);

    const auto *request =
            ran::fapi::assume_cast<const scf_fapi_ul_tti_req_t>(request_buffers[0].data());

    const auto tx_windows = create_test_tx_windows();
    std::vector<ran::oran::OranCPlaneMsgInfo> cplane_msg_infos{};

    const std::size_t body_len = request_buffers[0].size() - sizeof(scf_fapi_body_header_t);
    const std::error_code ec = ran::oran::convert_ul_tti_request_to_cplane(
            *request,
            body_len,
            NUM_ANTENNA_PORTS,
            create_30khz_numerology(),
            tx_windows,
            cplane_msg_infos);

    ASSERT_FALSE(ec) << "Error: " << ec.message();
    ASSERT_GT(cplane_msg_infos.size(), 0);

    // Test with standard MTU - may cause fragmentation
    const auto packet_header = create_test_packet_header();
    ran::oran::SimpleOranFlow flow{packet_header};
    ran::oran::SimpleOranPeer peer{};

    const std::size_t predicted_count =
            ran::oran::count_cplane_packets({cplane_msg_infos.data(), 1}, TEST_MTU);

    ASSERT_GT(predicted_count, 0);

    // Create buffers
    std::vector<ran::oran::VecBuf> test_buffers{};
    test_buffers.reserve(predicted_count);

    for (std::size_t i = 0; i < predicted_count; ++i) {
        test_buffers.emplace_back(TEST_MTU);
    }

    const std::uint16_t packet_count = ran::oran::prepare_cplane_message(
            cplane_msg_infos[0], flow, peer, std::span<ran::oran::VecBuf>{test_buffers}, TEST_MTU);

    EXPECT_GT(packet_count, 0);
    EXPECT_EQ(packet_count, predicted_count) << "Actual packet count should match prediction";

    // Verify all generated packets are within MTU
    for (std::size_t i = 0; i < packet_count; ++i) {
        EXPECT_GT(test_buffers[i].size(), 0) << "Packet " << i << " is empty";
        EXPECT_LE(test_buffers[i].size(), TEST_MTU) << "Packet " << i << " exceeds MTU";
    }

    RT_LOGC_DEBUG(
            ran::oran::Oran::OranFapi,
            "Fragmentation test: Generated {} packets from 1 C-plane message ({})",
            packet_count,
            packet_count > 1 ? "fragmentation occurred" : "no fragmentation");
}

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

} // namespace
