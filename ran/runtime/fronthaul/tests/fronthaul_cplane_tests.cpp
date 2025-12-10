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
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <scf_5g_fapi.h>
#include <tl/expected.hpp>

#include <gtest/gtest.h>

#include "fapi/fapi_file_replay.hpp"
#include "fapi_test_utils.hpp"
#include "fronthaul/fronthaul.hpp"
#include "net/doca_rxq.hpp"
#include "net/doca_txq.hpp"
#include "net/dpdk_txq.hpp"
#include "net/dpdk_types.hpp"
#include "net/env.hpp"
#include "net/mempool.hpp"
#include "net/nic.hpp"
#include "oran/numerology.hpp"
#include "task/time.hpp"

namespace {

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

/**
 * Test fixture for Fronthaul::send_ul_cplane() tests
 *
 * Creates a single shared Fronthaul instance for all tests to avoid
 * multiple DPDK EAL initializations. Tests use stats deltas to verify behavior.
 */
class FronthaulSendUlCplaneTest : public ::testing::Test {
protected:
    /**
     * Set up test suite - runs once for all tests
     *
     * Reads FAPI_CAPTURE_DIR and TEST_CELLS environment variables,
     * constructs capture filename dynamically, validates file exists,
     * and creates a shared Fronthaul instance.
     */
    static void SetUpTestSuite() {
        if (!fapi_capture_file_.empty()) {
            return; // Already set up
        }

        const auto result = ran::fapi::get_fapi_capture_file_path();
        ASSERT_TRUE(result.has_value()) << result.error();
        fapi_capture_file_ = result.value().string();

        // Load FAPI file to determine cell count
        const ran::fapi::FapiFileReplay replay(fapi_capture_file_, SLOTS_PER_SUBFRAME_30_KHZ);
        const auto cell_count = replay.get_cell_count();
        ASSERT_GT(cell_count, 0U);

        // Create shared Fronthaul instance with matching cell count
        const auto config = create_shared_config(cell_count);
        shared_fronthaul_ = std::make_unique<ran::fronthaul::Fronthaul>(config);
    }

    /**
     * Tear down test suite - runs once after all tests
     */
    static void TearDownTestSuite() { shared_fronthaul_.reset(); }

    /**
     * Set up before each test - reset statistics
     */
    void SetUp() override {
        if (shared_fronthaul_) {
            shared_fronthaul_->reset_stats();
        }
    }

    /**
     * Create a valid network config for testing
     *
     * @return Network environment configuration
     */
    static framework::net::EnvConfig create_network_config() {
        const auto available_nics = framework::net::discover_mellanox_nics();
        if (available_nics.empty()) {
            throw std::runtime_error("No Mellanox NICs available for testing");
        }

        framework::net::EnvConfig config{};
        config.nic_config.nic_pcie_addr = available_nics.front();
        config.dpdk_config.file_prefix = "fronthaul_send_ul_cplane_test";
        config.dpdk_config.dpdk_core_id = 0;
        config.dpdk_config.enable_rt_priority_for_lcores = false;

        // Create one DPDK TX queue
        framework::net::DpdkTxQConfig dpdk_txq_config{};
        dpdk_txq_config.txq_size = 128;
        config.nic_config.dpdk_txq_configs.push_back(dpdk_txq_config);

        // Create one mempool
        framework::net::MempoolConfig mempool_config{};
        mempool_config.name = "fronthaul_test_mempool";
        mempool_config.num_mbufs = 1024;
        mempool_config.mtu_size = 1514;
        mempool_config.host_pinned = framework::net::HostPinned::No;
        config.nic_config.mempool_configs.push_back(mempool_config);

        return config;
    }

    /**
     * Create the shared fronthaul configuration
     *
     * @param[in] num_cells Number of cells to configure
     * @return Fronthaul configuration for shared instance
     */
    static ran::fronthaul::FronthaulConfig create_shared_config(std::size_t num_cells) {
        ran::fronthaul::FronthaulConfig config{};
        config.net_config = create_network_config();
        config.numerology = ran::oran::from_scs(ran::oran::SubcarrierSpacing::Scs30Khz);
        config.num_antenna_ports = DEFAULT_NUM_ANTENNA_PORTS;
        config.slot_ahead = 2;
        config.t1a_max_cp_ul_ns = 500'000; // 500 us
        config.t1a_min_cp_ul_ns = 250'000; // 250 us

        // Create cell destination MAC addresses
        for (std::size_t i = 0; i < num_cells; ++i) {
            // Create unique MAC addresses for each cell
            const auto mac_str = std::format("00:11:22:33:44:{:02x}", static_cast<unsigned>(i));
            const auto mac_result = framework::net::MacAddress::from_string(mac_str);
            if (mac_result.has_value()) {
                config.cell_dest_macs.push_back(mac_result.value());
            }
        }

        // Create matching VLAN TCI values for each cell (100, 101, 102, etc.)
        config.cell_vlan_tcis.resize(num_cells);
        std::iota(config.cell_vlan_tcis.begin(), config.cell_vlan_tcis.end(), 100);

        return config;
    }

    /**
     * Create a valid config for constructor edge case tests
     *
     * @param[in] num_cells Number of cells to configure
     * @return Valid fronthaul configuration
     */
    static ran::fronthaul::FronthaulConfig create_valid_config(std::size_t num_cells = 1) {
        return create_shared_config(num_cells);
    }

    /**
     * Create a test UL_TTI request
     *
     * @param[in] sfn System frame number
     * @param[in] slot Slot number
     * @param[in] num_pdus Number of PDUs to include
     * @return Test request structure
     */
    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    static scf_fapi_ul_tti_req_t
    create_test_request(std::uint16_t sfn = 0, std::uint16_t slot = 0, std::uint8_t num_pdus = 1) {
        scf_fapi_ul_tti_req_t request{};
        request.sfn = sfn;
        request.slot = slot;
        request.num_pdus = num_pdus;
        request.rach_present = 0;
        request.num_ulsch = num_pdus;
        request.num_ulcch = 0;
        return request;
        // NOLINTEND(bugprone-easily-swappable-parameters)
    }

    /**
     * Calculate body length for a minimal UL_TTI request struct
     *
     * For test requests created with create_test_request() that have no PDU payload data,
     * the body length is the struct size minus the body header.
     *
     * @return Body length for minimal test request
     */
    static constexpr std::size_t get_test_request_body_len() {
        return sizeof(scf_fapi_ul_tti_req_t) - sizeof(scf_fapi_body_header_t);
    }

    /// Path to FAPI capture file
    // NOLINTNEXTLINE(readability-identifier-naming)
    inline static std::string fapi_capture_file_{};

    /// Shared Fronthaul instance for all tests
    // NOLINTNEXTLINE(readability-identifier-naming)
    inline static std::unique_ptr<ran::fronthaul::Fronthaul> shared_fronthaul_{};

    static constexpr std::uint8_t SLOTS_PER_SUBFRAME_30_KHZ = 2;
    static constexpr std::uint32_t DEFAULT_NUM_ANTENNA_PORTS = 4;
};

// ============================================================================
// 1. Constructor Tests - Edge Cases
// ============================================================================

TEST_F(FronthaulSendUlCplaneTest, ConstructorEdgeCases) {
    // Test 1: Empty cell MACs - should throw std::invalid_argument
    {
        auto config = create_valid_config(1);
        config.cell_dest_macs.clear();

        EXPECT_THROW(
                {
                    try {
                        const ran::fronthaul::Fronthaul fronthaul(config);
                    } catch (const std::invalid_argument &e) {
                        EXPECT_STREQ("At least one cell MAC address required", e.what());
                        throw;
                    }
                },
                std::invalid_argument);
    }

    // Test 2: Invalid network config - should throw std::runtime_error
    {
        auto config = create_valid_config(1);
        config.net_config.nic_config.nic_pcie_addr = "invalid_pcie_address";

        EXPECT_THROW(
                {
                    try {
                        const ran::fronthaul::Fronthaul fronthaul(config);
                    } catch (const std::runtime_error &e) {
                        // Verify error message mentions network environment failure
                        const std::string error_msg = e.what();
                        EXPECT_TRUE(
                                error_msg.find("Failed to create network environment") !=
                                std::string::npos)
                                << "Error message: " << error_msg;
                        throw;
                    }
                },
                std::runtime_error);
    }
}

// ============================================================================
// 2. send_ul_cplane - Input Validation
// ============================================================================

TEST_F(FronthaulSendUlCplaneTest, SendUlCplaneInputValidation) {
    ASSERT_NE(shared_fronthaul_, nullptr) << "Shared fronthaul instance not initialized";

    const auto num_cells = shared_fronthaul_->config().cell_dest_macs.size();
    const auto t0 = std::chrono::nanoseconds{0};
    const auto tai_offset = std::chrono::nanoseconds{0};

    // Test 1: Invalid cell_id (>= num cells) - should log error, return early, no stats increment
    {
        const auto request = create_test_request(0, 0, 1);
        const std::size_t body_len = get_test_request_body_len();
        const std::uint64_t absolute_slot = 0;
        const auto invalid_cell_id = static_cast<std::uint16_t>(num_cells);

        shared_fronthaul_->reset_stats();
        shared_fronthaul_->send_ul_cplane(
                request, body_len, invalid_cell_id, absolute_slot, t0, tai_offset);
        const auto stats = shared_fronthaul_->get_stats();

        // Statistics should not increment for invalid cell_id
        EXPECT_EQ(stats.requests_sent, 0U);
    }

    // Test 2: Empty request (num_pdus = 0) - should log debug, return early, no stats increment
    {
        const auto request = create_test_request(0, 0, 0); // num_pdus = 0
        const std::size_t body_len = get_test_request_body_len();
        const std::uint64_t absolute_slot = 0;
        const std::uint16_t valid_cell_id = 0;

        shared_fronthaul_->reset_stats();
        shared_fronthaul_->send_ul_cplane(
                request, body_len, valid_cell_id, absolute_slot, t0, tai_offset);
        const auto stats = shared_fronthaul_->get_stats();

        // Statistics should not increment for empty request
        EXPECT_EQ(stats.requests_sent, 0U);
    }

    // Test 3: Valid cell_id boundary (max valid cell) - should process successfully
    {
        const auto request = create_test_request(0, 0, 0); // Empty request for simplicity
        const std::size_t body_len = get_test_request_body_len();
        const std::uint64_t absolute_slot = 0;
        const auto max_valid_cell_id = static_cast<std::uint16_t>(num_cells - 1);

        // Should not throw
        EXPECT_NO_THROW({
            shared_fronthaul_->send_ul_cplane(
                    request, body_len, max_valid_cell_id, absolute_slot, t0, tai_offset);
        });
    }

    // Test 4: Edge timing values - should process correctly
    {
        const auto request = create_test_request(0, 0, 0); // Empty request
        const std::size_t body_len = get_test_request_body_len();

        // t0 = 0
        EXPECT_NO_THROW({
            shared_fronthaul_->send_ul_cplane(
                    request, body_len, 0, 0, std::chrono::nanoseconds{0}, tai_offset);
        });

        // tai_offset = 0
        EXPECT_NO_THROW({
            shared_fronthaul_->send_ul_cplane(
                    request, body_len, 0, 0, t0, std::chrono::nanoseconds{0});
        });

        // absolute_slot = 0
        EXPECT_NO_THROW(
                { shared_fronthaul_->send_ul_cplane(request, body_len, 0, 0, t0, tai_offset); });

        // absolute_slot = 10000 - should not overflow
        EXPECT_NO_THROW({
            shared_fronthaul_->send_ul_cplane(request, body_len, 0, 10000, t0, tai_offset);
        });
    }
}

// ============================================================================
// 3. FAPI File Replay Integration Tests
// ============================================================================

TEST_F(FronthaulSendUlCplaneTest, SendUlCplaneWithFapiReplayIntegrationAndStats) {
    ASSERT_NE(shared_fronthaul_, nullptr) << "Shared fronthaul instance not initialized";

    // Load FAPI capture file
    ran::fapi::FapiFileReplay replay(fapi_capture_file_, SLOTS_PER_SUBFRAME_30_KHZ);

    const auto cell_count = replay.get_cell_count();
    ASSERT_GT(cell_count, 0U);
    ASSERT_EQ(cell_count, shared_fronthaul_->config().cell_dest_macs.size())
            << "Shared fronthaul cell count must match FAPI file";

    // Calculate timing parameters
    // t0 represents when slot 0 started, calculated as "now" minus elapsed time
    // Use same time source as fronthaul (system_clock via Time::now_ns())
    const auto now_ns = framework::task::Time::now_ns();
    const auto current_absolute_slot = replay.get_current_absolute_slot();
    const auto slot_period_ns =
            std::chrono::nanoseconds{shared_fronthaul_->config().numerology.slot_period_ns};
    const auto elapsed_time = current_absolute_slot * slot_period_ns;
    const auto t0 = now_ns - elapsed_time;
    const auto tai_offset = std::chrono::nanoseconds{0};

    // Process 100 slots
    constexpr std::size_t SLOTS_TO_PROCESS = 100;
    std::vector<std::size_t> requests_per_cell(cell_count, 0);

    const auto &cell_ids = replay.get_cell_ids();
    ASSERT_EQ(cell_ids.size(), cell_count);

    for (std::size_t slot_idx = 0; slot_idx < SLOTS_TO_PROCESS; ++slot_idx) {
        const auto absolute_slot = replay.get_current_absolute_slot();

        // For each cell, get request and send if available
        for (std::size_t cell_idx = 0; cell_idx < cell_count; ++cell_idx) {
            const auto cell_id = cell_ids.at(cell_idx);
            const auto request_opt = replay.get_request_for_current_slot(cell_id);

            if (request_opt.has_value()) {
                const auto &req_info = request_opt.value();
                ASSERT_NE(req_info.request, nullptr);

                // Send the request
                EXPECT_NO_THROW({
                    shared_fronthaul_->send_ul_cplane(
                            *req_info.request,
                            req_info.body_len,
                            static_cast<std::uint16_t>(cell_idx),
                            absolute_slot,
                            t0,
                            tai_offset);
                });

                ++requests_per_cell.at(cell_idx);
            }
        }

        std::ignore = replay.advance_slot();
    }

    const auto stats = shared_fronthaul_->get_stats();

    // Verify statistics
    EXPECT_GT(stats.requests_sent, 0U) << "Should have sent at least some requests";

    // Verify each cell had some requests processed
    std::size_t total_requests{0};
    for (std::size_t i = 0; i < cell_count; ++i) {
        EXPECT_GT(requests_per_cell.at(i), 0U)
                << "Cell " << cell_ids.at(i) << " should have processed requests";
        total_requests += requests_per_cell.at(i);
    }

    // Statistics should match actual sends
    EXPECT_EQ(stats.requests_sent, total_requests);
}

TEST_F(FronthaulSendUlCplaneTest, SendUlCplaneWithFapiReplayStressTest) {
    ASSERT_NE(shared_fronthaul_, nullptr) << "Shared fronthaul instance not initialized";

    // Load FAPI file
    ran::fapi::FapiFileReplay replay(fapi_capture_file_, SLOTS_PER_SUBFRAME_30_KHZ);

    const auto cell_count = replay.get_cell_count();
    ASSERT_GT(cell_count, 0U);
    ASSERT_EQ(cell_count, shared_fronthaul_->config().cell_dest_macs.size())
            << "Shared fronthaul cell count must match FAPI file";

    // Calculate timing
    // t0 represents when slot 0 started, calculated as "now" minus elapsed time
    // Use same time source as fronthaul (system_clock via Time::now_ns())
    const auto now_ns = framework::task::Time::now_ns();
    const auto current_absolute_slot = replay.get_current_absolute_slot();
    const auto slot_period_ns =
            std::chrono::nanoseconds{shared_fronthaul_->config().numerology.slot_period_ns};
    const auto elapsed_time = current_absolute_slot * slot_period_ns;
    const auto t0 = now_ns - elapsed_time;
    const auto tai_offset = std::chrono::nanoseconds{0};

    const auto &cell_ids = replay.get_cell_ids();

    // Process 1000 slots as fast as possible
    constexpr std::size_t STRESS_SLOT_COUNT = 1000;
    std::size_t total_requests_sent{0};

    for (std::size_t slot_idx = 0; slot_idx < STRESS_SLOT_COUNT; ++slot_idx) {
        const auto absolute_slot = replay.get_current_absolute_slot();

        // Process all cells
        for (std::size_t cell_idx = 0; cell_idx < cell_count; ++cell_idx) {
            const auto cell_id = cell_ids.at(cell_idx);
            const auto request_opt = replay.get_request_for_current_slot(cell_id);

            if (request_opt.has_value()) {
                const auto &req_info = request_opt.value();
                shared_fronthaul_->send_ul_cplane(
                        *req_info.request,
                        req_info.body_len,
                        static_cast<std::uint16_t>(cell_idx),
                        absolute_slot,
                        t0,
                        tai_offset);
                ++total_requests_sent;
            }
        }

        std::ignore = replay.advance_slot();
    }

    const auto stats = shared_fronthaul_->get_stats();

    // Verify all slots processed successfully
    EXPECT_EQ(stats.requests_sent, total_requests_sent);
    EXPECT_GT(stats.requests_sent, 0U) << "Should have processed requests";

    // Statistics should be accurate
    EXPECT_GE(stats.packets_sent, stats.requests_sent) << "Packets sent should be >= requests sent";
}

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

} // namespace
