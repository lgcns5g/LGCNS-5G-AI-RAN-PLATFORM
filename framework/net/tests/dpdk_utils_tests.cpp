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
 * @file dpdk_utils_tests.cpp
 * @brief Unit tests for DPDK utilities
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include <quill/LogMacros.h>
#include <rte_mbuf.h>
#include <rte_mbuf_core.h>
#include <rte_mempool.h>
#include <tl/expected.hpp>
#include <wise_enum_detail.h>

#include <gtest/gtest.h>
#include <wise_enum.h>

#include "log/rt_log_macros.hpp"
#include "net/details/dpdk_utils.hpp"
#include "net/dpdk_types.hpp"
#include "net/net_log.hpp"
#include "net_test_helpers.hpp"

namespace {

TEST(DpdkUtils, DiscoversMellanoxNics) {
    const auto nics = framework::net::discover_mellanox_nics();
    EXPECT_FALSE(nics.empty()) << "Must have at least one Mellanox NIC";
}

TEST(DpdkUtils, ParsesMacAddress) {
    struct TestCase {
        std::string_view input;
        bool should_succeed{};
        framework::net::MacAddress expected_mac{};
        std::string description;
    };

    const std::vector<TestCase> test_cases{
            // Valid MAC addresses
            {"00:11:22:33:44:55",
             true,
             framework::net::MacAddress{{{0x00, 0x11, 0x22, 0x33, 0x44, 0x55}}},
             "Standard MAC address"},
            {"ff:ff:ff:ff:ff:ff",
             true,
             framework::net::MacAddress{{{0xff, 0xff, 0xff, 0xff, 0xff, 0xff}}},
             "Broadcast MAC address"},
            {"00:00:00:00:00:00",
             true,
             framework::net::MacAddress{{{0x00, 0x00, 0x00, 0x00, 0x00, 0x00}}},
             "Zero MAC address"},
            {"aa:bb:cc:dd:ee:ff",
             true,
             framework::net::MacAddress{{{0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff}}},
             "Mixed case letters"},
            {"12:34:56:78:9a:bc",
             true,
             framework::net::MacAddress{{{0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc}}},
             "Numbers and letters"},

            // Invalid MAC addresses - wrong format
            {"", false, {}, "Empty string"},
            {"00:11:22:33:44", false, {}, "Too short - missing one octet"},
            {"00:11:22:33:44:55:66", false, {}, "Too long - extra octet"},
            {"00-11-22-33-44-55", false, {}, "Wrong separator - dashes"},
            {"00.11.22.33.44.55", false, {}, "Wrong separator - dots"},
            {"00:11:22:33:44:5g", false, {}, "Invalid hex character"},
            {"00:11:22:33:44:555", false, {}, "Invalid octet length"}};

    for (const auto &test_case : test_cases) {
        SCOPED_TRACE(test_case.description + " - Input: '" + std::string(test_case.input) + "'");

        const auto result = framework::net::MacAddress::from_string(test_case.input);

        if (test_case.should_succeed) {
            EXPECT_TRUE(result.has_value()) << "Expected parsing to succeed";
            if (result.has_value()) {
                EXPECT_EQ(*result, test_case.expected_mac) << "MAC address values should match";
            }
        } else {
            EXPECT_FALSE(result.has_value()) << "Expected parsing to fail";
        }
    }
}

TEST(DpdkUtils, EthernetHeaderConstruction) {
    // Test data
    const framework::net::MacAddress src_mac{{{0x00, 0x11, 0x22, 0x33, 0x44, 0x55}}};
    const framework::net::MacAddress dst_mac{{{0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff}}};
    static constexpr std::uint16_t IPV4_ETHER_TYPE = 0x0800;

    // Test parameterized constructor
    const framework::net::EthernetHeader header{src_mac, dst_mac, IPV4_ETHER_TYPE};

    EXPECT_EQ(header.src_mac(), src_mac) << "Source MAC should match";
    EXPECT_EQ(header.dest_mac(), dst_mac) << "Destination MAC should match";
    EXPECT_EQ(header.ether_type(), IPV4_ETHER_TYPE) << "EtherType should match";

    // Test equality operator
    const framework::net::EthernetHeader header_copy{src_mac, dst_mac, IPV4_ETHER_TYPE};
    EXPECT_EQ(header, header_copy) << "Headers with same values should be equal";

    // Test with different values
    const framework::net::MacAddress different_src{{{0xff, 0xee, 0xdd, 0xcc, 0xbb, 0xaa}}};
    const framework::net::EthernetHeader different_header{different_src, dst_mac, IPV4_ETHER_TYPE};
    EXPECT_NE(header, different_header) << "Headers with different values should not be equal";
}

TEST(DpdkUtils, SetPortMtu) {
    const auto setup_opt = framework::net::configure_test_dpdk_port();
    ASSERT_TRUE(setup_opt.has_value()) << setup_opt.error();
    const auto &setup = setup_opt.value(); // NOLINT(bugprone-unchecked-optional-access)
    const uint16_t valid_port_id = setup.port_id;

    struct TestCase {
        uint16_t port_id{};
        uint16_t mtu{};
        int expected_result{};
        std::string description;
    };

    const std::vector<TestCase> test_cases{
            // Valid MTU values with valid port (fail due to unconfigured port in test
            // environment)
            {valid_port_id, 1500, 0, "Standard Ethernet MTU"},
            {valid_port_id, 9000, 0, "Jumbo frame MTU (9K)"},
            {valid_port_id, 1024, 0, "Custom MTU value"},

            // Invalid port IDs
            {999, 1500, -1, "Non-existent port ID"},

            // Invalid MTU values with valid port (fail due to invalid MTU)
            {valid_port_id, 0, -1, "Zero MTU value"},
            {valid_port_id, 1, -1, "Below device minimum MTU"},
            {valid_port_id, 65535, -1, "Above device maximum frame size"}};

    for (const auto &test_case : test_cases) {
        SCOPED_TRACE(
                test_case.description + " - Port: " + std::to_string(test_case.port_id) +
                ", MTU: " + std::to_string(test_case.mtu));

        const auto result = framework::net::dpdk_set_port_mtu(test_case.port_id, test_case.mtu);
        if (test_case.expected_result == 0) {
            EXPECT_FALSE(result) << "Expected success but got: "
                                 << framework::net::get_error_name(result);
        } else {
            EXPECT_TRUE(result) << "Expected failure but got success";
        }
    }
}

TEST(DpdkUtils, TuneVirtualMemory) {
    // This test verifies that the function runs without crashing and returns 0
    // The actual behavior depends on system permissions and current VM settings
    // The function may not be able to read/write VM parameters in test
    // environments
    const auto result = framework::net::dpdk_try_tune_virtual_memory();
    EXPECT_FALSE(result) << "Expected success but got: " << framework::net::get_error_name(result);
}

TEST(DpdkUtils, DisableEthernetFlowControl) {
    const auto setup_opt = framework::net::configure_test_dpdk_port();
    ASSERT_TRUE(setup_opt.has_value()) << setup_opt.error();
    const auto &setup = setup_opt.value(); // NOLINT(bugprone-unchecked-optional-access)
    const uint16_t valid_port_id = setup.port_id;
    const auto result = framework::net::dpdk_disable_ethernet_flow_control(valid_port_id);
    EXPECT_FALSE(result) << "Expected success but got: " << framework::net::get_error_name(result);
}

TEST(DpdkUtils, CalculateTimestampOffsets) {
    int timestamp_offset{};
    uint64_t timestamp_mask{};

    const auto result =
            framework::net::dpdk_calculate_timestamp_offsets(timestamp_offset, timestamp_mask);

    EXPECT_FALSE(result) << "Expected success but got: " << framework::net::get_error_name(result);
    EXPECT_GE(timestamp_offset, 0) << "Timestamp offset should be non-negative";
    EXPECT_GT(timestamp_mask, 0) << "Timestamp mask should be greater than 0";
}

TEST(DpdkUtils, CheckPcieMaxReadRequestSize) {
    const auto nics = framework::net::discover_mellanox_nics();
    ASSERT_FALSE(nics.empty()) << "No Mellanox NICs found for testing";

    bool found_match = false;

    for (const auto &[value, name] : ::wise_enum::range<framework::net::PcieMrrs>) {
        const auto result =
                framework::net::dpdk_check_pcie_max_read_request_size(nics.front(), value);

        const auto success = result == framework::net::DpdkErrc::Success;
        const auto mismatch = result == framework::net::DpdkErrc::PcieVerifyMismatch;
        EXPECT_TRUE(success || mismatch)
                << "Unexpected result: " << framework::net::get_error_name(result)
                << " for MRRS: " << name;

        if (success) {
            found_match = true;
            RT_LOGC_DEBUG(framework::net::Net::NetDpdk, "Found matching MRRS: {}", name);
            break;
        }
    }

    EXPECT_TRUE(found_match);
}

TEST(DpdkUtils, LogLinkInfo) {
    const auto setup_opt = framework::net::configure_test_dpdk_port();
    ASSERT_TRUE(setup_opt.has_value()) << setup_opt.error();
    const auto &setup = setup_opt.value(); // NOLINT(bugprone-unchecked-optional-access)
    const uint16_t valid_port_id = setup.port_id;

    const auto result = framework::net::dpdk_log_link_info(valid_port_id);
    EXPECT_FALSE(result) << "Expected success but got: " << framework::net::get_error_name(result);
}

TEST(DpdkUtils, IsLinkStatusUp) {
    const auto setup_opt = framework::net::configure_test_dpdk_port();
    ASSERT_TRUE(setup_opt.has_value()) << setup_opt.error();
    const auto &setup = setup_opt.value(); // NOLINT(bugprone-unchecked-optional-access)
    const uint16_t valid_port_id = setup.port_id;

    const auto result = framework::net::dpdk_is_link_status_up(valid_port_id);
    if (result == framework::net::make_error_code(framework::net::DpdkErrc::LinkDown)) {
        EXPECT_EQ(result, framework::net::make_error_code(framework::net::DpdkErrc::LinkDown))
                << "Link is down for port " << valid_port_id;
    } else {
        EXPECT_FALSE(result) << "Expected success or link_down but got: "
                             << framework::net::get_error_name(result);
    }

    static constexpr uint16_t INVALID_PORT_ID = 999;
    const auto invalid_result = framework::net::dpdk_is_link_status_up(INVALID_PORT_ID);
    EXPECT_TRUE(invalid_result) << "Expected failure for invalid port ID";
    EXPECT_EQ(
            invalid_result,
            framework::net::make_error_code(framework::net::DpdkErrc::LinkInfoFailed))
            << "Expected link_info_failed for invalid port ID";
}

TEST(DpdkUtils, EnableFlowRuleIsolation) {
    const auto setup_opt = framework::net::configure_test_dpdk_port();
    ASSERT_TRUE(setup_opt.has_value()) << setup_opt.error();
    const auto &setup = setup_opt.value(); // NOLINT(bugprone-unchecked-optional-access)
    const uint16_t valid_port_id = setup.port_id;

    const auto result = framework::net::dpdk_enable_flow_rule_isolation(valid_port_id);
    EXPECT_FALSE(result) << "Expected success but got: " << framework::net::get_error_name(result);
}

TEST(DpdkUtils, LogStats) {
    const auto setup_opt = framework::net::configure_test_dpdk_port();
    ASSERT_TRUE(setup_opt.has_value()) << setup_opt.error();
    const auto &setup = setup_opt.value(); // NOLINT(bugprone-unchecked-optional-access)
    const uint16_t valid_port_id = setup.port_id;

    const auto result = framework::net::dpdk_log_stats(valid_port_id);
    EXPECT_FALSE(result) << "Expected success but got: " << framework::net::get_error_name(result);
}

TEST(DpdkUtils, StartStopEthDev) {
    const auto setup_opt = framework::net::configure_test_dpdk_port();
    ASSERT_TRUE(setup_opt.has_value()) << setup_opt.error();
    const auto &setup = setup_opt.value(); // NOLINT(bugprone-unchecked-optional-access)
    const uint16_t valid_port_id = setup.port_id;

    const auto start_result = framework::net::dpdk_start_eth_dev(valid_port_id);
    EXPECT_FALSE(start_result) << "Expected start success but got: "
                               << framework::net::get_error_name(start_result);
    const auto stop_result = framework::net::dpdk_stop_eth_dev(valid_port_id);
    EXPECT_FALSE(stop_result) << "Expected stop success but got: "
                              << framework::net::get_error_name(stop_result);
}

TEST(DpdkUtils, ValidateMellanoxDriver) {
    const auto setup_opt = framework::net::configure_test_dpdk_port();
    ASSERT_TRUE(setup_opt.has_value()) << setup_opt.error();
    const auto &setup = setup_opt.value(); // NOLINT(bugprone-unchecked-optional-access)
    const uint16_t valid_port_id = setup.port_id;

    const auto valid_result = framework::net::dpdk_validate_mellanox_driver(valid_port_id);
    EXPECT_FALSE(valid_result) << "Expected success but got: "
                               << framework::net::get_error_name(valid_result);

    static constexpr uint16_t INVALID_PORT_ID = 999;
    const auto invalid_result = framework::net::dpdk_validate_mellanox_driver(INVALID_PORT_ID);
    EXPECT_TRUE(invalid_result) << "Expected failure but got success";
}

TEST(DpdkUtils, MempoolOperations) {
    static constexpr uint32_t TEST_NUM_MBUFS = 1024;
    static constexpr uint32_t TEST_MTU_SIZE = 1500;

    const auto setup_opt = framework::net::configure_test_dpdk_port();
    ASSERT_TRUE(setup_opt.has_value()) << setup_opt.error();
    const auto &setup = setup_opt.value(); // NOLINT(bugprone-unchecked-optional-access)
    const uint16_t valid_port_id = setup.port_id;

    struct TestCase {
        std::string name;
        std::uint32_t num_mbufs{};
        std::uint32_t mtu_size{};
        framework::net::HostPinned host_pinned{};
        bool should_succeed{};
        std::string description;
    };

    const std::vector<TestCase> test_cases{
            // Valid mempool configurations
            {"test_mempool_regular",
             TEST_NUM_MBUFS,
             TEST_MTU_SIZE,
             framework::net::HostPinned::No,
             true,
             "Regular mempool with standard MTU"},
            {"test_mempool_pinned",
             512,
             1500,
             framework::net::HostPinned::Yes,
             true,
             "Host-pinned mempool with standard MTU"},
            {"test_mempool_large",
             2048,
             9000,
             framework::net::HostPinned::No,
             true,
             "Large mempool for jumbo frames"},
            {"test_mempool_small",
             256,
             64,
             framework::net::HostPinned::Yes,
             true,
             "Small host-pinned mempool"},

            // Edge cases and error conditions
            {"", 1024, 1500, framework::net::HostPinned::No, false, "Empty name should fail"},
            {"test_mempool_zero_mbufs",
             0,
             1500,
             framework::net::HostPinned::No,
             false,
             "Zero mbuf count should fail"}};

    for (const auto &test_case : test_cases) {
        SCOPED_TRACE(test_case.description + " - Name: '" + test_case.name + "'");

        rte_mempool *mempool{};
        const auto create_result = framework::net::dpdk_create_mempool(
                test_case.name,
                valid_port_id,
                test_case.num_mbufs,
                test_case.mtu_size,
                test_case.host_pinned,
                &mempool);

        // Handle failure cases first (early return pattern)
        if (!test_case.should_succeed) {
            EXPECT_TRUE(create_result) << "Expected mempool creation to fail but got success";
            EXPECT_EQ(mempool, nullptr) << "Mempool pointer should be null on failure";
            continue;
        }

        // Handle success cases with reduced indentation
        EXPECT_FALSE(create_result) << "Expected mempool creation to succeed but got: "
                                    << framework::net::get_error_name(create_result);
        EXPECT_NE(mempool, nullptr) << "Mempool pointer should not be null";

        if (mempool == nullptr) {
            continue; // Skip further tests if mempool creation failed
        }

        // Verify mempool properties
        EXPECT_STREQ(static_cast<const char *>(mempool->name), test_case.name.c_str())
                << "Mempool name should match";
        EXPECT_GT(rte_mempool_avail_count(mempool), 0U)
                << "Available count should be greater than 0";

        // Test basic bulk operations (simplified)
        static constexpr uint32_t TEST_BULK_SIZE = 4U;
        std::vector<void *> obj_table(TEST_BULK_SIZE);

        // Test bulk get and put
        const int get_ret = rte_mempool_get_bulk(mempool, obj_table.data(), TEST_BULK_SIZE);
        EXPECT_EQ(get_ret, 0) << "Bulk get should succeed";

        if (get_ret == 0) {
            // Verify objects are valid
            EXPECT_NE(obj_table[0], nullptr) << "First object should not be null";

            // Put them back
            rte_mempool_put_bulk(mempool, obj_table.data(), TEST_BULK_SIZE);

            // Verify count is restored
            EXPECT_EQ(rte_mempool_avail_count(mempool), test_case.num_mbufs)
                    << "Available count should be restored";
        }

        // Test destruction
        const auto destroy_result = framework::net::dpdk_destroy_mempool(mempool);
        EXPECT_FALSE(destroy_result) << "Expected mempool destruction to succeed but got: "
                                     << framework::net::get_error_name(destroy_result);
    }
}

TEST(DpdkUtils, MempoolNullOutputParameter) {
    static constexpr uint32_t TEST_NUM_MBUFS = 1024;
    static constexpr uint32_t TEST_MTU_SIZE = 1500;

    // Test null mempool output parameter should fail
    const auto null_result = framework::net::dpdk_create_mempool(
            "test_null_output",
            0,
            TEST_NUM_MBUFS,
            TEST_MTU_SIZE,
            framework::net::HostPinned::No,
            nullptr);
    EXPECT_TRUE(null_result) << "Expected failure for null mempool output parameter";
}

TEST(DpdkUtils, DestroyNullMempool) {
    // Test that destroying a null mempool is safe and returns success
    const auto destroy_null_result = framework::net::dpdk_destroy_mempool(nullptr);
    EXPECT_FALSE(destroy_null_result)
            << "Expected success but got: " << framework::net::get_error_name(destroy_null_result);
}

class EthSendTest : public ::testing::Test {
protected:
    static constexpr uint32_t TEST_NUM_MBUFS = 1024;
    static constexpr uint32_t TEST_MTU_SIZE = 1500;
    static constexpr uint16_t QUEUE_ID = 0;

    void SetUp() override {
        // Setup DPDK port
        setup_ = framework::net::configure_test_dpdk_port();
        ASSERT_TRUE(setup_.has_value()) << setup_.error();

        // Create mempool
        ASSERT_TRUE(setup_.has_value());
        const auto mempool_result = framework::net::dpdk_create_mempool(
                "test_eth_send_pool",
                setup_->port_id, // NOLINT(bugprone-unchecked-optional-access)
                TEST_NUM_MBUFS,
                TEST_MTU_SIZE,
                framework::net::HostPinned::No,
                &mempool_);
        ASSERT_FALSE(mempool_result)
                << "Mempool creation failed: " << framework::net::get_error_name(mempool_result);

        // Start ethernet device
        const auto start_result = framework::net::dpdk_start_eth_dev(
                setup_->port_id); // NOLINT(bugprone-unchecked-optional-access)
        ASSERT_FALSE(start_result)
                << "Device start failed: " << framework::net::get_error_name(start_result);
    }

    void TearDown() override {
        // Stop ethernet device
        if (setup_.has_value()) {
            const auto stop_result = framework::net::dpdk_stop_eth_dev(setup_->port_id);
            EXPECT_FALSE(stop_result)
                    << "Device stop failed: " << framework::net::get_error_name(stop_result);
        }

        // Destroy mempool
        if (mempool_ != nullptr) {
            const auto cleanup_result = framework::net::dpdk_destroy_mempool(mempool_);
            EXPECT_FALSE(cleanup_result)
                    << "Mempool cleanup failed: " << framework::net::get_error_name(cleanup_result);
        }
    }

    tl::expected<framework::net::TestDpdkSetup, std::string> setup_{
            tl::unexpected("Not initialized")};
    rte_mempool *mempool_{};
};

TEST_F(EthSendTest, EthSend) {
    // Test null mempool fails
    const framework::net::MacAddress src_mac{{{0x00, 0x11, 0x22, 0x33, 0x44, 0x55}}};
    const framework::net::MacAddress dst_mac{{{0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff}}};
    const framework::net::EthernetHeader eth_header{src_mac, dst_mac, 0x0800};

    // Create test messages as byte arrays directly
    static constexpr std::array<uint8_t, 5> MSG1{'t', 'e', 's', 't', '1'};
    static constexpr std::array<uint8_t, 5> MSG2{'t', 'e', 's', 't', '2'};

    const std::array<std::span<const uint8_t>, 2> messages{std::span{MSG1}, std::span{MSG2}};
    const std::span<const std::span<const uint8_t>> messages_span{messages};

    const auto null_result =
            framework::net::dpdk_eth_send(messages_span, eth_header, nullptr, QUEUE_ID, 0);
    EXPECT_EQ(
            null_result,
            framework::net::make_error_code(framework::net::DpdkErrc::MbufAllocFailed));

    // Test successful send with proper setup
    ASSERT_TRUE(setup_.has_value());
    const auto send_result = framework::net::dpdk_eth_send(
            messages_span,
            eth_header,
            mempool_,
            QUEUE_ID,
            setup_->port_id); // NOLINT(bugprone-unchecked-optional-access)
    EXPECT_FALSE(send_result) << "Packet send failed: "
                              << framework::net::get_error_name(send_result);
}

TEST_F(EthSendTest, EthSendMbufs) {
    // Allocate 2 mbufs using bulk allocation
    std::array<rte_mbuf *, 2> mbufs{};
    const int alloc_result = rte_pktmbuf_alloc_bulk(mempool_, mbufs.data(), 2);
    ASSERT_EQ(alloc_result, 0) << "Failed to allocate mbufs from pool";

    // Test sending the allocated mbufs
    const std::span<rte_mbuf *> mbuf_span{mbufs};
    ASSERT_TRUE(setup_.has_value());
    const auto send_result =
            // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
            framework::net::dpdk_eth_send_mbufs(mbuf_span, QUEUE_ID, setup_->port_id);
    EXPECT_FALSE(send_result) << "Mbuf send failed: "
                              << framework::net::get_error_name(send_result);
}

TEST_F(EthSendTest, EthSendMbufsEmpty) {
    // Test empty mbuf array - should succeed with no operation
    const std::span<rte_mbuf *> empty_span{};
    ASSERT_TRUE(setup_.has_value());
    const auto send_result =
            // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
            framework::net::dpdk_eth_send_mbufs(empty_span, QUEUE_ID, setup_->port_id);
    EXPECT_FALSE(send_result) << "Empty mbuf send should succeed: "
                              << framework::net::get_error_name(send_result);
}

TEST_F(EthSendTest, EthSendMbufsTooLarge) {
    // Test oversized mbuf array - create a vector larger than uint16_t::max
    static constexpr std::size_t OVERSIZED_COUNT =
            static_cast<std::size_t>(std::numeric_limits<std::uint16_t>::max()) + 1;

    // Create a vector with null pointers
    std::vector<rte_mbuf *> oversized_mbufs(OVERSIZED_COUNT, nullptr);
    const std::span<rte_mbuf *> oversized_span{oversized_mbufs};

    ASSERT_TRUE(setup_.has_value());
    const auto send_result =
            // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
            framework::net::dpdk_eth_send_mbufs(oversized_span, QUEUE_ID, setup_->port_id);

    // NEW BEHAVIOR: Function fails with bounds checking
    EXPECT_TRUE(send_result) << "Oversized mbuf array should fail with bounds checking";
    EXPECT_EQ(
            send_result,
            framework::net::make_error_code(framework::net::DpdkErrc::InvalidParameter));
}

} // namespace
