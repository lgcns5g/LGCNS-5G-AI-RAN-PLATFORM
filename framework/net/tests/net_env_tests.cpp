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
 * @file net_env_tests.cpp
 * @brief Unit tests for Net environment error checking logic
 */

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <tl/expected.hpp>

#include <gtest/gtest.h>

#include "net/doca_rxq.hpp"
#include "net/doca_txq.hpp"
#include "net/doca_types.hpp"
#include "net/dpdk_txq.hpp"
#include "net/dpdk_types.hpp"
#include "net/env.hpp"
#include "net/gpu.hpp"
#include "net/mempool.hpp"
#include "net/nic.hpp"

namespace {

namespace fn = ::framework::net;

TEST(EnvValidationTest, InvalidGpuDeviceIds) {
    fn::EnvConfig config{};
    config.nic_config.nic_pcie_addr = "0000:3a:00.0";
    config.nic_config.doca_txq_configs.push_back(fn::DocaTxQConfig{});

    // Test invalid large GPU device ID
    static constexpr uint32_t INVALID_LARGE_GPU_ID = 1000000U;
    config.gpu_device_id = fn::GpuDeviceId{INVALID_LARGE_GPU_ID};
    EXPECT_THROW((fn::Env{config}), std::invalid_argument);
}

TEST(EnvValidationTest, InvalidNicConfiguration) {
    fn::EnvConfig config{};
    config.gpu_device_id = fn::GpuDeviceId{0U};
    config.nic_config.doca_txq_configs.push_back(fn::DocaTxQConfig{});

    // Test invalid NIC PCIe address
    config.nic_config.nic_pcie_addr = "invalid_pcie_address";
    EXPECT_THROW((fn::Env{config}), std::invalid_argument);
}

TEST(GpuDeviceIdTest, ValidConstruction) {
    // Test valid construction with uint32_t (negative values impossible)
    EXPECT_NO_THROW((fn::GpuDeviceId{0U}));
    EXPECT_NO_THROW((fn::GpuDeviceId{1U}));
    EXPECT_NO_THROW((fn::GpuDeviceId{100U}));
}

TEST(NicValidationTest, InvalidArguments) {
    static constexpr std::uint16_t STANDARD_MTU_SIZE = 1500;
    fn::NicConfig config{};

    // Test null GPU device when DOCA queues are configured
    config.nic_pcie_addr = "0000:3a:00.0";
    config.max_mtu_size = STANDARD_MTU_SIZE;
    config.doca_txq_configs.push_back(fn::DocaTxQConfig{});
    EXPECT_THROW((fn::Nic{config, std::nullopt}), std::invalid_argument);

    // Test empty PCIe address
    config.nic_pcie_addr = "";
    config.max_mtu_size = STANDARD_MTU_SIZE;
    config.doca_txq_configs.push_back(fn::DocaTxQConfig{});
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    auto *dummy_gpu = reinterpret_cast<doca_gpu *>(0x1);
    EXPECT_THROW((fn::Nic{config, std::optional{dummy_gpu}}), std::invalid_argument);

    // Test zero MTU size
    config.nic_pcie_addr = "0000:3a:00.0";
    config.max_mtu_size = 0;
    EXPECT_THROW((fn::Nic{config, std::optional{dummy_gpu}}), std::invalid_argument);

    // Test no queue configurations
    config.max_mtu_size = STANDARD_MTU_SIZE;
    config.doca_txq_configs.clear();
    config.doca_rxq_configs.clear();
    config.dpdk_txq_configs.clear();
    EXPECT_THROW((fn::Nic{config, std::optional{dummy_gpu}}), std::invalid_argument);
}

TEST(DocaTxQValidationTest, InvalidArguments) {
    static constexpr std::uint32_t STANDARD_PKT_SIZE = 1500;
    static constexpr std::uint32_t STANDARD_PKT_NUM = 100;
    static constexpr std::uint32_t STANDARD_DESCR_NUM = 256;

    fn::DocaTxQConfig config{};

    // Test null devices
    EXPECT_THROW((fn::DocaTxQ{config, nullptr, nullptr}), std::invalid_argument);

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    auto *dummy_gpu = reinterpret_cast<doca_gpu *>(0x1);
    EXPECT_THROW((fn::DocaTxQ{config, dummy_gpu, nullptr}), std::invalid_argument);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    auto *dummy_dev = reinterpret_cast<doca_dev *>(0x1);
    EXPECT_THROW((fn::DocaTxQ{config, nullptr, dummy_dev}), std::invalid_argument);

    // Test empty addresses
    config.nic_pcie_addr = "";
    // NOLINTBEGIN(bugprone-unchecked-optional-access)

    config.dest_mac_addr = fn::MacAddress::from_string("aa:bb:cc:dd:ee:ff").value();
    config.pkt_size = STANDARD_PKT_SIZE;
    config.pkt_num = STANDARD_PKT_NUM;
    config.max_sq_descr_num = STANDARD_DESCR_NUM;
    EXPECT_THROW((fn::DocaTxQ{config, dummy_gpu, dummy_dev}), std::invalid_argument);

    config.nic_pcie_addr = "0000:3a:00.0";
    config.dest_mac_addr = fn::MacAddress{};
    EXPECT_THROW((fn::DocaTxQ{config, dummy_gpu, dummy_dev}), std::invalid_argument);

    // Test zero values
    config.dest_mac_addr = fn::MacAddress::from_string("aa:bb:cc:dd:ee:ff").value();
    // NOLINTEND(bugprone-unchecked-optional-access)

    config.pkt_size = 0;
    EXPECT_THROW((fn::DocaTxQ{config, dummy_gpu, dummy_dev}), std::invalid_argument);

    config.pkt_size = STANDARD_PKT_SIZE;
    config.pkt_num = 0;
    EXPECT_THROW((fn::DocaTxQ{config, dummy_gpu, dummy_dev}), std::invalid_argument);

    config.pkt_num = STANDARD_PKT_NUM;
    config.max_sq_descr_num = 0;
    EXPECT_THROW((fn::DocaTxQ{config, dummy_gpu, dummy_dev}), std::invalid_argument);
}

TEST(DocaRxQValidationTest, InvalidArguments) {
    static constexpr std::uint32_t STANDARD_PKT_NUM = 100;
    static constexpr std::uint32_t STANDARD_PKT_SIZE = 1500;
    static constexpr std::uint16_t IPV4_ETHER_TYPE = 0x0800;

    fn::DocaRxQConfig config{};

    // Test null devices
    EXPECT_THROW((fn::DocaRxQ{config, nullptr, nullptr}), std::invalid_argument);

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    auto *dummy_gpu = reinterpret_cast<doca_gpu *>(0x1);
    EXPECT_THROW((fn::DocaRxQ{config, dummy_gpu, nullptr}), std::invalid_argument);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    auto *dummy_dev = reinterpret_cast<doca_dev *>(0x1);
    EXPECT_THROW((fn::DocaRxQ{config, nullptr, dummy_dev}), std::invalid_argument);

    // Test empty addresses
    config.nic_pcie_addr = "";
    // NOLINTBEGIN(bugprone-unchecked-optional-access)
    config.sender_mac_addr = fn::MacAddress::from_string("aa:bb:cc:dd:ee:ff").value();
    config.max_pkt_num = STANDARD_PKT_NUM;
    config.max_pkt_size = STANDARD_PKT_SIZE;
    config.ether_type = IPV4_ETHER_TYPE;
    EXPECT_THROW((fn::DocaRxQ{config, dummy_gpu, dummy_dev}), std::invalid_argument);

    config.nic_pcie_addr = "0000:3a:00.0";
    config.sender_mac_addr = fn::MacAddress{};
    EXPECT_THROW((fn::DocaRxQ{config, dummy_gpu, dummy_dev}), std::invalid_argument);

    // Test zero values
    config.sender_mac_addr = fn::MacAddress::from_string("aa:bb:cc:dd:ee:ff").value();
    // NOLINTEND(bugprone-unchecked-optional-access)

    config.max_pkt_num = 0;
    EXPECT_THROW((fn::DocaRxQ{config, dummy_gpu, dummy_dev}), std::invalid_argument);

    config.max_pkt_num = STANDARD_PKT_NUM;
    config.max_pkt_size = 0;
    EXPECT_THROW((fn::DocaRxQ{config, dummy_gpu, dummy_dev}), std::invalid_argument);

    // Test zero ether_type
    config.max_pkt_size = STANDARD_PKT_SIZE;
    config.ether_type = 0;
    EXPECT_THROW((fn::DocaRxQ{config, dummy_gpu, dummy_dev}), std::invalid_argument);
}

TEST(DpdkTxQValidationTest, InvalidArguments) {
    static constexpr std::uint16_t STANDARD_PORT_ID = 0;
    static constexpr std::uint16_t STANDARD_TXQ_ID = 0;

    fn::DpdkTxQConfig config{};

    // Test zero txq_size
    config.txq_size = 0;
    EXPECT_THROW(
            (fn::DpdkTxQueue{STANDARD_PORT_ID, STANDARD_TXQ_ID, config}), std::invalid_argument);
}

// Test: Verifies Env construction fails with invalid DPDK configuration
TEST(EnvValidationTest, InvalidDpdkConfiguration) {
    const auto available_nics = fn::discover_mellanox_nics();
    ASSERT_FALSE(available_nics.empty()) << "No Mellanox NICs available for testing";

    fn::EnvConfig config{};
    config.gpu_device_id = fn::GpuDeviceId{0U};
    config.nic_config.nic_pcie_addr = available_nics.front();
    config.nic_config.doca_txq_configs.push_back(fn::DocaTxQConfig{});

    // Test with empty file_prefix (should cause validation error)
    config.dpdk_config.file_prefix = "";
    EXPECT_THROW((fn::Env{config}), std::invalid_argument);
}

TEST(EnvValidationTest, ValidConfiguration) {
    static constexpr std::uint32_t TX_PKT_SIZE = 1024;
    static constexpr std::uint32_t TX_PKT_NUM = 64;
    static constexpr std::uint32_t TX_DESCR_NUM = 8192;
    static constexpr std::uint32_t RX_PKT_SIZE = 4096;
    static constexpr std::uint32_t RX_PKT_NUM = 16384;
    static constexpr std::uint16_t TEST_ETHER_TYPE_1 = 0x88b5;
    static constexpr std::uint16_t TEST_ETHER_TYPE_2 = 0x88b6;
    static constexpr std::uint16_t DPDK_TXQ_SIZE_1 = 128;
    static constexpr std::uint16_t DPDK_TXQ_SIZE_2 = 128;
    static constexpr std::uint32_t MEMPOOL_NUM_MBUFS_1 = 1024;
    static constexpr std::uint32_t MEMPOOL_NUM_MBUFS_2 = 2048;
    static constexpr std::uint32_t MEMPOOL_MTU_SIZE = 1500;

    const auto available_nics = fn::discover_mellanox_nics();
    ASSERT_FALSE(available_nics.empty()) << "No Mellanox NICs available for testing";

    fn::EnvConfig config{};
    config.gpu_device_id = fn::GpuDeviceId{0U};               // Use first GPU
    config.nic_config.nic_pcie_addr = available_nics.front(); // Use first available NIC

    config.dpdk_config.file_prefix = "net_env_test_prefix";
    config.dpdk_config.dpdk_core_id = 0;

    // Create TX and RX queues for some test ether types
    const std::vector<std::uint16_t> ether_types{TEST_ETHER_TYPE_1, TEST_ETHER_TYPE_2};
    for (const auto ether_type : ether_types) {
        fn::DocaTxQConfig tx_config{};
        tx_config.nic_pcie_addr = available_nics.front();
        // NOLINTBEGIN(bugprone-unchecked-optional-access)
        tx_config.dest_mac_addr = fn::MacAddress::from_string("00:11:22:33:44:55").value();
        tx_config.pkt_size = TX_PKT_SIZE;
        tx_config.pkt_num = TX_PKT_NUM;
        tx_config.max_sq_descr_num = TX_DESCR_NUM;
        tx_config.ether_type = ether_type;
        config.nic_config.doca_txq_configs.push_back(tx_config);

        fn::DocaRxQConfig rx_config{};
        rx_config.nic_pcie_addr = available_nics.front();
        rx_config.sender_mac_addr = fn::MacAddress::from_string("00:11:22:33:44:55").value();
        // NOLINTEND(bugprone-unchecked-optional-access)
        rx_config.max_pkt_num = RX_PKT_NUM;
        rx_config.max_pkt_size = RX_PKT_SIZE;
        rx_config.ether_type = ether_type;
        config.nic_config.doca_rxq_configs.push_back(rx_config);
    }

    // Create two DPDK TX queues
    fn::DpdkTxQConfig dpdk_txq_config_1{};
    dpdk_txq_config_1.txq_size = DPDK_TXQ_SIZE_1;
    config.nic_config.dpdk_txq_configs.push_back(dpdk_txq_config_1);

    fn::DpdkTxQConfig dpdk_txq_config_2{};
    dpdk_txq_config_2.txq_size = DPDK_TXQ_SIZE_2;
    config.nic_config.dpdk_txq_configs.push_back(dpdk_txq_config_2);

    // Create two mempools
    fn::MempoolConfig mempool_config_1{};
    mempool_config_1.name = "test_mempool_1";
    mempool_config_1.num_mbufs = MEMPOOL_NUM_MBUFS_1;
    mempool_config_1.mtu_size = MEMPOOL_MTU_SIZE;
    mempool_config_1.host_pinned = fn::HostPinned::No;
    config.nic_config.mempool_configs.push_back(mempool_config_1);

    fn::MempoolConfig mempool_config_2{};
    mempool_config_2.name = "test_mempool_2";
    mempool_config_2.num_mbufs = MEMPOOL_NUM_MBUFS_2;
    mempool_config_2.mtu_size = MEMPOOL_MTU_SIZE;
    mempool_config_2.host_pinned = fn::HostPinned::Yes;
    config.nic_config.mempool_configs.push_back(mempool_config_2);

    // This should succeed with valid configuration
    EXPECT_NO_THROW((fn::Env{config}));
}

TEST(EnvValidationTest, MempoolInvalidArguments) {
    static constexpr std::uint32_t STANDARD_NUM_MBUFS = 1024;
    static constexpr std::uint32_t STANDARD_MTU_SIZE = 1500;
    static constexpr std::uint16_t STANDARD_PORT_ID = 0;

    fn::MempoolConfig config{};

    // Test empty name
    config.name = "";
    config.num_mbufs = STANDARD_NUM_MBUFS;
    config.mtu_size = STANDARD_MTU_SIZE;
    config.host_pinned = fn::HostPinned::No;
    EXPECT_THROW((fn::Mempool{STANDARD_PORT_ID, config}), std::invalid_argument);

    // Test zero num_mbufs
    config.name = "test_mempool";
    config.num_mbufs = 0;
    EXPECT_THROW((fn::Mempool{STANDARD_PORT_ID, config}), std::invalid_argument);

    // Test zero mtu_size
    config.num_mbufs = STANDARD_NUM_MBUFS;
    config.mtu_size = 0;
    EXPECT_THROW((fn::Mempool{STANDARD_PORT_ID, config}), std::invalid_argument);
}

} // namespace
