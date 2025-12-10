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
 * @file net_sample_tests.cpp
 * @brief Documentation sample tests for the net library
 *
 * This file contains tests that serve as executable documentation examples.
 * Code snippets from these tests are extracted and included in the Sphinx documentation.
 */

#include <iostream>
#include <optional>
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

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers,hicpp-uppercase-literal-suffix,readability-uppercase-literal-suffix)

using EnvConfig = framework::net::EnvConfig;
using GpuDeviceId = framework::net::GpuDeviceId;
using DocaTxQConfig = framework::net::DocaTxQConfig;
using MacAddress = framework::net::MacAddress;
using DocaRxQConfig = framework::net::DocaRxQConfig;
using DpdkTxQConfig = framework::net::DpdkTxQConfig;
using MempoolConfig = framework::net::MempoolConfig;
using HostPinned = framework::net::HostPinned;
using DpdkConfig = framework::net::DpdkConfig;
using Env = framework::net::Env;
using framework::net::discover_mellanox_nics;

// Test: Demonstrates basic environment configuration
TEST(NetSampleTests, BasicEnvironmentConfiguration) {
    const auto available_nics = framework::net::discover_mellanox_nics();
    ASSERT_FALSE(available_nics.empty()) << "No Mellanox NICs available for testing";

    // example-begin basic-environment-config-1
    EnvConfig config{};
    config.gpu_device_id = GpuDeviceId{0};            // Use first GPU
    config.nic_config.nic_pcie_addr = "0000:3a:00.0"; // NIC PCIe address

    // Configure DPDK
    config.dpdk_config.app_name = "my_app";
    config.dpdk_config.file_prefix = "my_prefix";
    config.dpdk_config.dpdk_core_id = 0;
    // example-end basic-environment-config-1

    config.nic_config.nic_pcie_addr = available_nics.front();
}

// Test: Demonstrates DOCA TX queue configuration
TEST(NetSampleTests, DocaTxQueueConfiguration) {
    const auto available_nics = framework::net::discover_mellanox_nics();
    ASSERT_FALSE(available_nics.empty()) << "No Mellanox NICs available for testing";

    // NOLINTBEGIN(bugprone-unchecked-optional-access)
    // example-begin doca-tx-queue-config-1
    DocaTxQConfig tx_config{};
    tx_config.nic_pcie_addr = "0000:3a:00.0";
    tx_config.dest_mac_addr = MacAddress::from_string("aa:bb:cc:dd:ee:ff").value();
    tx_config.pkt_size = 1024;
    tx_config.pkt_num = 64;
    tx_config.max_sq_descr_num = 8192;
    tx_config.ether_type = 0x88b5;
    // example-end doca-tx-queue-config-1
    // NOLINTEND(bugprone-unchecked-optional-access)

    tx_config.nic_pcie_addr = available_nics.front();
}

// Test: Demonstrates DOCA TX queue with VLAN tagging
TEST(NetSampleTests, DocaTxQueueWithVlan) {
    const auto available_nics = framework::net::discover_mellanox_nics();
    ASSERT_FALSE(available_nics.empty()) << "No Mellanox NICs available for testing";

    // NOLINTBEGIN(bugprone-unchecked-optional-access)
    // example-begin doca-tx-queue-vlan-1
    DocaTxQConfig tx_config{};
    tx_config.nic_pcie_addr = "0000:3a:00.0";
    tx_config.dest_mac_addr = MacAddress::from_string("aa:bb:cc:dd:ee:ff").value();
    tx_config.pkt_size = 1024;
    tx_config.pkt_num = 64;
    tx_config.max_sq_descr_num = 8192;
    tx_config.ether_type = 0x88b5;
    tx_config.vlan_tci = 100; // VLAN ID 100 (inserts 802.1Q tag)
    // example-end doca-tx-queue-vlan-1
    // NOLINTEND(bugprone-unchecked-optional-access)

    tx_config.nic_pcie_addr = available_nics.front();
}

// Test: Demonstrates DOCA RX queue configuration
TEST(NetSampleTests, DocaRxQueueConfiguration) {
    const auto available_nics = framework::net::discover_mellanox_nics();
    ASSERT_FALSE(available_nics.empty()) << "No Mellanox NICs available for testing";

    // NOLINTBEGIN(bugprone-unchecked-optional-access)
    // example-begin doca-rx-queue-config-1
    DocaRxQConfig rx_config{};
    rx_config.nic_pcie_addr = "0000:3a:00.0";
    rx_config.sender_mac_addr = MacAddress::from_string("aa:bb:cc:dd:ee:ff").value();
    rx_config.max_pkt_num = 16384; // Maximum packets in queue
    rx_config.max_pkt_size = 1024; // Maximum packet size
    rx_config.ether_type = 0x88b5; // Filter by EtherType
    // example-end doca-rx-queue-config-1
    // NOLINTEND(bugprone-unchecked-optional-access)

    rx_config.nic_pcie_addr = available_nics.front();
}

// Test: Demonstrates DOCA RX queue with VLAN filtering
TEST(NetSampleTests, DocaRxQueueWithVlan) {
    const auto available_nics = framework::net::discover_mellanox_nics();
    ASSERT_FALSE(available_nics.empty()) << "No Mellanox NICs available for testing";

    // NOLINTBEGIN(bugprone-unchecked-optional-access)
    // example-begin doca-rx-queue-vlan-1
    DocaRxQConfig rx_config{};
    rx_config.nic_pcie_addr = "0000:3a:00.0";
    rx_config.sender_mac_addr = MacAddress::from_string("aa:bb:cc:dd:ee:ff").value();
    rx_config.max_pkt_num = 16384;
    rx_config.max_pkt_size = 1024;
    rx_config.ether_type = 0x88b5;
    rx_config.vlan_tci = 100; // Filter VLAN ID 100
    // example-end doca-rx-queue-vlan-1
    // NOLINTEND(bugprone-unchecked-optional-access)

    rx_config.nic_pcie_addr = available_nics.front();
}

// Test: Demonstrates DPDK TX queue configuration for CPU-only mode
TEST(NetSampleTests, DpdkTxQueueConfiguration) {
    // example-begin dpdk-tx-queue-config-1
    DpdkTxQConfig dpdk_tx_config{};
    dpdk_tx_config.txq_size = 128; // TX queue size
    // example-end dpdk-tx-queue-config-1

    EXPECT_EQ(dpdk_tx_config.txq_size, 128);
}

// Test: Demonstrates mempool configuration
TEST(NetSampleTests, MempoolConfiguration) {
    // example-begin mempool-config-1
    MempoolConfig mempool_config{};
    mempool_config.name = "my_mempool";
    mempool_config.num_mbufs = 1024; // Number of buffers
    mempool_config.mtu_size = 1514;  // MTU size
    mempool_config.host_pinned = HostPinned::No;
    // example-end mempool-config-1
}

// Test: Demonstrates host-pinned mempool
TEST(NetSampleTests, HostPinnedMempool) {
    // example-begin mempool-host-pinned-1
    MempoolConfig mempool_config{};
    mempool_config.name = "pinned_mempool";
    mempool_config.num_mbufs = 2048;
    mempool_config.mtu_size = 9000;               // Jumbo frames
    mempool_config.host_pinned = HostPinned::Yes; // Pin memory
    // example-end mempool-host-pinned-1
}

// Test: Demonstrates multiple mempool configuration
TEST(NetSampleTests, MultipleMempools) {
    const auto available_nics = framework::net::discover_mellanox_nics();
    ASSERT_FALSE(available_nics.empty()) << "No Mellanox NICs available for testing";

    framework::net::EnvConfig config{};
    config.nic_config.nic_pcie_addr = available_nics.front();

    // example-begin multiple-mempools-1
    // Small packet mempool
    MempoolConfig small_pool{};
    small_pool.name = "small_packets";
    small_pool.num_mbufs = 1024;
    small_pool.mtu_size = 1514;
    small_pool.host_pinned = HostPinned::No;
    config.nic_config.mempool_configs.push_back(small_pool);

    // Large packet mempool
    MempoolConfig large_pool{};
    large_pool.name = "large_packets";
    large_pool.num_mbufs = 512;
    large_pool.mtu_size = 9000;
    large_pool.host_pinned = HostPinned::Yes;
    config.nic_config.mempool_configs.push_back(large_pool);
    // example-end multiple-mempools-1
}

// Test: Demonstrates MAC address operations
TEST(NetSampleTests, MacAddressOperations) {
    // NOLINTBEGIN(bugprone-unchecked-optional-access)
    // example-begin mac-address-operations-1
    // Parse MAC address from string
    const auto mac = MacAddress::from_string("aa:bb:cc:dd:ee:ff");
    if (mac) {
        std::cout << "Valid MAC: " << mac->to_string() << "\n";
    }

    // Check for zero MAC address
    const MacAddress mac_addr{};
    if (mac_addr.is_zero()) {
        std::cout << "MAC address is all zeros\n";
    }

    // Compare MAC addresses
    const auto mac1 = MacAddress::from_string("aa:bb:cc:dd:ee:ff").value();
    const auto mac2 = MacAddress::from_string("aa:bb:cc:dd:ee:ff").value();
    if (mac1 == mac2) {
        std::cout << "MAC addresses match\n";
    }
    // example-end mac-address-operations-1
    // NOLINTEND(bugprone-unchecked-optional-access)
}

// Test: Demonstrates discovering available NICs
TEST(NetSampleTests, DiscoverNics) {
    // example-begin discover-nics-1
    // Discover all Mellanox NICs in the system
    const auto nics = discover_mellanox_nics();

    for (const auto &nic_addr : nics) {
        std::cout << "Found NIC: " << nic_addr << "\n";
    }

    // Use the first available NIC
    if (!nics.empty()) {
        EnvConfig config{};
        config.nic_config.nic_pcie_addr = nics.front();
    }
    // example-end discover-nics-1
}

// Test: Demonstrates DPDK configuration options
TEST(NetSampleTests, DpdkConfiguration) {
    // example-begin dpdk-config-basic-1
    DpdkConfig dpdk_config{};
    dpdk_config.app_name = "my_network_app";
    dpdk_config.file_prefix = "myapp"; // Unique prefix for shared files
    dpdk_config.dpdk_core_id = 0;      // CPU core for DPDK main thread
    // example-end dpdk-config-basic-1

    // example-begin dpdk-config-verbose-1
    dpdk_config.verbose_logs = true; // Enable detailed DPDK logs
    // example-end dpdk-config-verbose-1

    // example-begin dpdk-config-rt-priority-1
    dpdk_config.enable_rt_priority_for_lcores = true; // Enable SCHED_FIFO
    // example-end dpdk-config-rt-priority-1
}

// Test: Demonstrates complete environment initialization and usage
TEST(NetSampleTests, CompleteEnvironmentUsage) {
    const auto available_nics = framework::net::discover_mellanox_nics();
    ASSERT_FALSE(available_nics.empty()) << "No Mellanox NICs available for testing";

    // Use actual NIC address for the example
    const std::string &nic_addr = available_nics.front();

    // NOLINTBEGIN(bugprone-unchecked-optional-access)
    // example-begin complete-environment-1
    // Configure environment
    EnvConfig config{};
    config.gpu_device_id = GpuDeviceId{0};
    config.nic_config.nic_pcie_addr = "0000:3a:00.0"; // Replace with your NIC address
    config.dpdk_config.file_prefix = "my_app_prefix";

    // Configure DOCA TX queue
    DocaTxQConfig tx_config{};
    tx_config.nic_pcie_addr = "0000:3a:00.0"; // Replace with your NIC address
    tx_config.dest_mac_addr = MacAddress::from_string("00:11:22:33:44:55").value();
    tx_config.pkt_size = 1024;
    tx_config.pkt_num = 64;
    tx_config.max_sq_descr_num = 8192;
    tx_config.ether_type = 0x88b5;
    config.nic_config.doca_txq_configs.push_back(tx_config);

    // Configure DOCA RX queue
    DocaRxQConfig rx_config{};
    rx_config.nic_pcie_addr = "0000:3a:00.0"; // Replace with your NIC address
    rx_config.sender_mac_addr = MacAddress::from_string("00:11:22:33:44:55").value();
    rx_config.max_pkt_num = 16384;
    rx_config.max_pkt_size = 1024;
    rx_config.ether_type = 0x88b5;
    config.nic_config.doca_rxq_configs.push_back(rx_config);
    // example-end complete-environment-1
    // NOLINTEND(bugprone-unchecked-optional-access)

    // Override example addresses with real NIC for testing
    config.nic_config.nic_pcie_addr = nic_addr;
    config.nic_config.doca_txq_configs.back().nic_pcie_addr = nic_addr;
    config.nic_config.doca_rxq_configs.back().nic_pcie_addr = nic_addr;

    // example-begin complete-environment-2
    // Initialize environment
    const Env env{config};

    // Check GPU availability
    if (env.has_gpu()) {
        const auto &gpu = env.gpu();
        // Verify GPU is initialized
        if (!gpu.pci_bus_id().empty()) {
            // Use the GPU
        }
    }

    // Access NIC and queues
    const auto &nic = env.nic();
    const auto &pci_addr = nic.pci_address();
    const auto &mac_addr = nic.mac_address();

    // Get DOCA TX queue for GPU kernel usage
    const auto *txq_params = nic.doca_tx_queue(0).params();

    // Get DOCA RX queue for GPU kernel usage
    const auto *rxq_params = nic.doca_rx_queue(0).params();
    // example-end complete-environment-2

    // Verify values
    EXPECT_FALSE(pci_addr.empty());
    EXPECT_FALSE(mac_addr.is_zero());
    EXPECT_NE(txq_params, nullptr);
    EXPECT_NE(rxq_params, nullptr);

    // example-begin queue-access-1
    // Access DOCA TX queues
    const auto &doca_tx_queues = nic.doca_tx_queues();
    const auto &doca_tx_queue_0 = nic.doca_tx_queue(0);

    // Access DOCA RX queues
    const auto &doca_rx_queues = nic.doca_rx_queues();
    const auto &doca_rx_queue_0 = nic.doca_rx_queue(0);
    // example-end queue-access-1

    // Verify queues
    EXPECT_FALSE(doca_tx_queues.empty());
    EXPECT_NE(doca_tx_queue_0.params(), nullptr);
    EXPECT_FALSE(doca_rx_queues.empty());
    EXPECT_NE(doca_rx_queue_0.params(), nullptr);

    // example-begin nic-information-1
    // Get NIC properties
    const auto &nic_pci_addr = nic.pci_address();
    const auto &nic_mac_addr = nic.mac_address();
    const auto nic_port_id = nic.dpdk_port_id();

    // Check RDMA support
    if (const auto rdma_result = nic.is_rdma_supported(); rdma_result) {
        if (*rdma_result) {
            std::cout << "RDMA supported\n";
        }
    }
    // example-end nic-information-1

    // Verify NIC info
    EXPECT_FALSE(nic_pci_addr.empty());
    EXPECT_FALSE(nic_mac_addr.is_zero());
    EXPECT_GE(nic_port_id, 0u);
}

// Test: Demonstrates CPU-only DPDK configuration
TEST(NetSampleTests, CpuOnlyConfiguration) {
    const auto available_nics = framework::net::discover_mellanox_nics();
    ASSERT_FALSE(available_nics.empty()) << "No Mellanox NICs available for testing";

    // example-begin cpu-only-config-1
    EnvConfig config{};
    config.nic_config.nic_pcie_addr = "0000:3a:00.0";
    config.dpdk_config.file_prefix = "cpu_app";

    // Configure DPDK TX queue instead of DOCA queue
    DpdkTxQConfig dpdk_tx_config{};
    dpdk_tx_config.txq_size = 128;
    config.nic_config.dpdk_txq_configs.push_back(dpdk_tx_config);

    // Configure mempool for buffer management
    MempoolConfig mempool_config{};
    mempool_config.name = "my_mempool";
    mempool_config.num_mbufs = 1024;
    mempool_config.mtu_size = 1514;
    mempool_config.host_pinned = HostPinned::No;
    config.nic_config.mempool_configs.push_back(mempool_config);
    // example-end cpu-only-config-1

    config.nic_config.nic_pcie_addr = available_nics.front();
}

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers,hicpp-uppercase-literal-suffix,readability-uppercase-literal-suffix)

} // namespace
