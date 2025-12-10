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
 * @file doca_utils_tests.cpp
 * @brief Unit tests for DOCA utilities
 */

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <doca_ctx.h>
#include <doca_dev.h>
#include <doca_error.h>
#include <doca_log.h>
#include <driver_types.h>
#include <quill/LogMacros.h>
#include <tl/expected.hpp>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include "log/rt_log_macros.hpp"
#include "net/details/doca_utils.hpp"
#include "net/doca_types.hpp"
#include "net/dpdk_types.hpp"
#include "net/net_log.hpp"
#include "net_test_helpers.hpp"

namespace {

void test_create_and_destroy_flow_rule(const std::optional<uint16_t> &vlan_opt) {
    const auto setup_opt = framework::net::configure_test_dpdk_port();
    ASSERT_TRUE(setup_opt.has_value()) << setup_opt.error();
    const auto &setup = setup_opt.value(); // NOLINT(bugprone-unchecked-optional-access)

    const auto pci_bus_id = framework::net::doca_device_id_to_pci_bus_id(0);
    doca_gpu *gpu_dev = nullptr;
    const auto gpu_result = framework::net::doca_open_cuda_device(pci_bus_id, &gpu_dev);
    ASSERT_EQ(gpu_result, DOCA_SUCCESS);

    static constexpr uint32_t TEST_MAX_PKT_NUM = 16384U;
    static constexpr uint32_t TEST_MAX_PKT_SIZE = 4096U;
    static constexpr uint32_t TEST_SEM_NUM_ITEMS = 4096U;
    static constexpr uint32_t TEST_SEM_ITEM_SIZE = 4U;

    const framework::net::DocaSemItems sem_items{
            .num_items = TEST_SEM_NUM_ITEMS, .item_size = TEST_SEM_ITEM_SIZE};

    framework::net::DocaRxQParams rxq{};
    const auto rxq_result = framework::net::doca_create_rxq(
            &rxq, gpu_dev, setup.ddev.get(), TEST_MAX_PKT_NUM, TEST_MAX_PKT_SIZE, sem_items);
    ASSERT_EQ(rxq_result, DOCA_SUCCESS);

    EXPECT_EQ(rxq.max_pkt_num, TEST_MAX_PKT_NUM);
    EXPECT_EQ(rxq.max_pkt_size, TEST_MAX_PKT_SIZE);

    static constexpr uint16_t TEST_ETHER_TYPE = 0x88B5;
    const auto test_mac = framework::net::MacAddress::from_string("00:11:22:33:44:55");
    ASSERT_TRUE(test_mac.has_value());
    const auto create_result = framework::net::doca_create_flow_rule(
            &rxq,
            setup.pcie_address,
            test_mac.value(), // NOLINT(bugprone-unchecked-optional-access)
            TEST_ETHER_TYPE,
            vlan_opt);
    EXPECT_EQ(create_result, DOCA_SUCCESS);

    const auto destroy_result = framework::net::doca_destroy_flow_rule(&rxq);
    EXPECT_EQ(destroy_result, DOCA_SUCCESS);

    const auto rxq_destroy_result = framework::net::doca_destroy_rxq(&rxq);
    EXPECT_EQ(rxq_destroy_result, DOCA_SUCCESS);

    EXPECT_EQ(framework::net::doca_close_cuda_device(gpu_dev), DOCA_SUCCESS);
}

TEST(DocaUtils, LogsDocaVersions) { framework::net::doca_log_versions(); }

TEST(DocaUtils, InitializesLogging) {
    doca_log_backend *sdk_log = nullptr;
    const auto result = framework::net::doca_init_logging(&sdk_log);
    EXPECT_EQ(result, DOCA_SUCCESS);
    EXPECT_NE(sdk_log, nullptr);
}

TEST(DocaUtils, InitializesDocaDevice) {
    doca_dev *ddev = nullptr;
    const auto nics = framework::net::discover_mellanox_nics();
    ASSERT_FALSE(nics.empty());

    // Try to initialize first NIC
    const auto result = framework::net::doca_open_and_probe_device(nics.front(), &ddev);
    EXPECT_EQ(result, DOCA_SUCCESS);
    EXPECT_NE(ddev, nullptr);

    EXPECT_EQ(framework::net::doca_close_device(ddev), DOCA_SUCCESS);
}

TEST(DocaUtils, SetsCudaDevice) {
    int device_count{};
    const auto cres = cudaGetDeviceCount(&device_count);
    ASSERT_EQ(cres, cudaSuccess) << "CUDA runtime error: " << cudaGetErrorString(cres);
    ASSERT_GT(device_count, 0) << "No CUDA devices available. At least one CUDA device required.";

    const auto pci_bus_id = framework::net::doca_device_id_to_pci_bus_id(0);
    EXPECT_FALSE(pci_bus_id.empty());

    doca_gpu *gpu_dev = nullptr;
    const auto dres = framework::net::doca_open_cuda_device(pci_bus_id, &gpu_dev);
    EXPECT_EQ(dres, DOCA_SUCCESS);
    EXPECT_NE(gpu_dev, nullptr);

    EXPECT_EQ(framework::net::doca_close_cuda_device(gpu_dev), DOCA_SUCCESS);
}

TEST(DocaUtils, GetsDpdkPortId) {
    const auto nics = framework::net::discover_mellanox_nics();
    ASSERT_FALSE(nics.empty());

    doca_dev *ddev = nullptr;
    const auto init_result = framework::net::doca_open_and_probe_device(nics.front(), &ddev);
    ASSERT_EQ(init_result, DOCA_SUCCESS);

    uint16_t port_id{};
    const auto result = framework::net::doca_get_dpdk_port_id(ddev, &port_id);
    EXPECT_EQ(result, DOCA_SUCCESS);

    EXPECT_EQ(framework::net::doca_close_device(ddev), DOCA_SUCCESS);
}

TEST(DocaUtils, GetsMacAddrFromPci) {
    const auto nics = framework::net::discover_mellanox_nics();
    ASSERT_FALSE(nics.empty());

    framework::net::MacAddress mac_addr{};
    const auto result = framework::net::doca_get_mac_addr_from_pci(nics.front(), mac_addr);
    EXPECT_EQ(result, DOCA_SUCCESS);
}

TEST(DocaUtils, CreatesAndDestroysRxq) {
    const auto setup_opt = framework::net::configure_test_dpdk_port();
    ASSERT_TRUE(setup_opt.has_value()) << setup_opt.error();
    const auto &setup = setup_opt.value(); // NOLINT(bugprone-unchecked-optional-access)

    const auto pci_bus_id = framework::net::doca_device_id_to_pci_bus_id(0);
    doca_gpu *gpu_dev = nullptr;
    const auto gpu_result = framework::net::doca_open_cuda_device(pci_bus_id, &gpu_dev);
    ASSERT_EQ(gpu_result, DOCA_SUCCESS);

    static constexpr uint32_t TEST_MAX_PKT_NUM = 16384U;
    static constexpr uint32_t TEST_MAX_PKT_SIZE = 4096U;

    static constexpr uint32_t TEST_SEM_NUM_ITEMS = 4096U;
    static constexpr uint32_t TEST_SEM_ITEM_SIZE = 4U;

    const framework::net::DocaSemItems sem_items{
            .num_items = TEST_SEM_NUM_ITEMS, .item_size = TEST_SEM_ITEM_SIZE};

    framework::net::DocaRxQParams rxq{};
    const auto create_result = framework::net::doca_create_rxq(
            &rxq, gpu_dev, setup.ddev.get(), TEST_MAX_PKT_NUM, TEST_MAX_PKT_SIZE, sem_items);
    EXPECT_EQ(create_result, DOCA_SUCCESS);

    EXPECT_EQ(rxq.max_pkt_num, TEST_MAX_PKT_NUM);
    EXPECT_EQ(rxq.max_pkt_size, TEST_MAX_PKT_SIZE);
    ASSERT_TRUE(rxq.has_sem_items);
    EXPECT_EQ(rxq.sem_items.num_items, TEST_SEM_NUM_ITEMS);
    EXPECT_EQ(rxq.sem_items.item_size, TEST_SEM_ITEM_SIZE);

    const auto destroy_result = framework::net::doca_destroy_rxq(&rxq);
    EXPECT_EQ(destroy_result, DOCA_SUCCESS);

    EXPECT_EQ(framework::net::doca_close_cuda_device(gpu_dev), DOCA_SUCCESS);
}

TEST(DocaUtils, CreatesAndDestroysFlowRule) {
    static constexpr auto TEST_VLAN_ID = std::nullopt;
    test_create_and_destroy_flow_rule(TEST_VLAN_ID);
}

TEST(DocaUtils, CreatesAndDestroysFlowRuleWithVlan) {
    static constexpr uint16_t TEST_VLAN_ID = 200;
    test_create_and_destroy_flow_rule(TEST_VLAN_ID);
}

TEST(DocaUtils, CreatesAndDestroysTxq) {
    const auto setup_opt = framework::net::configure_test_dpdk_port();
    ASSERT_TRUE(setup_opt.has_value()) << setup_opt.error();
    const auto &setup = setup_opt.value(); // NOLINT(bugprone-unchecked-optional-access)

    const auto pci_bus_id = framework::net::doca_device_id_to_pci_bus_id(0);
    doca_gpu *gpu_dev = nullptr;
    const auto gpu_result = framework::net::doca_open_cuda_device(pci_bus_id, &gpu_dev);
    ASSERT_EQ(gpu_result, DOCA_SUCCESS);

    static constexpr uint32_t TEST_MAX_SQ_DESCR_NUM = 8192U;
    static constexpr uint16_t TEST_ETHER_TYPE = 0x88B5;

    const auto test_dest_mac = framework::net::MacAddress::from_string("00:11:22:33:44:55");
    ASSERT_TRUE(test_dest_mac.has_value());

    framework::net::DocaTxQParams txq{};
    const auto create_result = framework::net::doca_create_txq(
            &txq,
            gpu_dev,
            setup.ddev.get(),
            1024,
            64,
            TEST_MAX_SQ_DESCR_NUM,
            setup.pcie_address,
            test_dest_mac.value(), // NOLINT(bugprone-unchecked-optional-access)
            TEST_ETHER_TYPE);
    EXPECT_EQ(create_result, DOCA_SUCCESS);

    const auto destroy_result = framework::net::doca_destroy_txq(&txq);
    EXPECT_EQ(destroy_result, DOCA_SUCCESS);

    EXPECT_EQ(framework::net::doca_close_cuda_device(gpu_dev), DOCA_SUCCESS);
}

TEST(DocaUtils, AlignsToGpuPage) {
    // Test various sizes to ensure proper alignment
    // GPU page size is 65536 bytes (64KB)
    EXPECT_EQ(framework::net::doca_align_to_gpu_page(0), 65536);
    EXPECT_EQ(framework::net::doca_align_to_gpu_page(1), 65536);
    EXPECT_EQ(framework::net::doca_align_to_gpu_page(65536), 65536);
    EXPECT_EQ(framework::net::doca_align_to_gpu_page(65537), 131072);
}

TEST(DocaUtils, DetectsCx6Device) {
    const auto nics = framework::net::discover_mellanox_nics();
    ASSERT_FALSE(nics.empty());

    doca_dev *ddev = nullptr;
    const auto init_result = framework::net::doca_open_and_probe_device(nics.front(), &ddev);
    ASSERT_EQ(init_result, DOCA_SUCCESS);
    ASSERT_NE(ddev, nullptr);

    // Test device detection
    const auto cx6_result = framework::net::is_device_cx6(ddev);
    // Device detection can succeed or return error depending on hardware capabilities
    if (cx6_result.has_value()) {
        RT_LOGC_DEBUG(
                framework::net::Net::NetDoca,
                "CX-6 detection: {}",
                cx6_result.value() ? "IS CX-6" : "NOT CX-6");
    } else {
        RT_LOGC_DEBUG(framework::net::Net::NetDoca, "CX-6 detection error: {}", cx6_result.error());
    }

    EXPECT_EQ(framework::net::doca_close_device(ddev), DOCA_SUCCESS);
}

TEST(DocaUtils, CreatesRxqWithoutSemaphore) {
    const auto setup_opt = framework::net::configure_test_dpdk_port();
    ASSERT_TRUE(setup_opt.has_value()) << setup_opt.error();
    const auto &setup = setup_opt.value(); // NOLINT(bugprone-unchecked-optional-access)

    const auto pci_bus_id = framework::net::doca_device_id_to_pci_bus_id(0);
    doca_gpu *gpu_dev = nullptr;
    const auto gpu_result = framework::net::doca_open_cuda_device(pci_bus_id, &gpu_dev);
    ASSERT_EQ(gpu_result, DOCA_SUCCESS);

    static constexpr uint32_t TEST_MAX_PKT_NUM = 16384U;
    static constexpr uint32_t TEST_MAX_PKT_SIZE = 4096U;

    const std::optional<framework::net::DocaSemItems> sem_items;
    // Leave sem_items unset - no semaphore should be created

    framework::net::DocaRxQParams rxq{};
    const auto create_result = framework::net::doca_create_rxq(
            &rxq, gpu_dev, setup.ddev.get(), TEST_MAX_PKT_NUM, TEST_MAX_PKT_SIZE, sem_items);
    ASSERT_EQ(create_result, DOCA_SUCCESS);

    EXPECT_EQ(rxq.max_pkt_num, TEST_MAX_PKT_NUM);
    EXPECT_EQ(rxq.max_pkt_size, TEST_MAX_PKT_SIZE);
    EXPECT_FALSE(rxq.has_sem_items);
    EXPECT_EQ(rxq.sem_cpu, nullptr);
    EXPECT_EQ(rxq.sem_gpu, nullptr);

    const auto destroy_result = framework::net::doca_destroy_rxq(&rxq);
    EXPECT_EQ(destroy_result, DOCA_SUCCESS);

    EXPECT_EQ(framework::net::doca_close_cuda_device(gpu_dev), DOCA_SUCCESS);
}

// Test RDMA capability detection for NIC
TEST(DocaUtils, IsRdmaSupported) {
    const auto setup_opt = framework::net::configure_test_dpdk_port();
    ASSERT_TRUE(setup_opt.has_value()) << setup_opt.error();
    const auto &setup = setup_opt.value(); // NOLINT(bugprone-unchecked-optional-access)

    const auto rdma_result = framework::net::doca_is_rdma_supported(setup.ddev.get());

    // Should succeed in checking capability (even if not supported)
    ASSERT_TRUE(rdma_result.has_value()) << rdma_result.error();

    // Log the result may or may not be supported on the actual NIC hardware
    RT_LOGC_DEBUG(
            framework::net::Net::NetDoca,
            "NIC RDMA capability: {}",
            rdma_result.value() ? "SUPPORTED" : "NOT SUPPORTED");
}

// Test RDMA capability detection with null device (error case)
TEST(DocaUtils, IsRdmaSupportedNullDevice) {
    const auto rdma_result = framework::net::doca_is_rdma_supported(nullptr);

    EXPECT_FALSE(rdma_result.has_value());
    EXPECT_EQ(rdma_result.error(), "Device is null");
}

} // namespace
