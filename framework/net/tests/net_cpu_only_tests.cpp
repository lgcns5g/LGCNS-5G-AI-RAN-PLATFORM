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
 * @file net_cpu_only_tests.cpp
 * @brief Tests for CPU-only DPDK configuration without GPU/DOCA components
 *
 * This test validates that the framework can operate in true CPU-only mode
 * with DPDK-only queues (no DOCA TX/RX queues). The GPU device is not
 * initialized when no DOCA queues are configured.
 */

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "net/doca_rxq.hpp"
#include "net/doca_txq.hpp"
#include "net/dpdk_txq.hpp"
#include "net/dpdk_types.hpp"
#include "net/env.hpp"
#include "net/mempool.hpp"
#include "net/nic.hpp"

namespace {

TEST(CpuOnlyEnvTest, CpuOnlyDpdkConfiguration) {
    static constexpr std::uint16_t DPDK_TXQ_SIZE = 128;
    static constexpr std::uint32_t MEMPOOL_NUM_MBUFS = 1024;
    static constexpr std::uint32_t MEMPOOL_MTU_SIZE = 1500;

    const auto available_nics = framework::net::discover_mellanox_nics();
    ASSERT_FALSE(available_nics.empty()) << "No Mellanox NICs available for testing";

    framework::net::EnvConfig config{};
    // GPU device ID is ignored when no DOCA queues are configured
    config.nic_config.nic_pcie_addr = available_nics.front();

    config.dpdk_config.file_prefix = "net_cpu_only_test_prefix";
    config.dpdk_config.dpdk_core_id = 0;

    // No DOCA TX queues - CPU-only configuration
    // No DOCA RX queues - CPU-only configuration

    // Create one DPDK TX queue
    framework::net::DpdkTxQConfig dpdk_txq_config{};
    dpdk_txq_config.txq_size = DPDK_TXQ_SIZE;
    config.nic_config.dpdk_txq_configs.push_back(dpdk_txq_config);

    // Create one mempool
    framework::net::MempoolConfig mempool_config{};
    mempool_config.name = "cpu_test_mempool";
    mempool_config.num_mbufs = MEMPOOL_NUM_MBUFS;
    mempool_config.mtu_size = MEMPOOL_MTU_SIZE;
    mempool_config.host_pinned = framework::net::HostPinned::No;
    config.nic_config.mempool_configs.push_back(mempool_config);

    // This should succeed with CPU-only DPDK configuration
    const framework::net::Env env{config};

    // Verify that no GPU device was created
    EXPECT_FALSE(env.has_gpu());

    // Verify that accessing GPU throws an exception
    EXPECT_THROW({ [[maybe_unused]] const auto &gpu = env.gpu(); }, std::runtime_error);

    // Verify that NIC and DPDK components are properly initialized
    EXPECT_TRUE(env.is_initialized());
    EXPECT_EQ(env.nic().dpdk_tx_queues().size(), 1);
    EXPECT_EQ(env.nic().mempools().size(), 1);
    EXPECT_TRUE(env.nic().doca_tx_queues().empty());
    EXPECT_TRUE(env.nic().doca_rx_queues().empty());
}

} // namespace
