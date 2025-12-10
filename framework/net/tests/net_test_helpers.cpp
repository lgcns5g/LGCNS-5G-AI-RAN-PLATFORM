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
 * @file net_test_helpers.cpp
 * @brief Implementation of common test helper functions for network utilities
 * tests
 */

// Cross-compiler sanitizer detection
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_LEAK__)
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LEAK_SANITIZER_ENABLED 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(leak_sanitizer)
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LEAK_SANITIZER_ENABLED 1
#endif // __has_feature(address_sanitizer) || __has_feature(leak_sanitizer)
#endif // defined(__has_feature)

#if LEAK_SANITIZER_ENABLED
#include <cerrno>
#include <cstring>

#include <sys/prctl.h>
#endif // LEAK_SANITIZER_ENABLED

#include <format>
#include <system_error>
#include <utility>
#include <vector>

#include <doca_dev.h>
#include <doca_error.h>
#include <driver_types.h>
#include <quill/LogMacros.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include "log/rt_log_macros.hpp"
#include "net/details/doca_utils.hpp"
#include "net/details/dpdk_utils.hpp"
#include "net/dpdk_types.hpp"
#include "net/net_log.hpp"
#include "net_test_helpers.hpp"

namespace framework::net {

void DocaDeviceDeleter::operator()(doca_dev *device) const noexcept {
    if (device != nullptr) {
        if (const auto close_result = doca_close_device(device); close_result != DOCA_SUCCESS) {
            RT_LOGC_ERROR(
                    Net::NetDoca,
                    "Failed to close DOCA device: {}",
                    doca_error_get_descr(close_result));
        }
    }
}

tl::expected<TestDpdkSetup, std::string> configure_test_dpdk_port() {
    const auto nics = discover_mellanox_nics();
    if (nics.empty()) {
        const std::string error_msg = "No Mellanox NICs found for testing";
        ADD_FAILURE() << error_msg;
        return tl::unexpected(error_msg);
    }

    const std::string &pcie_address = nics.front();
    doca_dev *raw_ddev = nullptr;
    const auto init_result = doca_open_and_probe_device(pcie_address, &raw_ddev);
    if (init_result != DOCA_SUCCESS) {
        const std::string error_msg = std::format(
                "Failed to open and probe DOCA device at {}: {}",
                pcie_address,
                doca_error_get_descr(init_result));
        ADD_FAILURE() << error_msg;
        return tl::unexpected(error_msg);
    }

    // Wrap the raw pointer in unique_ptr for automatic cleanup
    DocaDevicePtr ddev{raw_ddev};

    std::uint16_t port_id{};
    const auto port_id_result = doca_get_dpdk_port_id(ddev.get(), &port_id);
    if (port_id_result != DOCA_SUCCESS) {
        const std::string error_msg = std::format(
                "Failed to get DPDK port ID for device {}: {}",
                pcie_address,
                doca_error_get_descr(port_id_result));
        ADD_FAILURE() << error_msg;
        return tl::unexpected(error_msg);
    }

    // Configure and set up at least one TX queue (needed for port start)
    static constexpr auto TXQ_COUNT = 1;
    static constexpr auto RXQ_COUNT = 0;
    const auto result = dpdk_try_configure_port(port_id, RXQ_COUNT, TXQ_COUNT, false);
    if (result == DpdkPortState::ConfigureError) {
        const std::string error_msg =
                std::format("Failed to configure DPDK port {} (device {})", port_id, pcie_address);
        ADD_FAILURE() << error_msg;
        return tl::unexpected(error_msg);
    }

    static constexpr auto TXQ_ID = 0;
    static constexpr auto TXQ_SIZE = 128;
    if (const auto txq_result = dpdk_setup_tx_queue(port_id, TXQ_ID, TXQ_SIZE); txq_result) {
        const std::string error_msg = std::format(
                "Failed to setup DPDK TX queue {} on port {} (device {}): {}",
                TXQ_ID,
                port_id,
                pcie_address,
                get_error_name(txq_result));
        ADD_FAILURE() << error_msg;
        return tl::unexpected(error_msg);
    }

    // Success: return the setup with moved unique_ptr
    return TestDpdkSetup{port_id, std::move(ddev), pcie_address};
}

void enable_sanitizer_compatibility() {
#if LEAK_SANITIZER_ENABLED
    // Enable process dumpability to allow both leak sanitizer and real-time
    // scheduling When CAP_SYS_NICE is set, the process becomes non-dumpable by
    // default for security This prevents ptrace attachment needed by sanitizers
    // and debugging tools
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg,hicpp-vararg)
    if (prctl(PR_SET_DUMPABLE, 1) != 0) {
        const std::error_code ec(errno, std::generic_category());
        RT_LOG_WARN("Failed to set process as dumpable: {}", ec.message());
    }
#endif // LEAK_SANITIZER_ENABLED
}

bool has_cuda_device() {
    int device_count{};
    if (const auto cres = cudaGetDeviceCount(&device_count); cres != cudaSuccess) {
        ADD_FAILURE() << "Failed to get CUDA device count: " << cudaGetErrorString(cres);
        return false;
    }
    if (device_count == 0) {
        ADD_FAILURE() << "No CUDA devices available. At least one CUDA device is required.";
        return false;
    }
    return true;
}

bool has_mellanox_nic() {
    const auto nics = discover_mellanox_nics();
    if (nics.empty()) {
        ADD_FAILURE() << "No Mellanox NICs available. At least one Mellanox NIC is required.";
        return false;
    }
    return true;
}

} // namespace framework::net
