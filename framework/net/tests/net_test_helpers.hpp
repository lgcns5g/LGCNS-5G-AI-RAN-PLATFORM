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
 * @file net_test_helpers.hpp
 * @brief Common test helper functions for network utilities tests
 */

#ifndef FRAMEWORK_NET_TEST_HELPERS_HPP
#define FRAMEWORK_NET_TEST_HELPERS_HPP

#include <cstdint>
#include <memory>
#include <string>

#include <doca_dev.h>
#include <tl/expected.hpp>

namespace framework::net {

/**
 * Custom deleter for DOCA device
 */
struct DocaDeviceDeleter {
    void operator()(doca_dev *device) const noexcept;
};

/// Type alias for DOCA device unique pointer
using DocaDevicePtr = std::unique_ptr<doca_dev, DocaDeviceDeleter>;

/**
 * Test setup result containing configured port and device pointers
 */
struct TestDpdkSetup {
    std::uint16_t port_id{};  //!< Configured DPDK port ID
    DocaDevicePtr ddev;       //!< DOCA device smart pointer
    std::string pcie_address; //!< PCIe address of the NIC
};

/**
 * Get a configured DPDK port and device for testing
 *
 * This function discovers Mellanox NICs, opens and probes a DOCA device,
 * gets the DPDK port ID, and configures the port for testing.
 *
 * @return Configured port and device on success, error message on failure
 */
[[nodiscard]] tl::expected<TestDpdkSetup, std::string> configure_test_dpdk_port();

/**
 * Enable sanitizer compatibility for processes with elevated capabilities
 *
 * When a process has CAP_SYS_NICE (for real-time scheduling), it becomes
 * non-dumpable by default for security. This prevents LeakSanitizer and other
 * debugging tools from working. This function makes the process dumpable again
 * when sanitizers are enabled.
 */
void enable_sanitizer_compatibility();

/**
 * Check for CUDA device availability
 *
 * @return true if at least one CUDA device is available, false otherwise
 */
[[nodiscard]] bool has_cuda_device();

/**
 * Check for Mellanox NIC availability
 *
 * @return true if at least one Mellanox NIC is available, false otherwise
 */
[[nodiscard]] bool has_mellanox_nic();

} // namespace framework::net

#endif // FRAMEWORK_NET_TEST_HELPERS_HPP
