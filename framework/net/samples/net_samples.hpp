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
 * @file net_samples.hpp
 * @brief Common utilities for network sample applications
 */

#ifndef FRAMEWORK_NET_SAMPLES_HPP
#define FRAMEWORK_NET_SAMPLES_HPP

#include <cstddef>
#include <optional>
#include <string>       // for std::string
#include <system_error> // for std::error_code

#include <driver_types.h>
#include <tl/expected.hpp>
#include <wise_enum_detail.h>
#include <wise_enum_generated.h>

#include <wise_enum.h> // for WISE_ENUM_ADAPT

#include "log/rt_log_macros.hpp" // for RT_LOGGABLE_DEFERRED_FORMAT
#include "net/dpdk_types.hpp"    // for MacAddress
#include "net/env.hpp"           // for EnvConfig
#include "net/gpu.hpp"           // for GpuDeviceId

namespace framework::net {

/**
 * Network sample application type
 */
enum class NetSample {
    Sender,  //!< Network sender application
    Receiver //!< Network receiver application
};

} // namespace framework::net

// WISE_ENUM_ADAPT must be called at global namespace scope for ADL to work
WISE_ENUM_ADAPT(framework::net::NetSample, Sender, Receiver)

namespace framework::net {

/**
 * Command line arguments structure
 */
struct Arguments final {
    GpuDeviceId cuda_device_id{0}; //!< CUDA device ID
    std::string nic_pcie_addr;     //!< NIC PCIe address
    MacAddress mac_addr;           //!< MAC address
    int timeout_seconds{};         //!< Timeout in seconds (0 = no timeout)
    bool cpu_only{false};          //!< Use CPU-only DPDK mode (no GPU/DOCA)
};

} // namespace framework::net

/// @cond HIDE_FROM_DOXYGEN
// Must be in global namespace for quill to find it
// cppcheck-suppress functionStatic
RT_LOGGABLE_DEFERRED_FORMAT(
        framework::net::Arguments,
        "cuda_device_id: {}, nic_pcie_addr: {}, mac_addr: "
        "{}, timeout_seconds: {}, cpu_only: {}",
        obj.cuda_device_id.value(),
        obj.nic_pcie_addr,
        obj.mac_addr.to_string(),
        obj.timeout_seconds,
        obj.cpu_only)
/// @endcond

namespace framework::net {

/**
 * Setup logging for network sample applications
 */
void setup_logging();

/**
 * Parse command line arguments with system validation
 *
 * Performs the following operations:
 * 1. Parses command line arguments
 * 2. Validates CUDA device availability
 * 3. Discovers and validates Mellanox NICs
 * 4. Overrides NIC PCIe address if not specified
 *
 * @param[in] sample_type Type of network sample (SENDER or RECEIVER)
 * @param[in] argc Number of command line arguments
 * @param[in] argv Array of command line argument strings
 * @return Arguments on success, empty string if --help or --version shown, error message on failure
 */
[[nodiscard]] tl::expected<Arguments, std::string>
parse_arguments(NetSample sample_type, int argc, const char **argv);

/**
 * Create a network environment configuration for sending or receiving
 *
 * @param[in] sample_type Type of network sample (SENDER or RECEIVER)
 * @param[in] args Parsed command line arguments
 * @return Configured EnvConfig for network operations
 */
[[nodiscard]] EnvConfig create_net_env_config(NetSample sample_type, const Arguments &args);

/**
 * Send a message using DPDK-only mode (CPU-only, no GPU/DOCA)
 *
 * Sends "Hello DPDK" message using the NIC's MAC as source and
 * the configured destination MAC from arguments.
 *
 * @param[in] env Network environment (must be configured for DPDK-only mode)
 * @param[in] dest_mac Destination MAC address
 * @return Error code (success if no error)
 */
[[nodiscard]] std::error_code send_dpdk_message(const Env &env, const MacAddress &dest_mac);

/**
 * Copy data from host to device with error logging
 *
 * If stream is provided, uses cudaMemcpyAsync (caller responsible for synchronization).
 * If stream is not provided (nullopt), uses synchronous cudaMemcpy (data ready on return).
 *
 * @param[in] dst Destination device pointer
 * @param[in] src Source host pointer
 * @param[in] size Number of bytes to copy
 * @param[in] stream Optional CUDA stream for async copy (if nullopt, uses synchronous copy)
 * @return true on success, false on failure (error is logged)
 */
[[nodiscard]] bool cuda_memcpy_host_to_device(
        void *dst,
        const void *src,
        std::size_t size,
        std::optional<cudaStream_t> stream = std::nullopt) noexcept;

/**
 * Copy data from device to host with error logging
 *
 * If stream is provided, uses cudaMemcpyAsync (caller responsible for synchronization).
 * If stream is not provided (nullopt), uses synchronous cudaMemcpy (data ready on return).
 *
 * @param[in] dst Destination host pointer
 * @param[in] src Source device pointer
 * @param[in] size Number of bytes to copy
 * @param[in] stream Optional CUDA stream for async copy (if nullopt, uses synchronous copy)
 * @return true on success, false on failure (error is logged)
 */
[[nodiscard]] bool cuda_memcpy_device_to_host(
        void *dst,
        const void *src,
        std::size_t size,
        std::optional<cudaStream_t> stream = std::nullopt) noexcept;

} // namespace framework::net

#endif // FRAMEWORK_NET_SAMPLES_HPP
