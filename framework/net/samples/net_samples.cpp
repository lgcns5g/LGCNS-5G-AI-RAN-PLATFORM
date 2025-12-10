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
 * @file net_samples.cpp
 * @brief Common utilities for network sample applications
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

#include <algorithm>
#include <cstdint>
#include <format> // for format
#include <optional>
#include <span> // for std::span
#include <string_view>
#include <system_error> // for std::error_code
#include <vector>

#include <driver_types.h>
#include <quill/LogMacros.h>
#include <tl/expected.hpp>

#include <CLI/CLI.hpp> // for CLI::App, CLI::Option

#include <cuda_runtime.h> // for cudaGetDeviceCount

#include "internal_use_only/config.hpp" // for project_name, project_version
#include "log/components.hpp"           // for register_component
#include "log/rt_log.hpp"               // for Logger, LoggerConfig
#include "log/rt_log_macros.hpp"        // for RT_LOGC_*
#include "net/doca_rxq.hpp"
#include "net/doca_txq.hpp"
#include "net/doca_types.hpp"
#include "net/dpdk_txq.hpp"
#include "net/dpdk_types.hpp" // for discover_mellanox_nics
#include "net/mempool.hpp"
#include "net/net_log.hpp" // for Net component
#include "net/nic.hpp"
#include "net_samples.hpp"

namespace {

namespace fn = ::framework::net;

/// Ethernet type used for sample applications
constexpr std::uint16_t SAMPLE_ETHER_TYPE = 0x88b5;
constexpr std::uint16_t SAMPLE_VLAN_ID = 100;

/**
 * Check CUDA device availability
 *
 * @return true if CUDA devices are available, false otherwise
 */
[[nodiscard]] std::optional<int> get_cuda_device_count() noexcept {
    int device_count{};
    if (const auto cres = cudaGetDeviceCount(&device_count); cres != cudaSuccess) {
        RT_LOGC_ERROR(
                fn::Net::NetGpu, "Failed to get CUDA device count: {}", cudaGetErrorString(cres));
        return std::nullopt;
    }
    if (device_count == 0) {
        RT_LOGC_ERROR(
                fn::Net::NetGpu,
                "No CUDA devices available. At least one CUDA device is required.");
        return std::nullopt;
    }
    RT_LOGC_DEBUG(fn::Net::NetGpu, "Found {} CUDA device(s)", device_count);
    return device_count;
}

/**
 * Check Mellanox NIC availability and return available NICs
 *
 * @return Vector of available Mellanox NIC PCIe addresses, empty if none found
 */
[[nodiscard]] std::vector<std::string> check_mellanox_nics() {
    auto nics = fn::discover_mellanox_nics();
    if (nics.empty()) {
        RT_LOGC_ERROR(
                fn::Net::NetDpdk,
                "No Mellanox NICs available. At least one Mellanox NIC is required.");
        return {};
    }
    RT_LOGC_DEBUG(fn::Net::NetDpdk, "Found {} Mellanox NIC(s)", nics.size());
    return nics;
}

} // namespace

namespace framework::net {

void setup_logging() {
    framework::log::Logger::set_level(framework::log::LogLevel::Debug);
    framework::log::register_component<framework::net::Net>(framework::log::LogLevel::Debug);

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

tl::expected<Arguments, std::string>
parse_arguments(const NetSample sample_type, const int argc, const char **argv) {
    const std::string app_name =
            sample_type == NetSample::Sender ? "Network Sender Demo" : "Network Receiver Demo";
    CLI::App app{std::format(
            "{} - {} version {}",
            app_name,
            framework::cmake::project_name,
            framework::cmake::project_version)};

    Arguments args{};
    args.timeout_seconds = 0;

    uint32_t temp_cuda_device_id{0};
    app.add_option("-d,--device", temp_cuda_device_id, "CUDA device ID (default: 0)")
            ->check(CLI::NonNegativeNumber);

    app.add_option(
            "-n,--nic", args.nic_pcie_addr, "NIC PCIe address (default: first Mellanox NIC)");

    std::string mac_addr = "ff:ff:ff:ff:ff:ff"; // Broadcast MAC
    if (sample_type == NetSample::Sender) {
        app.add_option(
                "-m,--mac",
                mac_addr,
                "Destination MAC address (default: broadcast ff:ff:ff:ff:ff:ff)");
    } else {
        app.add_option(
                "-m,--mac",
                mac_addr,
                "Source MAC address to receive packets from (default: "
                "broadcast ff:ff:ff:ff:ff:ff)");

        if (sample_type == NetSample::Receiver) {
            app.add_option(
                       "-t,--timeout",
                       args.timeout_seconds,
                       "Timeout in seconds (default: 0, no timeout)")
                    ->check(CLI::NonNegativeNumber);
        }
    }

    if (sample_type == NetSample::Sender) {
        app.add_flag(
                "-c,--cpu-only",
                args.cpu_only,
                "Use CPU-only DPDK mode (no GPU/DOCA acceleration)");
    }

    app.set_version_flag(
            "--version",
            std::string{framework::cmake::project_version},
            "Show version information");

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
        const int exit_code = app.exit(e);
        if (exit_code == 0) {
            // Success codes (--help or --version) - return empty error string
            return tl::unexpected("");
        }
        // Actual error
        return tl::unexpected(std::format("Argument parsing failed: {}", e.what()));
    }

    args.cuda_device_id = GpuDeviceId{temp_cuda_device_id};

    // Check system requirements (skip CUDA validation in CPU-only mode)
    if (!args.cpu_only) {
        const auto dev_count = get_cuda_device_count();
        if (!dev_count.has_value()) {
            return tl::unexpected("Failed to detect CUDA devices");
        }
        if (args.cuda_device_id.value() >= dev_count.value()) {
            return tl::unexpected(std::format(
                    "Invalid CUDA device ID: {}.  Detected {} CUDA device(s).",
                    args.cuda_device_id.value(),
                    dev_count.value()));
        }
    }

    const auto available_nics = check_mellanox_nics();
    if (available_nics.empty()) {
        return tl::unexpected("No Mellanox NICs available. At least one Mellanox NIC is required.");
    }

    // Use specified NIC or first available Mellanox NIC
    if (args.nic_pcie_addr.empty()) {
        args.nic_pcie_addr = available_nics.front();
        RT_LOGC_INFO(
                framework::net::Net::NetDpdk,
                "Using first available Mellanox NIC: {}",
                args.nic_pcie_addr);
    }

    if (std::find(available_nics.cbegin(), available_nics.cend(), args.nic_pcie_addr) ==
        available_nics.cend()) {
        RT_LOGC_ERROR(
                framework::net::Net::NetDpdk,
                "Invalid NIC PCIe address: {} not in device list: {}.",
                args.nic_pcie_addr,
                available_nics);
        return tl::unexpected(std::format("Invalid NIC PCIe address: {}", args.nic_pcie_addr));
    }

    const auto parsed_mac_addr = framework::net::MacAddress::from_string(mac_addr);
    if (!parsed_mac_addr.has_value()) {
        return tl::unexpected(std::format("Invalid MAC address: {}", mac_addr));
    }
    args.mac_addr = parsed_mac_addr.value();

    // Print the parsed arguments using RT_LOGC
    RT_LOGC_INFO(framework::net::Net::NetGeneral, "Parsed arguments: {}", args);

    return args;
}

EnvConfig create_net_env_config(const NetSample sample_type, const Arguments &args) {
    // Queue configuration constants
    static constexpr std::uint32_t PKT_SIZE = 1024;     // 1KB packets (same for TX and RX)
    static constexpr std::uint32_t TX_PKT_NUM = 64;     // 64 packets
    static constexpr std::uint32_t TX_DESCR_NUM = 8192; // 8K descriptors
    static constexpr std::uint32_t RX_PKT_NUM = 16384;  // 16K packets
    static constexpr std::uint32_t DPDK_CORE_ID = 0;    // DPDK core ID

    // DPDK-only configuration constants
    static constexpr std::uint16_t DPDK_TXQ_SIZE = 128;
    static constexpr std::uint32_t MEMPOOL_NUM_MBUFS = 1024;
    static constexpr std::uint32_t MEMPOOL_MTU_SIZE = 1500;

    EnvConfig config{};

    // Basic configuration
    config.gpu_device_id = args.cuda_device_id;
    config.nic_config.nic_pcie_addr = args.nic_pcie_addr;

    // DPDK configuration
    const std::string app_name = sample_type == NetSample::Sender ? "net_sender" : "net_receiver";

    // Add cpu suffix to prefix for parallel test execution
    const std::string file_prefix = args.cpu_only ? std::format("{}_cpu_prefix", app_name)
                                                  : std::format("{}_prefix", app_name);

    config.dpdk_config.app_name = app_name;
    config.dpdk_config.file_prefix = file_prefix;
    config.dpdk_config.dpdk_core_id = DPDK_CORE_ID;

    if (args.cpu_only) {
        // CPU-only DPDK mode - no DOCA queues, only DPDK TX queue and mempool
        DpdkTxQConfig dpdk_tx_config{};
        dpdk_tx_config.txq_size = DPDK_TXQ_SIZE;
        config.nic_config.dpdk_txq_configs.push_back(dpdk_tx_config);

        MempoolConfig mempool_config{};
        mempool_config.name = app_name + "_mempool";
        mempool_config.num_mbufs = MEMPOOL_NUM_MBUFS;
        mempool_config.mtu_size = MEMPOOL_MTU_SIZE;
        mempool_config.host_pinned = HostPinned::No;
        config.nic_config.mempool_configs.push_back(mempool_config);
    } else {
        // Standard DOCA mode with GPU acceleration
        // Add TX queue configuration
        DocaTxQConfig tx_config{};
        tx_config.nic_pcie_addr = args.nic_pcie_addr;
        tx_config.dest_mac_addr = args.mac_addr;
        tx_config.pkt_size = PKT_SIZE;
        tx_config.pkt_num = TX_PKT_NUM;
        tx_config.max_sq_descr_num = TX_DESCR_NUM;
        tx_config.ether_type = SAMPLE_ETHER_TYPE;
        tx_config.vlan_tci = SAMPLE_VLAN_ID;
        config.nic_config.doca_txq_configs.push_back(tx_config);

        // Add RX queue configuration
        DocaRxQConfig rx_config{};
        rx_config.nic_pcie_addr = args.nic_pcie_addr;
        rx_config.sender_mac_addr = args.mac_addr; // Source MAC to filter for RX
        rx_config.max_pkt_num = RX_PKT_NUM;
        rx_config.max_pkt_size = PKT_SIZE;
        rx_config.ether_type = SAMPLE_ETHER_TYPE;
        rx_config.vlan_tci = SAMPLE_VLAN_ID;
        config.nic_config.doca_rxq_configs.push_back(rx_config);
    }

    return config;
}

std::error_code send_dpdk_message(const Env &env, const MacAddress &dest_mac) {
    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic,cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

    // example-begin net-samples-dpdk-send-1
    static constexpr std::string_view MESSAGE = "Hello DPDK";
    static constexpr std::uint32_t MESSAGE_LEN = MESSAGE.size();

    RT_LOGC_DEBUG(Net::NetDpdk, "Sending message via DPDK (CPU-only mode): '{}'", MESSAGE);

    const auto &src_mac = env.nic().mac_address();
    const EthernetHeader eth_header{src_mac, dest_mac, SAMPLE_ETHER_TYPE, SAMPLE_VLAN_ID};

    // Create packet payload with length prefix
    // Format: [4-byte length in big-endian][message content]
    std::vector<std::uint8_t> payload(4 + MESSAGE_LEN);

    // Write length in big-endian format
    payload[0] = (MESSAGE_LEN >> 24U) & 0xFFU;
    payload[1] = (MESSAGE_LEN >> 16U) & 0xFFU;
    payload[2] = (MESSAGE_LEN >> 8U) & 0xFFU;
    payload[3] = MESSAGE_LEN & 0xFFU;

    // Write message content
    const std::span<std::uint8_t> payload_content_span{payload.data() + 4, MESSAGE_LEN};
    std::copy(MESSAGE.begin(), MESSAGE.end(), payload_content_span.begin());
    const std::span<const std::uint8_t> payload_span{payload.data(), payload.size()};
    const std::vector<std::span<const std::uint8_t>> messages = {payload_span};

    return env.nic().send(0 /* dpdk_txq_id */, 0 /* mempool_id */, messages, eth_header);
    // example-end net-samples-dpdk-send-1

    // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic,cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
}

bool cuda_memcpy_host_to_device(
        void *dst,
        const void *src,
        const std::size_t size,
        std::optional<cudaStream_t> stream) noexcept {
    const auto result = [dst, src, size, stream]() {
        return stream.has_value() ? cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, *stream)
                                  : cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    }();

    if (result != cudaSuccess) {
        RT_LOGC_ERROR(
                Net::NetGpu, "Failed to copy from host to device: {}", cudaGetErrorString(result));
        return false;
    }
    return true;
}

bool cuda_memcpy_device_to_host(
        void *dst,
        const void *src,
        const std::size_t size,
        std::optional<cudaStream_t> stream) noexcept {
    const auto result = [dst, src, size, stream]() {
        return stream.has_value() ? cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, *stream)
                                  : cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    }();

    if (result != cudaSuccess) {
        RT_LOGC_ERROR(
                Net::NetGpu, "Failed to copy from device to host: {}", cudaGetErrorString(result));
        return false;
    }
    return true;
}

} // namespace framework::net
