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
#include <array>
#include <bit>
#include <cerrno>
#include <charconv>
#include <climits>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <driver_types.h>
#include <net/if.h>
#include <quill/LogMacros.h>
#include <rte_build_config.h>
#include <rte_byteorder.h>
#include <rte_common.h>
#include <rte_dev.h>
#include <rte_eal.h>
#include <rte_errno.h>
#include <rte_ethdev.h>
#include <rte_ether.h>
#include <rte_flow.h>
#include <rte_mbuf.h>
#include <rte_mbuf_core.h>
#include <rte_mbuf_dyn.h>
#include <rte_memcpy.h>
#include <rte_memory.h>
#include <rte_mempool.h>
#include <sched.h>
#include <tl/expected.hpp>

#include <gsl-lite/gsl-lite.hpp>
#include <wise_enum.h>

#include <cuda_runtime.h>

#include "log/rt_log_macros.hpp"
#include "net/details/dpdk_utils.hpp"
#include "net/dpdk_types.hpp"
#include "net/net_log.hpp"

namespace {

/// Private data stored in mempool for cleanup during destruction
struct MempoolPrivateData {
    bool is_host_pinned{};     //!< Whether this mempool uses host-pinned memory
    void *host_pinned_mem{};   //!< Host-pinned memory pointer (for cudaFreeHost)
    std::size_t buffer_size{}; //!< Buffer size for external memory operations
    uint16_t port_id{};        //!< Port ID for device info lookup
};

/**
 * Create a mempool with host-pinned memory for GPU operations
 *
 * @param[in] name Mempool name
 * @param[in] port_id DPDK port ID
 * @param[in] num_mbufs Number of mbufs in the pool
 * @param[in] droom_sz Data room size for each mbuf
 * @param[in] socket_id NUMA socket ID
 * @return Created mempool or nullptr on failure
 */
[[nodiscard]] rte_mempool *create_host_pinned_mempool(
        const std::string_view name,
        const uint16_t port_id,
        const uint32_t num_mbufs,
        const uint16_t droom_sz,
        const int socket_id) {

    static constexpr uint32_t NV_GPU_PAGE_SIZE = 65536U;
    static constexpr uint32_t CACHE_SIZE = 0;

    // Calculate buffer size: align total buffer to GPU page size
    const std::size_t buffer_size = RTE_ALIGN(num_mbufs * droom_sz, NV_GPU_PAGE_SIZE);

    // Allocate one contiguous block of host-pinned memory
    void *host_pinned_mem{};
    const cudaError_t cuda_result = cudaMallocHost(&host_pinned_mem, buffer_size);
    if (cuda_result != cudaSuccess) {
        RT_LOGC_ERROR(
                framework::net::Net::NetDpdk,
                "Failed to allocate {} bytes of contiguous host-pinned memory: {}",
                buffer_size,
                cudaGetErrorString(cuda_result));
        return nullptr;
    }

    // Track mempool creation success to control cleanup
    bool mempool_creation_success = false;

    // Setup cleanup guard for host-pinned memory - only free on failure
    const auto cuda_cleanup = gsl_lite::finally([host_pinned_mem, &mempool_creation_success] {
        if (!mempool_creation_success && host_pinned_mem != nullptr) {
            if (const cudaError_t free_result = cudaFreeHost(host_pinned_mem);
                free_result != cudaSuccess) {
                RT_LOGC_ERROR(
                        framework::net::Net::NetDpdk,
                        "Failed to free host-pinned memory during cleanup at {}: {}",
                        host_pinned_mem,
                        cudaGetErrorString(free_result));
                // Continue with cleanup even if this fails
            }
        }
    });

    RT_LOGC_DEBUG(
            framework::net::Net::NetDpdk,
            "Allocated {} bytes of contiguous host-pinned memory at {}",
            buffer_size,
            host_pinned_mem);

    // Register external memory with DPDK
    bool extmem_registered = false;
    if (const int ret = rte_extmem_register(
                host_pinned_mem,
                buffer_size,
                nullptr,
                static_cast<uint32_t>(RTE_BAD_IOVA),
                NV_GPU_PAGE_SIZE);
        ret != 0) {
        RT_LOGC_ERROR(
                framework::net::Net::NetDpdk,
                "Failed to register external memory with DPDK: {}",
                rte_strerror(-ret));
        return nullptr;
    }
    extmem_registered = true;

    // Setup cleanup guard for external memory registration
    const auto extmem_cleanup =
            gsl_lite::finally([host_pinned_mem, buffer_size, &extmem_registered] {
                if (extmem_registered) {
                    rte_extmem_unregister(host_pinned_mem, buffer_size);
                }
            });

    // Get device info for DMA mapping
    rte_eth_dev_info dev_info{};
    if (const int ret = rte_eth_dev_info_get(port_id, &dev_info); ret != 0) {
        RT_LOGC_ERROR(
                framework::net::Net::NetDpdk,
                "Failed to get device info for port {}: {}",
                port_id,
                rte_strerror(-ret));
        return nullptr;
    }

    // Map memory for DMA access by the device
    bool dma_mapped = false;
    if (const int ret =
                rte_dev_dma_map(dev_info.device, host_pinned_mem, RTE_BAD_IOVA, buffer_size);
        ret != 0) {
        RT_LOGC_ERROR(
                framework::net::Net::NetDpdk,
                "Failed to map DMA memory for device: {}",
                rte_strerror(-ret));
        return nullptr;
    }
    dma_mapped = true;

    // Setup cleanup guard for DMA mapping
    const auto dma_cleanup =
            gsl_lite::finally([&dev_info, host_pinned_mem, buffer_size, &dma_mapped] {
                if (dma_mapped) {
                    rte_dev_dma_unmap(dev_info.device, host_pinned_mem, RTE_BAD_IOVA, buffer_size);
                }
            });

    // Create mempool using external buffers (with private data for cleanup)
    const rte_pktmbuf_extmem ext_mem{host_pinned_mem, RTE_BAD_IOVA, buffer_size, droom_sz};
    rte_mempool *mp = rte_pktmbuf_pool_create_extbuf(
            name.data(),
            num_mbufs,
            CACHE_SIZE,
            sizeof(MempoolPrivateData),
            droom_sz,
            socket_id,
            &ext_mem,
            1 /* ext_num */);
    if (mp == nullptr) {
        RT_LOGC_ERROR(
                framework::net::Net::NetDpdk,
                "Failed to create external buffer mempool '{}': {}",
                name,
                rte_strerror(rte_errno));
        return nullptr;
    }

    // Store cleanup information in mempool private data
    auto *private_data = static_cast<MempoolPrivateData *>(rte_mempool_get_priv(mp));
    private_data->is_host_pinned = true;
    private_data->host_pinned_mem = host_pinned_mem;
    private_data->buffer_size = buffer_size;
    private_data->port_id = port_id;

    // Success - disable cleanup guards since mempool now owns the resources
    extmem_registered = false;
    dma_mapped = false;
    mempool_creation_success = true;

    return mp;
}

[[nodiscard]] std::string rx_offloads_to_string(const uint64_t offloads) {
    if (offloads == 0ULL) {
        return std::string{};
    }

    std::stringstream rx_offloads;
    rx_offloads << "RX offloads:";

    for (uint64_t single_offload = 1ULL; single_offload != 0; single_offload <<= 1ULL) {
        if ((offloads & single_offload) != 0ULL) {
            rx_offloads << " " << rte_eth_dev_rx_offload_name(single_offload);
        }
    }

    return rx_offloads.str();
}

[[nodiscard]] std::string tx_offloads_to_string(const uint64_t offloads) {
    if (offloads == 0ULL) {
        return std::string{};
    }

    std::stringstream tx_offloads;
    tx_offloads << "TX offloads:";

    const auto trailing_zeros = static_cast<uint64_t>(std::countr_zero(offloads));
    const uint64_t end = sizeof(offloads) * CHAR_BIT - trailing_zeros;

    uint64_t single_offload = 1ULL << trailing_zeros;
    for (uint64_t bit = trailing_zeros; bit < end; bit++) {
        if ((offloads & single_offload) != 0ULL) {
            tx_offloads << " " << rte_eth_dev_tx_offload_name(single_offload);
        }
        single_offload <<= 1ULL;
    }

    return tx_offloads.str();
}

/**
 * Read and optionally set a virtual memory parameter
 *
 * @param[in] param_name VM parameter name (e.g., "swappiness",
 * "zone_reclaim_mode")
 * @param[in] target_value Target value to set for the parameter
 * @return std::error_code indicating success or specific error condition
 */
[[nodiscard]] std::error_code
tune_vm_parameter(const std::string &param_name, const std::uint32_t target_value) {
    const auto param_path = std::filesystem::path{"/proc/sys/vm"} / param_name;
    const std::string full_param_name = "vm." + param_name;

    try {
        if (!std::filesystem::exists(param_path)) {
            RT_LOGC_ERROR(
                    framework::net::Net::NetDpdk,
                    "Cannot access {} for reading",
                    param_path.string());
            return framework::net::make_error_code(framework::net::DpdkErrc::VmTuneFailed);
        }

        std::ifstream read_file(param_path);
        if (!read_file.is_open()) {
            RT_LOGC_ERROR(
                    framework::net::Net::NetDpdk,
                    "Cannot access {} for reading",
                    param_path.string());
            return framework::net::make_error_code(framework::net::DpdkErrc::VmTuneFailed);
        }

        std::uint32_t current_value{};
        read_file >> current_value;
        read_file.close();

        if (current_value == target_value) {
            RT_LOGC_DEBUG(
                    framework::net::Net::NetDpdk,
                    "{} is already optimized ({})",
                    full_param_name,
                    current_value);
        } else {
            RT_LOGC_INFO(
                    framework::net::Net::NetDpdk,
                    "{} needs tuning: current={}, optimal={}",
                    full_param_name,
                    current_value,
                    target_value);

            std::ofstream write_file(param_path);
            if (write_file.is_open()) {
                write_file << target_value;
                RT_LOGC_INFO(
                        framework::net::Net::NetDpdk,
                        "Successfully set {}={}",
                        full_param_name,
                        target_value);
            } else {
                RT_LOGC_WARN(
                        framework::net::Net::NetDpdk,
                        "Failed to set {}={}. Current value: {}. "
                        "To optimize performance, run: sudo sysctl -w {}={}",
                        full_param_name,
                        target_value,
                        current_value,
                        full_param_name,
                        target_value);
            }
        }

        return framework::net::make_error_code(framework::net::DpdkErrc::Success);
    } catch (const std::filesystem::filesystem_error &e) {
        RT_LOGC_ERROR(
                framework::net::Net::NetDpdk,
                "Filesystem error while accessing {}: {}",
                param_path.string(),
                e.what());
        return framework::net::make_error_code(framework::net::DpdkErrc::VmTuneFailed);
    }
}

[[nodiscard]] std::string get_current_process_name() {
    try {
        return std::filesystem::read_symlink("/proc/self/exe").filename().string();
    } catch (const std::filesystem::filesystem_error &e) {
        RT_LOGC_ERROR(
                framework::net::Net::NetDpdk, "Failed to read current process name: {}", e.what());
        return "unknown_process_name";
    }
}

/**
 * Parse MAC address from string format
 *
 * @param[in] mac_str MAC address string in format "XX:XX:XX:XX:XX:XX"
 * @return MAC address array on success, error message on failure
 */
[[nodiscard]] tl::
        expected<std::array<std::uint8_t, framework::net::MacAddress::ADDRESS_LEN>, std::string>
        parse_mac_address(const std::string_view mac_str) {
    // Constants for MAC address parsing
    static constexpr std::size_t MAC_STR_LENGTH = 17; // xx:xx:xx:xx:xx:xx format
    static constexpr int HEX_BASE = 16;
    static constexpr std::size_t POS_MULTIPLIER = 3;

    std::array<std::uint8_t, framework::net::MacAddress::ADDRESS_LEN> mac{};

    if (mac_str.length() != MAC_STR_LENGTH) {
        return tl::unexpected(std::format(
                "Invalid MAC address length: expected {} characters, got {}",
                MAC_STR_LENGTH,
                mac_str.length()));
    }

    for (std::size_t i = 0; i < framework::net::MacAddress::ADDRESS_LEN; ++i) {
        const std::size_t pos = i * POS_MULTIPLIER;
        if (i > 0 && mac_str[pos - 1] != ':') {
            return tl::unexpected(std::format(
                    "Invalid delimiter at position {}: expected ':', got '{}'",
                    pos - 1,
                    mac_str[pos - 1]));
        }

        const auto hex_str = mac_str.substr(pos, 2);
        auto result = std::from_chars(
                hex_str.data(), hex_str.data() + hex_str.size(), mac.at(i), HEX_BASE);

        if (result.ec != std::errc{}) {
            return tl::unexpected(std::format(
                    "Invalid hexadecimal value '{}' at position {}-{}", hex_str, pos, pos + 1));
        }

        if (result.ptr != hex_str.data() + hex_str.size()) {
            return tl::unexpected(std::format(
                    "Incomplete hexadecimal value '{}' at position {}-{}", hex_str, pos, pos + 1));
        }
    }

    return mac;
}

[[nodiscard]] std::optional<std::function<void()>> set_scheduling_config(
        const std::optional<std::uint32_t> &core_id, const bool enable_rt_priority_for_lcores) {
    cpu_set_t original_affinity{};
    int original_policy{};
    sched_param original_param{};
    bool saved_state = false;

    // Save original state
    if (sched_getaffinity(0, sizeof(original_affinity), &original_affinity) == 0) {
        original_policy = sched_getscheduler(0);
        if (original_policy != -1 && sched_getparam(0, &original_param) == 0) {
            saved_state = true;
        }
    }

    // Set new affinity if core_id is specified
    if (core_id.has_value()) {
        cpu_set_t new_affinity{};
        CPU_ZERO(&new_affinity);
        CPU_SET(core_id.value(), &new_affinity);
        if (sched_setaffinity(0, sizeof(new_affinity), &new_affinity) != 0) {
            const std::error_code ec(errno, std::generic_category());
            RT_LOGC_ERROR(
                    framework::net::Net::NetDpdk,
                    "Failed to set CPU affinity to core {}: {}",
                    core_id.value(),
                    ec.message());
            return std::nullopt;
        }
    }

    // Set real-time priority if enabled
    if (enable_rt_priority_for_lcores) {
        static constexpr int RT_PRIORITY = 95;
        sched_param rt_param{};
        rt_param.sched_priority = RT_PRIORITY;
        if (sched_setscheduler(0, SCHED_FIFO, &rt_param) != 0) {
            const std::error_code ec(errno, std::generic_category());
            RT_LOGC_ERROR(
                    framework::net::Net::NetDpdk,
                    "Failed to set real-time priority {}: {}",
                    RT_PRIORITY,
                    ec.message());
            return std::nullopt;
        }
    }

    // Return cleanup function
    auto cleanup = [=]() {
        if (saved_state && (core_id.has_value() || enable_rt_priority_for_lcores)) {
            sched_setscheduler(0, original_policy, &original_param);
            sched_setaffinity(0, sizeof(original_affinity), &original_affinity);
        }
    };

    return cleanup;
}

} // namespace

namespace framework::net {

std::error_code dpdk_init_eal(const DpdkConfig &config) {
    std::vector<std::string> args;

    const std::string app_name =
            config.app_name.empty() ? get_current_process_name() : config.app_name;
    args.emplace_back(app_name);

    if (!config.file_prefix.empty()) {
        args.emplace_back(std::format("--file-prefix={}", config.file_prefix));
    }

    if (config.dpdk_core_id.has_value()) {
        args.emplace_back(std::format("-l{}", config.dpdk_core_id.value()));
        args.emplace_back(std::format("--main-lcore={}", config.dpdk_core_id.value()));
    }

    if (config.verbose_logs) {
        args.emplace_back("--log-level=,8");
        args.emplace_back("--log-level=pmd.net.mlx5:8");
    }

    args.emplace_back("--allow");
    args.emplace_back("0000:00:0.0");

    // Convert to C-style argument array for DPDK API compatibility
    std::vector<char *> argv;
    argv.reserve(args.size());
    // cppcheck-suppress constVariableReference
    for (auto &arg : args) {
        // DPDK API requires char** but we have const char* - this is safe since
        // DPDK only reads
        // cppcheck-suppress useStlAlgorithm
        argv.emplace_back(arg.data());
    }

    RT_LOGC_DEBUG(Net::NetDpdk, "DPDK EAL args: {}", args);

    // Set CPU affinity and RT priority, automatically restored on scope exit
    auto cleanup_func =
            set_scheduling_config(config.dpdk_core_id, config.enable_rt_priority_for_lcores);
    if (!cleanup_func.has_value()) {
        return make_error_code(DpdkErrc::EalInitFailed);
    }
    const auto sched_guard = gsl_lite::finally(std::move(cleanup_func.value()));

    const auto argc = static_cast<int>(argv.size());
    if (const int eal_ret = rte_eal_init(argc, argv.data()); eal_ret < 0) {
        RT_LOGC_ERROR(
                Net::NetDpdk,
                "Failed to initialize DPDK EAL with configuration: {}",
                rte_strerror(-eal_ret));
        return make_error_code(DpdkErrc::EalInitFailed);
    }

    RT_LOGC_DEBUG(Net::NetDpdk, "DPDK EAL initialized successfully with config: {}", config);
    return make_error_code(DpdkErrc::Success);
}

std::error_code dpdk_cleanup_eal() {
    const auto ret = rte_eal_cleanup();
    if (ret < 0) {
        RT_LOGC_ERROR(Net::NetDpdk, "DPDK EAL cleanup failed with error: {}", rte_strerror(-ret));
        return make_error_code(DpdkErrc::EalCleanupFailed);
    }

    RT_LOGC_DEBUG(Net::NetDpdk, "DPDK EAL cleanup completed successfully");
    return make_error_code(DpdkErrc::Success);
}

std::vector<std::string> discover_mellanox_nics() {
    // PCI device identification constants
    static constexpr std::string_view MELLANOX_VENDOR_ID = "0x15b3";
    static constexpr std::string_view ETHERNET_CLASS = "0x020000";

    std::vector<std::string> mellanox_pcis;

    try {
        // Scan /sys/bus/pci/devices/ for PCI devices
        const std::filesystem::path pci_devices_path = "/sys/bus/pci/devices";

        if (!std::filesystem::exists(pci_devices_path)) {
            RT_LOGC_WARN(
                    Net::NetDpdk, "PCI devices directory {} not found", pci_devices_path.c_str());
            return mellanox_pcis;
        }

        for (const auto &device_entry : std::filesystem::directory_iterator(pci_devices_path)) {
            if (!device_entry.is_directory()) {
                continue;
            }

            const std::string pci_address = device_entry.path().filename().string();
            const std::filesystem::path &device_path = device_entry.path();

            // Read vendor ID
            const std::filesystem::path vendor_path = device_path / "vendor";
            std::ifstream vendor_file(vendor_path);
            if (!vendor_file.is_open()) {
                continue; // Skip devices we can't read
            }

            std::string vendor_id;
            vendor_file >> vendor_id;
            vendor_file.close();

            // Check if it's a Mellanox device
            if (vendor_id != MELLANOX_VENDOR_ID) {
                continue;
            }

            // Read device class
            const std::filesystem::path class_path = device_path / "class";
            std::ifstream class_file(class_path);
            if (!class_file.is_open()) {
                continue; // Skip devices we can't read
            }

            std::string device_class;
            class_file >> device_class;
            class_file.close();

            // Check if it's an ethernet controller
            if (device_class != ETHERNET_CLASS) {
                continue;
            }

            mellanox_pcis.push_back(pci_address);
        }
    } catch (const std::exception &e) {
        RT_LOGC_WARN(Net::NetDpdk, "Error discovering Mellanox NICs: {}", e.what());
    }

    RT_LOGC_DEBUG(
            Net::NetDpdk,
            "Discovered {} Mellanox ethernet controllers {}",
            mellanox_pcis.size(),
            mellanox_pcis);
    return mellanox_pcis;
}

tl::expected<MacAddress, std::string> MacAddress::from_string(const std::string_view mac_str) {
    const auto mac_array = parse_mac_address(mac_str);
    if (!mac_array.has_value()) {
        return tl::unexpected(mac_array.error());
    }

    MacAddress mac_address{};
    mac_address.bytes = mac_array.value();
    return mac_address;
}

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
std::string MacAddress::to_string() const {
    return std::format(
            "{:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
            bytes[0],
            bytes[1],
            bytes[2],
            bytes[3],
            bytes[4],
            bytes[5]);
}
// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

bool MacAddress::is_zero() const {
    return std::all_of(
            bytes.begin(), bytes.end(), [](const std::uint8_t byte) { return byte == 0; });
}

std::error_code dpdk_set_port_mtu(const uint16_t port_id, const uint16_t mtu) {
    if (const auto ret = rte_eth_dev_set_mtu(port_id, mtu); ret != 0) {
        RT_LOGC_ERROR(
                Net::NetDpdk,
                "Failed to set MTU to {} for port {}: {}",
                mtu,
                port_id,
                rte_strerror(-ret));
        return make_error_code(DpdkErrc::PortMtuFailed);
    }

    RT_LOGC_DEBUG(Net::NetDpdk, "Successfully set MTU to {} for port {}", mtu, port_id);
    return make_error_code(DpdkErrc::Success);
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
DpdkPortState dpdk_try_configure_port(
        uint16_t port_id,
        uint16_t rxq_count,
        uint16_t txq_count,
        bool enable_accurate_send_scheduling) {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    rte_eth_conf eth_conf{};
    eth_conf.rxmode.offloads |= RTE_ETH_RX_OFFLOAD_TIMESTAMP;
    eth_conf.txmode.offloads |= RTE_ETH_TX_OFFLOAD_MULTI_SEGS;

    if (enable_accurate_send_scheduling) {
        eth_conf.txmode.offloads |= RTE_ETH_TX_OFFLOAD_SEND_ON_TIMESTAMP;
    }

    RT_LOGC_DEBUG(
            Net::NetDpdk,
            "Initializing DPDK port {} with {} RX queues and {} TX queues",
            port_id,
            rxq_count,
            txq_count);
    if (const auto ret = rte_eth_dev_configure(port_id, rxq_count, txq_count, &eth_conf);
        ret != 0) {
        if (ret == -EBUSY) {
            RT_LOGC_WARN(Net::NetDpdk, "Port {} was already configured", port_id);
            return DpdkPortState::AlreadyConfigured;
        }

        RT_LOGC_ERROR(
                Net::NetDpdk, "Failed to initialize port {}: {}", port_id, rte_strerror(-ret));
        return DpdkPortState::ConfigureError;
    }

    RT_LOGC_DEBUG(
            Net::NetDpdk,
            "RX offloads: {}, TX offloads {}",
            rx_offloads_to_string(eth_conf.rxmode.offloads),
            tx_offloads_to_string(eth_conf.txmode.offloads));

    return DpdkPortState::ConfigureSuccess;
}

std::error_code dpdk_try_tune_virtual_memory() {
    // Tune vm.zone_reclaim_mode=0
    if (auto result = tune_vm_parameter("zone_reclaim_mode", 0); result) {
        return result;
    }

    // Tune vm.swappiness=0
    if (auto result = tune_vm_parameter("swappiness", 0); result) {
        return result;
    }

    return make_error_code(DpdkErrc::Success);
}

std::error_code dpdk_enable_flow_rule_isolation(const uint16_t port_id) {
    rte_flow_error flow_error{};
    if (const auto ret = rte_flow_isolate(port_id, 1, &flow_error); ret != 0) {
        RT_LOGC_ERROR(
                Net::NetDpdk,
                "Failed to enable flow isolation on port {}: {}",
                port_id,
                flow_error.message);
        return make_error_code(DpdkErrc::FlowIsolationFailed);
    }

    RT_LOGC_DEBUG(Net::NetDpdk, "Successfully enabled flow isolation on port {}", port_id);
    return make_error_code(DpdkErrc::Success);
}

namespace {
[[nodiscard]] std::string_view fc_mode_to_string(rte_eth_fc_mode mode) {
    switch (mode) {
    case RTE_ETH_FC_NONE:
        return "RTE_ETH_FC_NONE";
    case RTE_ETH_FC_RX_PAUSE:
        return "RTE_ETH_FC_RX_PAUSE";
    case RTE_ETH_FC_TX_PAUSE:
        return "RTE_ETH_FC_TX_PAUSE";
    case RTE_ETH_FC_FULL:
        return "RTE_ETH_FC_FULL";
    default:
        return "UNKNOWN";
    }
}
} // namespace

std::error_code dpdk_disable_ethernet_flow_control(const uint16_t port_id) {
    rte_eth_fc_conf flow_control{};
    if (const auto ret = rte_eth_dev_flow_ctrl_get(port_id, &flow_control); ret != 0) {
        RT_LOGC_ERROR(
                Net::NetDpdk,
                "Failed to get port {} current Ethernet link flow control status: {}",
                port_id,
                rte_strerror(-ret));
        return make_error_code(DpdkErrc::FlowControlGetFailed);
    }

    const auto prev_mode = flow_control.mode;
    flow_control.mode = RTE_ETH_FC_NONE;

    if (const auto ret = rte_eth_dev_flow_ctrl_set(port_id, &flow_control); ret != 0) {
        RT_LOGC_WARN(
                Net::NetDpdk,
                "Failed to set port {} Ethernet link flow control: {}",
                port_id,
                rte_strerror(-ret));
        return make_error_code(DpdkErrc::FlowControlSetFailed);
    }

    RT_LOGC_DEBUG(
            Net::NetDpdk,
            "Successfully disabled Ethernet flow control for port {} "
            "(Previous mode: {}, Current mode: {}).",
            port_id,
            fc_mode_to_string(prev_mode),
            fc_mode_to_string(flow_control.mode));
    return make_error_code(DpdkErrc::Success);
}

std::error_code dpdk_calculate_timestamp_offsets(int &timestamp_offset, uint64_t &timestamp_mask) {

    const rte_mbuf_dynfield dynfield_desc{
            .name = RTE_MBUF_DYNFIELD_TIMESTAMP_NAME,
            .size = sizeof(uint64_t),
            .align = __alignof__(uint64_t),
            .flags = 0};

    const rte_mbuf_dynflag dynflag_desc{.name = RTE_MBUF_DYNFLAG_TX_TIMESTAMP_NAME, .flags = 0};

    timestamp_offset = rte_mbuf_dynfield_register(&dynfield_desc);
    if (timestamp_offset < 0) {
        RT_LOGC_ERROR(
                Net::NetDpdk,
                "{} registration error: {}",
                RTE_MBUF_DYNFIELD_TIMESTAMP_NAME,
                rte_strerror(rte_errno));
        return make_error_code(DpdkErrc::TimestampFieldFailed);
    }

    const int32_t dynflag_bitnum = rte_mbuf_dynflag_register(&dynflag_desc);
    if (dynflag_bitnum == -1) {
        RT_LOGC_ERROR(
                Net::NetDpdk,
                "{} registration error: {}",
                RTE_MBUF_DYNFLAG_TX_TIMESTAMP_NAME,
                rte_strerror(rte_errno));
        return make_error_code(DpdkErrc::TimestampFlagFailed);
    }

    const auto dynflag_shift = static_cast<uint8_t>(dynflag_bitnum);
    timestamp_mask = 1ULL << dynflag_shift;

    RT_LOGC_DEBUG(
            Net::NetDpdk,
            "Successfully calculated timestamp offsets: offset={}, mask=0x{:x}",
            timestamp_offset,
            timestamp_mask);
    return make_error_code(DpdkErrc::Success);
}

namespace {
/// PCIe Device Control register offset in config space
constexpr std::streamoff PCIE_DEVICE_CONTROL_OFFSET = 0x68;

/// Read PCIe Device Control register from sysfs config space
[[nodiscard]] std::optional<uint16_t>
read_pcie_device_control_register(const std::string_view pci_address) {

    const std::string config_path = std::format("/sys/bus/pci/devices/{}/config", pci_address);

    std::ifstream config_file(config_path, std::ios::binary);
    if (!config_file.is_open()) {
        return std::nullopt;
    }

    // Seek to Device Control register offset
    config_file.seekg(PCIE_DEVICE_CONTROL_OFFSET);
    if (!config_file.good()) {
        return std::nullopt;
    }

    // Read 16-bit Device Control register value
    uint16_t reg_value{};
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    config_file.read(reinterpret_cast<char *>(&reg_value), sizeof(reg_value));

    if (!config_file.good() && !config_file.eof()) {
        return std::nullopt;
    }

    return reg_value;
}

/// Extract MRRS value from PCIe Device Control register
constexpr PcieMrrs extract_mrrs_from_register(const uint16_t reg_value) {
    constexpr uint16_t MRRS_BIT_SHIFT = 12U;
    constexpr uint16_t MRRS_BIT_MASK = 0x7U;

    const auto mrrs_bits = static_cast<uint8_t>(
            static_cast<uint16_t>(reg_value >> MRRS_BIT_SHIFT) & MRRS_BIT_MASK);
    return static_cast<PcieMrrs>(mrrs_bits);
}
} // namespace

std::error_code dpdk_check_pcie_max_read_request_size(
        const std::string_view pci_address, const PcieMrrs expected_mrrs_value) {
    // Read current Device Control register value
    const auto current_reg_opt = read_pcie_device_control_register(pci_address);
    if (!current_reg_opt.has_value()) {
        RT_LOGC_WARN(
                Net::NetDpdk,
                "Failed to read PCIe Device Control register for device {}",
                pci_address);
        return make_error_code(DpdkErrc::PcieReadFailed);
    }

    const uint16_t current_reg_value = current_reg_opt.value();

    // Extract current MRRS value from register
    const auto current_mrrs_enum = extract_mrrs_from_register(current_reg_value);

    // Check if current value matches expected value
    if (current_mrrs_enum == expected_mrrs_value) {
        RT_LOGC_DEBUG(
                Net::NetDpdk,
                "PCIe MRRS for device {} is set to expected value {}",
                pci_address,
                ::wise_enum::to_string(expected_mrrs_value));
        return make_error_code(DpdkErrc::Success);
    }

    RT_LOGC_WARN(
            Net::NetDpdk,
            "PCIe MRRS for device {} is {} but expected {}. "
            "This may impact performance.",
            pci_address,
            ::wise_enum::to_string(current_mrrs_enum),
            ::wise_enum::to_string(expected_mrrs_value));
    return make_error_code(DpdkErrc::PcieVerifyMismatch);
}

std::error_code dpdk_log_link_info(const uint16_t port_id) {
    rte_eth_link link{};

    const auto ret = rte_eth_link_get(port_id, &link);
    if (ret != 0) {
        RT_LOGC_ERROR(
                Net::NetDpdk,
                "Failed to get link info for port {}: {}",
                port_id,
                rte_strerror(-ret));
        return make_error_code(DpdkErrc::LinkInfoFailed);
    }

    const std::string_view link_status = (link.link_status == RTE_ETH_LINK_DOWN) ? "DOWN" : "UP";
    const std::string_view link_duplex =
            (link.link_duplex == RTE_ETH_LINK_FULL_DUPLEX) ? "full-duplex" : "half-duplex";

    RT_LOGC_INFO(
            Net::NetDpdk,
            "Port {} link status: {}, {}, {}",
            port_id,
            link_status,
            rte_eth_link_speed_to_str(link.link_speed),
            link_duplex);
    return make_error_code(DpdkErrc::Success);
}

std::error_code dpdk_is_link_status_up(const uint16_t port_id) {
    rte_eth_link link{};

    const auto ret = rte_eth_link_get(port_id, &link);
    if (ret != 0) {
        RT_LOGC_ERROR(
                Net::NetDpdk,
                "Failed to get link status for port {}: {}",
                port_id,
                rte_strerror(-ret));
        return make_error_code(DpdkErrc::LinkInfoFailed);
    }

    if (link.link_status == RTE_ETH_LINK_DOWN) {
        return make_error_code(DpdkErrc::LinkDown);
    }

    return make_error_code(DpdkErrc::Success);
}

std::error_code dpdk_log_stats(const uint16_t port_id) {
    rte_eth_stats stats{};

    const auto ret = rte_eth_stats_get(port_id, &stats);
    if (ret != 0) {
        RT_LOGC_ERROR(Net::NetDpdk, "Failed to get port {} stats: {}", port_id, rte_strerror(-ret));
        return make_error_code(DpdkErrc::StatsFailed);
    }

    RT_LOGC_INFO(
            Net::NetDpdk,
            "Port {} stats: tx_packets={}, rx_packets={}, tx_bytes={}, rx_bytes={}, "
            "tx_errors={}, rx_errors={}, rx_missed={}, rx_nombuf={}",
            port_id,
            stats.opackets,
            stats.ipackets,
            stats.obytes,
            stats.ibytes,
            stats.oerrors,
            stats.ierrors,
            stats.imissed,
            stats.rx_nombuf);
    return make_error_code(DpdkErrc::Success);
}

std::error_code dpdk_start_eth_dev(const uint16_t port_id) {
    if (const auto ret = rte_eth_dev_start(port_id); ret != 0) {
        RT_LOGC_ERROR(
                Net::NetDpdk, "Failed to start DPDK port {}: {}", port_id, rte_strerror(-ret));
        return make_error_code(DpdkErrc::DevStartFailed);
    }

    RT_LOGC_DEBUG(Net::NetDpdk, "Successfully started DPDK port {}", port_id);
    return make_error_code(DpdkErrc::Success);
}

std::error_code dpdk_stop_eth_dev(const uint16_t port_id) {
    if (const auto ret = rte_eth_dev_stop(port_id); ret != 0) {
        RT_LOGC_WARN(Net::NetDpdk, "Failed to stop DPDK port {}: {}", port_id, rte_strerror(-ret));
        return make_error_code(DpdkErrc::DevStopFailed);
    }

    RT_LOGC_DEBUG(Net::NetDpdk, "DPDK port {} stopped successfully", port_id);
    return make_error_code(DpdkErrc::Success);
}

std::error_code
dpdk_setup_tx_queue(const uint16_t port_id, const uint16_t txq_id, const uint16_t txq_size) {
    const auto socket_id = static_cast<unsigned int>(rte_eth_dev_socket_id(port_id));
    if (const auto ret = rte_eth_tx_queue_setup(port_id, txq_id, txq_size, socket_id, nullptr);
        ret != 0) {
        RT_LOGC_ERROR(
                Net::NetDpdk,
                "Failed to setup DPDK TX queue {} on port {}: {}",
                txq_id,
                port_id,
                rte_strerror(-ret));
        return make_error_code(DpdkErrc::TxQueueSetupFailed);
    }

    RT_LOGC_DEBUG(Net::NetDpdk, "Successfully setup TX queue {} on port {}", txq_id, port_id);
    return make_error_code(DpdkErrc::Success);
}

std::error_code dpdk_validate_mellanox_driver(const uint16_t port_id) {
    static constexpr std::string_view MLX_PCI_DRIVER_NAME = "mlx5_pci";
    static constexpr std::string_view MLX_AUX_DRIVER_NAME = "mlx5_auxiliary";

    rte_eth_dev_info dev_info{};
    if (const auto ret = rte_eth_dev_info_get(port_id, &dev_info); ret != 0) {
        RT_LOGC_ERROR(
                Net::NetDpdk,
                "Failed to get device info for port {}: {}",
                port_id,
                rte_strerror(-ret));
        return make_error_code(DpdkErrc::DevInfoFailed);
    }
    if (dev_info.if_index == 0) {
        RT_LOGC_WARN(Net::NetDpdk, "No interface index available for port {}", port_id);
        return make_error_code(DpdkErrc::NoInterfaceIndex);
    }

    const auto driver_name = std::string(dev_info.driver_name);
    RT_LOGC_DEBUG(Net::NetDpdk, "Port {} is using {} driver", port_id, driver_name);
    if (driver_name != MLX_PCI_DRIVER_NAME && driver_name != MLX_AUX_DRIVER_NAME) {
        RT_LOGC_ERROR(Net::NetDpdk, "Non-Mellanox NICs are not supported");
        return make_error_code(DpdkErrc::UnsupportedDriver);
    }

    std::array<char, IFNAMSIZ> if_name{};
    if (if_indextoname(dev_info.if_index, if_name.data()) != nullptr) {
        RT_LOGC_DEBUG(
                Net::NetDpdk, "Port {} interface name: {}", port_id, std::string(if_name.data()));
    } else {
        RT_LOGC_ERROR(Net::NetDpdk, "Failed to get interface name for port {}", port_id);
        return make_error_code(DpdkErrc::InterfaceNameFailed);
    }

    RT_LOGC_DEBUG(Net::NetDpdk, "Mellanox driver validation passed for port {}", port_id);
    return make_error_code(DpdkErrc::Success);
}

std::error_code dpdk_create_mempool(
        const std::string_view name,
        const uint16_t port_id,
        const uint32_t num_mbufs,
        const uint32_t mtu_size,
        const HostPinned host_pinned,
        rte_mempool **mempool) {
    if (name.empty() || mempool == nullptr) {
        RT_LOGC_ERROR(Net::NetDpdk, "Invalid parameters for mempool creation");
        return make_error_code(DpdkErrc::MempoolCreateFailed);
    }

    *mempool = nullptr;

    // Calculate buffer sizes based on MTU and alignment requirements
    static constexpr std::size_t MBUF_POOL_DROOM_SZ_ALIGN = 128U;
    static constexpr uint32_t CACHE_SIZE = 0;
    const auto droom_sz = static_cast<uint16_t>(RTE_ALIGN_MUL_CEIL(
            mtu_size + RTE_PKTMBUF_HEADROOM + RTE_ETHER_HDR_LEN + RTE_ETHER_CRC_LEN,
            MBUF_POOL_DROOM_SZ_ALIGN));
    const uint32_t buff_size = droom_sz;

    const auto socket_id = rte_eth_dev_socket_id(port_id);

    RT_LOGC_DEBUG(
            Net::NetDpdk,
            "Creating DPDK mempool '{}' with {} mbufs, MTU={}, "
            "droom_sz={}, buff_size={}, cache_size={}, socket_id={}, host_pinned={}",
            name,
            num_mbufs,
            mtu_size,
            droom_sz,
            buff_size,
            CACHE_SIZE,
            socket_id,
            ::wise_enum::to_string(host_pinned));

    rte_mempool *mp{};

    // Setup cleanup guard - will clean up mp if it's set when we exit with error
    auto cleanup_guard = gsl_lite::finally([&mp] {
        if (mp != nullptr) {
            rte_mempool_free(mp);
        }
    });

    if (host_pinned == HostPinned::Yes) {
        mp = create_host_pinned_mempool(name, port_id, num_mbufs, droom_sz, socket_id);
        if (mp == nullptr) {
            return make_error_code(DpdkErrc::MempoolCreateFailed);
        }
    } else {
        // Create regular mempool with default memory operations
        mp = rte_pktmbuf_pool_create(
                name.data(),
                num_mbufs,
                CACHE_SIZE,
                sizeof(MempoolPrivateData),
                static_cast<uint16_t>(buff_size),
                socket_id);
        if (mp == nullptr) {
            RT_LOGC_ERROR(
                    Net::NetDpdk,
                    "Failed to create mempool '{}': {}",
                    name,
                    rte_strerror(rte_errno));
            return make_error_code(DpdkErrc::MempoolCreateFailed);
        }

        // Initialize private data for regular mempool (no special cleanup needed)
        auto *private_data = static_cast<MempoolPrivateData *>(rte_mempool_get_priv(mp));
        private_data->is_host_pinned = false;
        private_data->host_pinned_mem = nullptr;
        private_data->buffer_size = 0;
        private_data->port_id = 0;
    }

    RT_LOGC_DEBUG(
            Net::NetDpdk,
            "Successfully created mempool '{}' with {} elements",
            name,
            rte_mempool_avail_count(mp));

    *mempool = mp;
    mp = nullptr; // Success - transfer ownership, don't cleanup
    return make_error_code(DpdkErrc::Success);
}

std::error_code dpdk_destroy_mempool(rte_mempool *mempool) {
    if (mempool == nullptr) {
        RT_LOGC_DEBUG(Net::NetDpdk, "Mempool is nullptr, nothing to destroy");
        return make_error_code(DpdkErrc::Success);
    }

    RT_LOGC_DEBUG(Net::NetDpdk, "Destroying mempool '{}'", mempool->name);

    // Get private data to check if special cleanup is needed
    const auto *private_data =
            static_cast<const MempoolPrivateData *>(rte_mempool_get_priv(mempool));

    if (private_data->is_host_pinned && private_data->host_pinned_mem != nullptr) {
        // Perform external memory cleanup
        RT_LOGC_DEBUG(Net::NetDpdk, "Cleaning up host-pinned mempool external memory");

        // Get device info for DMA unmapping
        rte_eth_dev_info dev_info{};
        if (const int dev_info_ret = rte_eth_dev_info_get(private_data->port_id, &dev_info);
            dev_info_ret != 0) {
            RT_LOGC_ERROR(
                    Net::NetDpdk,
                    "Failed to get device info for port {} during cleanup: {}",
                    private_data->port_id,
                    rte_strerror(-dev_info_ret));
            // Continue with cleanup even if this fails
        } else {
            // Unmap DMA memory
            if (const int dma_unmap_ret = rte_dev_dma_unmap(
                        dev_info.device,
                        private_data->host_pinned_mem,
                        RTE_BAD_IOVA,
                        private_data->buffer_size);
                dma_unmap_ret != 0) {
                RT_LOGC_ERROR(
                        Net::NetDpdk,
                        "Failed to unmap DMA memory during cleanup: {}",
                        rte_strerror(-dma_unmap_ret));
                // Continue with cleanup even if this fails
            }
        }

        // Unregister external memory
        if (const int ret =
                    rte_extmem_unregister(private_data->host_pinned_mem, private_data->buffer_size);
            ret != 0) {
            RT_LOGC_ERROR(
                    Net::NetDpdk,
                    "Failed to unregister external memory during cleanup: {}",
                    rte_strerror(-ret));
            // Continue with cleanup even if this fails
        }

        // Capture host-pinned memory pointer before freeing mempool
        void *const host_pinned_mem = private_data->host_pinned_mem;

        // Free the mempool before freeing its backing memory to release references
        rte_mempool_free(mempool);

        // Free host-pinned memory
        if (const cudaError_t cuda_result = cudaFreeHost(host_pinned_mem);
            cuda_result != cudaSuccess) {
            RT_LOGC_ERROR(
                    Net::NetDpdk,
                    "Failed to free host-pinned memory at {}: {}",
                    host_pinned_mem,
                    cudaGetErrorString(cuda_result));
            // Continue with cleanup even if this fails
        }
    } else {
        // No host-pinned memory cleanup needed, just free the mempool
        rte_mempool_free(mempool);
    }

    RT_LOGC_DEBUG(Net::NetDpdk, "Successfully destroyed mempool");
    return make_error_code(DpdkErrc::Success);
}

std::error_code dpdk_eth_send(
        const std::span<const std::span<const uint8_t>> messages,
        const EthernetHeader &eth_header,
        rte_mempool *mempool,
        const uint16_t queue_id,
        const uint16_t port_id,
        const uint32_t max_retry_count) {
    if (messages.empty()) {
        RT_LOGC_DEBUG(Net::NetDpdk, "No messages to send");
        return make_error_code(DpdkErrc::Success);
    }

    if (mempool == nullptr) {
        RT_LOGC_ERROR(Net::NetDpdk, "Invalid mempool parameter");
        return make_error_code(DpdkErrc::MbufAllocFailed);
    }

    const auto nb_pkts = static_cast<uint16_t>(messages.size());
    RT_LOGC_DEBUG(
            Net::NetDpdk,
            "Sending {} packets: port={}, queue={}, max_retries={}",
            nb_pkts,
            port_id,
            queue_id,
            max_retry_count);

    static constexpr std::size_t CHUNK_SIZE = 256;
    std::array<rte_mbuf *, CHUNK_SIZE> mbuf_chunk{};

    // Process messages in chunks to avoid heap allocation
    for (std::size_t offset = 0; offset < messages.size(); offset += CHUNK_SIZE) {
        const std::size_t chunk_size = std::min(CHUNK_SIZE, messages.size() - offset);
        const auto chunk_messages = messages.subspan(offset, chunk_size);

        RT_LOGC_DEBUG(Net::NetDpdk, "Processing chunk: offset={}, size={}", offset, chunk_size);

        // Allocate mbufs for this chunk
        const int alloc_result = rte_pktmbuf_alloc_bulk(
                mempool, mbuf_chunk.data(), static_cast<unsigned int>(chunk_size));
        if (alloc_result != 0) {
            RT_LOGC_ERROR(
                    Net::NetDpdk,
                    "Failed to allocate {} mbufs from mempool for chunk at offset {}",
                    chunk_size,
                    offset);
            return make_error_code(DpdkErrc::MbufAllocFailed);
        }

        // Prepare each packet in this chunk
        for (std::size_t i = 0; i < chunk_size; ++i) {
            const auto &message = chunk_messages[i];
            // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
            rte_mbuf *mbuf = mbuf_chunk[i];

            // Calculate total packet size: Ethernet header (+ optional VLAN) + message
            const bool add_vlan = eth_header.has_vlan() && eth_header.vlan_tci().has_value();
            static constexpr std::size_t VLAN_HEADER_BYTES = 4;
            const std::size_t header_size =
                    EthernetHeader::NUM_BYTES + (add_vlan ? VLAN_HEADER_BYTES : 0);
            const std::size_t total_size = header_size + message.size();

            // Get pointer to packet data
            // cppcheck-suppress cstyleCast
            auto *pkt_data_base = rte_pktmbuf_mtod(mbuf, uint8_t *);
            std::size_t pkt_offset = 0;

            // Copy Ethernet header to packet
            // Destination MAC (6 bytes)
            // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
            rte_memcpy(
                    pkt_data_base + pkt_offset,
                    eth_header.dest_mac().bytes.data(),
                    MacAddress::ADDRESS_LEN);
            pkt_offset += MacAddress::ADDRESS_LEN;

            // Source MAC (6 bytes)
            rte_memcpy(
                    pkt_data_base + pkt_offset,
                    eth_header.src_mac().bytes.data(),
                    MacAddress::ADDRESS_LEN);
            pkt_offset += MacAddress::ADDRESS_LEN;

            if (add_vlan) {
                // VLAN TPID 0x8100 (2 bytes) + TCI (2 bytes) followed by inner EtherType
                const uint16_t tpid_be = rte_cpu_to_be_16(RTE_ETHER_TYPE_VLAN);
                rte_memcpy(pkt_data_base + pkt_offset, &tpid_be, sizeof(uint16_t));
                pkt_offset += sizeof(uint16_t);
                // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
                const uint16_t vlan_tci_be = rte_cpu_to_be_16(eth_header.vlan_tci().value());
                rte_memcpy(pkt_data_base + pkt_offset, &vlan_tci_be, sizeof(uint16_t));
                pkt_offset += sizeof(uint16_t);
            }

            // EtherType (outer or inner depending on VLAN presence; always payload EtherType here)
            const uint16_t ethertype_be = rte_cpu_to_be_16(eth_header.ether_type());
            rte_memcpy(pkt_data_base + pkt_offset, &ethertype_be, sizeof(uint16_t));
            pkt_offset += sizeof(uint16_t);

            // Copy message data to packet
            if (!message.empty()) {
                rte_memcpy(pkt_data_base + pkt_offset, message.data(), message.size());
            }
            // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)

            // Set packet length
            mbuf->data_len = static_cast<uint16_t>(total_size);
            mbuf->pkt_len = static_cast<uint32_t>(total_size);

            RT_LOGC_DEBUG(
                    Net::NetDpdk,
                    "Prepared packet {} (global {}): size={}, eth_type=0x{:04x}{}",
                    i,
                    offset + i,
                    total_size,
                    eth_header.ether_type(),
                    add_vlan ? " with VLAN" : "");
        }

        const std::span<rte_mbuf *> mbuf_span{mbuf_chunk.data(), chunk_size};
        const auto send_result = dpdk_eth_send_mbufs(mbuf_span, queue_id, port_id, max_retry_count);
        if (send_result) {
            RT_LOGC_ERROR(
                    Net::NetDpdk,
                    "Failed to send chunk at offset {}: {} packets on port {} queue {}",
                    offset,
                    chunk_size,
                    port_id,
                    queue_id);
            return send_result;
        }
    }

    RT_LOGC_DEBUG(
            Net::NetDpdk,
            "Successfully sent all {} packets on port {} queue {}",
            messages.size(),
            port_id,
            queue_id);
    return make_error_code(DpdkErrc::Success);
}

std::error_code dpdk_eth_send_mbufs(
        const std::span<rte_mbuf *> mbufs,
        const uint16_t queue_id,
        const uint16_t port_id,
        const uint32_t max_retry_count) {
    if (mbufs.empty()) {
        RT_LOGC_DEBUG(Net::NetDpdk, "No mbufs to send");
        return make_error_code(DpdkErrc::Success);
    }

    if (mbufs.size() > std::numeric_limits<uint16_t>::max()) {
        RT_LOGC_ERROR(Net::NetDpdk, "Mbuf count {} exceeds maximum DPDK burst size", mbufs.size());
        return make_error_code(DpdkErrc::InvalidParameter);
    }

    const auto nb_pkts = static_cast<uint16_t>(mbufs.size());
    RT_LOGC_DEBUG(
            Net::NetDpdk,
            "Sending {} mbufs: port={}, queue={}, max_retries={}",
            nb_pkts,
            port_id,
            queue_id,
            max_retry_count);

    // Send mbufs with retry loop and timeout
    uint16_t nb_sent = 0;
    uint32_t retry_count = 0;

    while (nb_sent < nb_pkts && retry_count < max_retry_count) {
        const uint16_t nb_remaining = nb_pkts - nb_sent;
        const uint16_t nb_tx = rte_eth_tx_burst(
                // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
                port_id,
                queue_id,
                mbufs.data() + nb_sent,
                nb_remaining);

        nb_sent += nb_tx;

        if (nb_tx == 0) {
            retry_count++;
        } else {
            retry_count = 0; // Reset retry count on successful send
        }
    }

    if (nb_sent != nb_pkts) {
        // Free the unsent mbufs
        const uint16_t nb_unsent = nb_pkts - nb_sent;
        for (uint16_t i = nb_sent; i < nb_pkts; ++i) {
            rte_pktmbuf_free(mbufs[i]);
        }

        RT_LOGC_ERROR(
                Net::NetDpdk,
                "Failed to send all mbufs on port {} queue {} after {} "
                "retries: sent {} of {}, freed {} unsent",
                port_id,
                queue_id,
                retry_count,
                nb_sent,
                nb_pkts,
                nb_unsent);
        return make_error_code(DpdkErrc::PacketSendFailed);
    }

    RT_LOGC_DEBUG(
            Net::NetDpdk,
            "Successfully sent {} mbufs on port {} queue {} (retries: {})",
            nb_sent,
            port_id,
            queue_id,
            retry_count);
    return make_error_code(DpdkErrc::Success);
}

} // namespace framework::net
