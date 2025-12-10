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

#ifndef FRAMEWORK_NET_DPDK_TYPES_HPP
#define FRAMEWORK_NET_DPDK_TYPES_HPP

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <vector>

#include <tl/expected.hpp>

#include <wise_enum.h>

#include "log/rt_log_macros.hpp"
#include "net/net_export.hpp"

namespace framework::net {

/**
 * Strongly typed MAC address
 *
 * Represents a 6-byte Ethernet MAC address with type safety.
 */
struct MacAddress {
    static constexpr std::size_t ADDRESS_LEN = 6U; //!< MAC address length

    std::array<std::uint8_t, ADDRESS_LEN> bytes{}; //!< MAC address bytes

    /**
     * Create MAC address from string
     *
     * @param[in] mac_str MAC address string in format "XX:XX:XX:XX:XX:XX"
     * @return MAC address on success, error message on failure
     */
    [[nodiscard]] static tl::expected<MacAddress, std::string>
    from_string(const std::string_view mac_str);

    /**
     * Convert MAC address to string representation
     *
     * @return String in format "XX:XX:XX:XX:XX:XX"
     */
    [[nodiscard]] std::string to_string() const;

    /**
     * Check if MAC address is zero (all bytes are 0x00)
     *
     * @return True if all bytes are zero
     */
    [[nodiscard]] bool is_zero() const;

    /**
     * Check if MAC address is equal to another MAC address
     *
     * @param[in] other The other MAC address to compare with
     * @return True if the MAC addresses are equal, false otherwise
     */
    [[nodiscard]] bool operator==(const MacAddress &other) const = default;
};

/**
 * Ethernet frame header
 *
 * Represents a standard Ethernet frame header with destination MAC,
 * source MAC, and EtherType fields.
 */
class EthernetHeader {
public:
    static constexpr std::size_t NUM_BYTES = 14; //!< Ethernet header size in bytes
    /**
     * Construct Ethernet header with specified parameters
     *
     * @param[in] src_mac Source MAC address
     * @param[in] dst_mac Destination MAC address
     * @param[in] ether_type EtherType value (e.g., 0x0800 for IPv4)
     * @param[in] vlan_tci Optional VLAN tag control information (in host byte order)
     */
    EthernetHeader(
            const MacAddress src_mac,
            const MacAddress dst_mac,
            const std::uint16_t ether_type,
            std::optional<std::uint16_t> vlan_tci = std::nullopt)
            : dst_mac_{dst_mac}, src_mac_{src_mac}, ether_type_{ether_type}, vlan_tci_{vlan_tci} {}

    /**
     * Get source MAC address
     *
     * @return Source MAC address
     */
    [[nodiscard]] const MacAddress &src_mac() const { return src_mac_; }

    /**
     * Get destination MAC address
     *
     * @return Destination MAC address
     */
    [[nodiscard]] const MacAddress &dest_mac() const { return dst_mac_; }

    /**
     * Get EtherType value
     *
     * @return EtherType value
     */
    [[nodiscard]] std::uint16_t ether_type() const { return ether_type_; }
    /**
     * Get optional VLAN TCI
     *
     * @return VLAN TCI if present
     */
    [[nodiscard]] const std::optional<std::uint16_t> &vlan_tci() const { return vlan_tci_; }
    /**
     * Whether a VLAN tag should be present
     *
     * @return True if a VLAN tag is configured and should be emitted, false otherwise
     */
    [[nodiscard]] bool has_vlan() const { return vlan_tci_.has_value(); }

    /**
     * Check if Ethernet header is equal to another header
     *
     * @param[in] other The other Ethernet header to compare with
     * @return True if the headers are equal, false otherwise
     */
    [[nodiscard]] bool operator==(const EthernetHeader &other) const = default;

private:
    MacAddress dst_mac_{};                  //!< Destination MAC address
    MacAddress src_mac_{};                  //!< Source MAC address
    std::uint16_t ether_type_{};            //!< EtherType field
    std::optional<std::uint16_t> vlan_tci_; //!< Optional VLAN tag control information
};

/**
 * DPDK configuration parameters
 */
struct DpdkConfig final {
    std::string app_name;    //!< Application name for DPDK EAL
    std::string file_prefix; //!< File prefix for DPDK shared files (--file-prefix)
    std::optional<std::uint32_t>
            dpdk_core_id;                 //!< DPDK core ID for main lcore (-l and --main-lcore=)
    bool verbose_logs{};                  //!< Enable verbose logging (--log-level=,8 and
                                          //!< --log-level=pmd.net.mlx5:8)
    bool enable_rt_priority_for_lcores{}; //!< Enable real-time priority
                                          //!< (SCHED_FIFO with priority 95) for
                                          //!< DPDK lcores
};

/**
 * PCIe Maximum Read Request Size values
 */
enum class PcieMrrs : std::uint8_t {
    Bytes128 = 0,  //!< 000b = 128 bytes
    Bytes256 = 1,  //!< 001b = 256 bytes
    Bytes512 = 2,  //!< 010b = 512 bytes
    Bytes1024 = 3, //!< 011b = 1024 bytes
    Bytes2048 = 4, //!< 100b = 2048 bytes
    Bytes4096 = 5  //!< 101b = 4096 bytes
};

/**
 * Host memory pinning configuration for DPDK mempool
 */
enum class HostPinned {
    No, //!< Host memory not pinned
    Yes //!< Host memory pinned
};

/**
 * DPDK error codes compatible with std::error_code
 *
 * This enum class provides type-safe error codes for DPDK operations
 * that integrate seamlessly with the standard C++ error handling framework.
 */
// clang-format off
enum class DpdkErrc : std::uint8_t {
    Success,                  //!< Operation succeeded
    EalInitFailed,           //!< EAL initialization failed
    EalCleanupFailed,        //!< EAL cleanup failed
    PortMtuFailed,           //!< Failed to set port MTU
    VmTuneFailed,            //!< Virtual memory tuning failed
    FlowIsolationFailed,     //!< Flow rule isolation failed
    FlowControlGetFailed,    //!< Failed to get flow control status
    FlowControlSetFailed,    //!< Failed to set flow control
    DevStartFailed,          //!< Failed to start ethernet device
    DevStopFailed,           //!< Failed to stop ethernet device
    TxQueueSetupFailed,      //!< TX queue setup failed
    TimestampFieldFailed,    //!< Timestamp field registration failed
    TimestampFlagFailed,     //!< Timestamp flag registration failed
    PcieReadFailed,          //!< PCIe register read failed
    PcieWriteFailed,         //!< PCIe register write failed
    PcieVerifyReadFailed,    //!< PCIe verification read failed
    PcieVerifyMismatch,      //!< PCIe verification value mismatch
    LinkInfoFailed,          //!< Link info retrieval failed
    LinkDown,                //!< Ethernet link is down
    StatsFailed,             //!< Statistics retrieval failed
    DevInfoFailed,           //!< Device info retrieval failed
    NoInterfaceIndex,        //!< No interface index available
    UnsupportedDriver,       //!< Non-Mellanox driver not supported
    InterfaceNameFailed,     //!< Interface name retrieval failed
    MempoolCreateFailed,     //!< Mempool creation failed
    MempoolDestroyFailed,    //!< Mempool destruction failed
    MbufAllocFailed,         //!< Mbuf allocation failed
    PacketSendFailed,        //!< Packet transmission failed
    InvalidParameter         //!< Invalid parameter provided
};
// clang-format on

static_assert(
        static_cast<std::uint32_t>(DpdkErrc::InvalidParameter) <=
                std::numeric_limits<std::uint8_t>::max(),
        "DpdkErrc enumerator values must fit in std::uint8_t");

} // namespace framework::net

// WISE_ENUM_ADAPT must be called at global namespace scope for ADL to work
WISE_ENUM_ADAPT(
        framework::net::PcieMrrs, Bytes128, Bytes256, Bytes512, Bytes1024, Bytes2048, Bytes4096)
WISE_ENUM_ADAPT(framework::net::HostPinned, No, Yes)
WISE_ENUM_ADAPT(
        framework::net::DpdkErrc,
        Success,
        EalInitFailed,
        EalCleanupFailed,
        PortMtuFailed,
        VmTuneFailed,
        FlowIsolationFailed,
        FlowControlGetFailed,
        FlowControlSetFailed,
        DevStartFailed,
        DevStopFailed,
        TxQueueSetupFailed,
        TimestampFieldFailed,
        TimestampFlagFailed,
        PcieReadFailed,
        PcieWriteFailed,
        PcieVerifyReadFailed,
        PcieVerifyMismatch,
        LinkInfoFailed,
        LinkDown,
        StatsFailed,
        DevInfoFailed,
        NoInterfaceIndex,
        UnsupportedDriver,
        InterfaceNameFailed,
        MempoolCreateFailed,
        MempoolDestroyFailed,
        MbufAllocFailed,
        PacketSendFailed,
        InvalidParameter)

// Register DpdkErrc as an error code enum to enable implicit conversion to
// std::error_code
// This must come before any functions that use std::error_code with DpdkErrc
namespace std {
template <> struct is_error_code_enum<framework::net::DpdkErrc> : true_type {};
} // namespace std

namespace framework::net {

/**
 * Custom error category for DPDK errors
 *
 * This class provides human-readable error messages and integrates DPDK errors
 * with the standard C++ error handling system.
 */
class DpdkErrorCategory final : public std::error_category {
private:
    // Compile-time table indexed by the enum's underlying value
    static constexpr std::array<std::string_view, 29> KMESSAGES{
            "Success: Operation completed successfully",
            "EAL initialization failed: Unable to initialize DPDK EAL",
            "EAL cleanup failed: Unable to cleanup DPDK EAL resources",
            "Port MTU failed: Unable to set port MTU size",
            "Virtual memory tuning failed: Unable to optimize VM parameters",
            "Flow isolation failed: Unable to enable flow rule isolation",
            "Flow control get failed: Unable to retrieve flow control status",
            "Flow control set failed: Unable to set flow control configuration",
            "Device start failed: Unable to start ethernet device",
            "Device stop failed: Unable to stop ethernet device",
            "TX queue setup failed: Unable to setup transmit queue",
            "Timestamp field failed: Unable to register timestamp dynamic field",
            "Timestamp flag failed: Unable to register timestamp dynamic flag",
            "PCIe read failed: Unable to read PCIe configuration register",
            "PCIe write failed: Unable to write PCIe configuration register",
            "PCIe verify read failed: Unable to verify PCIe register after write",
            "PCIe verify mismatch: PCIe register verification value mismatch",
            "Link info failed: Unable to retrieve ethernet link information",
            "Link down: Ethernet link is not up",
            "Statistics failed: Unable to retrieve ethernet port statistics",
            "Device info failed: Unable to retrieve ethernet device information",
            "No interface index: No network interface index available",
            "Unsupported driver: Non-Mellanox ethernet driver not supported",
            "Interface name failed: Unable to retrieve network interface name",
            "Mempool create failed: Unable to create DPDK mempool",
            "Mempool destroy failed: Unable to destroy DPDK mempool",
            "Mbuf allocation failed: Unable to allocate mbuf from mempool",
            "Packet send failed: Unable to transmit packet",
            "Invalid parameter: Parameter value is invalid or out of range"};

    // Ensure KMESSAGES array size matches the number of enum values
    static_assert(
            KMESSAGES.size() == ::wise_enum::size<DpdkErrc>,
            "KMESSAGES array size must match the number of DpdkErrc enum values");

public:
    /**
     * Get the name of this error category
     *
     * @return The category name as a C-style string
     */
    [[nodiscard]] const char *name() const noexcept override { return "adsp::net"; }

    /**
     * Get a descriptive message for the given error code
     *
     * @param[in] condition The error code value
     * @return A descriptive error message
     */
    [[nodiscard]] std::string message(const int condition) const override {
        const auto idx = static_cast<std::size_t>(condition);
        if (idx < KMESSAGES.size()) {
            return std::string{*std::next(KMESSAGES.begin(), static_cast<std::ptrdiff_t>(idx))};
        }
        return "Unknown DPDK error: " + std::to_string(condition);
    }

    /**
     * Map DPDK errors to standard error conditions where applicable
     *
     * @param[in] condition The error code value
     * @return The equivalent standard error condition, or a default-constructed
     * condition
     */
    [[nodiscard]] std::error_condition
    default_error_condition(const int condition) const noexcept override {
        switch (static_cast<DpdkErrc>(condition)) {
        case DpdkErrc::Success:
            return {};
        case DpdkErrc::UnsupportedDriver:
            return std::errc::operation_not_supported;
        case DpdkErrc::NoInterfaceIndex:
            return std::errc::no_such_device;
        default:
            // For DPDK-specific errors that don't map to standard conditions
            return std::error_condition{condition, *this};
        }
    }

    /**
     * Get the name of the error code enum value
     *
     * @param[in] condition The error code value
     * @return The enum name as a string
     */
    [[nodiscard]] static const char *name(const int condition) {
        const auto errc = static_cast<DpdkErrc>(condition);
        return ::wise_enum::to_string(errc).data();
    }
};

/**
 * Get the singleton instance of the DPDK error category
 *
 * @return Reference to the DPDK error category
 */
[[nodiscard]] inline const DpdkErrorCategory &dpdk_category() noexcept {
    static const DpdkErrorCategory instance{};
    return instance;
}

/**
 * Create an error_code from a DpdkErrc value
 *
 * @param[in] errc The DPDK error code
 * @return A std::error_code representing the DPDK error
 */
[[nodiscard]] inline std::error_code make_error_code(const DpdkErrc errc) noexcept {
    return {static_cast<int>(errc), dpdk_category()};
}

/**
 * Check if a DpdkErrc represents success
 *
 * @param[in] errc The error code to check
 * @return true if the error code represents success, false otherwise
 */
[[nodiscard]] constexpr bool is_success(const DpdkErrc errc) noexcept {
    return errc == DpdkErrc::Success;
}

/**
 * Check if an error_code represents DPDK success
 *
 * @param[in] errc The error code to check
 * @return true if the error code represents DPDK success, false otherwise
 */
[[nodiscard]] inline bool is_dpdk_success(const std::error_code &errc) noexcept {
    return errc.category() == dpdk_category() && errc.value() == 0;
}

/**
 * Get the name of a DpdkErrc enum value
 *
 * @param[in] errc The error code
 * @return The enum name as a string
 */
[[nodiscard]] inline const char *get_error_name(const DpdkErrc errc) noexcept {
    return ::wise_enum::to_string(errc).data();
}

/**
 * Get the name of a DpdkErrc from a std::error_code
 *
 * @param[in] ec The error code
 * @return The enum name as a string, or "unknown" if not a DPDK error
 */
[[nodiscard]] inline const char *get_error_name(const std::error_code &ec) noexcept {
    if (ec.category() != dpdk_category()) {
        return "unknown";
    }
    return get_error_name(static_cast<DpdkErrc>(ec.value()));
}

/**
 * Discover Mellanox NICs in the system
 *
 * Scans the system for Mellanox NICs by checking PCI vendor ID and device
 * class. Returns PCI addresses suitable for DPDK allowlisting.
 *
 * @return Vector of PCI addresses (e.g., "0000:29:00.1") for Mellanox ethernet
 * controllers
 */
[[nodiscard]] NET_EXPORT std::vector<std::string> discover_mellanox_nics();

} // namespace framework::net

// Must be in global namespace for quill to find it
// cppcheck-suppress functionStatic
RT_LOGGABLE_DEFERRED_FORMAT(
        framework::net::DpdkConfig,
        "app_name: {}, file_prefix: {}, verbose_logs: {}, dpdk_core_id: {}, "
        "enable_rt_priority_for_lcores: {}",
        obj.app_name,
        obj.file_prefix,
        obj.verbose_logs,
        obj.dpdk_core_id.has_value() ? std::to_string(obj.dpdk_core_id.value()) : "nullopt",
        obj.enable_rt_priority_for_lcores)

#endif // FRAMEWORK_NET_DPDK_TYPES_HPP
