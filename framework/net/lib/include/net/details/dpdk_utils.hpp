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

#ifndef FRAMEWORK_NET_DPDK_UTILS_HPP
#define FRAMEWORK_NET_DPDK_UTILS_HPP

#include <span>
#include <system_error>

#include "net/dpdk_types.hpp"

// Forward declarations
struct rte_mbuf;
// NOLINTNEXTLINE(readability-identifier-naming)
struct rte_mempool;

namespace framework::net {

/// Default maximum retry count for DPDK operations
static constexpr std::uint32_t DEFAULT_MAX_RETRY_COUNT = 1000;

/**
 * DPDK port configuration result states
 *
 * Represents the possible outcomes when attempting to configure a DPDK
 * ethernet port. Used by dpdk_try_configure_port to indicate whether the
 * configuration operation succeeded, failed, or was unnecessary.
 */
enum class DpdkPortState {
    ConfigureSuccess, //!< Port configuration succeeded
    ConfigureError,   //!< Port configuration failed
    AlreadyConfigured //!< Port was already configured
};

} // namespace framework::net

// WISE_ENUM_ADAPT must be called at global namespace scope for ADL to work
WISE_ENUM_ADAPT(framework::net::DpdkPortState, ConfigureSuccess, ConfigureError, AlreadyConfigured)

namespace framework::net {

/**
 * Initialize DPDK EAL with configuration parameters
 *
 * @param[in] config DPDK configuration parameters
 * @return std::error_code indicating success or specific error condition
 */
[[nodiscard]] NET_EXPORT std::error_code dpdk_init_eal(const DpdkConfig &config);

/**
 * Cleanup DPDK EAL resources
 *
 * @return std::error_code indicating success or specific error condition
 */
[[nodiscard]] NET_EXPORT std::error_code dpdk_cleanup_eal();

/**
 * Set MTU size for DPDK ethernet port
 *
 * @param[in] port_id DPDK port identifier
 * @param[in] mtu MTU size in bytes
 * @return std::error_code indicating success or specific error condition
 */
[[nodiscard]] NET_EXPORT std::error_code dpdk_set_port_mtu(uint16_t port_id, uint16_t mtu);

/**
 * @brief Configure DPDK ethernet port
 *
 * @param[in] port_id DPDK port identifier
 * @param[in] rxq_count Number of RX queues
 * @param[in] txq_count Number of TX queues
 * @param[in] enable_accurate_send_scheduling Whether to enable accurate send
 * scheduling
 * @return DpdkPortState indicating configuration result
 */
[[nodiscard]] NET_EXPORT DpdkPortState dpdk_try_configure_port(
        uint16_t port_id,
        uint16_t rxq_count,
        uint16_t txq_count,
        bool enable_accurate_send_scheduling);

/**
 * Tune virtual memory settings for optimal network performance
 *
 * Optimizes vm.zone_reclaim_mode and vm.swappiness kernel parameters
 * for DPDK network operations. Sets both parameters to 0 for best
 * performance.
 *
 * @return std::error_code indicating success or specific error condition
 */
[[nodiscard]] NET_EXPORT std::error_code dpdk_try_tune_virtual_memory();

/**
 * Restrict ingress traffic for DPDK ethernet port
 *
 * Enables flow isolation on the specified port to restrict ingress traffic.
 * This allows only explicitly configured flows to reach the application.
 *
 * @param[in] port_id DPDK port identifier
 * @return std::error_code indicating success or specific error condition
 */
[[nodiscard]] NET_EXPORT std::error_code dpdk_enable_flow_rule_isolation(uint16_t port_id);

/**
 * Disable Ethernet flow control for DPDK port
 *
 * Disables flow control on the specified port by setting the mode to
 * RTE_ETH_FC_NONE. This prevents the port from sending or responding
 * to flow control frames.
 *
 * @param[in] port_id DPDK port identifier
 * @return std::error_code indicating success or specific error condition
 */
[[nodiscard]] NET_EXPORT std::error_code dpdk_disable_ethernet_flow_control(uint16_t port_id);

/**
 * Start DPDK ethernet device
 *
 * Starts the specified DPDK ethernet port and enables packet processing.
 * The port must be configured before calling this function.
 *
 * @param[in] port_id DPDK port identifier
 * @return std::error_code indicating success or specific error condition
 */
[[nodiscard]] NET_EXPORT std::error_code dpdk_start_eth_dev(uint16_t port_id);

/**
 * Stop DPDK ethernet device
 *
 * Stops the specified DPDK ethernet port and disables packet processing.
 * This function can be called safely even if the port is already stopped.
 *
 * @param[in] port_id DPDK port identifier
 * @return std::error_code indicating success or specific error condition
 */
[[nodiscard]] NET_EXPORT std::error_code dpdk_stop_eth_dev(uint16_t port_id);

/**
 * Setup DPDK TX queue
 *
 * Sets up a TX queue for the specified DPDK ethernet port.
 * The port must be configured before calling this function.
 *
 * @param[in] port_id DPDK port identifier
 * @param[in] txq_id TX queue identifier
 * @param[in] txq_size TX queue size (number of descriptors)
 * @return std::error_code indicating success or specific error condition
 */
[[nodiscard]] NET_EXPORT std::error_code
dpdk_setup_tx_queue(uint16_t port_id, uint16_t txq_id, uint16_t txq_size);

/**
 * Calculate timestamp offsets for DPDK
 *
 * Registers dynamic fields and flags for timestamp-based packet transmission.
 * Calculates the timestamp offset and mask required for accurate send
 * scheduling.
 *
 * @param[out] timestamp_offset Offset for timestamp field in mbuf
 * @param[out] timestamp_mask Mask for timestamp flag
 * @return std::error_code indicating success or specific error condition
 */
[[nodiscard]] NET_EXPORT std::error_code
dpdk_calculate_timestamp_offsets(int &timestamp_offset, uint64_t &timestamp_mask);

/**
 * Check PCIe Maximum Read Request Size
 *
 * Uses sysfs to read the current PCIe MRRS and compare it with the expected
 * value. Logs a warning if the current value doesn't match the expected value,
 * which may impact network performance.
 *
 * @param[in] pci_address PCI address in format "XXXX:XX:XX.X"
 * @param[in] expected_mrrs_value Expected MRRS value
 * @return std::error_code success if values match, pcie_verify_mismatch if
 * different
 */
[[nodiscard]] NET_EXPORT std::error_code
dpdk_check_pcie_max_read_request_size(std::string_view pci_address, PcieMrrs expected_mrrs_value);

/**
 * Log DPDK ethernet port link information
 *
 * Retrieves and logs the current link status, speed, and duplex mode
 * for the specified DPDK port.
 *
 * @param[in] port_id DPDK port identifier
 * @return std::error_code indicating success or specific error condition
 */
[[nodiscard]] NET_EXPORT std::error_code dpdk_log_link_info(uint16_t port_id);

/**
 * Check if DPDK ethernet port link is up
 *
 * Retrieves the link status for the specified DPDK port and checks
 * if the link is up.
 *
 * @param[in] port_id DPDK port identifier
 * @return std::error_code success if link is up, link_down if link is down,
 * or other error on failure
 */
[[nodiscard]] NET_EXPORT std::error_code dpdk_is_link_status_up(uint16_t port_id);

/**
 * Log DPDK ethernet port statistics
 *
 * Retrieves and logs detailed statistics including packet counts,
 * byte counts, and error counts for the specified DPDK port on a single line.
 *
 * @param[in] port_id DPDK port identifier
 * @return std::error_code indicating success or specific error condition
 */
[[nodiscard]] NET_EXPORT std::error_code dpdk_log_stats(uint16_t port_id);

/**
 * Validate Mellanox driver for DPDK ethernet port
 *
 * Checks device driver information for the specified DPDK port and validates
 * it is a supported Mellanox driver. Also retrieves the network interface name.
 *
 * @param[in] port_id DPDK port identifier
 * @return std::error_code indicating success or specific error condition
 */
[[nodiscard]] NET_EXPORT std::error_code dpdk_validate_mellanox_driver(uint16_t port_id);

/**
 * Create DPDK mempool with optional host memory pinning
 *
 * Creates a DPDK mempool for packet buffers with configurable memory type.
 * When host pinning is enabled, uses CUDA host-pinned memory for better
 * GPU-CPU transfer performance. Calculates buffer sizes based on MTU and
 * uses proper alignment for GPU operations.
 *
 * @param[in] name Unique name for the mempool (DPDK copies internally)
 * @param[in] port_id DPDK port ID to determine NUMA socket
 * @param[in] num_mbufs Number of mbufs in the mempool
 * @param[in] mtu_size MTU size for buffer calculations
 * @param[in] host_pinned Whether to use host-pinned memory
 * @param[out] mempool Pointer to store the created mempool
 * @return std::error_code indicating success or specific error condition
 */
[[nodiscard]] NET_EXPORT std::error_code dpdk_create_mempool(
        std::string_view name,
        uint16_t port_id,
        uint32_t num_mbufs,
        uint32_t mtu_size,
        HostPinned host_pinned,
        rte_mempool **mempool);

/**
 * Destroy DPDK mempool and free associated resources
 *
 * Safely destroys a DPDK mempool created with dpdk_create_mempool.
 * Handles cleanup of both regular and host-pinned memory pools.
 *
 * @param[in] mempool Pointer to the mempool to destroy (can be nullptr)
 * @return std::error_code indicating success or specific error condition
 */
[[nodiscard]] NET_EXPORT std::error_code dpdk_destroy_mempool(rte_mempool *mempool);

/**
 * Send multiple Ethernet packets using DPDK
 *
 * Allocates mbufs in bulk from the mempool, copies the Ethernet header and
 * message data for each packet, and transmits them using rte_eth_tx_burst with
 * retry logic. Any unsent mbufs are automatically freed by this function.
 *
 * @param[in] messages 2D span of message data to send (each inner span is one
 * message)
 * @param[in] eth_header Ethernet header to prepend to each message
 * @param[in] mempool DPDK mempool for mbuf allocation
 * @param[in] queue_id TX queue identifier
 * @param[in] port_id DPDK port identifier
 * @param[in] max_retry_count Maximum number of retries when no progress is made
 * @return std::error_code indicating success or specific error condition
 */
[[nodiscard]] NET_EXPORT std::error_code dpdk_eth_send(
        std::span<const std::span<const uint8_t>> messages,
        const EthernetHeader &eth_header,
        rte_mempool *mempool,
        uint16_t queue_id,
        uint16_t port_id,
        uint32_t max_retry_count = DEFAULT_MAX_RETRY_COUNT);

/**
 * Send pre-allocated mbufs using DPDK
 *
 * Transmits a span of pre-allocated and prepared mbufs using rte_eth_tx_burst.
 * This function provides zero-copy transmission for high-performance scenarios.
 * The caller is responsible for mbuf allocation and data preparation.
 * Any unsent mbufs are automatically freed by this function.
 *
 * @note This is a high-performance path - mbufs should be pre-allocated
 *       and configured for optimal throughput
 * @note Maximum burst size is limited by DPDK's uint16_t constraints
 *
 * @param[in] mbufs Span of pre-allocated mbufs to send
 * @param[in] queue_id TX queue identifier
 * @param[in] port_id DPDK port identifier
 * @param[in] max_retry_count Maximum number of retries when no progress is made
 * @return std::error_code indicating success or specific error condition
 */
[[nodiscard]] NET_EXPORT std::error_code dpdk_eth_send_mbufs(
        const std::span<rte_mbuf *> mbufs,
        uint16_t queue_id,
        uint16_t port_id,
        uint32_t max_retry_count = DEFAULT_MAX_RETRY_COUNT);

} // namespace framework::net

#endif
