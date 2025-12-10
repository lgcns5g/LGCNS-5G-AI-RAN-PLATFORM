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

#ifndef FRAMEWORK_NET_NIC_HPP
#define FRAMEWORK_NET_NIC_HPP

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <doca_error.h>

#include "net/details/dpdk_utils.hpp"
#include "net/doca_rxq.hpp"
#include "net/doca_txq.hpp"
#include "net/doca_types.hpp"
#include "net/dpdk_txq.hpp"
#include "net/dpdk_types.hpp"
#include "net/mempool.hpp"
#include "net/net_export.hpp"

// Forward declarations
struct rte_mbuf;

namespace framework::net {

/**
 * Configuration structure for NIC creation
 */
struct NET_EXPORT NicConfig final {
    std::string nic_pcie_addr;                   //!< NIC PCIe address (e.g., "0000:3a:00.0")
    std::vector<DocaTxQConfig> doca_txq_configs; //!< DOCA TX queue configurations
    std::vector<DocaRxQConfig> doca_rxq_configs; //!< DOCA RX queue configurations
    std::vector<DpdkTxQConfig> dpdk_txq_configs; //!< DPDK TX queue configurations
    std::vector<MempoolConfig> mempool_configs;  //!< Mempool configurations
    bool enable_accurate_send_scheduling{false}; //!< Enable accurate send scheduling
    // NOLINTNEXTLINE(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
    std::uint16_t max_mtu_size{9216};        //!< Maximum MTU size in bytes (jumbo frame)
    PcieMrrs pcie_mrrs{PcieMrrs::Bytes4096}; //!< PCIe Maximum Read Request Size
};

/**
 * RAII wrapper for DOCA NIC device management
 *
 * Provides automatic resource management for DOCA network devices,
 * including MAC address resolution, DPDK port mapping, and queue management.
 */
class Nic final {
public:
    /**
     * Create and initialize a DOCA NIC device with queues
     *
     * @param[in] config Configuration parameters for the NIC and its queues
     * @param[in] gpu_dev DOCA GPU device for queue creation (optional, required
     * only for DOCA queues)
     * @throws std::runtime_error if NIC device creation fails
     * @throws std::invalid_argument if configuration parameters are invalid
     */
    explicit NET_EXPORT
    Nic(const NicConfig &config, std::optional<doca_gpu *> gpu_dev = std::nullopt);

    /**
     * Get the PCI address for this NIC
     *
     * @return PCI address string (e.g., "0000:3a:00.0")
     */
    [[nodiscard]] NET_EXPORT const std::string &pci_address() const noexcept;

    /**
     * Get the MAC address for this NIC
     *
     * @return Strongly typed MAC address
     */
    [[nodiscard]] NET_EXPORT const MacAddress &mac_address() const noexcept;

    /**
     * Get the DPDK port ID for this NIC
     *
     * @return DPDK port ID
     */
    [[nodiscard]] NET_EXPORT std::uint16_t dpdk_port_id() const noexcept;

    /**
     * Get the underlying DOCA device pointer
     *
     * @return Pointer to the DOCA device
     */
    [[nodiscard]] NET_EXPORT doca_dev *get() const noexcept;

    /**
     * Check if this is a CX-6 device
     *
     * @return True if CX-6 device, false otherwise
     */
    [[nodiscard]] NET_EXPORT bool is_cx6_device() const noexcept;

    /**
     * Check if NIC supports RDMA operations
     *
     * Verifies that this NIC can perform RDMA operations, which is required for
     * GPU-accelerated RDMA datapath. This checks if the device supports RDMA
     * write operations.
     *
     * @return True if RDMA is supported, error message on failure
     */
    [[nodiscard]] NET_EXPORT tl::expected<bool, std::string> is_rdma_supported() const noexcept;

    /**
     * Get the DOCA TX queues
     *
     * @return Reference to vector of DOCA TX queues
     */
    [[nodiscard]] NET_EXPORT const std::vector<DocaTxQ> &doca_tx_queues() const noexcept;

    /**
     * Get the DOCA RX queues
     *
     * @return Reference to vector of DOCA RX queues
     */
    [[nodiscard]] NET_EXPORT const std::vector<DocaRxQ> &doca_rx_queues() const noexcept;

    /**
     * Get a specific DOCA TX queue by index
     *
     * @param[in] index Queue index
     * @return Reference to the DOCA TX queue
     * @throws std::out_of_range if index is invalid
     */
    [[nodiscard]] NET_EXPORT const DocaTxQ &doca_tx_queue(std::size_t index) const;

    /**
     * Get a specific DOCA RX queue by index
     *
     * @param[in] index Queue index
     * @return Reference to the DOCA RX queue
     * @throws std::out_of_range if index is invalid
     */
    [[nodiscard]] NET_EXPORT const DocaRxQ &doca_rx_queue(std::size_t index) const;

    /**
     * Get the DPDK TX queues
     *
     * @return Reference to vector of DPDK TX queues
     */
    [[nodiscard]] NET_EXPORT const std::vector<DpdkTxQueue> &dpdk_tx_queues() const noexcept;

    /**
     * Get a specific DPDK TX queue by index
     *
     * @param[in] index Queue index
     * @return Reference to the DPDK TX queue
     * @throws std::out_of_range if index is invalid
     */
    [[nodiscard]] NET_EXPORT const DpdkTxQueue &dpdk_tx_queue(std::size_t index) const;

    /**
     * Get the mempools
     *
     * @return Reference to vector of mempools
     */
    [[nodiscard]] NET_EXPORT const std::vector<Mempool> &mempools() const noexcept;

    /**
     * Get a specific mempool by index
     *
     * @param[in] index Mempool index
     * @return Reference to the mempool
     * @throws std::out_of_range if index is invalid
     */
    [[nodiscard]] NET_EXPORT const Mempool &mempool(std::size_t index) const;

    /**
     * Send multiple Ethernet packets using DPDK TX queue
     *
     * Sends packets using the specified DPDK TX queue and mempool.
     * Allocates mbufs in bulk from the specified mempool, copies the Ethernet
     * header and message data for each packet, and transmits them using the
     * specified queue.
     *
     * @param[in] dpdk_txq_id DPDK TX queue index
     * @param[in] mempool_id Mempool index
     * @param[in] messages 2D span of message data to send (each inner span is one
     * message)
     * @param[in] eth_header Ethernet header to prepend to each message
     * @param[in] max_retry_count Maximum number of retries when no progress is
     * made
     * @return std::error_code indicating success or specific error condition
     * @throws std::out_of_range if dpdk_txq_id or mempool_id is invalid
     */
    [[nodiscard]] NET_EXPORT std::error_code
    send(std::size_t dpdk_txq_id,
         std::size_t mempool_id,
         std::span<const std::span<const std::uint8_t>> messages,
         const EthernetHeader &eth_header,
         std::uint32_t max_retry_count = DEFAULT_MAX_RETRY_COUNT) const;

    /**
     * Send pre-allocated mbufs using DPDK TX queue
     *
     * Transmits a span of pre-allocated and prepared mbufs using the specified
     * DPDK TX queue. The caller is responsible for mbuf allocation and data
     * preparation. Any unsent mbufs are automatically freed by this function.
     *
     * @param[in] dpdk_txq_id DPDK TX queue index
     * @param[in] mbufs Span of pre-allocated mbufs to send
     * @param[in] max_retry_count Maximum number of retries when no progress is
     * made
     * @return std::error_code indicating success or specific error condition
     * @throws std::out_of_range if dpdk_txq_id is invalid
     */
    [[nodiscard]] NET_EXPORT std::error_code send_mbufs(
            std::size_t dpdk_txq_id,
            std::span<rte_mbuf *> mbufs,
            std::uint32_t max_retry_count = DEFAULT_MAX_RETRY_COUNT) const;

private:
    /**
     * Custom deleter for doca_dev that calls doca_close_device
     */
    struct NicDeleter {
        void operator()(doca_dev *ddev) const noexcept;
    };

    // IMPORTANT: Member declaration order matters for RAII cleanup
    // Members are destroyed in reverse order of declaration, ensuring:
    // 1. queues destroyed first (releases references to mempool buffers)
    // 2. ddev_ destroyed second (stops DPDK port, releases queue resources)
    // 3. mempools_ destroyed last (frees memory after all references released)
    std::string pci_address_;                    //!< Cached PCI address
    MacAddress mac_address_{};                   //!< Cached MAC address
    std::uint16_t dpdk_port_id_{};               //!< Cached DPDK port ID
    bool is_cx6_device_{};                       //!< Whether this is a CX-6 device
    std::vector<Mempool> mempools_;              //!< DPDK mempools
    std::unique_ptr<doca_dev, NicDeleter> ddev_; //!< DOCA device
    std::vector<DocaTxQ> doca_tx_queues_;        //!< DOCA TX queues
    std::vector<DocaRxQ> doca_rx_queues_;        //!< DOCA RX queues
    std::vector<DpdkTxQueue> dpdk_tx_queues_;    //!< DPDK TX queues
};

} // namespace framework::net

#endif // FRAMEWORK_NET_NIC_HPP
