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

#ifndef FRAMEWORK_NET_DPDK_TXQ_HPP
#define FRAMEWORK_NET_DPDK_TXQ_HPP

#include <cstdint>
#include <span>
#include <system_error>

#include "net/details/dpdk_utils.hpp"
#include "net/dpdk_types.hpp"
#include "net/net_export.hpp"

// Forward declarations
struct rte_mbuf;
// NOLINTNEXTLINE(readability-identifier-naming)
struct rte_mempool;

namespace framework::net {

/**
 * Configuration structure for DpdkTxQueue creation
 */
struct NET_EXPORT DpdkTxQConfig final {
    std::uint16_t txq_size{}; //!< TX queue size (number of descriptors)
};

/**
 * RAII wrapper for DPDK TX queue management
 *
 * Provides automatic resource management for DPDK transmit queues,
 * ensuring proper cleanup on destruction. Manages a single TX queue
 * on a DPDK port and provides methods for sending packets.
 */
class DpdkTxQueue final {
public:
    /**
     * Create and initialize a DPDK TX queue
     *
     * Sets up a TX queue for the specified DPDK ethernet port using
     * dpdk_setup_tx_queue. The port must be configured before calling
     * this constructor.
     *
     * @param[in] port_id DPDK port identifier
     * @param[in] txq_id TX queue identifier
     * @param[in] config Configuration parameters for the TX queue
     * @throws std::runtime_error if queue setup fails
     */
    NET_EXPORT
    DpdkTxQueue(std::uint16_t port_id, std::uint16_t txq_id, const DpdkTxQConfig &config);

    /**
     * Send multiple Ethernet packets using DPDK
     *
     * Allocates mbufs in bulk from the mempool, copies the Ethernet header and
     * message data for each packet, and transmits them using the configured
     * TX queue. Any unsent mbufs are automatically freed.
     *
     * @param[in] messages 2D span of message data to send (each inner span is one
     * message)
     * @param[in] eth_header Ethernet header to prepend to each message
     * @param[in] mempool DPDK mempool for mbuf allocation
     * @param[in] max_retry_count Maximum number of retries when no progress is
     * made
     * @return std::error_code indicating success or specific error condition
     */
    [[nodiscard]] NET_EXPORT std::error_code
    send(std::span<const std::span<const std::uint8_t>> messages,
         const EthernetHeader &eth_header,
         rte_mempool *mempool,
         std::uint32_t max_retry_count = DEFAULT_MAX_RETRY_COUNT) const;

    /**
     * Send pre-allocated mbufs using DPDK
     *
     * Transmits a span of pre-allocated and prepared mbufs using the configured
     * TX queue. The caller is responsible for mbuf allocation and data
     * preparation. Any unsent mbufs are automatically freed.
     *
     * @param[in] mbufs Span of pre-allocated mbufs to send
     * @param[in] max_retry_count Maximum number of retries when no progress is
     * made
     * @return std::error_code indicating success or specific error condition
     */
    [[nodiscard]] NET_EXPORT std::error_code send_mbufs(
            std::span<rte_mbuf *> mbufs,
            std::uint32_t max_retry_count = DEFAULT_MAX_RETRY_COUNT) const;

    /**
     * Get the DPDK port ID for this TX queue
     *
     * @return DPDK port identifier
     */
    [[nodiscard]] std::uint16_t port_id() const noexcept { return port_id_; }

    /**
     * Get the TX queue ID
     *
     * @return TX queue identifier
     */
    [[nodiscard]] std::uint16_t queue_id() const noexcept { return queue_id_; }

private:
    std::uint16_t port_id_{};  //!< DPDK port identifier
    std::uint16_t queue_id_{}; //!< TX queue identifier
};

} // namespace framework::net

#endif // FRAMEWORK_NET_DPDK_TXQ_HPP
