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

#ifndef FRAMEWORK_NET_DOCA_RXQ_HPP
#define FRAMEWORK_NET_DOCA_RXQ_HPP

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include <doca_error.h>

#include "net/doca_types.hpp"
#include "net/dpdk_types.hpp"
#include "net/net_export.hpp"

namespace framework::net {

/**
 * Configuration structure for DocaRxQ creation
 */
struct NET_EXPORT DocaRxQConfig final {
    std::string nic_pcie_addr;    //!< NIC PCIe address (e.g., "0000:3a:00.0")
    MacAddress sender_mac_addr;   //!< Sender MAC address for flow rule
    std::uint32_t max_pkt_num{};  //!< Maximum number of packets in queue
    std::uint32_t max_pkt_size{}; //!< Maximum packet size in bytes
    std::uint16_t ether_type{};   //!< EtherType value for flow rule matching
    std::optional<std::uint16_t>
            vlan_tci; //!< Optional VLAN TCI (if not set, no VLAN matching in flow rule)
    std::optional<DocaSemItems>
            sem_items; //!< Semaphore configuration (if not set, no semaphores are created)
};

/**
 * RAII wrapper for DOCA RX queue management
 *
 * Provides automatic resource management for DOCA receive queues,
 * including flow rule setup and cleanup.
 */
class DocaRxQ final {
public:
    /**
     * Create and initialize a DOCA RX queue with flow rule
     *
     * @param[in] config Configuration parameters for the RX queue and flow rule
     * @param[in] gpu_dev DOCA GPU device
     * @param[in] ddev DOCA device
     * @throws std::runtime_error if queue or flow rule creation fails
     */
    NET_EXPORT DocaRxQ(const DocaRxQConfig &config, doca_gpu *gpu_dev, doca_dev *ddev);

    /**
     * Get the underlying DOCA RX queue structure
     *
     * @return Pointer to the internal DocaRxQParams structure
     */
    [[nodiscard]] NET_EXPORT const DocaRxQParams *params() const noexcept;

private:
    /**
     * Custom deleter for DocaRxQParams that calls doca_destroy_rxq
     */
    struct RxqDeleter {
        void operator()(DocaRxQParams *rxq) const noexcept;
    };

    std::unique_ptr<DocaRxQParams, RxqDeleter> rxq_; //!< Internal RX queue structure
};

} // namespace framework::net

#endif // FRAMEWORK_NET_DOCA_RXQ_HPP
