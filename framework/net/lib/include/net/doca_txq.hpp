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

#ifndef FRAMEWORK_NET_DOCA_TXQ_HPP
#define FRAMEWORK_NET_DOCA_TXQ_HPP

#include <cstdint>
#include <memory>
#include <string>

#include <doca_error.h>

#include "net/doca_types.hpp"
#include "net/dpdk_types.hpp"
#include "net/net_export.hpp"

namespace framework::net {

/**
 * Configuration structure for DocaTxQ creation
 */
struct NET_EXPORT DocaTxQConfig final {
    std::string nic_pcie_addr;        //!< NIC PCIe address (e.g., "0000:3a:00.0")
    MacAddress dest_mac_addr;         //!< Destination MAC address
    std::uint32_t pkt_size{};         //!< Packet size to send
    std::uint32_t pkt_num{};          //!< Number of packets
    std::uint32_t max_sq_descr_num{}; //!< Maximum number of send queue descriptors
    std::uint16_t ether_type{};       //!< EtherType value for packet headers
    std::optional<std::uint16_t>
            vlan_tci; //!< Optional VLAN TCI (if set, insert 802.1Q tag with inner EtherType)
};

/**
 * RAII wrapper for DOCA TX queue management
 *
 * Provides automatic resource management for DOCA transmit queues,
 * ensuring proper cleanup on destruction.
 */
class DocaTxQ final {
public:
    /**
     * Create and initialize a DOCA TX queue
     *
     * @param[in] config Configuration parameters for the TX queue
     * @param[in] gpu_dev DOCA GPU device
     * @param[in] ddev DOCA device
     * @throws std::runtime_error if queue creation fails
     */
    NET_EXPORT DocaTxQ(const DocaTxQConfig &config, doca_gpu *gpu_dev, doca_dev *ddev);

    /**
     * Get access to the internal DocaTxQParams structure for CUDA kernel usage
     *
     * @return Pointer to the internal DocaTxQParams structure
     * @note This method is intended for advanced usage with CUDA kernels
     */
    [[nodiscard]] NET_EXPORT const DocaTxQParams *params() const noexcept;

private:
    /**
     * Custom deleter for DocaTxQParams that calls doca_destroy_txq
     */
    struct TxqDeleter {
        void operator()(DocaTxQParams *txq) const noexcept;
    };

    std::unique_ptr<DocaTxQParams, TxqDeleter> txq_; //!< Internal TX queue structure
};

} // namespace framework::net

#endif // FRAMEWORK_NET_DOCA_TXQ_HPP
