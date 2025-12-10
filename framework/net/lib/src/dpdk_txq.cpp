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

#include <cstdint>
#include <format>
#include <span>
#include <stdexcept>
#include <system_error>

#include <quill/LogMacros.h>

#include "log/rt_log_macros.hpp"
#include "net/details/dpdk_utils.hpp"
#include "net/dpdk_txq.hpp"
#include "net/dpdk_types.hpp"
#include "net/net_log.hpp"

namespace framework::net {

DpdkTxQueue::DpdkTxQueue(
        const std::uint16_t port_id, const std::uint16_t txq_id, const DpdkTxQConfig &config)
        : port_id_{port_id}, queue_id_{txq_id} {

    if (config.txq_size == 0) {
        log_and_throw<std::invalid_argument>(
                Net::NetDpdk, "TX queue size must be greater than zero");
    }

    if (const auto result = dpdk_setup_tx_queue(port_id, txq_id, config.txq_size); result) {
        log_and_throw(
                Net::NetDpdk,
                "Failed to setup DPDK TX queue {} on port {}: {}",
                txq_id,
                port_id,
                get_error_name(result));
    }

    RT_LOGC_DEBUG(
            Net::NetDpdk,
            "Successfully created DpdkTxQueue: port={}, queue={}, size={}",
            port_id,
            txq_id,
            config.txq_size);
}

std::error_code DpdkTxQueue::send(
        const std::span<const std::span<const std::uint8_t>> messages,
        const EthernetHeader &eth_header,
        rte_mempool *mempool,
        const std::uint32_t max_retry_count) const {
    return dpdk_eth_send(messages, eth_header, mempool, queue_id_, port_id_, max_retry_count);
}

std::error_code DpdkTxQueue::send_mbufs(
        const std::span<rte_mbuf *> mbufs, const std::uint32_t max_retry_count) const {
    return dpdk_eth_send_mbufs(mbufs, queue_id_, port_id_, max_retry_count);
}

} // namespace framework::net
