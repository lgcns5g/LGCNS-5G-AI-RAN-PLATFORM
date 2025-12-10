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

#include <format>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <doca_ctx.h>
#include <doca_dev.h>
#include <doca_error.h>
#include <quill/LogMacros.h>

#include "log/rt_log_macros.hpp"
#include "net/details/doca_utils.hpp"
#include "net/doca_txq.hpp"
#include "net/doca_types.hpp"
#include "net/dpdk_types.hpp"
#include "net/net_log.hpp"

namespace framework::net {

void DocaTxQ::TxqDeleter::operator()(DocaTxQParams *txq) const noexcept {
    if (txq != nullptr) {
        const doca_error_t result = doca_destroy_txq(txq);
        if (result != DOCA_SUCCESS) {
            RT_LOGC_ERROR(
                    Net::NetDoca, "Failed to destroy TX queue: {}", doca_error_get_descr(result));
        }
        // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
        delete txq;
    }
}

DocaTxQ::DocaTxQ(const DocaTxQConfig &config, doca_gpu *gpu_dev, doca_dev *ddev) {
    if (gpu_dev == nullptr || ddev == nullptr) {
        log_and_throw<std::invalid_argument>(
                Net::NetDoca, "GPU device and DOCA device cannot be null");
    }

    if (config.nic_pcie_addr.empty() || config.dest_mac_addr.is_zero()) {
        log_and_throw<std::invalid_argument>(
                Net::NetDoca, "NIC PCIe address and destination MAC address cannot be empty");
    }

    if (config.pkt_size == 0 || config.pkt_num == 0 || config.max_sq_descr_num == 0) {
        log_and_throw<std::invalid_argument>(
                Net::NetDoca,
                "Packet size, packet number, and max descriptors must be greater than zero");
    }

    if (config.ether_type == 0) {
        log_and_throw<std::invalid_argument>(
                Net::NetDoca, "EtherType must be specified for flow rule");
    }

    // Create the raw DocaTxQParams object
    auto raw_txq = std::unique_ptr<DocaTxQParams, TxqDeleter>(new DocaTxQParams{});

    const doca_error_t result = doca_create_txq(
            raw_txq.get(),
            gpu_dev,
            ddev,
            config.pkt_size,
            config.pkt_num,
            config.max_sq_descr_num,
            config.nic_pcie_addr,
            config.dest_mac_addr,
            config.ether_type,
            config.vlan_tci);

    if (result != DOCA_SUCCESS) {
        log_and_throw(
                Net::NetDoca, "Failed to create DOCA TX queue: {}", doca_error_get_descr(result));
    }

    // Transfer ownership to member variable
    txq_ = std::move(raw_txq);
}

const DocaTxQParams *DocaTxQ::params() const noexcept { return txq_.get(); }

} // namespace framework::net
