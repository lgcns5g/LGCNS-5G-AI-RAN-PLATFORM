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
#include "net/doca_rxq.hpp"
#include "net/doca_types.hpp"
#include "net/dpdk_types.hpp"
#include "net/net_log.hpp"

namespace framework::net {

void DocaRxQ::RxqDeleter::operator()(DocaRxQParams *rxq) const noexcept {
    if (rxq != nullptr) {
        const doca_error_t result = doca_destroy_rxq(rxq);
        if (result != DOCA_SUCCESS) {
            RT_LOGC_ERROR(
                    Net::NetDoca, "Failed to destroy RX queue: {}", doca_error_get_descr(result));
        }
        // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
        delete rxq;
    }
}

DocaRxQ::DocaRxQ(const DocaRxQConfig &config, doca_gpu *gpu_dev, doca_dev *ddev) {
    if (gpu_dev == nullptr || ddev == nullptr) {
        log_and_throw<std::invalid_argument>(
                Net::NetDoca, "GPU device and DOCA device cannot be null");
    }

    if (config.nic_pcie_addr.empty() || config.sender_mac_addr.is_zero()) {
        log_and_throw<std::invalid_argument>(
                Net::NetDoca, "NIC PCIe address and sender MAC address cannot be empty");
    }

    if (config.max_pkt_num == 0 || config.max_pkt_size == 0) {
        log_and_throw<std::invalid_argument>(
                Net::NetDoca, "Maximum packet number and size must be greater than zero");
    }

    if (config.ether_type == 0) {
        log_and_throw<std::invalid_argument>(
                Net::NetDoca, "EtherType must be specified for flow rule");
    }

    // Create the raw DocaRxQParams object
    auto raw_rxq = std::unique_ptr<DocaRxQParams, RxqDeleter>(new DocaRxQParams{});

    // Create the RX queue
    doca_error_t result = doca_create_rxq(
            raw_rxq.get(),
            gpu_dev,
            ddev,
            config.max_pkt_num,
            config.max_pkt_size,
            config.sem_items);
    if (result != DOCA_SUCCESS) {
        log_and_throw(
                Net::NetDoca, "Failed to create DOCA RX queue: {}", doca_error_get_descr(result));
    }

    // Create the flow rule
    result = doca_create_flow_rule(
            raw_rxq.get(),
            config.nic_pcie_addr,
            config.sender_mac_addr,
            config.ether_type,
            config.vlan_tci);
    if (result != DOCA_SUCCESS) {
        // RX queue cleanup will be handled by the unique_ptr destructor
        log_and_throw(Net::NetDoca, "Failed to create flow rule: {}", doca_error_get_descr(result));
    }

    // Transfer ownership to member variable
    rxq_ = std::move(raw_rxq);
}

const DocaRxQParams *DocaRxQ::params() const noexcept { return rxq_.get(); }

} // namespace framework::net
