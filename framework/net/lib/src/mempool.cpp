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
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>

#include <quill/LogMacros.h>
#include <rte_mempool.h>

#include "log/rt_log_macros.hpp"
#include "net/details/dpdk_utils.hpp"
#include "net/dpdk_types.hpp"
#include "net/mempool.hpp"
#include "net/net_log.hpp"

namespace framework::net {

void Mempool::MempoolDeleter::operator()(rte_mempool *mempool) const noexcept {
    if (mempool != nullptr) {
        const std::error_code result = dpdk_destroy_mempool(mempool);
        if (result != DpdkErrc::Success) {
            RT_LOGC_ERROR(
                    Net::NetDpdk,
                    "Failed to destroy mempool '{}': {}",
                    mempool->name,
                    get_error_name(result));
        }
    }
}

Mempool::Mempool(const std::uint16_t port_id, const MempoolConfig &config) {
    if (config.name.empty()) {
        log_and_throw<std::invalid_argument>(Net::NetDpdk, "Mempool name cannot be empty");
    }

    if (config.num_mbufs == 0) {
        log_and_throw<std::invalid_argument>(
                Net::NetDpdk, "Number of mbufs must be greater than zero");
    }

    if (config.mtu_size == 0) {
        log_and_throw<std::invalid_argument>(Net::NetDpdk, "MTU size must be greater than zero");
    }

    // Create the raw mempool
    rte_mempool *raw_mempool{};
    const std::error_code result = dpdk_create_mempool(
            config.name,
            port_id,
            config.num_mbufs,
            config.mtu_size,
            config.host_pinned,
            &raw_mempool);

    if (result != DpdkErrc::Success) {
        log_and_throw(
                Net::NetDpdk,
                "Failed to create DPDK mempool '{}': {}",
                config.name,
                get_error_name(result));
    }

    // Transfer ownership to unique_ptr with custom deleter
    mempool_ = std::unique_ptr<rte_mempool, MempoolDeleter>(raw_mempool);
}

rte_mempool *Mempool::dpdk_mempool() const noexcept { return mempool_.get(); }

} // namespace framework::net
