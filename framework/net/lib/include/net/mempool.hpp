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

#ifndef FRAMEWORK_NET_MEMPOOL_HPP
#define FRAMEWORK_NET_MEMPOOL_HPP

#include <cstdint>
#include <memory>
#include <string>

#include <rte_mempool.h>

#include "net/dpdk_types.hpp"
#include "net/net_export.hpp"

namespace framework::net {

/**
 * Configuration structure for Mempool creation
 */
struct NET_EXPORT MempoolConfig final {
    std::string name;          //!< Unique name for the mempool
    std::uint32_t num_mbufs{}; //!< Number of mbufs in the mempool
    std::uint32_t mtu_size{};  //!< MTU size for buffer calculations
    HostPinned host_pinned{};  //!< Whether to use host-pinned memory
};

/**
 * RAII wrapper for DPDK mempool management
 *
 * Provides automatic resource management for DPDK mempools,
 * ensuring proper cleanup on destruction. Supports both regular
 * and host-pinned memory configurations.
 */
class Mempool final {
public:
    /**
     * Create and initialize a DPDK mempool
     *
     * @param[in] port_id DPDK port ID to determine NUMA socket
     * @param[in] config Configuration parameters for the mempool
     * @throws std::invalid_argument if configuration parameters are invalid
     * @throws std::runtime_error if mempool creation fails
     */
    NET_EXPORT Mempool(std::uint16_t port_id, const MempoolConfig &config);

    /**
     * Get access to the underlying DPDK mempool structure
     *
     * @return Pointer to the internal rte_mempool structure
     * @note This method is intended for advanced usage with DPDK operations
     */
    [[nodiscard]] NET_EXPORT rte_mempool *dpdk_mempool() const noexcept;

private:
    /**
     * Custom deleter for rte_mempool that calls dpdk_destroy_mempool
     */
    struct MempoolDeleter {
        void operator()(rte_mempool *mempool) const noexcept;
    };

    std::unique_ptr<rte_mempool, MempoolDeleter> mempool_; //!< Internal mempool structure
};

} // namespace framework::net

#endif // FRAMEWORK_NET_MEMPOOL_HPP
