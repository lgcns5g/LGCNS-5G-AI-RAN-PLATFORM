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

#ifndef RAN_FRONTHAUL_UPLANE_CONFIG_HPP
#define RAN_FRONTHAUL_UPLANE_CONFIG_HPP

/**
 * @file uplane_config.hpp
 * @brief U-Plane configuration for Order Kernel pipeline
 */

#include <cstdint>
#include <vector>

#include "fronthaul/fronthaul_export.hpp"
#include "fronthaul/order_kernel_descriptors.hpp"
#include "net/dpdk_types.hpp"
#include "net/env.hpp"

namespace ran::fronthaul {

/**
 * U-Plane configuration parameters
 *
 * Configuration for ORAN U-Plane packet reception using Order Kernel pipeline.
 * Default values are suitable for 30kHz SCS (500us slot duration) in production environments.
 */
struct FRONTHAUL_EXPORT UPlaneConfig final {
    // ========================================================================
    // Timing Parameters (Ta4 windows for ORAN packet timing)
    // ========================================================================
    // NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

    //! Ta4 early window (50us before slot start)
    std::uint64_t ta4_min_ns{50'000};

    //! Ta4 late window (450us after slot start)
    std::uint64_t ta4_max_ns{450'000};

    //! Slot duration (500us for 30kHz SCS)
    std::uint64_t slot_duration_ns{500'000};

    // ========================================================================
    // Timeout Configuration for Order Kernel
    // ========================================================================

    //! Timeout with no packets (default: 6s)
    std::uint64_t timeout_no_pkt_ns{DEFAULT_TIMEOUT_NO_PKT_NS};

    //! Timeout for first packet (default: 1500us)
    std::uint64_t timeout_first_pkt_ns{DEFAULT_TIMEOUT_FIRST_PKT_NS};

    //! Timeout log interval (default: 1s)
    std::uint64_t timeout_log_interval_ns{DEFAULT_TIMEOUT_LOG_INTERVAL_NS};

    //! Enable timeout logging
    bool timeout_log_enable{true};

    // ========================================================================
    // RX Configuration for DOCA GPUNetIO
    // ========================================================================

    //! Maximum RX packets to process per iteration (default: 512)
    std::uint32_t max_rx_pkts{DEFAULT_MAX_RX_PKTS};

    //! RX packet timeout (default: 100us)
    std::uint64_t rx_pkts_timeout_ns{DEFAULT_RX_PKTS_TIMEOUT_NS};

    // ========================================================================
    // DOCA RX Queue Configuration (production-ready defaults)
    // ========================================================================

    //! Number of packet buffers in RX queue (16K)
    std::uint32_t num_packets{16384};

    //! Maximum packet size (8KB, observed 1494 bytes in production)
    std::uint32_t max_packet_size{8192};

    //! GPU semaphore items (4096, must be power of 2)
    std::uint32_t gpu_semaphore_items{4096};

    // NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

    // ========================================================================
    // eAxC Configuration
    // ========================================================================

    //! UL eAxC IDs for antenna ports (default: [0, 1, 2, 3])
    std::vector<std::uint16_t> eaxc_ids{0, 1, 2, 3};
};

} // namespace ran::fronthaul

#endif // RAN_FRONTHAUL_UPLANE_CONFIG_HPP
