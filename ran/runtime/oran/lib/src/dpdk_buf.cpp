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

#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <system_error>

#include <quill/LogMacros.h>
#include <rte_mbuf.h>
#include <rte_mbuf_core.h>

#include "log/rt_log_macros.hpp"
#include "net/details/dpdk_utils.hpp"
#include "net/dpdk_types.hpp"
#include "oran/dpdk_buf.hpp"
#include "oran/oran_log.hpp"

namespace ran::oran {

void MBuf::initialize_timestamp_offsets() {
    std::int32_t offset{};
    std::uint64_t mask{};

    const auto result = framework::net::dpdk_calculate_timestamp_offsets(offset, mask);

    if (result) {
        RT_LOGC_ERROR(
                Oran::OranBuffer,
                "Failed to calculate DPDK timestamp offsets: {}",
                framework::net::get_error_name(result));
        throw std::runtime_error("Failed to initialize DPDK timestamp offsets");
    }

    timestamp_offset = offset;
    timestamp_mask = mask;

    RT_LOGC_INFO(
            Oran::OranBuffer,
            "DPDK timestamp offsets initialized: offset={}, mask={:#x}",
            offset,
            mask);
}

MBuf::MBuf(rte_mbuf *const mbuf) : mbuf_(mbuf) {
    // Thread-safe one-time initialization of timestamp offsets
    static std::once_flag init_flag;
    std::call_once(init_flag, &MBuf::initialize_timestamp_offsets);
}

std::uint8_t *MBuf::data() {
    // rte_pktmbuf_mtod is a DPDK macro that uses C-style casts for compatibility
    // across different DPDK versions and platforms
    return rte_pktmbuf_mtod(mbuf_, std::uint8_t *); // cppcheck-suppress cstyleCast
}

const std::uint8_t *MBuf::data() const {
    // rte_pktmbuf_mtod is a DPDK macro that uses C-style casts for compatibility
    // across different DPDK versions and platforms
    return rte_pktmbuf_mtod(mbuf_, const std::uint8_t *); // cppcheck-suppress cstyleCast
}

std::size_t MBuf::capacity() const { return rte_pktmbuf_tailroom(mbuf_) + mbuf_->data_len; }

std::size_t MBuf::size() const { return mbuf_->data_len; }

void MBuf::set_size(const std::size_t new_size) {
    const auto cap = capacity();
    if (new_size > cap) {
        throw std::length_error(std::format(
                "MBuf::set_size: requested size {} exceeds capacity {}", new_size, cap));
    }
    static constexpr auto MAX_MBUF_SIZE = std::numeric_limits<std::uint16_t>::max();
    if (new_size > MAX_MBUF_SIZE) {
        throw std::length_error(std::format(
                "MBuf::set_size: requested size {} exceeds maximum mbuf data length {}",
                new_size,
                MAX_MBUF_SIZE));
    }
    mbuf_->data_len = static_cast<std::uint16_t>(new_size);
    mbuf_->pkt_len = static_cast<std::uint32_t>(new_size);
}

void MBuf::set_timestamp(const std::uint64_t timestamp) {
    // If timestamp_offset is empty, timestamps are disabled
    if (!timestamp_offset.has_value()) {
        return;
    }

    // Automatically set the timestamp flag when setting a timestamp
    if (timestamp_mask != 0) {
        mbuf_->ol_flags |= timestamp_mask;
    }

    // Manually expand macro since C-style casts don't work with strict C++20
    // warnings RTE_MBUF_DYNFIELD(m, offset, type) expands to:
    // ((type)((uintptr_t)(m) + (offset)))
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast, performance-no-int-to-ptr)
    auto *timestamp_ptr = reinterpret_cast<std::uint64_t *>(
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
            reinterpret_cast<std::uintptr_t>(mbuf_) + static_cast<std::size_t>(*timestamp_offset));
    *timestamp_ptr = timestamp;
}

void MBuf::clear_flags() { mbuf_->ol_flags = 0; }

} // namespace ran::oran
