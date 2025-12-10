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

#ifndef RAN_ORAN_CPLANE_MESSAGE_HPP
#define RAN_ORAN_CPLANE_MESSAGE_HPP

#include <cstdint>
#include <span>

#include "oran/cplane_types.hpp"
#include "oran/cplane_utils.hpp"
#include "oran/oran_export.hpp"

namespace ran::oran {

/**
 * Prepare C-plane message packets
 *
 * Creates C-plane packets from message information, handling fragmentation
 * across multiple packets when sections don't fit in MTU.
 *
 * @tparam BufferType Type derived from OranBuf
 * @tparam Extent Span extent (static or dynamic)
 * @param[in,out] info Message information containing sections and headers
 * @param[in,out] flow Flow interface for packet templates and sequence IDs
 * @param[in,out] peer Peer interface for timestamp tracking
 * @param[in,out] buffers Span of buffer objects to fill
 * @param[in] mtu Maximum transmission unit size in bytes
 * @return Number of packets created
 */
template <std::derived_from<OranBuf> BufferType, std::size_t EXTENT = std::dynamic_extent>
std::uint16_t prepare_cplane_message(
        OranCPlaneMsgInfo &info,
        OranFlow &flow,
        OranPeer &peer,
        std::span<BufferType, EXTENT> buffers,
        std::uint16_t mtu);

/**
 * Count required C-plane packets
 *
 * Calculates how many packets will be needed to send the given C-plane
 * messages, accounting for MTU limitations and section fragmentation.
 *
 * @param[in,out] infos Span of message information structures
 * @param[in] mtu Maximum transmission unit size in bytes
 * @return Total number of packets needed
 */
ORAN_EXPORT [[nodiscard]] std::size_t
count_cplane_packets(std::span<OranCPlaneMsgInfo> infos, std::uint16_t mtu);

} // namespace ran::oran

#endif // RAN_ORAN_CPLANE_MESSAGE_HPP
