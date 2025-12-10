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

#ifndef RAN_ORAN_CPLANE_UTILS_HPP
#define RAN_ORAN_CPLANE_UTILS_HPP

#include <cstdint>
#include <limits>
#include <span>

#include "oran/cplane_types.hpp"
#include "oran/oran_export.hpp"

namespace ran::oran {

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

/**
 * Convert CPU byte order to big endian (network byte order)
 * @param[in] value Value in CPU byte order
 * @return Value in big endian byte order
 */
[[nodiscard]] inline std::uint16_t cpu_to_be_16(const std::uint16_t value) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    const auto shifted_left = static_cast<std::uint16_t>(value << 8U);
    const auto shifted_right = static_cast<std::uint16_t>(value >> 8U);
    return static_cast<std::uint16_t>(shifted_left | shifted_right);
#else
    return value;
#endif
}

/**
 * Convert CPU byte order to big endian (network byte order) for 64-bit values
 * @param[in] value Value in CPU byte order
 * @return Value in big endian byte order
 */
[[nodiscard]] inline std::uint64_t cpu_to_be_64(std::uint64_t value) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    static constexpr auto BYTE_MASK =
            static_cast<std::uint64_t>(std::numeric_limits<std::uint8_t>::max());
    return ((value & (BYTE_MASK << 0U)) << 56U) | ((value & (BYTE_MASK << 8U)) << 40U) |
           ((value & (BYTE_MASK << 16U)) << 24U) | ((value & (BYTE_MASK << 24U)) << 8U) |
           ((value & (BYTE_MASK << 32U)) >> 8U) | ((value & (BYTE_MASK << 40U)) >> 24U) |
           ((value & (BYTE_MASK << 48U)) >> 40U) | ((value & (BYTE_MASK << 56U)) >> 56U);
#else
    return value;
#endif
}

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

/**
 * Get C-plane message common header size for section type
 * @param[in] section_type Section type (0, 1, 3, 5)
 * @return Size of common header in bytes
 * @throws std::invalid_argument if section_type is out of bounds or unsupported
 */
[[nodiscard]] ORAN_EXPORT std::uint16_t get_cmsg_common_hdr_size(std::uint8_t section_type);

/**
 * Get C-plane message section size for section type
 * @param[in] section_type Section type (0, 1, 3, 5)
 * @return Size of section in bytes
 * @throws std::invalid_argument if section_type is out of bounds or unsupported
 */
[[nodiscard]] ORAN_EXPORT std::uint16_t get_cmsg_section_size(std::uint8_t section_type);

} // namespace ran::oran

#endif // RAN_ORAN_CPLANE_UTILS_HPP
