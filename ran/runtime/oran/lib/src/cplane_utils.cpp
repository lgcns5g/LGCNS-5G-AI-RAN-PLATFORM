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

#include <array>
#include <cstdint>
#include <format>
#include <stdexcept>

#include <aerial-fh-driver/oran.hpp>

#include "oran/cplane_utils.hpp"

namespace ran::oran {

std::uint16_t get_cmsg_common_hdr_size(const std::uint8_t section_type) {
    static constexpr std::array<std::uint16_t, 6> COMMON_HDR_SIZES = {
            sizeof(oran_cmsg_sect0_common_hdr), // Section Type 0
            sizeof(oran_cmsg_sect1_common_hdr), // Section Type 1
            std::uint16_t{0},                   // Section Type 2 (not supported)
            sizeof(oran_cmsg_sect3_common_hdr), // Section Type 3
            std::uint16_t{0},                   // Section Type 4 (not supported)
            sizeof(oran_cmsg_sect5_common_hdr), // Section Type 5
    };

    if (section_type >= COMMON_HDR_SIZES.size()) {
        throw std::invalid_argument(std::format(
                "Section type {} is out of bounds (max {})",
                section_type,
                COMMON_HDR_SIZES.size() - 1));
    }

    const auto size = COMMON_HDR_SIZES.at(section_type);
    if (size == 0) {
        throw std::invalid_argument(std::format("Section type {} is not supported", section_type));
    }

    return size;
}

std::uint16_t get_cmsg_section_size(const std::uint8_t section_type) {
    static constexpr std::array<std::uint16_t, 6> SECTION_SIZES = {
            sizeof(oran_cmsg_sect0), // Section Type 0
            sizeof(oran_cmsg_sect1), // Section Type 1
            std::uint16_t{0},        // Section Type 2 (not supported)
            sizeof(oran_cmsg_sect3), // Section Type 3
            std::uint16_t{0},        // Section Type 4 (not supported)
            sizeof(oran_cmsg_sect5), // Section Type 5
    };

    if (section_type >= SECTION_SIZES.size()) {
        throw std::invalid_argument(std::format(
                "Section type {} is out of bounds (max {})",
                section_type,
                SECTION_SIZES.size() - 1));
    }

    const auto size = SECTION_SIZES.at(section_type);
    if (size == 0) {
        throw std::invalid_argument(std::format("Section type {} is not supported", section_type));
    }

    return size;
}

} // namespace ran::oran
