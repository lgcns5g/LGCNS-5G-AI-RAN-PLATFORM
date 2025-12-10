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

#ifndef RAN_ORAN_NUMEROLOGY_HPP
#define RAN_ORAN_NUMEROLOGY_HPP

#include <cstdint>
#include <optional>

#include <wise_enum_detail.h>
#include <wise_enum_generated.h>

#include <wise_enum.h>

#include "oran/cplane_types.hpp"
#include "oran/oran_export.hpp"

namespace ran::fapi {
struct FapiSlotTiming;
} // namespace ran::fapi

namespace ran::oran {

/**
 * Subcarrier spacing values (3GPP TS 38.211)
 */
enum class SubcarrierSpacing : std::uint32_t {
    Scs15Khz = 15,  //!< 15 kHz subcarrier spacing
    Scs30Khz = 30,  //!< 30 kHz subcarrier spacing
    Scs60Khz = 60,  //!< 60 kHz subcarrier spacing
    Scs120Khz = 120 //!< 120 kHz subcarrier spacing
};

} // namespace ran::oran

// WISE_ENUM_ADAPT must be called at global namespace scope for ADL to work
WISE_ENUM_ADAPT(ran::oran::SubcarrierSpacing, Scs15Khz, Scs30Khz, Scs60Khz, Scs120Khz)

namespace ran::oran {

/**
 * Convert SubcarrierSpacing enum to kHz value
 * @param[in] scs Subcarrier spacing enum
 * @return SCS value in kHz
 */
[[nodiscard]] constexpr std::uint32_t to_khz(const SubcarrierSpacing scs) noexcept {
    return static_cast<std::uint32_t>(scs);
}

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
/**
 * Convert kHz value to SubcarrierSpacing enum
 * @param[in] khz SCS value in kHz (15, 30, 60, or 120)
 * @return SubcarrierSpacing enum or nullopt if invalid
 */
[[nodiscard]] constexpr std::optional<SubcarrierSpacing>
from_khz(const std::uint32_t khz) noexcept {
    switch (khz) {
    case 15:
        return SubcarrierSpacing::Scs15Khz;
    case 30:
        return SubcarrierSpacing::Scs30Khz;
    case 60:
        return SubcarrierSpacing::Scs60Khz;
    case 120:
        return SubcarrierSpacing::Scs120Khz;
    default:
        return std::nullopt;
    }
}
// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

// Timing constants (3GPP TS 38.211)
inline constexpr std::uint32_t SUBFRAMES_PER_FRAME = 10; //!< Subframes per frame
inline constexpr std::uint64_t SUBFRAME_NS = 1'000'000;  //!< Subframe period in ns (1ms)
inline constexpr std::uint64_t FRAME_PERIOD_NS =
        SUBFRAME_NS * SUBFRAMES_PER_FRAME;       //!< Frame period in ns (10ms)
inline constexpr std::uint64_t SFN_MAX = 1024LL; //!< System Frame Number maximum value
inline constexpr std::uint64_t SFN_PERIOD_NS =
        SFN_MAX * FRAME_PERIOD_NS; //!< SFN rollover period in ns

/**
 * ORAN numerology parameters derived from subcarrier spacing
 *
 * All timing parameters calculated from SCS according to 3GPP TS 38.211.
 * This struct encapsulates the relationship between subcarrier spacing
 * and derived timing values (slot period, symbol duration, etc.).
 */
struct ORAN_EXPORT OranNumerology final {
    SubcarrierSpacing subcarrier_spacing{}; //!< SCS enum value
    std::uint32_t slots_per_subframe{};     //!< Slots per 1ms subframe
    std::uint64_t slot_period_ns{};         //!< Slot period in nanoseconds
    std::uint64_t symbol_duration_ns{};     //!< OFDM symbol duration in ns

    /**
     * Calculate slot timing from absolute slot number
     *
     * Converts an absolute slot number (since SFN 0 slot 0) into
     * frame, subframe, and slot indices.
     *
     * @param[in] absolute_slot Absolute slot number since SFN 0 slot 0
     * @return Slot timing information (frame, subframe, slot)
     */
    [[nodiscard]] OranSlotTiming calculate_slot_timing(std::uint64_t absolute_slot) const;

    /**
     * Calculate timestamp for a given slot
     *
     * Converts frame, subframe, and slot indices into an absolute timestamp
     * (relative to SFN 0 slot 0).
     *
     * @param[in] timing Slot timing (frame, subframe, slot)
     * @return Timestamp in nanoseconds since SFN 0 slot 0
     */
    [[nodiscard]] std::uint64_t calculate_slot_timestamp(const OranSlotTiming &timing) const;
};

/**
 * Create numerology from SubcarrierSpacing enum
 *
 * Calculates all timing parameters based on 3GPP specifications:
 * - Slots per subframe = SCS / 15 kHz
 * - Slot period = 1ms / slots_per_subframe
 * - Symbol duration = slot_period / 14 symbols
 *
 * @param[in] scs Subcarrier spacing enum
 * @return Calculated numerology parameters
 */
ORAN_EXPORT [[nodiscard]] OranNumerology from_scs(SubcarrierSpacing scs);

/**
 * Create numerology from subcarrier spacing in kHz
 *
 * @param[in] scs_khz Subcarrier spacing in kHz (15, 30, 60, or 120)
 * @return Calculated numerology parameters
 * @throws std::invalid_argument if SCS is not valid
 */
ORAN_EXPORT [[nodiscard]] OranNumerology from_scs_khz(std::uint32_t scs_khz);

/**
 * Convert FAPI slot timing to ORAN slot timing
 *
 * Performs conversion from absolute slot number and slots-per-subframe
 * to frame, subframe, and slot IDs. Handles uint8_t wrapping at frame 256
 * and FAPI SFN wrapping at 1024.
 *
 * @param[in] fapi_timing FAPI slot timing containing absolute_slot and slots_per_subframe
 * @return ORAN slot timing with frame, subframe, and slot IDs
 */
ORAN_EXPORT [[nodiscard]] OranSlotTiming
fapi_to_oran_slot_timing(const ran::fapi::FapiSlotTiming &fapi_timing) noexcept;

} // namespace ran::oran

#endif // RAN_ORAN_NUMEROLOGY_HPP
