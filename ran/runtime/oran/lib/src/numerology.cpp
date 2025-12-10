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
#include <optional>
#include <stdexcept>

#include "fapi/fapi_file_replay.hpp"
#include "oran/cplane_types.hpp"
#include "oran/numerology.hpp"

namespace ran::oran {

OranNumerology from_scs(const SubcarrierSpacing scs) {
    // Base subcarrier spacing in kHz (15 kHz for LTE/NR)
    static constexpr std::uint32_t BASE_SCS_KHZ = 15;

    OranNumerology numerology{};

    numerology.subcarrier_spacing = scs;

    // Calculate slots per subframe (3GPP TS 38.211)
    // slots_per_subframe = SCS/15 (equivalent to 2^μ where μ = log2(SCS/15))
    const std::uint32_t scs_khz = to_khz(scs);
    numerology.slots_per_subframe = scs_khz / BASE_SCS_KHZ;

    // Calculate slot period in nanoseconds
    // slot_period = 1ms / slots_per_subframe
    numerology.slot_period_ns = SUBFRAME_NS / numerology.slots_per_subframe;

    // Calculate symbol duration (14 symbols per slot for normal CP)
    static constexpr std::uint32_t SYMBOLS_PER_SLOT = 14;
    numerology.symbol_duration_ns = numerology.slot_period_ns / SYMBOLS_PER_SLOT;

    return numerology;
}

OranNumerology from_scs_khz(const std::uint32_t scs_khz) {
    const auto scs_opt = from_khz(scs_khz);
    if (!scs_opt) {
        throw std::invalid_argument(std::format(
                "Invalid subcarrier spacing: {} kHz. Must be 15, 30, 60, or 120 kHz", scs_khz));
    }
    return from_scs(*scs_opt);
}

OranSlotTiming OranNumerology::calculate_slot_timing(const std::uint64_t absolute_slot) const {
    // Frame ID wrap value (8-bit field)
    static constexpr std::uint32_t FRAME_ID_WRAP = 256;

    // ORAN frame structure:
    // - 1 frame = 10 subframes = 10ms
    // - Frame ID wraps at 256 (8-bit field)
    const auto slots_per_frame = static_cast<std::uint64_t>(slots_per_subframe) *
                                 static_cast<std::uint64_t>(SUBFRAMES_PER_FRAME);

    OranSlotTiming timing{};

    // Calculate frame, subframe, and slot
    const std::uint64_t frames = absolute_slot / slots_per_frame;
    const std::uint64_t slots_in_frame = absolute_slot % slots_per_frame;

    timing.frame_id = static_cast<std::uint8_t>(frames % FRAME_ID_WRAP);
    timing.subframe_id = static_cast<std::uint8_t>(slots_in_frame / slots_per_subframe);
    timing.slot_id = static_cast<std::uint8_t>(slots_in_frame % slots_per_subframe);

    return timing;
}

std::uint64_t OranNumerology::calculate_slot_timestamp(const OranSlotTiming &timing) const {
    // Calculate absolute slot number
    const std::uint64_t slots_from_frames =
            static_cast<std::uint64_t>(timing.frame_id) * SUBFRAMES_PER_FRAME * slots_per_subframe;
    const std::uint64_t slots_from_subframes =
            static_cast<std::uint64_t>(timing.subframe_id) * slots_per_subframe;
    const std::uint64_t total_slots = slots_from_frames + slots_from_subframes + timing.slot_id;

    // Convert to timestamp
    return total_slots * slot_period_ns;
}

OranSlotTiming fapi_to_oran_slot_timing(const ran::fapi::FapiSlotTiming &fapi_timing) noexcept {
    // FAPI frame structure constants
    static constexpr std::uint16_t FAPI_SFN_MAX = 1024; // FAPI SFN wraps at 1024

    // Calculate slots per frame
    const std::uint64_t slots_per_frame =
            static_cast<std::uint64_t>(SUBFRAMES_PER_FRAME) * fapi_timing.slots_per_subframe;

    // Calculate frame, subframe, and slot from absolute slot
    const std::uint64_t frames = fapi_timing.absolute_slot / slots_per_frame;
    const std::uint64_t slots_in_frame = fapi_timing.absolute_slot % slots_per_frame;

    // OranSlotTiming uses uint8_t for frame_id which wraps at 256
    // FAPI SFN wraps at 1024, but we need to handle the narrowing to uint8_t
    const auto frame_id = static_cast<std::uint8_t>((frames % FAPI_SFN_MAX));
    const auto subframe_id =
            static_cast<std::uint8_t>(slots_in_frame / fapi_timing.slots_per_subframe);
    const auto slot_id = static_cast<std::uint8_t>(slots_in_frame % fapi_timing.slots_per_subframe);

    return OranSlotTiming{.frame_id = frame_id, .subframe_id = subframe_id, .slot_id = slot_id};
}

} // namespace ran::oran
