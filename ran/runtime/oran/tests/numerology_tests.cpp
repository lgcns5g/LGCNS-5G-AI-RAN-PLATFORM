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
#include <optional>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <wise_enum_detail.h>

#include <gtest/gtest.h>
#include <wise_enum.h>

#include "fapi/fapi_file_replay.hpp"
#include "oran/cplane_types.hpp"
#include "oran/numerology.hpp"

namespace {

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

namespace ro = ran::oran;

// Test SubcarrierSpacing enum conversion to kHz
TEST(NumerologyTest, ToKhz_AllValidValues) {
    EXPECT_EQ(ro::to_khz(ro::SubcarrierSpacing::Scs15Khz), 15U);
    EXPECT_EQ(ro::to_khz(ro::SubcarrierSpacing::Scs30Khz), 30U);
    EXPECT_EQ(ro::to_khz(ro::SubcarrierSpacing::Scs60Khz), 60U);
    EXPECT_EQ(ro::to_khz(ro::SubcarrierSpacing::Scs120Khz), 120U);
}

// Test kHz to SubcarrierSpacing enum conversion - valid values
TEST(NumerologyTest, FromKhz_ValidValues) {
    const auto scs_15 = ro::from_khz(15);
    // NOLINTBEGIN(bugprone-unchecked-optional-access)
    ASSERT_TRUE(scs_15.has_value());
    EXPECT_EQ(*scs_15, ro::SubcarrierSpacing::Scs15Khz);

    const auto scs_30 = ro::from_khz(30);
    ASSERT_TRUE(scs_30.has_value());
    EXPECT_EQ(*scs_30, ro::SubcarrierSpacing::Scs30Khz);

    const auto scs_60 = ro::from_khz(60);
    ASSERT_TRUE(scs_60.has_value());
    EXPECT_EQ(*scs_60, ro::SubcarrierSpacing::Scs60Khz);

    const auto scs_120 = ro::from_khz(120);
    ASSERT_TRUE(scs_120.has_value());
    EXPECT_EQ(*scs_120, ro::SubcarrierSpacing::Scs120Khz);
    // NOLINTEND(bugprone-unchecked-optional-access)
}

// Test kHz to SubcarrierSpacing enum conversion - invalid values
TEST(NumerologyTest, FromKhz_InvalidValues) {
    EXPECT_FALSE(ro::from_khz(0).has_value());
    EXPECT_FALSE(ro::from_khz(10).has_value());
    EXPECT_FALSE(ro::from_khz(20).has_value());
    EXPECT_FALSE(ro::from_khz(45).has_value());
    EXPECT_FALSE(ro::from_khz(100).has_value());
    EXPECT_FALSE(ro::from_khz(240).has_value());
    EXPECT_FALSE(ro::from_khz(1000).has_value());
}

// Test round-trip conversion: SCS -> kHz -> SCS
TEST(NumerologyTest, RoundTripConversion) {
    for (const auto scs_pair : ::wise_enum::range<ro::SubcarrierSpacing>) {
        const auto scs = scs_pair.value;
        const std::uint32_t khz = ro::to_khz(scs);
        const auto scs_back = ro::from_khz(khz);
        ASSERT_TRUE(scs_back.has_value());
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        EXPECT_EQ(*scs_back, scs);
    }
}

// Test from_scs for 15 kHz
TEST(NumerologyTest, FromScs_15kHz) {
    const auto num = ro::from_scs(ro::SubcarrierSpacing::Scs15Khz);

    EXPECT_EQ(num.subcarrier_spacing, ro::SubcarrierSpacing::Scs15Khz);
    EXPECT_EQ(num.slots_per_subframe, 1U);
    EXPECT_EQ(num.slot_period_ns, 1'000'000U);  // 1ms / 1 slot
    EXPECT_EQ(num.symbol_duration_ns, 71'428U); // ~1ms / 14 symbols
}

// Test from_scs for 30 kHz
TEST(NumerologyTest, FromScs_30kHz) {
    const auto num = ro::from_scs(ro::SubcarrierSpacing::Scs30Khz);

    EXPECT_EQ(num.subcarrier_spacing, ro::SubcarrierSpacing::Scs30Khz);
    EXPECT_EQ(num.slots_per_subframe, 2U);
    EXPECT_EQ(num.slot_period_ns, 500'000U);    // 1ms / 2 slots
    EXPECT_EQ(num.symbol_duration_ns, 35'714U); // ~500us / 14 symbols
}

// Test from_scs for 60 kHz
TEST(NumerologyTest, FromScs_60kHz) {
    const auto num = ro::from_scs(ro::SubcarrierSpacing::Scs60Khz);

    EXPECT_EQ(num.subcarrier_spacing, ro::SubcarrierSpacing::Scs60Khz);
    EXPECT_EQ(num.slots_per_subframe, 4U);
    EXPECT_EQ(num.slot_period_ns, 250'000U);    // 1ms / 4 slots
    EXPECT_EQ(num.symbol_duration_ns, 17'857U); // ~250us / 14 symbols
}

// Test from_scs for 120 kHz
TEST(NumerologyTest, FromScs_120kHz) {
    const auto num = ro::from_scs(ro::SubcarrierSpacing::Scs120Khz);

    EXPECT_EQ(num.subcarrier_spacing, ro::SubcarrierSpacing::Scs120Khz);
    EXPECT_EQ(num.slots_per_subframe, 8U);
    EXPECT_EQ(num.slot_period_ns, 125'000U);   // 1ms / 8 slots
    EXPECT_EQ(num.symbol_duration_ns, 8'928U); // ~125us / 14 symbols
}

// Test from_scs_khz for valid values
TEST(NumerologyTest, FromScsKhz_ValidValues) {
    const auto num_15 = ro::from_scs_khz(15);
    EXPECT_EQ(num_15.subcarrier_spacing, ro::SubcarrierSpacing::Scs15Khz);
    EXPECT_EQ(num_15.slots_per_subframe, 1U);

    const auto num_30 = ro::from_scs_khz(30);
    EXPECT_EQ(num_30.subcarrier_spacing, ro::SubcarrierSpacing::Scs30Khz);
    EXPECT_EQ(num_30.slots_per_subframe, 2U);

    const auto num_60 = ro::from_scs_khz(60);
    EXPECT_EQ(num_60.subcarrier_spacing, ro::SubcarrierSpacing::Scs60Khz);
    EXPECT_EQ(num_60.slots_per_subframe, 4U);

    const auto num_120 = ro::from_scs_khz(120);
    EXPECT_EQ(num_120.subcarrier_spacing, ro::SubcarrierSpacing::Scs120Khz);
    EXPECT_EQ(num_120.slots_per_subframe, 8U);
}

// Test from_scs_khz for invalid values
TEST(NumerologyTest, FromScsKhz_InvalidValues) {
    EXPECT_THROW(std::ignore = ro::from_scs_khz(0), std::invalid_argument);
    EXPECT_THROW(std::ignore = ro::from_scs_khz(10), std::invalid_argument);
    EXPECT_THROW(std::ignore = ro::from_scs_khz(45), std::invalid_argument);
    EXPECT_THROW(std::ignore = ro::from_scs_khz(240), std::invalid_argument);
}

// Test calculate_slot_timing at slot boundaries (30 kHz)
TEST(NumerologyTest, CalculateSlotTiming_30kHz_SlotBoundaries) {
    const auto num = ro::from_scs(ro::SubcarrierSpacing::Scs30Khz);

    // Slot 0 (frame 0, subframe 0, slot 0)
    auto timing = num.calculate_slot_timing(0);
    EXPECT_EQ(timing.frame_id, 0U);
    EXPECT_EQ(timing.subframe_id, 0U);
    EXPECT_EQ(timing.slot_id, 0U);

    // Slot 1 (frame 0, subframe 0, slot 1)
    timing = num.calculate_slot_timing(1); // Absolute slot 1
    EXPECT_EQ(timing.frame_id, 0U);
    EXPECT_EQ(timing.subframe_id, 0U);
    EXPECT_EQ(timing.slot_id, 1U);

    // Slot 2 (frame 0, subframe 1, slot 0)
    timing = num.calculate_slot_timing(2); // Absolute slot 2
    EXPECT_EQ(timing.frame_id, 0U);
    EXPECT_EQ(timing.subframe_id, 1U);
    EXPECT_EQ(timing.slot_id, 0U);

    // Frame 1 (20 slots per frame at 30 kHz)
    timing = num.calculate_slot_timing(20); // Absolute slot 20
    EXPECT_EQ(timing.frame_id, 1U);
    EXPECT_EQ(timing.subframe_id, 0U);
    EXPECT_EQ(timing.slot_id, 0U);
}

// Test calculate_slot_timing with various slot numbers
TEST(NumerologyTest, CalculateSlotTiming_30kHz_VariousSlots) {
    const auto num = ro::from_scs(ro::SubcarrierSpacing::Scs30Khz);

    // Slot 0
    auto timing = num.calculate_slot_timing(0);
    EXPECT_EQ(timing.frame_id, 0U);
    EXPECT_EQ(timing.subframe_id, 0U);
    EXPECT_EQ(timing.slot_id, 0U);

    // Slot 3 (frame 0, subframe 1, slot 1)
    timing = num.calculate_slot_timing(3);
    EXPECT_EQ(timing.frame_id, 0U);
    EXPECT_EQ(timing.subframe_id, 1U);
    EXPECT_EQ(timing.slot_id, 1U);

    // Slot 19 (frame 0, subframe 9, slot 1)
    timing = num.calculate_slot_timing(19);
    EXPECT_EQ(timing.frame_id, 0U);
    EXPECT_EQ(timing.subframe_id, 9U);
    EXPECT_EQ(timing.slot_id, 1U);
}

// Test calculate_slot_timing for 15 kHz
TEST(NumerologyTest, CalculateSlotTiming_15kHz) {
    const auto num = ro::from_scs(ro::SubcarrierSpacing::Scs15Khz);

    // Slot 0
    auto timing = num.calculate_slot_timing(0);
    EXPECT_EQ(timing.frame_id, 0U);
    EXPECT_EQ(timing.subframe_id, 0U);
    EXPECT_EQ(timing.slot_id, 0U);

    // Subframe 1 (also slot 1 for 15 kHz since 1 slot per subframe)
    timing = num.calculate_slot_timing(1); // Absolute slot 1
    EXPECT_EQ(timing.frame_id, 0U);
    EXPECT_EQ(timing.subframe_id, 1U);
    EXPECT_EQ(timing.slot_id, 0U);

    // Frame 1 (10 slots per frame at 15 kHz)
    timing = num.calculate_slot_timing(10); // Absolute slot 10
    EXPECT_EQ(timing.frame_id, 1U);
    EXPECT_EQ(timing.subframe_id, 0U);
    EXPECT_EQ(timing.slot_id, 0U);
}

// Test calculate_slot_timing for 60 kHz
TEST(NumerologyTest, CalculateSlotTiming_60kHz) {
    const auto num = ro::from_scs(ro::SubcarrierSpacing::Scs60Khz);

    // Slot boundaries (4 slots per subframe)
    auto timing = num.calculate_slot_timing(0);
    EXPECT_EQ(timing.frame_id, 0U);
    EXPECT_EQ(timing.subframe_id, 0U);
    EXPECT_EQ(timing.slot_id, 0U);

    timing = num.calculate_slot_timing(1); // Absolute slot 1
    EXPECT_EQ(timing.frame_id, 0U);
    EXPECT_EQ(timing.subframe_id, 0U);
    EXPECT_EQ(timing.slot_id, 1U);

    timing = num.calculate_slot_timing(2); // Absolute slot 2
    EXPECT_EQ(timing.frame_id, 0U);
    EXPECT_EQ(timing.subframe_id, 0U);
    EXPECT_EQ(timing.slot_id, 2U);

    timing = num.calculate_slot_timing(3); // Absolute slot 3
    EXPECT_EQ(timing.frame_id, 0U);
    EXPECT_EQ(timing.subframe_id, 0U);
    EXPECT_EQ(timing.slot_id, 3U);

    timing = num.calculate_slot_timing(4); // Subframe 1, slot 0
    EXPECT_EQ(timing.frame_id, 0U);
    EXPECT_EQ(timing.subframe_id, 1U);
    EXPECT_EQ(timing.slot_id, 0U);
}

// Test calculate_slot_timestamp (30 kHz)
TEST(NumerologyTest, CalculateSlotTimestamp_30kHz) {
    const auto num = ro::from_scs(ro::SubcarrierSpacing::Scs30Khz);

    // Frame 0, subframe 0, slot 0
    ro::OranSlotTiming timing{0, 0, 0};
    EXPECT_EQ(num.calculate_slot_timestamp(timing), 0U);

    // Frame 0, subframe 0, slot 1
    timing = {0, 0, 1};
    EXPECT_EQ(num.calculate_slot_timestamp(timing), 500'000U);

    // Frame 0, subframe 1, slot 0
    timing = {0, 1, 0};
    EXPECT_EQ(num.calculate_slot_timestamp(timing), 1'000'000U);

    // Frame 0, subframe 9, slot 1
    timing = {0, 9, 1};
    EXPECT_EQ(num.calculate_slot_timestamp(timing), 9'500'000U);

    // Frame 1, subframe 0, slot 0
    timing = {1, 0, 0};
    EXPECT_EQ(num.calculate_slot_timestamp(timing), 10'000'000U);

    // Frame 10, subframe 5, slot 1
    timing = {10, 5, 1};
    EXPECT_EQ(num.calculate_slot_timestamp(timing), 105'500'000U);
}

// Test round-trip: slot -> timing -> timestamp -> slot (30 kHz)
TEST(NumerologyTest, RoundTripSlot_30kHz) {
    const auto num = ro::from_scs(ro::SubcarrierSpacing::Scs30Khz);

    // Test various absolute slot numbers (30 kHz = 2 slots per subframe)
    const std::vector<std::uint64_t> test_slots = {
            0,   // Frame 0, subframe 0, slot 0
            1,   // Frame 0, subframe 0, slot 1
            2,   // Frame 0, subframe 1, slot 0
            10,  // Frame 0, subframe 5, slot 0
            20,  // Frame 1, subframe 0, slot 0
            200, // Frame 10, subframe 0, slot 0
    };

    for (const std::uint64_t original_slot : test_slots) {
        const auto timing = num.calculate_slot_timing(original_slot);
        const std::uint64_t timestamp = num.calculate_slot_timestamp(timing);
        const std::uint64_t calculated_slot = timestamp / num.slot_period_ns;
        EXPECT_EQ(calculated_slot, original_slot);
    }
}

// Test round-trip: timing -> timestamp -> slot -> timing (30 kHz)
TEST(NumerologyTest, RoundTripTiming_30kHz) {
    const auto num = ro::from_scs(ro::SubcarrierSpacing::Scs30Khz);

    // Test various timings
    const std::vector<ro::OranSlotTiming> test_timings = {
            {0, 0, 0},  // Frame 0, subframe 0, slot 0
            {0, 0, 1},  // Frame 0, subframe 0, slot 1
            {0, 1, 0},  // Frame 0, subframe 1, slot 0
            {0, 5, 1},  // Frame 0, subframe 5, slot 1
            {1, 0, 0},  // Frame 1, subframe 0, slot 0
            {10, 9, 1}, // Frame 10, subframe 9, slot 1
    };

    for (const auto &original_timing : test_timings) {
        const std::uint64_t timestamp = num.calculate_slot_timestamp(original_timing);
        const std::uint64_t absolute_slot = timestamp / num.slot_period_ns;
        const auto calculated_timing = num.calculate_slot_timing(absolute_slot);
        EXPECT_EQ(calculated_timing.frame_id, original_timing.frame_id);
        EXPECT_EQ(calculated_timing.subframe_id, original_timing.subframe_id);
        EXPECT_EQ(calculated_timing.slot_id, original_timing.slot_id);
    }
}

// Test large frame numbers
TEST(NumerologyTest, LargeFrameNumbers) {
    const auto num = ro::from_scs(ro::SubcarrierSpacing::Scs30Khz);

    // Frame 255 (max frame_id since it's uint8_t, wraps at 256)
    const ro::OranSlotTiming timing{255, 9, 1};
    const std::uint64_t timestamp = num.calculate_slot_timestamp(timing);
    EXPECT_EQ(timestamp, 2'559'500'000U);

    // Verify round-trip: timestamp -> absolute slot -> timing
    const std::uint64_t absolute_slot = timestamp / num.slot_period_ns;
    const auto calculated_timing = num.calculate_slot_timing(absolute_slot);
    EXPECT_EQ(calculated_timing.frame_id, 255U);
    EXPECT_EQ(calculated_timing.subframe_id, 9U);
    EXPECT_EQ(calculated_timing.slot_id, 1U);
}

// ============================================================================
// Tests for fapi_to_oran_slot_timing function
// ============================================================================

// Test basic conversion from FapiSlotTiming to OranSlotTiming
TEST(FapiToOranSlotTimingTest, BasicConversion) {
    static constexpr std::uint8_t SLOTS_PER_SUBFRAME = 2;
    // Absolute slot 105 at 30kHz (2 slots/subframe): frame 5, subframe 2, slot 1
    // 105 slots / 20 slots_per_frame = 5 frames, remainder 5 slots
    // 5 slots / 2 slots_per_subframe = 2 subframes, remainder 1 slot
    const ran::fapi::FapiSlotTiming fapi_timing{105, SLOTS_PER_SUBFRAME};
    const ro::OranSlotTiming oran_timing = ro::fapi_to_oran_slot_timing(fapi_timing);

    EXPECT_EQ(oran_timing.frame_id, 5);
    EXPECT_EQ(oran_timing.subframe_id, 2);
    EXPECT_EQ(oran_timing.slot_id, 1);
}

// Test uint8_t boundary values
TEST(FapiToOranSlotTimingTest, Uint8Boundaries) {
    static constexpr std::uint8_t SLOTS_PER_SUBFRAME = 2;
    static constexpr std::uint64_t SLOTS_PER_FRAME = 20;

    // Frame 255 start (max uint8_t value)
    const ran::fapi::FapiSlotTiming fapi_max{255 * SLOTS_PER_FRAME, SLOTS_PER_SUBFRAME};
    const ro::OranSlotTiming oran_max = ro::fapi_to_oran_slot_timing(fapi_max);

    EXPECT_EQ(oran_max.frame_id, 255);
    EXPECT_EQ(oran_max.subframe_id, 0);
    EXPECT_EQ(oran_max.slot_id, 0);

    // Zero values
    const ran::fapi::FapiSlotTiming fapi_zero{0, SLOTS_PER_SUBFRAME};
    const ro::OranSlotTiming oran_zero = ro::fapi_to_oran_slot_timing(fapi_zero);

    EXPECT_EQ(oran_zero.frame_id, 0);
    EXPECT_EQ(oran_zero.subframe_id, 0);
    EXPECT_EQ(oran_zero.slot_id, 0);
}

// Test round-trip: absolute_slot -> FapiSlotTiming -> OranSlotTiming
TEST(FapiToOranSlotTimingTest, ConsistencyWithDirectConversion) {
    static constexpr std::uint8_t SLOTS_PER_SUBFRAME = 2; // 30 kHz
    static constexpr std::uint64_t SLOTS_PER_FRAME = 20;

    // Test various absolute slots including boundary cases
    const std::vector<std::uint64_t> test_slots = {
            0,                          // Start
            1,                          // Basic
            19,                         // End of frame 0
            20,                         // Start of frame 1
            255 * SLOTS_PER_FRAME,      // Frame 255 start (uint8_t boundary)
            255 * SLOTS_PER_FRAME + 19, // Frame 255 end
            256 * SLOTS_PER_FRAME,      // Frame 256 wraps to 0
            1023 * SLOTS_PER_FRAME,     // Frame 1023 (FAPI SFN max)
            1024 * SLOTS_PER_FRAME,     // Frame 1024 wraps
    };

    for (const auto abs_slot : test_slots) {
        // Create FapiSlotTiming and convert to OranSlotTiming
        const auto fapi_timing = ran::fapi::FapiSlotTiming{abs_slot, SLOTS_PER_SUBFRAME};
        const ro::OranSlotTiming oran_timing = ro::fapi_to_oran_slot_timing(fapi_timing);

        // Calculate expected values manually
        const std::uint64_t frames = abs_slot / SLOTS_PER_FRAME;
        const std::uint64_t slots_in_frame = abs_slot % SLOTS_PER_FRAME;
        const auto expected_frame = static_cast<std::uint8_t>((frames % 1024));
        const auto expected_subframe =
                static_cast<std::uint8_t>(slots_in_frame / SLOTS_PER_SUBFRAME);
        const auto expected_slot = static_cast<std::uint8_t>(slots_in_frame % SLOTS_PER_SUBFRAME);

        EXPECT_EQ(oran_timing.frame_id, expected_frame)
                << "Frame mismatch at absolute slot " << abs_slot;
        EXPECT_EQ(oran_timing.subframe_id, expected_subframe)
                << "Subframe mismatch at absolute slot " << abs_slot;
        EXPECT_EQ(oran_timing.slot_id, expected_slot)
                << "Slot mismatch at absolute slot " << abs_slot;
    }
}

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

} // namespace
