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

#include <chrono>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "fronthaul/fronthaul.hpp"

namespace {

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

using namespace std::chrono_literals;

/**
 * Test data structure for packet send time calculation
 */
struct PacketSendTimeTestData {
    ran::fronthaul::PacketSendTimeParams params{};
    std::chrono::nanoseconds expected_start{};
    std::chrono::nanoseconds expected_threshold{};
    std::chrono::nanoseconds expected_tx{};
    bool expected_exceeds_threshold{};
    std::string_view description;
};

/**
 * Table-based test for packet send time calculation covering nominal cases,
 * edge cases, and threshold violations with various parameter combinations
 */
TEST(PacketSendTimeTest, CalculatePacketSendTime) {
    const std::vector<PacketSendTimeTestData> test_cases =
            {// Nominal case: slot 0, 1 slot ahead, 30kHz SCS
             {.params =
                      {
                              .t0 = 1s,
                              .tai_offset = 37s,
                              .absolute_slot = 0,
                              .slot_period = 500us, // 30kHz SCS
                              .slot_ahead = 1,
                              .t1a_max_cp_ul = 336us,
                              .actual_start = 1s - 500us // Processing 500us early (1 slot)
                      },
              .expected_start = 1s - 500us,          // t0 + 500us * (0 - 1) = 999.5ms
              .expected_threshold = 164us,           // 500us - 336us
              .expected_tx = 37s + 999500us + 164us, // 999.5ms + 164us + 37s TAI
              .expected_exceeds_threshold = false,
              .description = "Slot 0, 1 slot ahead, on time"},

             // Nominal case: slot 10, 1 slot ahead
             {.params =
                      {
                              .t0 = 1s,
                              .tai_offset = 37s,
                              .absolute_slot = 10,
                              .slot_period = 500us,
                              .slot_ahead = 1,
                              .t1a_max_cp_ul = 336us,
                              .actual_start = 1s + 5ms - 500us // 10 slots + processing early
                      },
              .expected_start = 1s + 4500us, // t0 + 500us * (10 - 1) = 1s + 4.5ms
              .expected_threshold = 164us,
              .expected_tx = 38s + 4664us, // 1s + 4.5ms + 164us + 37s = 38.004664s
              .expected_exceeds_threshold = false,
              .description = "Slot 10, 1 slot ahead, on time"},

             // Different slot_ahead: 2 slots ahead
             {.params =
                      {
                              .t0 = 1s,
                              .tai_offset = 37s,
                              .absolute_slot = 0,
                              .slot_period = 500us,
                              .slot_ahead = 2,
                              .t1a_max_cp_ul = 336us,
                              .actual_start = 1s - 1ms // Processing 1ms early (2 slots)
                      },
              .expected_start = 1s - 1ms,    // t0 + 500us * (0 - 2) = 999ms
              .expected_threshold = 664us,   // 1000us - 336us
              .expected_tx = 37s + 999664us, // 999ms + 664us + 37s
              .expected_exceeds_threshold = false,
              .description = "Slot 0, 2 slots ahead, on time"},

             // Different SCS: 15kHz (1ms slot period)
             {.params =
                      {.t0 = 1s,
                       .tai_offset = 37s,
                       .absolute_slot = 0,
                       .slot_period = 1ms, // 15kHz SCS
                       .slot_ahead = 1,
                       .t1a_max_cp_ul = 336us,
                       .actual_start = 1s - 1ms},
              .expected_start = 1s - 1ms,    // t0 + 1ms * (0 - 1) = 999ms
              .expected_threshold = 664us,   // 1000us - 336us
              .expected_tx = 37s + 999664us, // 999ms + 664us + 37s
              .expected_exceeds_threshold = false,
              .description = "Slot 0, 15kHz SCS, 1 slot ahead"},

             // Different SCS: 120kHz (125us slot period)
             {.params =
                      {.t0 = 1s,
                       .tai_offset = 37s,
                       .absolute_slot = 0,
                       .slot_period = 125us, // 120kHz SCS
                       .slot_ahead = 1,
                       .t1a_max_cp_ul = 336us,
                       .actual_start = 1s - 125us},
              .expected_start = 1s - 125us,            // t0 + 125us * (0 - 1) = 0.999875s
              .expected_threshold = -211us,            // 125us - 336us (negative!)
              .expected_tx = 37s + 1s - 125us - 211us, // 0.999875s + (-211us) + 37s = 37.999664s
              .expected_exceeds_threshold = false,
              .description = "Slot 0, 120kHz SCS, 1 slot ahead"},

             // Different t1a_max: smaller window
             {.params =
                      {.t0 = 1s,
                       .tai_offset = 37s,
                       .absolute_slot = 0,
                       .slot_period = 500us,
                       .slot_ahead = 1,
                       .t1a_max_cp_ul = 200us, // Smaller window
                       .actual_start = 1s - 500us},
              .expected_start = 1s - 500us,  // t0 + 500us * (0 - 1) = 999.5ms
              .expected_threshold = 300us,   // 500us - 200us
              .expected_tx = 37s + 999800us, // 999.5ms + 300us + 37s
              .expected_exceeds_threshold = false,
              .description = "Slot 0, smaller t1a_max window"},

             // Zero TAI offset
             {.params =
                      {.t0 = 1s,
                       .tai_offset = 0s,
                       .absolute_slot = 0,
                       .slot_period = 500us,
                       .slot_ahead = 1,
                       .t1a_max_cp_ul = 336us,
                       .actual_start = 1s - 500us},
              .expected_start = 1s - 500us, // t0 + 500us * (0 - 1) = 999.5ms
              .expected_threshold = 164us,
              .expected_tx = 999664us, // 999.5ms + 164us (no TAI offset)
              .expected_exceeds_threshold = false,
              .description = "Slot 0, zero TAI offset"},

             // Large slot number
             {.params =
                      {
                              .t0 = 1s,
                              .tai_offset = 37s,
                              .absolute_slot = 1'000'000,
                              .slot_period = 500us,
                              .slot_ahead = 1,
                              .t1a_max_cp_ul = 336us,
                              .actual_start = 501s -
                                              500us // 1M * 500us = 500s, so 1s + 500s - 500us
                      },
              .expected_start = 500s + 999500us, // t0 + 500us * (1M - 1) = 1s + 499999500us
              .expected_threshold = 164us,
              .expected_tx = 537s + 999664us, // 500999500us + 164us + 37s
              .expected_exceeds_threshold = false,
              .description = "Large slot number (slot 1000000)"},

             // Edge case: Processing exactly at threshold
             {.params =
                      {
                              .t0 = 1s,
                              .tai_offset = 37s,
                              .absolute_slot = 0,
                              .slot_period = 500us,
                              .slot_ahead = 1,
                              .t1a_max_cp_ul = 336us,
                              .actual_start = 1s - 336us // Exactly at threshold (expected_start +
                                                         // threshold)
                      },
              .expected_start = 1s - 500us, // t0 + 500us * (0 - 1) = 999.5ms
              .expected_threshold = 164us,
              .expected_tx = 37s + 999664us, // 999.5ms + 164us + 37s
              .expected_exceeds_threshold =
                      false, // actual is exactly at 164us threshold (not exceeded)
              .description = "Processing exactly at threshold boundary"},

             // Threshold violation: Processing too late
             {.params =
                      {
                              .t0 = 1s,
                              .tai_offset = 37s,
                              .absolute_slot = 0,
                              .slot_period = 500us,
                              .slot_ahead = 1,
                              .t1a_max_cp_ul = 336us,
                              .actual_start = 1s + 165us // 1us beyond threshold
                      },
              .expected_start = 1s - 500us, // t0 + 500us * (0 - 1) = 999.5ms
              .expected_threshold = 164us,
              .expected_tx = 37s + 999664us,      // 999.5ms + 164us + 37s
              .expected_exceeds_threshold = true, // actual is 665us late (exceeds threshold)
              .description = "Threshold violation - processing too late"},

             // Severe threshold violation
             {.params =
                      {
                              .t0 = 1s,
                              .tai_offset = 37s,
                              .absolute_slot = 0,
                              .slot_period = 500us,
                              .slot_ahead = 1,
                              .t1a_max_cp_ul = 336us,
                              .actual_start = 1s + 1ms // 1ms late!
                      },
              .expected_start = 1s - 500us, // t0 + 500us * (0 - 1) = 999.5ms
              .expected_threshold = 164us,
              .expected_tx = 37s + 999664us,      // 999.5ms + 164us + 37s
              .expected_exceeds_threshold = true, // actual is 1.5ms late (exceeds threshold)
              .description = "Severe threshold violation - very late"},

             // Edge case: Processing very early (before expected)
             {.params =
                      {
                              .t0 = 1s,
                              .tai_offset = 37s,
                              .absolute_slot = 0,
                              .slot_period = 500us,
                              .slot_ahead = 1,
                              .t1a_max_cp_ul = 336us,
                              .actual_start = 1s - 2ms // 2ms early
                      },
              .expected_start = 1s - 500us, // t0 + 500us * (0 - 1) = 999.5ms
              .expected_threshold = 164us,
              .expected_tx = 37s + 999664us,       // 999.5ms + 164us + 37s
              .expected_exceeds_threshold = false, // Negative delta doesn't exceed
              .description = "Processing very early - negative delta"},

             // Zero slot_ahead (shouldn't happen in practice, but test it)
             {.params =
                      {.t0 = 1s,
                       .tai_offset = 37s,
                       .absolute_slot = 0,
                       .slot_period = 500us,
                       .slot_ahead = 0,
                       .t1a_max_cp_ul = 336us,
                       .actual_start = 1s},
              .expected_start = 1s,         // t0 + 500us * (0 - 0) = 1s
              .expected_threshold = -336us, // Negative threshold
              .expected_tx = 38s - 336us,   // 1s + (-336us) + 37s = 37.999664s
              .expected_exceeds_threshold = false,
              .description = "Zero slot_ahead edge case"},

             // Large t1a_max (exceeds slot_period)
             {.params =
                      {.t0 = 1s,
                       .tai_offset = 37s,
                       .absolute_slot = 0,
                       .slot_period = 500us,
                       .slot_ahead = 1,
                       .t1a_max_cp_ul = 600us, // Exceeds slot_period
                       .actual_start = 1s - 500us},
              .expected_start = 1s - 500us,  // t0 + 500us * (0 - 1) = 999.5ms
              .expected_threshold = -100us,  // Negative threshold
              .expected_tx = 37s + 999400us, // 999.5ms + (-100us) + 37s
              .expected_exceeds_threshold = false,
              .description = "t1a_max exceeds slot period"},

             // Different t0 (not starting at 1 second)
             {.params =
                      {.t0 = 5s,
                       .tai_offset = 37s,
                       .absolute_slot = 0,
                       .slot_period = 500us,
                       .slot_ahead = 1,
                       .t1a_max_cp_ul = 336us,
                       .actual_start = 5s - 500us},
              .expected_start = 5s - 500us, // t0 + 500us * (0 - 1) = 4999.5ms
              .expected_threshold = 164us,
              .expected_tx = 41s + 999664us, // 4999.5ms + 164us + 37s TAI
              .expected_exceeds_threshold = false,
              .description = "Different t0 value"}};

    for (const auto &test_case : test_cases) {
        SCOPED_TRACE(test_case.description);

        const auto result = ran::fronthaul::calculate_packet_send_time(test_case.params);

        EXPECT_EQ(result.expected_start, test_case.expected_start);
        EXPECT_EQ(result.actual_start, test_case.params.actual_start);
        EXPECT_EQ(result.threshold, test_case.expected_threshold);
        EXPECT_EQ(result.start_tx, test_case.expected_tx);
        EXPECT_EQ(result.exceeds_threshold, test_case.expected_exceeds_threshold);

        // Verify time_delta calculation
        const auto expected_delta = test_case.params.actual_start - test_case.expected_start;
        EXPECT_EQ(result.time_delta, expected_delta);
    }
}

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

} // namespace
