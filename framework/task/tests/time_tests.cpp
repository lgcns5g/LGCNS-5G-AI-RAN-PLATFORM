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
#include <string>
#include <thread>

#include <gtest/gtest.h>

#include "task/time.hpp"

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

namespace {
namespace ft = framework::task;

using ft::Nanos;
using namespace std::chrono_literals;

constexpr ft::Nanos TOLERANCE = 10ms; // 10ms tolerance for timing tests

TEST(Time, NowNsReturnsMonotonicTime) {
    const ft::Nanos time1 = ft::Time::now_ns();
    std::this_thread::sleep_for(1ms);
    const ft::Nanos time2 = ft::Time::now_ns();

    // Time should advance
    EXPECT_GT(time2, time1);

    // Time difference should be reasonable (at least 1ms, less than 10ms)
    const ft::Nanos diff = time2 - time1;
    EXPECT_GE(diff, 1ms);
    EXPECT_LT(diff, 10ms);
}

TEST(Time, NowReturnsValidTimePoint) {
    const auto time1 = ft::Time::now();
    std::this_thread::sleep_for(1ms);
    const auto time2 = ft::Time::now();

    // Time should advance
    EXPECT_GT(time2, time1);

    // Time difference should be reasonable
    const auto diff = time2 - time1;
    EXPECT_GE(diff, 1ms);
    EXPECT_LT(diff, 10ms);
}

TEST(Time, NowNsAndNowAreConsistent) {
    // Get both timestamps as close as possible
    const ft::Nanos ns_time = ft::Time::now_ns();
    const auto chrono_time = ft::Time::now();
    const auto chrono_ns = std::chrono::duration_cast<Nanos>(chrono_time.time_since_epoch());

    // They should be very close (within 10μs)
    const auto diff = std::chrono::abs(chrono_ns - ns_time);
    EXPECT_LT(diff, 10us);
}

TEST(Time, SleepUntilNsWithPastTime) {
    const ft::Nanos past_time = ft::Time::now_ns() - 1ms;
    const auto start = ft::Time::now();

    // Should return immediately for past times
    ft::Time::sleep_until(past_time);

    const auto end = ft::Time::now();
    const auto elapsed = end - start;

    // Should complete very quickly (within tolerance)
    EXPECT_LT(elapsed, TOLERANCE);
}

TEST(Time, SleepUntilNsPrecisionShortDuration) {
    const ft::Nanos start_time = ft::Time::now_ns();
    const ft::Nanos target_time = start_time + 100us; // 100μs from now

    ft::Time::sleep_until(target_time);

    const ft::Nanos end_time = ft::Time::now_ns();
    const auto actual_duration = end_time - start_time;
    const auto expected_duration = target_time - start_time;

    // Should be close to expected duration
    const auto error = std::chrono::abs(actual_duration - expected_duration);
    EXPECT_LT(error, TOLERANCE);

    // Should not undershoot significantly
    EXPECT_GE(actual_duration, expected_duration - 10us); // Allow 10μs undershoot
}

TEST(Time, SleepUntilNsPrecisionLongDuration) {
    const ft::Nanos start_time = ft::Time::now_ns();
    const ft::Nanos target_time = start_time + 2ms; // 2ms from now

    ft::Time::sleep_until(target_time);

    const ft::Nanos end_time = ft::Time::now_ns();
    const auto actual_duration = end_time - start_time;
    const auto expected_duration = target_time - start_time;

    // Should be close to expected duration
    const auto error = std::chrono::abs(actual_duration - expected_duration);
    EXPECT_LT(error, TOLERANCE);

    // Should not undershoot significantly
    EXPECT_GE(actual_duration, expected_duration - 10us); // Allow 10μs undershoot
}

TEST(Time, SleepUntilTimePointWithPastTime) {
    const auto past_time = ft::Time::now() - 1ms;
    const auto start = ft::Time::now();

    // Should return immediately for past times
    ft::Time::sleep_until(past_time);

    const auto end = ft::Time::now();
    const auto elapsed = end - start;

    // Should complete very quickly
    EXPECT_LT(elapsed, TOLERANCE);
}

TEST(Time, SleepUntilTimePointPrecision) {
    const auto start_time = ft::Time::now();
    const auto target_time = start_time + 500us;

    ft::Time::sleep_until(target_time);

    const auto end_time = ft::Time::now();
    const auto actual_duration = end_time - start_time;
    const auto expected_duration = target_time - start_time;

    // Should be close to expected duration
    const auto error = std::chrono::abs(actual_duration - expected_duration);
    EXPECT_LT(error, TOLERANCE);

    // Should not undershoot significantly
    EXPECT_GE(actual_duration, expected_duration - 1ms);
}

TEST(Time, SleepUntilConsistency) {
    // Test that both overloads produce similar results
    const auto offset = 200us; // 200μs

    // Test Nanos version
    const ft::Nanos start1 = ft::Time::now_ns();
    ft::Time::sleep_until(start1 + offset);
    const ft::Nanos end1 = ft::Time::now_ns();
    const auto duration1 = end1 - start1;

    // Small delay between tests
    std::this_thread::sleep_for(10us);

    // Test time_point version
    const auto start2 = ft::Time::now();
    const auto target2 = start2 + offset;
    ft::Time::sleep_until(target2);
    const auto end2 = ft::Time::now();
    const auto duration2 = end2 - start2;

    // Both should be close to the expected offset
    const auto err1 = std::chrono::abs(duration1 - offset);
    const auto err2 = std::chrono::abs(duration2 - offset);
    EXPECT_LT(err1, TOLERANCE);
    EXPECT_LT(err2, TOLERANCE);

    // And close to each other
    const auto diff12 = std::chrono::abs(duration1 - duration2);
    EXPECT_LT(diff12, TOLERANCE);
}

TEST(Time, CpuPauseStatic) {
    // Test that cpu_pause() can be called
    // We can't easily test its effectiveness, but we can ensure it's callable
    EXPECT_NO_THROW(ft::Time::cpu_pause());

    // Call multiple times to ensure it's safe
    for (int i = 0; i < 10; ++i) {
        ft::Time::cpu_pause();
    }

    // Test that it can be used in a tight loop (simulating busy-wait)
    const auto start = ft::Time::now_ns();
    for (int i = 0; i < 1000; ++i) {
        ft::Time::cpu_pause();
    }
    const auto end = ft::Time::now_ns();
    const auto duration = end - start;

    // Should complete quickly (within 1ms) but not instantaneously
    EXPECT_LT(duration, 1ms);
    EXPECT_GT(duration, 0ns);
}

} // anonymous namespace

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
