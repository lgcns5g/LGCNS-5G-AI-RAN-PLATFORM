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

/**
 * @file timed_trigger_tests.cpp
 * @brief Unit tests for TimedTrigger classes
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <optional>
#include <ratio>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "task/task_errors.hpp"
#include "task/time.hpp"
#include "task/timed_trigger.hpp"

namespace {
namespace ft = framework::task;

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

using namespace std::chrono_literals;

/**
 * Validate Chrome trace format for TimedTrigger events
 * @param[in] content Chrome trace file content
 */
void validate_trigger_chrome_trace(const std::string_view content) {
    // Verify Chrome trace format structure
    EXPECT_NE(content.find("\"traceEvents\":["), std::string::npos);
    EXPECT_NE(content.find("\"name\":\"Trigger_"), std::string::npos);
    EXPECT_NE(content.find("\"cat\":\"TimedTrigger\""), std::string::npos);
    EXPECT_NE(content.find("\"ph\":\"X\""), std::string::npos); // Duration event
    EXPECT_NE(
            content.find("\"tid\":"),
            std::string::npos); // Thread ID (actual thread ID, not hardcoded)
    EXPECT_NE(content.find("\"ts\":"), std::string::npos);  // Timestamp
    EXPECT_NE(content.find("\"dur\":"), std::string::npos); // Duration
}

// ============================================================================
// TimedTrigger Builder Tests
// ============================================================================

TEST(TimedTriggerBuilder, BuilderBasic) {
    const auto interval = 10ms;

    auto trigger = ft::TimedTrigger::create([]() {}, interval).build();

    EXPECT_EQ(trigger.get_interval(), interval);
    EXPECT_FALSE(trigger.is_pinned());
    EXPECT_FALSE(trigger.has_thread_priority());

    // Stats thread should not be pinned by default
    EXPECT_FALSE(trigger.is_stats_pinned());
    EXPECT_EQ(trigger.get_stats_core_id(), 0U); // Should return 0 when not pinned
}

TEST(TimedTriggerBuilder, BuilderChaining) {
    const auto interval = 5ms;
    const std::uint32_t core_id = 4;
    const std::uint32_t priority = 80;

    // Test pinned RT trigger with chaining
    auto trigger1 = ft::TimedTrigger::create([]() {}, interval)
                            .pin_to_core(core_id)
                            .with_rt_priority(priority)
                            .build();

    EXPECT_EQ(trigger1.get_interval(), interval);
    EXPECT_TRUE(trigger1.is_pinned());
    EXPECT_EQ(trigger1.get_core_id(), core_id);
    EXPECT_TRUE(trigger1.has_thread_priority());
    EXPECT_EQ(trigger1.get_thread_priority(), priority);

    // Test trigger creation
    auto trigger2 = ft::TimedTrigger::create([]() {}, interval).build();
    EXPECT_EQ(trigger2.get_interval(), interval);
    EXPECT_FALSE(trigger2.is_pinned());
    EXPECT_FALSE(trigger2.has_thread_priority());
}

TEST(TimedTriggerBuilder, AutoCalculateThresholds) {
    const auto interval = 1ms;
    auto trigger = ft::TimedTrigger::create([]() {}, interval).build();

    // Trigger should be created successfully with auto-calculated thresholds
    EXPECT_EQ(trigger.get_interval(), interval);
}

TEST(TimedTriggerBuilder, StatsCorePinning) {
    const auto interval = 10ms;
    const auto tick_core = std::min(std::thread::hardware_concurrency() - 1U, 2U);
    const auto stats_core = std::min(std::thread::hardware_concurrency() - 1U, 3U);

    // Test trigger with stats thread pinned to different core than tick thread
    auto trigger = ft::TimedTrigger::create([]() {}, interval)
                           .pin_to_core(tick_core)
                           .with_stats_core(stats_core)
                           .build();

    EXPECT_EQ(trigger.get_interval(), interval);

    // Verify tick thread pinning
    EXPECT_TRUE(trigger.is_pinned());
    EXPECT_EQ(trigger.get_core_id(), tick_core);

    // Verify stats thread pinning
    EXPECT_TRUE(trigger.is_stats_pinned());
    EXPECT_EQ(trigger.get_stats_core_id(), stats_core);

    // Test that trigger can start and run with stats thread pinning
    std::atomic<int> callback_count{0};
    auto functional_trigger =
            ft::TimedTrigger::create([&callback_count]() { callback_count++; }, 5ms)
                    .with_stats_core(stats_core)
                    .build();

    EXPECT_TRUE(functional_trigger.is_stats_pinned());
    EXPECT_EQ(functional_trigger.get_stats_core_id(), stats_core);

    EXPECT_TRUE(ft::is_task_success(functional_trigger.start()));
    EXPECT_TRUE(functional_trigger.is_running());

    // Let it run for a short time
    std::this_thread::sleep_for(25ms);

    functional_trigger.stop();
    EXPECT_FALSE(functional_trigger.is_running());

    // Should have executed callbacks
    EXPECT_GE(callback_count.load(), 3);

    // Stats should work - this exercises the pinned stats thread
    EXPECT_NO_THROW(functional_trigger.print_summary());
}

// ============================================================================
// TimedTrigger Tests
// ============================================================================

TEST(TimedTrigger, Construction) {
    EXPECT_NO_THROW({ auto trigger = ft::TimedTrigger::create([]() {}, 10ms).build(); });
}

TEST(TimedTrigger, StartAndStop) {
    auto trigger = ft::TimedTrigger::create([]() {}, 10ms).build();

    EXPECT_FALSE(trigger.is_running());

    // Start the trigger
    const std::error_code start_result = trigger.start();
    EXPECT_TRUE(ft::is_task_success(start_result));
    EXPECT_TRUE(trigger.is_running());

    // Stop the trigger
    trigger.stop();
    EXPECT_FALSE(trigger.is_running());
}

TEST(TimedTrigger, DoubleStart) {
    auto trigger = ft::TimedTrigger::create([]() {}, 10ms).build();

    EXPECT_TRUE(ft::is_task_success(trigger.start()));
    EXPECT_TRUE(trigger.is_running());

    // Second start should fail
    EXPECT_FALSE(ft::is_task_success(trigger.start())); // Should fail - already running
    EXPECT_TRUE(trigger.is_running());

    trigger.stop();
}

TEST(TimedTrigger, CallbackExecution) {
    std::atomic<int> callback_count{0};

    auto trigger = ft::TimedTrigger::create([&callback_count]() { callback_count++; }, 5ms).build();

    EXPECT_TRUE(ft::is_task_success(trigger.start()));

    // Wait for several callbacks
    std::this_thread::sleep_for(50ms);

    const int final_count = callback_count.load();
    EXPECT_GE(final_count,
              8);               // Should have ~10 callbacks in 50ms with 5ms interval
    EXPECT_LE(final_count, 12); // But not too many more

    trigger.stop();
}

TEST(TimedTrigger, PreciseTiming) {
    const auto interval = 10ms;
    std::vector<ft::Nanos> callback_times;
    std::mutex times_mutex;

    auto trigger = ft::TimedTrigger::create(
                           [&callback_times, &times_mutex]() {
                               const std::lock_guard<std::mutex> lock(times_mutex);
                               callback_times.push_back(ft::Time::now_ns());
                           },
                           interval)
                           .build();

    EXPECT_TRUE(ft::is_task_success(trigger.start()));
    std::this_thread::sleep_for(50ms);
    trigger.stop();

    ASSERT_GE(callback_times.size(), 4U);

    // Check intervals between callbacks (skip first interval due to startup
    // overhead)
    for (std::size_t i = 2; i < callback_times.size(); ++i) {
        const auto actual_interval = callback_times[i] - callback_times[i - 1];
        static constexpr std::int64_t MS_TO_NS = 1'000'000;
        const auto expected_ns = interval.count() * MS_TO_NS;
        const auto actual_ns = actual_interval.count();
        const auto error = std::abs(actual_ns - expected_ns);

        static constexpr std::int64_t TIMING_TOLERANCE_NS = 10'000'000; // 10ms
        EXPECT_LT(error, TIMING_TOLERANCE_NS) << "Interval " << i << ": expected " << expected_ns
                                              << "ns, got " << actual_ns << "ns";
    }
}

TEST(TimedTrigger, StartTimeRespected) {
    const ft::Nanos future_start = ft::Time::now_ns() + 5ms;
    const auto interval = 10ms;

    ft::Nanos first_callback_time{0};

    auto trigger = ft::TimedTrigger::create(
                           [&first_callback_time]() {
                               if (first_callback_time.count() == 0) {
                                   first_callback_time = ft::Time::now_ns();
                               }
                           },
                           interval)
                           .pin_to_core(0)
                           .build();

    const ft::Nanos trigger_start_time = ft::Time::now_ns();
    EXPECT_TRUE(ft::is_task_success(trigger.start(future_start)));

    // Wait for first callback
    std::this_thread::sleep_for(50ms);
    trigger.stop();

    EXPECT_GT(first_callback_time.count(), 0);

    // First callback should be approximately at the scheduled start time
    const auto delay = first_callback_time - trigger_start_time;
    const auto expected_delay = future_start - trigger_start_time;
    const auto error = std::abs(delay.count() - expected_delay.count());

    static constexpr std::int64_t START_TIME_TOLERANCE_NS = 100'000'000; // 100ms
    EXPECT_LT(error, START_TIME_TOLERANCE_NS);
}

TEST(TimedTrigger, RestartNotSupported) {
    auto trigger = ft::TimedTrigger::create([]() {}, 10ms).build();

    // First start should succeed
    EXPECT_TRUE(ft::is_task_success(trigger.start()));
    EXPECT_TRUE(trigger.is_running());

    std::this_thread::sleep_for(20ms);

    // Stop the trigger
    trigger.stop();
    EXPECT_FALSE(trigger.is_running());

    // Attempt to restart should fail
    EXPECT_TRUE(trigger.start()); // Any error
    EXPECT_FALSE(trigger.is_running());
}

TEST(TimedTrigger, StatisticsCollectionWithStartOffsets) {
    struct StartOffsetTestCase {
        std::string name;
        std::chrono::milliseconds start_offset;
        std::string description;
    };

    const std::vector<StartOffsetTestCase> test_cases = {
            {"ImmediateStart", 0ms, "No offset - start immediately"},
            {"DelayedStart", 1ms, "1ms start offset"}};

    for (const auto &test_case : test_cases) {
        SCOPED_TRACE("Testing: " + test_case.name + " - " + test_case.description);

        std::atomic<int> callback_count{0};

        auto trigger = ft::TimedTrigger::create(
                               [&callback_count]() {
                                   callback_count++;
                                   // Add small delay to measure callback duration
                                   std::this_thread::sleep_for(10us);
                               },
                               5ms)
                               .build();

        // Start with specified offset from now to test different timing scenarios
        const auto start_time =
                ft::Time::now_ns() +
                std::chrono::duration_cast<std::chrono::nanoseconds>(test_case.start_offset);
        EXPECT_TRUE(ft::is_task_success(trigger.start(start_time)));

        std::this_thread::sleep_for(50ms);
        trigger.stop();

        EXPECT_GE(callback_count.load(), 3);

        // Print summary should not crash and should show meaningful data
        EXPECT_NO_THROW(trigger.print_summary());
    }
}

TEST(TimedTrigger, StatisticsCollection) {
    std::atomic<int> callback_count{0};

    auto trigger = ft::TimedTrigger::create(
                           [&callback_count]() {
                               callback_count++;
                               // Add small delay to measure callback duration
                               std::this_thread::sleep_for(100us);
                           },
                           5ms)
                           .build();

    EXPECT_TRUE(ft::is_task_success(trigger.start()));
    std::this_thread::sleep_for(50ms);
    trigger.stop();

    EXPECT_GE(callback_count.load(), 8);

    // Print summary should not crash and should show meaningful data
    EXPECT_NO_THROW(trigger.print_summary());
}

TEST(TimedTrigger, StatsFileOutput) {
    auto trigger = ft::TimedTrigger::create([]() {}, 100us).build();

    EXPECT_TRUE(ft::is_task_success(trigger.start()));
    std::this_thread::sleep_for(30ms);
    trigger.stop();

    const std::string filename = "test_trigger_stats.json";

    // Clean up any existing file
    std::filesystem::remove(filename);

    const std::error_code result = trigger.write_stats_to_file(filename);
    EXPECT_TRUE(ft::is_task_success(result));

    // Verify file exists and has content
    EXPECT_TRUE(std::filesystem::exists(filename));

    std::ifstream file(filename);
    EXPECT_TRUE(file.is_open());

    std::string line;
    int line_count = 0;
    bool header_found = false;

    while (std::getline(file, line)) {
        line_count++;
        EXPECT_FALSE(line.empty());

        if (line_count == 1) {
            // First line should be version header
            EXPECT_NE(line.find("\"version\":\"1.0\""), std::string::npos);
            header_found = true;
        } else {
            // Execution record lines should contain trigger_count
            EXPECT_NE(line.find("trigger_count"), std::string::npos);
        }
    }

    EXPECT_TRUE(header_found);
    EXPECT_GE(line_count,
              30); // Should have header + at least a few execution records

    // Also test Chrome trace format
    const std::string chrome_filename = "test_trigger_chrome_trace.json";
    std::filesystem::remove(chrome_filename);

    EXPECT_EQ(trigger.write_chrome_trace_to_file(chrome_filename), 0);
    EXPECT_TRUE(std::filesystem::exists(chrome_filename));

    std::ifstream chrome_file(chrome_filename);
    ASSERT_TRUE(chrome_file.is_open());
    std::stringstream buffer;
    buffer << chrome_file.rdbuf();
    chrome_file.close();

    validate_trigger_chrome_trace(buffer.str());
    std::filesystem::remove(chrome_filename);
}

TEST(TimedTrigger, DisabledStats) {
    auto trigger = ft::TimedTrigger::create([]() {}, 10ms).enable_statistics(false).build();

    EXPECT_TRUE(ft::is_task_success(trigger.start()));
    std::this_thread::sleep_for(30ms);
    trigger.stop();

    // Should not crash
    EXPECT_NO_THROW(trigger.print_summary());

    // File output should fail with no data
    const std::string filename = "no_stats_test.json";
    const std::error_code result = trigger.write_stats_to_file(filename);
    EXPECT_FALSE(ft::is_task_success(result)); // Should fail with no data
}

TEST(TimedTrigger, ClearStats) {
    auto trigger = ft::TimedTrigger::create([]() {}, 5ms).build();

    EXPECT_TRUE(ft::is_task_success(trigger.start()));
    std::this_thread::sleep_for(30ms);
    trigger.stop();

    // Clear stats
    trigger.clear_stats();

    // Should not crash and should show no data
    EXPECT_NO_THROW(trigger.print_summary());
}

TEST(TimedTrigger, ExcessiveCallbackDuration) {
    std::atomic<int> callback_count{0};

    // Create trigger with short interval and custom callback duration threshold
    auto trigger = ft::TimedTrigger::create(
                           [&callback_count]() {
                               callback_count++;
                               // Simulate long-running callback that exceeds threshold
                               std::this_thread::sleep_for(2ms);
                           },
                           5ms)
                           .with_callback_duration_threshold(1ms)
                           .build();

    EXPECT_TRUE(ft::is_task_success(trigger.start()));

    // Let a few callbacks execute with excessive duration
    std::this_thread::sleep_for(30ms);
    trigger.stop();

    EXPECT_GE(callback_count.load(), 4); // Should have several callbacks

    // Print summary should show duration warnings
    EXPECT_NO_THROW(trigger.print_summary());
}

TEST(TimedTrigger, MissedTriggerWindows) {
    std::vector<ft::Nanos> callback_times;
    std::mutex times_mutex;
    std::atomic<int> callback_count{0};

    // Create trigger with medium interval
    auto trigger =
            ft::TimedTrigger::create(
                    [&callback_times, &times_mutex, &callback_count]() {
                        const auto now = ft::Time::now_ns();
                        {
                            const std::lock_guard<std::mutex> lock(times_mutex);
                            callback_times.push_back(now);
                        }
                        callback_count++;

                        // First callback simulates a very long operation that
                        // causes missed windows
                        if (callback_count.load() == 1) {
                            // Sleep for much longer than the trigger interval to
                            // miss several windows
                            std::this_thread::sleep_for(20ms); // Much longer than 5ms interval
                        }
                    },
                    5ms)
                    .build();

    const auto start_time = ft::Time::now_ns();
    EXPECT_TRUE(ft::is_task_success(trigger.start(start_time)));

    // Wait for recovery and several more triggers
    std::this_thread::sleep_for(50ms);
    trigger.stop();

    EXPECT_GE(callback_count.load(), 3); // Should have at least a few callbacks
    ASSERT_GE(callback_times.size(), 3U);

    // Analyze timing behavior after the long first callback
    if (callback_times.size() >= 3) {
        // The gap between first and second callback should be longer than normal
        // interval
        const auto gap_1_to_2 = callback_times[1] - callback_times[0];
        const auto normal_interval_ns = 10ms;

        // First gap should be much longer due to the 30ms sleep in first callback
        EXPECT_GT(gap_1_to_2.count(), normal_interval_ns.count() * 2);
    }

    // Print summary to show any jump detections or warnings
    EXPECT_NO_THROW(trigger.print_summary());
}

TEST(TimedTrigger, JumpDetection) {
    std::vector<ft::Nanos> callback_times;
    std::mutex times_mutex;
    std::atomic<int> callback_count{0};

    // Create trigger with custom jump detection threshold
    auto trigger =
            ft::TimedTrigger::create(
                    [&callback_times, &times_mutex, &callback_count]() {
                        const auto now = ft::Time::now_ns();
                        {
                            const std::lock_guard<std::mutex> lock(times_mutex);
                            callback_times.push_back(now);
                        }
                        callback_count++;

                        // After the third callback, introduce a significant
                        // delay to trigger jump detection
                        if (callback_count.load() == 3) {
                            std::this_thread::sleep_for(25ms); // Much larger than 5ms interval
                        }
                    },
                    5ms)
                    .with_jump_threshold(15ms) // 15ms threshold (3x interval)
                    .build();

    EXPECT_TRUE(ft::is_task_success(trigger.start()));

    // Wait for several triggers including the one with delay
    std::this_thread::sleep_for(40ms);
    trigger.stop();

    EXPECT_GE(callback_count.load(), 5); // Should have several callbacks
    ASSERT_GE(callback_times.size(), 4U);

    // Verify that we can detect an abnormal timing jump
    if (callback_times.size() >= 4) {
        // Check intervals between callbacks
        bool found_jump = false;
        const auto normal_interval = 10ms;
        const auto jump_threshold = 15ms;

        for (std::size_t i = 1; i < callback_times.size(); ++i) {
            const auto interval = callback_times[i] - callback_times[i - 1];
            if (interval.count() > jump_threshold.count()) {
                found_jump = true;
                // The jump should be significantly larger than normal interval
                EXPECT_GT(interval.count(), normal_interval.count() * 2);
            }
        }

        // We should detect at least one jump due to the 25ms delay introduced in
        // callback 3
        EXPECT_TRUE(found_jump) << "Expected to detect timing jump, but none found";
    }

    // Print summary should show jump detection statistics
    EXPECT_NO_THROW(trigger.print_summary());
}

TEST(TimedTrigger, SkipAheadStrategy) {
    std::vector<ft::Nanos> callback_times;
    std::mutex times_mutex;
    std::atomic<int> callback_count{0};

    // Create trigger with SkipAhead strategy
    auto trigger =
            ft::TimedTrigger::create(
                    [&callback_times, &times_mutex, &callback_count]() {
                        const auto now = ft::Time::now_ns();
                        {
                            const std::lock_guard<std::mutex> lock(times_mutex);
                            callback_times.push_back(now);
                        }
                        callback_count++;

                        // First callback introduces significant delay that would
                        // cause missed windows
                        if (callback_count.load() == 1) {
                            std::this_thread::sleep_for(35ms); // Much longer than 5ms interval
                        }
                    },
                    5ms // 5ms interval
                    )
                    .with_skip_strategy(ft::SkipStrategy::SkipAhead)
                    .build();

    EXPECT_TRUE(ft::is_task_success(trigger.start()));

    // Wait for recovery and several more triggers
    std::this_thread::sleep_for(100ms);
    trigger.stop();

    EXPECT_GE(callback_count.load(), 2); // Should have at least 2 callbacks
    ASSERT_GE(callback_times.size(), 2U);

    // With skip ahead, we should NOT see rapid catch-up triggers
    // Instead, we should see normal intervals after the initial delay
    if (callback_times.size() >= 2) {
        const auto normal_interval = 5ms;

        // After the first long callback, subsequent intervals should be close to
        // normal (not the rapid-fire catch-up behavior we'd see with CatchupAll)
        for (std::size_t i = 1; i < callback_times.size(); ++i) {
            const auto interval = callback_times[i] - callback_times[i - 1];
            // Should be close to normal interval, not immediate (catch-up) triggers
            EXPECT_GT(interval.count(), normal_interval.count() / 2) // At least 5ms
                    << "Interval " << i << " should not be immediate (catch-up)";
        }
    }

    // Print summary should show skip statistics
    EXPECT_NO_THROW(trigger.print_summary());
}

// ============================================================================
// Stress Tests
// ============================================================================

TEST(TimedTrigger, HighFrequency) {
    std::atomic<int> callback_count{0};
    const auto core = std::min(std::thread::hardware_concurrency() - 1U, 5U);
    auto trigger = ft::TimedTrigger::create(
                           [&callback_count]() {
                               // Simulate some callback work
                               std::this_thread::sleep_for(50us);
                               callback_count++;
                           },
                           500us // 500μs = 2kHz
                           )
                           .pin_to_core(core)
                           .with_rt_priority(80)
                           .build();

    EXPECT_TRUE(ft::is_task_success(trigger.start()));
    std::this_thread::sleep_for(100ms);
    trigger.stop();

    const int final_count = callback_count.load();
    EXPECT_GE(final_count, 180); // Should get ~200 callbacks in 100ms at 2kHz
    EXPECT_LE(final_count, 220); // But not too many more

    // Write Chrome trace for high-frequency trigger analysis
    // Load this file in chrome://tracing to visualize
    EXPECT_EQ(trigger.write_chrome_trace_to_file("high_frequency_trigger_trace.json"), 0);
}

TEST(TimedTrigger, RealTimeStartupLatency) {
    std::atomic<int> callback_count{0};
    ft::Nanos first_callback_time{0};
    const auto core = std::min(std::thread::hardware_concurrency() - 1U, 5U);

    // RT-configured trigger with core pinning and high priority
    auto trigger = ft::TimedTrigger::create(
                           [&callback_count, &first_callback_time]() {
                               const auto now = ft::Time::now_ns();
                               if (callback_count.load() == 0) {
                                   first_callback_time = now;
                               }
                               callback_count++;
                           },
                           10ms // 10ms interval
                           )
                           .pin_to_core(core)
                           .with_rt_priority(80) // High RT priority
                           .build();

    const auto start_time = ft::Time::now_ns();
    EXPECT_TRUE(ft::is_task_success(trigger.start()));

    // Wait for first few callbacks
    while (callback_count.load() < 3) {
        std::this_thread::sleep_for(100us);
    }

    trigger.stop();

    // Calculate startup latency
    const auto startup_latency = first_callback_time - start_time;
    const auto latency_us = static_cast<double>(startup_latency.count()) / 1000.0;

    EXPECT_LT(latency_us, 5000.0); // Should be under 5000μs
}

TEST(TimedTriggerBuilder, MaxExecutionRecordsConfiguration) {
    std::atomic<int> callback_count{0};

    // Use a very small limit to force truncation
    auto trigger = ft::TimedTrigger::create(
                           [&callback_count]() { ++callback_count; },
                           1ms) // Fast callbacks
                           .enable_statistics(true)
                           .with_max_execution_records(5) // Very small limit to force truncation
                           .build();

    EXPECT_TRUE(ft::is_task_success(trigger.start()));

    // Let it run long enough to generate many records and force truncation
    std::this_thread::sleep_for(50ms);

    trigger.stop();

    // Should have executed many callbacks
    EXPECT_GT(callback_count.load(), 10);

    // Write stats to temporary file to check truncation
    const std::string temp_filename =
            (std::filesystem::temp_directory_path() / "test_trigger_truncation.json").string();
    const std::error_code write_result = trigger.write_stats_to_file(temp_filename);
    EXPECT_TRUE(ft::is_task_success(write_result));

    // Check that truncation warning is in the file
    std::ifstream file(temp_filename);
    ASSERT_TRUE(file.is_open());

    std::string line;
    bool found_truncation_warning = false;
    while (std::getline(file, line)) {
        if (line.find("execution_records_truncated") != std::string::npos) {
            found_truncation_warning = true;
            break;
        }
    }
    file.close();

    // Should have truncation warning due to tiny limit
    EXPECT_TRUE(found_truncation_warning);

    // Cleanup
    std::ignore = std::remove(temp_filename.c_str());
}

TEST(TimedTriggerBuilder, AutoCalculateMaxRecords) {
    std::atomic<int> callback_count{0};

    // Test auto-calculation (should default to 50GB worth)
    auto trigger = ft::TimedTrigger::create([&callback_count]() { ++callback_count; }, 10ms)
                           .enable_statistics(true)
                           // Don't set max_execution_records - should auto-calculate
                           .build();

    EXPECT_TRUE(ft::is_task_success(trigger.start()));

    // Let it run briefly
    std::this_thread::sleep_for(50ms);

    trigger.stop();

    // Should have executed some callbacks
    EXPECT_GT(callback_count.load(), 0);

    // Write stats to check no truncation with huge auto-calculated limit
    const std::string temp_filename =
            (std::filesystem::temp_directory_path() / "test_trigger_no_truncation.json").string();
    const std::error_code write_result = trigger.write_stats_to_file(temp_filename);
    EXPECT_TRUE(ft::is_task_success(write_result));

    // Check that NO truncation warning is in the file
    std::ifstream file(temp_filename);
    ASSERT_TRUE(file.is_open());

    std::string line;
    bool found_truncation_warning = false;
    while (std::getline(file, line)) {
        if (line.find("execution_records_truncated") != std::string::npos) {
            found_truncation_warning = true;
            break;
        }
    }
    file.close();

    // Should NOT have truncation warning with huge auto-calculated limit
    EXPECT_FALSE(found_truncation_warning);

    // Cleanup
    std::ignore = std::remove(temp_filename.c_str());
}

// ============================================================================
// TimedTrigger max_triggers Tests
// ============================================================================

TEST(TimedTrigger, MaxTriggersBuilder) {
    const auto interval = 10ms;
    const std::size_t max_count = 50;

    // Test with max_triggers set
    auto trigger_with_max =
            ft::TimedTrigger::create([]() {}, interval).max_triggers(max_count).build();

    EXPECT_EQ(trigger_with_max.get_interval(), interval);
    ASSERT_TRUE(trigger_with_max.max_triggers().has_value());
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    EXPECT_EQ(trigger_with_max.max_triggers().value(), max_count);

    // Test without max_triggers set
    auto trigger_without_max = ft::TimedTrigger::create([]() {}, interval).build();

    EXPECT_FALSE(trigger_without_max.max_triggers().has_value());
}

TEST(TimedTrigger, MaxTriggersAutoStop) {
    const auto interval = 5ms;
    const std::size_t max_count = 10;
    std::atomic<int> callback_count{0};

    auto trigger = ft::TimedTrigger::create([&callback_count]() { callback_count++; }, interval)
                           .max_triggers(max_count)
                           .build();

    EXPECT_TRUE(ft::is_task_success(trigger.start()));
    EXPECT_TRUE(trigger.is_running());

    // Wait for trigger to complete (joins threads automatically)
    trigger.wait_for_completion();

    // Trigger should have stopped automatically
    EXPECT_FALSE(trigger.is_running());

    // Should have executed exactly max_count callbacks
    EXPECT_EQ(callback_count.load(), static_cast<int>(max_count));
}

TEST(TimedTrigger, MaxTriggersExactCount) {
    const auto interval = 10ms;
    const std::size_t max_count = 5;
    std::atomic<int> callback_count{0};

    auto trigger = ft::TimedTrigger::create([&callback_count]() { callback_count++; }, interval)
                           .max_triggers(max_count)
                           .build();

    EXPECT_TRUE(ft::is_task_success(trigger.start()));

    // Wait for completion (joins threads automatically)
    trigger.wait_for_completion();

    // Should have stopped
    EXPECT_FALSE(trigger.is_running());

    // Verify exactly max_count callbacks executed
    EXPECT_EQ(callback_count.load(), static_cast<int>(max_count));
}

TEST(TimedTrigger, WaitForCompletionWithoutMaxTriggers) {
    const auto interval = 10ms;

    auto trigger = ft::TimedTrigger::create([]() {}, interval).build();

    EXPECT_TRUE(ft::is_task_success(trigger.start()));

    // Should throw since max_triggers is not set
    EXPECT_THROW(trigger.wait_for_completion(), std::logic_error);

    trigger.stop();
}

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

} // anonymous namespace
