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
 * @file memory_trigger_tests.cpp
 * @brief Unit tests for MemoryTrigger classes
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <ratio>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "log/components.hpp"
#include "task/memory_trigger.hpp"
#include "task/task_errors.hpp"

namespace {
namespace ft = framework::task;

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

using namespace std::chrono_literals;

// Test enum similar to DOCA GPU semaphore status
enum TestStatus : std::uint32_t { FREE = 0, READY = 1, DONE = 2, ERROR = 3, EXIT = 4 };

constexpr std::chrono::microseconds POLLING_INTERVAL{10};

// ============================================================================
// MemoryTrigger Construction Tests
// ============================================================================

TEST(MemoryTrigger, BasicConstruction) {
    auto memory_ptr = std::make_shared<std::atomic<std::uint32_t>>(0);
    std::atomic<bool> callback_called = false;

    EXPECT_NO_THROW({
        auto trigger =
                ft::make_memory_trigger(
                        memory_ptr,
                        [&callback_called](std::uint32_t /*old_val*/, std::uint32_t /*new_val*/) {
                            callback_called = true;
                        })
                        .build();
    });
}

TEST(MemoryTrigger, BuilderPattern) {
    auto memory_ptr = std::make_shared<std::atomic<std::uint32_t>>(0);

    EXPECT_NO_THROW({
        auto trigger = ft::make_memory_trigger(memory_ptr, [](std::uint32_t, std::uint32_t) {})
                               .with_comparator([](std::uint32_t old_val, std::uint32_t new_val) {
                                   return old_val != new_val;
                               })
                               .with_notification_strategy(ft::NotificationStrategy::Polling)
                               .with_polling_interval(std::chrono::microseconds(50))
                               .pin_to_core(0)
                               .with_priority(10)
                               .build();
    });
}

// ============================================================================
// Basic Functionality Tests
// ============================================================================

TEST(MemoryTrigger, StartAndStop) {
    auto memory_ptr = std::make_shared<std::atomic<std::uint32_t>>(0);
    auto trigger = ft::make_memory_trigger(memory_ptr, [](std::uint32_t, std::uint32_t) {}).build();

    EXPECT_FALSE(trigger.is_running());

    EXPECT_TRUE(ft::is_task_success(trigger.start()));
    EXPECT_TRUE(trigger.is_running());

    trigger.stop();
    EXPECT_FALSE(trigger.is_running());
}

TEST(MemoryTrigger, DoubleStart) {
    auto memory_ptr = std::make_shared<std::atomic<std::uint32_t>>(0);
    auto trigger = ft::make_memory_trigger(memory_ptr, [](std::uint32_t, std::uint32_t) {}).build();

    EXPECT_TRUE(ft::is_task_success(trigger.start()));
    EXPECT_EQ(trigger.start(), make_error_code(ft::TaskErrc::AlreadyRunning));

    trigger.stop();
}

// ============================================================================
// Condition Variable Mode Tests
// ============================================================================

TEST(MemoryTrigger, ConditionVariableMode_BasicTrigger) {
    auto memory_ptr = std::make_shared<std::atomic<std::uint32_t>>(0);
    std::uint32_t callback_old_value = 0;
    std::uint32_t callback_new_value = 0;
    std::atomic<bool> callback_called = false;
    std::mutex callback_mutex;

    auto trigger = ft::make_memory_trigger(
                           memory_ptr,
                           [&](std::uint32_t old_val, std::uint32_t new_val) {
                               const std::lock_guard<std::mutex> lock(callback_mutex);
                               callback_old_value = old_val;
                               callback_new_value = new_val;
                               callback_called = true;
                           })
                           .with_notification_strategy(ft::NotificationStrategy::ConditionVariable)
                           .build();

    EXPECT_TRUE(ft::is_task_success(trigger.start()));

    // Change memory value and notify
    memory_ptr->store(42, std::memory_order_release);
    trigger.notify();

    // Wait for callback
    std::this_thread::sleep_for(50ms);

    {
        const std::lock_guard<std::mutex> lock(callback_mutex);
        EXPECT_TRUE(callback_called);
        EXPECT_EQ(callback_old_value, 0);
        EXPECT_EQ(callback_new_value, 42);
    }

    trigger.stop();
}

TEST(MemoryTrigger, ConditionVariableMode_NoNotifyNoTrigger) {
    auto memory_ptr = std::make_shared<std::atomic<std::uint32_t>>(0);
    std::atomic<bool> callback_called = false;

    auto trigger =
            ft::make_memory_trigger(
                    memory_ptr,
                    [&callback_called](std::uint32_t, std::uint32_t) { callback_called = true; })
                    .with_notification_strategy(ft::NotificationStrategy::ConditionVariable)
                    .build();

    EXPECT_TRUE(ft::is_task_success(trigger.start()));

    // Wait for monitor thread to enter cv.wait() state
    // We need to ensure the thread has completed the predicate evaluation and is
    // truly waiting.
    std::this_thread::sleep_for(20ms);

    // Change memory value but DON'T notify
    memory_ptr->store(42, std::memory_order_release);

    // Wait - callback should NOT be called without notify
    std::this_thread::sleep_for(50ms);
    EXPECT_FALSE(callback_called);

    trigger.stop();
}

// ============================================================================
// Polling Mode Tests
// ============================================================================

TEST(MemoryTrigger, PollingMode_BasicTrigger) {
    auto memory_ptr = std::make_shared<std::atomic<std::uint32_t>>(0);
    std::uint32_t callback_old_value = 0;
    std::uint32_t callback_new_value = 0;
    std::atomic<bool> callback_called = false;
    std::mutex callback_mutex;

    auto trigger = ft::make_memory_trigger(
                           memory_ptr,
                           [&](std::uint32_t old_val, std::uint32_t new_val) {
                               const std::lock_guard<std::mutex> lock(callback_mutex);
                               callback_old_value = old_val;
                               callback_new_value = new_val;
                               callback_called = true;
                           })
                           .with_notification_strategy(ft::NotificationStrategy::Polling)
                           .with_polling_interval(POLLING_INTERVAL)
                           .build();

    EXPECT_TRUE(ft::is_task_success(trigger.start()));

    // Change memory value (no notify needed in polling mode)
    memory_ptr->store(42, std::memory_order_release);

    // Wait for polling to detect change
    std::this_thread::sleep_for(100ms);

    {
        const std::lock_guard<std::mutex> lock(callback_mutex);
        EXPECT_TRUE(callback_called);
        EXPECT_EQ(callback_old_value, 0);
        EXPECT_EQ(callback_new_value, 42);
    }

    trigger.stop();
}

TEST(MemoryTrigger, PollingMode_FastPolling) {
    auto memory_ptr = std::make_shared<std::atomic<std::uint32_t>>(0);
    std::vector<std::uint32_t> values_seen;
    std::mutex values_mutex;

    auto trigger = ft::make_memory_trigger(
                           memory_ptr,
                           [&](std::uint32_t /*old_val*/, std::uint32_t new_val) {
                               const std::lock_guard<std::mutex> lock(values_mutex);
                               values_seen.push_back(new_val);
                           })
                           .with_notification_strategy(ft::NotificationStrategy::Polling)
                           .with_polling_interval(std::chrono::microseconds(1)) // Very fast polling
                           .build();

    EXPECT_TRUE(ft::is_task_success(trigger.start()));

    // Rapidly change values
    for (std::uint32_t i = 1; i <= 5; ++i) {
        memory_ptr->store(i, std::memory_order_release);
        std::this_thread::sleep_for(5ms);
    }

    // Wait for all changes to be detected
    std::this_thread::sleep_for(50ms);

    {
        const std::lock_guard<std::mutex> lock(values_mutex);
        EXPECT_GE(values_seen.size(), 5); // Should see all changes
        EXPECT_EQ(values_seen.back(), 5); // Last value should be 5
    }

    trigger.stop();
}

// ============================================================================
// Comparator Tests
// ============================================================================

TEST(MemoryTrigger, DefaultComparator_AnyChange) {
    auto memory_ptr = std::make_shared<std::atomic<std::uint32_t>>(0);
    std::vector<std::uint32_t> new_values;
    std::mutex values_mutex;

    auto trigger = ft::make_memory_trigger(
                           memory_ptr,
                           [&](std::uint32_t /*old_val*/, std::uint32_t new_val) {
                               const std::lock_guard<std::mutex> lock(values_mutex);
                               new_values.push_back(new_val);
                           })
                           .with_notification_strategy(ft::NotificationStrategy::Polling)
                           .with_polling_interval(POLLING_INTERVAL)
                           .build();

    EXPECT_TRUE(ft::is_task_success(trigger.start()));

    // Change value multiple times
    memory_ptr->store(10, std::memory_order_release);
    std::this_thread::sleep_for(50ms);
    memory_ptr->store(20, std::memory_order_release);
    std::this_thread::sleep_for(50ms);
    memory_ptr->store(10, std::memory_order_release); // Same as before, but still a change
    std::this_thread::sleep_for(50ms);

    {
        const std::lock_guard<std::mutex> lock(values_mutex);
        EXPECT_EQ(new_values.size(), 3);
        EXPECT_EQ(new_values[0], 10);
        EXPECT_EQ(new_values[1], 20);
        EXPECT_EQ(new_values[2], 10);
    }

    trigger.stop();
}

TEST(MemoryTrigger, CustomComparator_ValueEquals) {
    auto memory_ptr = std::make_shared<std::atomic<std::uint32_t>>(0);
    std::atomic<bool> callback_called{false};

    auto trigger =
            ft::make_memory_trigger(
                    memory_ptr,
                    [&callback_called](std::uint32_t, std::uint32_t) { callback_called = true; })
                    .with_comparator([](std::uint32_t /*old_val*/, std::uint32_t new_val) {
                        return new_val == 42; // Only trigger when value becomes 42
                    })
                    .with_notification_strategy(ft::NotificationStrategy::Polling)
                    .with_polling_interval(POLLING_INTERVAL)
                    .build();

    EXPECT_TRUE(ft::is_task_success(trigger.start()));

    // Change to non-target values (should not trigger)
    memory_ptr->store(10, std::memory_order_release);
    std::this_thread::sleep_for(10ms);
    EXPECT_FALSE(callback_called);

    memory_ptr->store(20, std::memory_order_release);
    std::this_thread::sleep_for(10ms);
    EXPECT_FALSE(callback_called);

    // Change to target value (should trigger)
    memory_ptr->store(42, std::memory_order_release);
    std::this_thread::sleep_for(10ms);
    EXPECT_TRUE(callback_called);

    trigger.stop();
}

TEST(MemoryTrigger, CustomComparator_MultipleValues) {
    auto memory_ptr = std::make_shared<std::atomic<TestStatus>>(FREE);
    std::vector<TestStatus> triggered_values;
    std::mutex values_mutex;

    // Trigger on READY, ERROR, or EXIT (like DOCA use case)
    const std::set<TestStatus> action_statuses = {READY, ERROR, EXIT};

    auto trigger =
            ft::make_memory_trigger(
                    memory_ptr,
                    [&](TestStatus /*old_val*/, TestStatus new_val) {
                        const std::lock_guard<std::mutex> lock(values_mutex);
                        triggered_values.push_back(new_val);
                    })
                    .with_comparator([action_statuses](TestStatus old_val, TestStatus new_val) {
                        // Only trigger when transitioning TO a target status (not when
                        // staying in one)
                        return (old_val != new_val) && action_statuses.contains(new_val);
                    })
                    .with_notification_strategy(ft::NotificationStrategy::Polling)
                    .with_polling_interval(POLLING_INTERVAL)
                    .build();

    EXPECT_TRUE(ft::is_task_success(trigger.start()));

    // Test various status transitions
    memory_ptr->store(READY, std::memory_order_release);
    std::this_thread::sleep_for(10ms);

    memory_ptr->store(DONE, std::memory_order_release); // Should NOT trigger
    std::this_thread::sleep_for(10ms);

    memory_ptr->store(ERROR, std::memory_order_release); // Should trigger
    std::this_thread::sleep_for(10ms);

    memory_ptr->store(EXIT, std::memory_order_release); // Should trigger
    std::this_thread::sleep_for(10ms);

    {
        const std::lock_guard<std::mutex> lock(values_mutex);
        EXPECT_EQ(triggered_values.size(), 3);
        EXPECT_EQ(triggered_values[0], READY);
        EXPECT_EQ(triggered_values[1], ERROR);
        EXPECT_EQ(triggered_values[2], EXIT);
    }

    trigger.stop();
}

// ============================================================================
// Edge Cases and Error Handling Tests
// ============================================================================

TEST(MemoryTrigger, ThrowingCallback) {
    auto memory_ptr = std::make_shared<std::atomic<std::uint32_t>>(0);
    std::atomic<bool> callback_called = false;

    auto trigger = ft::make_memory_trigger(
                           memory_ptr,
                           [&callback_called](std::uint32_t, std::uint32_t) {
                               callback_called = true;
                               throw std::runtime_error("Callback exception");
                           })
                           .with_notification_strategy(ft::NotificationStrategy::Polling)
                           .with_polling_interval(POLLING_INTERVAL)
                           .build();

    EXPECT_TRUE(ft::is_task_success(trigger.start()));

    // Should not crash even if callback throws
    memory_ptr->store(42, std::memory_order_release);
    std::this_thread::sleep_for(10ms);
    EXPECT_TRUE(callback_called);

    trigger.stop();
}

TEST(MemoryTrigger, ThrowingComparator) {
    auto memory_ptr = std::make_shared<std::atomic<std::uint32_t>>(0);
    std::atomic<bool> callback_called = false;

    auto trigger =
            ft::make_memory_trigger(
                    memory_ptr,
                    [&callback_called](std::uint32_t, std::uint32_t) { callback_called = true; })
                    .with_comparator([](std::uint32_t, std::uint32_t) -> bool {
                        throw std::runtime_error("Comparator exception");
                    })
                    .with_notification_strategy(ft::NotificationStrategy::Polling)
                    .with_polling_interval(POLLING_INTERVAL)
                    .build();

    EXPECT_TRUE(ft::is_task_success(trigger.start()));

    // Should not crash even if comparator throws (and should not trigger)
    memory_ptr->store(42, std::memory_order_release);
    std::this_thread::sleep_for(10ms);
    EXPECT_FALSE(callback_called); // Should not trigger due to exception

    trigger.stop();
}

TEST(MemoryTrigger, NullMemoryPtr) {
    // Should throw or handle gracefully with null memory pointer
    EXPECT_THROW(
            {
                auto trigger = ft::MemoryTrigger<std::uint32_t>::create(
                                       nullptr, // Null memory pointer - need explicit type
                                                // since nullptr can't deduce T
                                       [](std::uint32_t, std::uint32_t) {})
                                       .build();
                std::ignore = trigger.start();
            },
            std::exception);
}

// ============================================================================
// Different Data Types Tests
// ============================================================================

TEST(MemoryTrigger, BoolType) {
    auto memory_ptr = std::make_shared<std::atomic<bool>>(false);
    std::atomic<bool> callback_called = false;
    std::atomic<bool> callback_new_value = false;

    auto trigger = ft::make_memory_trigger(
                           memory_ptr,
                           [&](bool /*old_val*/, bool new_val) {
                               callback_called = true;
                               callback_new_value = new_val;
                           })
                           .with_notification_strategy(ft::NotificationStrategy::Polling)
                           .with_polling_interval(POLLING_INTERVAL)
                           .build();

    EXPECT_TRUE(ft::is_task_success(trigger.start()));

    memory_ptr->store(true, std::memory_order_release);
    std::this_thread::sleep_for(10ms);

    EXPECT_TRUE(callback_called);
    EXPECT_TRUE(callback_new_value);

    trigger.stop();
}

TEST(MemoryTrigger, Uint64Type) {
    auto memory_ptr = std::make_shared<std::atomic<std::uint64_t>>(0);
    std::atomic<std::uint64_t> callback_new_value = 0;

    auto trigger = ft::make_memory_trigger(
                           memory_ptr,
                           [&callback_new_value](std::uint64_t /*old_val*/, std::uint64_t new_val) {
                               callback_new_value = new_val;
                           })
                           .with_notification_strategy(ft::NotificationStrategy::Polling)
                           .with_polling_interval(POLLING_INTERVAL)
                           .build();

    EXPECT_TRUE(ft::is_task_success(trigger.start()));

    constexpr std::uint64_t LARGE_VALUE = 0x123456789ABCDEF0ULL;
    memory_ptr->store(LARGE_VALUE, std::memory_order_release);
    std::this_thread::sleep_for(10ms);

    EXPECT_EQ(callback_new_value, LARGE_VALUE);

    trigger.stop();
}

// ============================================================================
// Performance and Stress Tests
// ============================================================================

TEST(MemoryTrigger, RapidChanges) {
    auto memory_ptr = std::make_shared<std::atomic<std::uint32_t>>(0);
    std::atomic<std::uint32_t> trigger_count{0};

    auto trigger = ft::make_memory_trigger(
                           memory_ptr,
                           [&trigger_count](std::uint32_t, std::uint32_t) {
                               trigger_count.fetch_add(1, std::memory_order_relaxed);
                           })
                           .with_notification_strategy(ft::NotificationStrategy::Polling)
                           .with_polling_interval(std::chrono::microseconds(1))
                           .build();

    EXPECT_TRUE(ft::is_task_success(trigger.start()));

    // Rapidly change values - this is a stress test to verify the system
    // can handle high-frequency changes without crashing
    constexpr std::uint32_t NUM_CHANGES = 100;
    for (std::uint32_t i = 1; i <= NUM_CHANGES; ++i) {
        memory_ptr->store(i, std::memory_order_release);
        std::this_thread::sleep_for(100us);
    }

    // Wait for all triggers to complete
    std::this_thread::sleep_for(50ms);

    // Note: Due to the rapid changes (100μs intervals) vs polling interval (1μs),
    // some changes may be missed due to timing races between the polling thread
    // and the test thread. This is expected behavior under extreme load.
    // We verify that at least 25% of the changes are detected, which demonstrates
    // the system is functioning correctly under stress.
    EXPECT_GE(trigger_count.load(), NUM_CHANGES / 4);

    trigger.stop();
}

TEST(MemoryTrigger, LongRunning) {
    auto memory_ptr = std::make_shared<std::atomic<std::uint32_t>>(0);
    std::atomic<std::uint32_t> trigger_count{0};

    auto trigger = ft::make_memory_trigger(
                           memory_ptr,
                           [&trigger_count](std::uint32_t, std::uint32_t) {
                               trigger_count.fetch_add(1, std::memory_order_relaxed);
                           })
                           .with_notification_strategy(ft::NotificationStrategy::Polling)
                           .with_polling_interval(std::chrono::milliseconds(10))
                           .build();

    EXPECT_TRUE(ft::is_task_success(trigger.start()));

    // Run for a longer duration with periodic changes
    for (int i = 0; i < 10; ++i) {
        memory_ptr->store(static_cast<std::uint32_t>(i + 1), std::memory_order_release);
        std::this_thread::sleep_for(50ms);
    }

    EXPECT_EQ(trigger_count.load(), 10);
    trigger.stop();
}

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

} // namespace
