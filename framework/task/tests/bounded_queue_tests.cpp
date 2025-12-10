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
 * @file bounded_queue_tests.cpp
 * @brief Unit tests for BoundedQueue class
 */

#include <atomic>
#include <cstddef>
#include <format>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "task/bounded_queue.hpp"

namespace {
namespace ft = framework::task;

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

/**
 * Basic functionality tests for BoundedQueue
 */
TEST(BoundedQueue, Construction) {
    // Test power-of-2 sizes
    const ft::BoundedQueue<int> queue2(2);
    EXPECT_EQ(queue2.capacity(), 2);

    const ft::BoundedQueue<int> queue4(4);
    EXPECT_EQ(queue4.capacity(), 4);

    const ft::BoundedQueue<int> queue8(8);
    EXPECT_EQ(queue8.capacity(), 8);

    const ft::BoundedQueue<int> queue1024(1024);
    EXPECT_EQ(queue1024.capacity(), 1024);

    // Test with different types
    const ft::BoundedQueue<std::string> string_queue(16);
    EXPECT_EQ(string_queue.capacity(), 16);

    const ft::BoundedQueue<double> double_queue(32);
    EXPECT_EQ(double_queue.capacity(), 32);
}

TEST(BoundedQueue, BasicEnqueueDequeue) {
    ft::BoundedQueue<int> queue(4);
    int value{};

    // Initially empty
    EXPECT_FALSE(queue.dequeue(value));

    // Add single item
    EXPECT_TRUE(queue.enqueue(42));
    EXPECT_TRUE(queue.dequeue(value));
    EXPECT_EQ(value, 42);

    // Queue should be empty again
    EXPECT_FALSE(queue.dequeue(value));
}

TEST(BoundedQueue, TryPushTryPop) {
    ft::BoundedQueue<std::string> queue(4);
    std::string value{};

    // Test try_push and try_pop
    EXPECT_TRUE(queue.try_push("hello"));
    EXPECT_TRUE(queue.try_push("world"));

    EXPECT_TRUE(queue.try_pop(value));
    EXPECT_EQ(value, "hello");

    EXPECT_TRUE(queue.try_pop(value));
    EXPECT_EQ(value, "world");

    // Should be empty
    EXPECT_FALSE(queue.try_pop(value));
}

TEST(BoundedQueue, OptionalInterface) {
    ft::BoundedQueue<int> queue(4);

    // Empty queue
    auto result = queue.try_pop();
    EXPECT_FALSE(result.has_value());

    // Add items and test optional return
    EXPECT_TRUE(queue.enqueue(10));
    EXPECT_TRUE(queue.enqueue(20));
    EXPECT_TRUE(queue.enqueue(30));

    result = queue.try_pop();
    ASSERT_TRUE(result.has_value());
    // NOLINTBEGIN(bugprone-unchecked-optional-access)
    EXPECT_EQ(result.value(), 10);

    result = queue.try_pop();
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 20);

    result = queue.try_pop();
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 30);

    // Should be empty again
    result = queue.try_pop();
    EXPECT_FALSE(result.has_value());
    // NOLINTEND(bugprone-unchecked-optional-access)
}

TEST(BoundedQueue, FIFOOrdering) {
    ft::BoundedQueue<int> queue(8);

    // Fill with sequence of numbers
    const std::vector<int> input = {1, 2, 3, 4, 5, 6, 7};
    for (const int val : input) {
        EXPECT_TRUE(queue.enqueue(val));
    }

    // Dequeue and verify order
    std::vector<int> output{};
    output.reserve(input.size());

    int value{};
    while (queue.dequeue(value)) {
        output.push_back(value);
    }

    EXPECT_EQ(output, input);
}

TEST(BoundedQueue, CapacityLimits) {
    ft::BoundedQueue<int> queue(4); // Capacity = 4

    // Fill to capacity
    EXPECT_TRUE(queue.enqueue(1));
    EXPECT_TRUE(queue.enqueue(2));
    EXPECT_TRUE(queue.enqueue(3));
    EXPECT_TRUE(queue.enqueue(4));

    // Should be full now
    EXPECT_FALSE(queue.enqueue(5));  // Should fail
    EXPECT_FALSE(queue.try_push(6)); // Should fail

    // Dequeue one item
    int value{};
    EXPECT_TRUE(queue.dequeue(value));
    EXPECT_EQ(value, 1);

    // Should be able to add one more
    EXPECT_TRUE(queue.enqueue(5));

    // Full again
    EXPECT_FALSE(queue.enqueue(6));
}

TEST(BoundedQueue, EmptyQueueBehavior) {
    ft::BoundedQueue<int> queue(4);
    int value{};

    // Multiple attempts on empty queue
    EXPECT_FALSE(queue.dequeue(value));
    EXPECT_FALSE(queue.try_pop(value));

    auto result = queue.try_pop();
    EXPECT_FALSE(result.has_value());

    // Add and remove one item
    EXPECT_TRUE(queue.enqueue(42));
    EXPECT_TRUE(queue.dequeue(value));
    EXPECT_EQ(value, 42);

    // Empty again
    EXPECT_FALSE(queue.dequeue(value));
    EXPECT_FALSE(queue.try_pop(value));
}

TEST(BoundedQueue, SingleElementOperations) {
    ft::BoundedQueue<std::string> queue(2); // Minimal size

    // Add and remove single element repeatedly
    for (int i = 0; i < 10; ++i) {
        const std::string test_str = std::format("test_{}", i);

        EXPECT_TRUE(queue.enqueue(test_str));

        auto result = queue.try_pop();
        ASSERT_TRUE(result.has_value());
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        EXPECT_EQ(result.value(), test_str);
    }
}

TEST(BoundedQueue, WrapAroundBehavior) {
    ft::BoundedQueue<int> queue(4);

    // Fill, drain, and refill to test wrap-around
    for (int cycle = 0; cycle < 3; ++cycle) {
        // Fill queue
        for (int i = 0; i < 4; ++i) {
            EXPECT_TRUE(queue.enqueue(cycle * 10 + i));
        }

        // Drain queue
        for (int i = 0; i < 4; ++i) {
            int value{};
            EXPECT_TRUE(queue.dequeue(value));
            EXPECT_EQ(value, cycle * 10 + i);
        }

        // Should be empty
        int value{};
        EXPECT_FALSE(queue.dequeue(value));
    }
}

TEST(BoundedQueue, MixedOperations) {
    ft::BoundedQueue<int> queue(8);

    // Mix enqueue and dequeue operations
    EXPECT_TRUE(queue.enqueue(1));
    EXPECT_TRUE(queue.enqueue(2));

    int value{};
    EXPECT_TRUE(queue.dequeue(value));
    EXPECT_EQ(value, 1);

    EXPECT_TRUE(queue.enqueue(3));
    EXPECT_TRUE(queue.enqueue(4));

    EXPECT_TRUE(queue.dequeue(value));
    EXPECT_EQ(value, 2);

    EXPECT_TRUE(queue.dequeue(value));
    EXPECT_EQ(value, 3);

    EXPECT_TRUE(queue.enqueue(5));

    EXPECT_TRUE(queue.dequeue(value));
    EXPECT_EQ(value, 4);

    EXPECT_TRUE(queue.dequeue(value));
    EXPECT_EQ(value, 5);

    // Should be empty
    EXPECT_FALSE(queue.dequeue(value));
}

/**
 * Thread safety tests for BoundedQueue
 */
TEST(BoundedQueue, SingleProducerSingleConsumer) {
    ft::BoundedQueue<int> queue(64);
    constexpr int NUM_ITEMS = 1000;
    std::atomic<bool> consumer_done{false};
    std::vector<int> consumed{};
    consumed.reserve(NUM_ITEMS);

    // Producer thread
    std::thread producer([&queue]() {
        for (int i = 0; i < NUM_ITEMS; ++i) {
            while (!queue.enqueue(i)) {
                std::this_thread::yield();
            }
        }
    });

    // Consumer thread
    std::thread consumer([&queue, &consumed, &consumer_done]() {
        int value{};
        for (int i = 0; i < NUM_ITEMS; ++i) {
            while (!queue.dequeue(value)) {
                std::this_thread::yield();
            }
            consumed.push_back(value);
        }
        consumer_done = true;
    });

    producer.join();
    consumer.join();

    EXPECT_TRUE(consumer_done.load());
    EXPECT_EQ(consumed.size(), NUM_ITEMS);

    // Verify ordering
    for (std::size_t i = 0; i < NUM_ITEMS; ++i) {
        EXPECT_EQ(consumed[i], static_cast<int>(i));
    }
}

TEST(BoundedQueue, MultipleProducersConsumers) {
    ft::BoundedQueue<int> queue(128);
    constexpr int NUM_PRODUCERS = 4;
    constexpr int NUM_CONSUMERS = 4;
    constexpr int ITEMS_PER_PRODUCER = 250;
    constexpr int TOTAL_ITEMS = NUM_PRODUCERS * ITEMS_PER_PRODUCER;

    std::atomic<int> items_consumed{0};
    std::atomic<int> producers_finished{0};
    std::vector<std::thread> producers{};
    producers.reserve(NUM_PRODUCERS);
    std::vector<std::thread> consumers{};
    consumers.reserve(NUM_CONSUMERS);

    // Start producers
    for (int p = 0; p < NUM_PRODUCERS; ++p) {
        producers.emplace_back([&queue, &producers_finished, p]() {
            for (int i = 0; i < ITEMS_PER_PRODUCER; ++i) {
                const int value = p * ITEMS_PER_PRODUCER + i;
                while (!queue.enqueue(value)) {
                    std::this_thread::yield();
                }
            }
            producers_finished.fetch_add(1, std::memory_order_relaxed);
        });
    }

    // Start consumers
    for (int c = 0; c < NUM_CONSUMERS; ++c) {
        consumers.emplace_back([&queue, &items_consumed, &producers_finished]() {
            int value{};
            while (items_consumed.load(std::memory_order_relaxed) < TOTAL_ITEMS) {
                if (queue.dequeue(value)) {
                    items_consumed.fetch_add(1, std::memory_order_relaxed);
                } else if (producers_finished.load(std::memory_order_relaxed) == NUM_PRODUCERS) {
                    // All producers done, but queue might still have items
                    if (!queue.dequeue(value)) {
                        break; // Truly empty
                    }
                    items_consumed.fetch_add(1, std::memory_order_relaxed);
                } else {
                    std::this_thread::yield();
                }
            }
        });
    }

    // Wait for all threads
    for (auto &producer : producers) {
        producer.join();
    }
    for (auto &consumer : consumers) {
        consumer.join();
    }

    EXPECT_EQ(producers_finished.load(), NUM_PRODUCERS);
    EXPECT_EQ(items_consumed.load(), TOTAL_ITEMS);

    // Queue should be empty
    int value{};
    EXPECT_FALSE(queue.dequeue(value));
}

TEST(BoundedQueue, HighContentionStressTest) {
    ft::BoundedQueue<int> queue(32); // Smaller queue for more contention
    constexpr int NUM_THREADS = 8;
    constexpr int OPERATIONS_PER_THREAD = 100;

    std::atomic<int> total_enqueued{0};
    std::atomic<int> total_dequeued{0};
    std::vector<std::thread> threads{};
    threads.reserve(NUM_THREADS);

    // Each thread does both enqueue and dequeue operations
    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&queue, &total_enqueued, &total_dequeued, t]() {
            int local_enqueued = 0;
            int local_dequeued = 0;

            for (int i = 0; i < OPERATIONS_PER_THREAD; ++i) {
                // Try to enqueue
                const int value = t * OPERATIONS_PER_THREAD + i;
                if (queue.enqueue(value)) {
                    ++local_enqueued;
                }

                // Try to dequeue
                int dequeued_value{};
                if (queue.dequeue(dequeued_value)) {
                    ++local_dequeued;
                }

                // Small delay to increase contention
                if (i % 10 == 0) {
                    std::this_thread::yield();
                }
            }

            total_enqueued.fetch_add(local_enqueued, std::memory_order_relaxed);
            total_dequeued.fetch_add(local_dequeued, std::memory_order_relaxed);
        });
    }

    for (auto &thread : threads) {
        thread.join();
    }

    // Drain any remaining items
    int remaining = 0;
    int value{};
    while (queue.dequeue(value)) {
        ++remaining;
    }

    // Total items dequeued should equal total items enqueued
    EXPECT_EQ(total_dequeued.load() + remaining, total_enqueued.load());
    EXPECT_GT(total_enqueued.load(), 0); // Should have enqueued something
}

TEST(BoundedQueue, DifferentDataTypes) {
    // Test with custom struct
    struct TestData {
        int id{};
        std::string name;
        double value{};

        bool operator==(const TestData &other) const {
            return id == other.id && name == other.name && value == other.value;
        }
    };

    ft::BoundedQueue<TestData> queue(8);

    const TestData data1{1, "first", 1.5};
    const TestData data2{2, "second", 2.5};
    const TestData data3{3, "third", 3.5};

    // Enqueue custom objects
    EXPECT_TRUE(queue.enqueue(data1));
    EXPECT_TRUE(queue.enqueue(data2));
    EXPECT_TRUE(queue.enqueue(data3));

    // Dequeue and verify
    TestData result{};
    EXPECT_TRUE(queue.dequeue(result));
    EXPECT_EQ(result, data1);

    EXPECT_TRUE(queue.dequeue(result));
    EXPECT_EQ(result, data2);

    EXPECT_TRUE(queue.dequeue(result));
    EXPECT_EQ(result, data3);

    EXPECT_FALSE(queue.dequeue(result));
}

/**
 * Test class for copy-only semantics testing
 */
class CopyOnlyType {
public:
    explicit CopyOnlyType(int val = 0) : value_{val} {}

    // Copy constructor
    CopyOnlyType(const CopyOnlyType &other) : value_{other.value_} { ++copy_count; }

    // Copy assignment
    CopyOnlyType &operator=(const CopyOnlyType &other) {
        if (this != &other) {
            value_ = other.value_;
            ++copy_count;
        }
        return *this;
    }

    // Delete move constructor and assignment
    CopyOnlyType(CopyOnlyType &&) = delete;
    CopyOnlyType &operator=(CopyOnlyType &&) = delete;

    // Destructor
    ~CopyOnlyType() = default;

    [[nodiscard]] int get_value() const noexcept { return value_; }
    [[nodiscard]] static int get_copy_count() noexcept { return copy_count; }
    static void reset_copy_count() noexcept { copy_count = 0; }

private:
    int value_{};
    static inline int copy_count = 0;
};

/**
 * Test class for move semantics testing
 */
class MoveOnlyType {
public:
    explicit MoveOnlyType(int val = 0) : value_{val} {}

    // Move constructor
    MoveOnlyType(MoveOnlyType &&other) noexcept : value_{other.value_} {
        other.value_ = -1; // Mark as moved
    }

    // Move assignment
    MoveOnlyType &operator=(MoveOnlyType &&other) noexcept {
        if (this != &other) {
            value_ = other.value_;
            other.value_ = -1; // Mark as moved
        }
        return *this;
    }

    // Delete copy constructor and assignment
    MoveOnlyType(const MoveOnlyType &) = delete;
    MoveOnlyType &operator=(const MoveOnlyType &) = delete;

    // Destructor
    ~MoveOnlyType() = default;

    [[nodiscard]] int get_value() const noexcept { return value_; }
    [[nodiscard]] bool was_moved() const noexcept { return value_ == -1; }

private:
    int value_{};
};

TEST(BoundedQueue, CopyOnlySemantics) {
    ft::BoundedQueue<CopyOnlyType> queue(4);

    // Reset copy counter
    CopyOnlyType::reset_copy_count();

    // Test enqueue with copy-only type
    const CopyOnlyType item1{42};
    const int initial_copies = CopyOnlyType::get_copy_count();
    EXPECT_TRUE(queue.enqueue(item1));
    EXPECT_EQ(item1.get_value(), 42); // Original should be unchanged
    EXPECT_GT(CopyOnlyType::get_copy_count(),
              initial_copies); // Should have copied

    // Test try_push with copy-only type
    const CopyOnlyType item2{99};
    const int copies_before_push = CopyOnlyType::get_copy_count();
    EXPECT_TRUE(queue.try_push(item2));
    EXPECT_EQ(item2.get_value(), 99); // Original should be unchanged
    EXPECT_GT(CopyOnlyType::get_copy_count(),
              copies_before_push); // Should have copied

    // Test dequeue copies items out (can't move copy-only types)
    CopyOnlyType result1{};
    const int copies_before_dequeue = CopyOnlyType::get_copy_count();
    EXPECT_TRUE(queue.dequeue(result1));
    EXPECT_EQ(result1.get_value(), 42);
    EXPECT_GT(
            CopyOnlyType::get_copy_count(),
            copies_before_dequeue); // Should have copied on dequeue

    CopyOnlyType result2{};
    EXPECT_TRUE(queue.try_pop(result2));
    EXPECT_EQ(result2.get_value(), 99);

    // Queue should be empty now
    CopyOnlyType dummy{};
    EXPECT_FALSE(queue.try_pop(dummy));
}

TEST(BoundedQueue, MoveSemantics) {
    ft::BoundedQueue<MoveOnlyType> queue(4);

    // Test move enqueue
    MoveOnlyType item1{42};
    EXPECT_TRUE(queue.enqueue(std::move(item1)));
    // Don't access item1 after move

    // Test move try_push
    MoveOnlyType item2{99};
    EXPECT_TRUE(queue.try_push(std::move(item2)));
    // Don't access item2 after move

    // Test dequeue moves items out
    MoveOnlyType result1{};
    EXPECT_TRUE(queue.dequeue(result1));
    EXPECT_EQ(result1.get_value(), 42);

    MoveOnlyType result2{};
    EXPECT_TRUE(queue.try_pop(result2));
    EXPECT_EQ(result2.get_value(), 99);

    // Queue should be empty now
    MoveOnlyType dummy{};
    EXPECT_FALSE(queue.try_pop(dummy));
}

TEST(BoundedQueue, UniquePtr) {
    using IntPtr = std::unique_ptr<int>;
    ft::BoundedQueue<IntPtr> queue(4);

    // Test enqueue with unique_ptr
    auto ptr1 = std::make_unique<int>(123);
    EXPECT_TRUE(queue.enqueue(std::move(ptr1)));
    EXPECT_EQ(ptr1, nullptr); // Should be moved from

    // Test try_push with unique_ptr
    auto ptr2 = std::make_unique<int>(456);
    EXPECT_TRUE(queue.try_push(std::move(ptr2)));
    EXPECT_EQ(ptr2, nullptr); // Should be moved from

    // Test dequeue
    IntPtr result1{};
    EXPECT_TRUE(queue.dequeue(result1));
    ASSERT_NE(result1, nullptr);
    EXPECT_EQ(*result1, 123);

    // Test try_pop
    IntPtr result2{};
    EXPECT_TRUE(queue.try_pop(result2));
    ASSERT_NE(result2, nullptr);
    EXPECT_EQ(*result2, 456);

    // Test try_pop with optional return
    auto ptr3 = std::make_unique<int>(789);
    EXPECT_TRUE(queue.try_push(std::move(ptr3)));

    auto optional_result = queue.try_pop();
    // NOLINTBEGIN(bugprone-unchecked-optional-access)
    ASSERT_TRUE(optional_result.has_value());
    ASSERT_NE(optional_result->get(), nullptr);
    EXPECT_EQ(**optional_result, 789);
    // NOLINTEND(bugprone-unchecked-optional-access)
}

TEST(BoundedQueue, MoveSemanticsWithCopyableTypes) {
    // Test that move semantics also work with copyable types
    ft::BoundedQueue<std::string> queue(4);

    // Test move enqueue
    std::string str1 = "hello world";
    const std::string original_str1 = str1;
    EXPECT_TRUE(queue.enqueue(std::move(str1)));
    // Don't access str1 after move

    // Test copy enqueue still works
    const std::string str2 = "copy me";
    EXPECT_TRUE(queue.enqueue(str2));
    EXPECT_EQ(str2, "copy me"); // Original should be unchanged

    // Dequeue and verify
    std::string result1{};
    EXPECT_TRUE(queue.dequeue(result1));
    EXPECT_EQ(result1, original_str1);

    std::string result2{};
    EXPECT_TRUE(queue.dequeue(result2));
    EXPECT_EQ(result2, "copy me");
}

TEST(BoundedQueue, MoveSemanticsConcurrency) {
    using IntPtr = std::unique_ptr<int>;
    ft::BoundedQueue<IntPtr> queue(128);

    const std::size_t num_items = 100;
    std::atomic<std::size_t> enqueued{0};
    std::atomic<std::size_t> dequeued{0};

    // Producer thread
    std::thread producer([&queue, &enqueued]() {
        for (std::size_t i = 0; i < num_items; ++i) {
            auto ptr = std::make_unique<int>(static_cast<int>(i));
            while (!queue.try_push(std::move(ptr))) {
                std::this_thread::yield();
            }
            enqueued.fetch_add(1, std::memory_order_relaxed);
        }
    });

    // Consumer thread
    std::thread consumer([&queue, &dequeued]() {
        std::size_t consumed = 0;
        while (consumed < num_items) {
            IntPtr ptr{};
            if (queue.try_pop(ptr) && ptr) {
                dequeued.fetch_add(1, std::memory_order_relaxed);
                ++consumed;
            } else {
                std::this_thread::yield();
            }
        }
    });

    producer.join();
    consumer.join();

    EXPECT_EQ(enqueued.load(), num_items);
    EXPECT_EQ(dequeued.load(), num_items);

    // Queue should be empty
    IntPtr dummy{};
    EXPECT_FALSE(queue.try_pop(dummy));
}

/**
 * Test class for non-default-constructible types
 */
class NonDefaultConstructible {
public:
    // No default constructor
    NonDefaultConstructible() = delete;

    // Explicit constructor with required parameter
    explicit NonDefaultConstructible(int val) : value_{val} {}

    // Copyable
    NonDefaultConstructible(const NonDefaultConstructible &other) = default;
    NonDefaultConstructible &operator=(const NonDefaultConstructible &other) = default;

    // Movable
    NonDefaultConstructible(NonDefaultConstructible &&other) noexcept = default;
    NonDefaultConstructible &operator=(NonDefaultConstructible &&other) noexcept = default;

    // Destructor
    ~NonDefaultConstructible() = default;

    [[nodiscard]] int get_value() const noexcept { return value_; }

    bool operator==(const NonDefaultConstructible &other) const { return value_ == other.value_; }

private:
    int value_{};
};

TEST(BoundedQueue, NonDefaultConstructibleType) {
    ft::BoundedQueue<NonDefaultConstructible> queue(8);

    // Test enqueue with non-default-constructible type
    const NonDefaultConstructible item1{42};
    EXPECT_TRUE(queue.enqueue(item1));

    // Test move enqueue
    NonDefaultConstructible item2{99};
    // NOLINTNEXTLINE(hicpp-move-const-arg,performance-move-const-arg)
    EXPECT_TRUE(queue.enqueue(std::move(item2)));

    // Test try_push
    const NonDefaultConstructible item3{123};
    EXPECT_TRUE(queue.try_push(item3));

    // Test move try_push
    NonDefaultConstructible item4{456};
    // NOLINTNEXTLINE(hicpp-move-const-arg,performance-move-const-arg)
    EXPECT_TRUE(queue.try_push(std::move(item4)));

    // Test dequeue
    NonDefaultConstructible result1{0};
    EXPECT_TRUE(queue.dequeue(result1));
    EXPECT_EQ(result1.get_value(), 42);

    // Test try_pop with reference
    NonDefaultConstructible result2{0};
    EXPECT_TRUE(queue.try_pop(result2));
    EXPECT_EQ(result2.get_value(), 99);

    // Test try_pop with optional return
    auto optional_result = queue.try_pop();
    // NOLINTBEGIN(bugprone-unchecked-optional-access)
    ASSERT_TRUE(optional_result.has_value());
    EXPECT_EQ(optional_result->get_value(), 123);
    // NOLINTEND(bugprone-unchecked-optional-access)

    // One more dequeue
    NonDefaultConstructible result4{0};
    EXPECT_TRUE(queue.dequeue(result4));
    EXPECT_EQ(result4.get_value(), 456);

    // Should be empty now
    NonDefaultConstructible dummy{0};
    EXPECT_FALSE(queue.dequeue(dummy));

    auto empty_result = queue.try_pop();
    EXPECT_FALSE(empty_result.has_value());
}

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

} // namespace
