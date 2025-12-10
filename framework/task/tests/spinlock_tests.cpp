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
 * @file spinlock_tests.cpp
 * @brief Unit tests for Spinlock, SpinlockGuard, and SpinlockTryGuard classes
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "task/spinlock.hpp"
#include "task/time.hpp"

namespace {
namespace ft = framework::task;

/**
 * Basic functionality tests for Spinlock
 */
TEST(Spinlock, BasicLockUnlock) {
    ft::Spinlock spinlock{};

    // Test basic lock/unlock sequence
    EXPECT_FALSE(spinlock.is_locked());
    spinlock.lock();
    EXPECT_TRUE(spinlock.is_locked());
    spinlock.unlock();
    EXPECT_FALSE(spinlock.is_locked());

    // Test multiple lock/unlock cycles
    spinlock.lock();
    EXPECT_TRUE(spinlock.is_locked());
    spinlock.unlock();
    EXPECT_FALSE(spinlock.is_locked());
    spinlock.lock();
    EXPECT_TRUE(spinlock.is_locked());
    spinlock.unlock();
    EXPECT_FALSE(spinlock.is_locked());
}

TEST(Spinlock, TryLock) {
    ft::Spinlock spinlock{};

    // Initially should be able to acquire lock
    EXPECT_TRUE(spinlock.try_lock());
    spinlock.unlock();

    // Multiple try_lock attempts when unlocked
    EXPECT_TRUE(spinlock.try_lock());
    spinlock.unlock();
    EXPECT_TRUE(spinlock.try_lock());
    spinlock.unlock();
}

TEST(Spinlock, TryLockWhenLocked) {
    ft::Spinlock spinlock{};

    // Acquire lock first
    spinlock.lock();

    // try_lock should fail when already locked
    EXPECT_FALSE(spinlock.try_lock());

    spinlock.unlock();

    // Should be able to acquire after unlock
    EXPECT_TRUE(spinlock.try_lock());
    spinlock.unlock();
}

TEST(Spinlock, IsLocked) {
    ft::Spinlock spinlock{};

    // Initially unlocked
    EXPECT_FALSE(spinlock.is_locked());

    // Locked after lock()
    spinlock.lock();
    EXPECT_TRUE(spinlock.is_locked());

    // Unlocked after unlock()
    spinlock.unlock();
    EXPECT_FALSE(spinlock.is_locked());

    // Locked after try_lock() success
    EXPECT_TRUE(spinlock.try_lock());
    EXPECT_TRUE(spinlock.is_locked());
    spinlock.unlock();
    EXPECT_FALSE(spinlock.is_locked());
}

TEST(Spinlock, IsLockedThreadSafety) {
    ft::Spinlock spinlock{};
    std::atomic<bool> thread_ready{false};
    std::atomic<bool> check_result{false};
    std::atomic<bool> lock_observed{false};

    std::thread checker([&thread_ready, &spinlock, &check_result, &lock_observed]() {
        thread_ready.store(true);
        // Busy wait until lock is acquired by main thread
        while (!spinlock.is_locked()) {
            std::this_thread::yield();
        }
        check_result.store(true);
        lock_observed.store(true); // Signal that we observed the lock
    });

    // Wait for checker thread to be ready
    while (!thread_ready.load()) {
        std::this_thread::yield();
    }

    // Acquire lock - checker should detect it
    spinlock.lock();

    // Wait for checker thread to actually observe the locked state
    while (!lock_observed.load()) {
        std::this_thread::yield();
    }

    spinlock.unlock();
    checker.join();

    EXPECT_TRUE(check_result.load());
}

/**
 * Tests for SpinlockGuard RAII wrapper
 */
TEST(SpinlockGuard, BasicRAII) {
    ft::Spinlock spinlock{};

    // Test RAII behavior with scope
    {
        const ft::SpinlockGuard guard(spinlock);
        // Lock should be held here
        EXPECT_TRUE(spinlock.is_locked());
        EXPECT_FALSE(spinlock.try_lock());
    }
    // Lock should be released after guard destruction
    EXPECT_FALSE(spinlock.is_locked());
    EXPECT_TRUE(spinlock.try_lock());
    spinlock.unlock();
}

TEST(SpinlockGuard, NestedScopes) {
    ft::Spinlock spinlock1{};
    ft::Spinlock spinlock2{};

    {
        const ft::SpinlockGuard guard1(spinlock1);
        EXPECT_TRUE(spinlock1.is_locked());
        EXPECT_FALSE(spinlock1.try_lock());

        {
            const ft::SpinlockGuard guard2(spinlock2);
            EXPECT_TRUE(spinlock1.is_locked());
            EXPECT_TRUE(spinlock2.is_locked());
            EXPECT_FALSE(spinlock1.try_lock());
            EXPECT_FALSE(spinlock2.try_lock());
        }

        // spinlock2 should be released, spinlock1 still held
        EXPECT_TRUE(spinlock1.is_locked());
        EXPECT_FALSE(spinlock2.is_locked());
        EXPECT_FALSE(spinlock1.try_lock());
        EXPECT_TRUE(spinlock2.try_lock());
        spinlock2.unlock();
    }

    // Both locks should be released
    EXPECT_FALSE(spinlock1.is_locked());
    EXPECT_FALSE(spinlock2.is_locked());
    EXPECT_TRUE(spinlock1.try_lock());
    EXPECT_TRUE(spinlock2.try_lock());
    spinlock1.unlock();
    spinlock2.unlock();
}

/**
 * Tests for SpinlockTryGuard RAII wrapper
 */
TEST(SpinlockTryGuard, BasicSuccess) {
    ft::Spinlock spinlock{};

    {
        const ft::SpinlockTryGuard guard(spinlock);

        // Should have acquired lock successfully
        EXPECT_TRUE(guard.owns_lock());
        EXPECT_TRUE(static_cast<bool>(guard)); // Test explicit operator bool
        EXPECT_TRUE(spinlock.is_locked());
        EXPECT_FALSE(spinlock.try_lock());
    }

    // Lock should be released after guard destruction
    EXPECT_FALSE(spinlock.is_locked());
    EXPECT_TRUE(spinlock.try_lock());
    spinlock.unlock();
}

TEST(SpinlockTryGuard, BasicFailure) {
    ft::Spinlock spinlock{};

    // First acquire the lock
    spinlock.lock();

    {
        const ft::SpinlockTryGuard guard(spinlock);

        // Should have failed to acquire lock
        EXPECT_FALSE(guard.owns_lock());
        EXPECT_FALSE(static_cast<bool>(guard)); // Test explicit operator bool
        EXPECT_TRUE(spinlock.is_locked());      // Original lock still held
    }

    // Original lock should still be held after guard destruction
    EXPECT_TRUE(spinlock.is_locked());
    spinlock.unlock();
    EXPECT_FALSE(spinlock.is_locked());
}

TEST(SpinlockTryGuard, NestedTryGuards) {
    ft::Spinlock spinlock{};

    {
        const ft::SpinlockTryGuard guard1(spinlock);
        EXPECT_TRUE(guard1.owns_lock());
        EXPECT_TRUE(spinlock.is_locked());

        {
            const ft::SpinlockTryGuard guard2(spinlock);
            // Should fail to acquire already locked spinlock
            EXPECT_FALSE(guard2.owns_lock());
            EXPECT_FALSE(static_cast<bool>(guard2));
            EXPECT_TRUE(spinlock.is_locked()); // Still locked by guard1
        }

        // After guard2 destruction, guard1 still holds lock
        EXPECT_TRUE(guard1.owns_lock());
        EXPECT_TRUE(spinlock.is_locked());
    }

    // After guard1 destruction, lock should be released
    EXPECT_FALSE(spinlock.is_locked());
}

TEST(SpinlockTryGuard, ThreadSafety) {
    ft::Spinlock spinlock{};
    std::atomic<int> successful_acquisitions{0};
    std::atomic<int> failed_acquisitions{0};
    std::atomic<int> threads_ready{0};     // Count threads ready to start
    std::atomic<int> threads_attempted{0}; // Count threads that attempted acquisition
    std::atomic<bool> start_flag{false};
    constexpr int NUM_THREADS = 8;

    std::vector<std::thread> threads{};
    threads.reserve(NUM_THREADS);

    for (int i = 0; i < NUM_THREADS; ++i) {
        threads.emplace_back([&threads_ready,
                              &start_flag,
                              &spinlock,
                              &successful_acquisitions,
                              &failed_acquisitions,
                              &threads_attempted]() {
            // Signal this thread is ready
            threads_ready.fetch_add(1);

            // Wait for start signal
            while (!start_flag.load()) {
                std::this_thread::yield();
            }

            const ft::SpinlockTryGuard guard(spinlock);
            threads_attempted.fetch_add(1);

            if (guard.owns_lock()) {
                successful_acquisitions.fetch_add(1);
                // Hold lock until all threads have attempted acquisition
                while (threads_attempted.load() < NUM_THREADS) {
                    std::this_thread::yield();
                }
                // Hold a bit longer to ensure other threads complete their attempts
                using namespace std::chrono_literals;
                std::this_thread::sleep_for(1ms);
            } else {
                failed_acquisitions.fetch_add(1);
            }
        });
    }

    // Wait for all threads to be ready before starting
    while (threads_ready.load() < NUM_THREADS) {
        std::this_thread::yield();
    }

    // Start all threads simultaneously
    start_flag.store(true);

    for (auto &thread : threads) {
        thread.join();
    }

    // Exactly one thread should have acquired the lock
    EXPECT_EQ(successful_acquisitions.load(), 1);
    EXPECT_EQ(failed_acquisitions.load(), NUM_THREADS - 1);
    EXPECT_FALSE(spinlock.is_locked());
}

/**
 * Thread safety and stress tests
 */
TEST(Spinlock, BasicThreadSafety) {
    ft::Spinlock spinlock{};
    std::uint64_t counter{0};
    constexpr int NUM_THREADS = 4;
    constexpr int INCREMENTS_PER_THREAD = 1000;

    std::vector<std::thread> threads{};
    threads.reserve(NUM_THREADS);

    // Launch threads that increment counter with lock protection
    for (int i = 0; i < NUM_THREADS; ++i) {
        threads.emplace_back([&spinlock, &counter]() {
            for (int j = 0; j < INCREMENTS_PER_THREAD; ++j) {
                const ft::SpinlockGuard guard(spinlock);
                ++counter;
            }
        });
    }

    // Wait for all threads
    for (auto &thread : threads) {
        thread.join();
    }

    // Verify counter reached expected value
    const auto exp = static_cast<std::uint64_t>(NUM_THREADS) * INCREMENTS_PER_THREAD;
    EXPECT_EQ(counter, exp);
    EXPECT_FALSE(spinlock.is_locked());
}

TEST(Spinlock, TryLockContention) {
    ft::Spinlock spinlock{};
    std::atomic<int> successful_tries{0};
    std::atomic<bool> start_flag{false};
    constexpr int NUM_THREADS = 4;

    std::vector<std::thread> threads{};
    threads.reserve(NUM_THREADS);

    // Launch threads that try to acquire lock
    for (int i = 0; i < NUM_THREADS; ++i) {
        threads.emplace_back([&spinlock, &successful_tries, &start_flag]() {
            // Wait for start signal
            while (!start_flag.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }

            // Try to acquire lock
            if (spinlock.try_lock()) {
                successful_tries.fetch_add(1, std::memory_order_relaxed);
                // Hold lock briefly
                constexpr auto BRIEF_HOLD_DURATION_US = 10;
                std::this_thread::sleep_for(std::chrono::microseconds(BRIEF_HOLD_DURATION_US));
                spinlock.unlock();
            }
        });
    }

    // Start all threads simultaneously
    start_flag.store(true, std::memory_order_release);

    // Wait for all threads
    for (auto &thread : threads) {
        thread.join();
    }

    // At least one thread should have acquired the lock
    EXPECT_GE(successful_tries.load(), 1);

    // Lock should be available after all threads finish
    EXPECT_FALSE(spinlock.is_locked());
    EXPECT_TRUE(spinlock.try_lock());
    spinlock.unlock();
}

namespace {
namespace ft = framework::task;

void perform_try_lock_work(ft::Spinlock &spinlock, std::atomic<int> &work_counter) {
    if (spinlock.try_lock()) {
        work_counter.fetch_add(1, std::memory_order_relaxed);
        spinlock.unlock();
    }
}

void perform_regular_lock_work(ft::Spinlock &spinlock, std::atomic<int> &work_counter) {
    spinlock.lock();
    work_counter.fetch_add(1, std::memory_order_relaxed);
    spinlock.unlock();
}

void perform_status_check_work(ft::Spinlock &spinlock, std::atomic<int> &work_counter) {
    if (spinlock.is_locked()) {
        ft::Time::cpu_pause();
    }
    // Still do some work
    perform_try_lock_work(spinlock, work_counter);
}
} // namespace

TEST(Spinlock, HighContentionStress) {
    ft::Spinlock spinlock{};
    std::atomic<int> work_counter{0};
    constexpr int NUM_THREADS = 8;
    constexpr int ITERATIONS = 500;

    std::vector<std::thread> threads{};
    threads.reserve(NUM_THREADS);

    for (int i = 0; i < NUM_THREADS; ++i) {
        threads.emplace_back([&spinlock, &work_counter]() {
            for (int j = 0; j < ITERATIONS; ++j) {
                // Mix of lock(), try_lock(), and is_locked() calls
                const int strategy = j % 3;
                if (strategy == 0) {
                    perform_try_lock_work(spinlock, work_counter);
                } else if (strategy == 1) {
                    perform_regular_lock_work(spinlock, work_counter);
                } else {
                    perform_status_check_work(spinlock, work_counter);
                }
            }
        });
    }

    for (auto &thread : threads) {
        thread.join();
    }

    // Verify some work was done and lock is released
    EXPECT_GT(work_counter.load(), 0);
    EXPECT_FALSE(spinlock.is_locked());
}

TEST(Spinlock, ExponentialBackoffBehavior) {
    ft::Spinlock spinlock{};
    std::atomic<bool> holder_ready{false};
    std::atomic<bool> waiter_started{false};
    std::atomic<bool> waiter_acquired{false};

    // Thread that holds the lock for a while
    std::thread holder([&spinlock, &holder_ready, &waiter_started]() {
        spinlock.lock();
        holder_ready.store(true);

        // Wait for waiter to start spinning
        while (!waiter_started.load()) {
            std::this_thread::yield();
        }

        // Hold lock long enough for exponential backoff to kick in
        constexpr auto BACKOFF_TRIGGER_DURATION_MS = 10;
        std::this_thread::sleep_for(std::chrono::milliseconds(BACKOFF_TRIGGER_DURATION_MS));
        spinlock.unlock();
    });

    // Thread that waits for lock (exercises exponential backoff)
    std::thread waiter([&holder_ready, &waiter_started, &spinlock, &waiter_acquired]() {
        // Wait for holder to acquire lock
        while (!holder_ready.load()) {
            std::this_thread::yield();
        }

        waiter_started.store(true);

        // This should trigger the slow path with exponential backoff
        spinlock.lock();
        waiter_acquired.store(true);
        spinlock.unlock();
    });

    holder.join();
    waiter.join();

    EXPECT_TRUE(waiter_acquired.load());
    EXPECT_FALSE(spinlock.is_locked());
}

} // namespace
