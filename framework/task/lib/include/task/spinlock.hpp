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
 * Cross-platform optimized spinlock implementation
 *
 * Provides high-performance user-space spinlock with try_lock support
 * Optimized for both x86 and ARM architectures with appropriate memory ordering
 * and architecture-specific pause/yield instructions.
 *
 * Features:
 * - Lock-free atomic operations using compare_exchange
 * - Architecture-specific optimizations (x86 pause, ARM yield)
 * - Proper memory ordering for weak memory models
 * - RAII lock guards for exception safety
 * - Minimal memory footprint (single atomic bool)
 * - Real-time friendly (no system calls)
 */

#ifndef FRAMEWORK_TASK_SPINLOCK_HPP
#define FRAMEWORK_TASK_SPINLOCK_HPP

#include <atomic>
#include <thread>

#include "task/time.hpp"

namespace framework::task {

/**
 * High-performance cross-platform spinlock
 *
 * Uses atomic compare_exchange operations with architecture-appropriate
 * memory ordering and CPU pause instructions for optimal performance
 * on both x86 (strong memory model) and ARM (weak memory model).
 */
class Spinlock final {
private:
    std::atomic<bool> locked_{false}; //!< Lock state (false = unlocked, true = locked)

public:
    /**
     * Acquire lock (blocking)
     * Spins until lock is acquired using architecture-optimized pause
     */
    void lock() noexcept {
        // Fast path: try to acquire immediately
        bool expected = false;
        if (locked_.compare_exchange_weak(
                    expected, true, std::memory_order_acquire, std::memory_order_relaxed)) {
            return; // Lock acquired immediately
        }

        // Slow path: spin with backoff
        lock_slow_path();
    }

    /**
     * Try to acquire lock (non-blocking)
     * @return true if lock acquired, false if already locked
     */
    [[nodiscard]] bool try_lock() noexcept {
        bool expected = false;
        return locked_.compare_exchange_weak(
                expected, true, std::memory_order_acquire, std::memory_order_relaxed);
    }

    /**
     * Release lock
     * Uses release memory ordering to ensure proper synchronization
     */
    void unlock() noexcept { locked_.store(false, std::memory_order_release); }

    /**
     * Check if lock is currently held (non-blocking read)
     * @return true if lock is held, false if available
     * @note This is a hint only - lock state may change immediately after check
     */
    [[nodiscard]] bool is_locked() const noexcept {
        return locked_.load(std::memory_order_relaxed);
    }

private:
    /**
     * Slow path for lock acquisition with exponential backoff
     * Uses architecture-specific pause instructions and adaptive spinning
     */
    void lock_slow_path() noexcept {
        static constexpr int MAX_PAUSE_CYCLES = 64;
        int pause_cycles = 1;

        while (true) {
            // Spin-read until lock appears available (reduces cache coherency
            // traffic)
            while (locked_.load(std::memory_order_relaxed)) {
                // Exponential backoff with architecture-specific pause
                for (int i = 0; i < pause_cycles; ++i) {
                    Time::cpu_pause();
                }

                // Exponential backoff up to maximum
                if (pause_cycles < MAX_PAUSE_CYCLES) {
                    pause_cycles *= 2;
                }
            }

            // Try to acquire lock
            bool expected = false;
            if (locked_.compare_exchange_weak(
                        expected, true, std::memory_order_acquire, std::memory_order_relaxed)) {
                return; // Successfully acquired
            }

            // Reset backoff on failed acquisition attempt
            pause_cycles = 1;
        }
    }
};

/**
 * RAII lock guard for Spinlock
 * Automatically acquires lock on construction and releases on destruction
 */
class SpinlockGuard final {
private:
    Spinlock &spinlock_; //!< Reference to the spinlock

public:
    /**
     * Constructor - acquires the lock
     * @param lock Spinlock to acquire
     */
    explicit SpinlockGuard(Spinlock &lock) : spinlock_(lock) { spinlock_.lock(); }

    /// Destructor - releases the lock
    ~SpinlockGuard() { spinlock_.unlock(); }

    // Non-copyable, non-movable
    SpinlockGuard(const SpinlockGuard &) = delete;
    SpinlockGuard &operator=(const SpinlockGuard &) = delete;
    SpinlockGuard(SpinlockGuard &&) = delete;
    SpinlockGuard &operator=(SpinlockGuard &&) = delete;
};

/**
 * RAII try-lock guard for Spinlock
 * Attempts to acquire lock on construction, provides success status
 */
class SpinlockTryGuard final {
private:
    Spinlock &spinlock_; //!< Reference to the spinlock
    bool acquired_{};    //!< Whether lock was successfully acquired

public:
    /**
     * Constructor - attempts to acquire the lock
     * @param lock Spinlock to try to acquire
     */
    explicit SpinlockTryGuard(Spinlock &lock) : spinlock_(lock), acquired_(spinlock_.try_lock()) {}

    /// Destructor - releases the lock if acquired
    ~SpinlockTryGuard() {
        if (acquired_) {
            spinlock_.unlock();
        }
    }

    /**
     * Check if lock was successfully acquired
     * @return true if lock is held, false if acquisition failed
     */
    [[nodiscard]] bool owns_lock() const noexcept { return acquired_; }

    /**
     * Explicit conversion to bool for convenient checking
     * @return true if lock is held, false if acquisition failed
     */
    [[nodiscard]] explicit operator bool() const noexcept { return acquired_; }

    // Non-copyable, non-movable
    SpinlockTryGuard(const SpinlockTryGuard &) = delete;
    SpinlockTryGuard &operator=(const SpinlockTryGuard &) = delete;
    SpinlockTryGuard(SpinlockTryGuard &&) = delete;
    SpinlockTryGuard &operator=(SpinlockTryGuard &&) = delete;
};

} // namespace framework::task

#endif // FRAMEWORK_TASK_SPINLOCK_HPP
