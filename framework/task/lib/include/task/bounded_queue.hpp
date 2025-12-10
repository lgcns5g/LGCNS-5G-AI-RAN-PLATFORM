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

#ifndef FRAMEWORK_TASK_BOUNDED_QUEUE_HPP
#define FRAMEWORK_TASK_BOUNDED_QUEUE_HPP

#include <array>
#include <atomic>
#include <bit>
#include <concepts>
#include <cstddef>
#include <optional>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace framework::task {

/**
 * Multi-Producer Multi-Consumer bounded queue based on Vyukov's algorithm
 */
// Suppress padding warnings - performance over memory efficiency for lock-free
// structures NOLINTNEXTLINE(clang-analyzer-optin.performance.Padding)
template <typename T> class BoundedQueue final {
private:
    /// Cell structure with sequence number for ABA protection
    struct Cell final {
        std::atomic<std::size_t> sequence; //!< Sequence number for this cell
        std::optional<T> data;             //!< Optional data storage
    };

// Architecture-specific cache line size
#if defined(__x86_64__) || defined(__i386__)
    static constexpr std::size_t CACHE_LINE_SIZE = 64;
#elif defined(__aarch64__) || defined(__arm__)
    static constexpr std::size_t CACHE_LINE_SIZE = 128;
#else
    static constexpr std::size_t CACHE_LINE_SIZE = 64;
#endif

    /// Cache line padding to prevent false sharing
    using CacheLinePad = std::array<char, CACHE_LINE_SIZE>;

    // Memory layout optimized to prevent false sharing
    CacheLinePad pad0_{};       //!< Padding before buffer
    std::vector<Cell> buffer_;  //!< Ring buffer of cells
    std::size_t buffer_mask_{}; //!< Mask for fast modulo (size - 1)
    CacheLinePad pad1_{};       //!< Padding before enqueue_pos
    alignas(CACHE_LINE_SIZE) std::atomic<std::size_t> enqueue_pos_; //!< Producer position
    CacheLinePad pad2_{};                                           //!< Padding before dequeue_pos
    alignas(CACHE_LINE_SIZE) std::atomic<std::size_t> dequeue_pos_; //!< Consumer position
    CacheLinePad pad3_{};                                           //!< Padding after dequeue_pos

public:
    /**
     * Constructor
     * @param buffer_size Queue capacity (must be power of 2)
     */
    explicit BoundedQueue(const std::size_t buffer_size)
            : buffer_(next_power_of_2(buffer_size)), buffer_mask_(next_power_of_2(buffer_size) - 1),
              enqueue_pos_(0), dequeue_pos_(0) {

        const std::size_t actual_size = buffer_mask_ + 1;

        // Initialize buffer sequence numbers
        for (std::size_t i = 0; i != actual_size; ++i) {
            buffer_[i].sequence.store(i, std::memory_order_relaxed);
        }
    }

    /**
     * Enqueue item (multiple producers) - copy version
     * @param data Item to enqueue
     * @return true if successful, false if queue full
     */
    [[nodiscard]] bool enqueue(const T &data) noexcept { return enqueue_impl(data); }

    /**
     * Enqueue item (multiple producers) - move version
     * @param data Item to enqueue (will be moved)
     * @return true if successful, false if queue full
     */
    [[nodiscard]] bool enqueue(T &&data) noexcept { return enqueue_impl(std::move(data)); }

    /**
     * Dequeue item (multiple consumers)
     * @param data Reference to store dequeued item
     * @return true if successful, false if queue empty
     */
    [[nodiscard]] bool dequeue(T &data) noexcept {
        auto result = dequeue_impl();
        if (result.has_value()) {
            if constexpr (std::is_move_constructible_v<T>) {
                data = std::move(*result);
            } else {
                data = *result;
            }
            return true;
        }
        return false;
    }

    /**
     * Try to enqueue with optional return - copy version
     * @param data Item to enqueue
     * @return true if successful, false if queue full
     */
    [[nodiscard]] bool try_push(const T &data) noexcept { return enqueue(data); }

    /**
     * Try to enqueue with optional return - move version
     * @param data Item to enqueue (will be moved)
     * @return true if successful, false if queue full
     */
    [[nodiscard]] bool try_push(T &&data) noexcept { return enqueue(std::move(data)); }

    /**
     * Try to dequeue with optional return
     * @param data Reference to store dequeued item
     * @return true if successful, false if queue empty
     */
    [[nodiscard]] bool try_pop(T &data) noexcept { return dequeue(data); }

    /**
     * Try to dequeue with std::optional return
     * @return Optional containing item if successful, nullopt if empty
     */
    [[nodiscard]] std::optional<T> try_pop() noexcept { return dequeue_impl(); }

    /**
     * Get buffer capacity
     * @return Maximum number of items the queue can hold
     */
    [[nodiscard]] std::size_t capacity() const noexcept { return buffer_mask_ + 1; }

private:
    /**
     * Template implementation for enqueue that works with both copy and move
     * @param data Item to enqueue (forwarded perfectly)
     * @return true if successful, false if queue full
     */
    template <typename U>
        requires std::constructible_from<T, U>
    [[nodiscard]] bool enqueue_impl(U &&data) noexcept {
        Cell *cell{};
        std::size_t pos = enqueue_pos_.load(std::memory_order_relaxed);

        for (;;) {
            cell = &buffer_[pos & buffer_mask_];
            const std::size_t seq = cell->sequence.load(std::memory_order_acquire);
            const std::intptr_t dif =
                    static_cast<std::intptr_t>(seq) - static_cast<std::intptr_t>(pos);

            if (dif == 0) {
                // Cell is ready for this position
                if (enqueue_pos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                    break;
                }
            } else if (dif < 0) {
                // Queue is full
                return false;
            } else {
                // Another thread got here first, reload position
                pos = enqueue_pos_.load(std::memory_order_relaxed);
            }
        }

        // Store data and update sequence (perfect forwarding)
        cell->data.emplace(std::forward<U>(data));
        cell->sequence.store(pos + 1, std::memory_order_release);

        return true;
    }

    /**
     * Internal dequeue implementation that returns std::optional
     * @return Optional containing item if successful, nullopt if empty
     */
    [[nodiscard]] std::optional<T> dequeue_impl() noexcept {
        Cell *cell{};
        std::size_t pos = dequeue_pos_.load(std::memory_order_relaxed);

        for (;;) {
            cell = &buffer_[pos & buffer_mask_];
            const std::size_t seq = cell->sequence.load(std::memory_order_acquire);
            const std::intptr_t dif =
                    static_cast<std::intptr_t>(seq) - static_cast<std::intptr_t>(pos + 1);

            if (dif == 0) {
                // Cell has data for this position
                if (dequeue_pos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                    break;
                }
            } else if (dif < 0) {
                // Queue is empty
                return std::nullopt;
            } else {
                // Another thread got here first, reload position
                pos = dequeue_pos_.load(std::memory_order_relaxed);
            }
        }

        // Move data out of cell and return as optional
        // NOLINTNEXTLINE(misc-const-correctness)
        std::optional<T> result = std::move(cell->data);
        cell->data.reset(); // Explicit destruction
        cell->sequence.store(pos + buffer_mask_ + 1, std::memory_order_release);

        return result;
    }

    /**
     * Calculate next power of 2 >= n
     * @param n Input value
     * @return Smallest power of 2 >= n
     */
    [[nodiscard]] static constexpr std::size_t next_power_of_2(const std::size_t n) noexcept {
        if (n <= 1) {
            return 2; // Minimum queue size
        }

        // Find next power of 2 using bit manipulation
        constexpr auto BITS_IN_SIZE_T = static_cast<unsigned>(sizeof(std::size_t) * 8);
        return std::size_t{1} << (BITS_IN_SIZE_T - static_cast<unsigned>(std::countl_zero(n - 1)));
    }
};
} // namespace framework::task

#endif
