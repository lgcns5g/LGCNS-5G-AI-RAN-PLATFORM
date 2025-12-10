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

#ifndef FRAMEWORK_TASK_TIME_HPP
#define FRAMEWORK_TASK_TIME_HPP

#include <chrono>
#include <cstdint>
#include <thread>

#if defined(__x86_64__) || defined(__i386__)
#include <emmintrin.h>
#include <immintrin.h>
#endif

namespace framework::task {

/// Time type for nanosecond precision timing
using Nanos = std::chrono::nanoseconds;

/**
 * High-precision timing utilities for real-time task scheduling
 *
 * Provides nanosecond precision timing and sleep functionality
 * optimized for low-latency task execution.
 */
class Time final {
public:
    /// System clock time point for time measurements
    using TimePoint = std::chrono::system_clock::time_point;

    /**
     * Get current time in nanoseconds
     *
     * @return Current time in nanoseconds since epoch
     */
    [[nodiscard]] static Nanos now_ns();

    /**
     * Get current time as chrono time point
     *
     * @return Current time point using system_clock
     */
    [[nodiscard]] static TimePoint now();

    /**
     * Sleep until the specified target time
     *
     * Uses hybrid approach: system sleep for longer waits,
     * then busy-wait for precision in the final microseconds.
     *
     * @param target_time_ns Target time in nanoseconds since epoch
     */
    static void sleep_until(Nanos target_time_ns);

    /**
     * Sleep until the specified target time point
     *
     * @param target_time Target time point to sleep until
     */
    static void sleep_until(TimePoint target_time);

    /**
     * Architecture-specific CPU pause/yield instruction
     * Reduces power consumption and improves performance in spin loops
     */
    static void cpu_pause() noexcept {
#if defined(__x86_64__) || defined(__i386__)
        _mm_pause(); // x86 PAUSE instruction
#elif defined(__aarch64__)
        // NOLINTNEXTLINE(hicpp-no-assembler)
        asm volatile("yield" ::: "memory"); // ARM64 yield
#elif defined(__arm__)
        // NOLINTNEXTLINE(hicpp-no-assembler)
        asm volatile("yield" ::: "memory"); // ARM32 yield
#else
        std::this_thread::yield(); // Fallback to OS yield
#endif
    }
};

} // namespace framework::task

#endif // FRAMEWORK_TASK_TIME_HPP
