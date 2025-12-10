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
#include <compare>
#include <thread>

#include "task/time.hpp"

namespace framework::task {

Nanos Time::now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch());
}

Time::TimePoint Time::now() { return std::chrono::system_clock::now(); }

void Time::sleep_until(const Nanos target_time_ns) {
    // Check if we're already past the target time
    const Nanos current_time_ns = now_ns();
    if (current_time_ns >= target_time_ns) {
        return;
    }
    static constexpr Nanos PRECISION_THRESHOLD_NS{100'000}; // 100μs
    static constexpr Nanos BUSY_WAIT_MARGIN_NS{75'000};     // 75μs
    static_assert(
            PRECISION_THRESHOLD_NS > BUSY_WAIT_MARGIN_NS,
            "PRECISION_THRESHOLD_NS must be greater than BUSY_WAIT_MARGIN_NS");

    // Calculate remaining time
    const Nanos wait_time_ns = target_time_ns - current_time_ns;

    // Hybrid approach for accurate timing
    // For longer waits, use system sleep with margin
    if (wait_time_ns > PRECISION_THRESHOLD_NS) {
        // Sleep for slightly less than the wait time to avoid oversleeping
        // Leave margin for final busy-wait to ensure precision
        const auto sleep_duration = wait_time_ns - BUSY_WAIT_MARGIN_NS;
        std::this_thread::sleep_for(sleep_duration);
    }

    // Busy-wait for the remaining time for precision
    // This ensures we don't undershoot or overshoot the target time
    while (now_ns() < target_time_ns) {
        // CPU pause instruction to reduce power consumption during busy-wait
        cpu_pause();
    }
}

void Time::sleep_until(const TimePoint target_time) {
    // Check if we're already past the target time
    const TimePoint current_time = now();
    if (current_time >= target_time) {
        return;
    }

    // Use the nanoseconds version for consistency
    const Nanos target_time_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(target_time.time_since_epoch());

    sleep_until(target_time_ns);
}

} // namespace framework::task
