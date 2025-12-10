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

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <system_error>
#include <thread>

#include <quill/LogMacros.h>

#include "log/rt_log_macros.hpp"
#include "task/memory_trigger.hpp"
#include "task/task_errors.hpp"
#include "task/task_log.hpp"
#include "task/task_utils.hpp"

namespace framework::task::detail {

std::error_code configure_thread(
        const std::optional<std::uint32_t> core_id, const std::optional<std::uint32_t> priority) {
    return configure_current_thread(ThreadConfig{core_id, priority});
}

// Type-erased function for the main monitoring loop
void monitor_loop_impl(
        const NotificationStrategy strategy,
        const std::chrono::nanoseconds polling_interval,
        const std::optional<std::uint32_t> core_id,
        const std::optional<std::uint32_t> priority,
        std::atomic<bool> &stop_flag,
        std::atomic<bool> &is_running,
        std::mutex &cv_mutex,
        std::condition_variable &cv,
        const std::function<bool()> &check_condition_fn,
        const std::function<void()> &execute_callback_fn) {

    // Configure thread (pinning and priority)
    const std::error_code config_result = configure_thread(core_id, priority);
    if (config_result) {
        RT_LOGC_WARN(
                TaskLog::TaskTrigger,
                "Failed to configure monitor thread: {}",
                get_error_name(config_result));
    }

    // Signal that we're running
    is_running.store(true, std::memory_order_release);

    if (strategy == NotificationStrategy::ConditionVariable) {
        // Condition variable loop
        std::unique_lock<std::mutex> lock(cv_mutex);
        while (!stop_flag.load(std::memory_order_acquire)) {
            cv.wait(lock, [&stop_flag, &check_condition_fn] {
                return stop_flag.load(std::memory_order_acquire) || check_condition_fn();
            });

            // If woken up by condition (not stop), execute callback outside lock
            if (!stop_flag.load(std::memory_order_acquire)) {
                lock.unlock();
                execute_callback_fn();
                lock.lock();
            }
        }
    } else {
        // Polling loop
        while (!stop_flag.load(std::memory_order_acquire)) {
            if (check_condition_fn()) {
                execute_callback_fn();
            }
            std::this_thread::sleep_for(polling_interval);
        }
    }

    is_running.store(false, std::memory_order_release);
}

std::error_code start_monitor_thread(
        std::thread &monitor_thread,
        std::atomic<bool> &stop_flag,
        std::atomic<bool> &is_running,
        const std::function<void()> &monitor_loop_fn) {
    if (is_running.load(std::memory_order_acquire)) {
        RT_LOGC_WARN(TaskLog::TaskTrigger, "MemoryTrigger is already running");
        return make_error_code(TaskErrc::AlreadyRunning);
    }

    RT_LOGC_INFO(TaskLog::TaskTrigger, "Starting MemoryTrigger");

    stop_flag.store(false, std::memory_order_release);
    monitor_thread = std::thread([monitor_loop_fn] { monitor_loop_fn(); });

    while (!is_running.load(std::memory_order_acquire)) {
        using namespace std::chrono_literals;
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
        std::this_thread::sleep_for(100us);
    }

    return {}; // Success
}

void stop_monitor_thread(
        std::thread &monitor_thread,
        std::atomic<bool> &stop_flag,
        std::atomic<bool> &is_running,
        std::mutex &cv_mutex,
        std::condition_variable &cv) {
    RT_LOGC_INFO(TaskLog::TaskTrigger, "Stopping MemoryTrigger");

    stop_flag.store(true, std::memory_order_release);

    // Notify condition variable to wake up waiting thread
    {
        const std::unique_lock<std::mutex> lock(cv_mutex);
        cv.notify_all();
    }

    if (monitor_thread.joinable()) {
        monitor_thread.join();
    }
    is_running.store(false, std::memory_order_release);
}

} // namespace framework::task::detail
