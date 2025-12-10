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
 * @file memory_trigger.hpp
 * @brief Generic memory location monitoring with callbacks
 *
 * Provides MemoryTrigger for monitoring CPU memory locations and executing
 * callbacks when values change. Default comparator prevents double-triggering
 * by only firing on value changes (old != new). Custom comparators should
 * include transition logic to avoid repeated triggers.
 */

#ifndef FRAMEWORK_TASK_MEMORY_TRIGGER_HPP
#define FRAMEWORK_TASK_MEMORY_TRIGGER_HPP

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <format>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>

#include <quill/LogMacros.h>

#include <wise_enum.h>

#include "log/rt_log_macros.hpp"
#include "task/task_log.hpp"

namespace framework::task {

/// Concept defining requirements for MemoryTrigger template parameter
template <typename T>
concept MemoryTriggerRequirements =
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
        std::is_trivially_copyable_v<T> && (sizeof(T) <= 8) && std::atomic<T>::is_always_lock_free;

/// Concept defining requirements for MemoryTrigger callback function
template <typename F, typename T>
concept MemoryTriggerCallback =
        std::invocable<F, T, T> && std::same_as<std::invoke_result_t<F, T, T>, void>;

/// Notification strategy for memory monitoring
enum class NotificationStrategy {
    ConditionVariable, //!< Use condition variable (requires explicit notify)
    Polling            //!< Polling (when explicit notification not possible)
};

} // namespace framework::task

// WISE_ENUM_ADAPT must be called at global namespace scope for ADL to work
WISE_ENUM_ADAPT(framework::task::NotificationStrategy, ConditionVariable, Polling)

namespace framework::task {

namespace detail {

/// Configure thread priority and CPU affinity
/// @param[in] core_id Optional core ID to pin thread to
/// @param[in] priority Optional thread priority to set
/// @return Error code indicating success or failure
std::error_code configure_thread(
        const std::optional<std::uint32_t> core_id, const std::optional<std::uint32_t> priority);

/// Type-erased monitor loop implementation
/// @param[in] strategy Notification strategy to use
/// @param[in] polling_interval Polling interval for polling mode
/// @param[in] core_id Optional core ID to pin thread to
/// @param[in] priority Optional thread priority to set
/// @param[in,out] stop_flag Atomic flag to signal thread stop
/// @param[in,out] is_running Atomic flag indicating thread running state
/// @param[in,out] cv_mutex Mutex for condition variable
/// @param[in,out] cv Condition variable for notification
/// @param[in] check_condition_fn Function to check if trigger condition is met
/// @param[in] execute_callback_fn Function to execute callback (called outside
/// lock)
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
        const std::function<void()> &execute_callback_fn);

/// Start monitor thread
/// @param[in,out] monitor_thread Thread object to start
/// @param[in,out] stop_flag Atomic flag to signal thread stop
/// @param[in,out] is_running Atomic flag indicating thread running state
/// @param[in] monitor_loop_fn Function to run in the monitor thread
/// @return Error code indicating success or failure
std::error_code start_monitor_thread(
        std::thread &monitor_thread,
        std::atomic<bool> &stop_flag,
        std::atomic<bool> &is_running,
        const std::function<void()> &monitor_loop_fn);

/// Stop monitor thread
/// @param[in,out] monitor_thread Thread object to stop
/// @param[in,out] stop_flag Atomic flag to signal thread stop
/// @param[in,out] is_running Atomic flag indicating thread running state
/// @param[in,out] cv_mutex Mutex for condition variable
/// @param[in,out] cv Condition variable for notification
void stop_monitor_thread(
        std::thread &monitor_thread,
        std::atomic<bool> &stop_flag,
        std::atomic<bool> &is_running,
        std::mutex &cv_mutex,
        std::condition_variable &cv);

} // namespace detail

/**
 * Generic memory-based trigger for monitoring CPU memory locations
 *
 * Monitors a CPU memory location and executes callback when trigger condition
 * is met. Supports any atomic-compatible type. Default comparator prevents
 * double-triggering by only firing on value changes (old != new).
 *
 * @par Custom Comparators:
 * Include transition logic to prevent repeated triggers:
 * @code
 * // Safe: Triggers on transition to READY
 * .with_comparator([](Status old, Status new_val) {
 *     return old != new_val && new_val == READY;
 * })
 * // Unsafe: Triggers repeatedly when value == READY
 * .with_comparator([](Status old, Status new_val) {
 *     return new_val == READY;  // Missing old != new_val check!
 * })
 * @endcode
 *
 * @tparam T Type of memory location to monitor (must satisfy
 * MemoryTriggerRequirements)
 */
template <MemoryTriggerRequirements T> class MemoryTrigger final {
public:
    using CallbackType = std::function<void(
            T old_value,
            T new_value)>; //!< Callback function type for memory change notifications
    using ComparatorType =
            std::function<bool(T old_value, T new_value)>; //!< Comparator function type for
                                                           //!< determining when to trigger
    using MemoryPtr = std::shared_ptr<std::atomic<T>>;     //!< Shared pointer to
                                                           //!< atomic memory location

    /**
     * Builder pattern for configuring MemoryTrigger
     */
    class Builder final {
    public:
        /**
         * Create builder with required parameters
         * @param[in] memory_ptr Shared pointer to atomic memory location to monitor
         * @param[in] callback Function to execute when triggered (must satisfy
         * MemoryTriggerCallback concept)
         */
        template <MemoryTriggerCallback<T> CallbackT>
        Builder(MemoryPtr memory_ptr, CallbackT &&callback)
                : memory_ptr_(std::move(memory_ptr)), callback_(std::forward<CallbackT>(callback)),
                  comparator_([](const T &old_val, const T &new_val) {
                      // The default comparator is any change
                      return old_val != new_val;
                  }) {}

        /**
         * Set custom comparator to determine when to trigger
         *
         * Include transition detection (old != new) to prevent double-triggering.
         * @param[in] comparator Function that returns true if trigger should fire
         * @return Reference to this builder for chaining
         */
        [[nodiscard]] Builder &with_comparator(ComparatorType comparator) noexcept {
            comparator_ = std::move(comparator);
            return *this;
        }

        /**
         * Set notification strategy (default: ConditionVariable)
         * @param[in] strategy Notification strategy to use
         * @return Reference to this builder for chaining
         */
        [[nodiscard]] Builder &
        with_notification_strategy(const NotificationStrategy strategy) noexcept {
            strategy_ = strategy;
            return *this;
        }

        /**
         * Set polling interval for polling mode (default: 100μs)
         * @param[in] interval Polling interval (any std::chrono::duration type)
         * @return Reference to this builder for chaining
         */
        template <typename Rep, typename Period>
        [[nodiscard]] Builder &
        with_polling_interval(std::chrono::duration<Rep, Period> interval) noexcept {
            polling_interval_ = std::chrono::duration_cast<std::chrono::nanoseconds>(interval);
            return *this;
        }

        /**
         * Pin monitoring thread to specific CPU core
         * @param[in] core CPU core ID to pin to
         * @return Reference to this builder for chaining
         * @throws std::invalid_argument if core >= hardware_concurrency
         */
        [[nodiscard]] Builder &pin_to_core(const std::uint32_t core) {
            const auto max_cores = std::thread::hardware_concurrency();
            if (core >= max_cores) {
                const std::string error_msg = std::format(
                        "Invalid core ID {} for MemoryTrigger: system has {} cores (0-{})",
                        core,
                        max_cores,
                        max_cores - 1);
                RT_LOGC_ERROR(TaskLog::TaskTrigger, "{}", error_msg);
                throw std::invalid_argument(error_msg);
            }
            core_id_ = core;
            return *this;
        }

        /**
         * Set thread priority (1-99, higher = more priority)
         * @param[in] priority Thread priority level
         * @return Reference to this builder for chaining
         */
        [[nodiscard]] Builder &with_priority(const std::uint32_t priority) noexcept {
            priority_ = priority;
            return *this;
        }

        /**
         * Build the MemoryTrigger
         * @return Configured MemoryTrigger instance
         */
        [[nodiscard]] MemoryTrigger build();

    private:
        MemoryPtr memory_ptr_;
        CallbackType callback_;
        ComparatorType comparator_;
        NotificationStrategy strategy_{NotificationStrategy::ConditionVariable};
        static constexpr std::chrono::microseconds DEFAULT_POLLING_INTERVAL{100};
        std::chrono::nanoseconds polling_interval_{DEFAULT_POLLING_INTERVAL};
        std::optional<std::uint32_t> core_id_;
        std::optional<std::uint32_t> priority_;
    };

    /**
     * Create builder for memory trigger
     * @param[in] memory_ptr Shared pointer to atomic memory location to monitor
     * @param[in] callback Function to execute when triggered (must satisfy
     * MemoryTriggerCallback concept)
     * @return Builder instance for configuring the trigger
     */
    template <MemoryTriggerCallback<T> CallbackT>
    [[nodiscard]] static Builder create(MemoryPtr memory_ptr, CallbackT &&callback) {
        return Builder{std::move(memory_ptr), std::forward<CallbackT>(callback)};
    }

    /// Destructor ensures clean shutdown
    ~MemoryTrigger();

    // Non-copyable, non-movable
    MemoryTrigger(const MemoryTrigger &) = delete;
    MemoryTrigger &operator=(const MemoryTrigger &) = delete;
    MemoryTrigger(MemoryTrigger &&) = delete;
    MemoryTrigger &operator=(MemoryTrigger &&) = delete;

    /**
     * Start monitoring
     * @return Error code indicating success or failure
     */
    [[nodiscard]] std::error_code start();

    /// Stop monitoring
    void stop();

    /**
     * Check if currently running
     * @return true if monitoring is active, false otherwise
     */
    [[nodiscard]] bool is_running() const noexcept;

    /// Notify trigger of memory change (for ConditionVariable mode only)
    void notify() noexcept;

private:
    /// Private constructor for builder
    MemoryTrigger(
            MemoryPtr memory_ptr,
            CallbackType callback,
            ComparatorType comparator,
            const NotificationStrategy strategy,
            const std::chrono::nanoseconds polling_interval,
            const std::optional<std::uint32_t> core_id,
            const std::optional<std::uint32_t> priority);

    /// Main monitoring loop - supports both condition variable and polling modes
    void monitor_loop();

    /// Check if memory condition is met (without executing callback)
    [[nodiscard]] bool check_condition();

    /// Execute callback with current and last seen values
    void execute_trigger();

    /// Check if trigger condition is met using user-provided comparator
    [[nodiscard]] bool check_trigger_condition(const T old_val, const T new_val) const noexcept;

    // Member variables
    MemoryPtr memory_ptr_;
    CallbackType callback_;
    ComparatorType comparator_;
    T last_seen_value_{};
    T current_value_{}; //!< Current value for callback execution
    NotificationStrategy strategy_;
    std::chrono::nanoseconds polling_interval_;
    std::optional<std::uint32_t> core_id_;
    std::optional<std::uint32_t> priority_;

    // Thread control
    std::thread monitor_thread_;
    std::atomic<bool> stop_flag_{false};
    std::atomic<bool> is_running_{false};

    // Condition variable support
    mutable std::mutex cv_mutex_;
    mutable std::condition_variable cv_;
};

// ============================================================================
// Template Implementation
// ============================================================================

template <MemoryTriggerRequirements T> MemoryTrigger<T> MemoryTrigger<T>::Builder::build() {
    return MemoryTrigger{
            std::move(memory_ptr_),
            std::move(callback_),
            std::move(comparator_),
            strategy_,
            polling_interval_,
            core_id_,
            priority_};
}

template <MemoryTriggerRequirements T>
MemoryTrigger<T>::MemoryTrigger(
        MemoryPtr memory_ptr,
        CallbackType callback,
        ComparatorType comparator,
        const NotificationStrategy strategy,
        const std::chrono::nanoseconds polling_interval,
        const std::optional<std::uint32_t> core_id,
        const std::optional<std::uint32_t> priority)
        : memory_ptr_(std::move(memory_ptr)), callback_(std::move(callback)),
          comparator_(std::move(comparator)), strategy_(strategy),
          polling_interval_(polling_interval), core_id_(core_id), priority_(priority) {
    if (!memory_ptr_) {
        throw std::invalid_argument("MemoryTrigger: memory_ptr cannot be null");
    }
    if (!callback_) {
        throw std::invalid_argument("MemoryTrigger: callback cannot be null");
    }
    // Initialize last seen value after null check
    last_seen_value_ = memory_ptr_->load(std::memory_order_seq_cst);
}

template <MemoryTriggerRequirements T> MemoryTrigger<T>::~MemoryTrigger() { stop(); }

template <MemoryTriggerRequirements T> std::error_code MemoryTrigger<T>::start() {
    return detail::start_monitor_thread(
            monitor_thread_, stop_flag_, is_running_, [this]() { monitor_loop(); });
}

template <MemoryTriggerRequirements T> void MemoryTrigger<T>::stop() {
    detail::stop_monitor_thread(monitor_thread_, stop_flag_, is_running_, cv_mutex_, cv_);
}

template <MemoryTriggerRequirements T> bool MemoryTrigger<T>::is_running() const noexcept {
    return is_running_.load(std::memory_order_acquire);
}

template <MemoryTriggerRequirements T> void MemoryTrigger<T>::notify() noexcept {
    const std::unique_lock<std::mutex> lock(cv_mutex_);
    cv_.notify_one();
}

template <MemoryTriggerRequirements T> void MemoryTrigger<T>::monitor_loop() {
    // Initialize last seen value
    last_seen_value_ = memory_ptr_->load(std::memory_order_seq_cst);

    detail::monitor_loop_impl(
            strategy_,
            polling_interval_,
            core_id_,
            priority_,
            stop_flag_,
            is_running_,
            cv_mutex_,
            cv_,
            [this]() { return check_condition(); },
            [this]() { execute_trigger(); });
}

template <MemoryTriggerRequirements T> bool MemoryTrigger<T>::check_condition() {
    current_value_ = memory_ptr_->load(std::memory_order_seq_cst);
    return check_trigger_condition(last_seen_value_, current_value_);
}

template <MemoryTriggerRequirements T> void MemoryTrigger<T>::execute_trigger() {
    // Execute callback with old and new values (called outside lock)
    try {
        callback_(last_seen_value_, current_value_);
    } catch (const std::exception &e) {
        RT_LOGC_ERROR(TaskLog::TaskTrigger, "MemoryTrigger callback threw exception: {}", e.what());
    } catch (...) {
        RT_LOGC_ERROR(TaskLog::TaskTrigger, "MemoryTrigger callback threw unknown exception");
    }

    // Update last seen value after trigger
    last_seen_value_ = current_value_;
}

template <MemoryTriggerRequirements T>
bool MemoryTrigger<T>::check_trigger_condition(const T old_val, const T new_val) const noexcept {
    try {
        return comparator_(old_val, new_val);
    } catch (const std::exception &e) {
        // Comparator should not throw, but be defensive
        RT_LOGC_ERROR(
                TaskLog::TaskTrigger, "MemoryTrigger comparator threw exception: {}", e.what());
        return false;
    } catch (...) {
        // Catch any other exceptions
        RT_LOGC_ERROR(TaskLog::TaskTrigger, "MemoryTrigger comparator threw unknown exception");
        return false;
    }
}

/// Create MemoryTrigger with automatic type deduction
/// @param[in] memory_ptr Shared pointer to atomic memory location
/// @param[in] callback Function to call when trigger condition is met
/// @return Builder for configuring the MemoryTrigger
template <MemoryTriggerRequirements T, MemoryTriggerCallback<T> CallbackType>
[[nodiscard]] auto
make_memory_trigger(std::shared_ptr<std::atomic<T>> memory_ptr, CallbackType &&callback) {
    return MemoryTrigger<T>::create(std::move(memory_ptr), std::forward<CallbackType>(callback));
}

} // namespace framework::task

#endif // FRAMEWORK_TASK_MEMORY_TRIGGER_HPP
