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
 * @file timed_trigger.hpp
 * @brief Task trigger system for event and time-based triggering
 *
 * Provides TimedTrigger for high-precision periodic callbacks
 * with comprehensive timing analysis and jump detection.
 */

#ifndef FRAMEWORK_TASK_TIMED_TRIGGER_HPP
#define FRAMEWORK_TASK_TIMED_TRIGGER_HPP

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <semaphore>
#include <string>
#include <thread>
#include <vector>

#include <wise_enum.h>

#include "task/bounded_queue.hpp"
#include "task/task_log.hpp"
#include "task/task_utils.hpp"
#include "task/time.hpp"

namespace framework::task {

/// Concept defining requirements for TimedTrigger callback function
template <typename F>
concept TimedTriggerCallback = std::invocable<F> && std::same_as<std::invoke_result_t<F>, void> &&
                               std::copy_constructible<F> && std::move_constructible<F>;

/// Parameter struct for TimedTrigger constructor thresholds
struct TriggerThresholds {
    std::chrono::nanoseconds latency_warning_threshold{};   //!< Threshold for triggering latency
                                                            //!< warnings
    std::chrono::nanoseconds jump_detection_threshold{};    //!< Threshold for detecting time jumps
    std::chrono::nanoseconds callback_duration_threshold{}; //!< Threshold for callback execution
                                                            //!< duration warnings
};

// ============================================================================
// TimedTrigger - High-precision periodic triggering with comprehensive stats
// ============================================================================

/// Skip strategy for handling missed trigger windows
enum class SkipStrategy {
    CatchupAll, //!< Catch up all missed triggers (default)
    SkipAhead   //!< Skip missed triggers, jump to next future interval
};

/// Trigger event types for statistics tracking
enum class TriggerEventType {
    TriggerStart,            //!< Trigger tick started
    CallbackStart,           //!< Callback execution started
    CallbackEnd,             //!< Callback execution completed
    LatencyWarning,          //!< Latency exceeded threshold
    CallbackDurationWarning, //!< Callback duration exceeded threshold
    JumpDetected,            //!< Sudden timing jump detected
    TriggersSkipped          //!< Multiple triggers were skipped
};

} // namespace framework::task

// WISE_ENUM_ADAPT must be called at global namespace scope for ADL to work
WISE_ENUM_ADAPT(framework::task::SkipStrategy, CatchupAll, SkipAhead)
WISE_ENUM_ADAPT(
        framework::task::TriggerEventType,
        TriggerStart,
        CallbackStart,
        CallbackEnd,
        LatencyWarning,
        CallbackDurationWarning,
        JumpDetected,
        TriggersSkipped)

namespace framework::task {

/**
 * Statistics event for lock-free communication
 * Used to pass timing data from real-time thread to stats thread
 */
struct TriggerStatsEvent final {
    TriggerEventType type{TriggerEventType::TriggerStart}; //!< Event type
    Nanos timestamp{};                                     //!< Event timestamp
    Nanos scheduled_time{};                                //!< For TriggerStart events
    Nanos callback_duration{};                             //!< For CallbackEnd events
    Nanos latency{};                                       //!< For LatencyWarning events
    Nanos jump_size{};                                     //!< For JumpDetected events
    Nanos inter_trigger_time{};                            //!< For JumpDetected events
    std::uint64_t trigger_count{};                         //!< Sequential trigger number
    std::uint64_t skipped_triggers{};                      //!< For TriggersSkipped events
};

/**
 * Comprehensive trigger execution record for detailed statistics
 * Stores complete execution data for performance analysis
 */
struct TriggerExecutionRecord final {
    Nanos scheduled_time{};                 //!< When trigger should have fired
    Nanos actual_time{};                    //!< When trigger actually fired
    Nanos callback_start_time{};            //!< When callback started
    Nanos callback_end_time{};              //!< When callback completed
    Nanos latency_ns{};                     //!< Scheduled vs actual (jitter)
    Nanos callback_duration_ns{};           //!< Callback execution time
    Nanos inter_trigger_actual{};           //!< Actual time since last trigger
    Nanos inter_trigger_expected{};         //!< Expected time since last trigger
    std::uint64_t trigger_count{};          //!< Sequential trigger number
    std::uint64_t skipped_triggers{};       //!< Number of triggers skipped before this one
    bool exceeded_latency_threshold{false}; //!< Latency warning flag
    bool exceeded_callback_duration_threshold{false}; //!< Callback duration warning flag
    bool jump_detected{false};                        //!< Jump detection flag
};

/**
 * High-precision periodic trigger with comprehensive timing analysis
 *
 * Provides nanosecond-precision periodic callbacks with real-time optimization.
 * Features include jump detection, latency monitoring,
 * and detailed percentile statistics following TaskMonitor patterns.
 */
// NOLINTNEXTLINE(clang-analyzer-optin.performance.Padding)
class TimedTrigger final {
public:
    /// Callback function type for trigger execution
    using CallbackType = std::function<void()>;

    /**
     * Builder pattern for safe TimedTrigger construction
     */
    class Builder final {
    public:
        /**
         * Create builder with required parameters
         * @param[in] callback Function to execute on each trigger (must satisfy
         * TimedTriggerCallback concept)
         * @param[in] interval Trigger interval (any std::chrono::duration type)
         */
        template <TimedTriggerCallback CallbackT, typename Rep, typename Period>
        Builder(CallbackT &&callback, std::chrono::duration<Rep, Period> interval)
                : callback_(std::forward<CallbackT>(callback)),
                  interval_(std::chrono::duration_cast<std::chrono::nanoseconds>(interval)) {}

        /**
         * Pin trigger to specific CPU core
         * @param[in] core CPU core ID
         * @return Reference to builder for chaining
         * @throws std::invalid_argument if core >= hardware_concurrency
         */
        [[nodiscard]] Builder &pin_to_core(std::uint32_t core);

        /**
         * Set real-time thread priority
         * @param[in] priority RT priority (1-99, higher = more priority)
         * @return Reference to builder for chaining
         */
        [[nodiscard]] Builder &with_rt_priority(std::uint32_t priority) noexcept;

        /**
         * Enable or disable statistics collection
         * @param[in] enabled Whether to collect statistics
         * @return Reference to builder for chaining
         */
        [[nodiscard]] Builder &enable_statistics(bool enabled = true) noexcept;

        /**
         * Set custom latency warning threshold
         * @param[in] threshold Custom threshold (0 = auto-calculate, any
         * std::chrono::duration type)
         * @return Reference to builder for chaining
         */
        template <typename Rep, typename Period>
        [[nodiscard]] Builder &
        with_latency_threshold(std::chrono::duration<Rep, Period> threshold) noexcept {
            latency_warning_threshold_ =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(threshold);
            return *this;
        }

        /**
         * Set custom jump detection threshold
         * @param[in] threshold Custom threshold (0 = auto-calculate, any
         * std::chrono::duration type)
         * @return Reference to builder for chaining
         */
        template <typename Rep, typename Period>
        [[nodiscard]] Builder &
        with_jump_threshold(std::chrono::duration<Rep, Period> threshold) noexcept {
            jump_detection_threshold_ =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(threshold);
            return *this;
        }

        /**
         * Set custom callback duration warning threshold
         * @param[in] threshold Custom threshold (0 = auto-calculate, any
         * std::chrono::duration type)
         * @return Reference to builder for chaining
         */
        template <typename Rep, typename Period>
        [[nodiscard]] Builder &
        with_callback_duration_threshold(std::chrono::duration<Rep, Period> threshold) noexcept {
            callback_duration_threshold_ =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(threshold);
            return *this;
        }

        /**
         * Set skip strategy for handling missed trigger windows
         * @param[in] strategy Skip strategy (default: CatchupAll)
         * @return Reference to builder for chaining
         */
        [[nodiscard]] Builder &with_skip_strategy(SkipStrategy strategy) noexcept;

        /**
         * Set CPU core for stats thread pinning
         * @param[in] core_id Core ID to pin stats thread to
         * @return Reference to builder for chaining
         * @throws std::invalid_argument if core_id >= hardware_concurrency()
         */
        [[nodiscard]] Builder &with_stats_core(std::uint32_t core_id);

        /**
         * Set maximum execution records (omit for auto-calculate to 50GB)
         * @param[in] max_records Maximum records to keep
         * @return Reference to builder for chaining
         */
        [[nodiscard]] Builder &with_max_execution_records(std::size_t max_records) noexcept;

        /**
         * Set maximum number of triggers (auto-stop after count reached)
         * @param[in] count Maximum triggers to execute
         * @return Reference to builder for chaining
         */
        [[nodiscard]] Builder &max_triggers(std::size_t count) noexcept;

        /**
         * Build final TimedTrigger with auto-calculated thresholds
         * @return Fully configured TimedTrigger
         */
        [[nodiscard]] TimedTrigger build();

    private:
        CallbackType callback_;
        std::chrono::nanoseconds interval_{};
        std::optional<std::uint32_t> core_id_;
        std::optional<std::uint32_t> thread_priority_;
        bool enable_stats_{true};
        std::chrono::nanoseconds latency_warning_threshold_{};
        std::chrono::nanoseconds jump_detection_threshold_{};
        std::chrono::nanoseconds callback_duration_threshold_{};
        SkipStrategy skip_strategy_{SkipStrategy::CatchupAll};
        std::optional<std::uint32_t> stats_core_id_;
        std::optional<std::size_t>
                max_execution_records_;           //!< Max records (nullopt = auto-calculate)
        std::optional<std::size_t> max_triggers_; //!< Max triggers (auto-stop)
    };

    /**
     * Create builder for periodic trigger
     * @param[in] callback Function to execute on each trigger (must satisfy
     * TimedTriggerCallback concept)
     * @param[in] interval Trigger interval (any std::chrono::duration type)
     * @return Builder instance
     */
    template <TimedTriggerCallback CallbackT, typename Rep, typename Period>
    [[nodiscard]] static Builder
    create(CallbackT &&callback, std::chrono::duration<Rep, Period> interval) {
        return Builder{std::forward<CallbackT>(callback), interval};
    }

    /// Destructor ensures clean shutdown
    ~TimedTrigger();

    // Non-copyable, non-movable
    TimedTrigger(const TimedTrigger &) = delete;
    TimedTrigger &operator=(const TimedTrigger &) = delete;
    TimedTrigger(TimedTrigger &&) = delete;
    TimedTrigger &operator=(TimedTrigger &&) = delete;

    /**
     * Start the timed trigger
     * @param[in] start_time When to start the trigger (default: now)
     * @return Error code indicating success or failure
     * @note TimedTrigger cannot be restarted after stop(). Create new instance
     * instead.
     */
    [[nodiscard]] std::error_code start(Nanos start_time = Time::now_ns());

    /// Stop the timed trigger
    void stop();

    /**
     * Check if trigger is currently running
     * @return True if trigger is running, false otherwise
     */
    [[nodiscard]] bool is_running() const noexcept;

    /**
     * Wait for trigger to complete execution
     *
     * Blocks until the trigger finishes (e.g., max_triggers reached) or until
     * the optional stop flag is set. Automatically joins threads and finalizes
     * statistics.
     *
     * @param[in] stop_flag Optional reference to atomic bool for external stop signal
     *                      (e.g., from signal handler). If not provided, only waits
     *                      for trigger completion.
     *
     * @note When using stop_flag, the caller should use at least
     *       std::memory_order_release when setting it to true to ensure proper
     *       synchronization. This method uses std::memory_order_seq_cst for safety.
     *
     * @throws std::logic_error if max_triggers is not set and stop_flag is not provided
     */
    void wait_for_completion(
            std::optional<std::reference_wrapper<std::atomic_bool>> stop_flag = std::nullopt);

    /**
     * Check if tick thread is pinned to specific core
     * @return True if thread is pinned, false otherwise
     */
    [[nodiscard]] bool is_pinned() const noexcept;

    /**
     * Get core ID for pinned tick threads
     * @return Core ID if pinned, throws if not pinned
     */
    [[nodiscard]] std::uint32_t get_core_id() const;

    /**
     * Check if stats thread is pinned to specific core
     * @return True if stats thread is pinned, false otherwise
     */
    [[nodiscard]] bool is_stats_pinned() const noexcept;

    /**
     * Get core ID for pinned stats threads
     * @return Core ID if stats thread is pinned, throws if not pinned
     */
    [[nodiscard]] std::uint32_t get_stats_core_id() const;

    /**
     * Check if thread uses RT priority
     * @return True if thread has RT priority, false otherwise
     */
    [[nodiscard]] bool has_thread_priority() const noexcept;

    /**
     * Get thread priority level
     * @return Thread priority (1-99), throws if no priority set
     */
    [[nodiscard]] std::uint32_t get_thread_priority() const;

    /**
     * Get trigger interval
     * @return Trigger interval in nanoseconds
     */
    [[nodiscard]] std::chrono::nanoseconds get_interval() const noexcept;

    /**
     * Get maximum trigger count
     * @return Maximum triggers if set, std::nullopt otherwise
     */
    [[nodiscard]] std::optional<std::size_t> max_triggers() const noexcept;

    /// Current trigger tracking for building execution records
    struct CurrentTriggerData {
        Nanos scheduled_time{};                //!< When trigger was scheduled to fire
        Nanos actual_start_time{};             //!< When trigger actually started
        Nanos callback_start_time{};           //!< When callback execution began
        std::uint64_t trigger_count{};         //!< Sequential trigger number
        std::uint64_t skipped_triggers{};      //!< Number of triggers skipped
        bool latency_warning{false};           //!< Latency threshold exceeded flag
        bool callback_duration_warning{false}; //!< Callback duration threshold exceeded flag
        bool jump_detected{false};             //!< Timing jump detected flag
    };

    /// Clear execution statistics
    void clear_stats();

    /// Print comprehensive execution statistics
    void print_summary() const;

    /**
     * Write execution statistics to JSON file
     * @param[in] filename Output file path
     * @param[in] mode Write mode (OVERWRITE or APPEND)
     * @return Error code indicating success or failure
     */
    [[nodiscard]] std::error_code write_stats_to_file(
            const std::string &filename, TraceWriteMode mode = TraceWriteMode::Overwrite) const;

    /**
     * Write execution statistics to Chrome trace format file
     * @param[in] filename Output file path
     * @param[in] mode Write mode (OVERWRITE or APPEND)
     * @return Error code indicating success or failure
     */
    [[nodiscard]] int write_chrome_trace_to_file(
            const std::string &filename, TraceWriteMode mode = TraceWriteMode::Overwrite) const;

private:
    /// Main tick loop with optimized callback execution
    void tick_loop();

    /// Configure tick thread (pinning and priority)
    [[nodiscard]] std::error_code configure_tick_thread();

    /// Stats thread main function
    void stats_thread_function();

    /// Process pending stats events
    std::size_t process_stats_events();

    /// Handle different types of stats events
    void handle_callback_end(const TriggerStatsEvent &event);

    /// Private constructor for builder
    TimedTrigger(
            CallbackType callback,
            std::chrono::nanoseconds interval,
            std::optional<std::uint32_t> core_id,
            std::optional<std::uint32_t> thread_priority,
            bool enable_stats,
            const TriggerThresholds &thresholds,
            SkipStrategy skip_strategy,
            std::optional<std::uint32_t> stats_core_id,
            std::optional<std::size_t> max_execution_records,
            std::optional<std::size_t> max_triggers);

    /// Auto-calculate thresholds based on interval
    void auto_calculate_thresholds();

    CallbackType callback_;

    /// Configuration fields
    Nanos start_time_{};
    std::chrono::nanoseconds interval_{};
    std::optional<std::uint32_t> core_id_;
    std::optional<std::uint32_t> thread_priority_;
    bool enable_stats_{true};
    std::chrono::nanoseconds latency_warning_threshold_{};
    std::chrono::nanoseconds jump_detection_threshold_{};
    std::chrono::nanoseconds callback_duration_threshold_{};
    SkipStrategy skip_strategy_{SkipStrategy::CatchupAll};
    std::optional<std::uint32_t> stats_core_id_;
    std::optional<std::size_t> max_triggers_; //!< Max triggers (auto-stop)

    /// Threading
    std::thread tick_thread_;
    std::thread stats_thread_;
    std::counting_semaphore<2> start_semaphore_{0};
    std::atomic<bool> stop_flag_{false};
    std::atomic<std::uint32_t> threads_ready_{0};
    std::atomic<std::uint64_t> tick_thread_id_{0}; //!< Actual OS thread ID for Chrome tracing

    /// Statistics system
    BoundedQueue<TriggerStatsEvent> stats_queue_;
    std::vector<TriggerExecutionRecord> executions_;
    std::atomic<std::uint64_t> trigger_counter_{0};
    mutable std::mutex stats_mutex_;

    /// Record management
    std::size_t max_execution_records_{0};   //!< Maximum records to keep
    std::uint64_t total_records_created_{0}; //!< Total records created (including truncated)
    std::uint64_t records_truncated_{0};     //!< Count of records removed due to limits

    std::optional<CurrentTriggerData> current_trigger_;
};

} // namespace framework::task

#endif // FRAMEWORK_TASK_TIMED_TRIGGER_HPP
