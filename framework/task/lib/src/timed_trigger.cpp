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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <compare>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <mutex>
#include <numeric>
#include <optional>
#include <ratio>
#include <semaphore>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include <quill/LogMacros.h>
#include <unistd.h>

#include "log/rt_log_macros.hpp"
#include "task/bounded_queue.hpp"
#include "task/task_errors.hpp"
#include "task/task_log.hpp"
#include "task/task_utils.hpp"
#include "task/time.hpp"
#include "task/timed_trigger.hpp"

namespace framework::task {

namespace {
namespace ft = framework::task;

/// Parameter struct for apply_skip_strategy function
struct SkipStrategyParams {
    SkipStrategy skip_strategy{};
    Nanos start_time{};
    Nanos interval{};
    std::uint64_t trigger_count{};
    bool enable_stats{};
};

/// Parameter struct for check_jump_detection function
struct JumpDetectionParams {
    std::uint64_t trigger_count{};
    Nanos inter_trigger_actual{};
    Nanos inter_trigger_expected{};
    Nanos jump_detection_threshold{};
    Nanos actual_trigger_time{};
};

/// Return type for apply_skip_strategy function
struct SkipStrategyResult {
    std::uint64_t skipped_triggers{};
    std::uint64_t updated_trigger_count{};
};

/**
 * Cleanup old execution records by removing oldest 10%
 * @param[in,out] executions Vector of execution records to clean
 * @param[in] max_records Maximum records allowed
 * @return Number of records removed
 */
template <typename RecordType>
std::size_t
cleanup_old_records_if_needed(std::vector<RecordType> &executions, std::size_t max_records) {
    if (executions.size() >= max_records) {
        // Remove oldest 10% or at least 1 record for small limits
        static constexpr std::size_t REMOVE_PERCENT = 10;
        const std::size_t to_remove = std::max(1UL, max_records / REMOVE_PERCENT);

        executions.erase(
                executions.begin(), executions.begin() + static_cast<std::ptrdiff_t>(to_remove));

        return to_remove;
    }
    return 0;
}

/**
 * Handle triggers skipped event (free function)
 * @param[in] event Skipped triggers event (unused)
 */
void handle_triggers_skipped(const TriggerStatsEvent & /* event */) {
    // This event is for logging purposes and doesn't affect trigger state
    // The skipped_triggers count is already recorded in the TriggerStart event
}

/**
 * Push stats event to queue with logging on failure
 * @param[in] stats_queue Queue to push event to
 * @param[in] event Event to push
 */
template <typename Queue>
void push_stats_event(Queue &stats_queue, const TriggerStatsEvent &event) {
    if (!stats_queue.enqueue(event)) {
        RT_LOGC_WARN(TaskLog::TaskTrigger, "Failed to push stats event for TimedTrigger");
    }
}

/**
 * Apply skip strategy for missed trigger windows
 * @param[in] params Skip strategy parameters
 * @param[in,out] stats_queue Queue for pushing stats events
 * @return SkipStrategyResult containing skipped_triggers and
 * updated_trigger_count
 */
template <typename Queue>
SkipStrategyResult apply_skip_strategy(const SkipStrategyParams &params, Queue &stats_queue) {
    if (params.skip_strategy != SkipStrategy::SkipAhead) {
        return {.skipped_triggers = 0,
                .updated_trigger_count =
                        params.trigger_count}; // No triggers skipped, trigger count unchanged
    }

    std::uint64_t skipped_triggers = 0;
    std::uint64_t updated_trigger_count = params.trigger_count;

    const auto now = Time::now_ns();
    const auto next_absolute_time =
            params.start_time + (params.interval * (params.trigger_count + 1));

    if (next_absolute_time < now) {
        // Calculate how many triggers we're behind
        const auto time_behind = now - next_absolute_time;
        const auto time_behind_ns = time_behind.count();
        const auto interval_ns = static_cast<std::uint64_t>(params.interval.count());
        const auto intervals_behind = (time_behind_ns + interval_ns - 1) / interval_ns; // Round up

        if (intervals_behind > 0) {
            skipped_triggers = intervals_behind;
            updated_trigger_count += skipped_triggers;

            // Log the skip event if stats are enabled
            if (params.enable_stats && skipped_triggers > 0) {
                TriggerStatsEvent skip_event{};
                skip_event.type = TriggerEventType::TriggersSkipped;
                skip_event.timestamp = now;
                skip_event.skipped_triggers = skipped_triggers;
                skip_event.trigger_count = updated_trigger_count;
                push_stats_event(stats_queue, skip_event);

                RT_LOGC_WARN(
                        TaskLog::TaskTrigger,
                        "TimedTrigger skipped {} triggers due to missed windows",
                        skipped_triggers);
            }
        }
    }

    return {.skipped_triggers = skipped_triggers, .updated_trigger_count = updated_trigger_count};
}

/**
 * Create and push trigger start stats event
 * @param[in] actual_trigger_time When trigger actually fired
 * @param[in] scheduled_time When trigger was scheduled to fire
 * @param[in] trigger_count Current trigger count
 * @param[in] skipped_triggers Number of triggers skipped this iteration
 * @param[in,out] stats_queue Queue for pushing stats events
 */
template <typename Queue>
void record_trigger_start_event(
        const Nanos actual_trigger_time,
        const Nanos scheduled_time,
        const std::uint64_t trigger_count,
        const std::uint64_t skipped_triggers,
        Queue &stats_queue) {
    TriggerStatsEvent start_event{};
    start_event.type = TriggerEventType::TriggerStart;
    start_event.timestamp = actual_trigger_time;
    start_event.scheduled_time = scheduled_time;
    start_event.trigger_count = trigger_count;
    start_event.skipped_triggers = skipped_triggers;
    push_stats_event(stats_queue, start_event);
}

/**
 * Create and push callback execution stats events
 * @param[in] callback_start_time When callback started
 * @param[in] callback_end_time When callback ended
 * @param[in] trigger_count Current trigger count
 * @param[in] callback_duration Duration of callback execution
 * @param[in,out] stats_queue Queue for pushing stats events
 */
template <typename Queue>
void record_callback_events(
        const Nanos callback_start_time,
        const Nanos callback_end_time,
        const std::uint64_t trigger_count,
        const Nanos callback_duration,
        Queue &stats_queue) {
    // Record callback execution
    TriggerStatsEvent callback_start_event{};
    callback_start_event.type = TriggerEventType::CallbackStart;
    callback_start_event.timestamp = callback_start_time;
    callback_start_event.trigger_count = trigger_count;
    push_stats_event(stats_queue, callback_start_event);

    TriggerStatsEvent callback_end_event{};
    callback_end_event.type = TriggerEventType::CallbackEnd;
    callback_end_event.timestamp = callback_end_time;
    callback_end_event.callback_duration = callback_duration;
    callback_end_event.trigger_count = trigger_count;
    push_stats_event(stats_queue, callback_end_event);
}

/**
 * Check for excessive latency and create warning event if needed
 * @param[in] latency Calculated latency (actual - scheduled time)
 * @param[in] latency_warning_threshold Threshold for latency warnings
 * @param[in] actual_trigger_time When trigger actually fired
 * @param[in] trigger_count Current trigger count
 * @param[in,out] stats_queue Queue for pushing stats events
 */
template <typename Queue>
void check_latency_warning(
        const Nanos latency,
        const Nanos latency_warning_threshold,
        const Nanos actual_trigger_time,
        const std::uint64_t trigger_count,
        Queue &stats_queue) {
    if (std::abs(latency.count()) > latency_warning_threshold.count()) {
        TriggerStatsEvent warning_event{};
        warning_event.type = TriggerEventType::LatencyWarning;
        warning_event.timestamp = actual_trigger_time;
        warning_event.latency = latency;
        warning_event.trigger_count = trigger_count;
        push_stats_event(stats_queue, warning_event);

        RT_LOGC_WARN(
                TaskLog::TaskTrigger,
                "TimedTrigger latency warning: {:.3f} us (threshold: {:.3f} us)",
                nanos_to_micros_int64(std::abs(latency.count())),
                nanos_to_micros_int64(latency_warning_threshold.count()));
    }
}

/**
 * Check for excessive callback duration and create warning event if needed
 * @param[in] callback_duration Duration of callback execution
 * @param[in] callback_duration_threshold Threshold for duration warnings
 * @param[in] callback_end_time When callback finished
 * @param[in] trigger_count Current trigger count
 * @param[in,out] stats_queue Queue for pushing stats events
 */
template <typename Queue>
void check_callback_duration_warning(
        const Nanos callback_duration,
        const Nanos callback_duration_threshold,
        const Nanos callback_end_time,
        const std::uint64_t trigger_count,
        Queue &stats_queue) {
    if (callback_duration.count() > callback_duration_threshold.count()) {
        TriggerStatsEvent duration_warning_event{};
        duration_warning_event.type = TriggerEventType::CallbackDurationWarning;
        duration_warning_event.timestamp = callback_end_time;
        duration_warning_event.callback_duration = callback_duration;
        duration_warning_event.trigger_count = trigger_count;
        push_stats_event(stats_queue, duration_warning_event);

        RT_LOGC_WARN(
                TaskLog::TaskTrigger,
                "TimedTrigger excessive callback duration: {:.3f} us "
                "(threshold: {:.3f} us)",
                nanos_to_micros_int64(callback_duration.count()),
                nanos_to_micros_int64(callback_duration_threshold.count()));
    }
}

/**
 * Check for sudden timing jumps and create warning event if needed
 * @param[in] params Jump detection parameters
 * @param[in,out] stats_queue Queue for pushing stats events
 */
template <typename Queue>
void check_jump_detection(const JumpDetectionParams &params, Queue &stats_queue) {
    if (params.trigger_count > 1) {
        const Nanos inter_trigger_error = Nanos{
                std::abs((params.inter_trigger_actual - params.inter_trigger_expected).count())};

        if (inter_trigger_error > params.jump_detection_threshold) {
            TriggerStatsEvent jump_event{};
            jump_event.type = TriggerEventType::JumpDetected;
            jump_event.timestamp = params.actual_trigger_time;
            jump_event.jump_size = params.inter_trigger_actual - params.inter_trigger_expected;
            jump_event.inter_trigger_time = params.inter_trigger_actual;
            jump_event.trigger_count = params.trigger_count;
            push_stats_event(stats_queue, jump_event);

            RT_LOGC_WARN(
                    TaskLog::TaskTrigger,
                    "TimedTrigger jump detected: expected {:.3f} us, got {:.3f} us",
                    nanos_to_micros_int64(params.inter_trigger_expected.count()),
                    nanos_to_micros_int64(params.inter_trigger_actual.count()));
        }
    }
}

/**
 * Handle trigger start event by updating current trigger data
 * @param[in,out] current_trigger Current trigger tracking data
 * @param[in] event Trigger start event
 */
void handle_trigger_start_event(
        std::optional<TimedTrigger::CurrentTriggerData> &current_trigger,
        const TriggerStatsEvent &event) {
    // Start building a new execution record
    TimedTrigger::CurrentTriggerData data{};
    data.scheduled_time = event.scheduled_time;
    data.actual_start_time = event.timestamp;
    data.trigger_count = event.trigger_count;
    data.skipped_triggers = event.skipped_triggers;
    current_trigger = data;
}

/**
 * Handle latency warning by marking current trigger
 * @param[in,out] current_trigger Current trigger tracking data
 * @param[in] event Latency warning event (unused)
 */
void handle_latency_warning_event(
        std::optional<TimedTrigger::CurrentTriggerData> &current_trigger,
        const TriggerStatsEvent & /* event */) {
    if (current_trigger.has_value()) {
        current_trigger.value().latency_warning = true;
    }
}

/**
 * Handle callback duration warning by marking current trigger
 * @param[in,out] current_trigger Current trigger tracking data
 * @param[in] event Callback duration warning event (unused)
 */
void handle_callback_duration_warning_event(
        std::optional<TimedTrigger::CurrentTriggerData> &current_trigger,
        const TriggerStatsEvent & /* event */) {
    if (current_trigger.has_value()) {
        current_trigger.value().callback_duration_warning = true;
    }
}

/**
 * Handle jump detected event by marking current trigger
 * @param[in,out] current_trigger Current trigger tracking data
 * @param[in] event Jump detected event (unused)
 */
void handle_jump_detected_event(
        std::optional<TimedTrigger::CurrentTriggerData> &current_trigger,
        const TriggerStatsEvent & /* event */) {
    if (current_trigger.has_value()) {
        current_trigger.value().jump_detected = true;
    }
}

/**
 * Print latency statistics from execution records
 * @param[in] executions Vector of execution records
 */
void print_latency_statistics_impl(const std::vector<TriggerExecutionRecord> &executions) {
    std::vector<double> latency_values;
    std::size_t latency_warnings = 0;

    for (const auto &record : executions) {
        latency_values.push_back(nanos_to_micros_int64(record.latency_ns.count()));
        if (record.exceeded_latency_threshold) {
            latency_warnings++;
        }
    }

    if (latency_values.empty()) {
        return;
    }

    std::sort(latency_values.begin(), latency_values.end());

    const double avg = std::accumulate(latency_values.begin(), latency_values.end(), 0.0) /
                       static_cast<double>(latency_values.size());
    const double min_val = latency_values.front();
    const double max_val = latency_values.back();
    const double median = ft::calculate_percentile(latency_values, 0.5);
    const double p95 = ft::calculate_percentile(latency_values, 0.95);
    const double p99 = ft::calculate_percentile(latency_values, 0.99);
    const double std_dev = ft::calculate_standard_deviation(latency_values, avg);

    RT_LOGC_INFO(TaskLog::TaskTrigger, "=== Latency Statistics ===");
    RT_LOGC_INFO(
            TaskLog::TaskTrigger,
            "Min: {:.3f} us, Max: {:.3f} us, Avg: {:.3f} us",
            min_val,
            max_val,
            avg);
    RT_LOGC_INFO(
            TaskLog::TaskTrigger,
            "Median: {:.3f} us, 95th: {:.3f} us, 99th: {:.3f} us",
            median,
            p95,
            p99);
    RT_LOGC_INFO(TaskLog::TaskTrigger, "Std: {:.3f} us", std_dev);
    RT_LOGC_INFO(
            TaskLog::TaskTrigger,
            "Latency warnings: {} / {} triggers",
            latency_warnings,
            executions.size());
}

/**
 * Print callback duration statistics from execution records
 * @param[in] executions Vector of execution records
 */
void print_callback_duration_statistics_impl(
        const std::vector<TriggerExecutionRecord> &executions) {
    std::vector<double> duration_values;
    duration_values.reserve(executions.size());
    std::size_t duration_warnings = 0;

    for (const auto &record : executions) {
        duration_values.push_back(nanos_to_micros_int64(record.callback_duration_ns.count()));
        if (record.exceeded_callback_duration_threshold) {
            duration_warnings++;
        }
    }

    if (duration_values.empty()) {
        return;
    }

    std::sort(duration_values.begin(), duration_values.end());

    const double avg = std::accumulate(duration_values.begin(), duration_values.end(), 0.0) /
                       static_cast<double>(duration_values.size());
    const double min_val = duration_values.front();
    const double max_val = duration_values.back();
    const double median = ft::calculate_percentile(duration_values, 0.5);
    const double p95 = ft::calculate_percentile(duration_values, 0.95);
    const double p99 = ft::calculate_percentile(duration_values, 0.99);
    const double std_dev = ft::calculate_standard_deviation(duration_values, avg);

    RT_LOGC_INFO(TaskLog::TaskTrigger, "=== Callback Duration Statistics ===");
    RT_LOGC_INFO(
            TaskLog::TaskTrigger,
            "Min: {:.3f} us, Max: {:.3f} us, Avg: {:.3f} us",
            min_val,
            max_val,
            avg);
    RT_LOGC_INFO(
            TaskLog::TaskTrigger,
            "Median: {:.3f} us, 95th: {:.3f} us, 99th: {:.3f} us",
            median,
            p95,
            p99);
    RT_LOGC_INFO(TaskLog::TaskTrigger, "Std: {:.3f} us", std_dev);
    RT_LOGC_INFO(
            TaskLog::TaskTrigger,
            "Duration warnings: {} / {} triggers",
            duration_warnings,
            executions.size());
}

/**
 * Print jump statistics from execution records
 * @param[in] executions Vector of execution records
 */
void print_jump_statistics_impl(const std::vector<TriggerExecutionRecord> &executions) {
    std::size_t jump_count = 0;
    Nanos max_jump{0};

    for (const auto &record : executions) {
        if (record.jump_detected) {
            jump_count++;
            const Nanos jump_size = Nanos{std::abs(
                    (record.inter_trigger_actual - record.inter_trigger_expected).count())};
            max_jump = std::max(max_jump, jump_size);
        }
    }

    RT_LOGC_INFO(TaskLog::TaskTrigger, "=== Jump Detection Statistics ===");
    RT_LOGC_INFO(
            TaskLog::TaskTrigger,
            "Jumps detected: {} / {} triggers",
            jump_count,
            executions.size());
    if (jump_count > 0) {
        RT_LOGC_INFO(
                TaskLog::TaskTrigger,
                "Maximum jump: {:.3f} us",
                nanos_to_micros_int64(max_jump.count()));
    }
}

/**
 * Print skip statistics from execution records
 * @param[in] executions Vector of execution records
 * @param[in] skip_strategy Skip strategy used by the trigger
 */
void print_skip_statistics_impl(
        const std::vector<TriggerExecutionRecord> &executions, const SkipStrategy skip_strategy) {
    std::uint64_t total_skipped = 0;
    std::size_t triggers_with_skips = 0;
    std::uint64_t max_skipped_in_one_event = 0;

    for (const auto &record : executions) {
        if (record.skipped_triggers > 0) {
            total_skipped += record.skipped_triggers;
            triggers_with_skips++;
            max_skipped_in_one_event = std::max(max_skipped_in_one_event, record.skipped_triggers);
        }
    }

    RT_LOGC_INFO(TaskLog::TaskTrigger, "=== Skip Statistics ===");
    RT_LOGC_INFO(
            TaskLog::TaskTrigger,
            "Skip strategy: {}",
            skip_strategy == SkipStrategy::CatchupAll ? "CatchupAll" : "SkipAhead");
    RT_LOGC_INFO(TaskLog::TaskTrigger, "Total triggers skipped: {}", total_skipped);
    RT_LOGC_INFO(
            TaskLog::TaskTrigger,
            "Triggers with skips: {} / {} triggers",
            triggers_with_skips,
            executions.size());
    if (max_skipped_in_one_event > 0) {
        RT_LOGC_INFO(
                TaskLog::TaskTrigger, "Maximum skipped in one event: {}", max_skipped_in_one_event);
    }
}

} // anonymous namespace

// ============================================================================
// TimedTrigger Auto-Calculate Thresholds
// ============================================================================

void TimedTrigger::auto_calculate_thresholds() {
    if (latency_warning_threshold_.count() == 0) {
        latency_warning_threshold_ = interval_ / 2; // 50% of interval
    }

    if (jump_detection_threshold_.count() == 0) {
        jump_detection_threshold_ = interval_ * 2; // 200% of interval
    }

    if (callback_duration_threshold_.count() == 0) {
        callback_duration_threshold_ = interval_ / 4; // 25% of interval
    }
}

// TimedTrigger Builder Implementation

TimedTrigger::Builder &TimedTrigger::Builder::pin_to_core(const std::uint32_t core) {
    const auto max_cores = std::thread::hardware_concurrency();
    if (core >= max_cores) {
        const std::string error_msg = std::format(
                "Invalid core ID {} for TimedTrigger: system has {} cores (0-{})",
                core,
                max_cores,
                max_cores - 1);
        RT_LOGC_ERROR(TaskLog::TaskTrigger, "{}", error_msg);
        throw std::invalid_argument(error_msg);
    }
    core_id_ = core;
    return *this;
}

TimedTrigger::Builder &TimedTrigger::Builder::with_rt_priority(std::uint32_t priority) noexcept {
    thread_priority_ = priority;
    return *this;
}

TimedTrigger::Builder &TimedTrigger::Builder::enable_statistics(bool enabled) noexcept {
    enable_stats_ = enabled;
    return *this;
}

TimedTrigger::Builder &TimedTrigger::Builder::with_skip_strategy(SkipStrategy strategy) noexcept {
    skip_strategy_ = strategy;
    return *this;
}

TimedTrigger::Builder &TimedTrigger::Builder::with_stats_core(const std::uint32_t core_id) {
    const auto max_cores = std::thread::hardware_concurrency();
    if (core_id >= max_cores) {
        const std::string error_msg = std::format(
                "Invalid stats core ID {} for TimedTrigger: system has {} cores (0-{})",
                core_id,
                max_cores,
                max_cores - 1);
        RT_LOGC_ERROR(TaskLog::TaskTrigger, "{}", error_msg);
        throw std::invalid_argument(error_msg);
    }
    stats_core_id_ = core_id;
    return *this;
}

TimedTrigger::Builder &
TimedTrigger::Builder::with_max_execution_records(const std::size_t max_records) noexcept {
    max_execution_records_ = max_records;
    return *this;
}

TimedTrigger::Builder &TimedTrigger::Builder::max_triggers(const std::size_t count) noexcept {
    max_triggers_ = count;
    return *this;
}

TimedTrigger TimedTrigger::Builder::build() {
    const ft::TriggerThresholds thresholds{
            .latency_warning_threshold = latency_warning_threshold_,
            .jump_detection_threshold = jump_detection_threshold_,
            .callback_duration_threshold = callback_duration_threshold_};
    return {std::move(callback_),
            interval_,
            core_id_,
            thread_priority_,
            enable_stats_,
            thresholds,
            skip_strategy_,
            stats_core_id_,
            max_execution_records_,
            max_triggers_};
}

// ============================================================================
// TimedTrigger Implementation
// ============================================================================

namespace {
namespace ft = framework::task;

constexpr std::uint64_t BYTES_PER_KB = 1024ULL;
constexpr std::uint64_t DEFAULT_MAX_MEMORY_GB = 50ULL;
constexpr auto GB = BYTES_PER_KB * BYTES_PER_KB * BYTES_PER_KB;

} // namespace

TimedTrigger::TimedTrigger(
        CallbackType callback,
        std::chrono::nanoseconds interval,
        std::optional<std::uint32_t> core_id,
        std::optional<std::uint32_t> thread_priority,
        bool enable_stats,
        const ft::TriggerThresholds &thresholds,
        SkipStrategy skip_strategy,
        std::optional<std::uint32_t> stats_core_id,
        std::optional<std::size_t> max_execution_records,
        std::optional<std::size_t> max_triggers)
        : callback_(std::move(callback)), interval_(interval), core_id_(core_id),
          thread_priority_(thread_priority), enable_stats_(enable_stats),
          latency_warning_threshold_(thresholds.latency_warning_threshold),
          jump_detection_threshold_(thresholds.jump_detection_threshold),
          callback_duration_threshold_(thresholds.callback_duration_threshold),
          skip_strategy_(skip_strategy), stats_core_id_(stats_core_id), max_triggers_(max_triggers),
          stats_queue_(BYTES_PER_KB),
          max_execution_records_(max_execution_records.value_or(
                  calculate_max_records_for_bytes<TriggerExecutionRecord>(
                          DEFAULT_MAX_MEMORY_GB * GB))) {
    // Log record configuration
    RT_LOGC_DEBUG(
            TaskLog::TaskTrigger,
            "TimedTrigger configured for max {} execution records (~{:.1f} GB)",
            max_execution_records_,
            static_cast<double>(max_execution_records_ * sizeof(TriggerExecutionRecord)) /
                    static_cast<double>(GB));

    // Ensure thresholds are calculated
    auto_calculate_thresholds();

    // Create threads in constructor for reduced startup latency
    stop_flag_.store(false, std::memory_order_release);
    threads_ready_.store(0, std::memory_order_release);

    // Start stats thread first
    if (enable_stats_) {
        stats_thread_ = std::thread([this] { stats_thread_function(); });
    }

    // Start tick thread
    tick_thread_ = std::thread([this] { tick_loop(); });

    // Configure tick thread
    const std::error_code config_result = configure_tick_thread();
    if (config_result) {
        RT_LOGC_ERROR(
                TaskLog::TaskTrigger,
                "Failed to configure tick thread for TimedTrigger: {}",
                get_error_name(config_result));
        // Clean up threads
        stop_flag_.store(true, std::memory_order_release);
        if (tick_thread_.joinable()) {
            tick_thread_.join();
        }
        if (stats_thread_.joinable()) {
            stats_thread_.join();
        }
        throw std::runtime_error("Failed to configure TimedTrigger tick thread");
    }

    // Wait for threads to be ready
    static constexpr std::chrono::microseconds THREAD_READY_WAIT_INTERVAL{10};
    const auto expected_threads = enable_stats_ ? 2U : 1U;
    while (threads_ready_.load(std::memory_order_acquire) < expected_threads) {
        std::this_thread::sleep_for(THREAD_READY_WAIT_INTERVAL);
    }

    // Reset counter - threads ready, but trigger not started yet
    threads_ready_.store(0, std::memory_order_release);

    RT_LOGC_DEBUG(TaskLog::TaskTrigger, "All {} TimedTrigger threads ready", expected_threads);
}

TimedTrigger::~TimedTrigger() {
    try {
        stop();
    } catch (...) {
        // Destructor must not throw - log any exceptions
        RT_LOGC_ERROR(TaskLog::TaskTrigger, "Exception in TimedTrigger destructor");
    }
}

bool TimedTrigger::is_pinned() const noexcept { return core_id_.has_value(); }

std::uint32_t TimedTrigger::get_core_id() const { return core_id_.value_or(0); }

bool TimedTrigger::is_stats_pinned() const noexcept { return stats_core_id_.has_value(); }

std::uint32_t TimedTrigger::get_stats_core_id() const { return stats_core_id_.value_or(0); }

bool TimedTrigger::has_thread_priority() const noexcept { return thread_priority_.has_value(); }

std::uint32_t TimedTrigger::get_thread_priority() const { return thread_priority_.value_or(0); }

std::chrono::nanoseconds TimedTrigger::get_interval() const noexcept { return interval_; }

std::optional<std::size_t> TimedTrigger::max_triggers() const noexcept { return max_triggers_; }

std::error_code TimedTrigger::start(Nanos start_time) {
    if (threads_ready_.load(std::memory_order_acquire) > 0) {
        RT_LOGC_WARN(TaskLog::TaskTrigger, "TimedTrigger is already running");
        return make_error_code(TaskErrc::AlreadyRunning);
    }

    // Check if threads are dead (after previous stop) - restart not supported
    if (!tick_thread_.joinable() || (enable_stats_ && !stats_thread_.joinable())) {
        RT_LOGC_ERROR(
                TaskLog::TaskTrigger,
                "TimedTrigger cannot be restarted after stop() - create new instance");
        return make_error_code(TaskErrc::NotStarted);
    }

    RT_LOGC_INFO(
            TaskLog::TaskTrigger,
            "Starting TimedTrigger with {:.3f} us interval",
            nanos_to_micros_int64(interval_.count()));

    start_time_ = start_time;

    // Mark as running and unblock threads
    const auto expected_threads = enable_stats_ ? 2U : 1U;
    threads_ready_.store(expected_threads, std::memory_order_release);
    start_semaphore_.release(); // Unblock tick thread
    if (enable_stats_) {
        start_semaphore_.release(); // Unblock stats thread
    }
    RT_LOGC_INFO(TaskLog::TaskTrigger, "TimedTrigger started successfully");

    return {}; // Success
}

void TimedTrigger::stop() {
    RT_LOGC_INFO(TaskLog::TaskTrigger, "Stopping TimedTrigger");

    stop_flag_.store(true, std::memory_order_release);

    // Release start semaphores to wake up waiting threads
    start_semaphore_.release(); // Wake up tick thread if waiting to start
    if (enable_stats_) {
        start_semaphore_.release(); // Wake up stats thread if waiting to start
    }

    // Stop threads
    if (tick_thread_.joinable()) {
        tick_thread_.join();
    }

    if (stats_thread_.joinable()) {
        stats_thread_.join();
    }

    // Reset state for potential restart
    threads_ready_.store(0, std::memory_order_release);
    stop_flag_.store(false, std::memory_order_release);

    // Process any remaining stats events
    if (enable_stats_) {
        process_stats_events();
    }

    RT_LOGC_INFO(TaskLog::TaskTrigger, "TimedTrigger stopped");
}

bool TimedTrigger::is_running() const noexcept {
    return threads_ready_.load(std::memory_order_acquire) > 0 &&
           !stop_flag_.load(std::memory_order_acquire);
}

void TimedTrigger::wait_for_completion(
        const std::optional<std::reference_wrapper<std::atomic_bool>> stop_flag) {
    if (!max_triggers_.has_value() && !stop_flag.has_value()) {
        throw std::logic_error("wait_for_completion() requires either max_triggers to be set or "
                               "a stop_flag. Without either, the trigger runs indefinitely "
                               "and this would block forever.");
    }

    // Cache stop_flag presence to avoid repeated has_value() checks
    const bool has_stop_flag = stop_flag.has_value();

    // Wait until trigger completes execution or stop flag is set
    static constexpr std::chrono::milliseconds WAIT_INTERVAL{100};
    while (is_running()) {
        if (has_stop_flag && stop_flag->get().load()) {
            break;
        }
        std::this_thread::sleep_for(WAIT_INTERVAL);
    }

    // Join threads and finalize stats
    stop();
}

void TimedTrigger::clear_stats() {
    const std::lock_guard<std::mutex> lock(stats_mutex_);
    executions_.clear();
    trigger_counter_.store(0, std::memory_order_release);
}

void TimedTrigger::tick_loop() {
    RT_LOGC_DEBUG(TaskLog::TaskTrigger, "TimedTrigger tick thread started");

    // Capture actual OS thread ID for Chrome tracing
    tick_thread_id_.store(
            std::hash<std::thread::id>{}(std::this_thread::get_id()), std::memory_order_release);

    // Signal that this thread is ready
    threads_ready_.fetch_add(1, std::memory_order_release);

    // Block until start() is called
    start_semaphore_.acquire();

    if (stop_flag_.load(std::memory_order_acquire)) {
        return; // Exit if stopped before started
    }

    // Initialize next trigger time (loop will handle the initial sleep)
    Nanos next_trigger_time = start_time_;
    std::uint64_t trigger_count = 0;
    Nanos last_actual_trigger_time = start_time_;

    // Note: With absolute scheduling, drift tracking is not needed
    // Each trigger targets its absolute scheduled time independently

    while (!stop_flag_.load(std::memory_order_acquire)) {
        // ===== CRITICAL TIMING SECTION - MINIMIZE LATENCY =====

        // 1. Sleep until target time
        Time::sleep_until(next_trigger_time);

        // 2. IMMEDIATELY capture wake-up time and execute callback
        const Nanos actual_trigger_time = Time::now_ns();

        // 3. Execute callback immediately - minimal critical path
        const Nanos callback_start_time = Time::now_ns();
        if (callback_) {
            callback_();
        }

        // 4. Capture callback completion time
        const Nanos callback_end_time = Time::now_ns();

        // ===== END CRITICAL TIMING SECTION =====

        // Schedule next trigger (deterministic calculation, timing doesn't matter)
        const Nanos current_scheduled_time = next_trigger_time; // Save for calculations

        // Apply skip strategy for missed trigger windows
        const SkipStrategyParams skip_params{
                skip_strategy_, start_time_, interval_, trigger_count, enable_stats_};
        const auto [skipped_triggers, updated_trigger_count] =
                apply_skip_strategy(skip_params, stats_queue_);
        trigger_count = updated_trigger_count;

        next_trigger_time = start_time_ + (interval_ * (trigger_count + 1));

        // Now we can do all our analysis and logging

        // Calculate all timing metrics
        const Nanos latency = actual_trigger_time - current_scheduled_time;

        const Nanos callback_duration = callback_end_time - callback_start_time;

        // Now increment trigger count for next iteration
        trigger_count++;
        trigger_counter_.store(trigger_count, std::memory_order_release);

        // Calculate inter-trigger timing for jump detection
        const Nanos inter_trigger_actual = actual_trigger_time - last_actual_trigger_time;
        const Nanos inter_trigger_expected = interval_;

        // Log all events after callback completion
        if (enable_stats_) {
            // Record trigger start event
            record_trigger_start_event(
                    actual_trigger_time,
                    current_scheduled_time,
                    trigger_count,
                    skipped_triggers,
                    stats_queue_);

            // Record callback execution events
            record_callback_events(
                    callback_start_time,
                    callback_end_time,
                    trigger_count,
                    callback_duration,
                    stats_queue_);

            // Check for excessive latency
            check_latency_warning(
                    latency,
                    latency_warning_threshold_,
                    actual_trigger_time,
                    trigger_count,
                    stats_queue_);

            // Check for excessive callback duration
            check_callback_duration_warning(
                    callback_duration,
                    callback_duration_threshold_,
                    callback_end_time,
                    trigger_count,
                    stats_queue_);

            // Check for sudden jumps
            const JumpDetectionParams jump_params{
                    trigger_count,
                    inter_trigger_actual,
                    inter_trigger_expected,
                    jump_detection_threshold_,
                    actual_trigger_time};
            check_jump_detection(jump_params, stats_queue_);
        }

        // Prepare for next iteration (do this after logging to minimize next-loop
        // latency)
        last_actual_trigger_time = actual_trigger_time;

        // Check if max_triggers limit reached (after stats are recorded)
        if (max_triggers_.has_value() && trigger_count >= max_triggers_.value()) {
            break; // Stop after reaching max triggers
        }

        // Loop back to sleep_until - all logging/analysis done
    }

    RT_LOGC_DEBUG(TaskLog::TaskTrigger, "TimedTrigger tick thread stopped");

    // Signal stop when thread exits (regardless of exit reason)
    stop_flag_.store(true, std::memory_order_release);
}

std::error_code TimedTrigger::configure_tick_thread() {
    if (!tick_thread_.joinable()) {
        return make_error_code(TaskErrc::InvalidParameter);
    }

    // Configure thread using common utilities
    const std::optional<std::uint32_t> core_id =
            is_pinned() ? std::make_optional(get_core_id()) : std::nullopt;
    const std::optional<std::uint32_t> priority =
            has_thread_priority() ? std::make_optional(get_thread_priority()) : std::nullopt;

    const std::error_code config_result =
            configure_thread(tick_thread_, ThreadConfig{core_id, priority});
    if (config_result) {
        RT_LOGC_ERROR(
                TaskLog::TaskTrigger,
                "Failed to configure TimedTrigger thread: {}",
                get_error_name(config_result));
        return config_result;
    }

    // Log configuration status
    if (is_pinned() && has_thread_priority()) {
        RT_LOGC_DEBUG(
                TaskLog::TaskTrigger,
                "TimedTrigger pinned to core {}, RT priority {}",
                get_core_id(),
                get_thread_priority());
    } else if (is_pinned()) {
        RT_LOGC_DEBUG(TaskLog::TaskTrigger, "TimedTrigger pinned to core {}", get_core_id());
    } else if (has_thread_priority()) {
        RT_LOGC_DEBUG(
                TaskLog::TaskTrigger,
                "TimedTrigger configured with RT priority {}",
                get_thread_priority());
    }

    return {}; // Success (default-constructed error_code)
}

void TimedTrigger::stats_thread_function() {
    RT_LOGC_DEBUG(TaskLog::TaskTrigger, "TimedTrigger stats thread started");

    // Pin stats thread to core if specified
    if (stats_core_id_.has_value()) {
        const std::error_code pin_result = pin_current_thread_to_core(stats_core_id_.value());
        if (pin_result) {
            RT_LOGC_WARN(
                    TaskLog::TaskTrigger,
                    "Failed to pin stats thread to core {}: {}",
                    stats_core_id_.value(),
                    get_error_name(pin_result));
        } else {
            RT_LOGC_DEBUG(
                    TaskLog::TaskTrigger, "Stats thread pinned to core {}", stats_core_id_.value());
        }
    }

    // Signal that this thread is ready
    threads_ready_.fetch_add(1, std::memory_order_release);

    // Block until start() is called (using the same semaphore as tick thread)
    start_semaphore_.acquire();

    if (stop_flag_.load(std::memory_order_acquire)) {
        return; // Exit if stopped before started
    }

    while (!stop_flag_.load(std::memory_order_acquire)) {
        // Process pending events
        process_stats_events();

        // Sleep to avoid busy waiting
        static constexpr std::chrono::microseconds STATS_SLEEP_INTERVAL{100};
        std::this_thread::sleep_for(STATS_SLEEP_INTERVAL);
    }

    // Process any remaining events before shutdown
    process_stats_events();

    RT_LOGC_DEBUG(TaskLog::TaskTrigger, "TimedTrigger stats thread stopped");
}

std::size_t TimedTrigger::process_stats_events() {
    TriggerStatsEvent event{};
    std::size_t processed = 0;

    // Process events in batches to avoid starvation
    static constexpr std::size_t MAX_EVENTS_PER_BATCH = 1000;
    while (processed < MAX_EVENTS_PER_BATCH && stats_queue_.dequeue(event)) {
        switch (event.type) {
        case TriggerEventType::TriggerStart:
            handle_trigger_start_event(current_trigger_, event);
            break;
        case TriggerEventType::CallbackStart:
            // Just track the timing - will be processed in CallbackEnd
            break;
        case TriggerEventType::CallbackEnd:
            handle_callback_end(event);
            break;
        case TriggerEventType::LatencyWarning:
            handle_latency_warning_event(current_trigger_, event);
            break;
        case TriggerEventType::CallbackDurationWarning:
            handle_callback_duration_warning_event(current_trigger_, event);
            break;
        case TriggerEventType::JumpDetected:
            handle_jump_detected_event(current_trigger_, event);
            break;
        case TriggerEventType::TriggersSkipped:
            handle_triggers_skipped(event);
            break;
        default:
            RT_LOGC_WARN(
                    TaskLog::TaskTrigger,
                    "Unknown TriggerEventType: {}",
                    static_cast<int>(event.type));
            break;
        }
        processed++;
    }

    return processed;
}

void TimedTrigger::handle_callback_end(const TriggerStatsEvent &event) {
    if (!current_trigger_.has_value()) {
        RT_LOGC_WARN(TaskLog::TaskTrigger, "Received CallbackEnd without matching TriggerStart");
        return;
    }

    const auto &current = current_trigger_.value();

    // Create complete execution record
    TriggerExecutionRecord record{};
    record.scheduled_time = current.scheduled_time;
    record.actual_time = current.actual_start_time;
    record.callback_start_time = current.actual_start_time; // Immediate execution
    record.callback_end_time = event.timestamp;
    record.latency_ns = current.actual_start_time - current.scheduled_time;
    record.callback_duration_ns = event.callback_duration;
    record.trigger_count = current.trigger_count;
    record.skipped_triggers = current.skipped_triggers;
    record.exceeded_latency_threshold = current.latency_warning;
    record.exceeded_callback_duration_threshold = current.callback_duration_warning;
    record.jump_detected = current.jump_detected;

    // Find previous execution for inter-trigger calculation
    {
        const std::lock_guard<std::mutex> lock(stats_mutex_);
        if (!executions_.empty()) {
            const auto &prev = executions_.back();
            record.inter_trigger_actual = current.actual_start_time - prev.actual_time;
            record.inter_trigger_expected = interval_;
        }

        // Store execution record
        executions_.push_back(record);
        ++total_records_created_;

        // Check if cleanup needed
        const std::size_t removed =
                cleanup_old_records_if_needed(executions_, max_execution_records_);
        if (removed > 0) {
            records_truncated_ += removed;
            RT_LOGC_DEBUG(
                    TaskLog::TaskTrigger,
                    "Cleaned up {} old execution records, {} remaining, {} "
                    "total truncated",
                    removed,
                    executions_.size(),
                    records_truncated_);
        }
    }

    // Clear current trigger data
    current_trigger_.reset();
}

void TimedTrigger::print_summary() const {
    const std::lock_guard<std::mutex> lock(stats_mutex_);

    if (executions_.empty()) {
        RT_LOGC_INFO(TaskLog::TaskTrigger, "No execution data available");
        return;
    }

    const std::uint64_t total_triggers = trigger_counter_.load(std::memory_order_acquire);

    RT_LOGC_INFO(TaskLog::TaskTrigger, "===== TimedTrigger Summary =====");
    RT_LOGC_INFO(
            TaskLog::TaskTrigger,
            "Total triggers: {} (recorded: {})",
            total_triggers,
            executions_.size());
    RT_LOGC_INFO(
            TaskLog::TaskTrigger, "Interval: {:.3f} us", nanos_to_micros_int64(interval_.count()));
    if (max_triggers_.has_value()) {
        RT_LOGC_INFO(
                TaskLog::TaskTrigger,
                "Max triggers: {} (limit reached: {})",
                max_triggers_.value(),
                total_triggers >= max_triggers_.value() ? "yes" : "no");
    }

    print_latency_statistics_impl(executions_);
    print_callback_duration_statistics_impl(executions_);
    print_jump_statistics_impl(executions_);
    print_skip_statistics_impl(executions_, skip_strategy_);
}

std::error_code
TimedTrigger::write_stats_to_file(const std::string &filename, TraceWriteMode mode) const {
    const std::lock_guard<std::mutex> lock(stats_mutex_);

    if (executions_.empty()) {
        RT_LOGC_WARN(TaskLog::TaskTrigger, "No execution data to write");
        return make_error_code(TaskErrc::InvalidParameter);
    }

    const bool append_mode = (mode == TraceWriteMode::Append);
    const bool file_exists = std::filesystem::exists(filename);

    try {
        std::ofstream file;
        if (append_mode && file_exists) {
            // Append mode - open for appending
            file.open(filename, std::ios::out | std::ios::app);
            if (!file.is_open()) {
                RT_LOGC_ERROR(TaskLog::TaskTrigger, "Failed to open file for append: {}", filename);
                return make_error_code(TaskErrc::FileOpenFailed);
            }
        } else {
            // Overwrite mode or append to non-existing file
            file.open(filename, std::ios::out | std::ios::trunc);
            if (!file.is_open()) {
                RT_LOGC_ERROR(TaskLog::TaskTrigger, "Failed to open file: {}", filename);
                return make_error_code(TaskErrc::FileOpenFailed);
            }
        }

        // Write version header as first line (only for new files)
        if (!append_mode || !file_exists) {
            file << "{\"version\":\"1.0\"}\n";
        }

        // Write truncation information if records were truncated
        if (records_truncated_ > 0) {
            file << R"({"warning":"execution_records_truncated",)"
                 << "\"total_records_created\":" << total_records_created_ << ","
                 << "\"records_truncated\":" << records_truncated_ << ","
                 << "\"current_records\":" << executions_.size() << "}\n";
        }

        // Write each execution record as a JSON object per line
        for (const auto &record : executions_) {
            file << "{" << "\"trigger_count\":" << record.trigger_count << ","
                 << "\"scheduled_time_ns\":" << record.scheduled_time.count() << ","
                 << "\"actual_time_ns\":" << record.actual_time.count() << ","
                 << "\"callback_start_time_ns\":" << record.callback_start_time.count() << ","
                 << "\"callback_end_time_ns\":" << record.callback_end_time.count() << ","
                 << "\"latency_ns\":" << record.latency_ns.count() << ","
                 << "\"callback_duration_ns\":" << record.callback_duration_ns.count() << ","
                 << "\"inter_trigger_actual_ns\":" << record.inter_trigger_actual.count() << ","
                 << "\"inter_trigger_expected_ns\":" << record.inter_trigger_expected.count() << ","
                 << "\"skipped_triggers\":" << record.skipped_triggers << ","
                 << "\"exceeded_latency_threshold\":"
                 << (record.exceeded_latency_threshold ? "true" : "false") << ","
                 << "\"exceeded_callback_duration_threshold\":"
                 << (record.exceeded_callback_duration_threshold ? "true" : "false") << ","
                 << "\"jump_detected\":" << (record.jump_detected ? "true" : "false") << "}\n";
        }

        file.close();
        RT_LOGC_INFO(
                TaskLog::TaskTrigger,
                "Successfully wrote {} execution records to {} (total "
                "created: {}, truncated: {})",
                executions_.size(),
                filename,
                total_records_created_,
                records_truncated_);
        return {}; // Success

    } catch (const std::exception &e) {
        RT_LOGC_ERROR(TaskLog::TaskTrigger, "Error writing to file {}: {}", filename, e.what());
        return make_error_code(TaskErrc::FileWriteFailed);
    }
}

int TimedTrigger::write_chrome_trace_to_file(
        const std::string &filename, TraceWriteMode mode) const {
    const std::lock_guard<std::mutex> lock(stats_mutex_);

    if (executions_.empty()) {
        RT_LOGC_WARN(TaskLog::TaskTrigger, "No execution data to write");
        return -1;
    }

    try {
        const bool append_mode = (mode == TraceWriteMode::Append);
        bool file_exists = false;

        if (append_mode) {
            std::ifstream check_file(filename);
            file_exists = check_file.good();
            check_file.close();
        }

        std::ofstream file;
        if (append_mode && file_exists) {
            // Read the end of the file to find where "]}" starts
            std::ifstream read_file(filename);
            read_file.seekg(0, std::ios::end);
            const std::streampos file_size = read_file.tellg();

            if (file_size < 2) { // Need at least "]}"
                RT_LOGC_ERROR(
                        TaskLog::TaskTrigger,
                        "File too small for Chrome trace format: {}",
                        filename);
                return -2;
            }

            static constexpr std::streamoff MAX_RFIND_SIZE = 10;
            const std::streamoff read_size =
                    std::min(static_cast<std::streamoff>(file_size), MAX_RFIND_SIZE);
            read_file.seekg(-read_size, std::ios::end);

            std::string ending(static_cast<std::size_t>(read_size), '\0');
            read_file.read(ending.data(), read_size);
            read_file.close();

            const std::size_t close_pos = ending.rfind("]}");
            if (close_pos == std::string::npos) {
                RT_LOGC_ERROR(TaskLog::TaskTrigger, "Invalid Chrome trace format in {}", filename);
                return -2;
            }

            // Open file and truncate at the "]}" position
            file.open(filename, std::ios::in | std::ios::out);
            if (!file.is_open()) {
                RT_LOGC_ERROR(TaskLog::TaskTrigger, "Failed to open file for append: {}", filename);
                return -2;
            }

            const auto read_size_offset = static_cast<std::streamoff>(read_size);
            const auto close_pos_offset = static_cast<std::streamoff>(close_pos);
            const auto seek_offset = -(read_size_offset - close_pos_offset);
            file.seekp(seek_offset, std::ios::end);
            file << ",\n";
        } else {
            // Overwrite mode or append to non-existing file
            file.open(filename, std::ios::out | std::ios::trunc);
            if (!file.is_open()) {
                RT_LOGC_ERROR(
                        TaskLog::TaskTrigger, "Failed to open file for writing: {}", filename);
                return -2;
            }

            // Write Chrome trace format header
            file << "{\"traceEvents\":[\n";
        }

        // Write each execution record as a Chrome trace duration event per line
        for (std::size_t i = 0; i < executions_.size(); ++i) {
            const auto &record = executions_[i];

            // Convert nanoseconds to microseconds (Chrome expects microseconds)
            static constexpr double NS_TO_US_DIVISOR = 1000.0;
            const double callback_start_us =
                    static_cast<double>(record.callback_start_time.count()) / NS_TO_US_DIVISOR;
            const double callback_duration_us =
                    static_cast<double>(record.callback_duration_ns.count()) / NS_TO_US_DIVISOR;

            // Add comma separator except for first event
            // Note: when appending, we already added comma in the seek logic above
            if (i > 0) {
                file << ",\n";
            }

            // Trigger callback as duration event (use actual thread ID)
            file << "{" << R"("name":"Trigger_)" << record.trigger_count << R"(",)"
                 << R"("cat":"TimedTrigger",)" << R"("ph":"X",)" << "\"pid\":" << getpid() << ","
                 << "\"tid\":" << tick_thread_id_.load(std::memory_order_acquire) << ","
                 << "\"ts\":" << std::fixed << callback_start_us << "," << "\"dur\":" << std::fixed
                 << callback_duration_us << "}";
        }

        file << "\n]}\n";
        file.close();

        RT_LOGC_DEBUG(
                TaskLog::TaskTrigger,
                "Successfully wrote {} execution records to Chrome trace "
                "format {} (total "
                "created: {}, truncated: {})",
                executions_.size(),
                filename,
                total_records_created_,
                records_truncated_);
        return 0;

    } catch (const std::exception &e) {
        RT_LOGC_ERROR(
                TaskLog::TaskTrigger, "Error writing Chrome trace file {}: {}", filename, e.what());
        return -3;
    }
}

} // namespace framework::task
