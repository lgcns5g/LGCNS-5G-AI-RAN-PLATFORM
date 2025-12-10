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
 * @file task_monitor.cpp
 * @brief Implementation of TaskMonitor class and related functionality
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
#include <future>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <ratio>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <parallel_hashmap/phmap.h>
#include <quill/LogMacros.h>
#include <unistd.h>

#include <wise_enum.h>

#include "log/rt_log_macros.hpp"
#include "task/bounded_queue.hpp"
#include "task/flat_map.hpp"
#include "task/task.hpp"
#include "task/task_errors.hpp"
#include "task/task_log.hpp"
#include "task/task_monitor.hpp"
#include "task/task_utils.hpp"
#include "task/time.hpp"

namespace {
namespace ft = framework::task;

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

/**
 * Push event to queue
 * @param[in] event Event to push
 * @param[in] event_queue Queue to push to
 * @return true if pushed successfully, false if queue full
 */
bool push_event_to_queue(
        const ft::MonitorEvent &event, ft::BoundedQueue<ft::MonitorEvent> &event_queue) {
    return event_queue.enqueue(event);
}

// MonitorEvent factory functions
ft::MonitorEvent create_register_task(const ft::TaskHandle &task_handle) {
    ft::MonitorEvent event{};
    event.type = ft::MonitorEventType::RegisterTask;
    event.task_id = task_handle->get_task_id();
    event.timestamp = ft::Time::now_ns();
    event.task_handle = task_handle;
    return event;
}

ft::MonitorEvent create_record_start(std::uint64_t task_id, ft::WorkerId w, ft::Nanos start_time) {
    ft::MonitorEvent event{};
    event.type = ft::MonitorEventType::RecordStart;
    event.task_id = task_id;
    event.timestamp = start_time;
    event.worker = w;
    return event;
}

ft::MonitorEvent create_record_end(std::uint64_t task_id, ft::Nanos end_time, ft::TaskStatus s) {
    ft::MonitorEvent event{};
    event.type = ft::MonitorEventType::RecordEnd;
    event.task_id = task_id;
    event.timestamp = end_time;
    event.status = s;
    return event;
}

ft::MonitorEvent create_cancel_task(std::uint64_t task_id) {
    ft::MonitorEvent event{};
    event.type = ft::MonitorEventType::CancelTask;
    event.task_id = task_id;
    event.timestamp = ft::Time::now_ns();
    return event;
}

/**
 * Handle task registration event
 * @param[in] event Registration event
 * @param[in,out] task_data Task monitoring data map
 */
void handle_register_task(
        const ft::MonitorEvent &event, ft::FlatMap<std::uint64_t, ft::TaskMonitorData> &task_data) {
    auto it = task_data.find(event.task_id);
    if (it != task_data.end()) {
        RT_LOGC_WARN(
                ft::TaskLog::TaskMonitor,
                "Overwriting existing registration for task: {}",
                event.task_handle.has_value() ? event.task_handle.value()->get_task_name()
                                              : "unknown");
    }

    // Create monitoring data directly from TaskHandle
    if (event.task_handle.has_value()) {
        task_data[event.task_id] = ft::TaskMonitorData{*event.task_handle};
    }
}

/**
 * Handle execution start event
 * @param[in] event Start event
 * @param[in,out] task_data Task monitoring data map
 */
void handle_record_start(
        const ft::MonitorEvent &event, ft::FlatMap<std::uint64_t, ft::TaskMonitorData> &task_data) {
    auto it = task_data.find(event.task_id);
    if (it == task_data.end()) {
        RT_LOGC_WARN(
                ft::TaskLog::TaskMonitor,
                "Attempted to record start for unregistered task ID: {}",
                event.task_id);
        return;
    }

    it->second.start_time = event.timestamp;
    it->second.worker = event.worker;
}

/**
 * Handle task cancellation event
 * @param[in] event Cancellation event
 * @param[in,out] task_data Task monitoring data map
 */
void handle_cancel_task(
        const ft::MonitorEvent &event, ft::FlatMap<std::uint64_t, ft::TaskMonitorData> &task_data) {
    auto it = task_data.find(event.task_id);
    if (it == task_data.end()) {
        RT_LOGC_WARN(
                ft::TaskLog::TaskMonitor,
                "Attempted to cancel unregistered task: {}",
                event.task_id);
        return;
    }

    it->second.cancelled = true;

    // Cancel the task via TaskHandle
    if (it->second.task_handle.has_value()) {
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        (*it->second.task_handle)->cancel();
    }
}

/**
 * Log detailed statistics in standard format (7 lines)
 * @param[in] title Statistics section title
 * @param[in] avg_us Average value in microseconds
 * @param[in] median_us Median value in microseconds
 * @param[in] p95_us 95th percentile in microseconds
 * @param[in] p99_us 99th percentile in microseconds
 * @param[in] min_us Minimum value in microseconds
 * @param[in] max_us Maximum value in microseconds
 * @param[in] std_us Standard deviation in microseconds
 */
void log_detailed_statistics(
        const std::string_view title,
        double avg_us,
        double median_us,
        double p95_us,
        double p99_us,
        double min_us,
        double max_us,
        double std_us) {
    RT_LOGC_INFO(ft::TaskLog::TaskMonitor, "{}:", title);
    RT_LOGC_INFO(ft::TaskLog::TaskMonitor, "  Average: {:.3f} us", avg_us);
    RT_LOGC_INFO(ft::TaskLog::TaskMonitor, "  Median:  {:.3f} us", median_us);
    RT_LOGC_INFO(ft::TaskLog::TaskMonitor, "  95th:    {:.3f} us", p95_us);
    RT_LOGC_INFO(ft::TaskLog::TaskMonitor, "  99th:    {:.3f} us", p99_us);
    RT_LOGC_INFO(ft::TaskLog::TaskMonitor, "  Min:     {:.3f} us", min_us);
    RT_LOGC_INFO(ft::TaskLog::TaskMonitor, "  Max:     {:.3f} us", max_us);
    RT_LOGC_INFO(ft::TaskLog::TaskMonitor, "  Std:     {:.3f} us", std_us);
}

/**
 * Log compact statistics in 3-line format with optional prefix
 * @param[in] prefix Line prefix (e.g., "    " for indentation)
 * @param[in] min_us Minimum value in microseconds
 * @param[in] max_us Maximum value in microseconds
 * @param[in] avg_us Average value in microseconds
 * @param[in] median_us Median value in microseconds
 * @param[in] p95_us 95th percentile in microseconds
 * @param[in] p99_us 99th percentile in microseconds
 * @param[in] std_us Standard deviation in microseconds
 */
void log_compact_statistics(
        const std::string_view prefix,
        double min_us,
        double max_us,
        double avg_us,
        double median_us,
        double p95_us,
        double p99_us,
        double std_us) {
    RT_LOGC_INFO(
            ft::TaskLog::TaskMonitor,
            "{}Min: {:.3f} us, Max: {:.3f} us, Avg: {:.3f} us",
            prefix,
            min_us,
            max_us,
            avg_us);
    RT_LOGC_INFO(
            ft::TaskLog::TaskMonitor,
            "{}Median: {:.3f} us, 95th: {:.3f} us, 99th: {:.3f} us",
            prefix,
            median_us,
            p95_us,
            p99_us);
    RT_LOGC_INFO(ft::TaskLog::TaskMonitor, "{}Std: {:.3f} us", prefix, std_us);
}

/**
 * Log status counts with percentages
 * @param[in] prefix Line prefix (e.g., "    " for indentation)
 * @param[in] cancelled_count Number of cancelled tasks
 * @param[in] failed_count Number of failed tasks
 * @param[in] total_count Total number of tasks
 */
void log_status_counts(
        const std::string_view prefix,
        std::size_t cancelled_count,
        std::size_t failed_count,
        std::size_t total_count) {
    const double cancelled_percent =
            cancelled_count > 0
                    ? (static_cast<double>(cancelled_count) / static_cast<double>(total_count)) *
                              100.0
                    : 0.0;
    const double failed_percent =
            failed_count > 0
                    ? (static_cast<double>(failed_count) / static_cast<double>(total_count)) * 100.0
                    : 0.0;

    RT_LOGC_INFO(
            ft::TaskLog::TaskMonitor,
            "{}Cancelled: {} ({:.1f}%) [includes timeouts], Failed: {} ({:.1f}%)",
            prefix,
            cancelled_count,
            cancelled_percent,
            failed_count,
            failed_percent);
}

/**
 * Calculate parent generation for a dependent task
 * @param[in] task_generation Current task's dependency generation
 * @return Parent generation (task_generation - 1, or 0 if task_generation is 0)
 */
std::uint32_t calculate_parent_generation(std::uint32_t task_generation) {
    return task_generation > 0 ? task_generation - 1 : 0;
}

/**
 * Create parent filter predicate for finding immediate parents
 * @param[in] dependent_task Task execution record for the dependent task
 * @param[in] parent_generation Pre-calculated parent generation
 * @return Lambda function that filters for immediate parents
 */
auto create_parent_filter(
        const ft::TaskExecutionRecord &dependent_task, std::uint32_t parent_generation) {
    return [&dependent_task, parent_generation](const ft::TaskExecutionRecord &exec) {
        return exec.times_scheduled == dependent_task.times_scheduled &&
               exec.dependency_generation == parent_generation && // Immediate parent generation
               exec.end_time <= dependent_task.start_time;        // Completed before dependent
    };
}

/**
 * Find latest parent completion time for a dependent task
 * @param[in] dependent_task Task execution record for the dependent task
 * @param[in] all_executions Vector of all execution records to search
 * @return Latest parent completion time, or Nanos{0} if no parents found
 */
ft::Nanos find_latest_parent_completion(
        const ft::TaskExecutionRecord &dependent_task,
        const std::vector<ft::TaskExecutionRecord> &all_executions) {
    if (dependent_task.dependency_generation == 0) {
        return ft::Nanos{0}; // Root task has no parents
    }

    const std::uint32_t parent_generation =
            calculate_parent_generation(dependent_task.dependency_generation);
    auto parent_filter = create_parent_filter(dependent_task, parent_generation);

    auto max_parent_it = std::max_element(
            all_executions.begin(),
            all_executions.end(),
            [&](const ft::TaskExecutionRecord &a, const ft::TaskExecutionRecord &b) {
                const bool a_valid = parent_filter(a);
                const bool b_valid = parent_filter(b);

                if (!a_valid && !b_valid) {
                    return false;
                }
                if (!a_valid) {
                    return true; // a invalid, b valid -> a < b
                }
                if (!b_valid) {
                    return false; // a valid, b invalid -> a > b
                }

                return a.end_time < b.end_time; // Both valid, compare end times
            });

    if (max_parent_it != all_executions.end() && parent_filter(*max_parent_it)) {
        return max_parent_it->end_time;
    }

    return ft::Nanos{0}; // No valid parents found
}

/**
 * Apply jitter correction to dependent tasks based on parent completion times
 * @param[in] executions Vector of execution records to correct
 * @return Vector of task records with corrected jitter values for dependent
 * tasks
 */
[[nodiscard]] auto
apply_jitter_correction_to_dependent_tasks(const std::vector<ft::TaskExecutionRecord> &executions) {
    std::vector<ft::TaskExecutionRecord> corrected_tasks;
    corrected_tasks.reserve(executions.size());

    for (const auto &task : executions) {
        ft::TaskExecutionRecord corrected_task = task; // Copy the task

        if (corrected_task.dependency_generation > 0) { // Has parents
            const ft::Nanos dependency_satisfied_time =
                    find_latest_parent_completion(corrected_task, executions);

            // Recalculate jitter from dependency satisfaction point
            if (dependency_satisfied_time > ft::Nanos{0}) {
                corrected_task.jitter_ns = corrected_task.start_time - dependency_satisfied_time;

            } else {
                // Check if we have a parent in the same round but with timing issues
                if (corrected_task.dependency_generation > 0) {
                    const std::uint32_t parent_generation =
                            corrected_task.dependency_generation - 1;
                    for (const auto &exec : executions) {
                        if (exec.dependency_generation == parent_generation &&
                            exec.times_scheduled == corrected_task.times_scheduled &&
                            exec.end_time > corrected_task.start_time) {

                            auto time_diff = (exec.end_time - corrected_task.start_time).count();
                            RT_LOGC_WARN(
                                    ft::TaskLog::TaskMonitor,
                                    "TIMING ANOMALY: Dependent task {} (worker {}) started "
                                    "{:.3f}us before parent {} (worker {}) completed",
                                    corrected_task.task_name,
                                    corrected_task.worker,
                                    static_cast<double>(time_diff) / 1000.0,
                                    exec.task_name,
                                    exec.worker);
                            break;
                        }
                    }
                }
            }
        }

        corrected_tasks.push_back(std::move(corrected_task));
    }

    return corrected_tasks;
}

/**
 * Print jitter statistics for a category of tasks
 * @param[in] tasks Task records to analyze
 */
void print_jitter_statistics(const std::vector<ft::TaskExecutionRecord> &tasks) {
    if (tasks.empty()) {
        return;
    }

    // Calculate jitter statistics
    std::vector<int64_t> jitter_values;
    int64_t total_jitter = 0;
    int64_t max_jitter = 0;
    int64_t min_jitter = std::numeric_limits<int64_t>::max();

    for (const auto &exec : tasks) {
        const int64_t abs_jitter = std::abs(exec.jitter_ns.count());
        jitter_values.push_back(abs_jitter);
        total_jitter += abs_jitter;
        max_jitter = std::max(max_jitter, abs_jitter);
        min_jitter = std::min(min_jitter, abs_jitter);
    }

    // Calculate jitter statistics
    const double avg_jitter_us =
            static_cast<double>(total_jitter) / static_cast<double>(tasks.size()) / 1000.0;
    // Convert to double for percentile calculations
    std::vector<double> jitter_values_us;
    jitter_values_us.reserve(jitter_values.size());
    for (const auto &value : jitter_values) {
        jitter_values_us.push_back(static_cast<double>(value) / 1000.0);
    }
    std::sort(jitter_values_us.begin(), jitter_values_us.end());

    const double median_jitter_us = ft::calculate_percentile(jitter_values_us, 0.5);
    const double p95_jitter_us = ft::calculate_percentile(jitter_values_us, 0.95);
    const double p99_jitter_us = ft::calculate_percentile(jitter_values_us, 0.99);
    const double min_jitter_us = static_cast<double>(min_jitter) / 1000.0;
    const double max_jitter_us = static_cast<double>(max_jitter) / 1000.0;

    // Calculate standard deviation (on raw nanosecond values, then convert
    // result)
    const double avg_jitter_ns =
            static_cast<double>(total_jitter) / static_cast<double>(tasks.size());
    const double std_jitter_us =
            ft::calculate_standard_deviation(jitter_values, avg_jitter_ns) / 1000.0;
    // === JITTER STATISTICS ===
    log_detailed_statistics(
            "Overall Jitter Statistics",
            avg_jitter_us,
            median_jitter_us,
            p95_jitter_us,
            p99_jitter_us,
            min_jitter_us,
            max_jitter_us,
            std_jitter_us);

    // Print worker-specific jitter breakdown
    RT_LOGC_INFO(ft::TaskLog::TaskMonitor, "Jitter by Worker:");
    std::map<ft::WorkerId, std::vector<double>> worker_jitters;
    for (const auto &exec : tasks) {
        const double jitter_us = static_cast<double>(std::abs(exec.jitter_ns.count())) / 1000.0;
        worker_jitters[exec.worker].push_back(jitter_us);
    }

    for (const auto &[worker, jitters] : worker_jitters) {
        if (!jitters.empty()) {
            const double avg = std::accumulate(jitters.begin(), jitters.end(), 0.0) /
                               static_cast<double>(jitters.size());
            const double max_val = *std::max_element(jitters.begin(), jitters.end());
            RT_LOGC_INFO(
                    ft::TaskLog::TaskMonitor,
                    "  Worker {}: {:.3f} us avg, {:.3f} us max, {} tasks",
                    worker,
                    avg,
                    max_val,
                    jitters.size());
        }
    }
}

/**
 * Print execution time statistics for a category of tasks
 * @param[in] tasks Task records to analyze
 */
void print_execution_time_statistics(const std::vector<ft::TaskExecutionRecord> &tasks) {
    if (tasks.empty()) {
        return;
    }

    // Calculate execution time statistics
    std::vector<int64_t> execution_time_values;
    int64_t total_execution_time = 0;
    int64_t max_execution_time = 0;
    int64_t min_execution_time = std::numeric_limits<int64_t>::max();

    for (const auto &exec : tasks) {
        // Collect execution time data
        const int64_t execution_time = exec.duration_ns.count();
        execution_time_values.push_back(execution_time);
        total_execution_time += execution_time;
        max_execution_time = std::max(max_execution_time, execution_time);
        min_execution_time = std::min(min_execution_time, execution_time);
    }

    // Calculate execution time statistics
    const double avg_execution_time_us =
            static_cast<double>(total_execution_time) / static_cast<double>(tasks.size()) / 1000.0;
    // Convert to double for percentile calculations
    std::vector<double> execution_time_values_us;
    execution_time_values_us.reserve(execution_time_values.size());
    for (const auto &value : execution_time_values) {
        execution_time_values_us.push_back(static_cast<double>(value) / 1000.0);
    }
    std::sort(execution_time_values_us.begin(), execution_time_values_us.end());

    const double median_execution_time_us = ft::calculate_percentile(execution_time_values_us, 0.5);
    const double p95_execution_time_us = ft::calculate_percentile(execution_time_values_us, 0.95);
    const double p99_execution_time_us = ft::calculate_percentile(execution_time_values_us, 0.99);
    const double min_execution_time_us = static_cast<double>(min_execution_time) / 1000.0;
    const double max_execution_time_us = static_cast<double>(max_execution_time) / 1000.0;

    // Calculate standard deviation (on raw nanosecond values, then convert
    // result)
    const double avg_execution_time_ns =
            static_cast<double>(total_execution_time) / static_cast<double>(tasks.size());
    const double std_execution_time_us =
            ft::calculate_standard_deviation(execution_time_values, avg_execution_time_ns) / 1000.0;
    // === EXECUTION TIME STATISTICS ===
    log_detailed_statistics(
            "Overall Execution Time Distribution",
            avg_execution_time_us,
            median_execution_time_us,
            p95_execution_time_us,
            p99_execution_time_us,
            min_execution_time_us,
            max_execution_time_us,
            std_execution_time_us);

    // Print execution time distribution by task name
    RT_LOGC_INFO(ft::TaskLog::TaskMonitor, "Execution Time Distribution by Task:");
    std::map<std::string, std::vector<double>> task_execution_times;
    std::map<std::string, std::size_t> task_cancelled_counts;
    std::map<std::string, std::size_t> task_failed_counts;

    for (const auto &exec : tasks) {
        const double execution_time_us = static_cast<double>(exec.duration_ns.count()) / 1000.0;
        task_execution_times[exec.task_name].push_back(execution_time_us);

        // Track status counts per task
        if (exec.was_cancelled) {
            task_cancelled_counts[exec.task_name]++;
        }
        if (exec.status == ft::TaskStatus::Failed) {
            task_failed_counts[exec.task_name]++;
        }
    }

    for (const auto &[task_name, exec_times] : task_execution_times) {
        if (!exec_times.empty()) {
            // Sort for percentiles
            std::vector<double> sorted_times = exec_times;
            std::sort(sorted_times.begin(), sorted_times.end());

            // Calculate statistics
            const double avg = std::accumulate(sorted_times.begin(), sorted_times.end(), 0.0) /
                               static_cast<double>(sorted_times.size());
            const double min_val = sorted_times.front();
            const double max_val = sorted_times.back();
            const double median = ft::calculate_percentile(sorted_times, 0.5);
            const double p95 = ft::calculate_percentile(sorted_times, 0.95);
            const double p99 = ft::calculate_percentile(sorted_times, 0.99);
            const double std_dev = ft::calculate_standard_deviation(sorted_times, avg);

            // Get status counts for this task
            const std::size_t task_cancelled = task_cancelled_counts[task_name];
            const std::size_t task_failed = task_failed_counts[task_name];
            const std::size_t total_executions = exec_times.size();

            RT_LOGC_INFO(
                    ft::TaskLog::TaskMonitor,
                    "  Task '{}' ({} executions):",
                    task_name,
                    total_executions);
            log_compact_statistics("    ", min_val, max_val, avg, median, p95, p99, std_dev);
            log_status_counts("    ", task_cancelled, task_failed, total_executions);
        }
    }
}

/**
 * Cleanup old execution records by removing oldest 10%
 * @param[in,out] executions Vector of execution records to clean
 * @param[in] max_records Maximum records allowed
 * @return Number of records removed
 */
template <typename RecordType>
[[nodiscard]] std::size_t
cleanup_old_records_if_needed(std::vector<RecordType> &executions, std::size_t max_records) {
    if (executions.size() >= max_records) {
        // Remove oldest 10% or at least 1 record for small limits
        const std::size_t to_remove = std::max(1UL, max_records / 10);

        executions.erase(
                executions.begin(), executions.begin() + static_cast<std::ptrdiff_t>(to_remove));

        return to_remove;
    }
    return 0;
}

} // anonymous namespace

namespace framework::task {

void TaskMonitor::print_graph_statistics(const std::vector<TaskExecutionRecord> &executions) {
    if (executions.empty()) {
        return;
    }

    // Group tasks by graph name (aggregate all scheduling rounds)
    std::map<std::string, std::vector<TaskExecutionRecord>> graph_tasks_map;
    std::map<std::string, std::set<std::uint64_t>> graph_times_scheduled;

    for (const auto &exec : executions) {
        graph_tasks_map[exec.graph_name].push_back(exec);
        graph_times_scheduled[exec.graph_name].insert(exec.times_scheduled);
    }

    RT_LOGC_INFO(TaskLog::TaskMonitor, "===== Graph Execution Statistics =====");

    for (const auto &[graph_name, tasks] : graph_tasks_map) {
        if (tasks.empty()) {
            continue;
        }

        // Collect execution times for statistics calculation
        std::vector<double> execution_times_us;
        std::size_t cancelled_count = 0;
        std::size_t failed_count = 0;

        for (const auto &task : tasks) {
            const double execution_time_us = static_cast<double>(task.duration_ns.count()) / 1000.0;
            execution_times_us.push_back(execution_time_us);

            if (task.was_cancelled) {
                cancelled_count++;
            }
            if (task.status == TaskStatus::Failed) {
                failed_count++;
            }
        }

        // Sort for percentiles
        std::sort(execution_times_us.begin(), execution_times_us.end());

        // Calculate statistics
        const double avg =
                std::accumulate(execution_times_us.begin(), execution_times_us.end(), 0.0) /
                static_cast<double>(execution_times_us.size());
        const double min_val = execution_times_us.front();
        const double max_val = execution_times_us.back();
        const double median = ft::calculate_percentile(execution_times_us, 0.5);
        const double p95 = ft::calculate_percentile(execution_times_us, 0.95);
        const double p99 = ft::calculate_percentile(execution_times_us, 0.99);
        const double std_dev = ft::calculate_standard_deviation(execution_times_us, avg);

        const std::size_t times_scheduled = graph_times_scheduled[graph_name].size();

        RT_LOGC_INFO(
                TaskLog::TaskMonitor,
                "  Graph '{}' ({} tasks scheduled {} times):",
                graph_name,
                tasks.size(),
                times_scheduled);
        log_compact_statistics("    ", min_val, max_val, avg, median, p95, p99, std_dev);
        log_status_counts("    ", cancelled_count, failed_count, tasks.size());
    }
}

void TaskMonitor::print_category_stats(
        const std::vector<TaskExecutionRecord> &tasks, const std::string_view category_name) {
    if (tasks.empty()) {
        RT_LOGC_DEBUG(TaskLog::TaskMonitor, "No {} tasks were executed", category_name);
        return;
    }

    // Calculate status counts
    std::size_t cancelled_count = 0;
    std::size_t failed_count = 0;

    for (const auto &exec : tasks) {
        if (exec.was_cancelled) {
            cancelled_count++;
        }

        if (exec.status == TaskStatus::Failed) {
            failed_count++;
        }
    }

    // Print category header
    RT_LOGC_INFO(
            TaskLog::TaskMonitor, "===== {} Tasks ({} tasks) =====", category_name, tasks.size());

    // Print jitter statistics
    print_jitter_statistics(tasks);

    // Print execution time statistics
    print_execution_time_statistics(tasks);

    // === Task STATUS SUMMARY ===
    RT_LOGC_INFO(TaskLog::TaskMonitor, "Task Status Summary:");
    RT_LOGC_INFO(
            TaskLog::TaskMonitor,
            "  Cancelled: {} ({:.1f}%) [includes timeouts]",
            cancelled_count,
            cancelled_count > 0
                    ? (static_cast<double>(cancelled_count) / static_cast<double>(tasks.size())) *
                              100.0
                    : 0.0);
    RT_LOGC_INFO(
            TaskLog::TaskMonitor,
            "  Failed: {} ({:.1f}%)",
            failed_count,
            failed_count > 0
                    ? (static_cast<double>(failed_count) / static_cast<double>(tasks.size())) *
                              100.0
                    : 0.0);
}

namespace {

constexpr std::size_t EVENT_QUEUE_SIZE = 65536; //!< Event queue size (power of 2)
constexpr auto GB = 1024ULL * 1024ULL * 1024ULL;

} // namespace

// TaskMonitor implementation
TaskMonitor::TaskMonitor(const std::optional<std::size_t> max_execution_records)
        : event_queue_{EVENT_QUEUE_SIZE},
          max_execution_records_(max_execution_records.value_or(
                  calculate_max_records_for_bytes<TaskExecutionRecord>(50ULL * GB))) {
    // Log record configuration
    RT_LOGC_DEBUG(
            TaskLog::TaskMonitor,
            "TaskMonitor configured for max {} execution records (~{:.1f} GB)",
            max_execution_records_,
            static_cast<double>(max_execution_records_ * sizeof(TaskExecutionRecord)) /
                    static_cast<double>(GB));
}

TaskMonitor::~TaskMonitor() noexcept { stop(); }

std::error_code
TaskMonitor::start(std::optional<std::uint32_t> core_id, std::chrono::microseconds sleep_duration) {
    RT_LOGC_DEBUG(TaskLog::TaskMonitor, "TaskMonitor started");

    // Validate core_id upfront for better error reporting
    if (core_id.has_value()) {
        const auto max_cores = std::thread::hardware_concurrency();
        if (core_id.value() >= max_cores) {
            RT_LOGC_ERROR(
                    TaskLog::TaskMonitor,
                    "Invalid monitor core ID {}: system has {} cores",
                    core_id.value(),
                    max_cores);
            return make_error_code(TaskErrc::InvalidParameter);
        }
    }

    // Use promise/future for thread initialization synchronization
    std::promise<std::error_code> init_promise{};
    std::future<std::error_code> init_future = init_promise.get_future();

    monitor_thread_ = std::thread(
            [this, core_id, sleep_duration, init_promise = std::move(init_promise)]() mutable {
                if (core_id.has_value()) {
                    // Pin monitor thread to specific core using common utility
                    const std::error_code pin_result = pin_current_thread_to_core(core_id.value());
                    if (pin_result) {
                        init_promise.set_value(pin_result);
                        return;
                    }
                }

                // Signal successful initialization
                init_promise.set_value(make_error_code(TaskErrc::Success));

                // Run monitor loop
                monitor_function(sleep_duration);
            });

    // Wait for thread initialization to complete
    return init_future.get();
}

void TaskMonitor::stop() noexcept {
    RT_LOGC_DEBUG(TaskLog::TaskMonitor, "Stopping TaskMonitor...");
    stop_flag_.store(true, std::memory_order_release);

    if (monitor_thread_.joinable()) {
        monitor_thread_.join();
    }

    MonitorEvent event{};
    while (event_queue_.dequeue(event)) {
        ;
    }

    // We dequeued all events, so clear task data until next start()
    task_data_.clear();

    RT_LOGC_DEBUG(TaskLog::TaskMonitor, "TaskMonitor stopped");
}

void TaskMonitor::clear_stats() {
    const std::lock_guard<std::mutex> lock(stats_mutex_);
    executions_.clear();
    total_records_created_ = 0;
    records_truncated_ = 0;
}

std::error_code TaskMonitor::register_task(const TaskHandle &task_handle) {
    // Push registration event to queue for background processing (real-time safe)
    // No FlatMap operations or string allocations in real-time thread
    // TaskHandle is copied into the event, so no lifetime concerns for caller
    auto event = create_register_task(task_handle);
    return push_event(event) ? make_error_code(TaskErrc::Success)
                             : make_error_code(TaskErrc::QueueFull);
}

std::error_code
TaskMonitor::record_start(std::uint64_t task_id, WorkerId worker_id, Nanos start_time) {
    auto event = create_record_start(task_id, worker_id, start_time);
    return push_event(event) ? make_error_code(TaskErrc::Success)
                             : make_error_code(TaskErrc::QueueFull);
}

std::error_code TaskMonitor::record_end(std::uint64_t task_id, Nanos end_time, TaskStatus status) {
    auto event = create_record_end(task_id, end_time, status);
    return push_event(event) ? make_error_code(TaskErrc::Success)
                             : make_error_code(TaskErrc::QueueFull);
}

std::error_code TaskMonitor::cancel_task(std::uint64_t task_id) {
    auto event = create_cancel_task(task_id);
    return push_event(event) ? make_error_code(TaskErrc::Success)
                             : make_error_code(TaskErrc::QueueFull);
}

void TaskMonitor::monitor_function(std::chrono::microseconds sleep_duration) {
    RT_LOGC_DEBUG(TaskLog::TaskMonitor, "TaskMonitor thread started");

    while (!stop_flag_.load(std::memory_order_acquire)) {
        // Process pending events
        process_events();

        // Check for task timeouts
        check_timeouts();

        // Sleep to avoid busy waiting
        std::this_thread::sleep_for(sleep_duration);
    }

    // Process any remaining events before shutdown
    process_events();

    RT_LOGC_DEBUG(TaskLog::TaskMonitor, "TaskMonitor thread stopped");
}

std::size_t TaskMonitor::process_events() {
    MonitorEvent event{};
    std::size_t processed = 0;

    // Process events in batches to avoid starving timeout monitoring
    while (processed < 1000 && event_queue_.dequeue(event)) {
        switch (event.type) {
        case MonitorEventType::RegisterTask:
            handle_register_task(event, task_data_);
            break;
        case MonitorEventType::RecordStart:
            handle_record_start(event, task_data_);
            break;
        case MonitorEventType::RecordEnd:
            handle_record_end(event);
            break;
        case MonitorEventType::CancelTask:
            handle_cancel_task(event, task_data_);
            break;
        default:
            throw std::runtime_error(
                    std::format("Unknown MonitorEventType: {}", static_cast<int>(event.type)));
        }
        processed++;
    }

    return processed;
}

void TaskMonitor::handle_record_end(const MonitorEvent &event) {
    auto task_it = task_data_.find(event.task_id);

    if (task_it == task_data_.end()) {
        RT_LOGC_WARN(
                TaskLog::TaskMonitor,
                "Attempted to record end for unregistered task ID: {}",
                event.task_id);
        return;
    }

    const auto &task_data = task_it->second;

    if (!task_data.task_handle.has_value()) {
        RT_LOGC_WARN(
                TaskLog::TaskMonitor,
                "Task data missing TaskHandle for task ID: {}",
                event.task_id);
        return;
    }

    const auto &task_handle = task_data.task_handle.value();

    // Create execution record for statistics
    TaskExecutionRecord record{};
    record.task_name = std::string(task_handle->get_task_name());
    record.graph_name = std::string(task_handle->get_graph_name());
    record.end_time = event.timestamp;
    record.status = event.status;
    record.start_time = task_data.start_time;
    record.worker = task_data.worker;
    record.scheduled_time = task_handle->get_scheduled_time();
    record.was_cancelled = task_data.cancelled;
    record.times_scheduled = task_handle->get_times_scheduled();
    record.dependency_generation = task_handle->get_dependency_generation();

    // Calculate duration and jitter
    record.duration_ns = record.end_time - record.start_time;
    record.jitter_ns = record.start_time - record.scheduled_time;

    // Store execution record
    {
        const std::lock_guard<std::mutex> lock(stats_mutex_);
        executions_.push_back(record);
        ++total_records_created_;

        // Check if cleanup needed
        const std::size_t removed =
                cleanup_old_records_if_needed(executions_, max_execution_records_);
        if (removed > 0) {
            records_truncated_ += removed;
            RT_LOGC_DEBUG(
                    TaskLog::TaskMonitor,
                    "Cleaned up {} old execution records, {} remaining, {} "
                    "total truncated",
                    removed,
                    executions_.size(),
                    records_truncated_);
        }
    }

    // Clean up completed task data to prevent memory growth
    task_data_.erase(event.task_id);

    // Log large jitter with detailed context
    using namespace std::chrono_literals;
    if (std::chrono::abs(record.jitter_ns) > 100ms) {
        RT_LOGC_WARN(
                TaskLog::TaskMonitor,
                "LARGE JITTER DETECTED: Task {} had {:.3f}ms jitter",
                record.task_name,
                static_cast<double>(record.jitter_ns.count()) / 1000000.0);
        RT_LOGC_WARN(
                TaskLog::TaskMonitor,
                "  Worker ID: {} | Has Parents: {} | Status: {}",
                record.worker,
                record.dependency_generation > 0 ? "yes" : "no",
                record.status == TaskStatus::Completed ? "Completed"
                : record.status == TaskStatus::Failed  ? "Failed"
                                                       : "Other");
        RT_LOGC_WARN(
                TaskLog::TaskMonitor,
                "  Scheduled: {} ns | Started: {} ns | Duration: {:.3f}ms",
                record.scheduled_time.count(),
                record.start_time.count(),
                static_cast<double>(record.duration_ns.count()) / 1000000.0);
    }
}

bool TaskMonitor::push_event(const MonitorEvent &event) {
    return push_event_to_queue(event, event_queue_);
}

void TaskMonitor::check_timeouts() {
    const Nanos current_time = Time::now_ns();
    std::vector<std::uint64_t> timed_out_tasks;

    // Check all tasks with timeouts
    for (const auto &[task_id, task_data] : task_data_) {
        if (!task_data.task_handle.has_value()) {
            continue; // Skip tasks without valid TaskHandle
        }
        const Nanos timeout_ns = task_data.task_handle.value()->get_timeout_ns();
        const bool is_cancelled = task_data.cancelled;

        if (timeout_ns > Nanos{0} && task_data.start_time > Nanos{0} && !is_cancelled) {
            const Nanos elapsed = current_time - task_data.start_time;
            if (elapsed > timeout_ns) {
                timed_out_tasks.push_back(task_id);
            }
        }
    }

    // Cancel timed out tasks
    for (const auto &task_id : timed_out_tasks) {
        handle_cancel_task(create_cancel_task(task_id), task_data_);
        auto data_it = task_data_.find(task_id);
        const std::string task_name =
                (data_it != task_data_.end() && data_it->second.task_handle.has_value())
                        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
                        ? std::string(data_it->second.task_handle.value()->get_task_name())
                        : std::to_string(task_id);
        RT_LOGC_WARN(TaskLog::TaskMonitor, "Task {} timed out", task_name);
    }
}

void TaskMonitor::print_summary() const {
    const std::lock_guard<std::mutex> lock(stats_mutex_);

    if (executions_.empty()) {
        RT_LOGC_WARN(TaskLog::TaskMonitor, "Tried to print summary, but no tasks were executed");
        return;
    }

    RT_LOGC_INFO(TaskLog::TaskMonitor, "====== Task Execution Statistics ======");
    RT_LOGC_INFO(TaskLog::TaskMonitor, "Total tasks executed: {}", executions_.size());

    // Create corrected copy of all executions with jitter correction for
    // dependent tasks
    const auto corrected_executions = apply_jitter_correction_to_dependent_tasks(executions_);

    // Print overall statistics with corrected jitter values
    print_jitter_statistics(corrected_executions);

    // Separate tasks into independent and dependent categories
    std::vector<TaskExecutionRecord> root_tasks;
    std::vector<TaskExecutionRecord> dependent_tasks;

    for (const auto &exec : corrected_executions) {
        if (exec.dependency_generation == 0) { // Root task
            root_tasks.push_back(exec);
        } else {
            dependent_tasks.push_back(exec);
        }
    }

    // Print stats for each category
    print_category_stats(root_tasks, "Root");
    print_category_stats(dependent_tasks, "Dependent");

    print_graph_statistics(corrected_executions);

    RT_LOGC_INFO(TaskLog::TaskMonitor, "======================================");

    RT_LOGC_INFO(
            TaskLog::TaskMonitor,
            "TaskMonitor data structure size: active_tasks={}",
            task_data_.size());
}

std::error_code
TaskMonitor::write_stats_to_file(const std::string &filename, const TraceWriteMode mode) const {
    const std::lock_guard<std::mutex> lock(stats_mutex_);

    if (executions_.empty()) {
        RT_LOGC_WARN(TaskLog::TaskMonitor, "No execution records to write to file: {}", filename);
        return make_error_code(TaskErrc::InvalidParameter);
    }

    const bool append_mode = (mode == TraceWriteMode::Append);
    const bool file_exists = std::filesystem::exists(filename);

    std::ofstream file;
    if (append_mode && file_exists) {
        // Append mode - open for appending
        file.open(filename, std::ios::out | std::ios::app);
        if (!file.is_open()) {
            RT_LOGC_ERROR(TaskLog::TaskMonitor, "Failed to open file for append: {}", filename);
            return make_error_code(TaskErrc::FileOpenFailed);
        }
    } else {
        // Overwrite mode or append to non-existing file
        file.open(filename, std::ios::out | std::ios::trunc);
        if (!file.is_open()) {
            RT_LOGC_ERROR(TaskLog::TaskMonitor, "Failed to open file for writing: {}", filename);
            return make_error_code(TaskErrc::FileOpenFailed);
        }
    }

    try {
        // Write version header as first line (only for new files)
        if (!append_mode || !file_exists) {
            file << R"({"version":"1.0"})" << "\n";
        }

        // Write truncation information if records were truncated
        if (records_truncated_ > 0) {
            file << R"({"warning":"execution_records_truncated",)" << R"("total_records_created":)"
                 << total_records_created_ << "," << R"("records_truncated":)" << records_truncated_
                 << "," << R"("current_records":)" << executions_.size() << "}\n";
        }

        // Create corrected copy of execution records with jitter correction for
        // dependent tasks
        const auto corrected_executions = apply_jitter_correction_to_dependent_tasks(executions_);

        // Write each execution record as a JSON object per line
        for (const auto &record : corrected_executions) {
            file << "{" << R"("task_name":")" << record.task_name << R"(",)" << R"("graph_name":")"
                 << record.graph_name << R"(",)" << R"("dependency_generation":)"
                 << record.dependency_generation << "," << R"("times_scheduled":)"
                 << record.times_scheduled << "," << R"("scheduled_time_ns":)"
                 << record.scheduled_time.count() << "," << R"("worker":)" << record.worker << ","
                 << R"("start_time_ns":)" << record.start_time.count() << "," << R"("end_time_ns":)"
                 << record.end_time.count() << "," << R"("jitter_ns":)" << record.jitter_ns.count()
                 << "," << R"("duration_ns":)" << record.duration_ns.count() << ","
                 << R"("status":")" << ::wise_enum::to_string(record.status) << R"(",)"
                 << R"("was_cancelled":)" << (record.was_cancelled ? "true" : "false") << "}\n";
        }

        file.close();
        RT_LOGC_INFO(
                TaskLog::TaskMonitor,
                "Successfully wrote {} execution records to {} (total "
                "created: {}, truncated: {})",
                corrected_executions.size(),
                filename,
                total_records_created_,
                records_truncated_);
        return make_error_code(TaskErrc::Success);

    } catch (const std::exception &e) {
        RT_LOGC_ERROR(TaskLog::TaskMonitor, "Error writing to file {}: {}", filename, e.what());
        return make_error_code(TaskErrc::FileWriteFailed);
    }
}

std::error_code TaskMonitor::write_chrome_trace_to_file(
        const std::string &filename, const TraceWriteMode mode) const {
    const std::lock_guard<std::mutex> lock(stats_mutex_);

    if (executions_.empty()) {
        RT_LOGC_WARN(TaskLog::TaskMonitor, "No execution records to write to file: {}", filename);
        return make_error_code(TaskErrc::InvalidParameter);
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
                        TaskLog::TaskMonitor,
                        "File too small for Chrome trace format: {}",
                        filename);
                return make_error_code(TaskErrc::FileOpenFailed);
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
                RT_LOGC_ERROR(TaskLog::TaskMonitor, "Invalid Chrome trace format in {}", filename);
                return make_error_code(TaskErrc::FileOpenFailed);
            }

            // Open file and truncate at the "]}" position
            file.open(filename, std::ios::in | std::ios::out);
            if (!file.is_open()) {
                RT_LOGC_ERROR(TaskLog::TaskMonitor, "Failed to open file for append: {}", filename);
                return make_error_code(TaskErrc::FileOpenFailed);
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
                        TaskLog::TaskMonitor, "Failed to open file for writing: {}", filename);
                return make_error_code(TaskErrc::FileOpenFailed);
            }

            // Write Chrome trace format header
            file << R"({"traceEvents":[)" << "\n";
        }

        // Write each execution record as a Chrome trace duration event per line
        for (std::size_t i = 0; i < executions_.size(); ++i) {
            const auto &record = executions_[i];

            // Convert nanoseconds to microseconds (Chrome expects microseconds)
            const double start_us = static_cast<double>(record.start_time.count()) / 1000.0;
            const double duration_us = static_cast<double>(record.duration_ns.count()) / 1000.0;

            // Add comma separator except for first event
            // Note: when appending, we already added comma in the seek logic above
            if (i > 0) {
                file << ",\n";
            }

            // Duration event (X phase)
            file << "{" << R"("name":")" << record.task_name << R"(",)" << R"("cat":")"
                 << record.graph_name << R"(",)" << R"("ph":"X",)" << R"("pid":)" << getpid() << ","
                 << R"("tid":)" << record.worker << "," << R"("ts":)" << std::fixed << start_us
                 << "," << R"("dur":)" << std::fixed << duration_us << "}";
        }

        file << "\n]}\n";
        file.close();

        RT_LOGC_DEBUG(
                TaskLog::TaskMonitor,
                "Successfully wrote {} execution records to Chrome trace "
                "format {} (total "
                "created: {}, truncated: {})",
                executions_.size(),
                filename,
                total_records_created_,
                records_truncated_);
        return make_error_code(TaskErrc::Success);

    } catch (const std::exception &e) {
        RT_LOGC_ERROR(
                TaskLog::TaskMonitor, "Error writing Chrome trace file {}: {}", filename, e.what());
        return make_error_code(TaskErrc::FileWriteFailed);
    }
}

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

} // namespace framework::task
