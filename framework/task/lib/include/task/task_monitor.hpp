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
 * @file task_monitor.hpp
 * @brief Task monitoring system for tracking execution statistics and
 * visualization
 *
 * Provides lock-free monitoring of task execution with event-based tracking,
 * performance statistics, timeout detection, and dependency visualization.
 */

#ifndef FRAMEWORK_TASK_TASK_MONITOR_HPP
#define FRAMEWORK_TASK_TASK_MONITOR_HPP

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

#include <wise_enum.h>

#include "task/bounded_queue.hpp"
#include "task/flat_map.hpp"
#include "task/task.hpp"
#include "task/task_errors.hpp"
#include "task/task_utils.hpp"
#include "task/time.hpp"

namespace framework::task {

using WorkerId = std::uint32_t;

/// Task monitor event types for tracking task lifecycle
enum class MonitorEventType {
    RegisterTask, //!< Task registration event
    RecordStart,  //!< Task execution start
    RecordEnd,    //!< Task execution end
    CancelTask    //!< Task cancellation request
};

} // namespace framework::task

// WISE_ENUM_ADAPT must be called at global namespace scope for ADL to work
WISE_ENUM_ADAPT(framework::task::MonitorEventType, RegisterTask, RecordStart, RecordEnd, CancelTask)

namespace framework::task {

/**
 * Simplified monitor event for lock-free communication
 * Uses TaskHandle for registration, task_id for runtime events
 */
struct MonitorEvent final {
    MonitorEventType type{MonitorEventType::RegisterTask}; //!< Event type
    Nanos timestamp{};                                     //!< Event timestamp

    // Task identification
    std::uint64_t task_id{};               //!< Task ID (used for start/end/cancel events)
    std::optional<TaskHandle> task_handle; //!< Handle to task (used for RegisterTask events only)

    // Event-specific data (only relevant fields used per event type)
    WorkerId worker{};                        //!< Worker ID (for start/end events)
    TaskStatus status{TaskStatus::Completed}; //!< Status (for end events)
};

/// Consolidated task monitoring data
struct TaskMonitorData final {
    // Task information (all metadata accessed via TaskHandle)
    std::optional<TaskHandle> task_handle; //!< Handle to task (contains all task metadata)

    // Runtime monitoring data (unique to monitoring)
    Nanos start_time{};    //!< Actual execution start time
    WorkerId worker{};     //!< Worker assignment
    bool cancelled{false}; //!< Cancellation flag

    /// Default constructor (for container compatibility)
    TaskMonitorData() = default;

    /**
     * Constructor from TaskHandle
     * @param[in] handle Task handle to monitor
     */
    explicit TaskMonitorData(const TaskHandle &handle) : task_handle{handle} {}
};

/**
 * Task execution record for detailed statistics
 * Stores comprehensive execution data for performance analysis
 */
struct TaskExecutionRecord final {
    // Task information (metadata accessed via TaskHandle when needed)
    std::string task_name;                 //!< Task name (copied for persistence)
    std::string graph_name;                //!< Graph name (copied for persistence)
    std::uint32_t dependency_generation{}; //!< Dependency generation (copied for
                                           //!< persistence)
    std::uint64_t times_scheduled{};       //!< Number of times the task's graph has
                                           //!< been scheduled (copied for persistence)
    Nanos scheduled_time{};                //!< Originally scheduled time (copied for persistence)

    // Execution-specific data
    WorkerId worker{};                        //!< Worker that executed task
    Nanos start_time{};                       //!< Actual start time
    Nanos end_time{};                         //!< Completion time
    Nanos jitter_ns{};                        //!< Scheduling jitter
    Nanos duration_ns{};                      //!< Execution duration
    TaskStatus status{TaskStatus::Completed}; //!< Final execution status
    bool was_cancelled{};                     //!< Whether task was cancelled
};

/**
 * Lock-free task monitor using producer-consumer pattern
 *
 * Provides non-blocking task monitoring with event-based communication,
 * performance statistics, timeout detection, and dependency visualization.
 */
class TaskMonitor final {
public:
    /// Default monitoring sleep duration in microseconds
    static constexpr std::chrono::microseconds DEFAULT_MONITOR_SLEEP_US{1000};
    /**
     * Constructor initializes monitor
     *
     * @note Thread Safety: Not thread-safe. Must be called from single thread.
     *
     * @param[in] max_execution_records Maximum number of execution records to
     * keep (nullopt for unlimited)
     */
    explicit TaskMonitor(std::optional<std::size_t> max_execution_records = std::nullopt);

    /**
     * Destructor ensures clean shutdown
     *
     * Automatically calls stop() to ensure background thread is properly
     * terminated before object destruction. Safe to call even if stop()
     * was already called explicitly.
     *
     * @note Thread Safety: Safe. Multiple calls to stop() are handled gracefully.
     */
    ~TaskMonitor() noexcept;

    // Non-copyable, non-movable
    TaskMonitor(const TaskMonitor &) = delete;
    TaskMonitor &operator=(const TaskMonitor &) = delete;
    TaskMonitor(TaskMonitor &&) = delete;
    TaskMonitor &operator=(TaskMonitor &&) = delete;

    /**
     * Start monitoring thread
     *
     * @note Thread Safety: Not thread-safe. Must be called from single thread
     * before any monitoring operations begin.
     *
     * @param[in] core_id CPU core to pin monitor thread to (nullopt for no
     * pinning)
     * @param[in] sleep_duration Sleep duration between monitoring cycles
     * @return Error code indicating success or failure
     */
    [[nodiscard]] std::error_code
    start(std::optional<std::uint32_t> core_id = std::nullopt,
          std::chrono::microseconds sleep_duration = DEFAULT_MONITOR_SLEEP_US);

    /**
     * Stop monitoring thread
     *
     * Blocks until background thread terminates. Safe to call multiple times.
     * Automatically called by destructor.
     *
     * @note Thread Safety: Can be called safely from multiple threads.
     * Subsequent calls after first are no-ops.
     */
    void stop() noexcept;

    /**
     * Clear execution statistics
     *
     * Clears all stored execution records and resets statistics counters.
     * Task registration data is preserved for continued monitoring.
     *
     * @note Thread Safety: Thread-safe. Uses mutex protection and can be called
     * safely while monitoring is active. Only clears execution history, leaving
     * active task monitoring data intact.
     */
    void clear_stats();

    /**
     * Register task for monitoring (real-time safe)
     *
     * TaskHandle is copied, so no lifetime concerns for the caller.
     *
     * @note Thread Safety: Thread-safe and real-time safe. Uses lock-free queue
     * for communication with background thread. Can be called concurrently from
     * multiple threads.
     *
     * @param[in] task_handle TaskHandle to register for monitoring
     * @return Error code indicating success or failure
     */
    [[nodiscard]] std::error_code register_task(const TaskHandle &task_handle);

    /**
     * Record task execution start (real-time safe)
     *
     * @note Thread Safety: Thread-safe and real-time safe. Uses lock-free queue
     * for communication with background thread. Can be called concurrently from
     * multiple threads.
     *
     * @param[in] task_id Task ID (guaranteed unique 64-bit identifier)
     * @param[in] worker_id Worker executing the task
     * @param[in] start_time Execution start timestamp
     * @return Error code indicating success or failure
     */
    [[nodiscard]] std::error_code
    record_start(std::uint64_t task_id, WorkerId worker_id, Nanos start_time);

    /**
     * Record task execution completion (real-time safe)
     *
     * @note Thread Safety: Thread-safe and real-time safe. Uses lock-free queue
     * for communication with background thread. Can be called concurrently from
     * multiple threads.
     *
     * @param[in] task_id Task ID (guaranteed unique 64-bit identifier)
     * @param[in] end_time Completion timestamp
     * @param[in] status Final execution status
     * @return Error code indicating success or failure
     */
    [[nodiscard]] std::error_code
    record_end(std::uint64_t task_id, Nanos end_time, TaskStatus status = TaskStatus::Completed);

    /**
     * Cancel a task (real-time safe)
     *
     * @note Thread Safety: Thread-safe and real-time safe. Uses lock-free queue
     * for communication with background thread. Can be called concurrently from
     * multiple threads.
     *
     * @param[in] task_id Task ID (guaranteed unique 64-bit identifier)
     * @return Error code indicating success or failure
     */
    [[nodiscard]] std::error_code cancel_task(std::uint64_t task_id);

    /**
     * Print execution statistics summary
     *
     * @note Thread Safety: Thread-safe. Uses mutex protection and can be called
     * safely while monitoring is active.
     */
    void print_summary() const;

    /**
     * Write execution statistics to JSON file for later post-processing
     * Each execution record is written as one JSON object per line
     *
     * @note Thread Safety: Thread-safe. Uses mutex protection and can be called
     * safely while monitoring is active.
     *
     * @param[in] filename Output file path
     * @param[in] mode Write mode (OVERWRITE or APPEND)
     * @return Error code indicating success or failure
     */
    [[nodiscard]] std::error_code write_stats_to_file(
            const std::string &filename, TraceWriteMode mode = TraceWriteMode::Overwrite) const;

    /**
     * Write execution statistics to Chrome trace format file
     * Each execution record is written as one Chrome trace event per line
     *
     * @note Thread Safety: Thread-safe. Uses mutex protection and can be called
     * safely while monitoring is active.
     *
     * @param[in] filename Output file path
     * @param[in] mode Write mode (OVERWRITE or APPEND)
     * @return Error code indicating success or failure
     */
    [[nodiscard]] std::error_code write_chrome_trace_to_file(
            const std::string &filename, TraceWriteMode mode = TraceWriteMode::Overwrite) const;

private:
    /**
     * Print graph-level execution statistics
     * Groups tasks by graph name and scheduling round
     * @param[in] executions Execution records to analyze
     */
    static void print_graph_statistics(const std::vector<TaskExecutionRecord> &executions);

    /**
     * Print category-specific statistics
     * @param[in] tasks Tasks to analyze
     * @param[in] category_name Category name for logging
     */
    static void print_category_stats(
            const std::vector<TaskExecutionRecord> &tasks, const std::string_view category_name);
    BoundedQueue<MonitorEvent> event_queue_; //!< Lock-free event queue
    std::thread monitor_thread_;             //!< Monitor processing thread
    // NOLINTBEGIN(readability-redundant-member-init) - {} is required for std::atomic zero-init
    std::atomic<bool> stop_flag_{}; //!< Thread stop flag
    // NOLINTEND(readability-redundant-member-init)

    FlatMap<std::uint64_t, TaskMonitorData> task_data_; //!< Consolidated task monitoring data by ID
    std::vector<TaskExecutionRecord> executions_;       //!< Execution history
    mutable std::mutex stats_mutex_; //!< Protects executions_ for thread-safe access

    /// Record management
    std::size_t max_execution_records_{0};   //!< Maximum records to keep
    std::uint64_t total_records_created_{0}; //!< Total records created (including truncated)
    std::uint64_t records_truncated_{0};     //!< Count of records removed due to limits

    /**
     * Monitor thread main function
     * @param[in] sleep_duration Sleep duration between monitoring cycles
     */
    void monitor_function(std::chrono::microseconds sleep_duration);

    /**
     * Process pending events in batch
     * @return Number of events processed
     */
    std::size_t process_events();

    /**
     * Handle execution end event
     * @param[in] event End event
     */
    void handle_record_end(const MonitorEvent &event);

    /**
     * Push event to queue (non-blocking)
     * @param[in] event Event to push
     * @return true if pushed successfully, false if queue full
     */
    bool push_event(const MonitorEvent &event);

    /// Check for timed out tasks and cancel them
    void check_timeouts();
};

} // namespace framework::task

#endif // FRAMEWORK_TASK_TASK_MONITOR_HPP
