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
 * @file task_scheduler.hpp
 * @brief High-performance task scheduler with worker threads and category-based
 * queuing
 *
 * Provides a multi-threaded task execution system with support for task
 * categories, worker thread pinning, real-time scheduling, and high-precision
 * timing.
 */

#ifndef FRAMEWORK_TASK_TASK_SCHEDULER_HPP
#define FRAMEWORK_TASK_TASK_SCHEDULER_HPP

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <type_traits>
#include <vector>

#include <wise_enum.h>

#include "task/flat_map.hpp"
#include "task/spinlock.hpp"
#include "task/task.hpp"
#include "task/task_category.hpp"
#include "task/task_errors.hpp"
#include "task/task_graph.hpp"
#include "task/task_monitor.hpp"
#include "task/task_utils.hpp"
#include "task/task_worker.hpp"
#include "task/time.hpp"

namespace framework::task {

// Forward declarations for fluent API
class TaskGraph;
class TaskSchedulerBuilder;

/// Worker thread identifier type
using WorkerId = std::uint32_t;

/**
 * Shutdown behavior when joining workers
 */
enum class WorkerShutdownBehavior {
    /// Complete all pending tasks before shutdown (graceful)
    FinishPendingTasks,
    /// Cancel all pending tasks and shutdown immediately (forced)
    CancelPendingTasks
};

/// Worker startup behavior for constructor
enum class WorkerStartupBehavior {
    AutoStart, //!< Automatically start workers during construction
    Manual     //!< Require explicit start_workers() call
};

} // namespace framework::task

// WISE_ENUM_ADAPT must be called at global namespace scope for ADL to work
WISE_ENUM_ADAPT(framework::task::WorkerShutdownBehavior, FinishPendingTasks, CancelPendingTasks)
WISE_ENUM_ADAPT(framework::task::WorkerStartupBehavior, AutoStart, Manual)

namespace framework::task {

/**
 * Category queue containing priority queue and associated lock
 * Encapsulates a priority queue for a specific task category with thread-safe
 * access
 */
struct CategoryQueue final {
    /**
     * Task comparison for priority queue (dependency generation within same graph
     * first, then scheduled time)
     */
    struct TaskTimeComparator {
        /**
         * Compare TaskHandles by dependency generation (within same graph), then
         * scheduled time
         * @param[in] a First task handle to compare
         * @param[in] b Second task handle to compare
         * @return True if a should be scheduled after b
         */
        bool operator()(const TaskHandle &a, const TaskHandle &b) const noexcept {
            // Primary: If tasks are from same graph, use dependency generation for
            // ordering Only use dependency generation for tie-breaking within the
            // same graph
            if (a->get_graph_name() == b->get_graph_name() &&
                a->get_dependency_generation() != b->get_dependency_generation()) {
                return a->get_dependency_generation() > b->get_dependency_generation();
            }
            // Secondary: Earlier scheduled time has higher priority (applies to all
            // tasks)
            if (a->get_scheduled_time() != b->get_scheduled_time()) {
                return a->get_scheduled_time() > b->get_scheduled_time();
            }
            // Tie-breaker: Lexicographic order by name for deterministic behavior
            return a->get_task_name() > b->get_task_name();
        }
    };

    std::priority_queue<TaskHandle, std::vector<TaskHandle>, TaskTimeComparator>
            queue;   //!< Priority queue for task handles
    Spinlock lock{}; //!< Spinlock for thread-safe access

    /**
     * Reserve capacity in the underlying queue container
     *
     * Preserves existing elements while ensuring the underlying container
     * has at least the specified capacity. Existing elements will be
     * maintained in their proper priority order.
     *
     * @param[in] capacity Number of elements to reserve space for
     */
    void reserve(const std::size_t capacity) {
        std::vector<TaskHandle> container{};
        container.reserve(capacity);

        // Extract all elements from the current queue (no-op if empty)
        while (!queue.empty()) {
            container.push_back(queue.top());
            queue.pop();
        }

        // Create new queue with reserved capacity and existing elements
        queue = std::priority_queue<TaskHandle, std::vector<TaskHandle>, TaskTimeComparator>(
                TaskTimeComparator{}, std::move(container));
    }
};

/**
 * Builder for TaskScheduler configuration
 *
 * Provides a fluent interface for configuring TaskScheduler instances with
 * sensible defaults and method chaining for optional parameters.
 */
class TaskSchedulerBuilder final {
public:
    /// Default task readiness tolerance (300 microseconds)
    static constexpr Nanos DEFAULT_TASK_READINESS_TOLERANCE_NS{300'000};
    /// Default worker sleep duration (10 microseconds)
    static constexpr Nanos DEFAULT_WORKER_SLEEP_NS{10'000};
    /// Default worker blackout warning threshold (250 microseconds)
    static constexpr std::chrono::microseconds DEFAULT_WORKER_BLACKOUT_WARN_THRESHOLD{250};

    /// Default constructor with sensible defaults
    TaskSchedulerBuilder();

    /**
     * Set worker configuration
     * @param[in] config Worker configuration
     * @return Reference to this builder for chaining
     */
    TaskSchedulerBuilder &workers(const WorkersConfig &config);

    /**
     * Set worker configuration with number of workers
     * @param[in] num_workers Number of worker threads
     * @return Reference to this builder for chaining
     */
    TaskSchedulerBuilder &workers(std::size_t num_workers);

    /**
     * Set task readiness tolerance for early task execution
     * @tparam Rep Arithmetic type representing the number of ticks
     * @tparam Period std::ratio representing the tick period
     * @param[in] readiness_tolerance_duration Tolerance window for readiness
     * checking - allows task to be considered ready this amount of time before
     * its scheduled time. This tolerance window accounts for scheduling jitter
     * and ensures tasks can be queued for execution slightly before their exact
     * scheduled time, preventing delays due to timing precision limitations.
     * @return Reference to this builder for chaining
     */
    template <typename Rep, typename Period>
    TaskSchedulerBuilder &
    task_readiness_tolerance(std::chrono::duration<Rep, Period> readiness_tolerance_duration) {
        task_readiness_tolerance_ns_ =
                std::chrono::duration_cast<Nanos>(readiness_tolerance_duration);
        return *this;
    }

    /**
     * Set monitor thread core pinning
     * @param[in] core_id Core ID for monitor thread
     * @return Reference to this builder for chaining
     * @throws std::invalid_argument if core_id >= hardware_concurrency
     */
    TaskSchedulerBuilder &monitor_core(std::uint32_t core_id);

    /**
     * Disable monitor thread core pinning
     * @return Reference to this builder for chaining
     */
    TaskSchedulerBuilder &no_monitor_pinning();

    /**
     * Set worker sleep duration when no tasks are available
     * @tparam Rep Arithmetic type representing the number of ticks
     * @tparam Period std::ratio representing the tick period
     * @param[in] sleep_duration Sleep duration (any std::chrono::duration type)
     * @return Reference to this builder for chaining
     */
    template <typename Rep, typename Period>
    TaskSchedulerBuilder &worker_sleep(std::chrono::duration<Rep, Period> sleep_duration) {
        worker_sleep_ns_ = std::chrono::duration_cast<Nanos>(sleep_duration);
        return *this;
    }

    /**
     * Set worker blackout warning threshold
     *
     * Configures the maximum allowed gap between worker thread polling events
     * before logging a blackout warning. A blackout occurs when a worker thread
     * is blocked or delayed for longer than this threshold, indicating potential
     * performance issues or system contention.
     *
     * @tparam Rep Arithmetic type representing the number of ticks
     * @tparam Period std::ratio representing the tick period
     * @param[in] threshold_duration Maximum gap before warning (any
     * std::chrono::duration type)
     * @return Reference to this builder for chaining
     */
    template <typename Rep, typename Period>
    TaskSchedulerBuilder &
    worker_blackout_warn_threshold(std::chrono::duration<Rep, Period> threshold_duration) {
        worker_blackout_warn_threshold_ns_ = std::chrono::duration_cast<Nanos>(threshold_duration);
        return *this;
    }

    /**
     * Enable automatic worker startup during construction
     * @return Reference to this builder for chaining
     */
    TaskSchedulerBuilder &auto_start();

    /**
     * Require manual worker startup (call start_workers() explicitly)
     * @return Reference to this builder for chaining
     */
    TaskSchedulerBuilder &manual_start();

    /**
     * Set maximum tasks per category for queue preallocation
     * @param[in] tasks_per_category Number of tasks to preallocate per category
     * queue
     * @return Reference to this builder for chaining
     */
    TaskSchedulerBuilder &max_tasks_per_category(std::uint32_t tasks_per_category);

    /**
     * Set maximum execution records for TaskMonitor (omit for auto-calculate to
     * 50GB)
     * @param[in] max_records Maximum records to keep
     * @return Reference to builder for chaining
     */
    TaskSchedulerBuilder &max_execution_records(std::size_t max_records);

    /**
     * Build the TaskScheduler with configured parameters
     * @return Constructed TaskScheduler instance
     * @throws std::invalid_argument if configuration is invalid
     */
    [[nodiscard]] TaskScheduler build();

private:
    WorkersConfig workers_config_;
    Nanos task_readiness_tolerance_ns_{DEFAULT_TASK_READINESS_TOLERANCE_NS};
    std::optional<std::uint32_t> monitor_core_id_;
    Nanos worker_sleep_ns_{DEFAULT_WORKER_SLEEP_NS};
    Nanos worker_blackout_warn_threshold_ns_{
            std::chrono::duration_cast<Nanos>(DEFAULT_WORKER_BLACKOUT_WARN_THRESHOLD)};
    WorkerStartupBehavior startup_behavior_{WorkerStartupBehavior::AutoStart};
    std::optional<std::uint32_t> max_tasks_per_category_;
    std::optional<std::size_t> max_execution_records_; //!< Max records (nullopt = auto-calculate)
};

/**
 * High-performance task scheduler with category-based worker assignment
 *
 * Manages multiple worker threads that execute tasks based on categories,
 * with support for real-time scheduling, core pinning, and precise timing.
 * Uses lock-free monitoring and efficient task distribution.
 *
 * Use TaskSchedulerBuilder to construct instances with the builder pattern.
 */
class TaskScheduler final {
    friend class TaskSchedulerBuilder;

public:
    /**
     * Create a TaskSchedulerBuilder for fluent configuration
     * @return TaskSchedulerBuilder instance for method chaining
     */
    [[nodiscard]] static TaskSchedulerBuilder create();

private:
    /**
     * Private constructor - use TaskSchedulerBuilder::build() to create instances
     * @param[in] workers_config Configuration for all worker threads
     * @param[in] task_readiness_tolerance_ns Tolerance window for readiness
     * checking - allows task to be considered ready this amount of time before
     * its scheduled time (nanoseconds)
     * @param[in] monitor_core_id CPU core for monitor thread (nullopt = no
     * pinning)
     * @param[in] worker_sleep_ns Sleep time for worker threads when no tasks
     * found (nanoseconds)
     * @param[in] worker_blackout_warn_threshold_ns Maximum heartbeat gap before
     * blackout warning (nanoseconds)
     * @param[in] startup_behavior Whether to automatically start workers during
     * construction
     * @param[in] max_tasks_per_category Optional preallocation size for category
     * queues
     * @param[in] max_execution_records Maximum execution records to keep per
     * worker
     */
    explicit TaskScheduler(
            WorkersConfig workers_config,
            const Nanos task_readiness_tolerance_ns,
            const std::optional<std::uint32_t> monitor_core_id,
            const Nanos worker_sleep_ns,
            const Nanos worker_blackout_warn_threshold_ns,
            const WorkerStartupBehavior startup_behavior,
            const std::optional<std::uint32_t> max_tasks_per_category,
            const std::optional<std::size_t> max_execution_records);

public:
    /// Destructor - stops all threads and prints statistics
    ~TaskScheduler();

    // Non-copyable, non-movable
    TaskScheduler(const TaskScheduler &) = delete;
    TaskScheduler &operator=(const TaskScheduler &) = delete;
    TaskScheduler(TaskScheduler &&) = delete;
    TaskScheduler &operator=(TaskScheduler &&) = delete;

    /**
     * Get current worker configuration
     * @return Const reference to workers configuration
     */
    [[nodiscard]] const WorkersConfig &get_workers_config() const noexcept {
        return workers_config_;
    }

    /**
     * Print task monitor execution statistics
     */
    void print_monitor_stats() const { task_monitor_.print_summary(); }

    /**
     * Write task monitor execution statistics to file for later post-processing
     * @param[in] filename Output file path
     * @param[in] mode Write mode (OVERWRITE or APPEND)
     * @return Error code indicating success or failure
     */
    [[nodiscard]] std::error_code write_monitor_stats_to_file(
            const std::string &filename, TraceWriteMode mode = TraceWriteMode::Overwrite) const {
        return task_monitor_.write_stats_to_file(filename, mode);
    }

    /**
     * Write task monitor execution statistics to Chrome trace format file
     * @param[in] filename Output file path
     * @param[in] mode Write mode (OVERWRITE or APPEND)
     * @return Error code indicating success or failure
     */
    [[nodiscard]] std::error_code write_chrome_trace_to_file(
            const std::string &filename, TraceWriteMode mode = TraceWriteMode::Overwrite) const {
        return task_monitor_.write_chrome_trace_to_file(filename, mode);
    }

    /**
     * Schedule a task graph with dependencies
     *
     * @warning This method is NOT thread-safe for concurrent calls with the SAME
     * TaskGraph instance. Multiple threads can safely call schedule()
     * concurrently with DIFFERENT TaskGraph instances, but concurrent calls
     * with the same TaskGraph must be externally synchronized to prevent race
     * conditions in TaskGraph::prepare_tasks().
     *
     * @param[in] graph Task graph containing task specifications (must be built)
     * @param[in] execution_time Base execution time for all tasks (defaults to
     * now)
     * @throws std::runtime_error if graph has not been built
     */
    void schedule(TaskGraph &graph, const Nanos execution_time = Time::now_ns());

    /**
     * Schedule a task graph with dependencies (template version)
     *
     * @warning This method is NOT thread-safe for concurrent calls with the SAME
     * TaskGraph instance. Multiple threads can safely call schedule()
     * concurrently with DIFFERENT TaskGraph instances, but concurrent calls
     * with the same TaskGraph must be externally synchronized to prevent race
     * conditions in TaskGraph::prepare_tasks().
     *
     * @tparam Rep Arithmetic type representing the number of ticks
     * @tparam Period std::ratio representing the tick period
     * @param[in] graph Task graph containing task specifications (must be built)
     * @param[in] execution_time_duration Base execution time for all tasks (any
     * std::chrono::duration type)
     * @throws std::runtime_error if graph has not been built
     */
    template <typename Rep, typename Period>
    void schedule(TaskGraph &graph, std::chrono::duration<Rep, Period> execution_time_duration) {
        schedule(graph, std::chrono::duration_cast<Nanos>(execution_time_duration));
    }

    /**
     * Start all worker threads and wait for them to be ready
     * Should be called after constructor before scheduling tasks
     */
    void start_workers();

    /**
     * Join all worker threads after setting stop flag
     * Should be called to cleanly shutdown workers
     * @param[in] behavior How to handle pending tasks during shutdown
     */
    void join_workers(WorkerShutdownBehavior behavior = WorkerShutdownBehavior::FinishPendingTasks);

private:
    Nanos task_readiness_tolerance_ns_{}; //!< Tolerance window for readiness
                                          //!< checking
    Nanos worker_sleep_ns_{};             //!< Sleep time for worker threads when no tasks found
    Nanos worker_blackout_warn_threshold_ns_{};    //!< Maximum heartbeat gap before
                                                   //!< blackout warning
    std::optional<std::uint32_t> monitor_core_id_; //!< Core ID for task monitor pinning
    // NOLINTBEGIN(readability-redundant-member-init) - {} is required for std::atomic zero-init
    std::atomic<bool> stop_flag_{};            //!< Thread stop flag
    std::atomic<std::size_t> workers_ready_{}; //!< Number of workers ready
    // NOLINTEND(readability-redundant-member-init)
    TaskMonitor task_monitor_;         //!< Lock-free task monitor
    WorkersConfig workers_config_;     //!< Worker configuration
    std::vector<std::thread> workers_; //!< Worker threads
    FlatMap<TaskCategory, std::unique_ptr<CategoryQueue>>
            category_queues_; //!< Category-based task queues

    /**
     * Worker thread main function
     * @param[in] worker_index Worker index (0-based)
     */
    void worker_function(const std::size_t worker_index);

    /**
     * Configure and start all worker threads (internal implementation)
     */
    void start_workers_impl();

    /**
     * Configure individual worker thread
     * @param[in] worker_index Worker index
     * @param[in] config Worker configuration
     * @return Error code indicating success or failure
     */
    [[nodiscard]] std::error_code
    configure_worker(const std::size_t worker_index, const WorkerConfig &config);

    /**
     * Initialize category queues and reserve capacity for all task categories
     * @param[in] tasks_per_queue Number of tasks to reserve space for in each
     * queue
     */
    void preallocate_category_queues(std::uint32_t tasks_per_queue);
};

} // namespace framework::task

#endif // FRAMEWORK_TASK_TASK_SCHEDULER_HPP
