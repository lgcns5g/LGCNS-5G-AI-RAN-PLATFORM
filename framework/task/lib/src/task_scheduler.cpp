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
#include <compare>
#include <cstdint>
#include <cstring>
#include <format>
#include <functional>
#include <memory>
#include <optional>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include <parallel_hashmap/phmap.h>
#include <quill/LogMacros.h>

#include <wise_enum.h>

#include "log/rt_log_macros.hpp"
#include "task/flat_map.hpp"
#include "task/spinlock.hpp"
#include "task/task.hpp"
#include "task/task_category.hpp"
#include "task/task_errors.hpp"
#include "task/task_graph.hpp"
#include "task/task_log.hpp"
#include "task/task_monitor.hpp"
#include "task/task_scheduler.hpp"
#include "task/task_utils.hpp"
#include "task/task_worker.hpp"
#include "task/time.hpp"

namespace framework::task {

namespace {

/**
 * Schedule a task handle to execute at a specific time
 * Resets the task and updates its execution time before scheduling
 * @param[in] task_ptr Shared pointer to task to schedule
 * @param[in] task_monitor Task monitor for registration
 * @param[in] category_queues Category queues for task scheduling
 */
void schedule_at(
        const std::shared_ptr<Task> &task_ptr,
        TaskMonitor &task_monitor,
        const FlatMap<TaskCategory, std::unique_ptr<CategoryQueue>> &category_queues) {

    // Check for null task pointer
    if (!task_ptr) {
        RT_LOGC_ERROR(TaskLog::TaskScheduler, "Cannot schedule null task pointer");
        return;
    }

    // Get task info for registration and queue management
    const TaskCategory category = task_ptr->get_category();

    // Register with monitor
    const std::error_code register_result = task_monitor.register_task(TaskHandle(task_ptr));
    if (register_result) {
        RT_LOGC_ERROR(
                TaskLog::TaskScheduler, "Failed to register task {}", task_ptr->get_task_name());
    }

    // Add to appropriate category queue - no allocation needed
    auto queue_it = category_queues.find(category);
    if (queue_it != category_queues.end()) {
        const SpinlockGuard lock{queue_it->second->lock};
        // Emplace constructs TaskHandle in place from shared_ptr
        queue_it->second->queue.emplace(task_ptr);
    }
}

/**
 * Check for blackout condition and log warning if detected
 * @param[in] worker_id Worker identifier for logging
 * @param[in] blackout_threshold_ns Blackout warning threshold (nanoseconds)
 * @param[in,out] last_heartbeat_ns Last heartbeat timestamp (updated)
 * @param[in,out] last_operation Last operation description (updated)
 */
void check_and_log_blackout(
        const WorkerId worker_id,
        const std::int64_t blackout_threshold_ns,
        std::int64_t &last_heartbeat_ns,
        std::string_view &last_operation) {
    const std::int64_t now_ns = Time::now_ns().count();

    if (last_heartbeat_ns > 0) {
        const std::int64_t gap_ns = now_ns - last_heartbeat_ns;
        if (gap_ns > blackout_threshold_ns) {
            const std::int64_t gap_us = gap_ns / 1000;
            RT_LOGC_WARN(
                    TaskLog::TaskScheduler,
                    "Worker {} BLACKOUT: {} us gap, last operation: {}",
                    worker_id,
                    gap_us,
                    last_operation);
        }
    }
    last_heartbeat_ns = now_ns;
    last_operation = "scanning_categories";
}

/**
 * Validate that all task categories in graph are handled by at least one worker
 * Logs warnings for unhandled categories
 * @param[in] workers_config Worker configuration
 * @param[in] graph Task graph to validate
 */
void validate_task_categories(const WorkersConfig &workers_config, const TaskGraph &graph) {
    // Only validate on first schedule
    if (graph.get_times_scheduled() > 0) {
        return;
    }

    // Check each task's category against worker categories
    const auto &task_specs = graph.get_task_specs();
    for (const auto &task_spec : task_specs) {
        bool category_handled = false;

        // Check if any worker handles this category
        for (const auto &worker_config : workers_config.workers) {
            const auto it = std::find(
                    worker_config.categories.begin(),
                    worker_config.categories.end(),
                    task_spec.category);
            if (it != worker_config.categories.end()) {
                category_handled = true;
                break;
            }
        }

        if (!category_handled) {
            RT_LOGC_WARN(
                    TaskLog::TaskScheduler,
                    "Task '{}' in graph '{}' has category '{}' which is not handled by any "
                    "worker. Task will never execute!",
                    task_spec.task_name,
                    graph.get_graph_name(),
                    task_spec.category.name());
        }
    }
}

/**
 * Find the next ready task from assigned categories
 * @param[in] categories Categories this worker can process
 * @param[in] category_queues All category queues
 * @param[in] task_readiness_tolerance_ns Readiness tolerance window
 * @param[in,out] task_monitor Task monitor for cancellation recording
 * @return Ready task handle or nullopt if none found
 */
std::optional<TaskHandle> find_ready_task(
        const std::vector<TaskCategory> &categories,
        const FlatMap<TaskCategory, std::unique_ptr<CategoryQueue>> &category_queues,
        const Nanos task_readiness_tolerance_ns,
        TaskMonitor &task_monitor) {

    for (const auto &category : categories) {
        auto queue_it = category_queues.find(category);
        if (queue_it == category_queues.end()) {
            continue; // Category not found
        }

        CategoryQueue &category_queue = *queue_it->second;

        // Try to get lock without blocking
        const SpinlockTryGuard try_lock{category_queue.lock};
        if (!try_lock) {
            continue; // Skip if locked, try next category
        }

        // Check if queue has ready tasks
        if (category_queue.queue.empty()) {
            continue;
        }
        const TaskHandle &candidate = category_queue.queue.top();

        // Quick readiness check
        const Nanos now = Time::now_ns();
        const auto is_cancelled = candidate->is_cancelled();
        const auto candidate_status = candidate->status();
        const auto is_running = candidate_status == TaskStatus::Running;
        const auto is_completed = candidate_status == TaskStatus::Completed;
        const auto is_failed = candidate_status == TaskStatus::Failed;
        const auto is_ready = candidate->is_ready(now, task_readiness_tolerance_ns);
        const auto no_parents = candidate->has_no_parents();
        const auto all_parents_completed = candidate->all_parents_match(
                [](TaskStatus status) { return status == TaskStatus::Completed; });
        const auto any_parent_cancelled = candidate->any_parent_matches(
                [](TaskStatus status) { return status == TaskStatus::Cancelled; });
        const auto any_parent_failed = candidate->any_parent_matches(
                [](TaskStatus status) { return status == TaskStatus::Failed; });

        if (is_cancelled || is_running || is_completed || is_failed) {
            RT_LOGC_WARN(
                    TaskLog::TaskScheduler,
                    "Found task {} in state {} in pending worker queue.  "
                    "Removing it.",
                    candidate->get_task_name(),
                    ::wise_enum::to_string(candidate->status()));
            category_queue.queue.pop();
            continue; // Skip execution, continue to next iteration
        }

        if (any_parent_cancelled || any_parent_failed) {
            // The candidate task was dependent on the parent, so cancel it and
            // pop it off the queue
            candidate->set_status(TaskStatus::Cancelled);
            category_queue.queue.pop();
            const std::error_code cancel_result =
                    task_monitor.cancel_task(candidate->get_task_id());
            if (cancel_result) {
                RT_LOGC_WARN(
                        TaskLog::TaskScheduler,
                        "Failed to record task cancellation for {}",
                        candidate->get_task_name());
            }
            continue; // Skip execution, continue to next iteration
        }

        if (is_ready && (no_parents || all_parents_completed)) {
            // Take the task - copy it before popping from queue
            TaskHandle task_copy = candidate;
            category_queue.queue.pop();
            return task_copy; // Stop scanning categories
        }
    }

    return std::nullopt;
}

/**
 * Execute task with full monitoring and error handling
 * @param[in,out] task_handle Task to execute
 * @param[in] worker_id Worker identifier
 * @param[in,out] task_monitor Task monitor for recording
 * @param[in,out] last_heartbeat_ns Heartbeat timestamp (updated after
 * execution)
 * @param[in,out] last_operation Operation description (updated)
 */
void execute_task_with_monitoring(
        TaskHandle &task_handle,
        const WorkerId worker_id,
        TaskMonitor &task_monitor,
        std::int64_t &last_heartbeat_ns,
        std::string_view &last_operation) {
    const std::string_view task_name = task_handle->get_task_name();

    // Wait until scheduled time
    const Nanos scheduled_time = task_handle->get_scheduled_time();
    if (scheduled_time > Nanos{0}) {
        Time::sleep_until(scheduled_time);
    }

    last_operation = "executing_task";

    // Record start time
    const Nanos start_time = Time::now_ns();
    const std::error_code start_result =
            task_monitor.record_start(task_handle->get_task_id(), worker_id, start_time);
    if (start_result) {
        RT_LOGC_WARN(TaskLog::TaskScheduler, "Failed to record task start for {}", task_name);
    }

    // Execute the task
    const TaskResult result = task_handle->execute();
    const Nanos end_time = Time::now_ns();

    // Determine final status and record end
    const TaskStatus final_status =
            task_handle->is_cancelled() ? TaskStatus::Cancelled : result.status;

    const std::error_code end_result =
            task_monitor.record_end(task_handle->get_task_id(), end_time, final_status);
    task_handle->set_status(final_status);

    if (end_result) {
        RT_LOGC_WARN(TaskLog::TaskScheduler, "Failed to record task end for {}", task_name);
    }

    if (!result.is_success()) {
        RT_LOGC_WARN(TaskLog::TaskScheduler, "Task {} failed: {}", task_name, result.message);
    }

    // Update heartbeat after task completion
    last_heartbeat_ns = Time::now_ns().count();
    last_operation = "task_completed";
}

/**
 * Wait for all category queues to become empty
 * @param[in] category_queues Category queues to monitor
 */
void wait_for_queues_empty(
        const FlatMap<TaskCategory, std::unique_ptr<CategoryQueue>> &category_queues) {
    bool all_queues_empty = false;
    while (!all_queues_empty) {
        all_queues_empty = true;
        for (const auto &[category, queue_ptr] : category_queues) {
            const SpinlockGuard lock{
                    queue_ptr->lock}; // Use blocking lock to ensure we check every queue
            if (!queue_ptr->queue.empty()) {
                all_queues_empty = false;
                break;
            }
        }
        if (!all_queues_empty) {
            static constexpr std::int64_t QUEUE_POLL_INTERVAL_MS = 10;
            std::this_thread::sleep_for(std::chrono::milliseconds(QUEUE_POLL_INTERVAL_MS));
        }
    }
}

/**
 * Cancel all pending tasks in all category queues
 * @param[in] category_queues Category queues to process
 * @param[in,out] task_monitor Task monitor for cancellation recording
 * @return Number of tasks cancelled
 */
std::size_t cancel_pending_tasks(
        const FlatMap<TaskCategory, std::unique_ptr<CategoryQueue>> &category_queues,
        TaskMonitor &task_monitor) {
    std::size_t cancelled_tasks = 0;
    for (const auto &[category, queue_ptr] : category_queues) {
        const SpinlockGuard lock{queue_ptr->lock};
        while (!queue_ptr->queue.empty()) {
            const auto &task_handle = queue_ptr->queue.top();
            task_handle->set_status(TaskStatus::Cancelled);

            const std::error_code cancel_result =
                    task_monitor.cancel_task(task_handle->get_task_id());
            if (cancel_result) {
                RT_LOGC_WARN(
                        TaskLog::TaskScheduler,
                        "Failed to record task cancellation for {}",
                        task_handle->get_task_name());
            }

            queue_ptr->queue.pop();
            cancelled_tasks++;
        }
    }
    return cancelled_tasks;
}

} // anonymous namespace

// TaskSchedulerBuilder implementation

TaskSchedulerBuilder::TaskSchedulerBuilder() : workers_config_{WorkersConfig{}} {}

TaskSchedulerBuilder &TaskSchedulerBuilder::workers(const WorkersConfig &config) {
    workers_config_ = config;
    return *this;
}

TaskSchedulerBuilder &TaskSchedulerBuilder::workers(const std::size_t num_workers) {
    workers_config_ = WorkersConfig{num_workers};
    return *this;
}

TaskSchedulerBuilder &TaskSchedulerBuilder::monitor_core(const std::uint32_t core_id) {
    const auto max_cores = std::thread::hardware_concurrency();
    if (core_id >= max_cores) {
        const std::string error_msg = std::format(
                "Invalid monitor core ID {}: system has {} cores (0-{})",
                core_id,
                max_cores,
                max_cores - 1);
        RT_LOGC_ERROR(TaskLog::TaskScheduler, "{}", error_msg);
        throw std::invalid_argument(error_msg);
    }
    monitor_core_id_ = core_id;
    return *this;
}

TaskSchedulerBuilder &TaskSchedulerBuilder::no_monitor_pinning() {
    monitor_core_id_ = std::nullopt;
    return *this;
}

TaskSchedulerBuilder &TaskSchedulerBuilder::auto_start() {
    startup_behavior_ = WorkerStartupBehavior::AutoStart;
    return *this;
}

TaskSchedulerBuilder &TaskSchedulerBuilder::manual_start() {
    startup_behavior_ = WorkerStartupBehavior::Manual;
    return *this;
}

TaskSchedulerBuilder &
TaskSchedulerBuilder::max_tasks_per_category(const std::uint32_t tasks_per_category) {
    max_tasks_per_category_ = tasks_per_category;
    return *this;
}

TaskSchedulerBuilder &TaskSchedulerBuilder::max_execution_records(const std::size_t max_records) {
    max_execution_records_ = max_records;
    return *this;
}

TaskScheduler TaskSchedulerBuilder::build() {
    return TaskScheduler{
            workers_config_,
            task_readiness_tolerance_ns_,
            monitor_core_id_,
            worker_sleep_ns_,
            worker_blackout_warn_threshold_ns_,
            startup_behavior_,
            max_tasks_per_category_,
            max_execution_records_};
}

TaskSchedulerBuilder TaskScheduler::create() { return TaskSchedulerBuilder{}; }

// TaskScheduler implementation

TaskScheduler::TaskScheduler(
        WorkersConfig workers_config,
        const Nanos task_readiness_tolerance_ns,
        const std::optional<std::uint32_t> monitor_core_id,
        const Nanos worker_sleep_ns,
        const Nanos worker_blackout_warn_threshold_ns,
        const WorkerStartupBehavior startup_behavior,
        const std::optional<std::uint32_t> max_tasks_per_category,
        const std::optional<std::size_t> max_execution_records)
        : task_readiness_tolerance_ns_{task_readiness_tolerance_ns},
          worker_sleep_ns_{worker_sleep_ns},
          worker_blackout_warn_threshold_ns_{worker_blackout_warn_threshold_ns},
          monitor_core_id_{monitor_core_id}, task_monitor_{max_execution_records},
          workers_config_{std::move(workers_config)} {
    RT_LOGC_INFO(
            TaskLog::TaskScheduler,
            "Initializing TaskScheduler with {} workers, threshold {}ns",
            workers_config_.size(),
            task_readiness_tolerance_ns.count());

    // Validate configuration
    if (!workers_config_.is_valid()) {
        RT_LOGC_ERROR(TaskLog::TaskScheduler, "Invalid worker configuration");
        throw std::invalid_argument("Invalid worker configuration");
    }

    workers_config_.print();

    // Initialize and preallocate category queues (default 50 tasks per queue)
    constexpr std::uint32_t DEFAULT_TASKS_PER_QUEUE = 50;
    preallocate_category_queues(max_tasks_per_category.value_or(DEFAULT_TASKS_PER_QUEUE));

    // Start worker threads if requested
    if (startup_behavior == WorkerStartupBehavior::AutoStart) {
        start_workers();
        RT_LOGC_INFO(
                TaskLog::TaskScheduler,
                "TaskScheduler initialization complete with auto-started workers");
    } else {
        RT_LOGC_INFO(
                TaskLog::TaskScheduler,
                "TaskScheduler initialization complete - workers not started");
    }
}

TaskScheduler::~TaskScheduler() {
    RT_LOGC_INFO(TaskLog::TaskScheduler, "Shutting down TaskScheduler");

    // Join workers if any are running
    if (workers_ready_.load() > 0) {
        join_workers();
    }

    // Ensure TaskMonitor is fully stopped before destruction
    task_monitor_.stop();

    RT_LOGC_INFO(TaskLog::TaskScheduler, "TaskScheduler shutdown complete");
}

void TaskScheduler::worker_function(const std::size_t worker_index) {
    const auto id = static_cast<WorkerId>(worker_index + 1); // 1-based worker IDs

    RT_LOGC_DEBUG(TaskLog::TaskScheduler, "Worker {} starting", id);

    // Signal that this worker is ready
    workers_ready_.fetch_add(1);

    // Blackout detection variables
    std::int64_t last_heartbeat_ns = 0;
    std::string_view last_operation = "starting";

    while (!stop_flag_.load(std::memory_order_acquire)) {
        // Check for blackouts and update heartbeat
        check_and_log_blackout(
                id, worker_blackout_warn_threshold_ns_.count(), last_heartbeat_ns, last_operation);

        // Get the categories this worker is assigned to from config
        const auto &categories = workers_config_[worker_index].categories;

        // Find a ready task from assigned categories
        auto task_to_run = find_ready_task(
                categories, category_queues_, task_readiness_tolerance_ns_, task_monitor_);

        // Early continue if no task found
        if (!task_to_run.has_value()) {
            last_operation = "sleeping_no_tasks";
            std::this_thread::sleep_for(worker_sleep_ns_);
            continue;
        }

        // Execute the task with full monitoring
        execute_task_with_monitoring(
                task_to_run.value(), id, task_monitor_, last_heartbeat_ns, last_operation);
    }

    RT_LOGC_DEBUG(TaskLog::TaskScheduler, "Worker {} stopping", id);
}

void TaskScheduler::start_workers() {
    if (workers_ready_.load() > 0) {
        RT_LOGC_WARN(TaskLog::TaskScheduler, "Workers already started, ignoring call");
        return;
    }

    workers_ready_.store(0);

    start_workers_impl();

    // Wait for all workers to be ready
    const std::size_t expected_workers = workers_config_.size();
    RT_LOGC_DEBUG(TaskLog::TaskScheduler, "Waiting for {} workers to be ready", expected_workers);

    while (workers_ready_.load() < expected_workers) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    RT_LOGC_DEBUG(TaskLog::TaskScheduler, "All {} workers are ready", expected_workers);
}

void TaskScheduler::join_workers(const WorkerShutdownBehavior behavior) {
    if (workers_ready_.load() == 0) {
        RT_LOGC_DEBUG(TaskLog::TaskScheduler, "No workers running, nothing to join");
        return;
    }

    RT_LOGC_DEBUG(
            TaskLog::TaskScheduler,
            "Joining worker threads ({})",
            behavior == WorkerShutdownBehavior::FinishPendingTasks ? "finishing pending tasks"
                                                                   : "cancelling pending tasks");

    // Handle pending tasks based on behavior
    if (behavior == WorkerShutdownBehavior::FinishPendingTasks) {
        wait_for_queues_empty(category_queues_);
        RT_LOGC_DEBUG(TaskLog::TaskScheduler, "All queues empty, stopping workers");
    } else {
        const std::size_t cancelled_tasks = cancel_pending_tasks(category_queues_, task_monitor_);
        if (cancelled_tasks > 0) {
            RT_LOGC_INFO(TaskLog::TaskScheduler, "Cancelled {} pending tasks", cancelled_tasks);
        }
    }

    // Common shutdown logic: stop workers and join
    stop_flag_.store(true, std::memory_order_release);

    for (auto &worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }

    // Reset state for potential restart
    workers_ready_.store(0);
    stop_flag_.store(false, std::memory_order_release);
    workers_.clear();

    // Stop the task monitor
    task_monitor_.stop();

    RT_LOGC_INFO(TaskLog::TaskScheduler, "All worker threads joined");
}

void TaskScheduler::start_workers_impl() {
    RT_LOGC_INFO(TaskLog::TaskScheduler, "Starting {} worker threads", workers_config_.size());

    // Start task monitor
    const std::error_code monitor_start_result = task_monitor_.start(monitor_core_id_);
    if (monitor_start_result) {
        RT_LOGC_ERROR(
                TaskLog::TaskScheduler,
                "Failed to start task monitor: {}",
                get_error_name(monitor_start_result));
    }

    workers_.reserve(workers_config_.size());
    for (std::size_t i = 0; i < workers_config_.size(); ++i) {
        workers_.emplace_back(&TaskScheduler::worker_function, this, i);
    }

    // Configure each worker thread
    for (std::size_t i = 0; i < workers_.size(); ++i) {
        const std::error_code config_result = configure_worker(i, workers_config_[i]);
        if (config_result) {
            RT_LOGC_ERROR(
                    TaskLog::TaskScheduler,
                    "Failed to configure worker {}: {}",
                    i,
                    get_error_name(config_result));
        }
    }

    RT_LOGC_INFO(TaskLog::TaskScheduler, "All worker threads started and configured");
}

std::error_code
TaskScheduler::configure_worker(const std::size_t worker_index, const WorkerConfig &config) {
    if (worker_index >= workers_.size()) {
        RT_LOGC_ERROR(TaskLog::TaskScheduler, "Invalid worker index: {}", worker_index);
        return make_error_code(TaskErrc::InvalidParameter);
    }

    // Configure thread using common utilities
    const ThreadConfig thread_config{
            .core_id = config.is_pinned() ? config.get_core_id() : std::nullopt,
            .priority = config.has_thread_priority()
                                ? std::make_optional(config.get_thread_priority())
                                : std::nullopt};

    const std::error_code config_result = configure_thread(workers_[worker_index], thread_config);
    if (config_result) {
        RT_LOGC_ERROR(
                TaskLog::TaskScheduler,
                "Failed to configure worker {}: {}",
                worker_index,
                get_error_name(config_result));
        return config_result;
    }

    // Log worker configuration status
    if (config.is_pinned() && config.has_thread_priority()) {
        const auto core_id = config.get_core_id();
        if (core_id.has_value()) {
            RT_LOGC_DEBUG(
                    TaskLog::TaskScheduler,
                    "Worker {} pinned to core {}, RT priority {}",
                    worker_index,
                    *core_id,
                    config.get_thread_priority());
        }
    } else if (config.is_pinned()) {
        const auto core_id = config.get_core_id();
        if (core_id.has_value()) {
            RT_LOGC_DEBUG(
                    TaskLog::TaskScheduler, "Worker {} pinned to core {}", worker_index, *core_id);
        }
    } else if (config.has_thread_priority()) {
        RT_LOGC_DEBUG(
                TaskLog::TaskScheduler,
                "Worker {} configured with RT priority {}",
                worker_index,
                config.get_thread_priority());
    }

    return make_error_code(TaskErrc::Success);
}

void TaskScheduler::preallocate_category_queues(const std::uint32_t tasks_per_queue) {
    RT_LOGC_DEBUG(
            TaskLog::TaskScheduler,
            "Initializing category queues with {} reserved tasks each",
            tasks_per_queue);

    // Collect all unique categories from worker configurations
    std::set<TaskCategory> worker_categories{};
    for (const auto &worker_config : workers_config_.workers) {
        for (const auto &category : worker_config.categories) {
            worker_categories.insert(category);
        }
    }

    RT_LOGC_DEBUG(
            TaskLog::TaskScheduler,
            "Creating queues for {} unique categories from worker configs",
            worker_categories.size());

    // Initialize queues for all categories used by workers
    for (const auto &category : worker_categories) {
        auto queue_ptr = std::make_unique<CategoryQueue>();

        // Reserve capacity to minimize queue growth
        if (tasks_per_queue > 0) {
            const SpinlockGuard lock{queue_ptr->lock};
            queue_ptr->reserve(tasks_per_queue);
        }

        category_queues_.emplace(category, std::move(queue_ptr));
    }
}

void TaskScheduler::schedule(TaskGraph &graph, const Nanos execution_time) {
    // Require built graph for optimal performance
    if (!graph.is_built()) {
        throw std::runtime_error(
                "TaskGraph must be built before scheduling. Call graph.build() first.");
    }

    // Validate task categories match worker configuration
    validate_task_categories(workers_config_, graph);

    // Acquire fresh tasks from pool for this scheduling round
    const auto &scheduled_tasks = graph.prepare_tasks(execution_time);

    if (scheduled_tasks.empty()) {
        RT_LOGC_WARN(TaskLog::TaskScheduler, "No tasks to schedule in empty graph");
        return;
    }

    // Get scheduling count from the graph
    const std::uint64_t times_scheduled = graph.increment_times_scheduled();

    // Schedule all tasks with the same execution time and scheduling count
    std::size_t scheduled_count = 0;
    for (std::size_t task_index = 0; task_index < scheduled_tasks.size(); ++task_index) {
        const auto &task_ptr = scheduled_tasks[task_index];

        // Skip disabled tasks and tasks with disabled parents
        if (graph.is_task_or_parent_disabled(task_index)) {
            continue;
        }

        // Set scheduling count on the task
        task_ptr->set_times_scheduled(times_scheduled);

        // Schedule the task with the execution time
        // Dependency generation level ensures proper priority queue ordering
        schedule_at(task_ptr, task_monitor_, category_queues_);
        ++scheduled_count;
    }

    RT_LOGC_TRACE_L1(
            TaskLog::TaskScheduler,
            "Scheduled {} of {} tasks for scheduling round {}",
            scheduled_count,
            scheduled_tasks.size(),
            times_scheduled);
}

} // namespace framework::task
