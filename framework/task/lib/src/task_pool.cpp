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
 * @file task_pool.cpp
 * @brief Implementation of TaskPool for efficient Task object reuse
 */

#include <atomic>
#include <cstddef>
#include <memory>
#include <new>
#include <string>
#include <string_view>
#include <utility>

#include <quill/LogMacros.h>

#include "log/rt_log_macros.hpp"
#include "task/bounded_queue.hpp"
#include "task/task.hpp"
#include "task/task_log.hpp"
#include "task/task_pool.hpp"

namespace framework::task {

std::shared_ptr<TaskPool> TaskPool::create(
        const std::size_t initial_size,
        const std::size_t max_task_parents,
        const std::size_t max_task_name_length,
        const std::size_t max_graph_name_length) {
    // Use shared_ptr constructor that can access private constructor
    return std::shared_ptr<TaskPool>(new TaskPool(
            initial_size, max_task_parents, max_task_name_length, max_graph_name_length));
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
TaskPool::TaskPool(
        const std::size_t initial_size,
        const std::size_t max_task_parents,
        const std::size_t max_task_name_length,
        const std::size_t max_graph_name_length)
        : pool_(initial_size), max_task_parents_{max_task_parents},
          max_task_name_length_{max_task_name_length},
          max_graph_name_length_{max_graph_name_length} {
    // Pre-fill the pool with tasks
    for (std::size_t i = 0; i < initial_size; ++i) {
        auto task = create_new_task("pooled_task");
        if (!pool_.try_push(std::move(task))) {
            // Pool initialization failed - this should never happen with a newly
            // constructed pool
            throw std::bad_alloc{};
        }
    }
}
// NOLINTEND(bugprone-easily-swappable-parameters)

void TaskPool::return_to_pool(std::shared_ptr<Task> task) const noexcept {
    if (!task) {
        return;
    }

    // Clear parent references to break circular dependencies and prepare task
    // for reuse. This is safe here because the task is finished executing and
    // no worker threads have access to it anymore.
    task->clear_parent_tasks();

    // Increment released counter
    tasks_released_.fetch_add(1, std::memory_order_relaxed);

    // Capture task ID before potential move
    const auto task_id = task->get_task_id();

    // Try to return to pool
    if (!pool_.try_push(std::move(task))) {
        // Pool is full, log warning with stats and let shared_ptr handle cleanup
        const auto stats = get_stats();
        RT_LOGC_WARN(
                TaskLog::TaskPool,
                "Task pool is full, discarding task {} instead of reusing. "
                "Pool capacity: {}, Stats: {} pool hits, {} heap allocations "
                "({:.1f}% hit rate). "
                "Consider increasing pool size.",
                task_id,
                pool_.capacity(),
                stats.pool_hits,
                stats.new_tasks_created,
                stats.hit_rate_percent());
    }
}

std::shared_ptr<Task>
TaskPool::create_managed_task_ptr(const std::shared_ptr<Task> &task_ptr) const {
    // TaskPool is guaranteed to be managed by shared_ptr (enforced by private
    // constructor), so shared_from_this() will always succeed
    const std::weak_ptr<const TaskPool> weak_self = shared_from_this();
    return {task_ptr.get(), [weak_self, task_ptr](Task *const /*t*/) {
                if (auto pool = weak_self.lock()) {
                    pool->return_to_pool(task_ptr);
                }
                // If TaskPool destroyed, shared_ptr will handle cleanup
                // automatically
            }};
}

std::shared_ptr<Task>
TaskPool::acquire_task(const std::string_view task_name, const std::string_view graph_name) {
    std::shared_ptr<Task> task_ptr{};

    // Try to get from pool first
    if (pool_.try_pop(task_ptr) && task_ptr) {
        // Prepare existing task for reuse
        prepare_task_for_reuse(*task_ptr, task_name, graph_name);
        pool_hits_.fetch_add(1, std::memory_order_relaxed);

        return create_managed_task_ptr(task_ptr);
    }

    // Pool empty - create new task and log warning with stats
    const auto stats = get_stats();
    RT_LOGC_WARN(
            TaskLog::TaskPool,
            "Task pool exhausted, heap allocating new task '{}' for graph '{}'. "
            "Pool capacity: {}, Current stats: {} pool hits, {} new tasks created "
            "({:.1f}% hit rate). "
            "Consider increasing pool size for better performance.",
            task_name,
            graph_name,
            pool_.capacity(),
            stats.pool_hits,
            stats.new_tasks_created,
            stats.hit_rate_percent());

    new_tasks_created_.fetch_add(1, std::memory_order_relaxed);
    task_ptr = create_new_task(task_name);
    prepare_task_for_reuse(*task_ptr, task_name, graph_name);

    return create_managed_task_ptr(task_ptr);
}

TaskPoolStats TaskPool::get_stats() const noexcept {
    return TaskPoolStats{
            .pool_hits = pool_hits_.load(std::memory_order_relaxed),
            .new_tasks_created = new_tasks_created_.load(std::memory_order_relaxed),
            .tasks_released = tasks_released_.load(std::memory_order_relaxed)};
}

std::size_t TaskPool::capacity() const noexcept { return pool_.capacity(); }

std::shared_ptr<Task> TaskPool::create_new_task(const std::string_view task_name) const {
    // Create task using TaskBuilder
    TaskBuilder builder{std::string{task_name}};

    // Set default function that does nothing
    builder.function([]() { return TaskResult{}; });

    // Create the task and wrap in shared_ptr (no custom deleter yet)
    auto task = std::make_shared<Task>(builder.build());

    // Reserve space in parent_statuses vector to prevent runtime allocations
    task->reserve_parent_capacity(max_task_parents_);

    // Reserve capacity for task and graph name strings to prevent runtime
    // allocations
    task->reserve_name_capacity(max_task_name_length_, max_graph_name_length_);

    return task;
}

void TaskPool::prepare_task_for_reuse(
        Task &task, const std::string_view task_name, const std::string_view graph_name) const {
    // Use the Task's prepare_for_reuse method for clean reinitialization
    // The task already has a default function from create_new_task, so we reuse
    // the existing function to avoid redundant creation
    task.prepare_for_reuse(task_name, graph_name, task.func_);

    // Ensure parent_statuses has reserved capacity
    task.reserve_parent_capacity(max_task_parents_);

    // Ensure name strings have reserved capacity
    task.reserve_name_capacity(max_task_name_length_, max_graph_name_length_);
}

} // namespace framework::task
