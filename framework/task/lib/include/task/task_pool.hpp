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
 * @file task_pool.hpp
 * @brief Thread-safe object pool for Task instances using BoundedQueue
 *
 * Provides efficient reuse of Task objects to reduce memory allocation
 * overhead in high-frequency task scheduling scenarios.
 */

#ifndef FRAMEWORK_TASK_TASK_POOL_HPP
#define FRAMEWORK_TASK_TASK_POOL_HPP

#include <atomic>
#include <cstdint>
#include <memory>
#include <new>
#include <utility>

#include "task/bounded_queue.hpp"
#include "task/task.hpp"
#include "task/task_log.hpp"

namespace framework::task {

/// Default initial pool size
static constexpr std::size_t DEFAULT_POOL_SIZE = 64;

/// Default maximum expected parents per task
static constexpr std::size_t DEFAULT_MAX_TASK_PARENTS = 8;

/// Default maximum expected task name length
static constexpr std::size_t DEFAULT_MAX_TASK_NAME_LENGTH = 64;

/// Default maximum expected graph name length
static constexpr std::size_t DEFAULT_MAX_GRAPH_NAME_LENGTH = 32;

/**
 * Statistics for TaskPool performance monitoring
 */
struct TaskPoolStats final {
    std::uint64_t pool_hits{};         //!< Tasks served from pool (reused existing tasks)
    std::uint64_t new_tasks_created{}; //!< Tasks created new when pool was empty
    std::uint64_t tasks_released{};    //!< Tasks returned to pool for reuse

    /**
     * Get total tasks served by pool
     * @return Sum of pool hits and newly created tasks
     */
    [[nodiscard]] std::uint64_t total_tasks_served() const noexcept {
        return pool_hits + new_tasks_created;
    }

    /**
     * Get instantaneous pool hit rate as percentage
     * @return Current hit rate (0.0 to 100.0) based on cumulative stats snapshot,
     *         or 0.0 if no tasks have been served yet
     */
    [[nodiscard]] double hit_rate_percent() const noexcept {
        const auto total = total_tasks_served();
        return total > 0 ? (static_cast<double>(pool_hits) / static_cast<double>(total)) * 100.0
                         : 0.0;
    }
};

/**
 * Thread-safe object pool for Task instances
 *
 * Uses a BoundedQueue for lock-free pooling with automatic fallback
 * to heap allocation when pool is exhausted. Provides RAII semantics
 * through custom shared_ptr deleters.
 */
class TaskPool final : public std::enable_shared_from_this<TaskPool> {
private:
    mutable BoundedQueue<std::shared_ptr<Task>> pool_; //!< Lock-free task storage pool
    // NOLINTBEGIN(readability-redundant-member-init) - {} is required for std::atomic zero-init
    mutable std::atomic<std::uint64_t>
            new_tasks_created_{}; //!< Counter for new task creation when pool empty
    mutable std::atomic<std::uint64_t> pool_hits_{};      //!< Counter for successful pool reuse
    mutable std::atomic<std::uint64_t> tasks_released_{}; //!< Counter for tasks returned to pool
    // NOLINTEND(readability-redundant-member-init)
    std::size_t max_task_parents_{};      //!< Maximum expected parents per task
    std::size_t max_task_name_length_{};  //!< Maximum expected task name length
    std::size_t max_graph_name_length_{}; //!< Maximum expected graph name length

    /**
     * Custom deleter for shared_ptr that returns Task to pool
     * @param[in] task Shared pointer to task to return to pool
     */
    void return_to_pool(std::shared_ptr<Task> task) const noexcept;

    /**
     * Private constructor - use create() factory function instead
     * @param[in] initial_size Initial pool capacity (will be rounded up to power
     * of 2)
     * @param[in] max_task_parents Maximum expected parents per task for capacity
     * reservation
     * @param[in] max_task_name_length Maximum expected task name length for
     * string capacity reservation
     * @param[in] max_graph_name_length Maximum expected graph name length for
     * string capacity reservation
     * @throws std::bad_alloc if pool cannot be properly initialized to requested
     * capacity
     */
    explicit TaskPool(
            std::size_t initial_size = DEFAULT_POOL_SIZE,
            std::size_t max_task_parents = DEFAULT_MAX_TASK_PARENTS,
            std::size_t max_task_name_length = DEFAULT_MAX_TASK_NAME_LENGTH,
            std::size_t max_graph_name_length = DEFAULT_MAX_GRAPH_NAME_LENGTH);

public:
    /**
     * Factory function to create TaskPool managed by shared_ptr
     *
     * This ensures TaskPool is always managed by shared_ptr, enabling safe
     * lifetime management for tasks returned to the pool.
     *
     * @param[in] initial_size Initial pool capacity (will be rounded up to power
     * of 2)
     * @param[in] max_task_parents Maximum expected parents per task for capacity
     * reservation
     * @param[in] max_task_name_length Maximum expected task name length for
     * string capacity reservation
     * @param[in] max_graph_name_length Maximum expected graph name length for
     * string capacity reservation
     * @return Shared pointer to TaskPool instance
     * @throws std::bad_alloc if pool cannot be properly initialized to requested
     * capacity
     */
    [[nodiscard]] static std::shared_ptr<TaskPool>
    create(std::size_t initial_size = DEFAULT_POOL_SIZE,
           std::size_t max_task_parents = DEFAULT_MAX_TASK_PARENTS,
           std::size_t max_task_name_length = DEFAULT_MAX_TASK_NAME_LENGTH,
           std::size_t max_graph_name_length = DEFAULT_MAX_GRAPH_NAME_LENGTH);

    /// Default destructor
    ~TaskPool() = default;

    // Non-copyable, non-movable for thread safety
    TaskPool(const TaskPool &) = delete;
    TaskPool &operator=(const TaskPool &) = delete;
    TaskPool(TaskPool &&) = delete;
    TaskPool &operator=(TaskPool &&) = delete;

    /**
     * Acquire a Task from the pool or create new one
     * @param[in] task_name Task name
     * @param[in] graph_name Graph name
     * @return Shared pointer to Task with custom deleter for pool return
     */
    [[nodiscard]] std::shared_ptr<Task>
    acquire_task(std::string_view task_name, std::string_view graph_name);

    /**
     * Get pool statistics
     * @return TaskPoolStats with performance metrics
     */
    [[nodiscard]] TaskPoolStats get_stats() const noexcept;

    /**
     * Get current pool capacity
     * @return Maximum number of tasks the pool can hold
     */
    [[nodiscard]] std::size_t capacity() const noexcept;

private:
    /**
     * Create a new Task with reserved parent_statuses vector
     * @param[in] name Task name
     * @return Shared pointer to new Task with custom deleter
     */
    [[nodiscard]] std::shared_ptr<Task> create_new_task(std::string_view name) const;

    /**
     * Prepare a task for reuse (reset state and reserve parent_statuses)
     * @param[in,out] task Task to prepare
     * @param[in] task_name New task name
     * @param[in] graph_name New graph name
     */
    void prepare_task_for_reuse(
            Task &task, std::string_view task_name, std::string_view graph_name) const;

    /**
     * Create shared_ptr with custom deleter for safe task return to pool
     *
     * Uses weak_ptr in the deleter to safely check if TaskPool still exists
     * before attempting to return the task. If TaskPool has been destroyed,
     * the task is simply cleaned up normally without attempting pool return.
     *
     * This is safe because TaskPool is guaranteed to be managed by shared_ptr
     * (enforced by private constructor and factory function).
     *
     * @param[in] task_ptr Shared pointer to task (ownership transferred)
     * @return Shared pointer with safe custom deleter
     */
    [[nodiscard]] std::shared_ptr<Task>
    create_managed_task_ptr(const std::shared_ptr<Task> &task_ptr) const;
};

} // namespace framework::task

#endif // FRAMEWORK_TASK_TASK_POOL_HPP
