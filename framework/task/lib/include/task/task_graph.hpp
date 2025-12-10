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
 * @file task_graph.hpp
 * @brief Fluent API for building and managing task graphs with dependencies
 *
 * Provides a fluent interface for building complex task graphs with support
 * for multiple parent dependencies, reference capture for dynamic arguments,
 * and reset functionality for reuse across multiple execution cycles.
 */

#ifndef FRAMEWORK_TASK_TASK_GRAPH_HPP
#define FRAMEWORK_TASK_TASK_GRAPH_HPP

#include <atomic>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include "log/rt_log_macros.hpp"
#include "task/flat_map.hpp"
#include "task/task.hpp"
#include "task/task_log.hpp"
#include "task/task_pool.hpp"
#include "task/task_visualizer.hpp"
#include "task/time.hpp"

namespace framework::task {

// Forward declarations
class TaskGraphBuilder;
class SingleTaskGraphBuilder;
class TaskGraph;
class TaskScheduler;

/**
 * Schedulable task specification
 * Contains all information needed to create and schedule a task
 */
struct SchedulableTask final {
    std::string task_name; //!< Task name
    TaskFunction func;     //!< Function to execute (supports both signatures)
    TaskCategory category{TaskCategory{BuiltinTaskCategory::Default}}; //!< Task category
    Nanos timeout{0};                                                  //!< Timeout in nanoseconds
    std::any user_data;                        //!< User-defined data for task context. For large
                                               //!< objects, use std::shared_ptr<T> to avoid copies
    std::vector<std::string> dependency_names; //!< Names of parent tasks
    std::vector<std::size_t> parent_indices; //!< Pre-computed parent indices for efficient building
    bool disabled{false};                    //!< Whether this task is disabled from scheduling
};

/**
 * Graph-specific builder for creating task specifications in a TaskGraph
 * Provides a fluent interface for setting task properties with name-based
 * dependencies
 */
class TaskGraphBuilder final {
private:
    TaskGraph &graph_; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
    SchedulableTask current_task_;
    std::vector<std::string> dependencies_;

public:
    /**
     * Constructor
     * @param[in] graph Reference to parent task graph
     * @param[in] task_name Name for the task
     */
    TaskGraphBuilder(TaskGraph &graph, const std::string_view task_name);

    // Delete copy/move operations due to reference member
    TaskGraphBuilder(const TaskGraphBuilder &) = delete;
    TaskGraphBuilder &operator=(const TaskGraphBuilder &) = delete;
    TaskGraphBuilder(TaskGraphBuilder &&) = delete;
    TaskGraphBuilder &operator=(TaskGraphBuilder &&) = delete;
    ~TaskGraphBuilder() = default;

    /**
     * Set task execution timeout
     * @tparam Rep Arithmetic type representing the number of ticks
     * @tparam Period std::ratio representing the tick period
     * @param[in] timeout_duration Maximum execution duration (any
     * std::chrono::duration type)
     * @return Reference to this builder for chaining
     */
    template <typename Rep, typename Period>
    TaskGraphBuilder &timeout(std::chrono::duration<Rep, Period> timeout_duration) {
        current_task_.timeout = std::chrono::duration_cast<Nanos>(timeout_duration);
        return *this;
    }

    /**
     * Set task category
     * @param[in] cat Task category for worker assignment
     * @return Reference to this builder for chaining
     */
    TaskGraphBuilder &category(TaskCategory cat);

    /**
     * Set task category from builtin category
     * @param[in] builtin_cat Builtin task category for worker assignment
     * @return Reference to this builder for chaining
     */
    TaskGraphBuilder &category(BuiltinTaskCategory builtin_cat);

    /**
     * Set task pool capacity multiplier for the parent graph
     *
     * Multiplies the total tasks in the graph to get the total tasks in the
     * reuse pool. Higher values reduce heap allocations during bursty execution
     * patterns at cost of memory usage.
     *
     * @param[in] multiplier Multiplier for TaskPool capacity calculation
     * @return Reference to this builder for chaining
     */
    TaskGraphBuilder &task_pool_capacity_multiplier(const std::size_t multiplier);

    /**
     * Set function for current task
     * Supports function types with no arguments or with TaskContext
     * @param[in] func Function to execute
     * @return Reference to this builder for chaining
     */
    template <typename Func>
        requires ValidTaskFunction<Func>
    TaskGraphBuilder &function(Func &&func) {
        return function_impl(std::forward<Func>(func));
    }

    /**
     * Add single parent dependency
     * @param[in] parent_name Name of parent task
     * @return Reference to this builder for chaining
     */
    TaskGraphBuilder &depends_on(const std::string_view parent_name);

    /**
     * Add multiple parent dependencies
     * @param[in] parent_names Names of parent tasks
     * @return Reference to this builder for chaining
     */
    TaskGraphBuilder &depends_on(const std::vector<std::string_view> &parent_names);

    /**
     * Set user data for task context
     * @param[in] data User-defined data to pass to contextual functions.
     *                 For large objects, use std::shared_ptr<T> to avoid copies
     * @return Reference to this builder for chaining
     */
    TaskGraphBuilder &user_data(std::any data);

    /**
     * Set user data for task context (template convenience method)
     * @param[in] data User-defined data to pass to contextual functions.
     *                 For large objects, use std::shared_ptr<T> to avoid copies
     * @return Reference to this builder for chaining
     */
    template <typename T> TaskGraphBuilder &user_data(T &&data) {
        current_task_.user_data = std::any{std::forward<T>(data)};
        return *this;
    }

    /**
     * Add task to graph for multi-task graphs
     * @return Name of the created task for use in dependencies
     */
    std::string add();

private:
    /// Reset builder state for next task
    void reset_builder();

    /// Helper to create TaskFunction with TaskContext parameter
    template <typename Func> static TaskFunction create_context_function(Func &&func) {
        return [captured_func = std::forward<Func>(func)](const TaskContext &ctx) -> TaskResult {
            try {
                if constexpr (std::is_void_v<std::invoke_result_t<Func, const TaskContext &>>) {
                    captured_func(ctx);
                    return TaskResult{TaskStatus::Completed};
                } else {
                    auto result = captured_func(ctx);
                    if constexpr (std::is_same_v<
                                          std::invoke_result_t<Func, const TaskContext &>,
                                          TaskResult>) {
                        return result;
                    } else {
                        return TaskResult{TaskStatus::Completed};
                    }
                }
            } catch (const std::exception &e) {
                return TaskResult{TaskStatus::Failed, e.what()};
            } catch (...) {
                return TaskResult{TaskStatus::Failed, "Unknown exception"};
            }
        };
    }

    /// Helper to create TaskFunction with no parameters
    template <typename Func> static TaskFunction create_no_param_function(Func &&func) {
        return [captured_func = std::forward<Func>(func)]() -> TaskResult {
            try {
                if constexpr (std::is_void_v<std::invoke_result_t<Func>>) {
                    captured_func();
                    return TaskResult{TaskStatus::Completed};
                } else {
                    auto result = captured_func();
                    if constexpr (std::is_same_v<std::invoke_result_t<Func>, TaskResult>) {
                        return result;
                    } else {
                        return TaskResult{TaskStatus::Completed};
                    }
                }
            } catch (const std::exception &e) {
                return TaskResult{TaskStatus::Failed, e.what()};
            } catch (...) {
                return TaskResult{TaskStatus::Failed, "Unknown exception"};
            }
        };
    }

    /// Private helper to handle function setting logic
    template <typename Func> TaskGraphBuilder &function_impl(Func &&func) {
        if constexpr (std::invocable<Func, const TaskContext &>) {
            // Function with TaskContext parameter
            current_task_.func = create_context_function(std::forward<Func>(func));
        } else {
            // Function with no parameters
            current_task_.func = create_no_param_function(std::forward<Func>(func));
        }
        return *this;
    }
};

/**
 * Builder for single-task graphs that builds immediately
 * Provides a fluent interface for creating simple graphs with one task
 * Reuses existing TaskGraphBuilder to avoid code duplication
 */
class SingleTaskGraphBuilder final {
private:
    std::unique_ptr<TaskGraph> graph_;
    std::unique_ptr<TaskGraphBuilder> task_builder_;

public:
    /**
     * Constructor
     * @param[in] graph_name Name for the graph
     */
    explicit SingleTaskGraphBuilder(const std::string_view graph_name);

    /**
     * Set task name
     * @param[in] task_name Name for the task
     * @return Reference to this builder for chaining
     */
    SingleTaskGraphBuilder &single_task(const std::string_view task_name);

    /**
     * Set task execution timeout
     * @tparam Rep Arithmetic type representing the number of ticks
     * @tparam Period std::ratio representing the tick period
     * @param[in] timeout_duration Maximum execution duration (any
     * std::chrono::duration type)
     * @return Reference to this builder for chaining
     */
    template <typename Rep, typename Period>
    SingleTaskGraphBuilder &timeout(std::chrono::duration<Rep, Period> timeout_duration) {
        task_builder_->timeout(timeout_duration);
        return *this;
    }

    /**
     * Set task category
     * @param[in] cat Task category for worker assignment
     * @return Reference to this builder for chaining
     */
    SingleTaskGraphBuilder &category(TaskCategory cat) {
        task_builder_->category(cat);
        return *this;
    }

    /**
     * Set task function (supports reference capture)
     * @param[in] func Function to execute - supports lambda with reference
     * capture
     * @return Reference to this builder for chaining
     */
    template <typename Func> SingleTaskGraphBuilder &function(Func &&func) {
        task_builder_->function(std::forward<Func>(func));
        return *this;
    }

    /**
     * Set user data for task context
     * @param[in] data User-defined data to pass to contextual functions
     * @return Reference to this builder for chaining
     */
    template <typename T> SingleTaskGraphBuilder &user_data(T &&data) {
        task_builder_->user_data(std::forward<T>(data));
        return *this;
    }

    /**
     * Build complete task graph with single task
     * @return Built TaskGraph ready for scheduling
     */
    TaskGraph build();
};

/**
 * Task graph for managing complex task relationships
 * Provides fluent interface for building task graphs with dependencies
 */
class TaskGraph final {
private:
    static constexpr std::size_t DEFAULT_TASK_POOL_CAPACITY_MULTIPLIER = 20;

    std::string graph_name_; //!< Graph name for identification
    std::size_t task_pool_capacity_multiplier_{
            DEFAULT_TASK_POOL_CAPACITY_MULTIPLIER}; //!< Multiplier for TaskPool
                                                    //!< capacity calculation
    std::vector<SchedulableTask> task_specs_;
    FlatMap<std::string, size_t> task_name_to_index_;

    // Task pool must be declared before scheduled_tasks_ to ensure proper
    // destruction order (scheduled_tasks_ destroyed first, then task_pool_)
    std::shared_ptr<TaskPool> task_pool_; //!< Task pool for efficient task reuse

    // Scheduled tasks for current scheduling round - populated on-the-fly from
    // pool
    std::vector<std::shared_ptr<Task>> scheduled_tasks_;

    std::uint64_t times_scheduled_{
            0}; //!< Number of times this graph has been scheduled for execution

    bool is_built_{false};
    TaskVisualizer graph_visualizer_{}; //!< Always-enabled dependency graph visualizer

public:
    /**
     * Constructor with graph name
     * @param[in] graph_name Name for this graph
     */
    explicit TaskGraph(const std::string_view graph_name) : graph_name_(graph_name) {}

    /**
     * Constructor with graph name and task pool capacity multiplier
     * @param[in] graph_name Name for this graph
     * @param[in] task_pool_capacity_multiplier Multiplier for TaskPool capacity
     */
    TaskGraph(const std::string_view graph_name, const std::size_t task_pool_capacity_multiplier)
            : graph_name_(graph_name),
              task_pool_capacity_multiplier_(task_pool_capacity_multiplier) {}

    /**
     * Get graph name
     * @return Graph name
     */
    [[nodiscard]] const std::string &get_graph_name() const noexcept { return graph_name_; }

    /**
     * Get task pool capacity
     *
     * Note: Actual capacity may be larger than (task_count × multiplier) due to
     * BoundedQueue rounding up to the next power of 2 for performance.
     *
     * @return Current TaskPool capacity (rounded to power of 2), or 0 if not
     * built
     */
    [[nodiscard]] std::size_t get_task_pool_capacity() const noexcept {
        return task_pool_ ? task_pool_->capacity() : 0;
    }

    /**
     * Create a single-task graph builder with graph name
     * @param[in] graph_name Name for the graph
     * @return SingleTaskGraphBuilder instance for fluent single-task creation
     */
    static SingleTaskGraphBuilder create(const std::string_view graph_name);

    /**
     * Force cleanup of scheduled tasks to break circular references
     * Call this to ensure all tasks are properly released back to pool
     */
    void clear_scheduled_tasks();

    /**
     * Register new task builder for multi-task graphs
     * @param[in] task_name Name for the task
     * @return TaskGraphBuilder instance for fluent task creation
     */
    TaskGraphBuilder register_task(const std::string_view task_name);

    /**
     * Get task specifications for scheduling
     * @return Vector of schedulable task specifications
     */
    [[nodiscard]] const std::vector<SchedulableTask> &get_task_specs() const noexcept {
        return task_specs_;
    }

    /**
     * Clear the entire graph
     * Removes all task specifications, handles, and built state
     */
    void clear();

    /**
     * Get number of tasks in graph
     * @return Number of tasks
     */
    [[nodiscard]] size_t size() const noexcept { return task_specs_.size(); }

    /**
     * Check if graph is empty
     * @return true if graph has no tasks
     */
    [[nodiscard]] bool empty() const noexcept { return task_specs_.empty(); }

    /**
     * Check if a task has the specified status
     * @param[in] name Task name
     * @param[in] expected_status Expected status to check against
     * @return true if task exists and has the expected status, false otherwise
     * @throws std::runtime_error if graph has not been built
     */
    [[nodiscard]] bool
    task_has_status(const std::string_view name, TaskStatus expected_status) const;

    /**
     * Set the status of a specific task
     * @param[in] name Task name to modify
     * @param[in] new_status New status to set
     * @return true if task exists and status was set, false if task not found
     * @throws std::runtime_error if graph has not been built
     * @note When setting TaskStatus::Cancelled, this will properly cancel the
     * task using the cancel() method to set both status and cancellation token
     */
    bool set_task_status(const std::string_view name, TaskStatus new_status);

    /**
     * Build/finalize the task graph for optimized scheduling
     * Pre-processes all dependencies, creates task wrappers, and allocates status
     * handles Must be called before scheduling tasks for optimal performance
     * @throws std::runtime_error if graph has circular dependencies or other
     * issues
     */
    void build();

    /**
     * Check if graph has been built for optimized scheduling
     * @return true if graph has been built
     */
    [[nodiscard]] bool is_built() const noexcept { return is_built_; }

    /**
     * Get and increment times scheduled for this graph
     * @return Current times scheduled count (increments for each schedule call)
     */
    [[nodiscard]] std::uint64_t increment_times_scheduled() noexcept {
        return ++times_scheduled_; // increment first, then return
    }

    /**
     * Get current times scheduled count without incrementing
     * @return Current times scheduled count
     */
    [[nodiscard]] std::uint64_t get_times_scheduled() const noexcept { return times_scheduled_; }

    /**
     * Acquire tasks from pool for scheduling current scheduling round
     *
     * Populates scheduled_tasks_ with fresh tasks from the pool
     *
     * @warning This method is NOT thread-safe. Concurrent calls will cause race
     * conditions in scheduled_tasks_ vector manipulation and task dependency
     * setup. External synchronization is required for concurrent access.
     *
     * @param[in] execution_time Execution time for the tasks
     * @return Reference to scheduled_tasks_ vector
     * @throws std::runtime_error if graph has not been built
     */
    [[nodiscard]] std::vector<std::shared_ptr<Task>> &prepare_tasks(Nanos execution_time);

    /**
     * Acquire tasks from pool for scheduling current scheduling round (template
     * version)
     *
     * Populates scheduled_tasks_ with fresh tasks from the pool
     *
     * @warning This method is NOT thread-safe. Concurrent calls will cause race
     * conditions in scheduled_tasks_ vector manipulation and task dependency
     * setup. External synchronization is required for concurrent access.
     *
     * @tparam Rep Arithmetic type representing the number of ticks
     * @tparam Period std::ratio representing the tick period
     * @param[in] execution_time_duration Execution time for the tasks (any
     * std::chrono::duration type)
     * @return Reference to scheduled_tasks_ vector
     * @throws std::runtime_error if graph has not been built
     */
    template <typename Rep, typename Period>
    [[nodiscard]] std::vector<std::shared_ptr<Task>> &
    prepare_tasks(std::chrono::duration<Rep, Period> execution_time_duration) {
        return prepare_tasks(std::chrono::duration_cast<Nanos>(execution_time_duration));
    }

    /**
     * Generate string visualization of task dependency graph
     * @return String representation of the task graph
     */
    [[nodiscard]] std::string to_string() const;

    /**
     * Get task pool statistics (only available after build())
     * @return TaskPool statistics
     * @throws std::runtime_error if graph has not been built
     */
    [[nodiscard]] TaskPoolStats get_pool_stats() const;

    /**
     * Disable a task from being scheduled in future scheduling rounds
     * @param[in] task_name Name of the task to disable
     * @return true if task was found and disabled, false if task not found
     */
    bool disable_task(const std::string_view task_name);

    /**
     * Enable a previously disabled task for scheduling
     * @param[in] task_name Name of the task to enable
     * @return true if task was found and enabled, false if task not found
     */
    bool enable_task(const std::string_view task_name);

    /**
     * Check if a task is currently disabled
     * @param[in] task_name Name of the task to check
     * @return true if task is disabled, false if enabled or not found
     */
    [[nodiscard]] bool is_task_disabled(const std::string_view task_name) const;

    /**
     * Check if a task or any of its parents is disabled
     * @param[in] task_index Index of the task to check
     * @return true if task or any parent is disabled
     */
    [[nodiscard]] bool is_task_or_parent_disabled(std::size_t task_index) const;

    friend class TaskGraphBuilder;

private:
    /**
     * Add task specification (called by TaskGraphBuilder)
     * @param[in] spec Task specification to add
     * @throws std::runtime_error if graph has been built
     */
    void add_schedulable_task(SchedulableTask spec);

    /**
     * Validate that graph has not been built (for modification operations)
     * @throws std::runtime_error if graph has been built
     */
    void validate_not_built() const;

    /**
     * Calculate dependency generation levels for all tasks using breadth-first
     * traversal
     * @return Vector of dependency generations (index matches task_specs_)
     */
    [[nodiscard]] std::vector<std::uint32_t> calculate_dependency_generations() const;

    /**
     * Compute task indices in dependency order for single-pass building
     * @return Vector of task spec indices sorted by dependency generation
     */
    [[nodiscard]] std::vector<std::size_t> compute_dependency_ordered_indices() const;
};

} // namespace framework::task

#endif // FRAMEWORK_TASK_TASK_GRAPH_HPP
