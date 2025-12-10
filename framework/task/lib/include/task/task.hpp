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
 * @file task.hpp
 * @brief Minimal task system for scheduling and executing work
 *
 * Provides a lightweight task system with support for dependencies,
 * cancellation, timing constraints, and categorization. Tasks can be
 * scheduled with specific execution times and have dependencies on
 * other tasks.
 */

#ifndef FRAMEWORK_TASK_TASK_HPP
#define FRAMEWORK_TASK_TASK_HPP

#include <any>
#include <atomic>
#include <concepts>
#include <cstdint>
#include <format>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include <wise_enum.h>

#include "log/rt_log_macros.hpp"
#include "task/task_category.hpp"
#include "task/task_log.hpp"
#include "task/time.hpp"

namespace framework::task {

/// Task execution status
enum class TaskStatus {
    NotStarted, //!< Task has not been started yet
    Running,    //!< Task is currently executing
    Completed,  //!< Task completed successfully
    Failed,     //!< Task failed during execution
    Cancelled   //!< Task was cancelled before or during execution
};

} // namespace framework::task

// WISE_ENUM_ADAPT must be called at global namespace scope for ADL to work
WISE_ENUM_ADAPT(framework::task::TaskStatus, NotStarted, Running, Completed, Failed, Cancelled)

namespace framework::task {

/**
 * Result returned by task execution
 * Contains status information and optional message for diagnostics
 */
struct TaskResult final {
    TaskStatus status{TaskStatus::Completed}; //!< Execution status
    std::string message;                      //!< Optional message for details/errors

    /// Default constructor creates successful result
    TaskResult() = default;

    /**
     * Constructor with status and message
     * @param[in] s Task execution status
     * @param[in] msg Optional message describing the result
     */
    explicit TaskResult(const TaskStatus s, const std::string_view msg = "")
            : status{s}, message{msg} {}

    /**
     * Check if task completed successfully
     * @return true if task completed successfully, false otherwise
     */
    [[nodiscard]] bool is_success() const noexcept { return status == TaskStatus::Completed; }
};

/**
 * Cancellation token for cooperative task cancellation
 *
 * Allows tasks to check if they should stop execution and provides
 * a mechanism for external cancellation requests.
 */
class CancellationToken final {
private:
    // NOLINTBEGIN(readability-redundant-member-init) - {} is required for std::atomic zero-init
    std::atomic<bool> cancelled_{}; //!< Atomic cancellation flag
    // NOLINTEND(readability-redundant-member-init)

public:
    /// Default constructor
    CancellationToken() = default;

    /// Default destructor
    ~CancellationToken() = default;

    /// Non-copyable and non-movable
    CancellationToken(const CancellationToken &) = delete;
    CancellationToken &operator=(const CancellationToken &) = delete;
    CancellationToken(CancellationToken &&) = delete;
    CancellationToken &operator=(CancellationToken &&) = delete;

    /**
     * Check if cancellation has been requested
     * @return true if task should be cancelled, false otherwise
     */
    [[nodiscard]] bool is_cancelled() const noexcept {
        return cancelled_.load(std::memory_order_acquire);
    }

    /// Request cancellation of the task
    void cancel() noexcept { cancelled_.store(true, std::memory_order_release); }

    /// Reset cancellation state (mark as not cancelled)
    void reset() noexcept { cancelled_.store(false, std::memory_order_release); }
};

/**
 * Execution context passed to task functions
 * Provides access to cancellation and user-defined data
 */
struct TaskContext final {
    std::shared_ptr<CancellationToken>
            cancellation_token; //!< Cancellation token for cooperative cancellation
    std::any user_data;         //!< User-defined data for task-specific context. For
                                //!< large objects, use std::shared_ptr<T> to avoid copies

    /**
     * Constructor
     * @param[in] token Cancellation token for cooperative cancellation
     * @param[in] data User-defined data for task context
     */
    explicit TaskContext(std::shared_ptr<CancellationToken> token, std::any data = {})
            : cancellation_token(std::move(token)), user_data(std::move(data)) {}

    /**
     * Helper to safely get user data of specific type
     * @return Optional containing the data if type matches, nullopt otherwise
     */
    template <typename T> [[nodiscard]] std::optional<T> get_user_data() const {
        if (!user_data.has_value()) {
            return std::nullopt; // No data provided
        }

        try {
            return std::any_cast<T>(user_data);
        } catch (const std::bad_any_cast &) {
            RT_LOGC_ERROR(
                    TaskLog::Task,
                    "TaskContext::get_user_data() bad_any_cast - requested "
                    "type does not match stored type");
            return std::nullopt; // Wrong type
        }
    }
};

/// Concept to validate task functions with proper signatures and return types
// clang-format off
template <typename Func>
concept ValidTaskFunction = 
    // Functions that return TaskResult directly (no wrapping needed):
    (std::is_invocable_r_v<TaskResult, Func>) ||                              // TaskResult func()
    (std::is_invocable_r_v<TaskResult, Func, const TaskContext&>) ||          // TaskResult func(const TaskContext&)
    
    // Functions that return void (will be wrapped to return TaskResult):
    (std::is_invocable_v<Func> && std::is_void_v<std::invoke_result_t<Func>>) ||  // void func()
    (std::is_invocable_v<Func, const TaskContext&> &&                             // void func(const TaskContext&)
     std::is_void_v<std::invoke_result_t<Func, const TaskContext&>>);
// clang-format on

/// Type alias for task function variants to reduce verbosity
using TaskFunction =
        std::variant<std::function<TaskResult()>, std::function<TaskResult(const TaskContext &)>>;

/**
 * Task class for representing units of work
 *
 * A task encapsulates a function to execute along with metadata such as
 * scheduling time, timeout, and category. Tasks can be
 * executed independently or as part of dependency chains.
 *
 * Task stores owned copies of task and graph names for safe access
 * across different object lifetimes.
 */
class Task final {
private:
    static inline std::atomic<std::uint64_t> next_task_id{1}; //!< Global task ID counter
    std::uint64_t task_id_{}; //!< Unique task ID assigned at construction
    TaskFunction func_;       //!< Function to execute (supports both signatures)
    std::any user_data_;      //!< User-defined data for task context. For large
                              //!< objects, use std::shared_ptr<T> to avoid copies
    Nanos scheduled_time_{};  //!< When task should execute (nanoseconds)
    std::string task_name_;   //!< Task name for identification (owns string)
    std::string graph_name_;  //!< Graph name this task belongs to (owns string)
    Nanos timeout_ns_{};      //!< Maximum execution time
    std::shared_ptr<CancellationToken> cancel_token_{
            std::make_shared<CancellationToken>()};   //!< Cancellation token
    std::vector<std::shared_ptr<Task>> parent_tasks_; //!< Parent tasks for dependency tracking
    std::shared_ptr<std::atomic<TaskStatus>> status_{std::make_shared<std::atomic<TaskStatus>>(
            TaskStatus::NotStarted)}; //!< Current execution status
    TaskCategory category_{TaskCategory{BuiltinTaskCategory::Default}}; //!< Task category
    std::shared_ptr<std::atomic<std::uint32_t>> dependency_generation_{
            std::make_shared<std::atomic<std::uint32_t>>(
                    0)}; //!< Dependency generation level for scheduling order
    std::shared_ptr<std::atomic<std::uint64_t>> times_scheduled_{
            std::make_shared<std::atomic<std::uint64_t>>(
                    0)}; //!< Number of times this task's graph has been scheduled

    /**
     * Private constructor for internal use by TaskBuilder
     * @param[in] func Function to execute
     * @param[in] user_data User-defined data for task context
     * @param[in] task_name Task name for identification
     * @param[in] graph_name Graph name this task belongs to
     * @param[in] category Task category
     * @param[in] timeout_ns Maximum execution time
     * @param[in] scheduled_time When task should execute
     */
    template <typename FuncType>
    Task(FuncType &&func,
         std::any user_data,
         std::string_view task_name,
         std::string_view graph_name,
         TaskCategory category,
         Nanos timeout_ns,
         Nanos scheduled_time)
            : task_id_{next_task_id.fetch_add(1)}, func_{std::forward<FuncType>(func)},
              user_data_{std::move(user_data)}, scheduled_time_{scheduled_time},
              task_name_{task_name}, graph_name_{graph_name}, timeout_ns_{timeout_ns},
              category_{category} {}

    friend class TaskBuilder;
    friend class TaskPool;

public:
    /**
     * Check if task is ready to execute
     * @param[in] now Current time in nanoseconds
     * @param[in] readiness_tolerance_ns Tolerance window for readiness checking -
     * allows task to be considered ready this amount of time before its scheduled
     * time
     * @return true if task is ready to execute, false otherwise
     *
     * Task is ready when: now >= scheduled_time - readiness_tolerance_ns
     * This tolerance window accounts for scheduling jitter and ensures tasks can
     * be queued for execution slightly before their exact scheduled time,
     * preventing delays due to timing precision limitations.
     */
    [[nodiscard]] bool is_ready(const Nanos now, const Nanos readiness_tolerance_ns) const noexcept;

    /**
     * Check if task is ready to execute (template version)
     * @tparam NowRep Arithmetic type representing the number of ticks for now
     * @tparam NowPeriod std::ratio representing the tick period for now
     * @tparam ToleranceRep Arithmetic type representing the number of ticks for
     * tolerance
     * @tparam TolerancePeriod std::ratio representing the tick period for
     * tolerance
     * @param[in] now_duration Current time (any std::chrono::duration type)
     * @param[in] readiness_tolerance_duration Tolerance window for readiness
     * checking - allows task to be considered ready this amount of time before
     * its scheduled time
     * @return true if task is ready to execute, false otherwise
     *
     * Task is ready when: now >= scheduled_time - readiness_tolerance
     * This tolerance window accounts for scheduling jitter and ensures tasks can
     * be queued for execution slightly before their exact scheduled time,
     * preventing delays due to timing precision limitations.
     */
    template <typename NowRep, typename NowPeriod, typename ToleranceRep, typename TolerancePeriod>
    [[nodiscard]] bool is_ready(
            std::chrono::duration<NowRep, NowPeriod> now_duration,
            std::chrono::duration<ToleranceRep, TolerancePeriod> readiness_tolerance_duration)
            const noexcept {
        return is_ready(
                std::chrono::duration_cast<Nanos>(now_duration),
                std::chrono::duration_cast<Nanos>(readiness_tolerance_duration));
    }

    /**
     * Execute the task
     * Runs the task function and updates status and result
     * @return TaskResult indicating success or failure
     */
    [[nodiscard]] TaskResult execute() const;

    /**
     * Cancel the task
     * Sets cancellation token to request task termination
     */
    void cancel() const noexcept;

    /**
     * Get scheduled execution time
     * @return Scheduled time in nanoseconds
     */
    [[nodiscard]] Nanos get_scheduled_time() const noexcept;

    /**
     * Get task timeout
     * @return Timeout in nanoseconds
     */
    [[nodiscard]] Nanos get_timeout_ns() const noexcept;

    /**
     * Get task name
     * @return Task name reference
     */
    [[nodiscard]] std::string_view get_task_name() const noexcept;

    /**
     * Get graph name this task belongs to
     * @return Graph name reference
     */
    [[nodiscard]] std::string_view get_graph_name() const noexcept;

    /**
     * Get task ID (guaranteed unique across all tasks)
     * @return Task ID (64-bit unique identifier assigned at construction)
     */
    [[nodiscard]] std::uint64_t get_task_id() const noexcept;

    /**
     * Get task category
     * @return Task category
     */
    [[nodiscard]] TaskCategory get_category() const noexcept;

    /**
     * Get task dependency generation level
     * @return Dependency generation level (0=root, 1=child, 2=grandchild, etc.)
     */
    [[nodiscard]] std::uint32_t get_dependency_generation() const noexcept;

    /**
     * Get task's graph scheduling count
     * @return Number of times this task's graph has been scheduled
     */
    [[nodiscard]] std::uint64_t get_times_scheduled() const noexcept;

    /**
     * Set task's graph scheduling count
     * @param[in] times_scheduled Graph scheduling count to assign
     */
    void set_times_scheduled(std::uint64_t times_scheduled) const noexcept;

    /**
     * Get task status
     * Checks cancellation token first, then result status
     * @return Current task status
     */
    [[nodiscard]] TaskStatus status() const noexcept;

    /**
     * Set task status
     * @param[in] new_status New status to set
     */
    void set_status(TaskStatus new_status) const noexcept;

    /**
     * Check if task is cancelled
     * @return true if task is cancelled, false otherwise
     */
    [[nodiscard]] bool is_cancelled() const noexcept;

    /**
     * Check if task has no parent dependencies
     * Uses dependency generation for efficient check (generation 0 = no parents)
     * @return true if task has no parents, false otherwise
     */
    [[nodiscard]] bool has_no_parents() const noexcept;

    /**
     * Check if any parent matches the given predicate
     * @param[in] predicate Function to test each parent status
     * @return true if any parent matches the predicate, false otherwise
     */
    [[nodiscard]] bool any_parent_matches(std::function<bool(TaskStatus)> predicate) const noexcept;

    /**
     * Check if all parents match the given predicate
     * @param[in] predicate Callable to test each parent status
     * @return true if all parents match the predicate (or no parents), false
     * otherwise
     */
    template <std::invocable<TaskStatus> Predicate>
    [[nodiscard]] bool all_parents_match(Predicate predicate) const noexcept {
        // If no parents, return true (vacuous truth)
        if (parent_tasks_.empty()) {
            return true;
        }

        return std::all_of(
                parent_tasks_.begin(), parent_tasks_.end(), [&predicate](const auto &parent_task) {
                    return parent_task && predicate(parent_task->status());
                });
    }

    /**
     * Add parent task for dependency tracking
     * @param[in] parent_task Parent task to add as dependency
     */
    void add_parent_task(const std::shared_ptr<Task> &parent_task);

    /**
     * Reserve capacity for parent task statuses vector
     * @param[in] capacity Minimum capacity to reserve
     */
    void reserve_parent_capacity(std::size_t capacity);

    /**
     * Reserve capacity for task and graph name strings
     * @param[in] max_task_name_length Maximum expected task name length
     * @param[in] max_graph_name_length Maximum expected graph name length
     */
    void reserve_name_capacity(std::size_t max_task_name_length, std::size_t max_graph_name_length);

    /**
     * Clear all parent task dependencies
     */
    void clear_parent_tasks() noexcept;

private:
    /// Helper to wrap function with TaskContext parameter in exception handling
    template <typename Func> static TaskFunction wrap_context_function(Func &&func) {
        return std::function<TaskResult(const TaskContext &)>{
                [captured_func = std::forward<Func>(func)](const TaskContext &ctx) -> TaskResult {
                    try {
                        if constexpr (std::is_void_v<std::invoke_result_t<
                                              std::remove_reference_t<Func>,
                                              const TaskContext &>>) {
                            captured_func(ctx);
                            return TaskResult{TaskStatus::Completed};
                        } else {
                            auto result = captured_func(ctx);
                            if constexpr (std::is_same_v<
                                                  std::invoke_result_t<
                                                          std::remove_reference_t<Func>,
                                                          const TaskContext &>,
                                                  TaskResult>) {
                                return result;
                            } else {
                                return TaskResult{TaskStatus::Completed};
                            }
                        }
                    } catch (const std::exception &e) {
                        return TaskResult{
                                TaskStatus::Failed, std::format("Exception: {}", e.what())};
                    } catch (...) {
                        return TaskResult{TaskStatus::Failed, "Unknown exception occurred"};
                    }
                }};
    }

    /// Helper to wrap function with no parameters in exception handling
    template <typename Func> static TaskFunction wrap_no_param_function(Func &&func) {
        return std::function<
                TaskResult()>{[captured_func = std::forward<Func>(func)]() -> TaskResult {
            try {
                if constexpr (std::is_void_v<std::invoke_result_t<std::remove_reference_t<Func>>>) {
                    captured_func();
                    return TaskResult{TaskStatus::Completed};
                } else {
                    auto result = captured_func();
                    if constexpr (std::is_same_v<
                                          std::invoke_result_t<std::remove_reference_t<Func>>,
                                          TaskResult>) {
                        return result;
                    } else {
                        return TaskResult{TaskStatus::Completed};
                    }
                }
            } catch (const std::exception &e) {
                return TaskResult{TaskStatus::Failed, std::format("Exception: {}", e.what())};
            } catch (...) {
                return TaskResult{TaskStatus::Failed, "Unknown exception occurred"};
            }
        }};
    }

public:
    /**
     * Prepare task for reuse with new configuration (type-safe function version)
     * @tparam Func Function type that must be invocable with proper signature
     * @param[in] new_task_name New task name
     * @param[in] new_graph_name New graph name
     * @param[in] new_func New function to execute
     * @param[in] new_category New task category
     * @param[in] new_timeout_ns New timeout in nanoseconds
     * @param[in] new_scheduled_time New scheduled execution time
     * @param[in] new_user_data User-defined data for task context
     */
    template <typename Func>
        requires ValidTaskFunction<Func>
    void prepare_for_reuse(
            std::string_view new_task_name,
            std::string_view new_graph_name,
            Func &&new_func,
            TaskCategory new_category = TaskCategory{BuiltinTaskCategory::Default},
            Nanos new_timeout_ns = Nanos{0},
            Nanos new_scheduled_time = Nanos{0},
            const std::any &new_user_data = {}) {

        // Convert function to appropriate TaskFunction variant
        TaskFunction task_func;
        if constexpr (std::is_invocable_r_v<TaskResult, Func>) {
            // Function returns TaskResult with no parameters
            task_func = std::function<TaskResult()>{std::forward<Func>(new_func)};
        } else if constexpr (std::is_invocable_r_v<TaskResult, Func, const TaskContext &>) {
            // Function returns TaskResult with TaskContext parameter
            task_func =
                    std::function<TaskResult(const TaskContext &)>{std::forward<Func>(new_func)};
        } else if constexpr (std::is_invocable_v<Func, const TaskContext &>) {
            // Function with TaskContext parameter - wrap with exception handling
            task_func = wrap_context_function(std::forward<Func>(new_func));
        } else if constexpr (std::is_invocable_v<Func>) {
            // Function with no parameters - wrap with exception handling
            task_func = wrap_no_param_function(std::forward<Func>(new_func));
        }

        // Call the existing implementation
        prepare_for_reuse(
                new_task_name,
                new_graph_name,
                task_func,
                new_category,
                new_timeout_ns,
                new_scheduled_time,
                new_user_data);
    }

    /**
     * Prepare task for reuse with new configuration (TaskFunction variant
     * version)
     * @param[in] new_task_name New task name
     * @param[in] new_graph_name New graph name
     * @param[in] new_func New function to execute (supports both signatures)
     * @param[in] new_category New task category
     * @param[in] new_timeout_ns New timeout in nanoseconds
     * @param[in] new_scheduled_time New scheduled execution time
     * @param[in] new_user_data User-defined data for task context
     */
    void prepare_for_reuse(
            std::string_view new_task_name,
            std::string_view new_graph_name,
            const TaskFunction &new_func,
            TaskCategory new_category = TaskCategory{BuiltinTaskCategory::Default},
            Nanos new_timeout_ns = Nanos{0},
            Nanos new_scheduled_time = Nanos{0},
            const std::any &new_user_data = {});
};

/**
 * Handle to a scheduled task with reset capability
 * Provides access to task status and reset functionality
 */
class TaskHandle final {
private:
    std::shared_ptr<Task> task_;

public:
    /**
     * Constructor
     * @param[in] task Task instance (must be valid)
     * @throws std::invalid_argument if task is nullptr
     */
    explicit TaskHandle(std::shared_ptr<Task> task) : task_(std::move(task)) {
        if (!task_) {
            throw std::invalid_argument("TaskHandle requires a valid Task object");
        }
    }

    /**
     * Arrow operator for direct access to Task methods
     * @return Pointer to the underlying Task object
     */
    Task *operator->() const { return task_.get(); }
};

/**
 * Fluent builder for creating Task objects
 * Provides a fluent interface for setting task properties and dependencies
 */
class TaskBuilder final {
private:
    std::string task_name_;  //!< Task name for identification (owns string)
    std::string graph_name_; //!< Graph name this task belongs to (owns string)
    TaskFunction func_;      //!< Function to execute (supports both signatures)
    std::any user_data_;     //!< User-defined data for task context. For large
                             //!< objects, use std::shared_ptr<T> to avoid copies
    TaskCategory category_{TaskCategory{BuiltinTaskCategory::Default}};
    Nanos timeout_ns_{0};
    Nanos scheduled_time_{0};
    std::vector<std::shared_ptr<Task>> parent_tasks_;

public:
    /**
     * Constructor
     * @param[in] task_name Name for the task
     */
    explicit TaskBuilder(std::string task_name) : task_name_(std::move(task_name)) {}

    /**
     * Set task function
     * @tparam Func Function type that returns TaskResult with either no
     * parameters or const TaskContext&
     * @param[in] func Function to execute
     * @return Reference to this builder for chaining
     */
    template <typename Func>
        requires(
                std::is_invocable_r_v<TaskResult, Func> ||
                std::is_invocable_r_v<TaskResult, Func, const TaskContext &>)
    TaskBuilder &function(Func &&func) {
        if constexpr (std::is_invocable_r_v<TaskResult, Func>) {
            func_ = std::function<TaskResult()>{std::forward<Func>(func)};
        } else {
            func_ = std::function<TaskResult(const TaskContext &)>{std::forward<Func>(func)};
        }
        return *this;
    }

    /**
     * Set user data for task context
     * @param[in] data User-defined data to pass to contextual functions.
     *                 For large objects, use std::shared_ptr<T> to avoid copies
     * @return Reference to this builder for chaining
     */
    TaskBuilder &user_data(std::any data);

    /**
     * Set user data for task context (template convenience method)
     * @param[in] data User-defined data to pass to contextual functions.
     *                 For large objects, use std::shared_ptr<T> to avoid copies
     * @return Reference to this builder for chaining
     */
    template <typename T> TaskBuilder &user_data(T &&data) {
        user_data_ = std::any{std::forward<T>(data)};
        return *this;
    }

    /**
     * Set task timeout
     * @tparam Rep Arithmetic type representing the number of ticks
     * @tparam Period std::ratio representing the tick period
     * @param[in] timeout_duration Timeout duration (any std::chrono::duration
     * type)
     * @return Reference to this builder for chaining
     */
    template <typename Rep, typename Period>
    TaskBuilder &timeout(std::chrono::duration<Rep, Period> timeout_duration) {
        timeout_ns_ = std::chrono::duration_cast<Nanos>(timeout_duration);
        return *this;
    }

    /**
     * Set task category
     * @param[in] cat Task category for worker assignment
     * @return Reference to this builder for chaining
     */
    TaskBuilder &category(TaskCategory cat);

    /**
     * Set task category from builtin category
     * @param[in] builtin_cat Builtin task category for worker assignment
     * @return Reference to this builder for chaining
     */
    TaskBuilder &category(BuiltinTaskCategory builtin_cat);

    /**
     * Set task scheduled time
     * @tparam Rep Arithmetic type representing the number of ticks
     * @tparam Period std::ratio representing the tick period
     * @param[in] scheduled_time_duration When task should execute (any
     * std::chrono::duration type)
     * @return Reference to this builder for chaining
     */
    template <typename Rep, typename Period>
    TaskBuilder &scheduled_time(std::chrono::duration<Rep, Period> scheduled_time_duration) {
        scheduled_time_ = std::chrono::duration_cast<Nanos>(scheduled_time_duration);
        return *this;
    }

    /**
     * Set graph name
     * @param[in] graph_name Name of the graph this task belongs to
     * @return Reference to this builder for chaining
     */
    TaskBuilder &graph_name(std::string_view graph_name);

    /**
     * Add parent task dependency
     * @param[in] parent_task Shared pointer to parent task
     * @return Reference to this builder for chaining
     */
    TaskBuilder &depends_on(std::shared_ptr<Task> parent_task);

    /**
     * Add multiple parent task dependencies
     * @param[in] parent_tasks Vector of shared pointers to parent tasks
     * @return Reference to this builder for chaining
     */
    TaskBuilder &depends_on(const std::vector<std::shared_ptr<Task>> &parent_tasks);

    /**
     * Build the task
     * @return The created Task object
     * @throws std::invalid_argument if task name is empty
     */
    Task build();

    /**
     * Build the task as a shared_ptr
     * @return Shared pointer to the created Task object
     * @throws std::invalid_argument if task name is empty
     */
    [[nodiscard]] std::shared_ptr<Task> build_shared();
};

} // namespace framework::task

#endif // FRAMEWORK_TASK_TASK_HPP
