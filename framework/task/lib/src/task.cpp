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
 * @file task.cpp
 * @brief Implementation of Task class and related functionality
 */

#include <algorithm>
#include <any>
#include <atomic>
#include <chrono>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <format>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "task/task.hpp"
#include "task/task_category.hpp"
#include "task/time.hpp"

namespace framework::task {

// Static task ID counter is now defined inline in the header

bool Task::is_ready(const Nanos now, const Nanos readiness_tolerance_ns) const noexcept {
    if (status() != TaskStatus::NotStarted) {
        return false;
    }

    // If no scheduled time is set (0), task is immediately ready
    if (scheduled_time_ == Nanos{0}) {
        return true;
    }

    // Check if current time is within tolerance window of scheduled time
    // Task is ready if: now >= (scheduled_time - tolerance_window)
    return (now - scheduled_time_ >= -readiness_tolerance_ns);
}

TaskResult Task::execute() const {
    if (status() != TaskStatus::NotStarted) {
        return TaskResult{TaskStatus::Failed, "Task not in runnable state"};
    }

    // Reset cancellation token before marking as running
    cancel_token_->reset();

    // Mark task as running
    set_status(TaskStatus::Running);

    TaskResult result{};

    try {
        result = std::visit(
                [this](auto &&func) -> TaskResult {
                    using FuncType = std::decay_t<decltype(func)>;

                    if (!func) {
                        return TaskResult{TaskStatus::Failed, "Task has no function to execute"};
                    }

                    if constexpr (std::is_same_v<FuncType, std::function<TaskResult()>>) {
                        return std::forward<decltype(func)>(func)();
                    } else {
                        const TaskContext context{cancel_token_, user_data_};
                        return std::forward<decltype(func)>(func)(context);
                    }
                },
                func_);

        // Check if task was cancelled during execution
        if (cancel_token_->is_cancelled()) {
            result = TaskResult{TaskStatus::Cancelled, "Task was cancelled during execution"};
        }
    } catch (const std::exception &e) {
        result = TaskResult{TaskStatus::Failed, std::format("Exception: {}", e.what())};
    } catch (...) {
        result = TaskResult{TaskStatus::Failed, "Unknown exception occurred"};
    }

    // Update final status
    set_status(result.status);

    return result;
}

void Task::cancel() const noexcept {
    cancel_token_->cancel();

    // If task hasn't started yet, mark it as cancelled
    TaskStatus expected = TaskStatus::NotStarted;
    status_->compare_exchange_strong(
            expected, TaskStatus::Cancelled, std::memory_order_release, std::memory_order_acquire);
}

Nanos Task::get_scheduled_time() const noexcept { return scheduled_time_; }

Nanos Task::get_timeout_ns() const noexcept { return timeout_ns_; }

std::string_view Task::get_task_name() const noexcept { return task_name_; }

std::string_view Task::get_graph_name() const noexcept { return graph_name_; }

std::uint64_t Task::get_task_id() const noexcept { return task_id_; }

TaskCategory Task::get_category() const noexcept { return category_; }

std::uint32_t Task::get_dependency_generation() const noexcept {
    return dependency_generation_->load(std::memory_order_acquire);
}

std::uint64_t Task::get_times_scheduled() const noexcept {
    return times_scheduled_->load(std::memory_order_acquire);
}

void Task::set_times_scheduled(const std::uint64_t times_scheduled) const noexcept {
    times_scheduled_->store(times_scheduled, std::memory_order_release);
}

TaskStatus Task::status() const noexcept {
    if (cancel_token_->is_cancelled()) {
        return TaskStatus::Cancelled;
    }
    return status_->load(std::memory_order_acquire);
}

void Task::set_status(TaskStatus new_status) const noexcept {
    status_->store(new_status, std::memory_order_release);
}

bool Task::is_cancelled() const noexcept { return cancel_token_->is_cancelled(); }

bool Task::has_no_parents() const noexcept { return parent_tasks_.empty(); }

bool Task::any_parent_matches(std::function<bool(TaskStatus)> predicate) const noexcept {
    return std::any_of(
            parent_tasks_.begin(), parent_tasks_.end(), [&predicate](const auto &parent_task) {
                return parent_task && predicate(parent_task->status());
            });
}

void Task::add_parent_task(const std::shared_ptr<Task> &parent_task) {
    if (parent_task) {
        parent_tasks_.push_back(parent_task);

        // Update dependency generation to be max(all_parents) + 1
        const std::uint32_t parent_generation = parent_task->get_dependency_generation();
        const std::uint32_t new_generation = parent_generation + 1;

        // Thread-safe update using compare-and-swap loop
        std::uint32_t current_generation = dependency_generation_->load(std::memory_order_acquire);
        while (current_generation < new_generation &&
               !dependency_generation_->compare_exchange_weak(
                       current_generation,
                       new_generation,
                       std::memory_order_release,
                       std::memory_order_acquire)) {
            // Loop continues until we successfully update or current >=
            // new_generation
        }
    }
}

void Task::reserve_parent_capacity(const std::size_t capacity) { parent_tasks_.reserve(capacity); }

void Task::reserve_name_capacity(
        const std::size_t max_task_name_length, const std::size_t max_graph_name_length) {
    task_name_.reserve(max_task_name_length);
    graph_name_.reserve(max_graph_name_length);
}

void Task::clear_parent_tasks() noexcept {
    parent_tasks_.clear();
    // Reset dependency generation since task has no parents now
    dependency_generation_->store(0, std::memory_order_release);
}

void Task::prepare_for_reuse(
        const std::string_view new_task_name,
        const std::string_view new_graph_name,
        const TaskFunction &new_func,
        const TaskCategory new_category,
        const Nanos new_timeout_ns,
        const Nanos new_scheduled_time,
        const std::any &new_user_data) {
    // Reset task state
    set_status(TaskStatus::NotStarted);
    cancel_token_->reset();

    // Clear parent dependencies
    clear_parent_tasks();

    // Update task configuration
    task_name_ = new_task_name;
    graph_name_ = new_graph_name;
    func_ = new_func;
    user_data_ = new_user_data;
    category_ = new_category;
    timeout_ns_ = new_timeout_ns;
    scheduled_time_ = new_scheduled_time;
    dependency_generation_->store(0, std::memory_order_release);
    times_scheduled_->store(0, std::memory_order_release);

    // Handle case for null functions
    if (std::holds_alternative<std::function<TaskResult()>>(func_)) {
        const auto &func = std::get<std::function<TaskResult()>>(func_);
        if (!func) {
            func_ = std::function<TaskResult()>{[]() { return TaskResult{}; }};
        }
    } else if (std::holds_alternative<std::function<TaskResult(const TaskContext &)>>(func_)) {
        const auto &func = std::get<std::function<TaskResult(const TaskContext &)>>(func_);
        if (!func) {
            func_ = std::function<TaskResult(const TaskContext &)>{
                    [](const TaskContext &) { return TaskResult{}; }};
        }
    }
}

// TaskBuilder implementation

TaskBuilder &TaskBuilder::user_data(std::any data) {
    user_data_ = std::move(data);
    return *this;
}

TaskBuilder &TaskBuilder::category(const TaskCategory cat) {
    category_ = cat;
    return *this;
}

TaskBuilder &TaskBuilder::category(const BuiltinTaskCategory builtin_cat) {
    category_ = TaskCategory{builtin_cat};
    return *this;
}

TaskBuilder &TaskBuilder::graph_name(const std::string_view graph_name) {
    graph_name_ = graph_name;
    return *this;
}

TaskBuilder &TaskBuilder::depends_on(std::shared_ptr<Task> parent_task) {
    if (parent_task) {
        parent_tasks_.push_back(std::move(parent_task));
    }
    return *this;
}

TaskBuilder &TaskBuilder::depends_on(const std::vector<std::shared_ptr<Task>> &parent_tasks) {
    for (const auto &parent : parent_tasks) {
        if (parent) {
            parent_tasks_.push_back(parent);
        }
    }
    return *this;
}

Task TaskBuilder::build() {
    if (task_name_.empty()) {
        throw std::invalid_argument("Task name cannot be empty");
    }

    // Create the task (dependency generation starts at 0)
    Task task{func_, user_data_, task_name_, graph_name_, category_, timeout_ns_, scheduled_time_};

    // Set up dependencies - add_parent_task() will automatically calculate
    // dependency generation
    for (const auto &parent : parent_tasks_) {
        task.add_parent_task(parent);
    }

    return task;
}

std::shared_ptr<Task> TaskBuilder::build_shared() { return std::make_shared<Task>(build()); }

} // namespace framework::task
