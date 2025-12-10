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
 * @file task_graph.cpp
 * @brief Implementation of fluent API for building and managing task graphs
 */

#include <algorithm>
#include <any>
#include <cstddef>
#include <cstdint>
#include <format>
#include <memory>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <parallel_hashmap/phmap.h>
#include <quill/LogMacros.h>

#include "log/rt_log_macros.hpp"
#include "task/flat_map.hpp"
#include "task/task.hpp"
#include "task/task_category.hpp"
#include "task/task_graph.hpp"
#include "task/task_log.hpp"
#include "task/task_pool.hpp"
#include "task/task_visualizer.hpp"
#include "task/time.hpp"

namespace framework::task {

// TaskGraphBuilder implementation
TaskGraphBuilder::TaskGraphBuilder(TaskGraph &graph, const std::string_view task_name)
        : graph_(graph) {
    reset_builder();
    current_task_.task_name = task_name;
}

TaskGraphBuilder &TaskGraphBuilder::category(TaskCategory cat) {
    current_task_.category = cat;
    return *this;
}

TaskGraphBuilder &TaskGraphBuilder::category(BuiltinTaskCategory builtin_cat) {
    current_task_.category = TaskCategory{builtin_cat};
    return *this;
}

TaskGraphBuilder &TaskGraphBuilder::task_pool_capacity_multiplier(const std::size_t multiplier) {
    graph_.task_pool_capacity_multiplier_ = multiplier;
    return *this;
}

TaskGraphBuilder &TaskGraphBuilder::depends_on(const std::string_view parent_name) {
    if (!parent_name.empty()) {
        dependencies_.emplace_back(parent_name);
    }
    return *this;
}

TaskGraphBuilder &TaskGraphBuilder::depends_on(const std::vector<std::string_view> &parent_names) {
    for (const auto &parent_name : parent_names) {
        if (!parent_name.empty()) {
            dependencies_.emplace_back(parent_name);
        }
    }
    return *this;
}

TaskGraphBuilder &TaskGraphBuilder::user_data(std::any data) {
    current_task_.user_data = std::move(data);
    return *this;
}

std::string TaskGraphBuilder::add() {
    if (current_task_.task_name.empty()) {
        throw std::invalid_argument("Task name cannot be empty");
    }

    // Check if variant holds a valid function
    const bool has_function = std::visit(
            [](const auto &func) { return static_cast<bool>(func); }, current_task_.func);
    if (!has_function) {
        throw std::invalid_argument("Task function cannot be empty");
    }

    // Set dependencies
    current_task_.dependency_names = std::move(dependencies_);

    // Store the name before moving
    std::string task_name = current_task_.task_name;

    // Add to graph
    graph_.add_schedulable_task(std::move(current_task_));

    // Reset for next task
    reset_builder();

    return task_name;
}

void TaskGraphBuilder::reset_builder() {
    current_task_ = SchedulableTask{};
    dependencies_.clear();
}

// TaskGraph::create implementation
SingleTaskGraphBuilder TaskGraph::create(const std::string_view graph_name) {
    return SingleTaskGraphBuilder(graph_name);
}

// SingleTaskGraphBuilder implementation
SingleTaskGraphBuilder::SingleTaskGraphBuilder(const std::string_view graph_name)
        : graph_(std::make_unique<TaskGraph>(graph_name)) {}

SingleTaskGraphBuilder &SingleTaskGraphBuilder::single_task(const std::string_view task_name) {
    // Create the TaskGraphBuilder and store it
    task_builder_ = std::make_unique<TaskGraphBuilder>(*graph_, task_name);
    return *this;
}

// SingleTaskGraphBuilder methods are now inline in header

TaskGraph SingleTaskGraphBuilder::build() {
    if (!task_builder_) {
        throw std::invalid_argument("Task name not set - call single_task() first");
    }

    // Add the task and build the graph
    task_builder_->add();
    graph_->build();

    // Move the built graph out
    return std::move(*graph_);
}

// TaskGraph implementation
void TaskGraph::clear_scheduled_tasks() { scheduled_tasks_.clear(); }

// single_task method is removed - use TaskGraph::create(name).single_task(name)
// instead

TaskGraphBuilder TaskGraph::register_task(const std::string_view task_name) {
    return {*this, task_name};
}

void TaskGraph::clear() {
    task_specs_.clear();
    task_name_to_index_.clear();
    scheduled_tasks_.clear();
    task_pool_.reset(); // Release the TaskPool
    graph_visualizer_.clear();
    is_built_ = false;
    times_scheduled_ = 0;
}

void TaskGraph::add_schedulable_task(SchedulableTask spec) {
    validate_not_built();

    // Check for duplicate names
    if (task_name_to_index_.find(spec.task_name) != task_name_to_index_.end()) {
        throw std::invalid_argument("Task name '" + spec.task_name + "' already exists in graph");
    }

    // Validate dependencies exist and pre-compute parent indices
    spec.parent_indices.reserve(spec.dependency_names.size());
    for (const auto &dep_name : spec.dependency_names) {
        auto dep_it = task_name_to_index_.find(dep_name);
        if (dep_it == task_name_to_index_.end()) {
            throw std::invalid_argument("Dependency '" + dep_name + "' not found in graph");
        }
        spec.parent_indices.push_back(dep_it->second);
    }

    // Add to index
    task_name_to_index_[spec.task_name] = task_specs_.size();

    // Add to graph visualizer
    graph_visualizer_.add_node(spec.task_name, spec.category);

    // Add edges for dependencies
    for (const auto &dep : spec.dependency_names) {
        graph_visualizer_.add_edge(dep, spec.task_name);
    }

    // Add specification
    task_specs_.push_back(std::move(spec));
}

void TaskGraph::build() {
    if (is_built_) {
        RT_LOGC_WARN(TaskLog::TaskGraph, "TaskGraph already built, skipping");
        return;
    }

    if (task_specs_.empty()) {
        RT_LOGC_WARN(TaskLog::TaskGraph, "Building empty TaskGraph");
        is_built_ = true;
        return;
    }

    // Reset scheduling count when building
    times_scheduled_ = 0;

    RT_LOGC_INFO(TaskLog::TaskGraph, "Building TaskGraph with {} tasks", task_specs_.size());

    // Calculate maximum number of parents for TaskPool initialization
    std::size_t max_parents = 0;
    std::size_t max_task_name_length = 0;
    const std::size_t max_graph_name_length = graph_name_.length();

    for (const auto &spec : task_specs_) {
        max_parents = std::max(max_parents, spec.dependency_names.size());
        max_task_name_length = std::max(max_task_name_length, spec.task_name.length());
    }

    // Create TaskPool with configurable capacity multiplier
    const std::size_t pool_capacity = task_specs_.size() * task_pool_capacity_multiplier_;
    task_pool_ = TaskPool::create(
            pool_capacity, max_parents, max_task_name_length, max_graph_name_length);

    RT_LOGC_DEBUG(
            TaskLog::TaskGraph,
            "Initialized TaskPool with capacity {} ({}x graph size), max "
            "parents: {}, "
            "max task name length: {}, max graph name length: {}",
            pool_capacity,
            task_pool_capacity_multiplier_,
            max_parents,
            max_task_name_length,
            max_graph_name_length);

    scheduled_tasks_.clear();
    scheduled_tasks_.reserve(task_specs_.size());

    // Sort task_specs_ in dependency order
    const std::vector<std::size_t> build_order = compute_dependency_ordered_indices();

    // Create sorted specs and keep track of old→new mapping for updating
    // parent_indices
    std::vector<SchedulableTask> sorted_specs(task_specs_.size());
    std::vector<std::size_t> old_to_new_index(task_specs_.size());

    for (std::size_t new_idx = 0; new_idx < build_order.size(); ++new_idx) {
        const std::size_t old_idx = build_order[new_idx];
        sorted_specs[new_idx] = std::move(task_specs_[old_idx]);
        old_to_new_index[old_idx] = new_idx;
    }

    // Replace task_specs_ with sorted version
    task_specs_ = std::move(sorted_specs);

    // Update task_name_to_index_ to reflect new sorted positions
    task_name_to_index_.clear();
    for (std::size_t i = 0; i < task_specs_.size(); ++i) {
        task_name_to_index_[task_specs_[i].task_name] = i;
    }

    // Update parent_indices to reference new sorted positions
    for (auto &spec : task_specs_) {
        for (auto &parent_idx : spec.parent_indices) {
            parent_idx = old_to_new_index[parent_idx];
        }
    }

    // Build phase complete - tasks will be acquired from pool during scheduling
    // scheduled_tasks_ is reserved but not populated until prepare_tasks() is
    // called

    is_built_ = true;
    RT_LOGC_INFO(TaskLog::TaskGraph, "TaskGraph built successfully with dependencies resolved");
}

std::vector<std::shared_ptr<Task>> &TaskGraph::prepare_tasks(Nanos execution_time) {
    if (!is_built_) {
        throw std::runtime_error("TaskGraph has not been built. Call build() first.");
    }

    // Clear previous scheduling round
    clear_scheduled_tasks();

    // Acquire fresh tasks from pool for this scheduling round
    for (const auto &spec : task_specs_) {

        // Acquire task from pool and configure with variant function
        auto pooled_task = task_pool_->acquire_task(spec.task_name, graph_name_);
        pooled_task->prepare_for_reuse(
                spec.task_name,
                graph_name_,
                spec.func,
                spec.category,
                spec.timeout,
                execution_time,
                spec.user_data);

        // Reserve parent capacity to avoid allocations during dependency setup
        pooled_task->reserve_parent_capacity(spec.parent_indices.size());

        // Add parent dependencies - this will automatically calculate dependency
        // generation
        for (const std::size_t parent_idx : spec.parent_indices) {
            pooled_task->add_parent_task(scheduled_tasks_[parent_idx]);
        }

        scheduled_tasks_.push_back(pooled_task);
    }

    return scheduled_tasks_;
}

bool TaskGraph::task_has_status(
        const std::string_view name, const TaskStatus expected_status) const {
    if (!is_built_) {
        throw std::runtime_error("TaskGraph has not been built. Call build() first.");
    }

    auto it = task_name_to_index_.find(std::string{name});
    if (it != task_name_to_index_.end() && it->second < scheduled_tasks_.size()) {
        return scheduled_tasks_[it->second]->status() == expected_status;
    }
    return false; // Task not found
}

bool TaskGraph::set_task_status(const std::string_view name, const TaskStatus new_status) {
    if (!is_built_) {
        throw std::runtime_error("TaskGraph has not been built. Call build() first.");
    }

    auto it = task_name_to_index_.find(std::string{name});
    if (it != task_name_to_index_.end() && it->second < scheduled_tasks_.size()) {
        if (new_status == TaskStatus::Cancelled) {
            // Use proper cancellation method to set both status and cancellation
            // token
            scheduled_tasks_[it->second]->cancel();
        } else {
            // For other statuses, just set the status directly
            scheduled_tasks_[it->second]->set_status(new_status);
        }
        return true;
    }
    return false; // Task not found
}

void TaskGraph::validate_not_built() const {
    if (is_built_) {
        throw std::runtime_error("Cannot modify TaskGraph after build(). Call "
                                 "clear() first to reset the graph.");
    }
}

std::string TaskGraph::to_string() const { return graph_visualizer_.to_string(); }

TaskPoolStats TaskGraph::get_pool_stats() const {
    if (!is_built_ || !task_pool_) {
        throw std::runtime_error("TaskGraph has not been built. Call build() first.");
    }
    return task_pool_->get_stats();
}

std::vector<std::uint32_t> TaskGraph::calculate_dependency_generations() const {
    const std::size_t num_tasks = task_specs_.size();
    std::vector<std::uint32_t> generations(num_tasks, 0);

    if (num_tasks == 0) {
        return generations;
    }

    // Build adjacency lists for dependency traversal
    // parents[i] = indices of tasks that task i depends on
    // children[i] = indices of tasks that depend on task i
    std::vector<std::vector<std::size_t>> parents(num_tasks);
    std::vector<std::vector<std::size_t>> children(num_tasks);
    std::vector<std::size_t> in_degree(num_tasks, 0);

    for (std::size_t i = 0; i < num_tasks; ++i) {
        const auto &spec = task_specs_[i];

        for (const auto &parent_name : spec.dependency_names) {
            const auto parent_it = task_name_to_index_.find(parent_name);
            if (parent_it != task_name_to_index_.end()) {
                const std::size_t parent_idx = parent_it->second;
                parents[i].push_back(parent_idx);
                children[parent_idx].push_back(i);
                ++in_degree[i];
            }
        }
    }

    // Use Kahn's topological sorting algorithm to calculate generations
    std::queue<std::size_t> queue{};

    // Start with all root tasks (generation 0)
    for (std::size_t i = 0; i < num_tasks; ++i) {
        if (in_degree[i] == 0) {
            queue.push(i);
            generations[i] = 0;
        }
    }

    while (!queue.empty()) {
        const std::size_t current_idx = queue.front();
        queue.pop();

        // Calculate generation for all children
        for (const std::size_t child_idx : children[current_idx]) {
            // Child's generation is max of (all parents' generations + 1)
            const std::uint32_t max_parent_generation = std::accumulate(
                    parents[child_idx].begin(),
                    parents[child_idx].end(),
                    std::uint32_t{0},
                    [&generations](const std::uint32_t current_max, const std::size_t parent_idx) {
                        return std::max(current_max, generations[parent_idx]);
                    });

            generations[child_idx] = std::max(generations[child_idx], max_parent_generation + 1);

            // Decrease in-degree and add to queue if all dependencies processed
            --in_degree[child_idx];
            if (in_degree[child_idx] == 0) {
                queue.push(child_idx);
            }
        }
    }

    // Check for cycles - all tasks should have been processed
    for (std::size_t i = 0; i < num_tasks; ++i) {
        if (in_degree[i] > 0) {
            throw std::runtime_error(std::format(
                    "Circular dependency detected involving task '{}'", task_specs_[i].task_name));
        }
    }

    // Log generation assignments for debugging
    for (std::size_t i = 0; i < num_tasks; ++i) {
        RT_LOGC_DEBUG(
                TaskLog::TaskGraph,
                "Graph '{}' Task '{}' assigned dependency generation {}",
                graph_name_,
                task_specs_[i].task_name,
                generations[i]);
    }

    return generations;
}

std::vector<std::size_t> TaskGraph::compute_dependency_ordered_indices() const {
    const auto generations = calculate_dependency_generations();

    // Create build order indices
    std::vector<std::size_t> build_order(task_specs_.size());
    std::iota(build_order.begin(), build_order.end(), 0);

    // Sort by dependency generation, maintaining stable order within generations
    std::stable_sort(
            build_order.begin(),
            build_order.end(),
            [&generations](const std::size_t a, const std::size_t b) {
                return generations[a] < generations[b];
            });

    return build_order;
}

bool TaskGraph::disable_task(const std::string_view task_name) {
    const auto it = task_name_to_index_.find(std::string{task_name});
    if (it == task_name_to_index_.end()) {
        return false; // Task not found
    }

    task_specs_[it->second].disabled = true;
    return true;
}

bool TaskGraph::enable_task(const std::string_view task_name) {
    const auto it = task_name_to_index_.find(std::string{task_name});
    if (it == task_name_to_index_.end()) {
        return false; // Task not found
    }

    task_specs_[it->second].disabled = false;
    return true;
}

bool TaskGraph::is_task_disabled(const std::string_view task_name) const {
    const auto it = task_name_to_index_.find(std::string{task_name});
    if (it == task_name_to_index_.end()) {
        return false; // Task not found, consider as enabled
    }

    return task_specs_[it->second].disabled;
}

bool TaskGraph::is_task_or_parent_disabled(std::size_t task_index) const {
    // Bounds check to prevent out-of-range access
    if (task_index >= task_specs_.size()) {
        RT_LOGC_ERROR(
                TaskLog::TaskGraph, "invalid task index: {} (task specs size {})", task_index);
        return false; // Invalid index, consider as enabled
    }

    const auto &spec = task_specs_[task_index];

    // Check if this task is directly disabled
    if (spec.disabled) {
        return true;
    }

    // Check if any parent is disabled (recursive check)
    return std::any_of(
            spec.parent_indices.begin(),
            spec.parent_indices.end(),
            [this](const std::size_t parent_idx) {
                return is_task_or_parent_disabled(parent_idx);
            });
}

} // namespace framework::task
