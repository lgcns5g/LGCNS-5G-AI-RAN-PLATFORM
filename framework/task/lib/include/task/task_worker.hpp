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
 * @file task_worker.hpp
 * @brief Worker configuration for task scheduler threads
 *
 * Provides configuration structures and utilities for managing worker thread
 * setup, core pinning, thread priority scheduling, and task category
 * assignment.
 */

#ifndef FRAMEWORK_TASK_TASK_WORKER_HPP
#define FRAMEWORK_TASK_TASK_WORKER_HPP

#include <cstdint>
#include <format>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "task/flat_map.hpp"
#include "task/task_category.hpp"

namespace framework::task {

/**
 * Core assignment configuration for explicit worker setup
 */
struct CoreAssignment final {
    std::uint32_t core_id; //!< CPU core ID
    std::optional<std::uint32_t>
            thread_priority; //!< Thread priority (1-99, higher = more priority)

    /**
     * Create core assignment with normal scheduling
     * @param[in] core CPU core ID to assign
     */
    explicit CoreAssignment(std::uint32_t core) : core_id(core) {}

    /**
     * Create core assignment with thread priority
     * @param[in] core CPU core ID to assign
     * @param[in] priority Thread priority level (1-99)
     */
    CoreAssignment(std::uint32_t core, std::uint32_t priority)
            : core_id(core), thread_priority(priority) {}
};

/**
 * Worker configuration for individual worker thread setup
 *
 * Configures core pinning, thread priority, and task category assignment
 * for individual worker threads in the task scheduler.
 */
struct WorkerConfig final {
    static constexpr std::uint32_t DEFAULT_PRIORITY = 50; //!< Default thread priority level

    std::optional<std::uint32_t> core_id; //!< CPU core to pin worker to (nullopt = no pinning)
    std::optional<std::uint32_t> thread_priority; //!< Thread priority level (1-99, higher = more
                                                  //!< priority)
    std::vector<TaskCategory> categories;         //!< Task categories this worker handles

    /// Default constructor - no pinning, normal scheduling, all categories
    WorkerConfig() { categories.emplace_back(BuiltinTaskCategory::Default); }

    /**
     * Check if worker is pinned to a specific core
     * @return true if core pinning is enabled, false otherwise
     */
    [[nodiscard]] bool is_pinned() const noexcept { return core_id.has_value(); }

    /**
     * Get the core ID for pinned workers
     * @return Optional core ID (nullopt if worker is not pinned)
     */
    [[nodiscard]] std::optional<std::uint32_t> get_core_id() const noexcept { return core_id; }

    /**
     * Check if worker uses real-time priority
     * @return true if RT priority is enabled, false otherwise
     */
    [[nodiscard]] bool has_thread_priority() const noexcept { return thread_priority.has_value(); }

    /**
     * Get the thread priority level
     * @return Thread priority (should only be called if has_thread_priority()
     * returns true)
     */
    [[nodiscard]] std::uint32_t get_thread_priority() const {
        return thread_priority.value_or(DEFAULT_PRIORITY);
    }

    /**
     * Validate worker configuration
     * @return true if configuration is valid, false otherwise
     */
    [[nodiscard]] bool is_valid() const;

    /**
     * Print worker configuration details
     * @param[in] worker_index Worker index for display
     */
    void print(const std::size_t worker_index) const;

    // Factory methods for common configurations

    /**
     * Create worker with core pinning and thread priority
     * @param[in] core CPU core to pin to
     * @param[in] priority Thread priority (1-99)
     * @param[in] worker_categories Task categories to handle
     * @return WorkerConfig with pinning and thread priority enabled
     */
    [[nodiscard]] static WorkerConfig create_pinned_rt(
            const std::uint32_t core,
            const std::uint32_t priority = DEFAULT_PRIORITY,
            const std::vector<TaskCategory> &worker_categories = {
                    TaskCategory{BuiltinTaskCategory::Default}});

    /**
     * Create worker with only core pinning (normal scheduling)
     * @param[in] core CPU core to pin to
     * @param[in] worker_categories Task categories to handle
     * @return WorkerConfig with only pinning enabled
     */
    [[nodiscard]] static WorkerConfig create_pinned(
            const std::uint32_t core,
            const std::vector<TaskCategory> &worker_categories = {
                    TaskCategory{BuiltinTaskCategory::Default}});

    /**
     * Create worker with thread priority but no pinning
     * @param[in] priority Thread priority (1-99)
     * @param[in] worker_categories Task categories to handle
     * @return WorkerConfig with thread priority enabled
     */
    [[nodiscard]] static WorkerConfig create_rt_only(
            const std::uint32_t priority,
            const std::vector<TaskCategory> &worker_categories = {
                    TaskCategory{BuiltinTaskCategory::Default}});

    /**
     * Create worker for specific categories (normal priority, no pinning)
     * @param[in] worker_categories Task categories to handle
     * @return WorkerConfig for specific categories
     */
    [[nodiscard]] static WorkerConfig
    create_for_categories(const std::vector<TaskCategory> &worker_categories);
};

/**
 * Configuration for all workers in the task scheduler
 */
struct WorkersConfig final {
    std::vector<WorkerConfig> workers; //!< Individual worker configurations

    /**
     * Default constructor creates workers with default configuration
     * @param[in] num_workers Number of worker threads to create
     */
    explicit WorkersConfig(const std::size_t num_workers = 4) { workers.resize(num_workers); }

    /**
     * Constructor with explicit worker configs
     * @param[in] worker_configs Vector of worker configurations
     */
    explicit WorkersConfig(const std::vector<WorkerConfig> &worker_configs)
            : workers{worker_configs} {}

    /**
     * Validate all worker configurations
     * @return true if all configurations are valid, false otherwise
     */
    [[nodiscard]] bool is_valid() const;

    /// Print configuration details for all workers
    void print() const;

    /**
     * Get number of workers
     * @return Worker count
     */
    [[nodiscard]] std::size_t size() const noexcept { return workers.size(); }

    /**
     * Get worker configuration by index (const)
     * @param[in] index Worker index
     * @return Const reference to worker configuration
     * @throws std::out_of_range if index is out of bounds
     */
    [[nodiscard]] const WorkerConfig &operator[](const std::size_t index) const {
        if (index >= workers.size()) {
            throw std::out_of_range(std::format(
                    "Worker index {} is out of range (size: {})", index, workers.size()));
        }
        return workers[index];
    }

    /**
     * Get worker configuration by index (mutable)
     * @param[in] index Worker index
     * @return Reference to worker configuration
     * @throws std::out_of_range if index is out of bounds
     */
    [[nodiscard]] WorkerConfig &operator[](const std::size_t index) {
        if (index >= workers.size()) {
            throw std::out_of_range(std::format(
                    "Worker index {} is out of range (size: {})", index, workers.size()));
        }
        return workers[index];
    }

    // Factory methods for common configurations

    /**
     * Create configuration with workers for specific categories
     *
     * Creates workers for the specified categories without core pinning or
     * priority settings. Each worker is assigned to handle exactly one category.
     * The iteration order over categories is non-deterministic but the total
     * count of workers per category is guaranteed.
     *
     * @param[in] category_workers Map of categories to number of workers
     * @return WorkersConfig with unpinned workers for specified categories
     */
    [[nodiscard]] static WorkersConfig
    create_for_categories(const FlatMap<TaskCategory, std::size_t> &category_workers);
};

} // namespace framework::task

#endif // FRAMEWORK_TASK_TASK_WORKER_HPP
