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
#include <cstddef>
#include <cstdint>
#include <exception>
#include <format>
#include <fstream>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <parallel_hashmap/phmap.h>
#include <quill/LogMacros.h>

#include "log/rt_log_macros.hpp"
#include "task/flat_map.hpp"
#include "task/task_category.hpp"
#include "task/task_log.hpp"
#include "task/task_utils.hpp"
#include "task/task_worker.hpp"

namespace framework::task {

// WorkerConfig implementation

bool WorkerConfig::is_valid() const {
    // Validate core ID against system capabilities
    if (core_id.has_value()) {
        const auto max_cores = std::thread::hardware_concurrency();
        if (core_id.value() >= max_cores) {
            RT_LOGC_ERROR(
                    TaskLog::TaskScheduler,
                    "Invalid core ID {}: system has {} cores",
                    core_id.value(),
                    max_cores);
            return false;
        }
    }

    // Validate thread priority range
    static constexpr std::uint32_t MAX_PRIORITY = 99;
    if (thread_priority.has_value() &&
        (thread_priority.value() < 1 || thread_priority.value() > MAX_PRIORITY)) {
        RT_LOGC_ERROR(
                TaskLog::TaskScheduler,
                "Invalid thread priority {}: must be between 1 and 99",
                thread_priority.value());
        return false;
    }

    // Validate core configuration for RT workloads if thread priority is enabled
    // and pinned
    if (thread_priority.has_value() && is_pinned()) {
        try {
            std::ifstream cmdline_file{"/proc/cmdline"};
            if (cmdline_file.is_open()) {
                std::string cmdline;
                std::getline(cmdline_file, cmdline);
                cmdline_file.close();
                // Note: RT core validation warnings are logged in
                // validate_rt_core_config
                std::ignore =
                        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
                        validate_rt_core_config(cmdline, {get_core_id().value()});
            }
        } catch (const std::exception &e) {
            RT_LOGC_WARN(
                    TaskLog::TaskScheduler,
                    "Failed to read kernel cmdline for RT validation: {}",
                    e.what());
        }
    }

    // Must have at least one category
    if (categories.empty()) {
        RT_LOGC_ERROR(TaskLog::TaskScheduler, "Worker must handle at least one category");
        return false;
    }

    return true;
}

void WorkerConfig::print(const std::size_t worker_index) const {
    std::string worker_info = std::format("  Worker {}: ", worker_index);

    if (is_pinned()) {
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        worker_info += std::format("Core {}", get_core_id().value());
    } else {
        worker_info += "No pinning";
    }

    if (thread_priority.has_value()) {
        worker_info += std::format(", RT priority {}", thread_priority.value());
    } else {
        worker_info += ", Normal scheduling";
    }

    worker_info += ", Categories: ";
    for (std::size_t i = 0; i < categories.size(); ++i) {
        if (i > 0) {
            worker_info += ", ";
        }
        worker_info += std::string{categories[i].name()};
    }

    RT_LOGC_INFO(TaskLog::TaskScheduler, "{}", worker_info);
}

WorkerConfig WorkerConfig::create_pinned_rt(
        const std::uint32_t core,
        const std::uint32_t priority,
        const std::vector<TaskCategory> &worker_categories) {
    WorkerConfig config{};
    config.core_id = core;
    config.thread_priority = priority;
    config.categories = worker_categories;
    return config;
}

WorkerConfig WorkerConfig::create_pinned(
        const std::uint32_t core, const std::vector<TaskCategory> &worker_categories) {
    WorkerConfig config{};
    config.core_id = core;
    config.categories = worker_categories;
    return config;
}

WorkerConfig WorkerConfig::create_rt_only(
        const std::uint32_t priority, const std::vector<TaskCategory> &worker_categories) {
    WorkerConfig config{};
    config.thread_priority = priority;
    config.categories = worker_categories;
    return config;
}

WorkerConfig
WorkerConfig::create_for_categories(const std::vector<TaskCategory> &worker_categories) {
    WorkerConfig config{};
    config.categories = worker_categories;
    return config;
}

// WorkersConfig implementation

bool WorkersConfig::is_valid() const {
    if (workers.empty()) {
        RT_LOGC_ERROR(TaskLog::TaskScheduler, "No workers configured");
        return false;
    }

    // Validate each worker config
    if (!std::all_of(workers.begin(), workers.end(), [](const auto &worker) {
            return worker.is_valid();
        })) {
        return false;
    }

    // Check for duplicate core assignments (only for pinned workers)
    std::set<std::uint32_t> used_cores;
    for (const auto &worker : workers) {
        if (worker.is_pinned()) {
            // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
            const std::uint32_t core = worker.get_core_id().value();
            if (used_cores.contains(core)) {
                RT_LOGC_ERROR(TaskLog::TaskScheduler, "Duplicate core assignment: core {}", core);
                return false;
            }
            used_cores.insert(core);
        }
    }

    return true;
}

void WorkersConfig::print() const {
    RT_LOGC_INFO(TaskLog::TaskScheduler, "Workers Configuration ({} workers):", workers.size());
    for (std::size_t i = 0; i < workers.size(); ++i) {
        workers[i].print(i);
    }
}

WorkersConfig
WorkersConfig::create_for_categories(const FlatMap<TaskCategory, std::size_t> &category_workers) {
    std::vector<WorkerConfig> worker_configs{};

    for (const auto &[category, worker_count] : category_workers) {
        RT_LOGC_DEBUG(
                TaskLog::TaskScheduler,
                "Creating {} workers for category '{}'",
                worker_count,
                std::string{category.name()});

        for (std::size_t i = 0; i < worker_count; ++i) {
            RT_LOGC_DEBUG(
                    TaskLog::TaskScheduler,
                    "  Worker {}: No pinning, Normal scheduling, Category '{}'",
                    worker_configs.size(),
                    std::string{category.name()});
            worker_configs.push_back(WorkerConfig::create_for_categories({category}));
        }
    }

    return WorkersConfig{worker_configs};
}

} // namespace framework::task
