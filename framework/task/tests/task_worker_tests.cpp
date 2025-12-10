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
 * @file task_worker_tests.cpp
 * @brief Unit tests for WorkerConfig and WorkersConfig
 */

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "task/flat_map.hpp"
#include "task/task_category.hpp"
#include "task/task_worker.hpp"

namespace {
namespace ft = framework::task;

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

/**
 * WorkerConfig tests - basic functionality and validation
 */
TEST(WorkerConfig, DefaultConstruction) {
    const ft::WorkerConfig config{};

    EXPECT_FALSE(config.is_pinned());
    EXPECT_FALSE(config.has_thread_priority());
    EXPECT_EQ(config.get_thread_priority(), 50); // Default priority
    EXPECT_EQ(config.categories.size(), 1);
    EXPECT_EQ(config.categories[0], ft::TaskCategory{ft::BuiltinTaskCategory::Default});
    EXPECT_TRUE(config.is_valid());
}

TEST(WorkerConfig, ValidationMatrix) {
    struct ValidationTestCase {
        std::string name;
        std::optional<std::uint32_t> core_id;
        std::optional<std::uint32_t> thread_priority;
        std::vector<ft::TaskCategory> categories;
        bool expected_valid;
        std::string description;
    };

    const std::uint32_t max_cores = std::thread::hardware_concurrency();
    const std::vector<ValidationTestCase> test_cases = {
            {"Valid_Basic",
             std::nullopt,
             std::nullopt,
             {ft::TaskCategory{ft::BuiltinTaskCategory::Default}},
             true,
             "Basic valid config"},
            {"Valid_Pinned",
             std::optional<std::uint32_t>{0},
             std::nullopt,
             {ft::TaskCategory{ft::BuiltinTaskCategory::Default}},
             true,
             "Valid core pinning"},
            {"Valid_ThreadPriority",
             std::nullopt,
             std::optional<std::uint32_t>{75},
             {ft::TaskCategory{ft::BuiltinTaskCategory::Default}},
             true,
             "Valid thread priority"},
            {"Valid_Both",
             std::optional<std::uint32_t>{0},
             std::optional<std::uint32_t>{25},
             {ft::TaskCategory{ft::BuiltinTaskCategory::HighPriority}},
             true,
             "Valid pinning + thread priority"},
            {"Invalid_HighCore",
             std::optional<std::uint32_t>{max_cores + 10},
             std::nullopt,
             {ft::TaskCategory{ft::BuiltinTaskCategory::Default}},
             false,
             "Invalid high core ID"},
            {"Invalid_LowThreadPriority",
             std::nullopt,
             std::optional<std::uint32_t>{0},
             {ft::TaskCategory{ft::BuiltinTaskCategory::Default}},
             false,
             "Thread priority too low"},
            {"Invalid_HighThreadPriority",
             std::nullopt,
             std::optional<std::uint32_t>{100},
             {ft::TaskCategory{ft::BuiltinTaskCategory::Default}},
             false,
             "Thread priority too high"},
            {"Invalid_EmptyCategories",
             std::nullopt,
             std::nullopt,
             {},
             false,
             "No categories specified"},
            {"Valid_MultipleCategories",
             std::nullopt,
             std::nullopt,
             {ft::TaskCategory{ft::BuiltinTaskCategory::IO},
              ft::TaskCategory{ft::BuiltinTaskCategory::Compute}},
             true,
             "Multiple categories"}};

    for (const auto &test_case : test_cases) {
        SCOPED_TRACE("Testing: " + test_case.name + " - " + test_case.description);

        ft::WorkerConfig config{};
        config.core_id = test_case.core_id;
        config.thread_priority = test_case.thread_priority;
        config.categories = test_case.categories;

        EXPECT_EQ(config.is_valid(), test_case.expected_valid);

        if (test_case.core_id.has_value()) {
            EXPECT_TRUE(config.is_pinned());
            if (test_case.expected_valid) {
                ASSERT_TRUE(config.get_core_id().has_value());
                // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
                EXPECT_EQ(config.get_core_id().value(), test_case.core_id.value());
            }
        } else {
            EXPECT_FALSE(config.is_pinned());
        }

        if (test_case.thread_priority.has_value()) {
            EXPECT_TRUE(config.has_thread_priority());
            if (test_case.expected_valid) {
                EXPECT_EQ(config.get_thread_priority(), test_case.thread_priority.value());
            }
        } else {
            EXPECT_FALSE(config.has_thread_priority());
        }
    }
}

TEST(WorkerConfig, FactoryMethods) {
    // Test create_pinned_rt
    const auto pinned_rt = ft::WorkerConfig::create_pinned_rt(
            1, 80, {ft::TaskCategory{ft::BuiltinTaskCategory::HighPriority}});
    EXPECT_TRUE(pinned_rt.is_pinned());
    ASSERT_TRUE(pinned_rt.get_core_id().has_value());
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    EXPECT_EQ(pinned_rt.get_core_id().value(), 1);
    EXPECT_TRUE(pinned_rt.has_thread_priority());
    EXPECT_EQ(pinned_rt.get_thread_priority(), 80);
    EXPECT_EQ(pinned_rt.categories.size(), 1);
    EXPECT_EQ(pinned_rt.categories[0], ft::TaskCategory{ft::BuiltinTaskCategory::HighPriority});
    EXPECT_TRUE(pinned_rt.is_valid());

    // Test create_pinned
    const auto pinned_only = ft::WorkerConfig::create_pinned(
            2,
            {ft::TaskCategory{ft::BuiltinTaskCategory::IO},
             ft::TaskCategory{ft::BuiltinTaskCategory::Network}});
    EXPECT_TRUE(pinned_only.is_pinned());
    ASSERT_TRUE(pinned_only.get_core_id().has_value());
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    EXPECT_EQ(pinned_only.get_core_id().value(), 2);
    EXPECT_FALSE(pinned_only.has_thread_priority());
    EXPECT_EQ(pinned_only.categories.size(), 2);
    EXPECT_TRUE(pinned_only.is_valid());

    // Test create_rt_only
    const auto rt_only = ft::WorkerConfig::create_rt_only(
            60, {ft::TaskCategory{ft::BuiltinTaskCategory::Compute}});
    EXPECT_FALSE(rt_only.is_pinned());
    EXPECT_TRUE(rt_only.has_thread_priority());
    EXPECT_EQ(rt_only.get_thread_priority(), 60);
    EXPECT_EQ(rt_only.categories.size(), 1);
    EXPECT_EQ(rt_only.categories[0], ft::TaskCategory{ft::BuiltinTaskCategory::Compute});
    EXPECT_TRUE(rt_only.is_valid());

    // Test create_for_categories
    const auto for_categories = ft::WorkerConfig::create_for_categories(
            {ft::TaskCategory{ft::BuiltinTaskCategory::LowPriority}});
    EXPECT_FALSE(for_categories.is_pinned());
    EXPECT_FALSE(for_categories.has_thread_priority());
    EXPECT_EQ(for_categories.categories.size(), 1);
    EXPECT_EQ(for_categories.categories[0], ft::TaskCategory{ft::BuiltinTaskCategory::LowPriority});
    EXPECT_TRUE(for_categories.is_valid());
}

/**
 * WorkersConfig tests - configuration management
 */
TEST(WorkersConfig, DefaultConstruction) {
    const ft::WorkersConfig config{};

    EXPECT_EQ(config.size(), 4); // Default is 4 workers
    EXPECT_TRUE(config.is_valid());

    // Each worker should have default config
    for (std::size_t i = 0; i < config.size(); ++i) {
        const auto &worker = config[i];
        EXPECT_FALSE(worker.is_pinned());
        EXPECT_FALSE(worker.has_thread_priority());
        EXPECT_EQ(worker.categories.size(), 1);
        EXPECT_EQ(worker.categories[0], ft::TaskCategory{ft::BuiltinTaskCategory::Default});
    }
}

TEST(WorkersConfig, CustomConstruction) {
    std::vector<ft::WorkerConfig> worker_configs{};
    worker_configs.push_back(ft::WorkerConfig::create_pinned_rt(0, 70));
    worker_configs.push_back(ft::WorkerConfig::create_pinned(1));
    worker_configs.push_back(ft::WorkerConfig::create_rt_only(60));

    const ft::WorkersConfig config{worker_configs};

    EXPECT_EQ(config.size(), 3);
    EXPECT_TRUE(config.is_valid());

    EXPECT_TRUE(config[0].is_pinned());
    EXPECT_TRUE(config[0].has_thread_priority());

    EXPECT_TRUE(config[1].is_pinned());
    EXPECT_FALSE(config[1].has_thread_priority());

    EXPECT_FALSE(config[2].is_pinned());
    EXPECT_TRUE(config[2].has_thread_priority());
}

TEST(WorkersConfig, ValidationEdgeCases) {
    // Empty workers config should be invalid
    const ft::WorkersConfig empty_config{std::vector<ft::WorkerConfig>{}};
    EXPECT_FALSE(empty_config.is_valid());

    // Duplicate core assignments should be invalid
    std::vector<ft::WorkerConfig> duplicate_cores{};
    duplicate_cores.push_back(ft::WorkerConfig::create_pinned(0));
    duplicate_cores.push_back(ft::WorkerConfig::create_pinned(0)); // Same core

    const ft::WorkersConfig invalid_config{duplicate_cores};
    EXPECT_FALSE(invalid_config.is_valid());

    // One invalid worker should make entire config invalid
    std::vector<ft::WorkerConfig> mixed_configs{};
    mixed_configs.push_back(ft::WorkerConfig::create_pinned(0));
    ft::WorkerConfig invalid_worker{};
    invalid_worker.categories.clear(); // Make it invalid
    mixed_configs.push_back(invalid_worker);

    const ft::WorkersConfig mixed_invalid{mixed_configs};
    EXPECT_FALSE(mixed_invalid.is_valid());
}

TEST(WorkersConfig, FactoryCreateForCategoriesBasic) {
    ft::FlatMap<ft::TaskCategory, std::size_t> category_workers{};
    category_workers[ft::TaskCategory{ft::BuiltinTaskCategory::HighPriority}] = 2;
    category_workers[ft::TaskCategory{ft::BuiltinTaskCategory::IO}] = 1;
    category_workers[ft::TaskCategory{ft::BuiltinTaskCategory::Compute}] = 3;

    const auto config = ft::WorkersConfig::create_for_categories(category_workers);

    EXPECT_EQ(config.size(), 6); // 2+1+3
    EXPECT_TRUE(config.is_valid());

    // All workers should be unpinned with normal scheduling
    for (const auto &worker : config.workers) {
        EXPECT_FALSE(worker.is_pinned());
        EXPECT_FALSE(worker.has_thread_priority());
        EXPECT_EQ(worker.categories.size(),
                  1); // Each worker handles exactly one category
    }

    // Test category distribution (order-agnostic)
    std::map<ft::TaskCategory, std::size_t> actual_category_counts{};
    for (const auto &worker : config.workers) {
        ++actual_category_counts[worker.categories[0]];
    }

    std::map<ft::TaskCategory, std::size_t> expected_category_counts{};
    expected_category_counts[ft::TaskCategory{ft::BuiltinTaskCategory::HighPriority}] = 2;
    expected_category_counts[ft::TaskCategory{ft::BuiltinTaskCategory::IO}] = 1;
    expected_category_counts[ft::TaskCategory{ft::BuiltinTaskCategory::Compute}] = 3;

    EXPECT_EQ(actual_category_counts, expected_category_counts);
}

TEST(WorkersConfig, FactoryCreateForCategoriesMultiple) {
    ft::FlatMap<ft::TaskCategory, std::size_t> category_workers{};
    category_workers[ft::TaskCategory{ft::BuiltinTaskCategory::HighPriority}] = 2;
    category_workers[ft::TaskCategory{ft::BuiltinTaskCategory::IO}] = 1;
    category_workers[ft::TaskCategory{ft::BuiltinTaskCategory::Compute}] = 2;

    const auto config = ft::WorkersConfig::create_for_categories(category_workers);

    EXPECT_EQ(config.size(), 5);
    EXPECT_TRUE(config.is_valid());

    // All workers should be unpinned with normal scheduling
    for (const auto &worker : config.workers) {
        EXPECT_FALSE(worker.is_pinned());
        EXPECT_FALSE(worker.has_thread_priority());
        EXPECT_EQ(worker.categories.size(),
                  1); // Each worker handles exactly one category
    }

    // Test category distribution (order-agnostic)
    std::map<ft::TaskCategory, std::size_t> actual_category_counts{};
    for (const auto &worker : config.workers) {
        ++actual_category_counts[worker.categories[0]];
    }

    std::map<ft::TaskCategory, std::size_t> expected_category_counts{};
    expected_category_counts[ft::TaskCategory{ft::BuiltinTaskCategory::HighPriority}] = 2;
    expected_category_counts[ft::TaskCategory{ft::BuiltinTaskCategory::IO}] = 1;
    expected_category_counts[ft::TaskCategory{ft::BuiltinTaskCategory::Compute}] = 2;

    EXPECT_EQ(actual_category_counts, expected_category_counts);
}

TEST(WorkersConfig, FactoryCreateForCategoriesSingleCategory) {
    ft::FlatMap<ft::TaskCategory, std::size_t> category_workers{};
    category_workers[ft::TaskCategory{ft::BuiltinTaskCategory::Compute}] = 4;

    const auto config = ft::WorkersConfig::create_for_categories(category_workers);

    EXPECT_EQ(config.size(), 4);
    EXPECT_TRUE(config.is_valid());

    // All workers should handle Compute category and be unpinned
    for (const auto &worker : config.workers) {
        EXPECT_FALSE(worker.is_pinned());
        EXPECT_FALSE(worker.has_thread_priority());
        EXPECT_EQ(worker.categories.size(), 1);
        EXPECT_EQ(worker.categories[0], ft::TaskCategory{ft::BuiltinTaskCategory::Compute});
    }
}

TEST(CoreAssignment, Construction) {
    // Test construction with just core
    const ft::CoreAssignment normal_core{5};
    EXPECT_EQ(normal_core.core_id, 5);
    EXPECT_FALSE(normal_core.thread_priority.has_value());

    // Test construction with core and priority
    ft::CoreAssignment rt_core{8, 80};
    EXPECT_EQ(rt_core.core_id, 8);
    ASSERT_TRUE(rt_core.thread_priority.has_value());
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    EXPECT_EQ(rt_core.thread_priority.value(), 80);
}

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

} // namespace
