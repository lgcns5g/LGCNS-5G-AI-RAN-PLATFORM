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
 * @file task_category_tests.cpp
 * @brief Unit tests for TaskCategory and DECLARE_TASK_CATEGORIES macro
 */

#include <compare>
#include <cstddef>
#include <cstdint>
#include <unordered_set>
#include <vector>

#include <wise_enum_detail.h>
#include <wise_enum_generated.h>

#include <gtest/gtest.h>

#include "task/task_category.hpp"

namespace {
namespace ft = framework::task;

// Define test categories for uniqueness verification
DECLARE_TASK_CATEGORIES(TestCategories1, DataProcessing, NetworkIO, Rendering);
DECLARE_TASK_CATEGORIES(TestCategories2, AudioProcessing, VideoProcessing, FileIO);
DECLARE_TASK_CATEGORIES(TestCategories3, DataProcessing,
                        NetworkIO); // Same names, different type

TEST(TaskCategory, BuiltinCategoryBasics) {
    const ft::TaskCategory default_cat{ft::BuiltinTaskCategory::Default};
    const ft::TaskCategory compute_cat{ft::BuiltinTaskCategory::Compute};

    EXPECT_NE(default_cat.id(), compute_cat.id());
    EXPECT_EQ(default_cat.name(), "Default");
    EXPECT_EQ(compute_cat.name(), "Compute");

    // Test equality
    const ft::TaskCategory another_default{ft::BuiltinTaskCategory::Default};
    EXPECT_EQ(default_cat, another_default);
    EXPECT_NE(default_cat, compute_cat);
}

/**
 * Test user-defined categories work correctly
 */
TEST(TaskCategory, UserDefinedCategories) {
    const ft::TaskCategory data_cat{TestCategories1::DataProcessing};
    const ft::TaskCategory net_cat{TestCategories1::NetworkIO};
    const ft::TaskCategory render_cat{TestCategories1::Rendering};

    EXPECT_NE(data_cat.id(), net_cat.id());
    EXPECT_NE(data_cat.id(), render_cat.id());
    EXPECT_NE(net_cat.id(), render_cat.id());

    EXPECT_EQ(data_cat.name(), "DataProcessing");
    EXPECT_EQ(net_cat.name(), "NetworkIO");
    EXPECT_EQ(render_cat.name(), "Rendering");
}

TEST(TaskCategory, TypeBasedUniqueness) {
    const ft::TaskCategory cat1_data{TestCategories1::DataProcessing};
    const ft::TaskCategory cat1_net{TestCategories1::NetworkIO};
    const ft::TaskCategory cat3_data{TestCategories3::DataProcessing};
    const ft::TaskCategory cat3_net{TestCategories3::NetworkIO};

    // Same names from different enum types should have different IDs
    EXPECT_NE(cat1_data.id(), cat3_data.id());
    EXPECT_NE(cat1_net.id(), cat3_net.id());

    // But same names should be preserved
    EXPECT_EQ(cat1_data.name(), cat3_data.name());
    EXPECT_EQ(cat1_net.name(), cat3_net.name());
}

TEST(TaskCategory, BuiltinUserCategoryUniqueness) {
    const ft::TaskCategory builtin_default{ft::BuiltinTaskCategory::Default};
    const ft::TaskCategory builtin_compute{ft::BuiltinTaskCategory::Compute};
    const ft::TaskCategory user_data{TestCategories1::DataProcessing};
    const ft::TaskCategory user_audio{TestCategories2::AudioProcessing};

    // All categories should have unique IDs
    std::unordered_set<std::uint64_t> ids;
    ids.insert(builtin_default.id());
    ids.insert(builtin_compute.id());
    ids.insert(user_data.id());
    ids.insert(user_audio.id());

    EXPECT_EQ(ids.size(), 4); // All IDs should be unique
}

TEST(TaskCategory, ComparisonOperators) {
    const ft::TaskCategory cat1{TestCategories1::DataProcessing};
    const ft::TaskCategory cat2{TestCategories1::NetworkIO};
    const ft::TaskCategory cat1_copy{TestCategories1::DataProcessing};

    // Test equality
    EXPECT_EQ(cat1, cat1_copy);
    EXPECT_NE(cat1, cat2);

    // Test inequality
    EXPECT_FALSE(cat1 != cat1_copy);
    EXPECT_TRUE(cat1 != cat2);

    // Test ordering (should be deterministic based on ID)
    const bool cat1_less_than_cat2 = cat1 < cat2;
    const bool cat2_less_than_cat1 = cat2 < cat1;
    EXPECT_NE(cat1_less_than_cat2,
              cat2_less_than_cat1); // Exactly one should be true
}

TEST(TaskCategory, STLContainerCompatibility) {
    // Test in unordered_set (requires hash function)
    std::unordered_set<ft::TaskCategory> category_set;
    category_set.insert(ft::TaskCategory{ft::BuiltinTaskCategory::Default});
    category_set.insert(ft::TaskCategory{ft::BuiltinTaskCategory::Compute});
    category_set.insert(ft::TaskCategory{TestCategories1::DataProcessing});
    category_set.insert(ft::TaskCategory{TestCategories1::DataProcessing}); // Duplicate

    EXPECT_EQ(category_set.size(), 3); // Duplicate should be ignored

    // Test in vector
    std::vector<ft::TaskCategory> category_vec;
    category_vec.emplace_back(ft::BuiltinTaskCategory::Network);
    category_vec.emplace_back(TestCategories2::AudioProcessing);
    category_vec.emplace_back(TestCategories2::VideoProcessing);

    EXPECT_EQ(category_vec.size(), 3);
    EXPECT_EQ(category_vec[0].name(), "Network");
    EXPECT_EQ(category_vec[1].name(), "AudioProcessing");
    EXPECT_EQ(category_vec[2].name(), "VideoProcessing");
}

TEST(TaskCategory, IDStability) {
    // Create categories multiple times and verify IDs are consistent
    const std::uint64_t id1 = ft::TaskCategory{TestCategories1::DataProcessing}.id();
    const std::uint64_t id2 = ft::TaskCategory{TestCategories1::DataProcessing}.id();
    const std::uint64_t id3 = ft::TaskCategory{TestCategories1::DataProcessing}.id();

    EXPECT_EQ(id1, id2);
    EXPECT_EQ(id2, id3);

    // Built-in categories should also be stable
    const std::uint64_t builtin_id1 = ft::TaskCategory{ft::BuiltinTaskCategory::Default}.id();
    const std::uint64_t builtin_id2 = ft::TaskCategory{ft::BuiltinTaskCategory::Default}.id();

    EXPECT_EQ(builtin_id1, builtin_id2);
}

TEST(TaskCategory, LargeScaleUniqueness) {
    // Create a large set of categories and verify no collisions
    std::unordered_set<std::uint64_t> all_ids;

    // Add all built-in categories
    all_ids.insert(ft::TaskCategory{ft::BuiltinTaskCategory::Default}.id());
    all_ids.insert(ft::TaskCategory{ft::BuiltinTaskCategory::HighPriority}.id());
    all_ids.insert(ft::TaskCategory{ft::BuiltinTaskCategory::LowPriority}.id());
    all_ids.insert(ft::TaskCategory{ft::BuiltinTaskCategory::IO}.id());
    all_ids.insert(ft::TaskCategory{ft::BuiltinTaskCategory::Compute}.id());
    all_ids.insert(ft::TaskCategory{ft::BuiltinTaskCategory::Network}.id());
    all_ids.insert(ft::TaskCategory{ft::BuiltinTaskCategory::Message}.id());
    const std::size_t builtin_count = all_ids.size();

    // Add user-defined categories from multiple enum types
    all_ids.insert(ft::TaskCategory{TestCategories1::DataProcessing}.id());
    all_ids.insert(ft::TaskCategory{TestCategories1::NetworkIO}.id());
    all_ids.insert(ft::TaskCategory{TestCategories1::Rendering}.id());

    all_ids.insert(ft::TaskCategory{TestCategories2::AudioProcessing}.id());
    all_ids.insert(ft::TaskCategory{TestCategories2::VideoProcessing}.id());
    all_ids.insert(ft::TaskCategory{TestCategories2::FileIO}.id());

    all_ids.insert(ft::TaskCategory{TestCategories3::DataProcessing}.id());
    all_ids.insert(ft::TaskCategory{TestCategories3::NetworkIO}.id());

    // Should have no collisions
    const std::size_t expected_total = builtin_count + 8; // 8 user-defined categories
    EXPECT_EQ(all_ids.size(), expected_total);
}

} // namespace
