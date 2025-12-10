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
 * @file flat_map_tests.cpp
 * @brief Unit tests for FlatMap class
 */

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

#include <parallel_hashmap/phmap.h>

#include <gtest/gtest.h>

#include "task/flat_map.hpp"

namespace {
namespace ft = framework::task;

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

/**
 * Basic functionality tests for FlatMap
 */
TEST(FlatMap, Construction) {
    // Default construction (uses Allocate strategy by default)
    const ft::FlatMap<int, std::string> map{};
    EXPECT_TRUE(map.empty());
    EXPECT_EQ(map.size(), 0);
    EXPECT_EQ(map.max_size(), 64); // Default max_size
    EXPECT_EQ(map.growth_strategy(), ft::GrowthStrategy::Allocate);

    // Custom construction with explicit strategy
    ft::FlatMap<int, std::string> custom_map(100, ft::GrowthStrategy::Evict);
    custom_map.set_eviction_percentages(80, 30); // Configure eviction percentages
    EXPECT_TRUE(custom_map.empty());
    EXPECT_EQ(custom_map.max_size(), 100);
    EXPECT_EQ(custom_map.growth_strategy(), ft::GrowthStrategy::Evict);

    // Throw strategy construction
    const ft::FlatMap<int, std::string> throw_map(50, ft::GrowthStrategy::Throw);
    EXPECT_EQ(throw_map.growth_strategy(), ft::GrowthStrategy::Throw);
}

TEST(FlatMap, BasicOperations) {
    ft::FlatMap<int, std::string> map{};

    // Test operator[]
    map[1] = "one";
    map[2] = "two";
    map[3] = "three";

    EXPECT_EQ(map.size(), 3);
    EXPECT_FALSE(map.empty());
    EXPECT_EQ(map[1], "one");
    EXPECT_EQ(map[2], "two");
    EXPECT_EQ(map[3], "three");

    // Test find
    auto it = map.find(2);
    EXPECT_NE(it, map.end());
    EXPECT_EQ(it->second, "two");

    auto not_found = map.find(999);
    EXPECT_EQ(not_found, map.end());
}

TEST(FlatMap, InsertAndEmplace) {
    ft::FlatMap<int, std::string> map{};

    // Test insert
    auto result1 = map.insert({10, "ten"});
    EXPECT_TRUE(result1.second); // Insertion successful
    EXPECT_EQ(result1.first->second, "ten");

    // Test duplicate insert
    auto result2 = map.insert({10, "TEN"});
    EXPECT_FALSE(result2.second);            // Should not insert duplicate
    EXPECT_EQ(result2.first->second, "ten"); // Original value unchanged

    // Test emplace
    auto result3 = map.emplace(20, "twenty");
    EXPECT_TRUE(result3.second);
    EXPECT_EQ(result3.first->second, "twenty");

    EXPECT_EQ(map.size(), 2);
}

TEST(FlatMap, EraseAndClear) {
    ft::FlatMap<int, std::string> map{};

    // Add some entries
    map[1] = "one";
    map[2] = "two";
    map[3] = "three";
    EXPECT_EQ(map.size(), 3);

    // Test erase
    map.erase(2);
    EXPECT_EQ(map.size(), 2);
    EXPECT_EQ(map.find(2), map.end());
    EXPECT_NE(map.find(1), map.end());
    EXPECT_NE(map.find(3), map.end());

    // Test clear
    map.clear();
    EXPECT_TRUE(map.empty());
    EXPECT_EQ(map.size(), 0);
}

TEST(FlatMap, Iteration) {
    ft::FlatMap<int, std::string> map{};

    // Add entries
    map[1] = "one";
    map[2] = "two";
    map[3] = "three";

    // Test iteration
    std::size_t count = 0;
    for (const auto &pair : map) {
        EXPECT_GE(pair.first, 1);
        EXPECT_LE(pair.first, 3);
        EXPECT_FALSE(pair.second.empty());
        ++count;
    }
    EXPECT_EQ(count, 3);

    // Test const iteration
    const auto &const_map = map;
    count = 0;
    for (auto it = const_map.begin(); it != const_map.end(); ++it) {
        ++count;
    }
    EXPECT_EQ(count, 3);
}

/**
 * Eviction functionality tests
 */
TEST(FlatMap, AutomaticEviction) {
    // Create small map with Evict strategy (threshold = 90% of 10 = 9)
    ft::FlatMap<int, int> map(10, ft::GrowthStrategy::Evict);
    map.set_eviction_percentages(90, 50);

    // Fill to exactly the threshold (9 entries)
    for (int i = 0; i < 9; ++i) {
        map[i] = i * 10;
    }
    EXPECT_EQ(map.size(), 9);

    // Adding 10th entry should trigger eviction (9 >= threshold of 9)
    map[100] = 1000;

    // Should have evicted exactly 50% of 9 entries (4 entries), leaving 5 + 1 new
    // = 6
    EXPECT_EQ(map.size(), 6); // Verify exact expected result: 9 - 4 + 1 = 6

    // Verify the new entry was actually added
    EXPECT_EQ(map[100], 1000);
}

TEST(FlatMap, ManualEviction) {
    ft::FlatMap<int, std::string> map{};

    // Add 10 entries
    for (int i = 0; i < 10; ++i) {
        map[i] = std::to_string(i);
    }
    EXPECT_EQ(map.size(), 10);

    // Evict 30%
    map.evict_percentage(30);

    // Should have removed ~3 entries (30% of 10)
    EXPECT_LT(map.size(), 10);
    EXPECT_GT(map.size(), 5); // At least 5 should remain
}

TEST(FlatMap, EvictionEdgeCases) {
    ft::FlatMap<int, int> map{};

    // Test eviction on empty map (should not crash)
    map.evict_percentage(50);
    EXPECT_TRUE(map.empty());

    // Add one entry and evict 100%
    map[1] = 100;
    map.evict_percentage(100);
    EXPECT_TRUE(map.empty());

    // Add entries and evict very small percentage
    for (int i = 0; i < 100; ++i) {
        map[i] = i;
    }
    const std::size_t initial_size = map.size();
    map.evict_percentage(1); // 1% of 100 = 1 entry
    EXPECT_LT(map.size(), initial_size);
}

TEST(FlatMap, EvictionDivisionByZeroProtection) {
    ft::FlatMap<int, int> map{};

    // Test case 1: Small map with percentage that results in target_removals = 0
    // With 1 entry and 50% eviction: target_removals = (1 * 50) / 100 = 0
    map[1] = 100;
    EXPECT_NO_THROW(map.evict_percentage(50));
    EXPECT_EQ(map.size(), 1); // Should remain unchanged due to early return

    // Test case 2: Small map with very small percentage
    // With 2 entries and 1% eviction: target_removals = (2 * 1) / 100 = 0
    map[2] = 200;
    EXPECT_EQ(map.size(), 2);
    EXPECT_NO_THROW(map.evict_percentage(1));
    EXPECT_EQ(map.size(), 2); // Should remain unchanged due to early return

    // Test case 3: Edge case with percentage resulting in zero removals
    // With 5 entries and 10% eviction: target_removals = (5 * 10) / 100 = 0
    map[3] = 300;
    map[4] = 400;
    map[5] = 500;
    EXPECT_EQ(map.size(), 5);
    EXPECT_NO_THROW(map.evict_percentage(10));
    EXPECT_EQ(map.size(), 5); // Should remain unchanged due to early return
}

/**
 * Configuration and validation tests
 */
TEST(FlatMap, ParameterValidation) {
    // Test invalid constructor parameters
    EXPECT_THROW((ft::FlatMap<int, int>(0)),
                 std::invalid_argument); // max_size = 0

    // Test invalid evict_percentage values (through manual eviction method)
    ft::FlatMap<int, int> map{};
    EXPECT_THROW(map.evict_percentage(0), std::invalid_argument);
    EXPECT_THROW(map.evict_percentage(101), std::invalid_argument);

    // Test invalid eviction percentages
    EXPECT_THROW(map.set_eviction_percentages(0, 50), std::invalid_argument);
    EXPECT_THROW(map.set_eviction_percentages(50, 0), std::invalid_argument);
    EXPECT_THROW(map.set_eviction_percentages(101, 50), std::invalid_argument);
    EXPECT_THROW(map.set_eviction_percentages(50, 101), std::invalid_argument);

    // Valid parameters should not throw
    EXPECT_NO_THROW((ft::FlatMap<int, int>(1000)));
    EXPECT_NO_THROW((ft::FlatMap<int, int>(1000, ft::GrowthStrategy::Allocate)));
    EXPECT_NO_THROW((ft::FlatMap<int, int>(1000, ft::GrowthStrategy::Evict)));
    EXPECT_NO_THROW((ft::FlatMap<int, int>(1000, ft::GrowthStrategy::Throw)));
}

TEST(FlatMap, ConfigurationMethods) {
    ft::FlatMap<int, int> map(100);

    // Test set_max_size
    map.set_max_size(200);
    EXPECT_EQ(map.max_size(), 200);

    // Test set_eviction_percentages
    map.set_eviction_percentages(80, 30);

    // Test invalid parameters
    EXPECT_THROW(map.set_max_size(0), std::invalid_argument);
}

TEST(FlatMap, UnderlyingAccess) {
    ft::FlatMap<int, std::string> map{};

    map[1] = "one";
    map[2] = "two";

    // Test const underlying access
    const auto &const_map = map;
    const auto &underlying_const = const_map.underlying();
    EXPECT_EQ(underlying_const.size(), 2);

    // Test non-const underlying access
    auto &underlying = map.underlying();
    EXPECT_EQ(underlying.size(), 2);

    // Modify through underlying reference
    underlying[3] = "three";
    EXPECT_EQ(map.size(), 3);
    EXPECT_EQ(map[3], "three");
}

/**
 * Growth strategy specific tests
 */
TEST(FlatMap, AllocateStrategy) {
    // Allocate strategy allows unlimited growth
    ft::FlatMap<int, int> map(5, ft::GrowthStrategy::Allocate);

    // Fill beyond max_size - should not trigger any action
    for (int i = 0; i < 10; ++i) {
        map[i] = i * 10;
    }

    // Should have all 10 entries (no eviction, no exception)
    EXPECT_EQ(map.size(), 10);
    EXPECT_GT(map.size(), map.max_size()); // Exceeded max_size but allowed
}

TEST(FlatMap, EvictStrategy) {
    // Evict strategy removes entries at threshold
    ft::FlatMap<int, int> map(10, ft::GrowthStrategy::Evict);
    map.set_eviction_percentages(70, 40); // threshold = 7

    // Fill to just below threshold
    for (int i = 0; i < 7; ++i) {
        map[i] = i;
    }
    EXPECT_EQ(map.size(), 7);

    // Adding 8th entry should trigger eviction (7 >= 7)
    map[100] = 100;

    // Should have evicted ~40% of 7 entries (~2-3 entries)
    EXPECT_LT(map.size(), 7); // Some entries were evicted
    EXPECT_GT(map.size(), 3); // But not too many
}

TEST(FlatMap, EvictionThresholdRecalculationOnMaxSizeChange) {
    // Test that eviction threshold is correctly recalculated when max_size
    // changes
    ft::FlatMap<int, int> map(100, ft::GrowthStrategy::Evict);
    map.set_eviction_percentages(80, 30); // threshold = 80% of 100 = 80

    // Fill to just below threshold (80 entries)
    for (int i = 0; i < 79; ++i) {
        map[i] = i;
    }
    EXPECT_EQ(map.size(), 79);

    // Adding one more should NOT trigger eviction (79 < 80)
    map[100] = 100;
    EXPECT_EQ(map.size(), 80); // No eviction should occur

    // Change max_size to 50 - eviction threshold should recalculate to 80% of 50
    // = 40
    map.set_max_size(50);
    EXPECT_EQ(map.max_size(), 50);

    // Current size is 80, which is way above new threshold of 40
    // Adding another entry should trigger eviction
    const std::size_t size_before_insertion = map.size();
    map[200] = 200;

    // Should have triggered eviction since 80 >= 40 (new threshold)
    // After evicting 30% of ~80 entries (~24 entries), plus adding 1 new = ~57
    // entries
    EXPECT_LT(map.size(), size_before_insertion); // Some eviction occurred
    EXPECT_GT(map.size(), 40);                    // But not too much eviction
}

TEST(FlatMap, EvictionThresholdRecalculationPreservesPercentage) {
    // Test that the percentage relationship is preserved when changing max_size
    ft::FlatMap<int, int> map(20, ft::GrowthStrategy::Evict);
    map.set_eviction_percentages(75, 50); // threshold = 75% of 20 = 15

    // Fill to exactly threshold
    for (int i = 0; i < 15; ++i) {
        map[i] = i;
    }
    EXPECT_EQ(map.size(), 15);

    // Adding one more should trigger eviction (15 >= 15)
    map[100] = 100;
    const std::size_t size_after_first_eviction = map.size();
    EXPECT_LT(size_after_first_eviction, 15); // Eviction occurred

    // Clear and refill for clean test
    map.clear();
    for (int i = 0; i < 14; ++i) {
        map[i] = i;
    }

    // Change max_size to 40 - threshold should be 75% of 40 = 30
    map.set_max_size(40);
    EXPECT_EQ(map.max_size(), 40);

    // Fill to just below new threshold (30 entries)
    for (int i = 14; i < 29; ++i) {
        map[i] = i;
    }
    EXPECT_EQ(map.size(), 29);

    // Adding one more should NOT trigger eviction (29 < 30)
    map[200] = 200;
    EXPECT_EQ(map.size(), 30); // No eviction should occur

    // Adding one more should trigger eviction (30 >= 30)
    map[300] = 300;
    EXPECT_LT(map.size(), 30); // Eviction should occur
}

TEST(FlatMap, EvictionThresholdRecalculationEdgeCases) {
    // Test edge cases for eviction threshold recalculation

    // Case 1: Very small max_size
    ft::FlatMap<int, int> small_map(2, ft::GrowthStrategy::Evict);
    small_map.set_eviction_percentages(50, 50); // threshold = 1
    small_map[1] = 1;
    EXPECT_EQ(small_map.size(), 1);

    // Change to even smaller max_size
    small_map.set_max_size(1);
    EXPECT_EQ(small_map.max_size(), 1);
    // threshold should now be 50% of 1 = 0, so any addition should trigger
    // eviction

    // Case 2: Increase max_size significantly
    ft::FlatMap<int, int> growing_map(10, ft::GrowthStrategy::Evict);
    growing_map.set_eviction_percentages(90, 20); // threshold = 9

    for (int i = 0; i < 8; ++i) {
        growing_map[i] = i;
    }
    EXPECT_EQ(growing_map.size(), 8);

    // Increase max_size significantly - threshold should be 90% of 100 = 90
    growing_map.set_max_size(100);
    EXPECT_EQ(growing_map.max_size(), 100);

    // Should be able to add many more entries without eviction
    for (int i = 8; i < 89; ++i) {
        growing_map[i] = i;
    }
    EXPECT_EQ(growing_map.size(), 89);

    // Adding one more should NOT trigger eviction (89 < 90)
    growing_map[200] = 200;
    EXPECT_EQ(growing_map.size(), 90);

    // Adding one more should trigger eviction (90 >= 90)
    growing_map[300] = 300;
    EXPECT_LT(growing_map.size(), 90);
}

TEST(FlatMap, ThrowStrategy) {
    // Throw strategy throws when max_size is reached
    ft::FlatMap<int, int> map(5, ft::GrowthStrategy::Throw);

    // Fill to max_size
    for (int i = 0; i < 5; ++i) {
        map[i] = i;
    }
    EXPECT_EQ(map.size(), 5);

    // Test throwing with different insertion methods
    EXPECT_THROW(map[100] = 100, std::length_error);
    EXPECT_THROW(map.insert({101, 101}), std::length_error);
    EXPECT_THROW(map.emplace(102, 102), std::length_error);

    // Size should remain at max_size after all failed attempts
    EXPECT_EQ(map.size(), 5);
}

TEST(FlatMap, GrowthStrategyConfiguration) {
    ft::FlatMap<int, int> map{};

    // Test setting growth strategy
    map.set_growth_strategy(ft::GrowthStrategy::Evict);
    EXPECT_EQ(map.growth_strategy(), ft::GrowthStrategy::Evict);

    map.set_growth_strategy(ft::GrowthStrategy::Throw);
    EXPECT_EQ(map.growth_strategy(), ft::GrowthStrategy::Throw);

    map.set_growth_strategy(ft::GrowthStrategy::Allocate);
    EXPECT_EQ(map.growth_strategy(), ft::GrowthStrategy::Allocate);
}

TEST(FlatMap, GrowthStrategyValidation) {
    ft::FlatMap<int, int> map(100);

    // Start with default valid percentages, then modify to invalid values
    // by setting eviction_percentage_ to an invalid value through internal state
    // Since we can't directly set invalid values (they're validated),
    // we test the validation logic by checking the constructor behavior

    // Test that switching strategies works with valid parameters
    map.set_eviction_percentages(80, 25);
    EXPECT_NO_THROW(map.set_growth_strategy(ft::GrowthStrategy::Evict));
    EXPECT_NO_THROW(map.set_growth_strategy(ft::GrowthStrategy::Throw));
    EXPECT_NO_THROW(map.set_growth_strategy(ft::GrowthStrategy::Allocate));
}

TEST(FlatMap, CapacityAndDefaultValues) {
    ft::FlatMap<int, int> map(1000);

    // Test capacity method
    EXPECT_GE(map.capacity(), 0);

    // Test that default eviction percentages (90%, 25%) are used
    map.set_growth_strategy(ft::GrowthStrategy::Evict);

    // Fill to 90% threshold (900 entries)
    for (int i = 0; i < 900; ++i) {
        map[i] = i;
    }
    EXPECT_EQ(map.size(), 900);

    // Adding 901st entry should trigger eviction with default 25%
    map[2000] = 2000;

    // Should have evicted ~25% of 900 entries (~225), leaving ~675 + 1 new = ~676
    EXPECT_LT(map.size(), 900); // Some entries were evicted
    EXPECT_GT(map.size(), 650); // But not too many (rough bounds)
}

TEST(FlatMap, InsertEmplaceWithStrategies) {
    // Test that insert() and emplace() also trigger capacity handling

    // Test with Evict strategy
    ft::FlatMap<int, int> evict_map(5, ft::GrowthStrategy::Evict);
    evict_map.set_eviction_percentages(80, 50); // threshold = 4

    // Fill to threshold using different methods
    evict_map.insert({1, 1});
    evict_map.emplace(2, 2);
    evict_map[3] = 3;
    evict_map.insert({4, 4});
    EXPECT_EQ(evict_map.size(), 4);

    // Next insertion should trigger eviction
    evict_map.emplace(100, 100);
    EXPECT_LT(evict_map.size(), 4); // Eviction occurred
}

TEST(FlatMap, EvictionBehaviorVerification) {
    // More precise testing of eviction behavior
    ft::FlatMap<int, int> map(100, ft::GrowthStrategy::Evict);
    map.set_eviction_percentages(50, 20); // threshold = 50, evict 20%

    // Fill exactly to threshold
    for (int i = 0; i < 50; ++i) {
        map[i] = i;
    }
    EXPECT_EQ(map.size(), 50);

    // Trigger eviction
    map[999] = 999;

    // Should evict 20% of 50 = 10 entries, leaving 40 + 1 new = 41
    const std::size_t expected_remaining = 50 - (50 * 20 / 100) + 1; // 41
    const std::size_t actual_size = map.size();

    // Allow some variance due to eviction algorithm implementation
    EXPECT_GE(actual_size, expected_remaining - 2);
    EXPECT_LE(actual_size, expected_remaining + 2);

    // Verify the new entry was added
    EXPECT_NE(map.find(999), map.end());
}

TEST(FlatMap, EvictionPercentagesConfiguration) {
    ft::FlatMap<int, int> map(100);

    // Test setting both eviction percentages
    map.set_eviction_percentages(80, 30);

    // Verify the configuration took effect by switching to Evict strategy and
    // testing behavior
    map.set_growth_strategy(ft::GrowthStrategy::Evict);

    // Fill to threshold (80% of 100 = 80)
    for (int i = 0; i < 80; ++i) {
        map[i] = i;
    }
    EXPECT_EQ(map.size(), 80);

    // Adding 81st entry should trigger 30% eviction
    map[200] = 200;

    // Should have evicted ~30% of 80 entries (~24 entries), leaving ~56 + 1 new =
    // ~57
    EXPECT_LT(map.size(), 80); // Some entries were evicted
    EXPECT_GT(map.size(), 50); // But not too many
}

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

} // namespace
