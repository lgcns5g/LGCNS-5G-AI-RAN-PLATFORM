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
 * @file function_cache_tests.cpp
 * @brief Unit tests for FixedString and FunctionCache classes
 */

#include <array>
#include <cstring>
#include <format>
#include <string>
#include <tuple>

#include <gtest/gtest.h>

#include "task/function_cache.hpp"

namespace {
namespace ft = framework::task;

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

/**
 * Tests for FixedString template class
 */
TEST(FixedString, Construction) {
    // Default construction
    ft::FixedString<64> str1{};
    EXPECT_TRUE(str1.empty());
    EXPECT_EQ(str1.size(), 0);
    EXPECT_EQ(str1.capacity(), 64);
    EXPECT_STREQ(str1.c_str(), "");

    // Construction with string
    ft::FixedString<64> str2("hello");
    EXPECT_FALSE(str2.empty());
    EXPECT_EQ(str2.size(), 5);
    EXPECT_STREQ(str2.c_str(), "hello");

    // Construction with nullptr
    ft::FixedString<64> str3(nullptr);
    EXPECT_TRUE(str3.empty());
    EXPECT_EQ(str3.size(), 0);
    EXPECT_STREQ(str3.c_str(), "");
}

TEST(FixedString, Assignment) {
    ft::FixedString<32> str{};

    // Assignment from string
    str = "world";
    EXPECT_EQ(str.size(), 5);
    EXPECT_STREQ(str.c_str(), "world");

    // Assignment from nullptr
    str = nullptr;
    EXPECT_TRUE(str.empty());
    EXPECT_STREQ(str.c_str(), "");

    // Assignment from longer string
    str = "test_assignment";
    EXPECT_EQ(str.size(), 15);
    EXPECT_STREQ(str.c_str(), "test_assignment");
}

TEST(FixedString, Comparison) {
    const ft::FixedString<64> str1("hello");
    const ft::FixedString<64> str2("hello");
    const ft::FixedString<64> str3("world");

    // FixedString comparison
    EXPECT_TRUE(str1 == str2);
    EXPECT_FALSE(str1 == str3);

    // C-string comparison
    EXPECT_TRUE(str1 == "hello");
    EXPECT_FALSE(str1 == "world");

    // nullptr comparison
    const ft::FixedString<64> empty_str{};
    EXPECT_TRUE(empty_str == nullptr);
    EXPECT_TRUE(empty_str.empty());
}

TEST(FixedString, StringTruncation) {
    // Test string longer than capacity
    ft::FixedString<8> str("this_is_a_very_long_string");

    // Should be truncated to fit capacity - 1 (for null terminator)
    EXPECT_EQ(str.size(), 7); // 8 - 1 for null terminator
    EXPECT_EQ(std::strlen(str.c_str()), 7);

    // Should be null-terminated
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    EXPECT_EQ(str.c_str()[str.size()], '\0');

    // Should contain truncated content
    EXPECT_STREQ(str.c_str(), "this_is");
}

TEST(FixedString, DataAccess) {
    ft::FixedString<32> str("test");

    // Test const data access
    const auto &const_str = str;
    const auto &const_data = const_str.data();
    EXPECT_EQ(const_data.size(), 32);
    EXPECT_STREQ(const_data.data(), "test");

    // Test non-const data access
    auto &data = str.data();
    EXPECT_EQ(data.size(), 32);

    // Modify through direct access
    data[0] = 'b';
    EXPECT_STREQ(str.c_str(), "best");
}

TEST(FixedString, EdgeCases) {
    // Size 1 FixedString (only null terminator)
    ft::FixedString<1> tiny{};
    EXPECT_TRUE(tiny.empty());
    EXPECT_EQ(tiny.capacity(), 1);

    // Assign to size 1 FixedString
    tiny = "x";
    EXPECT_TRUE(tiny.empty()); // Should be truncated to empty
    EXPECT_STREQ(tiny.c_str(), "");

    // Large capacity
    ft::FixedString<1024> large("test");
    EXPECT_EQ(large.capacity(), 1024);
    EXPECT_EQ(large.size(), 4);
    EXPECT_STREQ(large.c_str(), "test");
}

/**
 * Tests for FunctionCache class
 */
TEST(FunctionCache, Construction) {
    // Default construction
    const ft::FunctionCache cache{};
    EXPECT_EQ(cache.size(), 0);
    EXPECT_EQ(cache.get_cache_hits(), 0);
    EXPECT_EQ(cache.get_cache_misses(), 0);
    EXPECT_EQ(cache.get_cache_attempts(), 0);
    EXPECT_EQ(cache.get_hit_ratio(), 0.0);

    // Custom construction
    const ft::FunctionCache custom_cache(100, 80, 30);
    EXPECT_EQ(custom_cache.size(), 0);
}

TEST(FunctionCache, BasicOperations) {
    ft::FunctionCache cache{};

    // Create some function pointers for testing
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast,performance-no-int-to-ptr)
    void *addr1 = reinterpret_cast<void *>(0x1000);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast,performance-no-int-to-ptr)
    void *addr2 = reinterpret_cast<void *>(0x2000);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast,performance-no-int-to-ptr)
    void *addr3 = reinterpret_cast<void *>(0x3000);

    // Initially empty
    EXPECT_EQ(cache.get(addr1), nullptr);
    EXPECT_EQ(cache.get_cache_misses(), 1);
    EXPECT_EQ(cache.get_cache_attempts(), 1);

    // Add entries
    cache.add(addr1, "function_one");
    cache.add(addr2, "function_two");
    cache.add(addr3, "function_three");

    EXPECT_EQ(cache.size(), 3);

    // Test retrieval
    const char *name1 = cache.get(addr1);
    ASSERT_NE(name1, nullptr);
    EXPECT_STREQ(name1, "function_one");
    EXPECT_EQ(cache.get_cache_hits(), 1);
    EXPECT_EQ(cache.get_cache_attempts(), 2); // 1 miss + 1 hit

    const char *name2 = cache.get(addr2);
    ASSERT_NE(name2, nullptr);
    EXPECT_STREQ(name2, "function_two");

    const char *name3 = cache.get(addr3);
    ASSERT_NE(name3, nullptr);
    EXPECT_STREQ(name3, "function_three");

    // Test non-existent address
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast,performance-no-int-to-ptr)
    void *addr4 = reinterpret_cast<void *>(0x4000);
    EXPECT_EQ(cache.get(addr4), nullptr);
}

TEST(FunctionCache, CacheStatistics) {
    ft::FunctionCache cache{};
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast,performance-no-int-to-ptr)
    void *addr1 = reinterpret_cast<void *>(0x1000);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast,performance-no-int-to-ptr)
    void *addr2 = reinterpret_cast<void *>(0x2000);

    // Add one entry
    cache.add(addr1, "test_function");

    // Multiple hits
    std::ignore = cache.get(addr1); // hit
    std::ignore = cache.get(addr1); // hit
    std::ignore = cache.get(addr1); // hit

    // Multiple misses
    std::ignore = cache.get(addr2); // miss
    std::ignore = cache.get(addr2); // miss

    EXPECT_EQ(cache.get_cache_hits(), 3);
    EXPECT_EQ(cache.get_cache_misses(), 2);
    EXPECT_EQ(cache.get_cache_attempts(), 5);

    // Test hit ratio calculation
    const double expected_ratio = (3.0 / 5.0) * 100.0; // 60%
    EXPECT_DOUBLE_EQ(cache.get_hit_ratio(), expected_ratio);
}

TEST(FunctionCache, CacheOverwriting) {
    ft::FunctionCache cache{};
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast,performance-no-int-to-ptr)
    void *addr = reinterpret_cast<void *>(0x1000);

    // Add initial entry
    cache.add(addr, "original_name");
    EXPECT_EQ(cache.size(), 1);

    const char *name1 = cache.get(addr);
    ASSERT_NE(name1, nullptr);
    EXPECT_STREQ(name1, "original_name");

    // Overwrite with new name
    cache.add(addr, "updated_name");
    EXPECT_EQ(cache.size(), 1); // Size should remain the same

    const char *name2 = cache.get(addr);
    ASSERT_NE(name2, nullptr);
    EXPECT_STREQ(name2, "updated_name");
}

TEST(FunctionCache, ClearCache) {
    ft::FunctionCache cache{};

    // Add multiple entries
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast,performance-no-int-to-ptr)
    void *addr1 = reinterpret_cast<void *>(0x1000);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast,performance-no-int-to-ptr)
    void *addr2 = reinterpret_cast<void *>(0x2000);
    cache.add(addr1, "function_one");
    cache.add(addr2, "function_two");

    // Generate some statistics
    std::ignore = cache.get(addr1);
    std::ignore = cache.get(addr2);

    EXPECT_EQ(cache.size(), 2);
    EXPECT_GT(cache.get_cache_attempts(), 0);

    // Clear cache
    cache.clear();

    EXPECT_EQ(cache.size(), 0);
    // Note: Statistics are typically not reset by clear() in caching systems
    // but this depends on implementation
}

TEST(FunctionCache, ManualEviction) {
    ft::FunctionCache cache{};

    // Add multiple entries
    for (int i = 0; i < 10; ++i) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast,performance-no-int-to-ptr)
        void *addr = reinterpret_cast<void *>(0x1000 + i * 0x100);
        cache.add(addr, std::format("function_{}", i).c_str());
    }

    EXPECT_EQ(cache.size(), 10);

    // Evict 40%
    cache.evict_percentage(40);

    // Should have fewer entries
    EXPECT_LT(cache.size(), 10);
    EXPECT_GT(cache.size(), 4); // At least some should remain
}

TEST(FunctionCache, LongFunctionNames) {
    ft::FunctionCache cache{};
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast,performance-no-int-to-ptr)
    void *addr = reinterpret_cast<void *>(0x1000);

    // Test with very long function name (longer than FixedString capacity)
    const std::string long_name(300, 'x'); // 300 'x' characters
    cache.add(addr, long_name.c_str());

    const char *retrieved = cache.get(addr);
    ASSERT_NE(retrieved, nullptr);

    // Should be truncated to fit FixedString capacity - 1
    EXPECT_LT(std::strlen(retrieved), 256); // NAME_SIZE in FunctionCache

    // Should start with the expected characters
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    EXPECT_EQ(retrieved[0], 'x');
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    EXPECT_EQ(retrieved[1], 'x');
}

TEST(FunctionCache, NullPointers) {
    ft::FunctionCache cache{};

    // Test with null address (edge case)
    void *null_addr = nullptr;
    cache.add(null_addr, "null_function");

    const char *name = cache.get(null_addr);
    ASSERT_NE(name, nullptr);
    EXPECT_STREQ(name, "null_function");

    // Test with null name
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast,performance-no-int-to-ptr)
    void *addr = reinterpret_cast<void *>(0x1000);
    cache.add(addr, nullptr);

    const char *retrieved = cache.get(addr);
    ASSERT_NE(retrieved, nullptr);
    EXPECT_STREQ(retrieved, ""); // Should be empty string
}

TEST(FunctionCache, HitRatioEdgeCases) {
    ft::FunctionCache cache{};

    // Test hit ratio with no attempts
    EXPECT_EQ(cache.get_hit_ratio(), 0.0);

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast,performance-no-int-to-ptr)
    void *addr = reinterpret_cast<void *>(0x1000);

    // Only misses
    std::ignore = cache.get(addr); // miss
    std::ignore = cache.get(addr); // miss
    EXPECT_EQ(cache.get_hit_ratio(), 0.0);

    // Add entry and test only hits
    cache.add(addr, "test");
    std::ignore = cache.get(addr); // hit
    std::ignore = cache.get(addr); // hit

    // Should be 50% (2 hits out of 4 attempts total)
    EXPECT_DOUBLE_EQ(cache.get_hit_ratio(), 50.0);
}

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

} // namespace
