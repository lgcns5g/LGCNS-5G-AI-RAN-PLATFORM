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

#include <array>    // for array
#include <cstddef>  // for size_t
#include <iterator> // for next
#include <memory>   // for allocator
#include <numbers>  // for e

#include <gtest/gtest.h> // for Test, Message, TestPartResult, AssertionResult

#include "utils/arr.hpp" // for Arr, operator==, operator!=

namespace {

// Basic Construction Tests

// Test: Default constructor zero-initializes all elements
TEST(ArrTest, DefaultConstruction) {
    const framework::utils::Arr<int, 3> arr{};

    // All elements should be zero-initialized
    EXPECT_EQ(0, arr[0]);
    EXPECT_EQ(0, arr[1]);
    EXPECT_EQ(0, arr[2]);
}

// Test: Default constructor works with different numeric types
TEST(ArrTest, DefaultConstructionDifferentTypes) {
    const framework::utils::Arr<float, 2> float_arr{};
    const framework::utils::Arr<double, 4> double_arr{};
    const framework::utils::Arr<bool, 1> bool_arr{};

    EXPECT_FLOAT_EQ(0.0F, float_arr[0]);
    EXPECT_FLOAT_EQ(0.0F, float_arr[1]);

    EXPECT_DOUBLE_EQ(0.0, double_arr[0]);
    EXPECT_DOUBLE_EQ(0.0, double_arr[1]);
    EXPECT_DOUBLE_EQ(0.0, double_arr[2]);
    EXPECT_DOUBLE_EQ(0.0, double_arr[3]);

    EXPECT_FALSE(bool_arr[0]);
}

// Array Constructor Tests

// Test: C-array constructor properly copies integer values (smoke test)
TEST(ArrTest, CArrayConstructorInt) {
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays)
    const int arr[3] = {1, 2, 3};
    const framework::utils::Arr<int, 3> vec(arr);

    EXPECT_EQ(1, vec[0]);
    EXPECT_EQ(2, vec[1]);
    EXPECT_EQ(3, vec[2]);
}

// Test: std::array constructor properly copies integer values
TEST(ArrTest, StdArrayConstructorInt) {
    const std::array<int, 3> arr = {1, 2, 3};
    const framework::utils::Arr<int, 3> vec(arr);

    EXPECT_EQ(1, vec[0]);
    EXPECT_EQ(2, vec[1]);
    EXPECT_EQ(3, vec[2]);
}

// Test: std::array constructor properly copies floating-point values
TEST(ArrTest, StdArrayConstructorFloat) {
    const std::array<float, 4> arr = {1.5F, 2.5F, 3.5F, 4.5F};
    const framework::utils::Arr<float, 4> vec(arr);

    EXPECT_FLOAT_EQ(1.5F, vec[0]);
    EXPECT_FLOAT_EQ(2.5F, vec[1]);
    EXPECT_FLOAT_EQ(3.5F, vec[2]);
    EXPECT_FLOAT_EQ(4.5F, vec[3]);
}

// Test: std::array constructor works with single-element arrays
TEST(ArrTest, StdArrayConstructorSingleElement) {
    const std::array<int, 1> arr = {42};
    const framework::utils::Arr<int, 1> vec(arr);

    EXPECT_EQ(42, vec[0]);
}

// Test: std::array constructor dimension matching works correctly
TEST(ArrTest, StdArrayConstructorDimensionMatching) {
    // This should compile fine
    const std::array<int, 3> arr = {1, 2, 3};
    const framework::utils::Arr<int, 3> vec(arr);

    EXPECT_EQ(1, vec[0]);
    EXPECT_EQ(2, vec[1]);
    EXPECT_EQ(3, vec[2]);

    // Note: We can't test static_assert failure at runtime,
    // but compilation would fail if we tried: framework::utils::Arr<int, 2> vec2(arr);
}

// Fill Method Tests

// Test: Fill method sets all elements to the same value
TEST(ArrTest, FillMethod) {
    framework::utils::Arr<int, 3> arr{};
    static constexpr int FILL_VALUE = 42;
    arr.fill(FILL_VALUE);

    EXPECT_EQ(FILL_VALUE, arr[0]);
    EXPECT_EQ(FILL_VALUE, arr[1]);
    EXPECT_EQ(FILL_VALUE, arr[2]);
}

// Test: Fill method overwrites existing values
TEST(ArrTest, FillMethodOverwrite) {
    const std::array<int, 2> arr = {1, 2};
    framework::utils::Arr<int, 2> vec(arr);

    // Verify initial values
    EXPECT_EQ(1, vec[0]);
    EXPECT_EQ(2, vec[1]);

    // Fill with new value
    static constexpr int NEW_VALUE = 99;
    vec.fill(NEW_VALUE);

    EXPECT_EQ(NEW_VALUE, vec[0]);
    EXPECT_EQ(NEW_VALUE, vec[1]);
}

// Test: Fill method works with different numeric types
TEST(ArrTest, FillMethodDifferentTypes) {
    framework::utils::Arr<float, 2> float_arr{};
    framework::utils::Arr<double, 3> double_arr{};

    static constexpr float FLOAT_VALUE = 3.14F;
    float_arr.fill(FLOAT_VALUE);
    double_arr.fill(std::numbers::e);

    EXPECT_FLOAT_EQ(FLOAT_VALUE, float_arr[0]);
    EXPECT_FLOAT_EQ(FLOAT_VALUE, float_arr[1]);

    EXPECT_DOUBLE_EQ(std::numbers::e, double_arr[0]);
    EXPECT_DOUBLE_EQ(std::numbers::e, double_arr[1]);
    EXPECT_DOUBLE_EQ(std::numbers::e, double_arr[2]);
}

// Index Operator Tests

// Test: Index operator provides read access to elements
TEST(ArrTest, IndexOperatorRead) {
    const std::array<int, 4> arr = {10, 20, 30, 40};
    const framework::utils::Arr<int, 4> vec(arr);

    EXPECT_EQ(10, vec[0]);
    EXPECT_EQ(20, vec[1]);
    EXPECT_EQ(30, vec[2]);
    EXPECT_EQ(40, vec[3]);
}

// Test: Index operator provides write access to elements
TEST(ArrTest, IndexOperatorWrite) {
    framework::utils::Arr<int, 3> arr{};
    static constexpr int VALUE_1 = 100;
    static constexpr int VALUE_2 = 200;
    static constexpr int VALUE_3 = 300;

    arr[0] = VALUE_1;
    arr[1] = VALUE_2;
    arr[2] = VALUE_3;

    EXPECT_EQ(VALUE_1, arr[0]);
    EXPECT_EQ(VALUE_2, arr[1]);
    EXPECT_EQ(VALUE_3, arr[2]);
}

// Test: Index operator allows element modification
TEST(ArrTest, IndexOperatorModify) {
    const std::array<int, 2> arr = {5, 10};
    framework::utils::Arr<int, 2> vec(arr);

    static constexpr int MULTIPLIER = 2;
    static constexpr int ADDITION = 5;
    vec[0] *= MULTIPLIER;
    vec[1] += ADDITION;

    EXPECT_EQ(MULTIPLIER * 5, vec[0]);
    EXPECT_EQ(ADDITION + 10, vec[1]);
}

// Test: Const index operator provides read-only access
TEST(ArrTest, ConstIndexOperator) {
    const std::array<int, 3> arr = {1, 2, 3};
    const framework::utils::Arr<int, 3> vec(arr);

    // Test const version of operator[]
    EXPECT_EQ(1, vec[0]);
    EXPECT_EQ(2, vec[1]);
    EXPECT_EQ(3, vec[2]);
}

// Iterator Tests

// Test: Begin and end iterators work correctly
TEST(ArrTest, BeginEndIterators) {
    const std::array<int, 3> arr = {10, 20, 30};
    framework::utils::Arr<int, 3> vec(arr);

    const auto *begin_it = vec.begin();
    const auto *end_it = vec.end();

    EXPECT_NE(begin_it, end_it);
    EXPECT_EQ(10, *begin_it);
    EXPECT_EQ(20, *(std::next(begin_it, 1)));
    EXPECT_EQ(30, *(std::next(begin_it, 2)));
    EXPECT_EQ(std::next(begin_it, 3), end_it);
}

// Test: Const begin and end iterators work correctly
TEST(ArrTest, ConstBeginEndIterators) {
    const std::array<int, 2> arr = {100, 200};
    const framework::utils::Arr<int, 2> vec(arr);

    const auto *begin_it = vec.begin();
    const auto *end_it = vec.end();

    EXPECT_NE(begin_it, end_it);
    EXPECT_EQ(100, *begin_it);
    EXPECT_EQ(200, *(std::next(begin_it, 1)));
    EXPECT_EQ(std::next(begin_it, 2), end_it);
}

// Test: Range-based for loop works correctly
TEST(ArrTest, RangeBasedLoop) {
    const std::array<int, 4> arr = {1, 2, 3, 4};
    const framework::utils::Arr<int, 4> vec(arr);

    int expected = 1;
    for (const auto &element : vec) {
        EXPECT_EQ(expected, element);
        ++expected;
    }
    EXPECT_EQ(5, expected); // Should have incremented 4 times
}

// Test: Modification through iterators works correctly
TEST(ArrTest, ModifyThroughIterators) {
    framework::utils::Arr<int, 3> arr{};
    static constexpr int INITIAL_VALUE = 1;
    arr.fill(INITIAL_VALUE);

    // Modify through iterators
    static constexpr int MULTIPLIER = 10;
    for (auto &element : arr) {
        element *= MULTIPLIER;
    }

    EXPECT_EQ(MULTIPLIER * INITIAL_VALUE, arr[0]);
    EXPECT_EQ(MULTIPLIER * INITIAL_VALUE, arr[1]);
    EXPECT_EQ(MULTIPLIER * INITIAL_VALUE, arr[2]);
}

// Equality Operator Tests

// Test: Equality operator returns true for identical arrays
TEST(ArrTest, EqualityOperatorTrue) {
    const std::array<int, 3> arr1 = {1, 2, 3};
    const std::array<int, 3> arr2 = {1, 2, 3};

    const framework::utils::Arr<int, 3> vec1(arr1);
    const framework::utils::Arr<int, 3> vec2(arr2);

    EXPECT_TRUE(vec1 == vec2);
    EXPECT_TRUE(vec2 == vec1);
}

// Test: Equality operator returns false for different arrays
TEST(ArrTest, EqualityOperatorFalse) {
    const std::array<int, 3> arr1 = {1, 2, 3};
    const std::array<int, 3> arr2 = {1, 2, 4}; // Different last element

    const framework::utils::Arr<int, 3> vec1(arr1);
    const framework::utils::Arr<int, 3> vec2(arr2);

    EXPECT_FALSE(vec1 == vec2);
    EXPECT_FALSE(vec2 == vec1);
}

// Test: Equality operator works with arrays from same array
TEST(ArrTest, EqualityOperatorIdentical) {
    const std::array<int, 2> arr = {42, 84};
    const framework::utils::Arr<int, 2> vec1(arr);
    const framework::utils::Arr<int, 2> vec2(arr);

    EXPECT_TRUE(vec1 == vec2);
}

// Test: Equality operator works for self-comparison
TEST(ArrTest, EqualityOperatorSelfComparison) {
    const std::array<int, 4> arr = {10, 20, 30, 40};
    const framework::utils::Arr<int, 4> vec(arr);

    EXPECT_TRUE(vec == vec);
}

// Test: Equality operator works with different numeric types
TEST(ArrTest, EqualityOperatorDifferentTypes) {
    const std::array<float, 2> arr1 = {1.0F, 2.0F};
    const std::array<float, 2> arr2 = {1.0F, 2.0F};

    const framework::utils::Arr<float, 2> vec1(arr1);
    const framework::utils::Arr<float, 2> vec2(arr2);

    EXPECT_TRUE(vec1 == vec2);
}

// Inequality Operator Tests

// Test: Inequality operator returns true for different arrays
TEST(ArrTest, InequalityOperatorTrue) {
    const std::array<int, 3> arr1 = {1, 2, 3};
    const std::array<int, 3> arr2 = {1, 2, 4};

    const framework::utils::Arr<int, 3> vec1(arr1);
    const framework::utils::Arr<int, 3> vec2(arr2);

    EXPECT_TRUE(vec1 != vec2);
    EXPECT_TRUE(vec2 != vec1);
}

// Test: Inequality operator returns false for identical arrays
TEST(ArrTest, InequalityOperatorFalse) {
    const std::array<int, 2> arr1 = {10, 20};
    const std::array<int, 2> arr2 = {10, 20};

    const framework::utils::Arr<int, 2> vec1(arr1);
    const framework::utils::Arr<int, 2> vec2(arr2);

    EXPECT_FALSE(vec1 != vec2);
    EXPECT_FALSE(vec2 != vec1);
}

// Test: Inequality operator works for self-comparison
TEST(ArrTest, InequalityOperatorSelfComparison) {
    const std::array<int, 3> arr = {5, 10, 15};
    const framework::utils::Arr<int, 3> vec(arr);

    EXPECT_FALSE(vec != vec);
}

// Consistency Between Equality and Inequality

// Test: Equality and inequality operators are consistent
TEST(ArrTest, EqualityInequalityConsistency) {
    const std::array<int, 4> arr1 = {1, 2, 3, 4};
    const std::array<int, 4> arr2 = {1, 2, 3, 5}; // Different last element
    const std::array<int, 4> arr3 = {1, 2, 3, 4}; // Same as arr1

    const framework::utils::Arr<int, 4> vec1(arr1);
    const framework::utils::Arr<int, 4> vec2(arr2);
    const framework::utils::Arr<int, 4> vec3(arr3);

    // vec1 and vec2 should be different
    EXPECT_TRUE((vec1 == vec2) != (vec1 != vec2));
    EXPECT_FALSE(vec1 == vec2);
    EXPECT_TRUE(vec1 != vec2);

    // vec1 and vec3 should be equal
    EXPECT_TRUE((vec1 == vec3) != (vec1 != vec3));
    EXPECT_TRUE(vec1 == vec3);
    EXPECT_FALSE(vec1 != vec3);
}

// Edge Cases and Special Scenarios

// Test: Single element array works correctly (C-array smoke test)
TEST(ArrTest, SingleElementArrayCArray) {
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays)
    const int arr[1] = {42};
    framework::utils::Arr<int, 1> vec(arr);

    static constexpr int INITIAL_VALUE = 42;
    EXPECT_EQ(INITIAL_VALUE, vec[0]);

    static constexpr int FILL_VALUE = 99;
    vec.fill(FILL_VALUE);
    EXPECT_EQ(FILL_VALUE, vec[0]);

    framework::utils::Arr<int, 1> other_vec{};
    other_vec[0] = INITIAL_VALUE;

    EXPECT_FALSE(vec == other_vec);
    EXPECT_TRUE(vec != other_vec);
}

// Test: Large array works correctly
TEST(ArrTest, LargeArray) {
    static constexpr std::size_t SIZE = 10;
    framework::utils::Arr<int, SIZE> vec{};

    // Fill with sequential values
    for (std::size_t i = 0; i < SIZE; ++i) {
        vec[i] = static_cast<int>(i * SIZE);
    }

    // Verify values
    for (std::size_t i = 0; i < SIZE; ++i) {
        EXPECT_EQ(static_cast<int>(i * SIZE), vec[i]);
    }

    // Test iterators
    std::size_t expected = 0;
    for (const auto &element : vec) {
        EXPECT_EQ(expected * SIZE, element);
        ++expected;
    }
    EXPECT_EQ(SIZE, expected);
}

// Test: Zero values work correctly
TEST(ArrTest, ZeroValues) {
    const framework::utils::Arr<int, 3> vec1{};
    framework::utils::Arr<int, 3> vec2{};

    // Both should be zero-initialized
    EXPECT_TRUE(vec1 == vec2);

    vec2.fill(0);
    EXPECT_TRUE(vec1 == vec2); // Still equal after explicit zero fill
}

// Test: Negative values work correctly
TEST(ArrTest, NegativeValues) {
    static constexpr int VALUE_1 = -1;
    static constexpr int VALUE_2 = -2;
    static constexpr int VALUE_3 = -3;
    const std::array<int, 3> arr = {VALUE_1, VALUE_2, VALUE_3};
    framework::utils::Arr<int, 3> vec(arr);

    EXPECT_EQ(VALUE_1, vec[0]);
    EXPECT_EQ(VALUE_2, vec[1]);
    EXPECT_EQ(VALUE_3, vec[2]);

    static constexpr int FILL_VALUE = -42;
    vec.fill(FILL_VALUE);
    EXPECT_EQ(FILL_VALUE, vec[0]);
    EXPECT_EQ(FILL_VALUE, vec[1]);
    EXPECT_EQ(FILL_VALUE, vec[2]);
}

// Test: Floating-point precision comparison works
TEST(ArrTest, FloatingPointPrecision) {
    const std::array<float, 2> arr1 = {1.0F, 2.0F};
    const std::array<float, 2> arr2 = {1.0F, 2.0F};

    const framework::utils::Arr<float, 2> vec1(arr1);
    framework::utils::Arr<float, 2> vec2(arr2);

    EXPECT_TRUE(vec1 == vec2);

    // Modify slightly
    static constexpr float VALUE = 2.0001F;
    vec2[1] = VALUE;
    EXPECT_FALSE(vec1 == vec2);
}

// Type System Tests

// Test: Different numeric types work correctly
TEST(ArrTest, DifferentNumericTypes) {
    // Test that the template works with different numeric types
    framework::utils::Arr<char, 2> char_arr{};
    framework::utils::Arr<short, 2> short_arr{};
    framework::utils::Arr<long, 2> long_arr{};
    framework::utils::Arr<unsigned int, 2> uint_arr{};

    static constexpr char CHAR_VALUE = 'A';
    static constexpr short SHORT_VALUE = 1'000;
    static constexpr long LONG_VALUE = 1'000'000L;
    static constexpr unsigned int UINT_VALUE = 4'000'000'000U;
    char_arr.fill(CHAR_VALUE);
    short_arr.fill(SHORT_VALUE);
    long_arr.fill(LONG_VALUE);
    uint_arr.fill(UINT_VALUE);

    EXPECT_EQ('A', char_arr[0]);
    EXPECT_EQ('A', char_arr[1]);

    EXPECT_EQ(SHORT_VALUE, short_arr[0]);
    EXPECT_EQ(SHORT_VALUE, short_arr[1]);

    EXPECT_EQ(LONG_VALUE, long_arr[0]);
    EXPECT_EQ(LONG_VALUE, long_arr[1]);

    EXPECT_EQ(UINT_VALUE, uint_arr[0]);
    EXPECT_EQ(UINT_VALUE, uint_arr[1]);
}

// Memory Layout Tests

// Test: Memory layout is contiguous
TEST(ArrTest, MemoryLayoutContiguous) {
    framework::utils::Arr<int, 4> vec{};
    vec[0] = 1;
    vec[1] = 2;
    vec[2] = 3;
    vec[3] = 4;

    // Check that elements are contiguous in memory
    const int *ptr = vec.begin();
    EXPECT_EQ(1, *(std::next(ptr, 0)));
    EXPECT_EQ(2, *(std::next(ptr, 1)));
    EXPECT_EQ(3, *(std::next(ptr, 2)));
    EXPECT_EQ(4, *(std::next(ptr, 3)));
}

// Test: Array size matches expected size
TEST(ArrTest, SizeOfArray) {
    // Verify that Arr doesn't add overhead beyond the contained array
    EXPECT_EQ(sizeof(int) * 3, sizeof(framework::utils::Arr<int, 3>));
    EXPECT_EQ(sizeof(float) * 5, sizeof(framework::utils::Arr<float, 5>));
    EXPECT_EQ(sizeof(double) * 2, sizeof(framework::utils::Arr<double, 2>));
}

// Mathematical Operations Tests

// Test: Manual dot product calculation with Arr
TEST(ArrTest, ManualDotProductCalculation) {
    const std::array<int, 4> arr1 = {1, 2, 3, 4};
    const std::array<int, 4> arr2 = {5, 6, 7, 8};

    const framework::utils::Arr<int, 4> vec1(arr1);
    const framework::utils::Arr<int, 4> vec2(arr2);

    // Manual dot product calculation
    int result = 0;
    for (std::size_t i = 0; i < 4; ++i) {
        result += vec1[i] * vec2[i];
    }

    // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
    EXPECT_EQ(result, 70);
}

// Test: Element-wise operations with Arr
TEST(ArrTest, ElementWiseOperations) {
    const std::array<int, 3> arr1 = {1, 2, 3};
    const std::array<int, 3> arr2 = {4, 5, 6};

    const framework::utils::Arr<int, 3> vec1(arr1);
    const framework::utils::Arr<int, 3> vec2(arr2);

    // Element-wise addition
    framework::utils::Arr<int, 3> sum{};
    for (std::size_t i = 0; i < 3; ++i) {
        sum[i] = vec1[i] + vec2[i];
    }

    EXPECT_EQ(5, sum[0]); // 1 + 4
    EXPECT_EQ(7, sum[1]); // 2 + 5
    EXPECT_EQ(9, sum[2]); // 3 + 6

    // Element-wise multiplication
    framework::utils::Arr<int, 3> product{};
    for (std::size_t i = 0; i < 3; ++i) {
        product[i] = vec1[i] * vec2[i];
    }

    EXPECT_EQ(4, product[0]);  // 1 * 4
    EXPECT_EQ(10, product[1]); // 2 * 5
    EXPECT_EQ(18, product[2]); // 3 * 6
}

// Product Method Tests

// Test: Product method calculates correctly for normal cases
TEST(ArrTest, ProductMethodNormalCases) {
    const std::array<int, 3> arr1 = {2, 3, 4};
    const framework::utils::Arr<int, 3> vec1(arr1);

    EXPECT_EQ(24, vec1.product()); // 2 * 3 * 4 = 24

    const std::array<float, 2> arr2 = {1.5F, 2.0F};
    const framework::utils::Arr<float, 2> vec2(arr2);

    EXPECT_FLOAT_EQ(3.0F, vec2.product()); // 1.5 * 2.0 = 3.0
}

// Test: Product method with single element
TEST(ArrTest, ProductMethodSingleElement) {
    const std::array<int, 1> arr = {42};
    const framework::utils::Arr<int, 1> vec(arr);

    EXPECT_EQ(42, vec.product());
}

// Test: Product method with zeros
TEST(ArrTest, ProductMethodWithZeros) {
    const std::array<int, 3> arr = {1, 0, 3};
    const framework::utils::Arr<int, 3> vec(arr);

    EXPECT_EQ(0, vec.product()); // Any zero makes product zero
}

// Test: Product method with negative numbers
TEST(ArrTest, ProductMethodWithNegatives) {
    const std::array<int, 4> arr = {-1, 2, -3, 4};
    const framework::utils::Arr<int, 4> vec(arr);

    EXPECT_EQ(24, vec.product()); // (-1) * 2 * (-3) * 4 = 24 (even negatives)

    const std::array<int, 3> arr2 = {-2, 3, 5};
    const framework::utils::Arr<int, 3> vec2(arr2);

    EXPECT_EQ(-30, vec2.product()); // (-2) * 3 * 5 = -30 (odd negatives)
}

// Constexpr Tests

// Test: Constexpr array initialization works
TEST(ArrTest, ConstexprArrayInitialization) {
    constexpr std::array<int, 3> ARR = {10, 20, 30};
    const framework::utils::Arr<int, 3> vec(ARR);

    EXPECT_EQ(ARR[0], vec[0]);
    EXPECT_EQ(ARR[1], vec[1]);
    EXPECT_EQ(ARR[2], vec[2]);
}

} // namespace
