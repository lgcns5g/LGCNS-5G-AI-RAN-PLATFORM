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
 * @file instrumented_functions.cpp
 * @brief Implementation of instrumented functions for testing
 * -finstrument-function and NVTX
 */

#include "instrumented_functions.hpp"

namespace test_functions {

int compute_sum(const int a, const int b) {
    // Simple computation to create some work
    const int result = a + b;
    return result;
}

double compute_product(const double x, const double y) {
    // Some floating point work
    static constexpr int MAX_ITERATIONS = 100;
    static constexpr double INCREMENT_FACTOR = 0.001;

    double result = x * y;
    for (int i = 0; i < MAX_ITERATIONS; ++i) {
        result += i * INCREMENT_FACTOR;
    }
    return result;
}

int perform_iterations(const int count) {
    int sum{};
    for (int i = 0; i < count; ++i) {
        sum += i;
    }

    return sum;
}

int fibonacci(const int n) {
    if (n <= 1) {
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}

int orchestrate_computations() {
    static constexpr int DEFAULT_ITERATIONS = 1000;

    // Call other instrumented functions
    const int sum = compute_sum(42, 58);
    const double product = compute_product(3.14, 2.71);

    perform_iterations(DEFAULT_ITERATIONS);

    // Small fibonacci to avoid excessive recursion in tests
    const int fib = fibonacci(8);

    // Use results to prevent optimization
    return sum + static_cast<int>(product) + fib;
}

} // namespace test_functions
