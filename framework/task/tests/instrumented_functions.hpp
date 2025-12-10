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
 * @file instrumented_functions.hpp
 * @brief Instrumented functions for testing -finstrument-function and NVTX
 * functionality
 */

#ifndef FRAMEWORK_TASK_TESTS_INSTRUMENTED_FUNCTIONS_HPP
#define FRAMEWORK_TASK_TESTS_INSTRUMENTED_FUNCTIONS_HPP

namespace test_functions {

/// Simple computational function for instrumentation testing
__attribute__((visibility("default"))) int compute_sum(int a, int b);

/// Function with floating point operations for instrumentation testing
__attribute__((visibility("default"))) double compute_product(double x, double y);

/// Function with loop for instrumentation testing
__attribute__((visibility("default"))) int perform_iterations(int count);

/// Recursive function for instrumentation testing
__attribute__((visibility("default"))) int fibonacci(int n);

/// Function that calls other functions for testing call graph instrumentation
__attribute__((visibility("default"))) int orchestrate_computations();

} // namespace test_functions

#endif // FRAMEWORK_TASK_TESTS_INSTRUMENTED_FUNCTIONS_HPP
