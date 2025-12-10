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
 * @file nvtx_tests.cpp
 * @brief Unit tests for NVTX functions
 */

#include <iterator>
#include <memory>
#include <tuple>

#include <quill/LogMacros.h>

#include <gtest/gtest.h>

#include "instrumented_functions.hpp"
#include "log/rt_log_macros.hpp"
#include "task/nvtx.hpp"
#include "task/task_log.hpp"

// Declare C profiling functions for testing
extern "C" {
// NOLINTNEXTLINE(bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp,readability-identifier-naming)
void __cyg_profile_func_enter(void *func, void *caller);
// NOLINTNEXTLINE(bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp,readability-identifier-naming)
void __cyg_profile_func_exit(void *func, void *caller);
}

namespace {
namespace ft = framework::task;

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

void test_nvtx_function() {
    NVTX_FUNCTION();
    { NVTX_RANGE("test_nvtx_range"); }
}

TEST(Nvtx, TestFunctionRange) {
    // NVTX is automatically enabled/disabled based on nsys/ncu detection
    // Just test that the functions work and log stats
    for (int i = 0; i < 10000; i++) {
        test_nvtx_function();
    }

    // Log stats using the loggable Stats struct
    const auto stats = ft::Nvtx::get_stats();
    RT_LOGC_INFO(ft::TaskLog::TaskNvtx, "NVTX Stats: {}", stats);
}

TEST(Nvtx, TestInstrumentedFunctions) {
    // Test individual instrumented functions
    for (int i = 0; i < 1000; ++i) {
        test_functions::compute_sum(i, i + 1);
    }

    for (int i = 0; i < 500; ++i) {
        test_functions::compute_product(1.5 + i * 0.1, 2.0 + i * 0.05);
    }

    std::ignore = test_functions::perform_iterations(2000);

    // Test recursive function (limited depth to avoid stack overflow)
    for (int i = 0; i < 100; ++i) {
        std::ignore = test_functions::fibonacci(6); // Small fibonacci number
    }

    // Test orchestration function that calls multiple other functions
    for (int i = 0; i < 200; ++i) {
        std::ignore = test_functions::orchestrate_computations();
    }

    const auto stats = ft::Nvtx::get_stats();
    RT_LOGC_INFO(ft::TaskLog::TaskNvtx, "NVTX Stats: {}", stats);
}

TEST(Nvtx, TestEnabledState) {
    // Test that we can check the enabled state
    // The actual state depends on whether nsys/ncu profiling is detected
    const bool enabled = ft::Nvtx::is_enabled();
    RT_LOGC_INFO(ft::TaskLog::TaskNvtx, "NVTX enabled state: {}", enabled);
    EXPECT_EQ(enabled, ft::Nvtx::is_enabled());
}

TEST(Nvtx, TestFallbackFormatting) {
    // Test the fallback path by directly calling the C profiling functions
    // with invalid function pointers that can't be resolved to symbol names

    // Only test if NVTX is enabled, otherwise the C functions return early
    if (!ft::Nvtx::is_enabled()) {
        GTEST_SKIP() << "NVTX not enabled, skipping fallback test";
    }

    // Get initial stats
    const auto initial_stats = ft::Nvtx::get_stats();

    // Use some arbitrary invalid function pointers that won't resolve
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays)
    void *invalid_ptrs[] = {// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
                            reinterpret_cast<void *>(0x1234),
                            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
                            reinterpret_cast<void *>(0xDEADBEEF),
                            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
                            reinterpret_cast<void *>(0x7FFF0000)};

    // Directly call the C profiling functions to trigger fallback
    for (void *ptr : invalid_ptrs) {
        // This should trigger the fallback formatting path in
        // __cyg_profile_func_enter
        __cyg_profile_func_enter(ptr, nullptr);
        __cyg_profile_func_exit(ptr, nullptr);
    }

    // Get final stats and verify fallback path was exercised
    const auto final_stats = ft::Nvtx::get_stats();

    // Verify that fallback formatting was triggered
    EXPECT_GT(final_stats.fallback_functions, initial_stats.fallback_functions)
            << "Fallback count should have increased";

    // Verify total function count increased
    EXPECT_GT(final_stats.total_functions, initial_stats.total_functions)
            << "Total function count should have increased";

    // Verify the exact number of fallbacks (should be 3 for our 3 test pointers)
    const int expected_fallbacks = static_cast<int>(std::size(invalid_ptrs));
    EXPECT_EQ(final_stats.fallback_functions - initial_stats.fallback_functions, expected_fallbacks)
            << "Should have exactly " << expected_fallbacks << " fallback calls";

    // Log stats for debugging using the loggable Stats struct
    RT_LOGC_INFO(ft::TaskLog::TaskNvtx, "NVTX Stats after fallback test: {}", final_stats);
}

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

} // namespace
