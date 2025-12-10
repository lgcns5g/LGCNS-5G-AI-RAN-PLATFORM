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
 * @file dmrs_plugin_tests.cpp
 * @brief C++ Google Test suite for DMRS TensorRT plugin
 *
 * This test suite validates the DMRS TensorRT plugin.
 */

#include <algorithm>
#include <cstdint>
#include <format>
#include <iostream>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <NvInfer.h>
#include <driver_types.h>

#include <gtest/gtest.h>

#include "trt_test_utils.hpp"
#include "trt_test_utils_impl.hpp"

namespace {

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

using namespace ran::trt_utils;

/// Test the DMRS plugin engine with specified execution mode
void test_dmrs_plugin_engine(const ExecutionMode mode) {
    StdioLogger logger(nvinfer1::ILogger::Severity::kVERBOSE);

    // Load custom plugins
    ASSERT_TRUE(init_ran_plugins(&logger)) << "Failed to load custom plugins";

    // Load engine
    const TrtEngine engine("dmrs_test.trtengine", logger);

    // Print engine information
    engine.print_engine_info(logger);

    // Get execution context
    auto *context = engine.get_context();
    auto *cuda_engine = engine.get_engine();

    // DMRS parameters
    static constexpr std::int32_t SLOT_NUMBER = 0;
    static constexpr std::int32_t N_F = 3276;      // Length of Gold sequence per port
    static constexpr std::int32_t N_T = 2;         // Number of OFDM symbols per slot
    static constexpr std::int32_t N_DMRS_ID = 0;   // DMRS identity
    static constexpr std::int32_t N_SYMBOLS = 14;  // Number of symbols per slot
    static constexpr std::int32_t N_REAL_IMAG = 2; // [real, imag]

    // Create CUDA stream for async operations
    const CudaStream stream;

    // Create tensors (type is automatically derived from template parameter)
    CudaTensor<std::int32_t> slot({.nbDims = 0}, "Slot");
    CudaTensor<std::int32_t> dmrs_id({.nbDims = 0}, "DMRS ID");
    CudaTensor<std::int32_t> scr_seq(
            {.nbDims = 3, .d = {N_SYMBOLS, N_T, N_F}}, "Scrambling Sequence");
    CudaTensor<float> rdmrs({.nbDims = 4, .d = {N_REAL_IMAG, N_SYMBOLS, N_T, N_F / 2}}, "DMRS");

    // Initialize input data
    slot[0] = SLOT_NUMBER;
    dmrs_id[0] = N_DMRS_ID;
    std::fill(scr_seq.host().begin(), scr_seq.host().end(), 0);
    std::fill(rdmrs.host().begin(), rdmrs.host().end(), 0.0F);

    // Copy input data to GPU asynchronously
    slot.copy_to_device(stream.get());
    dmrs_id.copy_to_device(stream.get());
    scr_seq.copy_to_device(stream.get());
    rdmrs.copy_to_device(stream.get());

    // Bind tensors with shape and type validation
    // Note: result0 is complex DMRS (2, 14, 2, 1638), result1 is binary sequence (14, 2, 3276)
    TensorBinder binder;
    ASSERT_TRUE(binder.bind("arg0", slot, "slot buffer")
                        .bind("arg1", dmrs_id, "dmrs_id buffer")
                        .bind("result0", rdmrs, "r_dmrs buffer")
                        .bind("result1", scr_seq, "scr_seq buffer")
                        .apply(context, cuda_engine, logger))
            << "Failed to bind tensors";

    // Execute
    TrtExecutor executor(mode);
    executor.prepare(context, stream.get());
    executor.execute(stream.get());
    stream.synchronize();

    // Copy outputs back to host asynchronously
    scr_seq.copy_from_device(stream.get());
    rdmrs.copy_from_device(stream.get());

    // Final synchronization to ensure all copies complete
    stream.synchronize();

    // Print results
    std::cout << std::format(
            "\nDMRS Plugin {} Mode Test Results: SLOT {}, N_F: {}, N_DMRS_SYMBOLS: {}, N_SYMBOLS: "
            "{}, "
            "N_DMRS_ID: {}\n",
            execution_mode_to_string(mode),
            SLOT_NUMBER,
            N_F,
            N_T,
            N_SYMBOLS,
            N_DMRS_ID);

    // Print DMRS values (r_dmrs has shape (N_REAL_IMAG, N_SYMBOLS, N_T, N_F/2) where dim 0 is
    // [real, imag])
    std::cout << rdmrs.format_complex({0, 0, Range(0, 20)}) << '\n';

    // Print scrambling sequence (shape (N_SYMBOLS, N_T, N_F))
    std::cout << scr_seq.format({0, 0, Range(0, 20)}) << '\n';

    // Basic validation: Check that output is non-zero (computed)
    EXPECT_TRUE(rdmrs.has_non_zero()) << "DMRS output should contain non-zero values";
    EXPECT_TRUE(scr_seq.has_non_zero())
            << "Scrambling sequence output should contain non-zero values";
}

// Test: Verifies DMRS plugin execution in stream mode
TEST(DmrsPluginTest, DmrsEngineStreamMode) {
    ASSERT_TRUE(set_cuda_device(0));
    EXPECT_NO_THROW(test_dmrs_plugin_engine(ExecutionMode::Stream));
}

// Test: Verifies DMRS plugin execution in graph mode
TEST(DmrsPluginTest, DmrsEngineGraphMode) {
    ASSERT_TRUE(set_cuda_device(0));
    EXPECT_NO_THROW(test_dmrs_plugin_engine(ExecutionMode::Graph));
}

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

} // namespace
