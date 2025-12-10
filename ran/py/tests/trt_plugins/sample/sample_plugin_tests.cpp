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
 * @file sample_plugin_tests.cpp
 * @brief C++ Google Test suite for TensorRT sample plugins
 *
 * This test suite loads TensorRT engines created by Python tests and runs
 * the same tests to verify C++ and Python interoperability.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <driver_types.h>

#include <gtest/gtest.h>

#include "trt_test_utils.hpp"

namespace {

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

using namespace ran::trt_utils;

/// Test the sequential sum plugin engine with specified execution mode
void test_sequential_sum_engine(const ExecutionMode mode) {
    StdioLogger logger(nvinfer1::ILogger::Severity::kVERBOSE);

    // Load custom plugins
    ASSERT_TRUE(init_ran_plugins(&logger)) << "Failed to load custom plugins";

    // Load engine
    const TrtEngine engine("sequential_sum_test.trtengine", logger);

    // Print engine information
    engine.print_engine_info(logger);

    // Get execution context
    auto *context = engine.get_context();
    auto *cuda_engine = engine.get_engine();

    // For the sequential sum test, input is (1, 5) and output is (1, 5)
    static constexpr std::int32_t BATCH_SIZE = 1;
    static constexpr std::int32_t INPUT_SIZE = 5;

    // Create CUDA stream for async operations
    const CudaStream stream;

    // Create tensors (type is automatically derived from template parameter)
    CudaTensor<float> input({.nbDims = 2, .d = {BATCH_SIZE, INPUT_SIZE}}, "Input");
    CudaTensor<float> output({.nbDims = 2, .d = {BATCH_SIZE, INPUT_SIZE}}, "Output");

    // Test data: [1.0, 2.0, 3.0, 4.0, 5.0]
    input.host() = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F};

    // Expected output: [1.0, 3.0, 6.0, 10.0, 15.0] (sequential sum)
    const std::vector<float> expected_output = {1.0F, 3.0F, 6.0F, 10.0F, 15.0F};

    // Copy input data to GPU asynchronously
    input.copy_to_device(stream.get());

    // Bind tensors with shape and type validation
    TensorBinder binder;
    ASSERT_TRUE(binder.bind("input", input, "input buffer")
                        .bind("output", output, "output buffer")
                        .apply(context, cuda_engine, logger))
            << "Failed to bind tensors";

    // Execute
    TrtExecutor executor(mode);
    executor.prepare(context, stream.get());
    executor.execute(stream.get());
    stream.synchronize();

    // Copy output back to host asynchronously
    output.copy_from_device(stream.get());

    // Final synchronization to ensure all copies complete
    stream.synchronize();

    // Verify results
    static constexpr float TOLERANCE = 1e-5F;

    std::cout << std::format(
            "\nSequential Sum Engine {} Mode Test Results:\n", execution_mode_to_string(mode));
    for (std::int32_t i = 0; i < INPUT_SIZE; ++i) {
        const float diff = std::abs(output[i] - expected_output[i]);

        EXPECT_NEAR(output[i], expected_output[i], TOLERANCE)
                << std::format("Element [{}] mismatch", i);

        std::cout << std::format(
                "  [{}] output: {:.6f}, expected: {:.6f}, diff: {:.6e}\n",
                i,
                output[i],
                expected_output[i],
                diff);
    }
}

/// Test the hybrid model engine (TensorRT portion only) with specified execution mode
void test_hybrid_model_engine(const ExecutionMode mode) {
    StdioLogger logger(nvinfer1::ILogger::Severity::kVERBOSE);

    // Load custom plugins
    ASSERT_TRUE(init_ran_plugins(&logger)) << "Failed to load custom plugins";

    // Load engine
    const TrtEngine engine("torch_model_with_trt_plugin.trtengine", logger);

    // Print engine information
    engine.print_engine_info(logger);

    // Get execution context
    auto *context = engine.get_context();
    auto *cuda_engine = engine.get_engine();

    // For the hybrid model test, we'll use a batch of 2 samples
    static constexpr std::int32_t BATCH_SIZE = 2;
    static constexpr std::int32_t INPUT_SIZE = 5;

    // Create CUDA stream for async operations
    const CudaStream stream;

    // Create tensors (type is automatically derived from template parameter)
    CudaTensor<float> input({.nbDims = 2, .d = {BATCH_SIZE, INPUT_SIZE}}, "Input");
    CudaTensor<float> output({.nbDims = 2, .d = {BATCH_SIZE, INPUT_SIZE}}, "Output");

    // Test data (this will be the output from PyTorch preprocessing: scale by 2)
    // Original: [[1,2,3,4,5], [2,3,4,5,6]]
    // After scale by 2: [[2,4,6,8,10], [4,6,8,10,12]]
    input.host() = {
            2.0F,
            4.0F,
            6.0F,
            8.0F,
            10.0F, // First row
            4.0F,
            6.0F,
            8.0F,
            10.0F,
            12.0F // Second row
    };

    // Expected output from TensorRT plugin (sequential sum of flattened tensor):
    // Flattened: [2,4,6,8,10,4,6,8,10,12]
    // Sequential sum: [2,6,12,20,30,34,40,48,58,70]
    // Reshaped: [[2,6,12,20,30], [34,40,48,58,70]]
    const std::vector<float> expected_output = {
            2.0F,
            6.0F,
            12.0F,
            20.0F,
            30.0F, // First sample
            34.0F,
            40.0F,
            48.0F,
            58.0F,
            70.0F // Second sample
    };

    // Set input shape for dynamic batch
    nvinfer1::Dims input_dims{};
    input_dims.nbDims = 2;
    input_dims.d[0] = BATCH_SIZE;
    input_dims.d[1] = INPUT_SIZE;

    ASSERT_TRUE(context->setInputShape("input", input_dims)) << "Failed to set input shape";

    // Copy input data to GPU asynchronously
    input.copy_to_device(stream.get());

    // Bind tensors with shape and type validation
    TensorBinder binder;
    ASSERT_TRUE(binder.bind("input", input, "input buffer")
                        .bind("output", output, "output buffer")
                        .apply(context, cuda_engine, logger))
            << "Failed to bind tensors";

    // Execute
    TrtExecutor executor(mode);
    executor.prepare(context, stream.get());
    executor.execute(stream.get());
    stream.synchronize();

    // Copy output back to host asynchronously
    output.copy_from_device(stream.get());

    // Final synchronization to ensure all copies complete
    stream.synchronize();

    // Verify results
    static constexpr float TOLERANCE = 1e-5F;

    std::cout << std::format(
            "\nHybrid Model Engine {} Mode Test Results:\n", execution_mode_to_string(mode));
    for (std::int32_t i = 0; i < BATCH_SIZE * INPUT_SIZE; ++i) {
        const float diff = std::abs(output[i] - expected_output[i]);

        const std::int32_t batch = i / INPUT_SIZE;
        const std::int32_t elem = i % INPUT_SIZE;

        EXPECT_NEAR(output[i], expected_output[i], TOLERANCE)
                << std::format("Element [{},{}] mismatch", batch, elem);

        std::cout << std::format(
                "  [{},{}] output: {:.6f}, expected: {:.6f}, diff: {:.6e}\n",
                batch,
                elem,
                output[i],
                expected_output[i],
                diff);
    }
}

// Test: Verifies sequential sum plugin with single input vector in stream mode
TEST(TensorRTPluginTest, SequentialSumEngineStreamMode) {
    ASSERT_TRUE(set_cuda_device(0));
    EXPECT_NO_THROW(test_sequential_sum_engine(ExecutionMode::Stream));
}

// Test: Verifies sequential sum plugin with single input vector in graph mode
TEST(TensorRTPluginTest, SequentialSumEngineGraphMode) {
    ASSERT_TRUE(set_cuda_device(0));
    EXPECT_NO_THROW(test_sequential_sum_engine(ExecutionMode::Graph));
}

// Test: Verifies hybrid model TensorRT engine with batched input in stream mode
TEST(TensorRTPluginTest, HybridModelEngineStreamMode) {
    ASSERT_TRUE(set_cuda_device(0));
    EXPECT_NO_THROW(test_hybrid_model_engine(ExecutionMode::Stream));
}

// Test: Verifies hybrid model TensorRT engine with batched input in graph mode
TEST(TensorRTPluginTest, HybridModelEngineGraphMode) {
    ASSERT_TRUE(set_cuda_device(0));
    EXPECT_NO_THROW(test_hybrid_model_engine(ExecutionMode::Graph));
}

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

} // namespace
