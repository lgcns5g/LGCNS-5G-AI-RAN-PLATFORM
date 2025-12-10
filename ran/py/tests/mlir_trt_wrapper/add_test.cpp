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
 * @file add_test.cpp
 * @brief Google Test suite for MLIR-TensorRT add operation
 *
 * This test verifies that Python-compiled MLIR-TensorRT engines can be
 * loaded and executed from C++ using the TensorRT API.
 *
 * Supports: float32, float16, bfloat16, int32
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <iostream>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <driver_types.h>
#include <trt_test_utils.hpp>

#include <gtest/gtest.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

using namespace ran::trt_utils;

/// Template test function for any numeric type
template <typename T>
void test_add_operation(const std::string &engine_path, const float tolerance) {
    StdioLogger logger;

    // Load engine created by Python test
    const TrtEngine engine(engine_path, logger);

    // Print engine information
    engine.print_engine_info(logger);

    // Get execution context
    auto *context = engine.get_context();
    auto *cuda_engine = engine.get_engine();

    // Create CUDA stream for async operations
    const CudaStream stream;

    // Create tensors
    static constexpr std::int32_t SIZE = 2;
    CudaTensor<T> input1({.nbDims = 1, .d = {SIZE}}, "Input1");
    CudaTensor<T> input2({.nbDims = 1, .d = {SIZE}}, "Input2");
    CudaTensor<T> output({.nbDims = 1, .d = {SIZE}}, "Output");

    // Test data: [12.34, 56.78] + [23.45, 67.89] = [35.79, 124.67]
    // For int32: rounds to [12, 57] + [23, 68] = [35, 125]
    input1.host() = {from_float<T>(12.34F), from_float<T>(56.78F)};
    input2.host() = {from_float<T>(23.45F), from_float<T>(67.89F)};

    std::vector<float> expected_output(SIZE);
    for (std::int32_t i = 0; i < SIZE; ++i) {
        expected_output[i] = to_float(input1[i]) + to_float(input2[i]);
    }

    // Copy input data to GPU asynchronously
    input1.copy_to_device(stream.get());
    input2.copy_to_device(stream.get());

    // Bind tensors with shape and type validation
    TensorBinder binder;
    ASSERT_TRUE(binder.bind("arg0", input1, "first input")
                        .bind("arg1", input2, "second input")
                        .bind("result0", output, "output")
                        .apply(context, cuda_engine, logger))
            << "Failed to bind tensors";

    // Execute with custom stream
    ASSERT_TRUE(context->enqueueV3(stream.get())) << "TensorRT execution failed";

    // Synchronize stream
    stream.synchronize();

    // Copy output back to host asynchronously
    output.copy_from_device(stream.get());

    // Final synchronization to ensure all copies complete
    stream.synchronize();

    // Verify results with detailed output
    std::cout << std::format("\n=== {} Test Results ===\n", engine_path);
    std::cout << std::format("Tolerance: {:.2e}\n\n", tolerance);

    float max_diff = 0.0F;
    for (std::int32_t i = 0; i < SIZE; ++i) {
        const float input1_f = to_float(input1[i]);
        const float input2_f = to_float(input2[i]);
        const float output_f = to_float(output[i]);
        const float diff = std::abs(output_f - expected_output[i]);

        max_diff = std::max(max_diff, diff);

        std::cout << std::format(
                "  [{}] {:.6f} + {:.6f} = {:.6f} (expected: {:.6f}, diff: {:.2e})\n",
                i,
                input1_f,
                input2_f,
                output_f,
                expected_output[i],
                diff);

        EXPECT_NEAR(output_f, expected_output[i], tolerance)
                << std::format("Element [{}] mismatch", i);
    }

    std::cout << std::format("\nMax diff: {:.2e}, Tolerance: {:.2e}\n", max_diff, tolerance);
}

/// Note: Requires running Python test: test_compile_and_execute_add_function[float32]
TEST(MlirTensorRTTest, AddOperationFloat32) {
    test_add_operation<float>("add_func_float32.trtengine", 1e-5F);
}

/// Note: Requires running Python test: test_compile_and_execute_add_function[float16]
TEST(MlirTensorRTTest, AddOperationFloat16) {
    test_add_operation<__half>("add_func_float16.trtengine", 1e-1F);
}

/// Note: Requires running Python test: test_compile_and_execute_add_function[bfloat16]
TEST(MlirTensorRTTest, AddOperationBFloat16) {
    test_add_operation<__nv_bfloat16>("add_func_bfloat16.trtengine", 0.5F);
}

/// Note: Requires running Python test: test_compile_and_execute_add_function[int32]
TEST(MlirTensorRTTest, AddOperationInt32) {
    test_add_operation<std::int32_t>("add_func_int32.trtengine", 0.0F);
}

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

} // namespace
