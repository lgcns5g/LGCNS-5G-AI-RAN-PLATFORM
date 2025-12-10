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
 * @file cholesky_factor_inv_plugin_tests.cpp
 * @brief C++ Google Test suite for Cholesky Factor Inverse TensorRT plugin
 *
 * This test suite validates the Cholesky Factor Inverse TensorRT plugin.
 */

#include <algorithm>
#include <array>
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

/// Test the Cholesky Factor Inverse plugin engine with specified execution mode
void test_cholesky_factor_inv_plugin_engine(const ExecutionMode mode) {
    StdioLogger logger(nvinfer1::ILogger::Severity::kVERBOSE);

    // Load custom plugins
    ASSERT_TRUE(init_ran_plugins(&logger)) << "Failed to load custom plugins";

    // Load engine
    const TrtEngine engine("cholesky_test.trtengine", logger);

    // Print engine information
    engine.print_engine_info(logger);

    // Get execution context
    auto *context = engine.get_context();
    auto *cuda_engine = engine.get_engine();

    // Cholesky inversion parameters (must match Python test that generates engine)
    static constexpr std::int32_t MATRIX_SIZE = 2;
    static constexpr std::int32_t BATCH_SIZE = 1;

    // Create CUDA stream for async operations
    const CudaStream stream;

    // Create tensors (type is automatically derived from template parameter)
    CudaTensor<float> input({.nbDims = 3, .d = {BATCH_SIZE, MATRIX_SIZE, MATRIX_SIZE}}, "Input");
    CudaTensor<float> output({.nbDims = 3, .d = {BATCH_SIZE, MATRIX_SIZE, MATRIX_SIZE}}, "Output");

    // Initialize input data: 2x2 positive definite matrix [[4.0, 2.0], [2.0, 3.0]] for all batches
    for (std::int32_t batch = 0; batch < BATCH_SIZE; ++batch) {
        input.at(batch, 0, 0) = 4.0F;
        input.at(batch, 0, 1) = 2.0F;
        input.at(batch, 1, 0) = 2.0F;
        input.at(batch, 1, 1) = 3.0F;
    }

    std::fill(output.host().begin(), output.host().end(), 0.0F);

    // Copy input data to GPU asynchronously
    input.copy_to_device(stream.get());
    output.copy_to_device(stream.get());

    // Bind tensors with shape and type validation
    TensorBinder binder;
    ASSERT_TRUE(binder.bind("arg0", input, "input buffer")
                        .bind("result0", output, "output buffer")
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

    // Print results
    std::cout << std::format(
            "\nCholesky Factor Inverse Plugin {} Mode Test Results: Matrix size: {}x{}, Batch "
            "size: {}\n\n",
            execution_mode_to_string(mode),
            MATRIX_SIZE,
            MATRIX_SIZE,
            BATCH_SIZE);

    // Print output matrix (each row separately since format() accepts only one Range)
    std::cout << "Output L^-1 (batch 0):\n";
    for (std::int32_t row = 0; row < MATRIX_SIZE; ++row) {
        std::cout << output.format({0, row, Range(0, MATRIX_SIZE)}) << "\n";
    }

    // Expected L^-1 for input matrix [[4.0, 2.0], [2.0, 3.0]]
    // Cholesky factor L = [[2.0, 0.0], [1.0, 1.414...]]
    // L^-1 should be approximately [[0.5, 0.0], [-0.354, 0.707]]
    const std::array<std::array<float, 2>, 2> expected = {{
            {{0.5F, 0.0F}}, {{-0.3535534F, 0.7071068F}} // -1/(2*sqrt(2)), 1/sqrt(2)
    }};

    static constexpr float TOLERANCE = 1e-3F;

    std::cout << "\nValidation:\n";

    // Validate all matrix elements for all batches
    for (std::int32_t batch = 0; batch < BATCH_SIZE; ++batch) {
        for (std::int32_t row = 0; row < MATRIX_SIZE; ++row) {
            for (std::int32_t col = 0; col < MATRIX_SIZE; ++col) {
                EXPECT_NEAR(output.at(batch, row, col), expected.at(row).at(col), TOLERANCE)
                        << std::format("Batch {}: L^-1[{},{}] mismatch", batch, row, col);
            }
        }

        // Verify lower triangular property for each batch
        for (std::int32_t row = 0; row < MATRIX_SIZE; ++row) {
            for (std::int32_t col = row + 1; col < MATRIX_SIZE; ++col) {
                EXPECT_NEAR(output.at(batch, row, col), 0.0F, TOLERANCE) << std::format(
                        "Batch {}: L^-1[{},{}] should be zero (lower triangular)", batch, row, col);
            }
        }

        std::cout << std::format("Batch {}: All elements validated ✓\n", batch);
    }
}

// Test: Verifies Cholesky Factor Inverse plugin execution in stream mode
TEST(CholeskyFactorInvPluginTest, CholeskyEngineStreamMode) {
    ASSERT_TRUE(set_cuda_device(0));
    EXPECT_NO_THROW(test_cholesky_factor_inv_plugin_engine(ExecutionMode::Stream));
}

// Test: Verifies Cholesky Factor Inverse plugin execution in graph mode
TEST(CholeskyFactorInvPluginTest, CholeskyEngineGraphMode) {
    ASSERT_TRUE(set_cuda_device(0));
    EXPECT_NO_THROW(test_cholesky_factor_inv_plugin_engine(ExecutionMode::Graph));
}

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

} // namespace
