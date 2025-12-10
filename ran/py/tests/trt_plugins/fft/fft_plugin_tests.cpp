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
 * @file fft_plugin_tests.cpp
 * @brief C++ Google Test suite for FFT TensorRT plugin
 *
 * This test suite validates the FFT TensorRT plugin.
 */

#include <algorithm>
#include <cmath>
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

/// Test the FFT plugin engine with specified execution mode
void test_fft_plugin_engine(const ExecutionMode mode) {
    StdioLogger logger(nvinfer1::ILogger::Severity::kVERBOSE);

    // Load custom plugins
    ASSERT_TRUE(init_ran_plugins(&logger)) << "Failed to load custom plugins";

    // Load engine
    const TrtEngine engine("fft_test.trtengine", logger);

    // Print engine information
    engine.print_engine_info(logger);

    // Get execution context
    auto *context = engine.get_context();
    auto *cuda_engine = engine.get_engine();

    // FFT parameters (must match Python test that generates engine)
    static constexpr std::int32_t FFT_SIZE = 2048;
    static constexpr std::int32_t BATCH_SIZE = 4;

    // Create CUDA stream for async operations
    const CudaStream stream;

    // Create tensors for FFT input and output
    CudaTensor<float> input_real({.nbDims = 2, .d = {BATCH_SIZE, FFT_SIZE}}, "Input Real");
    CudaTensor<float> input_imag({.nbDims = 2, .d = {BATCH_SIZE, FFT_SIZE}}, "Input Imag");
    CudaTensor<float> output_real({.nbDims = 2, .d = {BATCH_SIZE, FFT_SIZE}}, "Output Real");
    CudaTensor<float> output_imag({.nbDims = 2, .d = {BATCH_SIZE, FFT_SIZE}}, "Output Imag");

    // Initialize input data: real part = 1.0, imaginary part = 0.0
    std::fill(input_real.host().begin(), input_real.host().end(), 1.0F);
    std::fill(input_imag.host().begin(), input_imag.host().end(), 0.0F);
    std::fill(output_real.host().begin(), output_real.host().end(), 0.0F);
    std::fill(output_imag.host().begin(), output_imag.host().end(), 0.0F);

    // Copy input data to GPU asynchronously
    input_real.copy_to_device(stream.get());
    input_imag.copy_to_device(stream.get());
    output_real.copy_to_device(stream.get());
    output_imag.copy_to_device(stream.get());

    // Bind tensors with shape and type validation
    TensorBinder binder;
    ASSERT_TRUE(binder.bind("arg0", input_real, "input_real buffer")
                        .bind("arg1", input_imag, "input_imag buffer")
                        .bind("result0", output_real, "output_real buffer")
                        .bind("result1", output_imag, "output_imag buffer")
                        .apply(context, cuda_engine, logger))
            << "Failed to bind tensors";

    // Execute
    TrtExecutor executor(mode);
    executor.prepare(context, stream.get());
    executor.execute(stream.get());
    stream.synchronize();

    // Copy outputs back to host asynchronously
    output_real.copy_from_device(stream.get());
    output_imag.copy_from_device(stream.get());

    // Final synchronization to ensure all copies complete
    stream.synchronize();

    // Print results
    std::cout << std::format(
            "\nFFT Plugin {} Mode Test Results: FFT size: {}, Batch size: {}\n\n",
            execution_mode_to_string(mode),
            FFT_SIZE,
            BATCH_SIZE);
    std::cout << std::format(
            "Expected per batch: [{}+0j, 0+0j, 0+0j, ...] (impulse at DC)\n", FFT_SIZE);

    // Print first 10 values for batch 0
    std::cout << "\nBatch 0 - First 10 values:\n";
    std::cout << output_real.format({0, Range(0, 10)}) << "\n";
    std::cout << output_imag.format({0, Range(0, 10)}) << "\n";

    // Validate DC component and non-DC magnitude for each batch
    static constexpr float DC_TOLERANCE = 1e-3F;
    static constexpr float NON_DC_TOLERANCE = 1e-3F;
    const auto expected_dc = static_cast<float>(FFT_SIZE);

    for (std::int32_t batch = 0; batch < BATCH_SIZE; ++batch) {
        // Check DC component (first element in batch)
        const float dc_real = output_real.at(batch, 0);
        const float dc_imag = output_imag.at(batch, 0);

        EXPECT_NEAR(dc_real, expected_dc, DC_TOLERANCE)
                << std::format("Batch {}: DC real component mismatch", batch);
        EXPECT_NEAR(dc_imag, 0.0F, DC_TOLERANCE)
                << std::format("Batch {}: DC imaginary component should be zero", batch);

        // Check non-DC components
        float max_non_dc{};
        for (int64_t i = 1; i < FFT_SIZE; ++i) {
            const float mag = std::sqrt(
                    output_real.at(batch, i) * output_real.at(batch, i) +
                    output_imag.at(batch, i) * output_imag.at(batch, i));
            max_non_dc = std::max(max_non_dc, mag);
        }

        EXPECT_LT(max_non_dc, NON_DC_TOLERANCE) << std::format(
                "Batch {}: Non-DC components too large (max: {:.2e})", batch, max_non_dc);

        const float dc_error = std::abs(dc_real - expected_dc) + std::abs(dc_imag);
        std::cout << std::format(
                "Batch {}: DC error: {:.2e}, Non-DC max: {:.2e}\n", batch, dc_error, max_non_dc);
    }
}

// Test: Verifies FFT plugin execution in stream mode
TEST(FftPluginTest, FftEngineStreamMode) {
    ASSERT_TRUE(set_cuda_device(0));
    EXPECT_NO_THROW(test_fft_plugin_engine(ExecutionMode::Stream));
}

// Test: Verifies FFT plugin execution in graph mode
TEST(FftPluginTest, FftEngineGraphMode) {
    ASSERT_TRUE(set_cuda_device(0));
    EXPECT_NO_THROW(test_fft_plugin_engine(ExecutionMode::Graph));
}

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

} // namespace
