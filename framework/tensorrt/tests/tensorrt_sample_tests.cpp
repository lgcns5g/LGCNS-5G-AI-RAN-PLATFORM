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
 * @file tensorrt_sample_tests.cpp
 * @brief Sample tests for TensorRT library documentation
 */

#include <array>
#include <bit>
#include <cstdint>
#include <memory>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <NvInfer.h>
#include <driver_types.h>

#include <gtest/gtest.h>

#include "tensor/data_types.hpp"
#include "tensorrt/mlir_trt_engine.hpp"
#include "tensorrt/trt_engine_interface.hpp"
#include "tensorrt/trt_engine_params.hpp"
#include "utils/errors.hpp"

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

namespace {

namespace tensor = framework::tensor;
namespace utils = framework::utils;
using ITrtEngine = framework::tensorrt::ITrtEngine;
using MLIRTensorParams = framework::tensorrt::MLIRTensorParams;
using MLIRTrtEngine = framework::tensorrt::MLIRTrtEngine;

/// Creates a mock pointer from an address for testing purposes
[[nodiscard]] void *mock_ptr(const std::uintptr_t address) noexcept {
    return std::bit_cast<void *>(address);
}

/// Creates a mock CUDA stream from an address for testing purposes
[[nodiscard]] cudaStream_t mock_stream(const std::uintptr_t address) noexcept {
    return std::bit_cast<cudaStream_t>(address);
}

/**
 * @brief Mock TensorRT engine for documentation examples
 */
class MockTrtEngine final : public ITrtEngine {
public:
    [[nodiscard]]
    utils::NvErrc set_input_shape(
            [[maybe_unused]] const std::string_view tensor_name,
            [[maybe_unused]] const nvinfer1::Dims &dims) final {
        return utils::NvErrc::Success;
    }

    [[nodiscard]]
    utils::NvErrc set_tensor_address(
            [[maybe_unused]] const std::string_view tensor_name,
            [[maybe_unused]] void *address) final {
        return utils::NvErrc::Success;
    }

    [[nodiscard]]
    utils::NvErrc enqueue_inference([[maybe_unused]] cudaStream_t cu_stream) final {
        return utils::NvErrc::Success;
    }

    [[nodiscard]]
    bool all_input_dimensions_specified() const final {
        return true;
    }
};

TEST(TensorRTSampleTests, TensorParams) {
    // example-begin tensor-params-1
    // Define tensor parameters with name, data type, rank, and dimensions
    MLIRTensorParams input_params{
            .name = "input_data", .data_type = tensor::TensorR32F, .rank = 2, .dims = {128, 256}};

    // Access tensor properties
    const auto rank = input_params.rank;
    const auto batch_size = input_params.dims[0];
    const auto feature_size = input_params.dims[1];
    // example-end tensor-params-1

    EXPECT_EQ(rank, 2);
    EXPECT_EQ(batch_size, 128);
    EXPECT_EQ(feature_size, 256);
}

TEST(TensorRTSampleTests, TensorParamsWithStrides) {
    // example-begin tensor-params-strides-1
    // Define tensor parameters with explicit strides
    MLIRTensorParams params{
            .name = "data",
            .data_type = tensor::TensorR32F,
            .rank = 3,
            .dims = {4, 8, 16},
            .strides = {128, 16, 1}};

    // Verify stride configuration
    const auto outer_stride = params.strides[0];
    const auto middle_stride = params.strides[1];
    const auto inner_stride = params.strides[2];
    // example-end tensor-params-strides-1

    EXPECT_EQ(outer_stride, 128);
    EXPECT_EQ(middle_stride, 16);
    EXPECT_EQ(inner_stride, 1);
}

TEST(TensorRTSampleTests, EngineConstruction) {
    // example-begin engine-construction-1
    // Define input and output tensor parameters
    std::vector<MLIRTensorParams> input_params = {
            {.name = "input", .data_type = tensor::TensorR32F, .rank = 1, .dims = {1024}}};

    std::vector<MLIRTensorParams> output_params = {
            {.name = "output", .data_type = tensor::TensorR32F, .rank = 1, .dims = {1024}}};

    // Create TensorRT runtime (mock for documentation)
    auto runtime = std::make_unique<MockTrtEngine>();

    // Construct MLIR TensorRT engine
    const MLIRTrtEngine engine(
            std::move(input_params), std::move(output_params), std::move(runtime));
    // example-end engine-construction-1

    SUCCEED();
}

TEST(TensorRTSampleTests, EngineSetup) {
    // Create engine
    std::vector<MLIRTensorParams> input_params = {
            {.name = "input", .data_type = tensor::TensorR32F, .rank = 1, .dims = {1024}}};

    std::vector<MLIRTensorParams> output_params = {
            {.name = "output", .data_type = tensor::TensorR32F, .rank = 1, .dims = {1024}}};

    auto runtime = std::make_unique<MockTrtEngine>();
    MLIRTrtEngine engine(std::move(input_params), std::move(output_params), std::move(runtime));

    // example-begin engine-setup-1
    // Prepare buffer pointers (CUDA device memory)
    const std::vector<void *> input_buffers = {mock_ptr(0x1000)};
    const std::vector<void *> output_buffers = {mock_ptr(0x2000)};

    // Setup engine with buffer addresses
    const auto setup_result = engine.setup(input_buffers, output_buffers);
    // example-end engine-setup-1

    EXPECT_EQ(setup_result, utils::NvErrc::Success);
}

TEST(TensorRTSampleTests, EngineInference) {
    // Create and setup engine
    std::vector<MLIRTensorParams> input_params = {
            {.name = "input", .data_type = tensor::TensorR32F, .rank = 1, .dims = {1024}}};

    std::vector<MLIRTensorParams> output_params = {
            {.name = "output", .data_type = tensor::TensorR32F, .rank = 1, .dims = {1024}}};

    auto runtime = std::make_unique<MockTrtEngine>();
    MLIRTrtEngine engine(std::move(input_params), std::move(output_params), std::move(runtime));

    const std::vector<void *> input_buffers = {mock_ptr(0x1000)};
    const std::vector<void *> output_buffers = {mock_ptr(0x2000)};
    std::ignore = engine.setup(input_buffers, output_buffers);

    // example-begin engine-execution-1
    // Create CUDA stream for execution
    cudaStream_t stream = mock_stream(0x100);

    // Execute the engine
    const auto result = engine.run(stream);
    // example-end engine-execution-1

    EXPECT_EQ(result, utils::NvErrc::Success);
}

TEST(TensorRTSampleTests, CompleteWorkflow) {
    // example-begin complete-workflow-1
    // Step 1: Define tensor parameters
    std::vector<MLIRTensorParams> inputs = {
            {.name = "input0", .data_type = tensor::TensorR32F, .rank = 2, .dims = {32, 128}},
            {.name = "input1", .data_type = tensor::TensorR32F, .rank = 2, .dims = {32, 128}}};

    std::vector<MLIRTensorParams> outputs = {
            {.name = "result", .data_type = tensor::TensorR32F, .rank = 2, .dims = {32, 128}}};

    // Step 2: Create engine with TensorRT runtime
    auto trt_runtime = std::make_unique<MockTrtEngine>();
    MLIRTrtEngine engine(std::move(inputs), std::move(outputs), std::move(trt_runtime));

    // Step 3: Setup buffers
    const std::vector<void *> input_addrs = {mock_ptr(0x1000), mock_ptr(0x2000)};
    const std::vector<void *> output_addrs = {mock_ptr(0x3000)};
    const auto setup_err = engine.setup(input_addrs, output_addrs);

    // Step 4: Run the engine
    cudaStream_t cu_stream = mock_stream(0x100);
    const auto run_err = engine.run(cu_stream);
    // example-end complete-workflow-1

    EXPECT_EQ(setup_err, utils::NvErrc::Success);
    EXPECT_EQ(run_err, utils::NvErrc::Success);
}

TEST(TensorRTSampleTests, MultiRankTensors) {
    // example-begin multi-rank-tensors-1
    // Define tensors with different ranks
    const MLIRTensorParams scalar{.name = "scalar", .data_type = tensor::TensorR32F, .rank = 0};

    const MLIRTensorParams vector{
            .name = "vector", .data_type = tensor::TensorR32F, .rank = 1, .dims = {256}};

    const MLIRTensorParams matrix{
            .name = "matrix", .data_type = tensor::TensorR32F, .rank = 2, .dims = {32, 64}};

    const MLIRTensorParams tensor_3d{
            .name = "tensor_3d", .data_type = tensor::TensorR32F, .rank = 3, .dims = {16, 32, 64}};

    // Access rank information
    const auto vec_rank = vector.rank;
    const auto mat_rank = matrix.rank;
    // example-end multi-rank-tensors-1

    EXPECT_EQ(vec_rank, 1);
    EXPECT_EQ(mat_rank, 2);
}

} // namespace

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
