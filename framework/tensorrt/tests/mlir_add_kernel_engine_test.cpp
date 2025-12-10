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

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <driver_types.h>
#include <quill/LogMacros.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include "log/rt_log_macros.hpp"
#include "tensor/data_types.hpp"
#include "tensorrt/mlir_trt_engine.hpp"
#include "tensorrt/trt_engine.hpp"
#include "tensorrt/trt_engine_interface.hpp"
#include "tensorrt/trt_engine_logger.hpp"
#include "tensorrt/trt_engine_params.hpp"
#include "utils/core_log.hpp"
#include "utils/errors.hpp"

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

namespace framework::pipeline::tests {

using framework::tensor::TensorR32F;
using framework::tensorrt::MLIRTensorParams;
using framework::tensorrt::MLIRTrtEngine;
using framework::tensorrt::TrtEngine;
using framework::tensorrt::TrtLogger;
using framework::utils::Core;
using framework::utils::NvErrc;

namespace {

// Test constants
constexpr std::size_t TENSOR_SIZE = 16384;
constexpr std::size_t TENSOR_RANK = 1;
constexpr float FLOAT_TOLERANCE = 1e-5F;

/// Path to the compiled TensorRT engine file
constexpr std::string_view ENGINE_FILE_PATH =
        "framework/pipeline/samples/engines/tensorrt_cluster_engine_data.trtengine";

/**
 * @brief Read a binary file into a byte vector
 * @param[in] file_path Path to the file to read
 * @return Vector containing file contents, empty on error
 */
std::vector<std::byte> read_file_to_bytes(const std::string_view file_path) {
    std::ifstream file(std::string(file_path), std::ios::binary | std::ios::ate);
    if (!file) {
        return {};
    }

    const std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> temp_buffer(static_cast<std::size_t>(size));
    if (!file.read(temp_buffer.data(), size)) {
        return {};
    }

    std::vector<std::byte> buffer(static_cast<std::size_t>(size));
    std::memcpy(buffer.data(), temp_buffer.data(), static_cast<std::size_t>(size));

    return buffer;
}

} // anonymous namespace

/**
 * @brief Test fixture for MLIR TensorRT add kernel engine tests
 */
class MLIRAddKernelEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Check if CUDA device is available
        int device_count = 0;
        const cudaError_t error = cudaGetDeviceCount(&device_count);
        if (error != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "CUDA device not available: " << cudaGetErrorString(error);
        }

        // Initialize CUDA (required for device operations)
        ASSERT_EQ(cudaSuccess, cudaSetDevice(0));

        // Load engine file
        engine_data_ = read_file_to_bytes(ENGINE_FILE_PATH);
        ASSERT_FALSE(engine_data_.empty()) << "Failed to load engine file: " << ENGINE_FILE_PATH;

        // Create tensor parameters matching the compiled engine
        // Input tensors: arg0, arg1 (both rank-1, 16384 float32 elements)
        // Output tensor: result0 (rank-1, 16384 float32 elements)
        input_params_ = {
                {.name = "arg0",
                 .data_type = TensorR32F,
                 .rank = TENSOR_RANK,
                 .dims = {TENSOR_SIZE}},
                {.name = "arg1",
                 .data_type = TensorR32F,
                 .rank = TENSOR_RANK,
                 .dims = {TENSOR_SIZE}}};

        output_params_ = {
                {.name = "result0",
                 .data_type = TensorR32F,
                 .rank = TENSOR_RANK,
                 .dims = {TENSOR_SIZE}}};

        // Allocate device memory for tensors
        const std::size_t tensor_bytes = TENSOR_SIZE * sizeof(float);
        ASSERT_EQ(cudaSuccess, cudaMalloc(&input0_dev_, tensor_bytes));
        ASSERT_EQ(cudaSuccess, cudaMalloc(&input1_dev_, tensor_bytes));
        ASSERT_EQ(cudaSuccess, cudaMalloc(&output_dev_, tensor_bytes));

        // Allocate and initialize host memory
        input0_host_.resize(TENSOR_SIZE);
        input1_host_.resize(TENSOR_SIZE);
        output_host_.resize(TENSOR_SIZE);

        // Initialize input data with test pattern (similar to add_driver.cpp)
        for (std::size_t i = 0; i < TENSOR_SIZE; ++i) {
            input0_host_[i] = static_cast<float>(i) + 4.0F;
            input1_host_[i] = static_cast<float>(i + 1) * 2.0F;
        }

        // Copy input data to device
        ASSERT_EQ(
                cudaSuccess,
                cudaMemcpy(input0_dev_, input0_host_.data(), tensor_bytes, cudaMemcpyHostToDevice));
        ASSERT_EQ(
                cudaSuccess,
                cudaMemcpy(input1_dev_, input1_host_.data(), tensor_bytes, cudaMemcpyHostToDevice));

        // Setup buffer pointer vectors
        input_buffers_ = {input0_dev_, input1_dev_};
        output_buffers_ = {output_dev_};

        // Create CUDA events for timing
        ASSERT_EQ(cudaSuccess, cudaEventCreate(&start_event_));
        ASSERT_EQ(cudaSuccess, cudaEventCreate(&stop_event_));
    }

    void TearDown() override {
        // Free device memory
        if (input0_dev_ != nullptr) {
            cudaFree(input0_dev_);
        }
        if (input1_dev_ != nullptr) {
            cudaFree(input1_dev_);
        }
        if (output_dev_ != nullptr) {
            cudaFree(output_dev_);
        }

        // Destroy CUDA events
        if (start_event_ != nullptr) {
            cudaEventDestroy(start_event_);
        }
        if (stop_event_ != nullptr) {
            cudaEventDestroy(stop_event_);
        }

        // Reset device to ensure clean state
        cudaDeviceReset();
    }

    /// Verify that output equals input0 + input1 element-wise
    void verify_add_results() {
        const std::size_t tensor_bytes = TENSOR_SIZE * sizeof(float);
        ASSERT_EQ(
                cudaSuccess,
                cudaMemcpy(output_host_.data(), output_dev_, tensor_bytes, cudaMemcpyDeviceToHost));

        for (std::size_t i = 0; i < TENSOR_SIZE; ++i) {
            const float expected = input0_host_[i] + input1_host_[i];
            const float actual = output_host_[i];
            EXPECT_NEAR(expected, actual, FLOAT_TOLERANCE) << "Mismatch at index " << i;
        }
    }

    // Engine data and parameters
    std::vector<std::byte> engine_data_;
    std::vector<MLIRTensorParams> input_params_;
    std::vector<MLIRTensorParams> output_params_;

    // CUDA device memory
    void *input0_dev_ = nullptr;
    void *input1_dev_ = nullptr;
    void *output_dev_ = nullptr;

    // Host memory
    std::vector<float> input0_host_;
    std::vector<float> input1_host_;
    std::vector<float> output_host_;

    // Buffer pointer vectors
    std::vector<void *> input_buffers_;
    std::vector<void *> output_buffers_;

    // CUDA events for timing
    cudaEvent_t start_event_ = nullptr;
    cudaEvent_t stop_event_ = nullptr;
};

TEST_F(MLIRAddKernelEngineTest, DISABLED_LoadEngineSuccess) {
    TrtLogger logger;
    EXPECT_NO_THROW({ const TrtEngine runtime(engine_data_, logger); });
}

TEST_F(MLIRAddKernelEngineTest, DISABLED_ConstructEngineSuccess) {
    TrtLogger logger;
    auto runtime = std::make_unique<TrtEngine>(engine_data_, logger);

    EXPECT_NO_THROW(
            { const MLIRTrtEngine engine(input_params_, output_params_, std::move(runtime)); });
}

TEST_F(MLIRAddKernelEngineTest, DISABLED_SetupWithCorrectBuffers) {
    TrtLogger logger;
    auto runtime = std::make_unique<TrtEngine>(engine_data_, logger);
    MLIRTrtEngine engine(input_params_, output_params_, std::move(runtime));

    EXPECT_EQ(NvErrc::Success, engine.setup(input_buffers_, output_buffers_));
}

TEST_F(MLIRAddKernelEngineTest, DISABLED_SetupWithWrongInputCount) {
    TrtLogger logger;
    auto runtime = std::make_unique<TrtEngine>(engine_data_, logger);
    MLIRTrtEngine engine(input_params_, output_params_, std::move(runtime));

    const std::vector<void *> wrong_inputs = {input0_dev_}; // Only 1, need 2
    EXPECT_EQ(NvErrc::InvalidArgument, engine.setup(wrong_inputs, output_buffers_));
}

TEST_F(MLIRAddKernelEngineTest, DISABLED_SetupWithWrongOutputCount) {
    TrtLogger logger;
    auto runtime = std::make_unique<TrtEngine>(engine_data_, logger);
    MLIRTrtEngine engine(input_params_, output_params_, std::move(runtime));

    const std::vector<void *> wrong_outputs = {output_dev_, input0_dev_}; // 2 buffers, need 1
    EXPECT_EQ(NvErrc::InvalidArgument, engine.setup(input_buffers_, wrong_outputs));
}

TEST_F(MLIRAddKernelEngineTest, DISABLED_WarmupExecution) {
    TrtLogger logger;
    auto runtime = std::make_unique<TrtEngine>(engine_data_, logger);
    MLIRTrtEngine engine(input_params_, output_params_, std::move(runtime));

    ASSERT_EQ(NvErrc::Success, engine.setup(input_buffers_, output_buffers_));
    EXPECT_EQ(NvErrc::Success, engine.warmup(cudaStreamDefault));
}

/**
 * @brief Test full inference execution with result validation and timing
 */
TEST_F(MLIRAddKernelEngineTest, DISABLED_InferenceAndValidation) {
    TrtLogger logger;
    auto runtime = std::make_unique<TrtEngine>(engine_data_, logger);
    MLIRTrtEngine engine(input_params_, output_params_, std::move(runtime));

    // Setup engine
    ASSERT_EQ(NvErrc::Success, engine.setup(input_buffers_, output_buffers_));

    // Warmup phase
    ASSERT_EQ(NvErrc::Success, engine.warmup(cudaStreamDefault));

    // Timed inference run
    ASSERT_EQ(cudaSuccess, cudaEventRecord(start_event_, cudaStreamDefault));
    ASSERT_EQ(NvErrc::Success, engine.run(cudaStreamDefault));
    ASSERT_EQ(cudaSuccess, cudaEventRecord(stop_event_, cudaStreamDefault));
    ASSERT_EQ(cudaSuccess, cudaEventSynchronize(stop_event_));

    // Calculate and print timing
    float elapsed_ms = 0.0F;
    ASSERT_EQ(cudaSuccess, cudaEventElapsedTime(&elapsed_ms, start_event_, stop_event_));
    RT_LOGC_DEBUG(Core::CoreNvApi, "Inference time: {} ms", elapsed_ms);

    // Verify results
    verify_add_results();
}

TEST_F(MLIRAddKernelEngineTest, DISABLED_RunWithoutSetup) {
    TrtLogger logger;
    auto runtime = std::make_unique<TrtEngine>(engine_data_, logger);
    const MLIRTrtEngine engine(input_params_, output_params_, std::move(runtime));

    EXPECT_EQ(NvErrc::InternalError, engine.run(cudaStreamDefault));
}

TEST_F(MLIRAddKernelEngineTest, DISABLED_WarmupWithoutSetup) {
    TrtLogger logger;
    auto runtime = std::make_unique<TrtEngine>(engine_data_, logger);
    MLIRTrtEngine engine(input_params_, output_params_, std::move(runtime));

    EXPECT_EQ(NvErrc::InternalError, engine.warmup(cudaStreamDefault));
}

} // namespace framework::pipeline::tests

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
