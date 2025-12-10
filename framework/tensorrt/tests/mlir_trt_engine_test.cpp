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

#include <bit>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <NvInfer.h>
#include <driver_types.h>

#include <gtest/gtest.h>

#include "tensor/data_types.hpp"
#include "tensorrt/mlir_trt_engine.hpp"
#include "tensorrt/trt_engine_interface.hpp"
#include "tensorrt/trt_engine_interfaces.hpp"
#include "tensorrt/trt_engine_params.hpp"
#include "utils/errors.hpp"

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

namespace framework::pipeline::tests {

using framework::tensor::TensorR32F;
using framework::tensorrt::IPrePostTrtEngEnqueue;
using framework::tensorrt::ITrtEngine;
using framework::tensorrt::MLIRTensorParams;
using framework::tensorrt::MLIRTrtEngine;
using framework::utils::NvErrc;

// Test constants for magic number elimination
namespace {
constexpr std::uintptr_t TEST_STREAM_ADDR = 0x1234;
constexpr std::uintptr_t INPUT_BUFFER_ADDR_1 = 0x1000;
constexpr std::uintptr_t INPUT_BUFFER_ADDR_2 = 0x2000;
constexpr std::uintptr_t OUTPUT_BUFFER_ADDR_1 = 0x3000;

} // anonymous namespace

/// Creates a mock pointer from an address for testing purposes
[[nodiscard]] void *mock_ptr(std::uintptr_t address) noexcept {
    return std::bit_cast<void *>(address);
}

/// Creates a mock CUDA stream from an address for testing purposes
[[nodiscard]] cudaStream_t mock_stream(std::uintptr_t address) noexcept {
    return std::bit_cast<cudaStream_t>(address);
}

/**
 * @brief Test implementation of ITrtEngine for unit testing
 */
class TestTrtEngine final : public ITrtEngine {
public:
    [[nodiscard]]
    NvErrc set_input_shape(
            [[maybe_unused]] const std::string_view tensor_name,
            [[maybe_unused]] const nvinfer1::Dims &dims) final {
        set_input_shape_called_ = true;
        last_tensor_name_ = std::string(tensor_name);
        return return_value_;
    }

    [[nodiscard]]
    NvErrc set_tensor_address(
            [[maybe_unused]] const std::string_view tensor_name,
            [[maybe_unused]] void *address) final {
        set_tensor_address_called_ = true;
        last_tensor_name_ = std::string(tensor_name);
        last_address_ = address;
        return return_value_;
    }

    [[nodiscard]]
    NvErrc enqueue_inference([[maybe_unused]] cudaStream_t cu_stream) final {
        enqueue_inference_called_ = true;
        return return_value_;
    }

    [[nodiscard]]
    bool all_input_dimensions_specified() const final {
        return all_dimensions_specified_;
    }

    // Test control methods
    void set_return_value(NvErrc value) { return_value_ = value; }
    void set_all_dimensions_specified(bool value) { all_dimensions_specified_ = value; }

    // Query methods for verification
    [[nodiscard]] bool was_set_input_shape_called() const { return set_input_shape_called_; }
    [[nodiscard]] bool was_set_tensor_address_called() const { return set_tensor_address_called_; }
    [[nodiscard]] bool was_enqueue_inference_called() const { return enqueue_inference_called_; }
    [[nodiscard]] const std::string &get_last_tensor_name() const { return last_tensor_name_; }
    [[nodiscard]] void *get_last_address() const { return last_address_; }

private:
    NvErrc return_value_ = NvErrc::Success;
    bool all_dimensions_specified_ = true;
    bool set_input_shape_called_ = false;
    bool set_tensor_address_called_ = false;
    bool enqueue_inference_called_ = false;
    std::string last_tensor_name_;
    void *last_address_ = nullptr;
};

/**
 * @brief Test implementation of IPrePostTrtEngEnqueue for unit testing
 */
class TestPrePostTrtEngEnqueue final : public IPrePostTrtEngEnqueue {
public:
    [[nodiscard]]
    NvErrc pre_enqueue([[maybe_unused]] cudaStream_t cu_stream) final {
        pre_enqueue_called_ = true;
        return return_value_;
    }

    [[nodiscard]]
    NvErrc post_enqueue([[maybe_unused]] cudaStream_t cu_stream) final {
        post_enqueue_called_ = true;
        return return_value_;
    }

    // Test control methods
    void set_return_value(NvErrc value) { return_value_ = value; }

    // Query methods for verification
    [[nodiscard]] bool was_pre_enqueue_called() const { return pre_enqueue_called_; }
    [[nodiscard]] bool was_post_enqueue_called() const { return post_enqueue_called_; }

private:
    NvErrc return_value_ = NvErrc::Success;
    bool pre_enqueue_called_ = false;
    bool post_enqueue_called_ = false;
};

/**
 * @brief Test fixture for MLIRTrtEngine tests
 */
class MLIRTrtEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test engine
        auto test_runtime = std::make_unique<TestTrtEngine>();
        test_runtime_ptr_ = test_runtime.get();
        test_runtime_ptr_->set_all_dimensions_specified(true);

        // Create test tensor parameters with rank and dims
        input_params_ = {
                {.name = "input0",
                 .data_type = TensorR32F,
                 .rank = 2,
                 .dims = {1024, 768}}, // Strides auto-computed: [768, 1]
                {.name = "input1",
                 .data_type = TensorR32F,
                 .rank = 1,
                 .dims = {512}} // Strides auto-computed: [1]
        };

        output_params_ = {
                {.name = "output0",
                 .data_type = TensorR32F,
                 .rank = 2,
                 .dims = {1024, 512}}}; // Strides auto-computed: [512, 1]

        // Create fake CUDA stream and data buffer pointers (not descriptor
        // pointers)
        test_stream_ = mock_stream(TEST_STREAM_ADDR);
        input_buffers_ = {mock_ptr(INPUT_BUFFER_ADDR_1), mock_ptr(INPUT_BUFFER_ADDR_2)};
        output_buffers_ = {mock_ptr(OUTPUT_BUFFER_ADDR_1)};

        // Store the runtime pointer before moving it
        stored_runtime_ = std::move(test_runtime);
    }

    std::unique_ptr<TestTrtEngine> stored_runtime_;
    TestTrtEngine *test_runtime_ptr_{}; // Non-owning pointer for verification

    std::vector<MLIRTensorParams> input_params_;
    std::vector<MLIRTensorParams> output_params_;
    cudaStream_t test_stream_{};
    std::vector<void *> input_buffers_;
    std::vector<void *> output_buffers_;
};

TEST_F(MLIRTrtEngineTest, ConstructorValid) {
    EXPECT_NO_THROW({
        const MLIRTrtEngine engine(input_params_, output_params_, std::move(stored_runtime_));
    });
}

TEST_F(MLIRTrtEngineTest, ConstructorNullRuntime) {
    EXPECT_THROW(
            { const MLIRTrtEngine engine(input_params_, output_params_, nullptr); },
            std::invalid_argument);
}

TEST_F(MLIRTrtEngineTest, ConstructorEmptyInputs) {
    const std::vector<MLIRTensorParams> empty_inputs;
    EXPECT_THROW(
            {
                const MLIRTrtEngine engine(
                        empty_inputs, output_params_, std::move(stored_runtime_));
            },
            std::invalid_argument);
}

TEST_F(MLIRTrtEngineTest, ConstructorEmptyOutputs) {
    const std::vector<MLIRTensorParams> empty_outputs;
    EXPECT_THROW(
            {
                const MLIRTrtEngine engine(
                        input_params_, empty_outputs, std::move(stored_runtime_));
            },
            std::invalid_argument);
}

TEST_F(MLIRTrtEngineTest, ConstructorEmptyTensorName) {
    input_params_[0].name = ""; // Empty name
    EXPECT_THROW(
            {
                const MLIRTrtEngine engine(
                        input_params_, output_params_, std::move(stored_runtime_));
            },
            std::invalid_argument);
}

TEST_F(MLIRTrtEngineTest, ConstructorScalarRank) {
    input_params_[0].rank = 0; // Scalar rank (valid)
    EXPECT_NO_THROW({
        const MLIRTrtEngine engine(input_params_, output_params_, std::move(stored_runtime_));
    });
}

TEST_F(MLIRTrtEngineTest, ConstructorRankExceedsMax) {
    input_params_[0].rank = 99; // Exceeds MAX_TENSOR_RANK
    EXPECT_THROW(
            {
                const MLIRTrtEngine engine(
                        input_params_, output_params_, std::move(stored_runtime_));
            },
            std::invalid_argument);
}

TEST_F(MLIRTrtEngineTest, SetupValid) {
    MLIRTrtEngine engine(input_params_, output_params_, std::move(stored_runtime_));

    EXPECT_EQ(NvErrc::Success, engine.setup(input_buffers_, output_buffers_));
}

TEST_F(MLIRTrtEngineTest, SetupMismatchedInputCount) {
    MLIRTrtEngine engine(input_params_, output_params_, std::move(stored_runtime_));

    const std::vector<void *> wrong_input_buffers = {
            mock_ptr(INPUT_BUFFER_ADDR_1)}; // Only 1 buffer, need 2
    EXPECT_EQ(NvErrc::InvalidArgument, engine.setup(wrong_input_buffers, output_buffers_));
}

TEST_F(MLIRTrtEngineTest, SetupMismatchedOutputCount) {
    MLIRTrtEngine engine(input_params_, output_params_, std::move(stored_runtime_));

    const std::vector<void *> wrong_output_buffers = {
            mock_ptr(OUTPUT_BUFFER_ADDR_1),
            mock_ptr(0x4000) // Test mock address
    }; // 2 buffers, need 1
    EXPECT_EQ(NvErrc::InvalidArgument, engine.setup(input_buffers_, wrong_output_buffers));
}

TEST_F(MLIRTrtEngineTest, WarmupWithoutSetup) {
    MLIRTrtEngine engine(input_params_, output_params_, std::move(stored_runtime_));

    EXPECT_EQ(NvErrc::InternalError, engine.warmup(test_stream_));
}

TEST_F(MLIRTrtEngineTest, RunWithoutSetup) {
    const MLIRTrtEngine engine(input_params_, output_params_, std::move(stored_runtime_));

    EXPECT_EQ(NvErrc::InternalError, engine.run(test_stream_));
}

TEST_F(MLIRTrtEngineTest, RunSuccess) {
    MLIRTrtEngine engine(input_params_, output_params_, std::move(stored_runtime_));

    EXPECT_EQ(NvErrc::Success, engine.setup(input_buffers_, output_buffers_));
    EXPECT_EQ(NvErrc::Success, engine.run(test_stream_));

    // Verify that tensor addresses were set with user-provided names
    EXPECT_TRUE(test_runtime_ptr_->was_set_tensor_address_called());
    EXPECT_TRUE(test_runtime_ptr_->was_set_input_shape_called());
    EXPECT_TRUE(test_runtime_ptr_->was_enqueue_inference_called());
}

// Scalar tensor tests
TEST_F(MLIRTrtEngineTest, ScalarInputTensor) {
    // Create scalar input tensor (rank 0)
    const std::vector<MLIRTensorParams> scalar_input = {
            {.name = "scalar_input", .data_type = TensorR32F, .rank = 0, .dims = {}}};

    const std::vector<void *> scalar_input_buffers = {mock_ptr(INPUT_BUFFER_ADDR_1)};

    EXPECT_NO_THROW({
        const MLIRTrtEngine engine(scalar_input, output_params_, std::move(stored_runtime_));
    });
}

TEST_F(MLIRTrtEngineTest, ScalarOutputTensor) {
    // Create scalar output tensor (rank 0)
    const std::vector<MLIRTensorParams> scalar_output = {
            {.name = "scalar_output", .data_type = TensorR32F, .rank = 0, .dims = {}}};

    const std::vector<void *> scalar_output_buffers = {mock_ptr(OUTPUT_BUFFER_ADDR_1)};

    EXPECT_NO_THROW({
        const MLIRTrtEngine engine(input_params_, scalar_output, std::move(stored_runtime_));
    });
}

TEST_F(MLIRTrtEngineTest, MixedScalarAndVectorTensors) {
    // Mix scalar and vector tensors
    const std::vector<MLIRTensorParams> mixed_inputs = {
            {.name = "scalar_input", .data_type = TensorR32F, .rank = 0, .dims = {}},
            {.name = "vector_input", .data_type = TensorR32F, .rank = 1, .dims = {512}}};

    const std::vector<void *> mixed_input_buffers = {
            mock_ptr(INPUT_BUFFER_ADDR_1), mock_ptr(INPUT_BUFFER_ADDR_2)};

    EXPECT_NO_THROW({
        const MLIRTrtEngine engine(mixed_inputs, output_params_, std::move(stored_runtime_));
    });
}

TEST_F(MLIRTrtEngineTest, ScalarTensorRunSuccess) {
    // Create engine with scalar input
    const std::vector<MLIRTensorParams> scalar_input = {
            {.name = "scalar_input", .data_type = TensorR32F, .rank = 0, .dims = {}}};

    const std::vector<void *> scalar_input_buffers = {mock_ptr(INPUT_BUFFER_ADDR_1)};

    MLIRTrtEngine engine(scalar_input, output_params_, std::move(stored_runtime_));

    EXPECT_EQ(NvErrc::Success, engine.setup(scalar_input_buffers, output_buffers_));
    EXPECT_EQ(NvErrc::Success, engine.run(test_stream_));

    // Verify that tensor operations were called
    EXPECT_TRUE(test_runtime_ptr_->was_set_tensor_address_called());
    EXPECT_TRUE(test_runtime_ptr_->was_set_input_shape_called());
    EXPECT_TRUE(test_runtime_ptr_->was_enqueue_inference_called());
}

} // namespace framework::pipeline::tests

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
