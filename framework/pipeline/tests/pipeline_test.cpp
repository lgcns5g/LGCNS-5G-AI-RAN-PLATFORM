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

#include <any> // for any, bad_any_cast
#include <bit>
#include <cstddef>  // for size_t
#include <cstdint>  // for uintptr_t
#include <format>   // for format
#include <iostream> // for cout
#include <span>
#include <string> // for allocator, string
#include <string_view>
#include <typeinfo> // for type_info
#include <vector>   // for vector

#include <driver_types.h> // for CUstream_st, cudaS...

#include <gtest/gtest.h> // for Test, AssertionResult

#include <cuda/std/complex> // for complex

#include "pipeline/ipipeline.hpp" // for IPipeline
#include "pipeline/types.hpp"     // for ModuleMemoryRequir...
#include "tensor/data_types.hpp"  // for nv_get_data_type_s...
#include "tensor/tensor_info.hpp" // for tensor::TensorInfo

namespace {

using namespace framework::pipeline;
namespace tensor = framework::tensor;

// Test constants to avoid magic numbers
namespace test_constants {
constexpr std::size_t STATIC_KERNEL_BYTES = 1024;
constexpr std::size_t DYNAMIC_KERNEL_BYTES = 512;
constexpr std::size_t DEVICE_TENSOR_BYTES = 8192;
constexpr std::size_t NUM_INPUTS = 2;
constexpr std::size_t NUM_OUTPUTS = 1;
constexpr std::size_t TENSOR_DIM_32 = 32;

} // namespace test_constants

namespace test_helpers {
[[nodiscard]] void *to_ptr(std::uintptr_t addr) noexcept { return std::bit_cast<void *>(addr); }

[[nodiscard]] cudaStream_t to_cuda_stream(std::uintptr_t addr) noexcept {
    return std::bit_cast<cudaStream_t>(addr);
}
} // namespace test_helpers

/**
 * Mock implementation of IPipeline for testing interface functionality.
 * Uses std::cout << std::format to verify method invocation and demonstrate
 * interaction with types.hpp and tensor_info.hpp components.
 */
class MockPipeline final : public IPipeline {
public:
    explicit MockPipeline(const std::any &static_params = std::any{}) {
        std::cout << "MockPipeline::MockPipeline() called with static_params\n";

        // Demonstrate std::any usage from types.hpp pattern
        try {
            // Try to extract a PipelineModuleConfig as an example
            if (static_params.has_value()) {
                std::cout << std::format(
                        "  - static_params has value (type: {})\n", static_params.type().name());
                // In real implementation, would use std::any_cast<ExpectedType>
            } else {
                std::cout << "  - static_params is empty\n";
            }
        } catch (const std::bad_any_cast &e) {
            std::cout << std::format("  - bad_any_cast exception: {}\n", e.what());
        }

        initialized_ = true;
    }

    ~MockPipeline() override = default;
    // Explicitly defaulted copy/move operations
    MockPipeline(const MockPipeline &) = default;
    MockPipeline(MockPipeline &&) = default;
    MockPipeline &operator=(const MockPipeline &) = default;
    MockPipeline &operator=(MockPipeline &&) = default;

    [[nodiscard]] std::string_view get_pipeline_id() const override {
        std::cout << "MockPipeline::get_pipeline_id() called\n";
        return "mock_pipeline_test_v1.0";
    }

    void setup() override {
        std::cout << "MockPipeline::setup() called\n";

        // Demonstrate ModuleMemoryRequirements from types.hpp
        memory_requirements_.static_kernel_descriptor_bytes = test_constants::STATIC_KERNEL_BYTES;
        memory_requirements_.dynamic_kernel_descriptor_bytes = test_constants::DYNAMIC_KERNEL_BYTES;
        memory_requirements_.device_tensor_bytes = test_constants::DEVICE_TENSOR_BYTES;
        memory_requirements_.alignment = ModuleMemoryRequirements::DEFAULT_ALIGNMENT;

        std::cout << "  - Memory requirements configured:\n";
        std::cout << std::format(
                "    * static_kernel_descriptor_bytes: {}\n",
                memory_requirements_.static_kernel_descriptor_bytes);
        std::cout << std::format(
                "    * dynamic_kernel_descriptor_bytes: {}\n",
                memory_requirements_.dynamic_kernel_descriptor_bytes);
        std::cout << std::format(
                "    * device_tensor_bytes: {}\n", memory_requirements_.device_tensor_bytes);

        setup_called_ = true;
    }

    void configure_io(
            const DynamicParams &params,
            std::span<const PortInfo> external_inputs,
            std::span<PortInfo> external_outputs,
            [[maybe_unused]] cudaStream_t stream) override {
        std::cout << std::format("MockPipeline::configure_io() called\n");
        std::cout << std::format("  - external_inputs.size(): {}\n", external_inputs.size());
        std::cout << std::format("  - external_outputs.size(): {}\n", external_outputs.size());

        // Demonstrate DynamicParams usage from types.hpp
        if (params.module_specific_params.has_value()) {
            std::cout << std::format(
                    "  - module_specific_params has value (type: {})\n",
                    params.module_specific_params.type().name());
        } else {
            std::cout << std::format("  - module_specific_params is empty\n");
        }

        io_config_called_ = true;
    }

    void execute_stream(cudaStream_t stream) override {
        std::cout << std::format("MockPipeline::execute_stream() called\n");
        std::cout << std::format("  - stream pointer: {}\n", static_cast<void *>(stream));

        // Mock CUDA stream execution (no actual CUDA calls)
        if (stream != nullptr) {
            std::cout << std::format("  - Valid CUDA stream provided, executing mock kernels\n");
        } else {
            std::cout << std::format("  - Warning: null CUDA stream provided\n");
        }

        stream_execution_count_++;
    }

    void execute_graph(cudaStream_t stream) override {
        std::cout << std::format("MockPipeline::execute_graph() called\n");
        std::cout << std::format("  - stream pointer: {}\n", static_cast<void *>(stream));

        // Mock CUDA graph execution (no actual CUDA calls)
        if (stream != nullptr) {
            std::cout << std::format("  - Valid CUDA stream provided, launching mock CUDA graph\n");
        } else {
            std::cout << std::format("  - Warning: null CUDA stream provided\n");
        }

        graph_execution_count_++;
    }

    [[nodiscard]] std::size_t get_num_external_inputs() const override {
        std::cout << std::format("MockPipeline::get_num_external_inputs() called\n");
        std::cout << std::format("  - returning: {}\n", test_constants::NUM_INPUTS);
        return test_constants::NUM_INPUTS;
    }

    [[nodiscard]] std::size_t get_num_external_outputs() const override {
        std::cout << std::format("MockPipeline::get_num_external_outputs() called\n");
        std::cout << std::format("  - returning: {}\n", test_constants::NUM_OUTPUTS);
        return test_constants::NUM_OUTPUTS;
    }

    // Test helper methods
    [[nodiscard]] bool is_initialized() const { return initialized_; }
    [[nodiscard]] bool is_setup_called() const { return setup_called_; }
    [[nodiscard]] bool is_io_config_called() const { return io_config_called_; }
    [[nodiscard]] std::size_t get_stream_execution_count() const { return stream_execution_count_; }
    [[nodiscard]] std::size_t get_graph_execution_count() const { return graph_execution_count_; }
    // Test-only helper to access memory requirements (not part of IPipeline
    // interface)
    [[nodiscard]] ModuleMemoryRequirements get_memory_requirements_for_test() const {
        return memory_requirements_;
    }

private:
    bool initialized_{false};
    bool setup_called_{false};
    bool io_config_called_{false};
    std::size_t stream_execution_count_{0};
    std::size_t graph_execution_count_{0};
    ModuleMemoryRequirements memory_requirements_{};
};

// Test basic interface methods
TEST(PipelineTest, GetPipelineId_ReturnsValidId) {
    const MockPipeline pipeline;
    const std::string_view pipeline_id = pipeline.get_pipeline_id();

    EXPECT_FALSE(pipeline_id.empty());
    EXPECT_EQ("mock_pipeline_test_v1.0", pipeline_id);
}

TEST(PipelineTest, Init_WithEmptyParams_CallsInitialization) {
    const std::any empty_params;
    const MockPipeline pipeline(empty_params);

    EXPECT_TRUE(pipeline.is_initialized());
}

TEST(PipelineTest, Init_WithPipelineModuleConfig_CallsInitialization) {
    const PipelineModuleConfig config;
    const std::any params = config;
    const MockPipeline pipeline(params);

    EXPECT_TRUE(pipeline.is_initialized());
}

TEST(PipelineTest, Setup_CallsSetupSuccessfully) {
    MockPipeline pipeline;

    pipeline.setup();

    EXPECT_TRUE(pipeline.is_setup_called());
}

TEST(PipelineTest, ConfigureIO_WithDynamicParams_ConfiguresIO) {
    MockPipeline pipeline;
    DynamicParams params;
    params.module_specific_params = std::string("test_params");

    // Create PortInfo structures for external inputs and outputs
    std::vector<PortInfo> inputs(2);
    inputs[0].name = "input0";
    inputs[1].name = "input1";

    std::vector<PortInfo> outputs(1);
    outputs[0].name = "output0";

    pipeline.configure_io(params, inputs, outputs, nullptr);

    EXPECT_TRUE(pipeline.is_io_config_called());
}

TEST(PipelineTest, ExecuteStream_WithValidStream_IncrementsExecutionCount) {
    constexpr std::uintptr_t MOCK_STREAM_1 = 0x10000;

    MockPipeline pipeline;
    auto *mock_stream = test_helpers::to_cuda_stream(MOCK_STREAM_1);

    pipeline.execute_stream(mock_stream);

    EXPECT_EQ(1U, pipeline.get_stream_execution_count());
    EXPECT_EQ(0U, pipeline.get_graph_execution_count());
}

TEST(PipelineTest, ExecuteStream_WithNullStream_StillExecutes) {
    MockPipeline pipeline;
    cudaStream_t null_stream = nullptr;

    pipeline.execute_stream(null_stream);

    EXPECT_EQ(1U, pipeline.get_stream_execution_count());
}

TEST(PipelineTest, ExecuteGraph_WithValidStream_IncrementsExecutionCount) {
    constexpr std::uintptr_t MOCK_STREAM_2 = 0x20000;

    MockPipeline pipeline;
    auto *mock_stream = test_helpers::to_cuda_stream(MOCK_STREAM_2);

    pipeline.execute_graph(mock_stream);

    EXPECT_EQ(1U, pipeline.get_graph_execution_count());
    EXPECT_EQ(0U, pipeline.get_stream_execution_count());
}

TEST(PipelineTest, GetNumExternalInputs_ReturnsExpectedCount) {
    const MockPipeline pipeline;
    const std::size_t num_inputs = pipeline.get_num_external_inputs();

    EXPECT_EQ(test_constants::NUM_INPUTS, num_inputs);
}

TEST(PipelineTest, GetNumExternalOutputs_ReturnsExpectedCount) {
    const MockPipeline pipeline;
    const std::size_t num_outputs = pipeline.get_num_external_outputs();

    EXPECT_EQ(test_constants::NUM_OUTPUTS, num_outputs);
}

// Memory requirements testing removed - no longer part of IPipeline interface
// (now handled internally by pipeline implementations)

// Test demonstrating interaction with tensor_info.hpp types
TEST(PipelineTest, TensorInfoIntegration_CreateAndVerifyTensorInfo) {
    constexpr std::size_t TENSOR_DIM_64 = 64;
    constexpr std::size_t TENSOR_DIM_128 = 128;

    auto data_type = tensor::NvDataType::TensorR32F;
    std::vector<std::size_t> dimensions = {
            test_constants::TENSOR_DIM_32, TENSOR_DIM_64, TENSOR_DIM_128};

    tensor::TensorInfo tensor_info(data_type, dimensions);
    tensor_info.set_size_bytes(
            test_constants::TENSOR_DIM_32 * TENSOR_DIM_64 * TENSOR_DIM_128 * sizeof(float));

    EXPECT_EQ(data_type, tensor_info.get_type());
    EXPECT_EQ(dimensions, tensor_info.get_dimensions());
    EXPECT_EQ(
            test_constants::TENSOR_DIM_32 * TENSOR_DIM_64 * TENSOR_DIM_128,
            tensor_info.get_total_elements());
    EXPECT_EQ(
            test_constants::TENSOR_DIM_32 * TENSOR_DIM_64 * TENSOR_DIM_128 * sizeof(float),
            tensor_info.get_total_size_in_bytes());

    std::cout << std::format("tensor::TensorInfo integration test:\n");
    std::cout << std::format("  - Type: {}\n", nv_get_data_type_string(tensor_info.get_type()));
    std::cout << std::format(
            "  - Dimensions: [{}, {}, {}]\n", dimensions[0], dimensions[1], dimensions[2]);
    std::cout << std::format("  - Total elements: {}\n", tensor_info.get_total_elements());
    std::cout << std::format("  - Total size: {} bytes\n", tensor_info.get_total_size_in_bytes());
}

// Test demonstrating types.hpp data structures
TEST(PipelineTest, TypesIntegration_DemostratePortInfoUsage) {
    constexpr std::uintptr_t MOCK_DEVICE_PTR = 0x40000;
    constexpr std::size_t TENSOR_DIM_16 = 16;

    tensor::TensorInfo tensor_info(
            tensor::NvDataType::TensorC32F, {TENSOR_DIM_16, test_constants::TENSOR_DIM_32});
    tensor_info.set_size_bytes(
            TENSOR_DIM_16 * test_constants::TENSOR_DIM_32 * sizeof(cuda::std::complex<float>));

    PortInfo input_port;
    input_port.name = "complex_input_matrix";
    input_port.tensors.push_back(DeviceTensor{
            .device_ptr = test_helpers::to_ptr(MOCK_DEVICE_PTR), .tensor_info = tensor_info});

    EXPECT_EQ("complex_input_matrix", input_port.name);
    EXPECT_EQ(1, input_port.tensors.size());
    EXPECT_NE(nullptr, input_port.tensors[0].device_ptr);
    EXPECT_TRUE(input_port.tensors[0].tensor_info.is_compatible_with(tensor_info));

    std::cout << std::format("PortInfo integration test:\n");
    std::cout << std::format("  - Port name: {}\n", input_port.name);
    std::cout << std::format("  - Number of tensors: {}\n", input_port.tensors.size());
    std::cout << std::format("  - Device pointer: {}\n", input_port.tensors[0].device_ptr);
    std::cout << std::format(
            "  - Tensor type: {}\n",
            nv_get_data_type_string(input_port.tensors[0].tensor_info.get_type()));
    std::cout << std::format(
            "  - Tensor dimensions: [{}, {}]\n",
            input_port.tensors[0].tensor_info.get_dimensions()[0],
            input_port.tensors[0].tensor_info.get_dimensions()[1]);
}

// Integration test combining multiple components
TEST(PipelineTest, FullWorkflow_InitSetupExecute_CompletesSuccessfully) {
    constexpr std::uintptr_t MOCK_STREAM_WORKFLOW = 0x30000;

    // Create a PipelineModuleConfig with ModuleSpec
    PipelineModuleConfig config;
    ModuleCreationInfo module_creation_info;
    module_creation_info.module_type = "gemm";
    module_creation_info.instance_id = "gemm_0";
    module_creation_info.init_params = std::string("gemm_config_params");

    const ModuleSpec module_spec{module_creation_info};
    config.modules.push_back(module_spec);

    const std::any init_params = config;

    // Initialize pipeline with static parameters in constructor
    MockPipeline pipeline(init_params);

    DynamicParams params;
    params.module_specific_params = std::vector<int>{1, 2, 3, 4};

    // Create PortInfo structures for external inputs and outputs
    std::vector<PortInfo> inputs(2);
    inputs[0].name = "input0";
    inputs[1].name = "input1";

    std::vector<PortInfo> outputs(1);
    outputs[0].name = "output0";

    auto *mock_stream = test_helpers::to_cuda_stream(MOCK_STREAM_WORKFLOW);
    pipeline.setup();
    pipeline.configure_io(params, inputs, outputs, nullptr);
    pipeline.execute_stream(mock_stream);
    pipeline.execute_graph(mock_stream);

    EXPECT_TRUE(pipeline.is_initialized());
    EXPECT_TRUE(pipeline.is_setup_called());
    EXPECT_TRUE(pipeline.is_io_config_called());
    EXPECT_EQ(1U, pipeline.get_stream_execution_count());
    EXPECT_EQ(1U, pipeline.get_graph_execution_count());
    EXPECT_EQ(test_constants::NUM_INPUTS, pipeline.get_num_external_inputs());
    EXPECT_EQ(test_constants::NUM_OUTPUTS, pipeline.get_num_external_outputs());
}

} // namespace
