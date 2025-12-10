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
#include <any>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <driver_types.h>

#include <gsl-lite/gsl-lite.hpp>
#include <gtest/gtest.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include "aerial_tv/aerial_tv_utils.hpp"
#include "aerial_tv/cuphy_pusch_tv.hpp"
#include "ldpc/ldpc_decoder_module.hpp"
#include "ldpc/ldpc_params.hpp"
#include "ldpc/outer_rx_params.hpp"
#include "log/components.hpp"
#include "log/rt_log_macros.hpp"
#include "memory/unique_ptr_utils.hpp"
#include "pipeline/graph_manager.hpp"
#include "pipeline/igraph_node_provider.hpp"
#include "pipeline/types.hpp"
#include "ran_common.hpp"
#include "tensor/data_types.hpp"
#include "tensor/tensor_info.hpp"
#include "utils/core_log.hpp"
#include "utils/cuda_stream.hpp"

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

namespace {

using ran::ldpc::LdpcDecoderModule;
namespace pipeline = framework::pipeline;
namespace tensor = framework::tensor;
using framework::memory::make_unique_device;
using framework::memory::UniqueDevicePtr;
using framework::utils::CudaStream;

constexpr std::size_t BITS_PER_BYTE = 8;

const LdpcDecoderModule::StaticParams STATIC_PARAMS{
        .clamp_value = ran::ldpc::LDPC_CLAMP_VALUE,
        .max_num_iterations = ran::ldpc::LDPC_MAX_ITERATIONS,
        .max_num_cbs_per_tb = ran::common::MAX_NUM_CBS_PER_TB,
        .max_num_tbs = ran::common::MAX_NUM_TBS,
        .normalization_factor = ran::ldpc::LDPC_NORMALIZATION_FACTOR,
        .max_iterations_method = ran::ldpc::LdpcMaxIterationsMethod::Fixed,
        .max_num_ldpc_het_configs = ran::ldpc::LDPC_MAX_HET_CONFIGS};

/**
 * Test fixture for LdpcDecoderModule
 *
 * Sets up common test infrastructure including CUDA context,
 * memory management, and H5 file handling for test vectors.
 */
class LdpcDecoderModuleTest : public ::testing::Test {
protected:
    void SetUp() override {
        ldpc_decoder_module_ =
                std::make_unique<LdpcDecoderModule>("ldpc_decoder_instance", STATIC_PARAMS);
    }

    std::unique_ptr<LdpcDecoderModule> ldpc_decoder_module_;
};

/**
 * Parameterized test fixture for LdpcDecoderModule with H5 test vector files
 *
 * Tests the module with different H5 test vector files to ensure
 * robustness across various input scenarios.
 */
class LdpcDecoderModuleH5Test : public LdpcDecoderModuleTest,
                                public ::testing::WithParamInterface<std::string_view> {
protected:
    void SetUp() override {
        LdpcDecoderModuleTest::SetUp();

        // Get the test vector path from compile definition and form full file path
        const std::filesystem::path test_vector_dir{TEST_VECTOR_PATH};
        const std::filesystem::path full_file_path = test_vector_dir / GetParam();

        // Check if the file exists and fail the test directly if not
        if (!std::filesystem::exists(full_file_path)) {
            FAIL() << "Test HDF5 file not found: " << full_file_path.string()
                   << "\nExpected location: " << test_vector_dir.string();
        }

        // Store the full path for use in tests
        test_file_path_ = full_file_path.string();
    }

    /**
     * Get the full path to the test vector file
     */
    [[nodiscard]] const std::string &get_test_file_path() const { return test_file_path_; }

private:
    std::string test_file_path_;
};

/**
 * Test successful module identification.
 */
TEST_F(LdpcDecoderModuleTest, ModuleIdentification) {

    // Test type ID
    EXPECT_STREQ(ldpc_decoder_module_->get_type_id().data(), "ldpc_decoder_module");

    // Instance ID should be empty before init
    EXPECT_STREQ(ldpc_decoder_module_->get_instance_id().data(), "ldpc_decoder_instance");
}

/**
 * Verifies module correctly reports its input and output port names.
 */
TEST_F(LdpcDecoderModuleTest, PortNames) {

    // Test input port names
    const auto input_ports = ldpc_decoder_module_->get_input_port_names();
    EXPECT_EQ(input_ports.size(), 1);
    EXPECT_EQ(input_ports[0], "llrs");

    // Test output port names
    const auto output_ports = ldpc_decoder_module_->get_output_port_names();
    EXPECT_EQ(output_ports.size(), 1);
    EXPECT_EQ(output_ports[0], "decoded_bits");
}

/**
 * Test setting inputs and verifying them with getInputTensorInfo.
 */
TEST_F(LdpcDecoderModuleTest, SetInputsAndVerify) {

    // First, we need to call setup to complete module setup
    pipeline::ModuleMemorySlice memory_slice{};
    // Use nullptr for device tensor pointer - not needed for this test
    memory_slice.device_tensor_ptr = nullptr;

    ldpc_decoder_module_->setup_memory(memory_slice);

    // Create input tensor info - FLOAT16 with 2D dimensions
    const std::vector<std::size_t> input_dimensions{
            1000, ran::common::MAX_NUM_TBS}; // Example dimensions
    const tensor::TensorInfo input_tensor_info{tensor::NvDataType::TensorR16F, input_dimensions};

    // Create PortInfo for the input - use dummy non-null pointer to pass
    // validation
    const pipeline::PortInfo input_port{
            .name = "llrs",

            .tensors = {pipeline::DeviceTensor{
                    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
                    .device_ptr = reinterpret_cast<void *>(0x1),
                    .tensor_info = input_tensor_info}}};

    // Set the inputs
    const std::vector<pipeline::PortInfo> inputs{input_port};
    ASSERT_NO_THROW(ldpc_decoder_module_->set_inputs(inputs));

    // Verify the input was set correctly using get_input_tensor_info
    const tensor::TensorInfo retrieved_tensor_info =
            ldpc_decoder_module_->get_input_tensor_info("llrs")[0];

    // Verify the tensor info matches what we set
    EXPECT_EQ(retrieved_tensor_info.get_type(), tensor::NvDataType::TensorR16F);
    const auto &retrieved_dimensions = retrieved_tensor_info.get_dimensions();
    EXPECT_EQ(retrieved_dimensions.size(), input_dimensions.size());
    for (std::size_t i = 0; i < input_dimensions.size(); ++i) {
        EXPECT_EQ(retrieved_dimensions[i], input_dimensions[i]);
    }
}

/**
 * Helper function to run LDPC decoder module test with H5 test vectors
 *
 * This function contains all the common setup and validation logic for testing
 * the LDPC decoder module, supporting both stream and graph execution modes.
 *
 * @param ldpc_module The LDPC decoder module instance to test
 * @param test_file Path to the H5 test vector file
 * @param execution_mode Whether to use stream or graph mode execution
 */
void run_ldpc_decoder_e2e(
        LdpcDecoderModule &ldpc_module,
        const std::string_view test_file,
        const pipeline::ExecutionMode execution_mode) {

    // Create a CUDA stream
    const CudaStream stream;

    // Read test vector and copy parameters to GPU
    const ran::aerial_tv::CuphyPuschTestVector test_vector{std::string(test_file)};
    auto pusch_outer_rx_params = ran::aerial_tv::to_pusch_outer_rx_params(test_vector);
    pusch_outer_rx_params.copy_tb_params_to_gpu(stream.get());

    // Allocate memory for outputs and setup the module with memory slice
    pipeline::ModuleMemorySlice memory_slice{};
    const auto memory_requirements = ldpc_module.get_requirements();
    const std::size_t output_size = memory_requirements.device_tensor_bytes;
    auto device_output = make_unique_device<std::byte>(output_size);
    memory_slice.device_tensor_ptr = device_output.get();
    ldpc_module.setup_memory(memory_slice);

    // Set up input port
    const auto num_tbs = pusch_outer_rx_params.num_tbs();
    pipeline::PortInfo input_port{
            .name = "llrs", .tensors = std::vector<pipeline::DeviceTensor>(num_tbs)};

    std::vector<UniqueDevicePtr<std::byte>> input_device_ptrs;
    input_device_ptrs.reserve(num_tbs);
    for (std::size_t tb_idx = 0; tb_idx < num_tbs; tb_idx++) {
        const auto ldpc_params = pusch_outer_rx_params[tb_idx].ldpc_params();

        // Read input LLRs for this TB
        const auto input_data_array =
                test_vector.read_array<float>(std::format("reference_rmOutLLRs{}", tb_idx));
        const auto &input_llrs = input_data_array.data;

        // Transform to 16-bit float after transpose
        std::vector<__half> input_llrs_half(input_llrs.size());
        std::transform(
                input_llrs.begin(),
                input_llrs.end(),
                input_llrs_half.begin(),
                [](const float value) { return __float2half(value); });

        // Calculate input size for CUDA memory allocation using __half size
        const std::size_t input_size = input_llrs_half.size() * sizeof(__half);
        input_device_ptrs.push_back(make_unique_device<std::byte>(input_size));

        // Copy the converted half-precision data to device
        ASSERT_EQ(
                cudaMemcpyAsync(
                        input_device_ptrs[tb_idx].get(),
                        input_llrs_half.data(),
                        input_size,
                        cudaMemcpyHostToDevice,
                        stream.get()),
                cudaSuccess);

        // Create input tensor info matching LDPC parameters
        const std::vector<std::size_t> input_dimensions{
                ldpc_params.circular_buffer_size_padded(), ldpc_params.num_code_blocks()};
        const tensor::TensorInfo input_tensor_info(
                tensor::NvDataType::TensorR16F, input_dimensions);

        // Set input tensor for this TB
        input_port.tensors[tb_idx] = pipeline::DeviceTensor{
                .device_ptr = input_device_ptrs[tb_idx].get(), .tensor_info = input_tensor_info};
    }
    const std::vector<pipeline::PortInfo> inputs{input_port};
    ASSERT_NO_THROW(ldpc_module.set_inputs(inputs));

    // Setup dynamic parameters
    pipeline::DynamicParams params{};
    params.module_specific_params = pusch_outer_rx_params;

    // Variables for validation
    pipeline::IGraphNodeProvider *graph_node_provider{};
    std::span<const CUgraphNode> nodes{};

    // example-begin configure-and-execute-1
    // Configure I/O with dynamic parameters
    ldpc_module.configure_io(params, stream.get());

    // Execute based on mode
    if (execution_mode == pipeline::ExecutionMode::Stream) {
        // Stream mode: Execute directly
        RT_LOG_DEBUG("Executing LDPC decoder module in stream mode");
        ldpc_module.execute(stream.get());
    } else {
        // Graph mode: Create graph, add node, instantiate, and launch
        auto graph_manager = std::make_unique<pipeline::GraphManager>();

        graph_node_provider = ldpc_module.as_graph_node_provider();

        // Add module node to graph with no dependencies
        const std::vector<CUgraphNode> no_deps{};
        nodes = graph_manager->add_kernel_node(
                gsl_lite::not_null<pipeline::IGraphNodeProvider *>(graph_node_provider), no_deps);

        // Instantiate and upload graph
        graph_manager->instantiate_graph();
        graph_manager->upload_graph(stream.get());

        // Update graph node parameters
        auto *const exec = graph_manager->get_exec();
        graph_node_provider->update_graph_node_params(exec, params);

        // Launch graph
        RT_LOG_DEBUG("Executing LDPC decoder module in graph mode");
        graph_manager->launch_graph(stream.get());
    }
    // example-end configure-and-execute-1

    // Synchronize stream to ensure all GPU operations are complete before reading values
    ASSERT_TRUE(stream.synchronize());
    RT_LOG_DEBUG("Finished executing LDPC decoder module");

    // Validate graph mode results
    if (execution_mode == pipeline::ExecutionMode::Graph) {
        ASSERT_NE(graph_node_provider, nullptr);
        ASSERT_FALSE(nodes.empty());
    }

    // Verify outputs
    const auto outputs = ldpc_module.get_outputs();
    ASSERT_EQ(outputs.size(), 1);
    EXPECT_EQ(outputs[0].name, "decoded_bits");
    EXPECT_EQ(outputs[0].tensors[0].device_ptr, device_output.get());

    // Calculate output size based on the actual output tensor dimensions
    const auto output_tensor_info = outputs[0].tensors[0].tensor_info;
    const auto &output_dimensions = output_tensor_info.get_dimensions();

    // The output tensor is of type TensorBit, so get_total_elements() returns number of bits
    // We need to convert to bytes for the memory copy
    const std::size_t num_output_bits = output_tensor_info.get_total_elements();
    const std::size_t num_output_bytes = (num_output_bits + BITS_PER_BYTE - 1) / BITS_PER_BYTE;
    std::vector<std::uint8_t> output_bytes(num_output_bytes);

    // Use stream-aware async copy for proper synchronization
    ASSERT_EQ(
            cudaMemcpyAsync(
                    output_bytes.data(),
                    outputs[0].tensors[0].device_ptr,
                    num_output_bytes,
                    cudaMemcpyDeviceToHost,
                    stream.get()),
            cudaSuccess);
    // Synchronize stream to ensure async copy completes before validation
    ASSERT_TRUE(stream.synchronize());

    // Extract individual bits
    std::vector<std::uint8_t> output_bits(num_output_bits);
    for (std::size_t i = 0; i < num_output_bits; ++i) {
        const std::size_t byte_index = i / BITS_PER_BYTE;
        const std::size_t bit_index = i % BITS_PER_BYTE;
        output_bits[i] = static_cast<std::uint8_t>(
                (static_cast<unsigned int>(output_bytes[byte_index]) >> bit_index) & 1U);
    }

    std::size_t first_cb_idx = 0;
    for (std::size_t tb_idx = 0; tb_idx < num_tbs; tb_idx++) {
        const auto ldpc_params = pusch_outer_rx_params[tb_idx].ldpc_params();

        // Extract only the information bits for comparison
        // The decoder outputs MAX_CODE_BLOCK_SIZE_BG1 bits per code block,
        // but we only want to compare the actual information bits (K bits)
        const auto num_code_block_info_bits = ldpc_params.num_code_block_info_bits();
        const auto num_code_blocks = ldpc_params.num_code_blocks();
        const auto num_info_bits = num_code_block_info_bits * num_code_blocks;

        std::vector<std::uint8_t> info_bits{};
        info_bits.reserve(num_info_bits);

        // Extract first num_code_block_info_bits bits from each code block
        for (std::size_t cb = 0; cb < num_code_blocks; ++cb) {
            const std::size_t cb_start_index = (first_cb_idx + cb) * output_dimensions[0];
            for (std::size_t bit = 0; bit < num_code_block_info_bits; ++bit) {
                info_bits.emplace_back(output_bits[cb_start_index + bit]);
            }
        }
        first_cb_idx += num_code_blocks;

        // Validate output data against expected results from H5 file
        const auto expected_outputs_array =
                test_vector.read_array<std::uint8_t>(std::format("reference_TbCbs_est{}", tb_idx));
        const auto &expected_outputs = expected_outputs_array.data;

        ASSERT_EQ(info_bits.size(), expected_outputs.size());
        ASSERT_EQ(info_bits, expected_outputs);
    }
}

/**
 * Test full LDPC decoder module using H5 test vectors in stream mode.
 * This test loads input data from H5 file, executes the LDPC decoder module
 * using direct stream execution, and validates the output formats and basic functionality.
 */
TEST_P(LdpcDecoderModuleH5Test, FullModuleTestWithH5TestVectorsStreamMode) {
    const std::string &test_file_path = get_test_file_path();

    // Run test using stream mode
    run_ldpc_decoder_e2e(*ldpc_decoder_module_, test_file_path, pipeline::ExecutionMode::Stream);
}

/**
 * Test full LDPC decoder module using H5 test vectors in graph mode.
 * This test loads input data from H5 file, executes the LDPC decoder module
 * using CUDA graph execution, and validates the output formats and basic functionality.
 */
TEST_P(LdpcDecoderModuleH5Test, FullModuleTestWithH5TestVectorsGraphMode) {
    const std::string &test_file_path = get_test_file_path();

    // Run test using graph mode
    run_ldpc_decoder_e2e(*ldpc_decoder_module_, test_file_path, pipeline::ExecutionMode::Graph);
}

// Instantiate the parameterized test with all available H5 test files
INSTANTIATE_TEST_SUITE_P(
        MultipleH5Files,
        LdpcDecoderModuleH5Test,
        ::testing::ValuesIn(ran::aerial_tv::TEST_HDF5_FILES),
        [](const ::testing::TestParamInfo<std::string_view> &test_info) {
            std::string name = std::filesystem::path(test_info.param).stem().string();
            // Replace non-alphanumeric characters with underscores for valid test
            // names
            std::replace_if(name.begin(), name.end(), [](char c) { return !std::isalnum(c); }, '_');
            return name;
        });

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

} // namespace
