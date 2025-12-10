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
#include <functional>
#include <memory>
#include <numeric>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <driver_types.h>

#include <gsl-lite/gsl-lite.hpp>
#include <gtest/gtest.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "aerial_tv/aerial_tv_utils.hpp"
#include "aerial_tv/cuphy_pusch_tv.hpp"
#include "ldpc/crc_decoder_module.hpp"
#include "ldpc/ldpc_params.hpp"
#include "ldpc/outer_rx_params.hpp"
#include "log/components.hpp"
#include "log/rt_log_macros.hpp"
#include "memory/unique_ptr_utils.hpp"
#include "pipeline/graph_manager.hpp"
#include "pipeline/igraph_node_provider.hpp"
#include "pipeline/kernel_descriptor_accessor.hpp"
#include "pipeline/types.hpp"
#include "tensor/data_types.hpp"
#include "tensor/tensor_info.hpp"
#include "utils/core_log.hpp"
#include "utils/cuda_stream.hpp"

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

namespace {

using ran::ldpc::CrcDecoderModule;
namespace pipeline = framework::pipeline;
namespace tensor = framework::tensor;
using framework::memory::make_unique_device;
using framework::memory::make_unique_pinned;
using framework::memory::UniqueDevicePtr;
using framework::memory::UniquePinnedPtr;
using framework::utils::CudaStream;

const CrcDecoderModule::StaticParams STATIC_PARAMS{
        .reverse_bytes = true, .max_num_cbs_per_tb = 152, .max_num_tbs = 8};

/**
 * Helper function to run CRC decoder module test with H5 test vectors
 *
 * This function contains all the common setup and validation logic for testing
 * the CRC decoder module, supporting both stream and graph execution modes.
 *
 * @param crc_module The CRC decoder module instance to test
 * @param test_file Path to the H5 test vector file
 * @param execution_mode Whether to use stream or graph mode execution
 */
void run_crc_decoder_e2e(
        CrcDecoderModule &crc_module,
        const std::string_view test_file,
        const pipeline::ExecutionMode execution_mode) {

    // Create a CUDA stream
    const CudaStream stream;

    // Read test vector and copy parameters to GPU
    const ran::aerial_tv::CuphyPuschTestVector test_vector(std::string{test_file});
    auto pusch_outer_rx_params = ran::aerial_tv::to_pusch_outer_rx_params(test_vector);
    pusch_outer_rx_params.copy_tb_params_to_gpu(stream.get());

    // Setup CrcDecoderModule
    pipeline::ModuleMemorySlice crc_decoder_memory_slice{};
    const auto crc_decoder_memory_requirements = crc_module.get_requirements();
    const std::size_t crc_decoder_output_size = crc_decoder_memory_requirements.device_tensor_bytes;
    auto crc_decoder_device_output = make_unique_device<std::byte>(crc_decoder_output_size);
    crc_decoder_memory_slice.device_tensor_ptr = crc_decoder_device_output.get();

    // Allocate dynamic kernel descriptor memory for CRC decoder module
    auto crc_decoder_dyn_descr_cpu = make_unique_pinned<std::byte>(
            crc_decoder_memory_requirements.dynamic_kernel_descriptor_bytes);
    auto crc_decoder_dyn_descr_gpu = make_unique_device<std::byte>(
            crc_decoder_memory_requirements.dynamic_kernel_descriptor_bytes);
    crc_decoder_memory_slice.dynamic_kernel_descriptor_cpu_ptr = crc_decoder_dyn_descr_cpu.get();
    crc_decoder_memory_slice.dynamic_kernel_descriptor_gpu_ptr = crc_decoder_dyn_descr_gpu.get();

    crc_decoder_memory_slice.static_kernel_descriptor_bytes =
            crc_decoder_memory_requirements.static_kernel_descriptor_bytes;
    crc_decoder_memory_slice.dynamic_kernel_descriptor_bytes =
            crc_decoder_memory_requirements.dynamic_kernel_descriptor_bytes;
    crc_decoder_memory_slice.device_tensor_bytes =
            crc_decoder_memory_requirements.device_tensor_bytes;

    crc_module.setup_memory(crc_decoder_memory_slice);

    // Setup input data for CRC decoder module
    constexpr std::uint32_t BITS_PER_BYTE = 8;

    // Count total number of code blocks across all TBs
    const auto num_tbs = pusch_outer_rx_params.num_tbs();
    const auto total_code_blocks = std::transform_reduce(
            std::ranges::iota_view{std::size_t{0}, num_tbs}.begin(),
            std::ranges::iota_view{std::size_t{0}, num_tbs}.end(),
            std::size_t{0},
            std::plus{},
            [&](const std::size_t tb_idx) {
                return pusch_outer_rx_params[tb_idx].ldpc_params().num_code_blocks();
            });

    const std::size_t input_size_bytes{
            (ran::ldpc::LdpcParams::MAX_CODE_BLOCK_SIZE_BG1 / BITS_PER_BYTE) * total_code_blocks};
    auto input_device = make_unique_device<std::byte>(input_size_bytes);
    void *input_device_ptr = input_device.get();

    // Pack input data into bytes.
    std::size_t offset_bytes = 0;
    std::vector<std::uint8_t> input_bytes(input_size_bytes, 0);
    for (std::size_t tb_idx = 0; tb_idx < num_tbs; ++tb_idx) {
        const auto ldpc_params = pusch_outer_rx_params[tb_idx].ldpc_params();
        const auto num_code_blocks = ldpc_params.num_code_blocks();
        const auto num_code_block_info_bits = ldpc_params.num_code_block_info_bits();

        // Read input data from H5 file
        const auto input_data_array =
                test_vector.read_array<std::uint8_t>(std::format("reference_TbCbs_est{}", tb_idx));
        const auto &input_data = input_data_array.data;

        for (std::size_t cb = 0; cb < num_code_blocks; ++cb) {

            for (std::size_t i = 0; i < num_code_block_info_bits; ++i) {
                const std::size_t byte_index = i / BITS_PER_BYTE;
                const std::size_t bit_index = i % BITS_PER_BYTE;
                input_bytes[offset_bytes + byte_index] |=
                        (input_data[cb * num_code_block_info_bits + i] & 1U) << bit_index;
            }

            offset_bytes += (ran::ldpc::LdpcParams::MAX_CODE_BLOCK_SIZE_BG1 / BITS_PER_BYTE);
        }
    }

    ASSERT_EQ(
            cudaMemcpyAsync(
                    input_device_ptr,
                    input_bytes.data(),
                    input_size_bytes,
                    cudaMemcpyHostToDevice,
                    stream.get()),
            cudaSuccess);

    // Create input tensor info matching CRC decoder parameters
    const std::vector input_dimensions{
            static_cast<std::size_t>(ran::ldpc::LdpcParams::MAX_CODE_BLOCK_SIZE_BG1),
            total_code_blocks};
    const tensor::TensorInfo input_tensor_info(tensor::NvDataType::TensorBit, input_dimensions);

    // Create and set inputs
    const pipeline::PortInfo input_port{
            .name = "decoded_bits",
            .tensors = {pipeline::DeviceTensor{
                    .device_ptr = input_device_ptr, .tensor_info = input_tensor_info}}};
    const std::vector<pipeline::PortInfo> inputs{input_port};
    ASSERT_NO_THROW(crc_module.set_inputs(inputs));

    // Configure I/O
    pipeline::DynamicParams params{};
    params.module_specific_params = pusch_outer_rx_params;
    ASSERT_NO_THROW(crc_module.configure_io(params, stream.get()));

    // Copy dynamic descriptors to device (normally done by PipelineMemoryManager)
    // This is required for both stream and graph modes before kernel execution
    const pipeline::KernelDescriptorAccessor descriptor_accessor(crc_decoder_memory_slice);
    descriptor_accessor.copy_dynamic_descriptors_to_device(stream.get());

    // Execute based on mode
    if (execution_mode == pipeline::ExecutionMode::Stream) {
        // Stream mode: Execute directly (descriptor copy already done above)
        RT_LOG_DEBUG("Executing CRC decoder module in stream mode");
        ASSERT_NO_THROW(crc_module.execute(stream.get()));
    } else {
        // Graph mode: Create graph, add node, instantiate, and launch
        auto graph_manager = std::make_unique<pipeline::GraphManager>();

        auto *graph_node_provider = crc_module.as_graph_node_provider();
        ASSERT_NE(graph_node_provider, nullptr);

        // Add module node(s) to graph with no dependencies
        const std::vector<CUgraphNode> no_deps{};
        const auto nodes = graph_manager->add_kernel_node(
                gsl_lite::not_null<pipeline::IGraphNodeProvider *>(graph_node_provider), no_deps);
        ASSERT_FALSE(nodes.empty());
        ASSERT_NE(nodes[0], nullptr);

        // Instantiate and upload graph
        graph_manager->instantiate_graph();
        graph_manager->upload_graph(stream.get());

        // Update graph node parameters
        auto *const exec = graph_manager->get_exec();
        graph_node_provider->update_graph_node_params(exec, params);

        // Launch graph
        RT_LOG_DEBUG("Executing CRC decoder module in graph mode");
        graph_manager->launch_graph(stream.get());
    }

    ASSERT_TRUE(stream.synchronize());
    RT_LOG_DEBUG("Finished executing CRC decoder module");

    // Verify outputs
    const auto outputs = crc_module.get_outputs();
    ASSERT_EQ(outputs.size(), 3);
    EXPECT_EQ(outputs[0].name, "cb_crcs");
    EXPECT_EQ(outputs[1].name, "tb_crcs");
    EXPECT_EQ(outputs[2].name, "tb_payloads");

    // Verify output tensor information
    for (std::size_t tb_idx = 0; tb_idx < num_tbs; ++tb_idx) {

        EXPECT_EQ(
                outputs[0].tensors[tb_idx].tensor_info.get_type(),
                tensor::NvDataType::TensorR32U); // CB CRCs
        EXPECT_EQ(
                outputs[1].tensors[tb_idx].tensor_info.get_type(),
                tensor::NvDataType::TensorR32U); // TB CRCs
        EXPECT_EQ(
                outputs[2].tensors[tb_idx].tensor_info.get_type(),
                tensor::NvDataType::TensorR8U); // TB payloads

        // Check that outputs are not null.
        EXPECT_NE(outputs[0].tensors[tb_idx].device_ptr, nullptr);
        EXPECT_NE(outputs[1].tensors[tb_idx].device_ptr, nullptr);
        EXPECT_NE(outputs[2].tensors[tb_idx].device_ptr, nullptr);
    }

    // Verify transport block payloads
    ran::aerial_tv::check_tb_payloads(outputs[2].tensors, test_vector, stream.get());

    // Verify transport block CRCs pass
    std::vector<std::uint32_t> tb_crc_values(num_tbs);
    ASSERT_EQ(
            cudaMemcpyAsync(
                    tb_crc_values.data(),
                    outputs[1].tensors[0].device_ptr,
                    sizeof(std::uint32_t) * num_tbs,
                    cudaMemcpyDeviceToHost,
                    stream.get()),
            cudaSuccess);
    ASSERT_TRUE(stream.synchronize());
    for (std::size_t tb_idx = 0; tb_idx < num_tbs; ++tb_idx) {
        EXPECT_EQ(tb_crc_values[tb_idx], 0U) << "TB " << tb_idx << " CRC check failed";
    }
}

/**
 * Test fixture for CrcDecoderModule
 *
 * Sets up common test infrastructure including CUDA context,
 * memory management, and module initialization for test cases.
 */
class CrcDecoderModuleTest : public ::testing::Test {
protected:
    void SetUp() override {
        crc_decoder_module_ =
                std::make_unique<CrcDecoderModule>("crc_decoder_instance", STATIC_PARAMS);
    }

    std::unique_ptr<CrcDecoderModule> crc_decoder_module_;
};

/**
 * Parameterized test fixture for CrcDecoderModule with H5 test vector files
 *
 * Tests the module with different H5 test vector files to ensure
 * robustness across various input scenarios.
 */
class CrcDecoderModuleH5Test : public CrcDecoderModuleTest,
                               public ::testing::WithParamInterface<std::string_view> {
protected:
    void SetUp() override {
        CrcDecoderModuleTest::SetUp();

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
TEST_F(CrcDecoderModuleTest, ModuleIdentification) {

    // Test type ID
    EXPECT_STREQ(crc_decoder_module_->get_type_id().data(), "crc_decoder_module");

    // Test instance ID
    EXPECT_STREQ(crc_decoder_module_->get_instance_id().data(), "crc_decoder_instance");
}

/**
 * Verifies module correctly reports its input and output port names.
 */
TEST_F(CrcDecoderModuleTest, PortNames) {

    // Test input port names - CrcDecoderModule has 1 predefined input
    const auto input_ports = crc_decoder_module_->get_input_port_names();
    EXPECT_EQ(input_ports.size(), 1);
    EXPECT_EQ(input_ports[0], "decoded_bits");

    // Test output port names - CrcDecoderModule has 3 predefined outputs
    const auto output_ports = crc_decoder_module_->get_output_port_names();
    EXPECT_EQ(output_ports.size(), 3);
    EXPECT_EQ(output_ports[0], "cb_crcs");
    EXPECT_EQ(output_ports[1], "tb_crcs");
    EXPECT_EQ(output_ports[2], "tb_payloads");
}

/**
 * Test memory requirements calculation.
 */
TEST_F(CrcDecoderModuleTest, MemoryRequirements) {

    // Get memory requirements
    const auto requirements = crc_decoder_module_->get_requirements();

    // Verify requirements are reasonable
    EXPECT_GT(requirements.device_tensor_bytes, 0);
    EXPECT_GE(requirements.alignment, 1);

    // Static kernel descriptor bytes should be 0 according to implementation
    EXPECT_EQ(requirements.static_kernel_descriptor_bytes, 0);

    // Dynamic kernel descriptor bytes should be > 0 (from cuphy API)
    EXPECT_GT(requirements.dynamic_kernel_descriptor_bytes, 0);
}

/**
 * Test full CRC decoder module using H5 test vectors in stream mode.
 * This test loads input data from H5 file, executes the CRC decoder module
 * using direct stream execution, and validates the output formats and basic functionality.
 */
TEST_P(CrcDecoderModuleH5Test, FullModuleTestWithH5TestVectorsStreamMode) {
    const std::string &test_file_path = get_test_file_path();

    // Run test using stream mode
    run_crc_decoder_e2e(*crc_decoder_module_, test_file_path, pipeline::ExecutionMode::Stream);
}

/**
 * Test full CRC decoder module using H5 test vectors in graph mode.
 * This test loads input data from H5 file, executes the CRC decoder module
 * using CUDA graph execution, and validates the output formats and basic functionality.
 */
TEST_P(CrcDecoderModuleH5Test, FullModuleTestWithH5TestVectorsGraphMode) {
    const std::string &test_file_path = get_test_file_path();

    // Run test using graph mode
    run_crc_decoder_e2e(*crc_decoder_module_, test_file_path, pipeline::ExecutionMode::Graph);
}

// Instantiate the parameterized test with all available H5 test files
INSTANTIATE_TEST_SUITE_P(
        MultipleH5Files,
        CrcDecoderModuleH5Test,
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
