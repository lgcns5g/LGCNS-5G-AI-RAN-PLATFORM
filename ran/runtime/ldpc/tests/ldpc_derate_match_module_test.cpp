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
#include <filesystem>
#include <format>
#include <functional>
#include <memory>
#include <numeric>
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
#include "ldpc/derate_match_params.hpp"
#include "ldpc/ldpc_derate_match_module.hpp"
#include "ldpc/ldpc_params.hpp"
#include "ldpc/outer_rx_params.hpp"
#include "log/components.hpp"
#include "log/rt_log_macros.hpp"
#include "memory/unique_ptr_utils.hpp"
#include "pipeline/graph_manager.hpp"
#include "pipeline/igraph_node_provider.hpp"
#include "pipeline/kernel_descriptor_accessor.hpp"
#include "pipeline/types.hpp"
#include "ran_common.hpp"
#include "tensor/data_types.hpp"
#include "tensor/tensor_info.hpp"
#include "utils/core_log.hpp"
#include "utils/cuda_stream.hpp"

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

namespace {

using ran::ldpc::LdpcDerateMatchModule;
namespace pipeline = framework::pipeline;
namespace tensor = framework::tensor;
using framework::memory::make_unique_device;
using framework::memory::make_unique_pinned;
using framework::memory::UniqueDevicePtr;
using framework::memory::UniquePinnedPtr;
using framework::utils::CudaStream;

const LdpcDerateMatchModule::StaticParams STATIC_PARAMS{
        .enable_scrambling = true,
        .max_num_tbs = ran::common::MAX_NUM_TBS,
        .max_num_cbs_per_tb = ran::common::MAX_NUM_CBS_PER_TB,
        .max_num_rm_llrs_per_cb = ran::ldpc::MAX_NUM_RM_LLRS_PER_CB,
        .max_num_ue_grps = ran::common::MAX_NUM_UE_GRPS};

/**
 * Execution modes for testing the LDPC derate match module
 */
enum class ExecutionMode {
    Stream, //!< Execute using CUDA streams directly
    Graph   //!< Execute using CUDA graphs
};

/**
 * Helper function to run LDPC derate match module test with H5 test vectors
 *
 * This function contains all the common setup and validation logic for testing
 * the LDPC derate match module, supporting both stream and graph execution modes.
 *
 * @param ldpc_module The LDPC derate match module instance to test
 * @param test_file Path to the H5 test vector file
 * @param execution_mode Whether to use stream or graph mode execution
 */
void run_ldpc_derate_match_e2e(
        LdpcDerateMatchModule &ldpc_module,
        const std::string_view test_file,
        const ExecutionMode execution_mode) {

    // Create a CUDA stream
    const CudaStream stream;

    // Read test vector and copy parameters to GPU
    const ran::aerial_tv::CuphyPuschTestVector test_vector{std::string{test_file}};
    auto pusch_outer_rx_params = ran::aerial_tv::to_pusch_outer_rx_params(test_vector);
    pusch_outer_rx_params.copy_tb_params_to_gpu(stream.get());

    // Setup memory
    pipeline::ModuleMemorySlice de_rm_memory_slice{};
    const auto de_rm_memory_requirements = ldpc_module.get_requirements();
    const std::size_t de_rm_output_size = de_rm_memory_requirements.device_tensor_bytes;
    auto device_output = make_unique_device<std::byte>(de_rm_output_size);
    de_rm_memory_slice.device_tensor_ptr = device_output.get();

    // Allocate dynamic kernel descriptor memory for LDPC derate match module
    auto de_rm_dyn_descr_cpu = make_unique_pinned<std::byte>(
            de_rm_memory_requirements.dynamic_kernel_descriptor_bytes);
    auto de_rm_dyn_descr_gpu = make_unique_device<std::byte>(
            de_rm_memory_requirements.dynamic_kernel_descriptor_bytes);
    de_rm_memory_slice.dynamic_kernel_descriptor_cpu_ptr = de_rm_dyn_descr_cpu.get();
    de_rm_memory_slice.dynamic_kernel_descriptor_gpu_ptr = de_rm_dyn_descr_gpu.get();

    de_rm_memory_slice.static_kernel_descriptor_bytes =
            de_rm_memory_requirements.static_kernel_descriptor_bytes;
    de_rm_memory_slice.dynamic_kernel_descriptor_bytes =
            de_rm_memory_requirements.dynamic_kernel_descriptor_bytes;
    de_rm_memory_slice.device_tensor_bytes = de_rm_memory_requirements.device_tensor_bytes;

    ldpc_module.setup_memory(de_rm_memory_slice);

    // Setup input port info for LDPC derate match module
    const auto num_ue_grps = test_vector.get_ue_grp_params().size();
    pipeline::PortInfo llrs_port{
            .name = "llrs", .tensors = std::vector<pipeline::DeviceTensor>(num_ue_grps)};
    pipeline::PortInfo llrs_cdm1_port{
            .name = "llrs_cdm1", .tensors = std::vector<pipeline::DeviceTensor>(num_ue_grps)};

    // Allocate device input pointers vector
    std::vector<UniqueDevicePtr<std::byte>> device_inputs;
    device_inputs.reserve(num_ue_grps);

    // Read rate matching input LLRs for each UE group
    for (std::size_t i = 0; i < num_ue_grps; ++i) {
        const auto input_data_array =
                test_vector.read_array<float>(std::format("reference_eqOutLLRs{}", i));
        const auto &input_rm_llrs = input_data_array.data;

        // Transform to 16-bit float
        std::vector<__half> input_rm_llrs_half(input_rm_llrs.size());
        std::transform(
                input_rm_llrs.begin(),
                input_rm_llrs.end(),
                input_rm_llrs_half.begin(),
                [](const float value) { return __float2half(value); });

        // Allocate and copy input data to device
        const std::size_t input_size = input_rm_llrs_half.size() * sizeof(__half);
        device_inputs.push_back(make_unique_device<std::byte>(input_size));
        ASSERT_EQ(
                cudaMemcpyAsync(
                        device_inputs[i].get(),
                        input_rm_llrs_half.data(),
                        input_size,
                        cudaMemcpyHostToDevice,
                        stream.get()),
                cudaSuccess);

        // Create input tensor info for derate match module
        const auto &ue_grp_params = test_vector.get_ue_grp_params(i);
        const auto first_ue_idx = ue_grp_params.ue_prm_idxs[0];
        const auto &tb_params = test_vector.get_tb_params(first_ue_idx);

        const auto num_prb = ue_grp_params.n_prb;
        const auto num_layers =
                pusch_outer_rx_params[first_ue_idx].de_rm_params().num_ue_grp_layers;
        const auto num_symbols = ue_grp_params.nr_of_symbols;
        const auto dmrs_addln_pos = tb_params.dmrs_addl_position;
        const auto dmrs_max_len = tb_params.dmrs_max_length;
        const auto num_dmrs_symbols = (1 + dmrs_addln_pos) * dmrs_max_len;
        const auto num_data_symbols = num_symbols - num_dmrs_symbols;
        constexpr int NUM_SUBC_PER_PRB = 12;
        constexpr int QAM_STRIDE_VALUE = 8;

        const std::vector input_dimensions{
                static_cast<std::size_t>(QAM_STRIDE_VALUE),
                static_cast<std::size_t>(num_layers),
                static_cast<std::size_t>(num_prb * NUM_SUBC_PER_PRB),
                static_cast<std::size_t>(num_data_symbols)};

        const tensor::TensorInfo input_tensor_info(
                tensor::NvDataType::TensorR16F, input_dimensions);

        llrs_port.tensors[i] = pipeline::DeviceTensor{
                .device_ptr = device_inputs[i].get(), .tensor_info = input_tensor_info};
        llrs_cdm1_port.tensors[i] = pipeline::DeviceTensor{
                .device_ptr = device_inputs[i].get(), .tensor_info = input_tensor_info};
    }

    const std::vector<pipeline::PortInfo> de_rm_inputs{llrs_port, llrs_cdm1_port};
    ASSERT_NO_THROW(ldpc_module.set_inputs(de_rm_inputs));

    // Configure I/O
    pipeline::DynamicParams params{};
    params.module_specific_params = pusch_outer_rx_params;
    ASSERT_NO_THROW(ldpc_module.configure_io(params, stream.get()));

    // Copy dynamic descriptors to device (normally done by PipelineMemoryManager)
    // This is required for both stream and graph modes before kernel execution
    const pipeline::KernelDescriptorAccessor descriptor_accessor(de_rm_memory_slice);
    descriptor_accessor.copy_dynamic_descriptors_to_device(stream.get());

    // Execute based on mode
    if (execution_mode == ExecutionMode::Stream) {
        RT_LOG_DEBUG("Executing LDPC derate match module in stream mode");
        ASSERT_NO_THROW(ldpc_module.execute(stream.get()));
    } else {
        // example-begin graph-execution-1
        // Graph mode: Create graph manager and get module's graph interface
        auto graph_manager = std::make_unique<pipeline::GraphManager>();

        auto *graph_node_provider = ldpc_module.as_graph_node_provider();
        // example-end graph-execution-1

        // Verify graph node provider is available
        ASSERT_NE(graph_node_provider, nullptr);

        // example-begin graph-execution-2
        // Add module node(s) to graph with no dependencies
        const std::vector<CUgraphNode> no_deps{};
        const auto nodes = graph_manager->add_kernel_node(
                gsl_lite::not_null<pipeline::IGraphNodeProvider *>(graph_node_provider), no_deps);
        // example-end graph-execution-2

        // Verify nodes were created successfully
        ASSERT_FALSE(nodes.empty());
        ASSERT_NE(nodes[0], nullptr);

        // example-begin graph-execution-3
        // Instantiate and upload graph
        graph_manager->instantiate_graph();
        graph_manager->upload_graph(stream.get());

        // Update graph node parameters
        auto *const exec = graph_manager->get_exec();
        graph_node_provider->update_graph_node_params(exec, params);

        // Launch graph
        RT_LOG_DEBUG("Executing LDPC derate match module in graph mode");
        graph_manager->launch_graph(stream.get());
        // example-end graph-execution-3
    }

    ASSERT_TRUE(stream.synchronize());
    RT_LOG_DEBUG("Finished executing LDPC derate match module");

    // Get derate match outputs and verify
    const auto de_rm_outputs = ldpc_module.get_outputs();
    ASSERT_EQ(de_rm_outputs.size(), 1);
    EXPECT_EQ(de_rm_outputs[0].name, "derate_matched_llrs");

    // Verify derate match module outputs
    const auto num_tbs = pusch_outer_rx_params.num_tbs();
    for (std::size_t tb_idx = 0; tb_idx < num_tbs; ++tb_idx) {
        const auto output_tensor_info = de_rm_outputs[0].tensors[tb_idx].tensor_info;
        const auto &output_dimensions = output_tensor_info.get_dimensions();
        const std::size_t num_output_elements = std::accumulate(
                output_dimensions.begin(),
                output_dimensions.end(),
                std::size_t{1},
                std::multiplies<>());

        // Copy output data from device to host using stream-aware async copy
        const std::size_t output_size_bytes = num_output_elements * sizeof(__half);
        std::vector<__half> output_half(num_output_elements);
        ASSERT_EQ(
                cudaMemcpyAsync(
                        output_half.data(),
                        de_rm_outputs[0].tensors[tb_idx].device_ptr,
                        output_size_bytes,
                        cudaMemcpyDeviceToHost,
                        stream.get()),
                cudaSuccess);
        // Synchronize stream to ensure async copy completes before validation
        ASSERT_TRUE(stream.synchronize());

        // Convert from half precision to float for comparison
        std::vector<float> output_llrs(num_output_elements);
        std::transform(
                output_half.begin(),
                output_half.end(),
                output_llrs.begin(),
                [](const __half value) { return __half2float(value); });

        // Validate output data against expected derate match results from H5 file
        const auto expected_outputs_array =
                test_vector.read_array<float>(std::format("reference_rmOutLLRs{}", tb_idx));
        const auto &expected_outputs = expected_outputs_array.data;
        ASSERT_EQ(output_llrs.size(), expected_outputs.size());

        // Compare with tolerance for floating point precision
        constexpr float TOLERANCE_VALUE = 1e-5F;
        for (std::size_t i = 0; i < output_llrs.size(); ++i) {
            EXPECT_NEAR(output_llrs[i], expected_outputs[i], TOLERANCE_VALUE) << std::format(
                    "Mismatch at TB {} index {}: got {}, expected {}",
                    tb_idx,
                    i,
                    output_llrs[i],
                    expected_outputs[i]);
        }
    }
}

/**
 * Test fixture for LdpcDerateMatchModule
 */
class LdpcDerateMatchModuleTest : public ::testing::Test {
protected:
    void SetUp() override {
        ldpc_derate_match_module_ = std::make_unique<LdpcDerateMatchModule>(
                "ldpc_derate_match_instance", STATIC_PARAMS);
    }

    std::unique_ptr<LdpcDerateMatchModule> ldpc_derate_match_module_;
};

/**
 * Parameterized test fixture for LdpcDerateMatchModule with H5 test vector
 * files
 */
class LdpcDerateMatchModuleH5Test : public LdpcDerateMatchModuleTest,
                                    public ::testing::WithParamInterface<std::string_view> {
protected:
    void SetUp() override {
        LdpcDerateMatchModuleTest::SetUp();

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
TEST_F(LdpcDerateMatchModuleTest, ModuleIdentification) {

    // Test type ID
    EXPECT_STREQ(ldpc_derate_match_module_->get_type_id().data(), "ldpc_derate_match_module");

    // Test instance ID
    EXPECT_STREQ(ldpc_derate_match_module_->get_instance_id().data(), "ldpc_derate_match_instance");
}

/**
 * Verifies module correctly reports its input and output port names.
 */
TEST_F(LdpcDerateMatchModuleTest, PortNames) {

    // Test input port names
    const auto input_ports = ldpc_derate_match_module_->get_input_port_names();
    EXPECT_EQ(input_ports.size(), 2);
    EXPECT_EQ(input_ports[0], "llrs");
    EXPECT_EQ(input_ports[1], "llrs_cdm1");

    // Test output port names
    const auto output_ports = ldpc_derate_match_module_->get_output_port_names();
    EXPECT_EQ(output_ports.size(), 1);
    EXPECT_EQ(output_ports[0], "derate_matched_llrs");
}

/**
 * Test setting inputs and verifying them with getInputTensorInfo.
 */
TEST_F(LdpcDerateMatchModuleTest, SetInputsAndVerify) {

    // First, we need to call setup to complete module setup
    pipeline::ModuleMemorySlice memory_slice{};
    memory_slice.device_tensor_ptr = nullptr; // Not needed for this test
    ldpc_derate_match_module_->setup_memory(memory_slice);

    // Create input tensor info - FLOAT16 with 2D dimensions
    const std::vector<std::size_t> input_dimensions{1000, 8}; // Example dimensions
    const tensor::TensorInfo input_tensor_info{tensor::NvDataType::TensorR16F, input_dimensions};

    // Create PortInfo for both required inputs - use dummy non-null pointers for
    // validation
    const pipeline::PortInfo llrs_port{
            .name = "llrs",
            .tensors = {pipeline::DeviceTensor{
                    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
                    .device_ptr = reinterpret_cast<void *>(0x1),
                    .tensor_info = input_tensor_info}}};

    const pipeline::PortInfo llrs_cdm1_port{
            .name = "llrs_cdm1",
            .tensors = {pipeline::DeviceTensor{
                    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
                    .device_ptr = reinterpret_cast<void *>(0x2),
                    .tensor_info = input_tensor_info}}};

    // Set the inputs
    const std::vector<pipeline::PortInfo> inputs{llrs_port, llrs_cdm1_port};
    ASSERT_NO_THROW(ldpc_derate_match_module_->set_inputs(inputs));

    // Verify the inputs were set correctly using get_input_tensor_info
    const tensor::TensorInfo retrieved_llrs_info =
            ldpc_derate_match_module_->get_input_tensor_info("llrs")[0];
    const tensor::TensorInfo retrieved_llrs_cdm1_info =
            ldpc_derate_match_module_->get_input_tensor_info("llrs_cdm1")[0];

    // Verify the tensor info matches what we set
    EXPECT_EQ(retrieved_llrs_info.get_type(), tensor::NvDataType::TensorR16F);
    EXPECT_EQ(retrieved_llrs_info.get_dimensions(), input_dimensions);

    EXPECT_EQ(retrieved_llrs_cdm1_info.get_type(), tensor::NvDataType::TensorR16F);
    EXPECT_EQ(retrieved_llrs_cdm1_info.get_dimensions(), input_dimensions);
}

/**
 * Test memory requirements calculation.
 */
TEST_F(LdpcDerateMatchModuleTest, MemoryRequirements) {

    // Get memory requirements
    const auto requirements = ldpc_derate_match_module_->get_requirements();

    // Verify requirements are reasonable
    EXPECT_GT(requirements.device_tensor_bytes, 0);
    EXPECT_GE(requirements.alignment, 1);

    // Static kernel descriptor bytes should be 0 according to implementation
    EXPECT_EQ(requirements.static_kernel_descriptor_bytes, 0);

    // Dynamic kernel descriptor bytes should be > 0 (from cuphy API)
    EXPECT_GT(requirements.dynamic_kernel_descriptor_bytes, 0);
}

/**
 * Test full LDPC derate match module using H5 test vectors in stream mode.
 * This test loads input data from H5 file, executes the derate match module
 * using direct stream execution, and validates the output against expected reference data.
 */
TEST_P(LdpcDerateMatchModuleH5Test, FullModuleTestWithH5TestVectorsStreamMode) {
    const std::string &test_file_path = get_test_file_path();

    // Run test using stream mode
    run_ldpc_derate_match_e2e(*ldpc_derate_match_module_, test_file_path, ExecutionMode::Stream);
}

/**
 * Test full LDPC derate match module using H5 test vectors in graph mode.
 * This test loads input data from H5 file, executes the derate match module
 * using CUDA graph execution, and validates the output against expected reference data.
 */
TEST_P(LdpcDerateMatchModuleH5Test, FullModuleTestWithH5TestVectorsGraphMode) {

    const std::string &test_file_path = get_test_file_path();

    // Run test using graph mode
    run_ldpc_derate_match_e2e(*ldpc_derate_match_module_, test_file_path, ExecutionMode::Graph);
}

// Instantiate the parameterized test with all available H5 test files
INSTANTIATE_TEST_SUITE_P(
        MultipleH5Files,
        LdpcDerateMatchModuleH5Test,
        ::testing::ValuesIn(ran::aerial_tv::TEST_HDF5_FILES),
        [](const ::testing::TestParamInfo<std::string_view> &test_info) {
            std::string name = std::filesystem::path(test_info.param).stem().string();
            // Replace non-alphanumeric characters with underscores for valid test
            // names
            std::replace_if(name.begin(), name.end(), [](char c) { return !std::isalnum(c); }, '_');
            return name;
        });

} // namespace

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
