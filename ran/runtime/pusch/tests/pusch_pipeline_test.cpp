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
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <driver_types.h>

#include <gsl-lite/gsl-lite.hpp>
#include <gtest/gtest.h>
#include <wise_enum.h>

#include <cuda_runtime_api.h>

#include "aerial_tv/aerial_tv_utils.hpp"
#include "ldpc/outer_rx_params.hpp"
#include "log/rt_log_macros.hpp"
#include "pipeline/imodule_factory.hpp"
#include "pipeline/types.hpp"
#include "pusch/pusch_defines.hpp"
#include "pusch/pusch_module_factories.hpp"
#include "pusch/pusch_pipeline.hpp"
#include "pusch_pipeline_runner.hpp"
#include "pusch_test_utils.hpp"
#include "ran_common.hpp"
#include "tensor/data_types.hpp"
#include "tensor/tensor_info.hpp"
#include "utils/cuda_stream.hpp"

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
namespace {

namespace tensor = framework::tensor;
namespace pipeline = framework::pipeline;

using framework::utils::CudaStream;
using ran::common::PhyParams;
using ran::pusch::PuschModuleFactory;
using ran::pusch::PuschPipeline;

/**
 * Abstract base test fixture for all PUSCH pipeline tests
 *
 * Defines the contract for pipeline creation and provides common infrastructure.
 * Derived classes must implement create_pipeline() to specify how the pipeline is created.
 */
class PuschPipelineTestBase : public ::testing::Test {
protected:
    void SetUp() override {
        // Create module factory
        module_factory_ = std::make_unique<PuschModuleFactory>();

        // Let derived class create the pipeline
        pusch_pipeline_ = create_pipeline();
    }

    /**
     * Create and configure the PUSCH pipeline
     *
     * Derived classes must implement this to provide their specific pipeline configuration.
     *
     * @return Unique pointer to the created pipeline
     */
    [[nodiscard]] virtual std::unique_ptr<PuschPipeline> create_pipeline() = 0;

    PhyParams phy_params_{};
    std::unique_ptr<PuschModuleFactory> module_factory_;
    std::unique_ptr<PuschPipeline> pusch_pipeline_;
};

/**
 * Test fixture using dummy PhyParams values
 *
 * Tests pipeline behavior with dummy configuration suitable for
 * unit testing specific functionality without requiring test vector files.
 */
class PuschPipelineTest : public PuschPipelineTestBase {
protected:
    [[nodiscard]] std::unique_ptr<PuschPipeline> create_pipeline() override {
        // Use default values appropriate for unit testing
        phy_params_ = PhyParams{
                .num_rx_ant = 4,
                .cyclic_prefix = ran::common::CYCLIC_PREFIX_NORMAL,
                .bandwidth = 100,
                .num_prb = 273};

        const pipeline::PipelineSpec spec =
                ran::pusch::create_pusch_pipeline_spec("pusch_pipeline", phy_params_);

        return std::make_unique<PuschPipeline>(
                "test_pipeline",
                gsl_lite::not_null<pipeline::IModuleFactory *>(module_factory_.get()),
                spec);
    }
};

/**
 * Abstract base test fixture for PUSCH pipeline tests using H5 test vectors
 *
 * Provides common test logic for both Stream and Graph execution modes.
 * Derived classes must implement get_execution_mode() to specify the execution mode.
 */
class PuschPipelineH5TestBase : public ::testing::Test,
                                public ::testing::WithParamInterface<std::string_view> {
protected:
    /**
     * Get execution mode for the test
     *
     * Derived classes must implement this to specify Stream or Graph mode.
     */
    [[nodiscard]] virtual pipeline::ExecutionMode get_execution_mode() const = 0;

    /**
     * Get the full path to the test vector file
     */
    [[nodiscard]] static std::string get_test_vector_path() {
        const std::filesystem::path test_vector_dir{TEST_VECTOR_PATH};
        const std::filesystem::path full_file_path = test_vector_dir / GetParam();

        // Check if the file exists and fail the test directly if not
        if (!std::filesystem::exists(full_file_path)) {
            ADD_FAILURE() << "Test HDF5 file not found: " << full_file_path.string()
                          << "\nExpected location: " << test_vector_dir.string();
            return "";
        }

        return full_file_path.string();
    }

    /**
     * Common test logic for full pipeline test
     *
     * Executes the complete pipeline flow and validates outputs.
     */
    void run_full_pipeline_test();
};

/**
 * Test fixture for PUSCH pipeline in Stream execution mode
 */
class PuschPipelineH5StreamTest : public PuschPipelineH5TestBase {
protected:
    [[nodiscard]] pipeline::ExecutionMode get_execution_mode() const override {
        return pipeline::ExecutionMode::Stream;
    }
};

/**
 * Test fixture for PUSCH pipeline in Graph execution mode
 */
class PuschPipelineH5GraphTest : public PuschPipelineH5TestBase {
protected:
    [[nodiscard]] pipeline::ExecutionMode get_execution_mode() const override {
        return pipeline::ExecutionMode::Graph;
    }
};

/**
 * Test pipeline setup phase without external data.
 */
TEST_F(PuschPipelineTest, BasicSetup) {

    // Pipeline setup should succeed
    ASSERT_NO_THROW(pusch_pipeline_->setup());
}

/**
 * Test full PUSCH pipeline in Stream mode using H5 test vectors.
 * This test loads input data from H5 file, executes the complete pipeline,
 * and validates the final output against expected reference data.
 */
TEST_P(PuschPipelineH5StreamTest, FullPipelineTest) { run_full_pipeline_test(); }

/**
 * Test full PUSCH pipeline in Graph mode using H5 test vectors.
 * Tests the graph-based execution path with dynamic graph updates.
 */
TEST_P(PuschPipelineH5GraphTest, FullPipelineTest) { run_full_pipeline_test(); }

/**
 * Common test logic for full pipeline test
 *
 * Executes the complete pipeline flow and validates outputs using PuschPipelineRunner.
 */
void PuschPipelineH5TestBase::run_full_pipeline_test() {

    // Create a CUDA stream
    const CudaStream stream;

    // Create pipeline runner
    ran::pusch::PuschPipelineRunner runner{get_test_vector_path(), get_execution_mode()};

    // Configure pipeline
    ASSERT_NO_THROW({ runner.configure(stream); });

    // Execute pipeline
    RT_LOG_DEBUG(
            "Executing PUSCH pipeline in mode: {}",
            ::wise_enum::to_string(runner.get_execution_mode()));
    runner.execute_once(stream);

    // Synchronize stream to ensure all GPU operations are complete before reading values
    ASSERT_TRUE(stream.synchronize());
    RT_LOG_DEBUG("Finished executing PUSCH pipeline");

    // Get test vector and outer_rx parameters for validation
    const auto &test_vector = runner.get_test_vector();
    const auto pusch_outer_rx_params = ran::aerial_tv::to_pusch_outer_rx_params(test_vector);
    const auto &external_outputs = runner.get_external_outputs();

    // Verify external outputs
    ASSERT_EQ(external_outputs.size(), ran::pusch::NUM_EXTERNAL_OUTPUTS);
    ASSERT_EQ(runner.get_num_external_outputs(), ran::pusch::NUM_EXTERNAL_OUTPUTS);

    // Verify external output names match expected indices
    EXPECT_EQ(external_outputs[0].name, "post_eq_noise_var_db")
            << "Index 0 should be post_eq_noise_var_db";
    EXPECT_EQ(external_outputs[1].name, "post_eq_sinr_db") << "Index 1 should be post_eq_sinr_db";
    EXPECT_EQ(external_outputs[2].name, "tb_crcs") << "Index 2 should be tb_crcs";
    EXPECT_EQ(external_outputs[3].name, "tb_payloads") << "Index 3 should be tb_payloads";

    // Verify output tensor information
    EXPECT_EQ(
            external_outputs[0].tensors[0].tensor_info.get_type(),
            tensor::NvDataType::TensorR32F); // Post-EQ noise var db
    EXPECT_EQ(
            external_outputs[1].tensors[0].tensor_info.get_type(),
            tensor::NvDataType::TensorR32F); // Post-EQ SINR db
    EXPECT_NE(external_outputs[0].tensors[0].device_ptr, nullptr);
    EXPECT_NE(external_outputs[1].tensors[0].device_ptr, nullptr);
    const auto num_tbs = pusch_outer_rx_params.num_tbs();
    for (std::size_t tb_idx = 0; tb_idx < num_tbs; ++tb_idx) {

        EXPECT_EQ(
                external_outputs[2].tensors[tb_idx].tensor_info.get_type(),
                tensor::NvDataType::TensorR32U); // TB CRCs
        EXPECT_EQ(
                external_outputs[3].tensors[tb_idx].tensor_info.get_type(),
                tensor::NvDataType::TensorR8U); // TB payloads

        // Check that outputs are not null.
        EXPECT_NE(external_outputs[2].tensors[tb_idx].device_ptr, nullptr);
        EXPECT_NE(external_outputs[3].tensors[tb_idx].device_ptr, nullptr);
    }

    // Verify post-EQ noise variance and SINR
    const auto post_eq_noise_var_db = ran::pusch::tensor_to_host_vector<float>(
            external_outputs[0].tensors[0].tensor_info,
            external_outputs[0].tensors[0].device_ptr,
            stream.get());
    const auto post_eq_sinr_db = ran::pusch::tensor_to_host_vector<float>(
            external_outputs[1].tensors[0].tensor_info,
            external_outputs[1].tensors[0].device_ptr,
            stream.get());
    ran::aerial_tv::check_post_eq_noise_var(post_eq_noise_var_db, test_vector, 0.5F);
    ran::aerial_tv::check_post_eq_sinr(post_eq_sinr_db, test_vector, 0.5F);

    // Verify transport block payloads
    ran::aerial_tv::check_tb_payloads(external_outputs[3].tensors, test_vector, stream.get());

    // Verify transport block CRCs pass
    std::vector<std::uint32_t> tb_crc_values(num_tbs);
    ASSERT_EQ(
            cudaMemcpyAsync(
                    tb_crc_values.data(),
                    external_outputs[2].tensors[0].device_ptr,
                    sizeof(std::uint32_t) * num_tbs,
                    cudaMemcpyDeviceToHost,
                    stream.get()),
            cudaSuccess);
    // Synchronize after async copy before reading host data
    ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);
    for (std::size_t tb_idx = 0; tb_idx < num_tbs; ++tb_idx) {
        EXPECT_EQ(tb_crc_values[tb_idx], 0U) << "TB " << tb_idx << " CRC check failed";
    }
}

/**
 * Instantiate parameterized test for Stream mode with all test vector files.
 */
INSTANTIATE_TEST_SUITE_P(
        StreamMode,
        PuschPipelineH5StreamTest,
        ::testing::ValuesIn(ran::aerial_tv::TEST_HDF5_FILES),
        [](const ::testing::TestParamInfo<std::string_view> &test_info) {
            std::string name = std::filesystem::path(test_info.param).stem().string();
            // Replace non-alphanumeric characters with underscores for valid test names
            std::replace_if(name.begin(), name.end(), [](char c) { return !std::isalnum(c); }, '_');
            return name;
        });

/**
 * Instantiate parameterized test for Graph mode with all test vector files.
 */
INSTANTIATE_TEST_SUITE_P(
        GraphMode,
        PuschPipelineH5GraphTest,
        ::testing::ValuesIn(ran::aerial_tv::TEST_HDF5_FILES),
        [](const ::testing::TestParamInfo<std::string_view> &test_info) {
            std::string name = std::filesystem::path(test_info.param).stem().string();
            // Replace non-alphanumeric characters with underscores for valid test names
            std::replace_if(name.begin(), name.end(), [](char c) { return !std::isalnum(c); }, '_');
            return name;
        });

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

} // namespace
