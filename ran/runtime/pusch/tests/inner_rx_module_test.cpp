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
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <driver_types.h>

#include <gtest/gtest.h>

#include <cuda_fp16.h>

#include "aerial_tv/aerial_tv_utils.hpp"
#include "aerial_tv/cuphy_pusch_tv.hpp"
#include "inner_rx_module_runner.hpp"
#include "memory/unique_ptr_utils.hpp"
#include "pipeline/types.hpp"
#include "pusch/inner_rx_module.hpp"
#include "pusch_test_utils.hpp"
#include "ran_common.hpp"
#include "utils/cuda_stream.hpp"

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
namespace {

using ran::pusch::InnerRxModule;
namespace pipeline = framework::pipeline;
using framework::memory::UniqueDevicePtr;
using framework::utils::CudaStream;

/**
 * Base test fixture for InnerRxModule with configurable PhyParams
 *
 * Provides common test infrastructure including module initialization.
 * Derived classes must override setup_phy_params() to configure their specific PhyParams source.
 */
class InnerRxModuleTestBase : public ::testing::Test {
protected:
    void SetUp() override {
        // Let derived class configure phy_params_
        setup_phy_params();

        // Create InnerRxModule instance
        const InnerRxModule::StaticParams params{
                .phy_params = phy_params_, .execution_mode = pipeline::ExecutionMode::Stream};

        inner_rx_module_ = std::make_unique<InnerRxModule>("inner_rx_test_instance", params);
    }

    /**
     * Configure PhyParams for testing
     *
     * Derived classes must implement this to provide their specific configuration source
     * (e.g., default values, test vectors, configuration files).
     */
    virtual void setup_phy_params() = 0;

    ran::common::PhyParams phy_params_{};
    std::unique_ptr<InnerRxModule> inner_rx_module_;
};

/**
 * Test fixture using dummy PhyParams values
 *
 * Tests module behavior with dummy configuration suitable for
 * unit testing specific functionality without requiring test vector files.
 */
class InnerRxModuleTest : public InnerRxModuleTestBase {
protected:
    void setup_phy_params() override {
        // Use default values appropriate for unit testing
        phy_params_ = ran::common::PhyParams{
                .num_rx_ant = 4,
                .cyclic_prefix = ran::common::CYCLIC_PREFIX_NORMAL,
                .bandwidth = 100,
                .num_prb = 273};
    }
};

/**
 * Parameterized test fixture using PhyParams from H5 test vectors
 *
 * Tests module end-to-end with real test vector data to validate
 * against reference implementations.
 */
class InnerRxModuleH5Test : public InnerRxModuleTestBase,
                            public ::testing::WithParamInterface<std::string_view> {
protected:
    void setup_phy_params() override {
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

        // Load test vector and extract PhyParams
        test_vector_ =
                std::make_unique<ran::aerial_tv::CuphyPuschTestVector>(test_file_path_.c_str());
        phy_params_ = ran::aerial_tv::to_phy_params(*test_vector_);
    }

    /**
     * Get the full path to the test vector file
     *
     * @return Full path to the loaded H5 test vector file
     */
    [[nodiscard]] const std::string &get_test_file_path() const { return test_file_path_; }

    /**
     * Get the loaded test vector
     *
     * @return Reference to the CUPHY PUSCH test vector
     */
    [[nodiscard]] const ran::aerial_tv::CuphyPuschTestVector &get_test_vector() const {
        return *test_vector_;
    }

private:
    std::string test_file_path_;
    std::unique_ptr<ran::aerial_tv::CuphyPuschTestVector> test_vector_;
};

/**
 * Test successful module identification.
 */
TEST_F(InnerRxModuleTest, ModuleIdentification) {

    // Test type ID
    EXPECT_STREQ(inner_rx_module_->get_type_id().data(), "inner_rx_module");

    // Test instance ID
    EXPECT_STREQ(inner_rx_module_->get_instance_id().data(), "inner_rx_test_instance");
}

/**
 * Verifies module correctly reports its input and output port names.
 */
TEST_F(InnerRxModuleTest, PortNames) {

    // Test input port names
    const auto input_ports = inner_rx_module_->get_input_port_names();
    EXPECT_EQ(input_ports.size(), 1);
    EXPECT_EQ(input_ports[0], "xtf");

    // Test output port names
    const auto outputs = inner_rx_module_->get_output_port_names();
    EXPECT_EQ(outputs.size(), 3);
    EXPECT_EQ(outputs[0], "llrs");
    EXPECT_EQ(outputs[1], "post_eq_noise_var_db");
    EXPECT_EQ(outputs[2], "post_eq_sinr_db");
}

/**
 * Helper function to test InnerRxModule using runner
 * Validates outputs against reference data for the given execution mode.
 *
 * @param[in] test_vector_path Path to H5 test vector file
 * @param[in] execution_mode Execution mode (Stream or Graph)
 */
void run_inner_rx_runner_test(
        const std::string &test_vector_path, const pipeline::ExecutionMode execution_mode) {
    // Create runner
    ran::pusch::InnerRxModuleRunner runner(test_vector_path, execution_mode);
    const CudaStream stream;

    // Configure (includes warmup)
    ASSERT_NO_THROW(runner.configure(stream));

    // Execute
    ASSERT_NO_THROW(runner.execute_once(stream));
    ASSERT_TRUE(stream.synchronize());

    // Get test vector for validation
    const auto &test_vector = runner.get_test_vector();

    // Verify outputs
    const auto &outputs = runner.get_outputs();
    EXPECT_EQ(outputs.size(), 3);
    EXPECT_EQ(outputs[0].name, "llrs");
    EXPECT_EQ(outputs[1].name, "post_eq_noise_var_db");
    EXPECT_EQ(outputs[2].name, "post_eq_sinr_db");

    // Read reference values from the test vector
    const auto ref_llrs_array = test_vector.read_array<float>("reference_eqOutLLRs0");
    const auto &ref_llrs = ref_llrs_array.data;

    // Copy outputs using stream-aware async operations for proper synchronization
    const auto post_eq_noise_var_db = ran::pusch::tensor_to_host_vector<float>(
            outputs[1].tensors[0].tensor_info, outputs[1].tensors[0].device_ptr, stream.get());
    const auto post_eq_sinr_db = ran::pusch::tensor_to_host_vector<float>(
            outputs[2].tensors[0].tensor_info, outputs[2].tensors[0].device_ptr, stream.get());

    // Check post-EQ noise variance and SINR using utility functions, 0.5 dB tolerance.
    ran::aerial_tv::check_post_eq_noise_var(post_eq_noise_var_db, test_vector, 0.5F);
    ran::aerial_tv::check_post_eq_sinr(post_eq_sinr_db, test_vector, 0.5F);

    const auto llrs_half = ran::pusch::tensor_to_host_vector<__half>(
            outputs[0].tensors[0].tensor_info, outputs[0].tensors[0].device_ptr, stream.get());
    ASSERT_EQ(llrs_half.size(), ref_llrs.size());

    // Convert to float
    std::vector<float> llrs(llrs_half.size());
    std::transform(llrs_half.begin(), llrs_half.end(), llrs.begin(), [](const __half value) {
        return __half2float(value);
    });

    // Count LLR sign errors
    // Note: Reference LLRs are padded with zeros to 256QAM, so we need to ignore them.
    std::size_t sign_errors = 0;
    for (std::size_t i = 0; i < ref_llrs.size(); ++i) {
        if (ref_llrs[i] != 0.0F && ((llrs[i] > 0.0F) != (ref_llrs[i] > 0.0F))) {
            ++sign_errors;
        }
    }
    const double error_percentage =
            100.0 * static_cast<double>(sign_errors) / static_cast<double>(ref_llrs.size());

    // Verify sign error rate is below 1%
    ASSERT_LT(error_percentage, 1.0) << "LLR sign error rate exceeds 1%";
}

/**
 * Test InnerRxModule using runner for stream mode
 */
TEST_P(InnerRxModuleH5Test, RunnerStreamMode) {
    run_inner_rx_runner_test(get_test_file_path(), pipeline::ExecutionMode::Stream);
}

/**
 * Test InnerRxModule using runner for graph mode
 */
TEST_P(InnerRxModuleH5Test, RunnerGraphMode) {
    run_inner_rx_runner_test(get_test_file_path(), pipeline::ExecutionMode::Graph);
}

// Instantiate the parameterized test with all available H5 test files
INSTANTIATE_TEST_SUITE_P(
        MultipleH5Files,
        InnerRxModuleH5Test,
        ::testing::ValuesIn(ran::aerial_tv::TEST_HDF5_FILES),
        [](const ::testing::TestParamInfo<std::string_view> &test_info) {
            std::string name = std::filesystem::path(test_info.param).stem().string();
            // Replace non-alphanumeric characters with underscores for valid test names
            std::replace_if(
                    name.begin(), name.end(), [](const char c) { return !std::isalnum(c); }, '_');
            return name;
        });

} // namespace

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
