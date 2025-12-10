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
#include <cstdint>
#include <filesystem>
#include <format>
#include <ios>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "aerial_tv/aerial_tv_utils.hpp"
#include "aerial_tv/cuphy_pusch_tv.hpp"
#include "ldpc/derate_match_params.hpp"
#include "ldpc/ldpc_params.hpp"
#include "ldpc/outer_rx_params.hpp"

namespace {

using ran::ldpc::DerateMatchParams;
using ran::ldpc::LdpcParams;
using ran::ldpc::ModulationOrder;
using ran::ldpc::NewDataIndicator;
using ran::ldpc::PuschOuterRxParams;
using ran::ldpc::SingleTbPuschOuterRxParams;

// Allow magic numbers in this file to check output against.
// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

/**
 * Test fixture for SingleTbPuschOuterRxParams tests
 */
class SingleTbPuschOuterRxParamsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a valid LdpcParams object for testing
        ldpc_params_ = std::make_unique<LdpcParams>(
                1000, // transport_block_size
                0.5F, // code_rate
                2000, // rate_matching_length
                0     // redundancy_version
        );
    }

    std::unique_ptr<LdpcParams> ldpc_params_;
};

/**
 * Test fixture for PuschOuterRxParams tests
 */
class PuschOuterRxParamsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create multiple valid LdpcParams objects for testing
        ldpc_params1_ = std::make_unique<LdpcParams>(1000, 0.5F, 2000, 0);
        ldpc_params2_ = std::make_unique<LdpcParams>(1500, 0.6F, 2500, 1);
        ldpc_params3_ = std::make_unique<LdpcParams>(2000, 0.7F, 3000, 2);
    }

    std::unique_ptr<LdpcParams> ldpc_params1_;
    std::unique_ptr<LdpcParams> ldpc_params2_;
    std::unique_ptr<LdpcParams> ldpc_params3_;
};

/**
 * Test constructor with all optional parameters provided
 */
TEST_F(SingleTbPuschOuterRxParamsTest, ConstructorAllParametersProvided) {
    const std::uint32_t mod_order = 4;
    const std::uint8_t n_dmrs_cdm_grps_no_data = 3;
    const std::uint32_t ndi = 1;
    const std::uint32_t num_layers = 2;
    const std::uint32_t user_group_idx = 5;
    const std::uint32_t num_ue_grp_layers = 3;
    const std::vector<std::uint32_t> layer_map{0, 1};
    const std::uint32_t scrambling_init = 0x1234;

    // Create DerateMatchParams object with test parameters
    DerateMatchParams de_rm_params{
            .mod_order = ModulationOrder::Qam16,
            .n_dmrs_cdm_grps_no_data = n_dmrs_cdm_grps_no_data,
            .ndi = static_cast<NewDataIndicator>(ndi),
            .num_layers = num_layers,
            .user_group_idx = user_group_idx,
            .num_ue_grp_layers = num_ue_grp_layers,
            .layer_map = layer_map,
            .scrambling_init = scrambling_init};

    const SingleTbPuschOuterRxParams params(*ldpc_params_, de_rm_params);

    // Verify all parameters are set correctly
    EXPECT_EQ(static_cast<std::uint32_t>(params.de_rm_params().mod_order), mod_order);
    EXPECT_EQ(params.de_rm_params().n_dmrs_cdm_grps_no_data, n_dmrs_cdm_grps_no_data);
    EXPECT_EQ(static_cast<std::uint32_t>(params.de_rm_params().ndi), ndi);
    EXPECT_EQ(params.de_rm_params().num_layers, num_layers);
    EXPECT_EQ(params.de_rm_params().user_group_idx, user_group_idx);
    EXPECT_EQ(params.de_rm_params().num_ue_grp_layers, num_ue_grp_layers);
    EXPECT_EQ(params.de_rm_params().layer_map, layer_map);
    EXPECT_EQ(params.de_rm_params().scrambling_init, scrambling_init);

    // Verify LDPC params are copied correctly
    EXPECT_EQ(params.ldpc_params().transport_block_size(), ldpc_params_->transport_block_size());
    EXPECT_FLOAT_EQ(params.ldpc_params().code_rate(), ldpc_params_->code_rate());
    EXPECT_EQ(params.ldpc_params().rate_matching_length(), ldpc_params_->rate_matching_length());
    EXPECT_EQ(params.ldpc_params().redundancy_version(), ldpc_params_->redundancy_version());
}

/**
 * Test layer mapping with different configurations
 */
TEST_F(SingleTbPuschOuterRxParamsTest, LayerMappingConfigurations) {
    struct TestCase {
        std::uint32_t num_layers;
        std::vector<std::uint32_t> expected_layer_map;
    };

    const std::array test_cases = std::to_array<TestCase>(
            {{1, {0}}, {2, {0, 1}}, {4, {0, 1, 2, 3}}, {8, {0, 1, 2, 3, 4, 5, 6, 7}}});

    for (const auto &test_case : test_cases) {
        // Create config with only the parameters we need to test
        DerateMatchParams de_rm_params{
                .mod_order = ModulationOrder::Qpsk,
                .n_dmrs_cdm_grps_no_data = 0,
                .ndi = NewDataIndicator::NewTransmission,
                .num_layers = test_case.num_layers,
                .user_group_idx = 0,
                .num_ue_grp_layers = 1,
                .layer_map = test_case.expected_layer_map,
                .scrambling_init = 0};

        const SingleTbPuschOuterRxParams params(*ldpc_params_, de_rm_params);

        EXPECT_EQ(params.de_rm_params().num_layers, test_case.num_layers);
        EXPECT_EQ(params.de_rm_params().layer_map, test_case.expected_layer_map);
    }
}

/**
 * Test basic constructor with single transport block
 */
TEST_F(PuschOuterRxParamsTest, ConstructorSingleTransportBlock) {
    // Create config with 16QAM modulation
    const DerateMatchParams de_rm_params{
            .mod_order = ModulationOrder::Qam16,
            .n_dmrs_cdm_grps_no_data = 0,
            .ndi = NewDataIndicator::NewTransmission,
            .num_layers = 1,
            .user_group_idx = 0,
            .num_ue_grp_layers = 1,
            .layer_map = {0},
            .scrambling_init = 0};

    const SingleTbPuschOuterRxParams single_tb_params(*ldpc_params1_, de_rm_params);

    const std::vector<SingleTbPuschOuterRxParams> tb_params{single_tb_params};
    const std::vector<std::uint16_t> sch_user_idxs{100};

    const PuschOuterRxParams params(tb_params, sch_user_idxs);

    // Test basic accessors
    EXPECT_EQ(params.get_num_sch_ues(), 1);
    EXPECT_EQ(params.get_sch_user_idxs().size(), 1);
    EXPECT_EQ(params.get_sch_user_idxs()[0], 100);

    // Test transport block parameter access
    const auto &retrieved_params = params.get_pusch_outer_rx_params_single_tb(0);
    EXPECT_EQ(retrieved_params.de_rm_params().mod_order, ModulationOrder::Qam16);
    EXPECT_EQ(
            retrieved_params.ldpc_params().transport_block_size(),
            ldpc_params1_->transport_block_size());
}

/**
 * Test constructor with multiple transport blocks
 */
TEST_F(PuschOuterRxParamsTest, ConstructorMultipleTransportBlocks) {
    // Create params with QPSK modulation (2)
    const DerateMatchParams de_rm_params1{
            .mod_order = ModulationOrder::Qpsk,
            .n_dmrs_cdm_grps_no_data = 0,
            .ndi = NewDataIndicator::NewTransmission,
            .num_layers = 1,
            .user_group_idx = 0,
            .num_ue_grp_layers = 1,
            .layer_map = {0},
            .scrambling_init = 0};
    const SingleTbPuschOuterRxParams tb_params1(*ldpc_params1_, de_rm_params1);

    // Create params with 16QAM modulation (4)
    const DerateMatchParams de_rm_params2{
            .mod_order = ModulationOrder::Qam16,
            .n_dmrs_cdm_grps_no_data = 0,
            .ndi = NewDataIndicator::NewTransmission,
            .num_layers = 1,
            .user_group_idx = 0,
            .num_ue_grp_layers = 1,
            .layer_map = {0},
            .scrambling_init = 0};
    const SingleTbPuschOuterRxParams tb_params2(*ldpc_params2_, de_rm_params2);

    // Create params with 64QAM modulation (6)
    const DerateMatchParams de_rm_params3{
            .mod_order = ModulationOrder::Qam64,
            .n_dmrs_cdm_grps_no_data = 0,
            .ndi = NewDataIndicator::NewTransmission,
            .num_layers = 1,
            .user_group_idx = 0,
            .num_ue_grp_layers = 1,
            .layer_map = {0},
            .scrambling_init = 0};
    const SingleTbPuschOuterRxParams tb_params3(*ldpc_params3_, de_rm_params3);

    const std::vector<SingleTbPuschOuterRxParams> tb_params{tb_params1, tb_params2, tb_params3};
    const std::vector<std::uint16_t> sch_user_idxs{100, 200, 300};

    const PuschOuterRxParams params(tb_params, sch_user_idxs);

    // Test basic accessors
    EXPECT_EQ(params.get_num_sch_ues(), 3);
    EXPECT_EQ(params.get_sch_user_idxs().size(), 3);

    // Test individual user indices
    EXPECT_EQ(params.get_sch_user_idxs()[0], 100);
    EXPECT_EQ(params.get_sch_user_idxs()[1], 200);
    EXPECT_EQ(params.get_sch_user_idxs()[2], 300);

    // Test transport block parameter access by index
    EXPECT_EQ(
            params.get_pusch_outer_rx_params_single_tb(0).de_rm_params().mod_order,
            ModulationOrder::Qpsk);
    EXPECT_EQ(
            params.get_pusch_outer_rx_params_single_tb(1).de_rm_params().mod_order,
            ModulationOrder::Qam16);
    EXPECT_EQ(
            params.get_pusch_outer_rx_params_single_tb(2).de_rm_params().mod_order,
            ModulationOrder::Qam64);

    // Test LDPC parameters
    EXPECT_EQ(
            params.get_pusch_outer_rx_params_single_tb(0).ldpc_params().transport_block_size(),
            ldpc_params1_->transport_block_size());
    EXPECT_EQ(
            params.get_pusch_outer_rx_params_single_tb(1).ldpc_params().transport_block_size(),
            ldpc_params2_->transport_block_size());
    EXPECT_EQ(
            params.get_pusch_outer_rx_params_single_tb(2).ldpc_params().transport_block_size(),
            ldpc_params3_->transport_block_size());
}

/**
 * Test GPU buffer pointer access
 */
TEST_F(PuschOuterRxParamsTest, GpuBufferAccess) {
    // Create derate matching parameters
    const DerateMatchParams de_rm_params{
            .mod_order = ModulationOrder::Qpsk,
            .n_dmrs_cdm_grps_no_data = 0,
            .ndi = NewDataIndicator::NewTransmission,
            .num_layers = 1,
            .user_group_idx = 0,
            .num_ue_grp_layers = 1,
            .layer_map = {0},
            .scrambling_init = 0};
    const SingleTbPuschOuterRxParams single_tb_params(*ldpc_params1_, de_rm_params);

    const std::vector<SingleTbPuschOuterRxParams> tb_params{single_tb_params};
    const std::vector<std::uint16_t> sch_user_idxs{200};

    const PuschOuterRxParams params(tb_params, sch_user_idxs);

    // Test that pointers are not null (buffers should be allocated)
    EXPECT_NE(params.get_per_tb_params_cpu_ptr(), nullptr);
    EXPECT_NE(params.get_per_tb_params_gpu_ptr(), nullptr);
}

/**
 * Test out of bounds access with at() method (should throw)
 */
TEST_F(PuschOuterRxParamsTest, OutOfBoundsAccess) {
    // Create derate matching parameters
    const DerateMatchParams de_rm_params{
            .mod_order = ModulationOrder::Qpsk,
            .n_dmrs_cdm_grps_no_data = 0,
            .ndi = NewDataIndicator::NewTransmission,
            .num_layers = 1,
            .user_group_idx = 0,
            .num_ue_grp_layers = 1,
            .layer_map = {0},
            .scrambling_init = 0};
    const SingleTbPuschOuterRxParams single_tb_params(*ldpc_params1_, de_rm_params);

    const std::vector<SingleTbPuschOuterRxParams> tb_params{single_tb_params};
    const std::vector<std::uint16_t> sch_user_idxs{100};

    const PuschOuterRxParams params(tb_params, sch_user_idxs);

    // Valid access should work
    EXPECT_NO_THROW(static_cast<void>(params.get_pusch_outer_rx_params_single_tb(0)));

    // Out of bounds access should throw
    EXPECT_THROW(
            static_cast<void>(params.get_pusch_outer_rx_params_single_tb(1)), std::out_of_range);
    EXPECT_THROW(
            static_cast<void>(params.get_pusch_outer_rx_params_single_tb(100)), std::out_of_range);
}

/**
 * Test validation: mismatched vector sizes between TB params and user indices
 */
TEST_F(PuschOuterRxParamsTest, MismatchedVectorSizes) {
    // Create derate matching parameters
    const DerateMatchParams de_rm_params{
            .mod_order = ModulationOrder::Qpsk,
            .n_dmrs_cdm_grps_no_data = 0,
            .ndi = NewDataIndicator::NewTransmission,
            .num_layers = 1,
            .user_group_idx = 0,
            .num_ue_grp_layers = 1,
            .layer_map = {0},
            .scrambling_init = 0};
    const SingleTbPuschOuterRxParams single_tb_params(*ldpc_params1_, de_rm_params);

    const std::vector<SingleTbPuschOuterRxParams> tb_params{single_tb_params};
    const std::vector<std::uint16_t> mismatched_sch_user_idxs{100, 200}; // Size mismatch

    // Should throw because implementation validates that sizes must match
    EXPECT_THROW(PuschOuterRxParams(tb_params, mismatched_sch_user_idxs), std::invalid_argument);
}

/**
 * Parameterized test fixture for get_rate_matching_length with H5 test vectors
 */
class RateMatchingLengthTest : public ::testing::TestWithParam<std::string_view> {
protected:
    void SetUp() override {
        const std::filesystem::path test_vector_dir{TEST_VECTOR_PATH};
        const std::filesystem::path full_file_path = test_vector_dir / GetParam();

        if (!std::filesystem::exists(full_file_path)) {
            FAIL() << "Test HDF5 file not found: " << full_file_path.string()
                   << "\nExpected location: " << test_vector_dir.string();
        }

        test_file_path_ = full_file_path.string();
    }

    [[nodiscard]] const std::string &get_test_file_path() const { return test_file_path_; }

private:
    std::string test_file_path_;
};

/**
 * Test get_rate_matching_length against reference values from test vectors
 */
TEST_P(RateMatchingLengthTest, CompareWithTestVectorReference) {
    const std::string &test_file_path = get_test_file_path();
    const ran::aerial_tv::CuphyPuschTestVector test_vector{test_file_path};

    const auto &tb_params_vec = test_vector.get_tb_params();
    const auto &ue_grp_params_vec = test_vector.get_ue_grp_params();

    for (const auto &tb_params : tb_params_vec) {
        const auto user_group_idx = tb_params.user_group_index;
        const auto mod_order = tb_params.qam_mod_order;

        // Get reference rate matching length from the actual LLRs.
        const auto de_rm_input_array =
                test_vector.read_array<float>(std::format("reference_eqOutLLRs{}", user_group_idx));
        const auto &de_rm_input = de_rm_input_array.data;
        constexpr std::uint8_t QAM_256_ORDER = 8;
        const std::uint32_t reference_rate_matching_len =
                static_cast<std::uint32_t>(de_rm_input.size()) * mod_order / QAM_256_ORDER;

        // Get UE group parameters for this transport block
        const auto &ue_grp_params = ue_grp_params_vec[user_group_idx];

        // Calculate rate matching length using the function under test
        const std::uint32_t calculated_rate_matching_len = ran::ldpc::get_rate_matching_length(
                ue_grp_params.n_prb,
                static_cast<std::uint8_t>(tb_params.num_layers),
                static_cast<ran::ldpc::ModulationOrder>(mod_order),
                ue_grp_params.nr_of_symbols,
                tb_params.num_dmrs_cdm_grps_no_data,
                ue_grp_params.dmrs_sym_loc_bmsk);

        // Compare calculated value with reference
        EXPECT_EQ(calculated_rate_matching_len, reference_rate_matching_len)
                << "Mismatch for user_group_idx=" << user_group_idx << ", mod_order=" << mod_order
                << ", num_prb=" << ue_grp_params.n_prb << ", num_layers=" << tb_params.num_layers
                << ", num_symbols=" << ue_grp_params.nr_of_symbols << ", num_dmrs_cdm_grps_no_data="
                << static_cast<int>(tb_params.num_dmrs_cdm_grps_no_data) << ", dmrs_sym_pos=0x"
                << std::hex << ue_grp_params.dmrs_sym_loc_bmsk;
    }
}

INSTANTIATE_TEST_SUITE_P(
        MultipleH5Files,
        RateMatchingLengthTest,
        ::testing::ValuesIn(ran::aerial_tv::TEST_HDF5_FILES),
        [](const ::testing::TestParamInfo<std::string_view> &test_info) {
            std::string name = std::filesystem::path(test_info.param).stem().string();
            std::replace_if(
                    name.begin(), name.end(), [](const char c) { return !std::isalnum(c); }, '_');
            return name;
        });

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

} // namespace
