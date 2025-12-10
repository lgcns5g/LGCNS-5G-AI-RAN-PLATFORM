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
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "aerial_tv/cuphy_pusch_tv.hpp"

namespace {

constexpr std::array<std::string_view, 1> TEST_HDF5_FILES{"TVnr_7204_PUSCH_gNB_CUPHY_s0p0.h5"};

/**
 * Parameterized test fixture for CuphyPuschTestVector with H5 test vector files
 */
class CuphyPuschTestVectorH5Test : public ::testing::Test,
                                   public ::testing::WithParamInterface<std::string_view> {
protected:
    void SetUp() override {
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

TEST_P(CuphyPuschTestVectorH5Test, ConstructorValidFileCreatesObject) {
    const std::string &test_file = get_test_file_path();

    EXPECT_NO_THROW({ const ran::aerial_tv::CuphyPuschTestVector test_vector(test_file); });
}

TEST_P(CuphyPuschTestVectorH5Test, ReadGnbParamsValidFileReadsSuccessfully) {
    const std::string &test_file = get_test_file_path();

    const auto gnb_params =
            ran::aerial_tv::CuphyPuschTestVector::read_gnb_params_from_file(test_file);

    EXPECT_GE(gnb_params.n_user_groups, 0.0);
    EXPECT_GE(gnb_params.n_rx, 0U);
    EXPECT_GE(gnb_params.n_prb, 0U);
    EXPECT_LE(gnb_params.mu, 5U);
}

TEST_P(CuphyPuschTestVectorH5Test, ReadUeGrpParamsValidFileReadsSuccessfully) {
    const std::string &test_file = get_test_file_path();

    const auto ue_grp_params_vec =
            ran::aerial_tv::CuphyPuschTestVector::read_ue_grp_params_from_file(test_file);

    ASSERT_GE(ue_grp_params_vec.size(), 1U);
    const auto &ue_grp_params = ue_grp_params_vec[0];

    EXPECT_GE(ue_grp_params.n_ues, 0U);
    EXPECT_GE(ue_grp_params.n_prb, 0U);
    EXPECT_GE(ue_grp_params.n_uplink_streams, 0.0);
    EXPECT_LE(ue_grp_params.start_symbol_index, 13U); // Max 14 symbols per slot
    EXPECT_LE(ue_grp_params.nr_of_symbols, 14U);
}

TEST_P(CuphyPuschTestVectorH5Test, ReadTbParamsValidFileReadsSuccessfully) {
    const std::string &test_file = get_test_file_path();

    const auto tb_params_vec =
            ran::aerial_tv::CuphyPuschTestVector::read_tb_params_from_file(test_file);

    ASSERT_GE(tb_params_vec.size(), 1U);
    const auto &tb_params = tb_params_vec[0];

    EXPECT_GE(tb_params.num_layers, 0U);
    EXPECT_LE(tb_params.mcs_index, 31U); // MCS index is 0-31
    EXPECT_LE(tb_params.rv, 3U);         // Redundancy version is 0-3
    EXPECT_GE(tb_params.n_tb_byte, 0U);
}

TEST_P(CuphyPuschTestVectorH5Test, GetGnbParamsValidFileLazyLoadsSuccessfully) {
    const std::string &test_file = get_test_file_path();

    const ran::aerial_tv::CuphyPuschTestVector test_vector(test_file);

    EXPECT_NO_THROW({
        const auto &params = test_vector.get_gnb_params();
        EXPECT_GE(params.n_user_groups, 0.0);
    });
}

TEST_P(CuphyPuschTestVectorH5Test, GetUeGrpParamsValidFileLazyLoadsSuccessfully) {
    const std::string &test_file = get_test_file_path();

    const ran::aerial_tv::CuphyPuschTestVector test_vector(test_file);

    EXPECT_NO_THROW({
        const auto &params_vec = test_vector.get_ue_grp_params();
        ASSERT_GE(params_vec.size(), 1U);
        EXPECT_GE(params_vec[0].n_ues, 0U);
    });
}

TEST_P(CuphyPuschTestVectorH5Test, GetTbParamsValidFileLazyLoadsSuccessfully) {
    const std::string &test_file = get_test_file_path();

    const ran::aerial_tv::CuphyPuschTestVector test_vector(test_file);

    EXPECT_NO_THROW({
        const auto &params_vec = test_vector.get_tb_params();
        ASSERT_GE(params_vec.size(), 1U);
        EXPECT_GE(params_vec[0].num_layers, 0U);
    });
}

// ============================================================================
// Parameter Value Verification Tests
// ============================================================================

TEST_P(CuphyPuschTestVectorH5Test, ReadGnbParamsValidFileReturnsExpectedValues) {
    const std::string &test_file = get_test_file_path();

    const auto gnb_params =
            ran::aerial_tv::CuphyPuschTestVector::read_gnb_params_from_file(test_file);

    EXPECT_EQ(1.0, gnb_params.n_user_groups);
    EXPECT_EQ(1U, gnb_params.mu);
    EXPECT_EQ(4U, gnb_params.n_rx);
    EXPECT_EQ(273U, gnb_params.n_prb);
    EXPECT_EQ(0U, gnb_params.cell_id);
    EXPECT_EQ(0U, gnb_params.slot_number);
    EXPECT_EQ(1U, gnb_params.num_tb);
    EXPECT_EQ(1U, gnb_params.enable_early_harq);
    EXPECT_EQ(0U, gnb_params.enable_cfo_correction);
    EXPECT_EQ(0U, gnb_params.enable_cfo_estimation);
}

TEST_P(CuphyPuschTestVectorH5Test, ReadUeGrpParamsValidFileReturnsExpectedValues) {
    const std::string &test_file = get_test_file_path();

    const auto ue_grp_params_vec =
            ran::aerial_tv::CuphyPuschTestVector::read_ue_grp_params_from_file(test_file);

    ASSERT_GE(ue_grp_params_vec.size(), 1U);
    const auto &ue_grp_params = ue_grp_params_vec[0];

    EXPECT_EQ(1U, ue_grp_params.n_ues);
    EXPECT_EQ(0U, ue_grp_params.ue_prm_idxs[0]);
    EXPECT_EQ(0U, ue_grp_params.start_prb);
    EXPECT_EQ(273U, ue_grp_params.n_prb);
    EXPECT_EQ(0U, ue_grp_params.start_symbol_index);
    EXPECT_EQ(14U, ue_grp_params.nr_of_symbols);
    EXPECT_EQ(273U, ue_grp_params.prg_size);
    EXPECT_EQ(4U, ue_grp_params.dmrs_sym_loc_bmsk);
    EXPECT_EQ(4U, ue_grp_params.rssi_sym_loc_bmsk);
    EXPECT_EQ(4.0, ue_grp_params.n_uplink_streams);
}

TEST_P(CuphyPuschTestVectorH5Test, ReadTbParamsValidFileReturnsExpectedValues) {
    const std::string &test_file = get_test_file_path();

    const auto tb_params_vec =
            ran::aerial_tv::CuphyPuschTestVector::read_tb_params_from_file(test_file);

    ASSERT_GE(tb_params_vec.size(), 1U);
    const auto &tb_params = tb_params_vec[0];

    EXPECT_EQ(0U, tb_params.n_rnti);
    EXPECT_EQ(1U, tb_params.num_layers);
    EXPECT_EQ(0U, tb_params.start_sym);
    EXPECT_EQ(14U, tb_params.num_sym);
    EXPECT_EQ(0U, tb_params.user_group_index);
    EXPECT_EQ(0U, tb_params.data_scram_id);
    EXPECT_EQ(2U, tb_params.mcs_table_index);
    EXPECT_EQ(27U, tb_params.mcs_index);
    EXPECT_EQ(0U, tb_params.rv);
    EXPECT_EQ(1U, tb_params.ndi);
}

// ============================================================================
// Scalar Reading Tests
// ============================================================================

TEST_P(CuphyPuschTestVectorH5Test, ReadScalarFloatDatasetReturnsCorrectValue) {
    const std::string &test_file = get_test_file_path();

    const ran::aerial_tv::CuphyPuschTestVector test_vector(test_file);

    EXPECT_NO_THROW({
        const auto rxx_inv = test_vector.read_scalar<float>("RxxInv");
        EXPECT_FLOAT_EQ(1.0F, rxx_inv);

        const auto cfo_angle = test_vector.read_scalar<float>("reference_cfoAngle");
        EXPECT_FLOAT_EQ(0.0F, cfo_angle);

        const auto noise_var = test_vector.read_scalar<float>("reference_noiseVardB");
        EXPECT_NEAR(-39.728413F, noise_var, 1e-5F);
    });
}

TEST_P(CuphyPuschTestVectorH5Test, ReadScalarNonexistentDatasetThrowsException) {
    const std::string &test_file = get_test_file_path();

    const ran::aerial_tv::CuphyPuschTestVector test_vector(test_file);

    EXPECT_THROW(
            { (void)test_vector.read_scalar<float>("nonexistent_dataset"); }, std::runtime_error);
}

TEST_P(CuphyPuschTestVectorH5Test, ReadArraySmallUintArrayReturnsCorrectData) {
    const std::string &test_file = get_test_file_path();

    const ran::aerial_tv::CuphyPuschTestVector test_vector(test_file);

    const auto array = test_vector.read_array<std::uint16_t>("StartPrb");
    const auto &data = array.data;
    const auto &dimensions = array.dimensions;

    EXPECT_EQ(2U, dimensions.size());
    EXPECT_EQ(1U, dimensions[0]);
    EXPECT_EQ(1U, dimensions[1]);
    EXPECT_EQ(1U, data.size());
    EXPECT_EQ(0U, data[0]);
}

TEST_P(CuphyPuschTestVectorH5Test, ReadArrayLargeFloatArrayReturnsCorrectDimensions) {
    const std::string &test_file = get_test_file_path();

    const ran::aerial_tv::CuphyPuschTestVector test_vector(test_file);

    const auto array = test_vector.read_array<float>("WFreq");
    const auto &data = array.data;
    const auto &dimensions = array.dimensions;

    EXPECT_EQ(3U, dimensions.size()); // Should be 3D: (3, 48, 49)
    EXPECT_EQ(3U, dimensions[0]);
    EXPECT_EQ(48U, dimensions[1]);
    EXPECT_EQ(49U, dimensions[2]);

    const std::size_t expected_size = dimensions[0] * dimensions[1] * dimensions[2];
    EXPECT_EQ(expected_size, data.size());
    EXPECT_EQ(7056U, data.size()); // 3 * 48 * 49 = 7056

    // Verify data contains finite values
    for (const auto &val : data) {
        EXPECT_TRUE(std::isfinite(val));
    }
}

TEST_P(CuphyPuschTestVectorH5Test, ReadArrayUintArrayReturnsCorrectType) {
    const std::string &test_file = get_test_file_path();

    const ran::aerial_tv::CuphyPuschTestVector test_vector(test_file);

    const auto array = test_vector.read_array<std::uint8_t>("Data_sym_loc");
    const auto &data = array.data;
    const auto &dimensions = array.dimensions;

    EXPECT_EQ(2U, dimensions.size()); // Should be 2D: (1, 13)
    EXPECT_EQ(1U, dimensions[0]);
    EXPECT_EQ(13U, dimensions[1]);
    EXPECT_EQ(13U, data.size());

    for (const auto &val : data) {
        EXPECT_LE(val, 255U);
    }
}

TEST_P(CuphyPuschTestVectorH5Test, ReadArrayNonexistentDatasetThrowsException) {
    const std::string &test_file = get_test_file_path();

    const ran::aerial_tv::CuphyPuschTestVector test_vector(test_file);

    EXPECT_THROW(
            { (void)test_vector.read_array<float>("nonexistent_array_dataset"); },
            std::runtime_error);
}

TEST_P(CuphyPuschTestVectorH5Test, GetParamsLazyLoadingReturnsSameValuesAsDirectRead) {
    const std::string &test_file = get_test_file_path();

    const ran::aerial_tv::CuphyPuschTestVector test_vector(test_file);

    const auto &gnb_params_lazy = test_vector.get_gnb_params();
    const auto &ue_grp_params_lazy_vec = test_vector.get_ue_grp_params();
    const auto &tb_params_lazy_vec = test_vector.get_tb_params();

    const auto gnb_params_direct =
            ran::aerial_tv::CuphyPuschTestVector::read_gnb_params_from_file(test_file);
    const auto ue_grp_params_direct_vec =
            ran::aerial_tv::CuphyPuschTestVector::read_ue_grp_params_from_file(test_file);
    const auto tb_params_direct_vec =
            ran::aerial_tv::CuphyPuschTestVector::read_tb_params_from_file(test_file);

    EXPECT_EQ(gnb_params_direct.n_user_groups, gnb_params_lazy.n_user_groups);
    EXPECT_EQ(gnb_params_direct.mu, gnb_params_lazy.mu);
    EXPECT_EQ(gnb_params_direct.n_rx, gnb_params_lazy.n_rx);
    EXPECT_EQ(gnb_params_direct.n_prb, gnb_params_lazy.n_prb);

    EXPECT_EQ(ue_grp_params_direct_vec.size(), ue_grp_params_lazy_vec.size());
    EXPECT_EQ(tb_params_direct_vec.size(), tb_params_lazy_vec.size());

    // Compare first UE group parameters (if available)
    if (!ue_grp_params_direct_vec.empty() && !ue_grp_params_lazy_vec.empty()) {
        const auto &ue_grp_params_direct = ue_grp_params_direct_vec[0];
        const auto &ue_grp_params_lazy = ue_grp_params_lazy_vec[0];

        EXPECT_EQ(ue_grp_params_direct.n_ues, ue_grp_params_lazy.n_ues);
        EXPECT_EQ(ue_grp_params_direct.n_prb, ue_grp_params_lazy.n_prb);
        EXPECT_EQ(ue_grp_params_direct.n_uplink_streams, ue_grp_params_lazy.n_uplink_streams);
    }

    // Compare first TB parameters (if available)
    if (!tb_params_direct_vec.empty() && !tb_params_lazy_vec.empty()) {
        const auto &tb_params_direct = tb_params_direct_vec[0];
        const auto &tb_params_lazy = tb_params_lazy_vec[0];

        EXPECT_EQ(tb_params_direct.num_layers, tb_params_lazy.num_layers);
        EXPECT_EQ(tb_params_direct.mcs_index, tb_params_lazy.mcs_index);
        EXPECT_EQ(tb_params_direct.rv, tb_params_lazy.rv);
    }
}

// Instantiate the parameterized test with all available H5 test files
INSTANTIATE_TEST_SUITE_P(
        MultipleH5Files,
        CuphyPuschTestVectorH5Test,
        ::testing::ValuesIn(TEST_HDF5_FILES),
        [](const ::testing::TestParamInfo<std::string_view> &test_info) {
            std::string name = std::filesystem::path(test_info.param).stem().string();
            // Replace non-alphanumeric characters with underscores for valid test names
            std::replace_if(
                    name.begin(), name.end(), [](const char c) { return !std::isalnum(c); }, '_');
            return name;
        });

} // namespace
