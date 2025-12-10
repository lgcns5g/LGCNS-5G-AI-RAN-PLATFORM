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

/**
 * @file cuphy_pusch_tv_sample_test.cpp
 * @brief Sample tests for aerial_tv library documentation
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "aerial_tv/aerial_tv_utils.hpp"
#include "aerial_tv/cuphy_pusch_tv.hpp"
#include "ldpc/outer_rx_params.hpp"
#include "log/rt_log_macros.hpp"
#include "ran_common.hpp"

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

namespace {

// NOLINTBEGIN(cert-err58-cpp)
// Get the test vector path from compile definition
const std::filesystem::path TEST_VECTOR_DIR{TEST_VECTOR_PATH};
const std::string TEST_FILE = (TEST_VECTOR_DIR / "TVnr_7204_PUSCH_gNB_CUPHY_s0p0.h5").string();
// NOLINTEND(cert-err58-cpp)

TEST(AerialTvSampleTests, LoadTestVector) {
    // example-begin load-test-vector-1
    // Load a PUSCH test vector from an HDF5 file
    const ran::aerial_tv::CuphyPuschTestVector test_vector(TEST_FILE);
    // example-end load-test-vector-1

    // Verify test vector loaded successfully by accessing parameters
    EXPECT_NO_THROW({
        const auto &params = test_vector.get_gnb_params();
        EXPECT_GT(params.n_rx, 0U);
    });
}

TEST(AerialTvSampleTests, ReadGnbParameters) {
    // example-begin read-gnb-params-1
    const ran::aerial_tv::CuphyPuschTestVector test_vector(TEST_FILE);

    // Access gNB parameters from the test vector
    const auto &gnb_params = test_vector.get_gnb_params();

    // Use gNB parameters for configuration
    const std::uint32_t num_rx_antennas = gnb_params.n_rx;
    const std::uint32_t num_prbs = gnb_params.n_prb;
    const std::uint32_t subcarrier_spacing = gnb_params.mu;
    // example-end read-gnb-params-1

    EXPECT_GT(num_rx_antennas, 0U);
    EXPECT_GT(num_prbs, 0U);
    EXPECT_LE(subcarrier_spacing, 5U);
}

TEST(AerialTvSampleTests, ReadTransportBlockParameters) {
    // example-begin read-tb-params-1
    const ran::aerial_tv::CuphyPuschTestVector test_vector(TEST_FILE);

    // Access transport block parameters
    const auto &tb_params_vec = test_vector.get_tb_params();

    // Process each transport block
    for (const auto &tb_params : tb_params_vec) {
        // Use TB parameters for processing
        const std::uint32_t num_layers = tb_params.num_layers;
        const std::uint32_t mcs_index = tb_params.mcs_index;
        const std::uint32_t tb_size_bytes = tb_params.n_tb_byte;

        RT_LOG_DEBUG(
                "TB params: layers=%u, MCS=%u, size=%u bytes",
                num_layers,
                mcs_index,
                tb_size_bytes);
    }
    // example-end read-tb-params-1

    // Validate parameters after example block
    EXPECT_GE(tb_params_vec.size(), 1U);
    for (const auto &tb_params : tb_params_vec) {
        const std::uint32_t num_layers = tb_params.num_layers;
        const std::uint32_t mcs_index = tb_params.mcs_index;
        const std::uint32_t tb_size_bytes = tb_params.n_tb_byte;

        EXPECT_GT(num_layers, 0U);
        EXPECT_LE(mcs_index, 31U);
        EXPECT_GT(tb_size_bytes, 0U);
    }
}

TEST(AerialTvSampleTests, ReadUeGroupParameters) {
    // example-begin read-ue-grp-params-1
    const ran::aerial_tv::CuphyPuschTestVector test_vector(TEST_FILE);

    // Access UE group parameters
    const auto &ue_grp_params_vec = test_vector.get_ue_grp_params();

    // Process each UE group
    for (const auto &ue_grp_params : ue_grp_params_vec) {
        const std::uint16_t start_prb = ue_grp_params.start_prb;
        const std::uint16_t num_prbs = ue_grp_params.n_prb;
        const std::uint8_t start_symbol = ue_grp_params.start_symbol_index;
        const std::uint8_t num_symbols = ue_grp_params.nr_of_symbols;

        RT_LOG_DEBUG(
                "UE group: PRB range [%u-%u], symbol range [%u-%u]",
                start_prb,
                start_prb + num_prbs - 1,
                start_symbol,
                start_symbol + num_symbols - 1);
    }
    // example-end read-ue-grp-params-1

    // Validate UE group parameters after example block
    EXPECT_GE(ue_grp_params_vec.size(), 1U);
    for (const auto &ue_grp_params : ue_grp_params_vec) {
        const std::uint16_t start_prb = ue_grp_params.start_prb;
        const std::uint16_t num_prbs = ue_grp_params.n_prb;
        const std::uint8_t start_symbol = ue_grp_params.start_symbol_index;
        const std::uint8_t num_symbols = ue_grp_params.nr_of_symbols;

        EXPECT_LT(start_prb, 273U);
        EXPECT_GT(num_prbs, 0U);
        EXPECT_LT(start_symbol, 14U);
        EXPECT_GT(num_symbols, 0U);
    }
}

TEST(AerialTvSampleTests, ReadScalarDataset) {
    // example-begin read-scalar-1
    const ran::aerial_tv::CuphyPuschTestVector test_vector(TEST_FILE);

    // Read scalar values from HDF5 datasets
    const auto noise_var_db = test_vector.read_scalar<float>("reference_noiseVardB");
    const auto cfo_angle = test_vector.read_scalar<float>("reference_cfoAngle");
    // example-end read-scalar-1

    EXPECT_TRUE(std::isfinite(noise_var_db));
    EXPECT_TRUE(std::isfinite(cfo_angle));
}

TEST(AerialTvSampleTests, ReadArrayDataset) {
    // example-begin read-array-1
    const ran::aerial_tv::CuphyPuschTestVector test_vector(TEST_FILE);

    // Read array data from HDF5 dataset
    const auto array_data = test_vector.read_array<float>("WFreq");

    // Access data and dimensions
    const std::vector<float> &data = array_data.data;
    const std::vector<std::size_t> &dims = array_data.dimensions;

    // Process array data (dimensions: [dim0, dim1, dim2, ...])
    const std::size_t total_elements = data.size();
    // example-end read-array-1

    EXPECT_GT(total_elements, 0U);
    EXPECT_GT(dims.size(), 0U);
}

TEST(AerialTvSampleTests, ConvertToPhyParams) {
    // example-begin convert-to-phy-params-1
    const ran::aerial_tv::CuphyPuschTestVector test_vector(TEST_FILE);

    // Convert test vector parameters to PhyParams structure
    const ran::common::PhyParams phy_params = ran::aerial_tv::to_phy_params(test_vector);

    // Use PhyParams for PHY layer configuration
    const std::uint32_t num_prb = phy_params.num_prb;
    const std::uint16_t num_rx_ant = phy_params.num_rx_ant;
    const std::uint16_t bandwidth = phy_params.bandwidth;
    // example-end convert-to-phy-params-1

    EXPECT_GT(num_prb, 0U);
    EXPECT_GT(num_rx_ant, 0U);
    EXPECT_GT(bandwidth, 0U);
}

TEST(AerialTvSampleTests, ConvertToPuschOuterRxParams) {
    // example-begin convert-to-outer-rx-params-1
    const ran::aerial_tv::CuphyPuschTestVector test_vector(TEST_FILE);

    // Convert test vector to PUSCH outer receiver parameters
    const ran::ldpc::PuschOuterRxParams outer_rx_params =
            ran::aerial_tv::to_pusch_outer_rx_params(test_vector);

    // Use outer receiver params for PUSCH receiver configuration
    // example-end convert-to-outer-rx-params-1

    EXPECT_NO_THROW({ (void)outer_rx_params; });
}

TEST(AerialTvSampleTests, StaticReadMethods) {
    // example-begin static-read-methods-1
    // Read parameters directly without creating test vector object
    const auto gnb_params =
            ran::aerial_tv::CuphyPuschTestVector::read_gnb_params_from_file(TEST_FILE);

    const auto ue_grp_params =
            ran::aerial_tv::CuphyPuschTestVector::read_ue_grp_params_from_file(TEST_FILE);

    const auto tb_params =
            ran::aerial_tv::CuphyPuschTestVector::read_tb_params_from_file(TEST_FILE);
    // example-end static-read-methods-1

    EXPECT_GT(gnb_params.n_rx, 0U);
    EXPECT_GE(ue_grp_params.size(), 1U);
    EXPECT_GE(tb_params.size(), 1U);
}

} // namespace

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
