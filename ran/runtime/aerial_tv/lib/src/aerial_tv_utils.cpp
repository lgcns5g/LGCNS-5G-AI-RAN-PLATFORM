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
#include <cstddef>
#include <cstdint>
#include <format>
#include <iterator>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <driver_types.h>
#include <range/v3/iterator/basic_iterator.hpp>
#include <range/v3/iterator/unreachable_sentinel.hpp>
#include <range/v3/view/enumerate.hpp>
#include <range/v3/view/facade.hpp>
#include <range/v3/view/view.hpp>
#include <range/v3/view/zip.hpp>

#include <gmock/gmock.h>
#include <gsl-lite/gsl-lite.hpp>
#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include "aerial_tv/aerial_tv_utils.hpp"
#include "aerial_tv/cuphy_pusch_tv.hpp"
#include "ldpc/derate_match_params.hpp"
#include "ldpc/ldpc_params.hpp"
#include "ldpc/outer_rx_params.hpp"
#include "pipeline/types.hpp"
#include "ran_common.hpp"
#include "tensor/tensor_info.hpp"

namespace ran::aerial_tv {

ran::common::PhyParams to_phy_params(const CuphyPuschTestVector &test_vector) {
    const auto &gnb_params = test_vector.get_gnb_params();

    ran::common::PhyParams phy_params{};

    // Extract number of receive antennas
    phy_params.num_rx_ant = gsl_lite::narrow_cast<std::uint16_t>(gnb_params.n_rx);

    // Set cyclic prefix to normal (0) - this is the standard value
    phy_params.cyclic_prefix = ran::common::CYCLIC_PREFIX_NORMAL;

    // Extract number of PRBs
    phy_params.num_prb = gnb_params.n_prb;

    // Calculate bandwidth from number of PRBs and subcarrier spacing
    // For μ=1 (30 kHz subcarrier spacing), bandwidth in MHz = (n_prb * 12 subcarriers/PRB * 30 kHz)
    // / 1000
    static constexpr std::uint32_t KHZ_TO_MHZ = 1000;
    const std::uint32_t subcarrier_spacing_khz = 15U << gnb_params.mu; // 15 * 2^μ kHz
    phy_params.bandwidth = gsl_lite::narrow_cast<std::uint16_t>(
            (gnb_params.n_prb * ran::common::NUM_SUBCARRIERS_PER_PRB * subcarrier_spacing_khz) /
            KHZ_TO_MHZ);

    return phy_params;
}

ran::ldpc::PuschOuterRxParams to_pusch_outer_rx_params(const CuphyPuschTestVector &test_vector) {

    const auto &tb_params_vec = test_vector.get_tb_params();
    const auto &ue_grp_params_vec = test_vector.get_ue_grp_params();
    const auto num_tbs = tb_params_vec.size();
    const auto num_ue_grps = ue_grp_params_vec.size();

    std::vector<ran::ldpc::SingleTbPuschOuterRxParams> pusch_outer_rx_params;
    pusch_outer_rx_params.reserve(num_tbs);
    std::vector<std::uint16_t> sch_user_idxs(num_tbs);

    // Count the total number of layers for each UE group
    std::vector<std::uint32_t> num_ue_grp_layers(num_ue_grps, 0);
    for (const auto &tb_params : tb_params_vec) {
        const auto user_group_idx = tb_params.user_group_index;
        num_ue_grp_layers[user_group_idx] += tb_params.num_layers;
    }

    // Process each transport block
    std::vector<std::uint32_t> layer_count(num_ue_grps, 0);
    for (const auto &[i, tb_params] : tb_params_vec | ranges::views::enumerate) {

        const auto tb_size = tb_params.n_tb_byte * 8U; // Convert bytes to bits
        // Convert target code rate to 0.0 to 1.0. In the test vectors, the target code rate is
        // from 0 to 10240.
        const auto code_rate = static_cast<float>(tb_params.target_code_rate) / 10240.0F;

        // Rate matching length is obtained from derate matching input.
        // Input is zero padded to maximum modulation order (8), compute the actual.
        const auto user_group_idx = tb_params.user_group_index;
        const auto mod_order = tb_params.qam_mod_order;
        const auto de_rm_input_array =
                test_vector.read_array<float>(std::format("reference_eqOutLLRs{}", user_group_idx));
        const auto &de_rm_input = de_rm_input_array.data;
        constexpr std::uint8_t QAM_256_ORDER = 8;
        const auto rate_matching_len = gsl_lite::narrow_cast<std::uint32_t>(de_rm_input.size()) *
                                       mod_order / QAM_256_ORDER;

        // Create LDPC parameters
        const ran::ldpc::LdpcParams ldpc_params(
                tb_size, code_rate, rate_matching_len, tb_params.rv);

        // Scrambling initialization
        const auto rnti = tb_params.n_rnti;
        const auto data_scram_id = tb_params.data_scram_id;
        const auto scrambling_init = ran::ldpc::get_scrambling_init(
                gsl_lite::narrow_cast<std::uint32_t>(rnti), data_scram_id);

        // Map UE layers to UE group layers
        std::vector<std::uint32_t> layer_map(tb_params.num_layers);
        for (std::size_t layer_idx = 0; layer_idx < tb_params.num_layers; ++layer_idx) {
            layer_map[layer_idx] = layer_count[user_group_idx];
            ++layer_count[user_group_idx];
        }

        // Create DerateMatchParams object
        const ran::ldpc::DerateMatchParams de_rm_params{
                .mod_order = static_cast<ran::ldpc::ModulationOrder>(mod_order),
                .n_dmrs_cdm_grps_no_data = tb_params.num_dmrs_cdm_grps_no_data,
                .ndi = static_cast<ran::ldpc::NewDataIndicator>(tb_params.ndi),
                .num_layers = tb_params.num_layers,
                .user_group_idx = user_group_idx,
                .num_ue_grp_layers = num_ue_grp_layers[user_group_idx],
                .layer_map = layer_map,
                .scrambling_init = scrambling_init};

        // Create single TB parameters
        const ran::ldpc::SingleTbPuschOuterRxParams single_tb_params{ldpc_params, de_rm_params};
        pusch_outer_rx_params.emplace_back(single_tb_params);
        sch_user_idxs[i] = gsl_lite::narrow_cast<std::uint16_t>(i);
    }

    return ran::ldpc::PuschOuterRxParams{pusch_outer_rx_params, sch_user_idxs};
}

void check_post_eq_noise_var(
        const std::vector<float> &obtained,
        const ran::aerial_tv::CuphyPuschTestVector &test_vector,
        const float tolerance) {
    const auto reference_array = test_vector.read_array<float>("reference_postEqNoiseVardBvec");
    const auto &reference = reference_array.data;

    ASSERT_EQ(obtained.size(), reference.size())
            << "Post-EQ noise variance size mismatch: obtained " << obtained.size()
            << " elements, expected " << reference.size();

    if (tolerance == 0.0F) {
        EXPECT_THAT(obtained, ::testing::ContainerEq(reference))
                << "Post-EQ noise variance mismatch";
    } else {
        EXPECT_THAT(obtained, ::testing::Pointwise(::testing::FloatNear(tolerance), reference))
                << "Post-EQ noise variance mismatch";
    }
}

void check_post_eq_sinr(
        const std::vector<float> &obtained,
        const ran::aerial_tv::CuphyPuschTestVector &test_vector,
        const float tolerance) {
    const auto reference_array = test_vector.read_array<float>("reference_postEqSinrdBVec");
    const auto &reference = reference_array.data;

    ASSERT_EQ(obtained.size(), reference.size())
            << "Post-EQ SINR size mismatch: obtained " << obtained.size() << " elements, expected "
            << reference.size();

    if (tolerance == 0.0F) {
        EXPECT_THAT(obtained, ::testing::ContainerEq(reference)) << "Post-EQ SINR mismatch";
    } else {
        EXPECT_THAT(obtained, ::testing::Pointwise(::testing::FloatNear(tolerance), reference))
                << "Post-EQ SINR mismatch";
    }
}

void check_tb_payloads(
        const std::span<const framework::pipeline::DeviceTensor> tb_payload_tensors,
        const ran::aerial_tv::CuphyPuschTestVector &test_vector,
        cudaStream_t stream) {

    constexpr std::uint32_t BITS_PER_BYTE = 8;

    // Read outer_rx params from test vector
    const auto outer_rx_params = to_pusch_outer_rx_params(test_vector);

    // Read expected payload from test vector (contains payloads for all TBs)
    const auto expected_tb_array = test_vector.read_array<std::uint8_t>("tb_payload");
    const auto &expected_tb_output = expected_tb_array.data;

    const auto num_tbs = outer_rx_params.num_tbs();
    ASSERT_EQ(tb_payload_tensors.size(), num_tbs)
            << "TB payload tensor count mismatch: obtained " << tb_payload_tensors.size()
            << " tensors, expected " << num_tbs;

    std::size_t tb_payload_offset = 0;

    // Allocate host buffers for all TBs upfront
    std::vector<std::vector<std::uint8_t>> tb_outputs;
    tb_outputs.reserve(num_tbs);

    // Launch async copies for all transport blocks
    for (std::size_t tb_idx = 0; tb_idx < num_tbs; ++tb_idx) {
        const auto &tb_payload_tensor = tb_payload_tensors[tb_idx];
        const auto &tb_payloads_dimension = tb_payload_tensor.tensor_info.get_dimensions()[0];

        // Verify tensor dimension matches expected TB size
        const auto ldpc_params = outer_rx_params[tb_idx].ldpc_params();
        const std::size_t tb_size_bytes = ldpc_params.transport_block_size() / BITS_PER_BYTE;
        ASSERT_EQ(tb_payloads_dimension, tb_size_bytes)
                << "TB " << tb_idx << " size mismatch: tensor has " << tb_payloads_dimension
                << " bytes, expected " << tb_size_bytes;

        // Allocate host buffer for this TB
        tb_outputs.emplace_back(tb_size_bytes);

        // Copy TB payload from device to host asynchronously on the same stream
        ASSERT_EQ(
                cudaMemcpyAsync(
                        tb_outputs[tb_idx].data(),
                        tb_payload_tensor.device_ptr,
                        tb_size_bytes * sizeof(std::uint8_t),
                        cudaMemcpyDeviceToHost,
                        stream),
                cudaSuccess)
                << "Failed to copy TB " << tb_idx << " payload from device to host";
    }

    // Synchronize stream to ensure all async copies complete
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess)
            << "Failed to synchronize stream after TB payload copies";

    // Validate each transport block
    tb_payload_offset = 0;
    for (std::size_t tb_idx = 0; tb_idx < num_tbs; ++tb_idx) {
        const auto ldpc_params = outer_rx_params[tb_idx].ldpc_params();
        const std::size_t tb_size_bytes = ldpc_params.transport_block_size() / BITS_PER_BYTE;

        // Extract expected data for this TB from the reference payload
        const std::vector<std::uint8_t> expected_tb(
                std::next(
                        expected_tb_output.begin(), static_cast<std::ptrdiff_t>(tb_payload_offset)),
                std::next(
                        expected_tb_output.begin(),
                        static_cast<std::ptrdiff_t>(tb_payload_offset + tb_size_bytes)));

        // Compare obtained vs expected
        EXPECT_EQ(tb_outputs.at(tb_idx), expected_tb) << "TB " << tb_idx << " payload mismatch";

        // Update offset to next TB (including CRC bytes in reference data)
        const std::size_t tb_size_bytes_with_crc =
                ldpc_params.get_tb_size_with_crc(ldpc_params.transport_block_size()) /
                BITS_PER_BYTE;
        tb_payload_offset += tb_size_bytes_with_crc;
    }
}

} // namespace ran::aerial_tv
