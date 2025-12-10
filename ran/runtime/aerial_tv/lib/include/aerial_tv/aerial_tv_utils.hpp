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

#ifndef RAN_AERIAL_TV_UTILS_HPP
#define RAN_AERIAL_TV_UTILS_HPP

#include <array>
#include <span>
#include <string_view>

#include <cuda_runtime_api.h>

#include "aerial_tv/cuphy_pusch_tv.hpp"
#include "ldpc/outer_rx_params.hpp"
#include "ran_common.hpp"

// Forward declarations
namespace framework::pipeline {
struct DeviceTensor;
} // namespace framework::pipeline

namespace ran::aerial_tv {

/// Common HDF5 test vector files for testing
inline constexpr std::array<std::string_view, 2> TEST_HDF5_FILES{
        "TVnr_7201_PUSCH_gNB_CUPHY_s0p0.h5", "TVnr_7204_PUSCH_gNB_CUPHY_s0p0.h5"};

/**
 * Convert CuphyPuschTestVector to PhyParams structure
 *
 * Extracts physical layer configuration parameters from the gNB parameters
 * section of the test vector and populates a PhyParams object.
 *
 * @param[in] test_vector PUSCH test vector containing gNB parameters
 * @return Physical layer parameters extracted from the test vector
 */
[[nodiscard]] ran::common::PhyParams to_phy_params(const CuphyPuschTestVector &test_vector);

/**
 * Read from a CuphyPuschTestVector into a PuschOuterRxParams object.
 *
 * @param[in] test_vector PUSCH test vector
 * @return PUSCH receiver outer_rx parameters object
 */
[[nodiscard]] ran::ldpc::PuschOuterRxParams
to_pusch_outer_rx_params(const CuphyPuschTestVector &test_vector);

/**
 * Read reference post-EQ noise variance from test vector and compare against obtained values
 *
 * @param[in] obtained Obtained post-EQ noise variance values
 * @param[in] test_vector Test vector containing reference data
 * @param[in] tolerance Absolute tolerance for comparison (default: 0.0 for exact match)
 */
void check_post_eq_noise_var(
        const std::vector<float> &obtained,
        const CuphyPuschTestVector &test_vector,
        const float tolerance = 0.0F);

/**
 * Read reference post-EQ SINR from test vector and compare against obtained values
 *
 * @param[in] obtained Obtained post-EQ SINR values
 * @param[in] test_vector Test vector containing reference data
 * @param[in] tolerance Absolute tolerance for comparison (default: 0.0 for exact match)
 */
void check_post_eq_sinr(
        const std::vector<float> &obtained,
        const CuphyPuschTestVector &test_vector,
        const float tolerance = 0.0F);

/**
 * Validate transport block payloads against reference data from test vector
 *
 * This function reads the expected TB payloads from the test vector, copies the
 * obtained TB payloads from device to host using stream-aware async copies,
 * and compares them byte-by-byte. It handles multiple transport blocks and
 * accounts for CRC bytes in the reference data.
 *
 * @param[in] tb_payload_tensors Device tensors containing the TB payloads to validate
 * @param[in] test_vector Test vector containing reference TB payload data and outer_rx params
 * @param[in] stream CUDA stream to use for async memory copies (ensures proper synchronization)
 */
void check_tb_payloads(
        const std::span<const framework::pipeline::DeviceTensor> tb_payload_tensors,
        const CuphyPuschTestVector &test_vector,
        cudaStream_t stream);

} // namespace ran::aerial_tv

#endif // RAN_AERIAL_TV_UTILS_HPP
