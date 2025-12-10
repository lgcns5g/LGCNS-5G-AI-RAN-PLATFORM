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

#ifndef RAN_LDPC_DERATE_MATCH_PARAMS_HPP
#define RAN_LDPC_DERATE_MATCH_PARAMS_HPP

#include <cstdint>
#include <vector>

#include <wise_enum.h>

namespace ran::ldpc {

/**
 * Modulation order enumeration for type-safe modulation scheme specification
 */
enum class ModulationOrder : std::uint32_t {
    Qpsk = 2,  //!< QPSK modulation (2 bits per symbol)
    Qam16 = 4, //!< 16-QAM modulation (4 bits per symbol)
    Qam64 = 6, //!< 64-QAM modulation (6 bits per symbol)
    Qam256 = 8 //!< 256-QAM modulation (8 bits per symbol)
};

/**
 * New Data Indicator enumeration for type-safe transmission type specification
 */
enum class NewDataIndicator : std::uint32_t {
    Retransmission = 0, //!< Retransmission of previous data
    NewTransmission = 1 //!< New data transmission
};

} // namespace ran::ldpc

// WISE_ENUM_ADAPT must be called at global namespace scope for ADL to work
WISE_ENUM_ADAPT(ran::ldpc::ModulationOrder, Qpsk, Qam16, Qam64, Qam256)
WISE_ENUM_ADAPT(ran::ldpc::NewDataIndicator, Retransmission, NewTransmission)

namespace ran::ldpc {

/**
 * Configuration parameters for rate matching
 *
 * Contains all additional parameters for rate matching.
 * @note LDPC parameters are needed too by the derate matcher.
 */
struct DerateMatchParams {
    ModulationOrder mod_order{ModulationOrder::Qpsk}; //!< Modulation order
    std::uint8_t n_dmrs_cdm_grps_no_data{};           //!< Number of DMRS CDM groups without data
    NewDataIndicator ndi{NewDataIndicator::NewTransmission}; //!< New data indicator
    std::uint32_t num_layers{1};                             //!< Number of transmission layers
    std::uint32_t user_group_idx{};          //!< User group index for multi-user scenarios
    std::uint32_t num_ue_grp_layers{1};      //!< Number of layers for UE group
    std::vector<std::uint32_t> layer_map{0}; //!< Layer mapping vector
    std::uint32_t scrambling_init{};         //!< Scrambling initialization value
};

} // namespace ran::ldpc

#endif // RAN_LDPC_DERATE_MATCH_PARAMS_HPP
