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

#ifndef RAN_PUSCH_DEFINES_HPP
#define RAN_PUSCH_DEFINES_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include "ldpc/outer_rx_params.hpp"
#include "ran_common.hpp"

namespace ran::pusch {

inline constexpr uint32_t NUM_PUSCH_MODULES = 4;       //!< Number of PUSCH modules
inline constexpr std::size_t NUM_EXTERNAL_INPUTS = 1;  //!< Number of external input ports
inline constexpr std::size_t NUM_EXTERNAL_OUTPUTS = 4; //!< Number of external output ports

/**
 * InnerRx parameters structure
 */
struct PuschInnerRxRxParams {};

/**
 * PUSCH UE parameters structure
 */
struct PuschUeParams {
    uint16_t cell_id{};          //!< Cell ID (0-65535)
    uint16_t sfn{};              //!< System Frame Number (0-1023)
    uint16_t rnti{};             //!< Radio Network Temporary Identifier (0-65535)
    uint32_t handle{};           //!< Handle for UL indication
    uint16_t target_code_rate{}; //!< Target code rate
    uint8_t qam_mod_order{}; //!< QAM modulation order (2,4,6,8 for TP disabled; 1,2,4,6,8 for TP
                             //!< enabled)
    uint8_t mcs_index{};     //!< MCS index (0-31)
    uint8_t mcs_table{};     //!< MCS table index (0-4)
    uint8_t transform_precoding{};       //!< Transform precoding (0-1)
    uint16_t data_scrambling_id{};       //!< Data scrambling ID (0-65535)
    uint8_t num_layers{};                //!< Number of layers (1-4)
    uint16_t dmrs_sym_pos_bmsk{};        //!< DMRS symbol position bitmask
    uint8_t num_dmrs_cdm_grps_no_data{}; //!< Number of DMRS CDM groups without data
    uint16_t start_prb{};                //!< Start physical resource block for RA type 1
    uint16_t num_prb{};                  //!< Number of physical resource blocks for RA type 1
    uint8_t num_symbols{};               //!< Number of symbols
    uint8_t start_symbol_index{};        //!< Start symbol index
    uint8_t rv_index{};                  //!< Redundancy version index (0-3)
    uint8_t harq_process_id{};           //!< HARQ process ID (0-15)
    uint8_t ndi{};                       //!< New data indicator (0-1)
    uint32_t tb_size{};                  //!< Transport block size in bytes
};

/**
 * PUSCH dynamic parameters data structure
 *
 * This structure contains the dynamic parameters for the PUSCH pipeline.
 * It is used to pass the dynamic parameters to the PUSCH pipeline.
 *
 * @param[in] inner_rx_params InnerRx parameters
 * @param[in] outer_rx_params OuterRx parameters
 */
struct PuschDynamicParams {
    PuschInnerRxRxParams inner_rx_params;          //!< InnerRx parameters
    ran::ldpc::PuschOuterRxParams outer_rx_params; //!< OuterRx parameters
};

/**
 * PUSCH Input Data
 *
 * Contains all input parameters and data for PUSCH processing.
 */
struct PuschInput {
    std::vector<PuschUeParams> ue_params; //!< UE parameters
    uint32_t ue_params_index{};           //!< UE parameters index
    std::vector<std::vector<int8_t>>
            ue_group_idx_map; //!< map between ue params index and ue group index. ue_group_idx_map
                              //!< is sized by number of ue groups each slot.
                              //!< ue_group_idx_map[ue_group_idx] contains the indexes into
                              //!< ue_params for the UE which belong to this group.
    uint32_t ue_group_idx_index{}; //!< UE group index index
    std::vector<float> xtf;        //!< XTF data (real/imag interleaved)

    /**
     * Check if indices are within bounds
     *
     * @return true if indices are within valid ranges
     */
    [[nodiscard]] bool check_bounds() const {
        return (ue_params_index < ran::common::MAX_UES_PER_SLOT) &&
               (ue_group_idx_index < ran::common::MAX_NUM_UE_GRPS);
    }
};

/**
 * PUSCH Output Data
 *
 * Contains all output results from PUSCH processing.
 */
struct PuschOutput {
    std::vector<std::uint32_t> tb_crcs;      //!< Transport block CRCs
    std::vector<std::uint8_t *> tb_payloads; //!< Transport block payloads
    void *d_tb_payloads{}; //!< pointer to device memory containing Transport block payloads
    std::vector<float> post_eq_noise_var_db; //!< Post-EQ noise variance db
    std::vector<float> post_eq_sinr_db;      //!< Post-EQ SINR db
};
} // namespace ran::pusch

#endif // RAN_PUSCH_DEFINES_HPP
