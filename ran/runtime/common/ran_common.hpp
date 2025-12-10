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

#ifndef RAN_COMMON_HPP
#define RAN_COMMON_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ran::common {

// Configuration constants

// Maximum supported values for memory allocation etc.
inline constexpr int NUM_CELLS_SUPPORTED = 1;         //!< Maximum number of cells supported
inline constexpr std::size_t MAX_PUSCH_PIPELINES = 1; //!< Maximum number of PUSCH pipelines
inline constexpr std::uint32_t MAX_UES_PER_CELL = 1;  //!< Maximum number of UEs per cell
inline constexpr std::uint32_t MAX_UES_PER_SLOT =
        MAX_UES_PER_CELL * NUM_CELLS_SUPPORTED;           //!< Maximum number of UEs per slot
inline constexpr std::size_t MAX_NUM_UES_PER_UE_GRP = 12; //!< Maximum number of UEs per UE group
inline constexpr std::size_t MAX_NUM_CBS_PER_TB = 152;    //!< Maximum number of code blocks per TB
inline constexpr std::size_t MAX_NUM_TBS = 1;             //!< Maximum number of transport blocks
inline constexpr std::size_t MAX_NUM_UE_GRPS = 1;         //!< Maximum number of UE groups
inline constexpr std::size_t MAX_NUM_LAYERS_PER_UE_GRP = 8;   //!< Maximum layers per UE group
inline constexpr std::uint32_t MAX_UL_LAYERS = 1;             //!< Maximum uplink layers
inline constexpr std::uint32_t MAX_ANTENNAS = 4;              //!< Maximum number of antennas
inline constexpr std::uint32_t BANDWIDTH_SUPPORTED_MHZ = 100; //!< Supported bandwidth in MHz
inline constexpr std::uint32_t NUM_PRBS_SUPPORTED = 273;      //!< Number of PRBs supported
inline constexpr std::uint32_t PUSCH_MAX_TB_SIZE_BYTES =
        311386; //!< Maximum bytes per transport block

// PHY parameter constants
inline constexpr int NUM_SLOTS_PER_SF = 20;     //!< Number of slots per subframe
inline constexpr int NUM_SFNS_PER_FRAME = 1024; //!< Number of system frame numbers per frame
inline constexpr std::uint16_t MAX_SFN = 1024;  //!< Maximum system frame number
inline constexpr std::uint16_t MAX_SLOT = 20;   //!< Maximum slot number
inline constexpr std::uint16_t INVALID_CELL_ID = 65535; //!< Invalid cell ID sentinel value
inline constexpr std::uint32_t OFDM_SYMBOLS_NORMAL_CYCLIC_PREFIX =
        14; //!< Number of OFDM symbols with normal cyclic prefix
inline constexpr std::uint32_t NUM_SUBCARRIERS_PER_PRB = 12; //!< Number of subcarriers per PRB
inline constexpr std::uint32_t MAX_DMRS_OFDM_SYMBOLS = 1;    //!< Maximum DMRS OFDM symbols
inline constexpr std::uint32_t REAL_IMAG_INTERLEAVED = 2;    //!< Real & imag interleaved
inline constexpr std::uint32_t MAX_QAM_BITS = 8;             //!< Maximum QAM bits supported
inline constexpr std::uint32_t CYCLIC_PREFIX_NORMAL = 0;     //!< Normal cyclic prefix
inline constexpr std::uint32_t SUBCARRIER_SPACING_MU_1 = 1;  //!< mu-1 subcarrier spacing

/**
 * Physical layer parameters
 */
struct PhyParams final {
    std::uint16_t num_rx_ant{};   //!< Number of receive antennas
    std::uint8_t cyclic_prefix{}; //!< Cyclic prefix
    std::uint16_t bandwidth{};    //!< Bandwidth in MHz
    std::uint32_t num_prb{};      //!< Number of PRBs is a function of bandwidth
};

} // namespace ran::common

#endif // RAN_COMMON_HPP
