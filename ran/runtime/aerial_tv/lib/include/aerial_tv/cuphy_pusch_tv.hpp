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

#ifndef RAN_AERIAL_TV_CUPHY_PUSCH_TV_HPP
#define RAN_AERIAL_TV_CUPHY_PUSCH_TV_HPP

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <cuda_fp16.h>

namespace ran::aerial_tv {

/**
 * Concept for types that can be stored in HDF5 arrays
 *
 * HDF5-compatible types include arithmetic types (integers or floating-point)
 * and CUDA half-precision types.
 */
template <typename T>
concept Hdf5Compatible = std::is_arithmetic_v<T> || std::is_same_v<T, __half>;

/**
 * Concept for types supported in complex arrays
 *
 * Includes standard floating-point types and CUDA half-precision types
 */
template <typename T>
concept ComplexArrayType = std::floating_point<T> || std::is_same_v<T, __half>;

/**
 * Array data and dimensions read from HDF5 test vector files
 *
 * @tparam T Element type of the array (must satisfy Hdf5Compatible)
 */
template <Hdf5Compatible T> struct Hdf5Array {
    std::vector<T> data;                 //!< Flattened array data
    std::vector<std::size_t> dimensions; //!< Dimensions of the array
};

// Constants for array sizes in PUSCH test vector structures
inline constexpr std::size_t MAX_CSI2_MAPS_BUFFER_SIZE = 20U; //!< Maximum CSI-2 maps buffer entries
inline constexpr std::size_t MAX_CSI2_REPORTS_PER_TB =
        16U; //!< Maximum CSI-2 reports per transport block
inline constexpr std::size_t MAX_TOTAL_CSI1_PARAMETERS_PER_TB =
        64U; //!< Maximum total CSI-1 parameters across all CSI-2 reports (16 reports × 4 params
             //!< each)

/**
 * gNB parameters for CUPHY PUSCH test vectors
 *
 * This structure matches the HDF5 compound data type fields in gnb_pars dataset
 */
struct CuphyPuschTvGnbParams {
    double n_user_groups{};      //!< Number of user groups
    std::uint32_t mu{};          //!< Subcarrier spacing parameter (μ)
    std::uint32_t n_rx{};        //!< Number of receive antennas
    std::uint32_t n_prb{};       //!< Number of physical resource blocks
    std::uint32_t cell_id{};     //!< Cell ID
    std::uint32_t slot_number{}; //!< Slot number
    std::uint32_t num_tb{};      //!< Number of transport blocks

    // Boolean flags (stored as uint8 in HDF5)
    std::uint8_t enable_early_harq{};                 //!< Enable early HARQ termination
    std::uint8_t enable_cfo_correction{};             //!< Enable CFO correction
    std::uint8_t enable_cfo_estimation{};             //!< Enable CFO estimation
    std::uint8_t enable_to_estimation{};              //!< Enable timing offset estimation
    std::uint8_t enable_to_correction{};              //!< Enable timing offset correction
    std::uint8_t tdi_mode{};                          //!< TDI mode
    std::uint8_t enable_dft_s_ofdm{};                 //!< Enable DFT-S-OFDM
    std::uint8_t enable_rssi_measurement{};           //!< Enable RSSI measurement
    std::uint8_t enable_sinr_measurement{};           //!< Enable SINR measurement
    std::uint8_t enable_static_dynamic_beamforming{}; //!< Enable static/dynamic
                                                      //!< beamforming

    // LDPC parameters
    std::uint32_t ldpc_early_termination{};  //!< LDPC early termination
    std::uint32_t ldpc_algo_index{};         //!< LDPC algorithm index
    std::uint32_t ldpc_flags{};              //!< LDPC flags
    std::uint32_t ldpc_use_half{};           //!< LDPC use half precision
    std::uint32_t num_bbu_layers{};          //!< Number of BBU layers
    std::uint8_t ldpc_max_num_itr{};         //!< LDPC maximum iterations
    std::uint8_t ldpc_max_num_itr_alg_idx{}; //!< LDPC max iterations algorithm index

    // Channel estimation parameters
    std::uint8_t dmrs_ch_est_alg_idx{};   //!< DMRS channel estimation algorithm index
    std::uint8_t enable_per_prg_ch_est{}; //!< Enable per-PRG channel estimation
    std::uint8_t eq_coeff_algo_idx{};     //!< Equalization coefficient algorithm index
    std::uint8_t list_length{};           //!< List length

    // CSI (Channel State Information) parameters
    std::uint8_t enable_csi_p2_fapiv3{}; //!< Enable CSI Part-2 processing using FAPI v3 interface
    std::uint16_t n_csi2_maps{};         //!< Number of active CSI-2 size maps for this cell
    std::array<std::uint8_t, 4>
            csi2_maps_sum_of_prm_sizes{}; //!< Total parameter sizes for each of 4 CSI-2 map types
    std::array<std::uint32_t, 4>
            csi2_maps_buffer_start_idxs{}; //!< Start indices into csi2_maps_buffer for each map
                                           //!< type
    std::array<std::uint16_t, MAX_CSI2_MAPS_BUFFER_SIZE>
            csi2_maps_buffer{}; //!< Buffer storing CSI-2 mapping configuration data
};

/**
 * Transport Block (TB) parameters for CUPHY PUSCH test vectors
 *
 * This structure matches the HDF5 compound data type fields in tb_pars dataset
 */
struct CuphyPuschTvTbParams {
    // Basic transport block parameters
    std::uint32_t n_rnti{};           //!< Radio Network Temporary Identifier
    std::uint32_t num_layers{};       //!< Number of layers
    std::uint32_t start_sym{};        //!< Start symbol index
    std::uint32_t num_sym{};          //!< Number of symbols
    std::uint32_t user_group_index{}; //!< User group index
    std::uint32_t data_scram_id{};    //!< Data scrambling ID

    // MCS and coding parameters
    std::uint32_t mcs_table_index{}; //!< MCS table index
    std::uint32_t mcs_index{};       //!< MCS index
    std::uint32_t rv{};              //!< Redundancy version
    std::uint32_t ndi{};             //!< New data indicator
    std::uint32_t n_tb_byte{};       //!< Transport block size in bytes
    std::uint32_t n_cb{};            //!< Number of code blocks

    // LBRM (Limited Buffer Rate Matching) parameters
    std::uint8_t i_lbrm{};      //!< LBRM indicator
    std::uint8_t max_layers{};  //!< Maximum layers
    std::uint8_t max_qm{};      //!< Maximum QAM modulation order
    std::uint16_t n_prb_lbrm{}; //!< Number of PRBs for LBRM

    // Modulation and coding parameters
    std::uint8_t qam_mod_order{};     //!< QAM modulation order
    std::uint16_t target_code_rate{}; //!< Target code rate

    // HARQ and CSI uplink control information parameters
    std::uint16_t n_bits_harq{};         //!< Number of HARQ acknowledgment bits in UCI payload
    std::uint16_t n_bits_csi1{};         //!< Number of CSI Part-1 bits (wideband channel quality)
    std::uint16_t pdu_bitmap{};          //!< PDU bitmap indicating presence of different UCI types
    std::uint8_t alpha_scaling{};        //!< Alpha scaling factor for UCI power control
    std::uint8_t beta_offset_harq_ack{}; //!< Beta offset for HARQ ACK power scaling
    std::uint8_t beta_offset_csi1{};     //!< Beta offset for CSI Part-1 power scaling
    std::uint8_t beta_offset_csi2{};     //!< Beta offset for CSI Part-2 power scaling

    // CSI parameter arrays (flattened across all CSI-2 reports for this TB)
    // Indexing scheme: CSI-2 Report N uses parameter indices [N*4, N*4+1, N*4+2, N*4+3]
    // Only the first n_part1_prms[N] entries are valid for each CSI-2 report N
    std::array<std::uint8_t, MAX_CSI2_REPORTS_PER_TB>
            n_part1_prms{}; //!< Number of CSI Part-1 parameters per CSI-2 report (1-4 each)
    std::array<std::uint8_t, MAX_TOTAL_CSI1_PARAMETERS_PER_TB>
            prm_sizes{}; //!< Flattened array of CSI Part-1 parameter sizes (bits)
    std::array<std::uint16_t, MAX_TOTAL_CSI1_PARAMETERS_PER_TB>
            prm_offsets{}; //!< Flattened array of CSI Part-1 parameter offsets (bit positions)
    std::array<std::uint16_t, MAX_CSI2_REPORTS_PER_TB>
            csi2_size_map_idx{}; //!< CSI-2 size map indices for each CSI-2 report

    // CSI Part-2 parameters (frequency-selective channel feedback)
    std::uint16_t n_csi2_reports{}; //!< Number of active CSI-2 reports for this TB
    std::uint16_t flag_csi_part2{}; //!< Bitmap indicating which CSI Part-2 fields are present
    std::uint8_t rank_bit_offset{}; //!< Bit offset where rank information starts in CSI payload
    std::uint8_t rank_bit_size{}; //!< Number of bits used to encode rank (1-4 bits for ranks 1-16)

    // DMRS parameters
    std::uint32_t dmrs_addl_position{};       //!< DMRS additional position
    std::uint32_t dmrs_max_length{};          //!< DMRS maximum length
    std::uint32_t dmrs_scram_id{};            //!< DMRS scrambling ID
    std::uint32_t n_scid{};                   //!< Scrambling ID
    std::uint32_t dmrs_port_bmsk{};           //!< DMRS port bitmask
    std::uint32_t dmrs_sym_loc_bmsk{};        //!< DMRS symbol location bitmask
    std::uint32_t rssi_sym_loc_bmsk{};        //!< RSSI symbol location bitmask
    std::uint8_t num_dmrs_cdm_grps_no_data{}; //!< Number of DMRS CDM groups without data

    // DTX and transform precoding
    float dtx_threshold{};         //!< DTX threshold
    std::uint8_t enable_tf_prcd{}; //!< Enable transform precoding

    // PUSCH and slot parameters
    std::uint8_t pusch_identity{}; //!< PUSCH identity
    std::uint8_t n_slot_frame{};   //!< Number of slots per frame
    std::uint8_t n_symb_slot{};    //!< Number of symbols per slot

    // Low PAPR parameters
    std::uint8_t group_or_sequence_hopping{}; //!< Group or sequence hopping
    std::uint8_t low_papr_group_number{};     //!< Low PAPR group number
    std::uint16_t low_papr_sequence_number{}; //!< Low PAPR sequence number
};

/**
 * UE group parameters for CUPHY PUSCH test vectors
 *
 * This structure matches the HDF5 compound data type fields in ueGrp_pars
 * dataset
 */
struct CuphyPuschTvUeGrpParams {
    std::uint16_t n_ues{};                  //!< Number of UEs in the group
    std::vector<std::uint16_t> ue_prm_idxs; //!< UE parameter indices
    std::uint16_t start_prb{};              //!< Starting physical resource block
    std::uint16_t n_prb{};                  //!< Number of physical resource blocks
    std::uint8_t start_symbol_index{};      //!< Start symbol index
    std::uint8_t nr_of_symbols{};           //!< Number of symbols
    std::uint16_t prg_size{};               //!< PRG (Physical Resource Group) size
    std::uint16_t dmrs_sym_loc_bmsk{};      //!< DMRS symbol location bitmask
    std::uint16_t rssi_sym_loc_bmsk{};      //!< RSSI symbol location bitmask
    double n_uplink_streams{};              //!< Number of uplink streams
};

/**
 * Test vector class for CUPHY (CUDA PHY) PUSCH operations in the Aerial
 * framework.
 *
 * This class provides functionality to load and manage PUSCH test vectors from
 * files for CUDA-based physical layer processing validation and testing.
 */
class CuphyPuschTestVector {
public:
    /**
     * Constructs a PUSCH test vector from the specified file.
     *
     * @param[in] filename Path to the test vector file to load
     */
    explicit CuphyPuschTestVector(std::string filename);

    /**
     * Gets the gNB parameters from the loaded PUSCH test vector.
     * Parameters are loaded lazily on first access.
     *
     * @return Reference to the gNB parameters structure
     */
    [[nodiscard]] const CuphyPuschTvGnbParams &get_gnb_params() const;

    /**
     * Gets the UE group parameters from the loaded PUSCH test vector.
     * Parameters are loaded lazily on first access.
     *
     * @return Reference to the vector of UE group parameters structures
     */
    [[nodiscard]] const std::vector<CuphyPuschTvUeGrpParams> &get_ue_grp_params() const;

    /**
     * Gets a specific UE group parameters by index from the loaded PUSCH test
     * vector.
     *
     * @param[in] ue_grp_idx Index of the UE group (0-based)
     * @return Reference to the UE group parameters structure at the specified
     * index
     * @throws std::out_of_range if ue_grp_idx is invalid
     */
    [[nodiscard]] const CuphyPuschTvUeGrpParams &get_ue_grp_params(std::size_t ue_grp_idx) const;

    /**
     * Gets the transport block parameters from the loaded PUSCH test vector.
     * Parameters are loaded lazily on first access.
     *
     * @return Reference to the vector of transport block parameters structures
     */
    [[nodiscard]] const std::vector<CuphyPuschTvTbParams> &get_tb_params() const;

    /**
     * Gets a specific transport block parameters by index from the loaded PUSCH
     * test vector.
     *
     * @param[in] tb_idx Index of the transport block (0-based)
     * @return Reference to the transport block parameters structure at the
     * specified index
     * @throws std::out_of_range if tb_idx is invalid
     */
    [[nodiscard]] const CuphyPuschTvTbParams &get_tb_params(std::size_t tb_idx) const;

    /**
     * Reads gNB parameters from an HDF5 PUSCH test vector file.
     *
     * @param[in] filename Path to the HDF5 test vector file
     * @return gNB parameters structure loaded from the file
     * @throws std::runtime_error if file cannot be opened or gnb_pars dataset is
     * missing/invalid
     */
    [[nodiscard]] static CuphyPuschTvGnbParams
    read_gnb_params_from_file(const std::string_view filename);

    /**
     * Reads UE group parameters from an HDF5 PUSCH test vector file.
     *
     * @param[in] filename Path to the HDF5 test vector file
     * @return Vector of UE group parameters structures loaded from the file
     * @throws std::runtime_error if file cannot be opened or ueGrp_pars dataset
     * is missing/invalid
     */
    [[nodiscard]] static std::vector<CuphyPuschTvUeGrpParams>
    read_ue_grp_params_from_file(const std::string_view filename);

    /**
     * Reads transport block parameters from an HDF5 PUSCH test vector file.
     *
     * @param[in] filename Path to the HDF5 test vector file
     * @return Vector of transport block parameters structures loaded from the
     * file
     * @throws std::runtime_error if file cannot be opened or tb_pars dataset is
     * missing/invalid
     */
    [[nodiscard]] static std::vector<CuphyPuschTvTbParams>
    read_tb_params_from_file(const std::string_view filename);

    /**
     * Reads a scalar value from an HDF5 dataset.
     *
     * @tparam T The data type to read (must satisfy Hdf5Compatible)
     * @param[in] dataset_name Name of the dataset in the HDF5 file
     * @return The scalar value from the dataset
     * @throws std::runtime_error if dataset is not found, empty, or not
     * scalar-compatible
     */
    template <Hdf5Compatible T>
    [[nodiscard]] T read_scalar(const std::string_view dataset_name) const;

    /**
     * Reads an array dataset of any dimensionality from the HDF5 file.
     *
     * @tparam T The element data type (must satisfy Hdf5Compatible)
     * @param[in] dataset_name Name of the dataset in the HDF5 file
     * @return Hdf5Array containing flattened data and dimensions
     * @throws std::runtime_error if dataset is not found or cannot be read
     */
    template <Hdf5Compatible T>
    [[nodiscard]] Hdf5Array<T> read_array(const std::string_view dataset_name) const;

    /**
     * Reads a complex-valued array from HDF5 as a real-valued array with extra dimension
     *
     * Reads HDF5 compound types with 're' and 'im' fields and converts them to
     * a real-valued array where the last dimension is 2 (index 0 = real, index 1 = imag).
     *
     * @tparam T The real-valued data type (must be floating point or half-precision)
     * @param[in] dataset_name Name of the dataset in the HDF5 file
     * @return Hdf5Array containing flattened data and dimensions (last dimension is 2)
     * @throws std::runtime_error if dataset is not found, not a complex type, or cannot be read
     *
     * @note The added dimension is the last dimension, i.e. the fastest changing in memory.
     */
    template <ComplexArrayType T>
    [[nodiscard]] Hdf5Array<T> read_complex_array(const std::string_view dataset_name) const;

private:
    std::string filename_;                                    //!< Path to the test vector file
    mutable std::optional<CuphyPuschTvGnbParams> gnb_params_; //!< gNB parameters (lazy loaded)
    mutable std::optional<std::vector<CuphyPuschTvUeGrpParams>>
            ue_grp_params_; //!< UE group parameters (lazy loaded)
    mutable std::optional<std::vector<CuphyPuschTvTbParams>>
            tb_params_; //!< Transport block parameters (lazy loaded)
};

} // namespace ran::aerial_tv

#endif // RAN_AERIAL_TV_CUPHY_PUSCH_TV_HPP
