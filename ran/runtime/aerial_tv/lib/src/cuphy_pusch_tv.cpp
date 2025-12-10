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
#include <cstddef>
#include <cstdint>
#include <exception>
#include <format>
#include <functional>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <hdf5hpp.hpp>

#include "aerial_tv/cuphy_pusch_tv.hpp"

namespace ran::aerial_tv {

CuphyPuschTestVector::CuphyPuschTestVector(std::string filename) : filename_{std::move(filename)} {
    // Parameters are loaded lazily on first access
}

const CuphyPuschTvGnbParams &CuphyPuschTestVector::get_gnb_params() const {
    if (!gnb_params_) {
        gnb_params_ = read_gnb_params_from_file(filename_);
    }
    return *gnb_params_;
}

const std::vector<CuphyPuschTvUeGrpParams> &CuphyPuschTestVector::get_ue_grp_params() const {
    if (!ue_grp_params_) {
        ue_grp_params_ = read_ue_grp_params_from_file(filename_);
    }
    return *ue_grp_params_;
}

const CuphyPuschTvUeGrpParams &
CuphyPuschTestVector::get_ue_grp_params(const std::size_t ue_grp_idx) const {
    const auto &ue_grp_params_vec = get_ue_grp_params();
    if (ue_grp_params_vec.empty() || ue_grp_idx >= ue_grp_params_vec.size()) {
        throw std::out_of_range(std::format(
                "UE group index {} is out of range (available: {})",
                ue_grp_idx,
                ue_grp_params_vec.size()));
    }
    return ue_grp_params_vec[ue_grp_idx];
}

const std::vector<CuphyPuschTvTbParams> &CuphyPuschTestVector::get_tb_params() const {
    if (!tb_params_) {
        tb_params_ = read_tb_params_from_file(filename_);
    }
    return *tb_params_;
}

const CuphyPuschTvTbParams &CuphyPuschTestVector::get_tb_params(const std::size_t tb_idx) const {
    const auto &tb_params_vec = get_tb_params();
    if (tb_params_vec.empty() || tb_idx >= tb_params_vec.size()) {
        throw std::out_of_range(std::format(
                "Transport block index {} is out of range (available: {})",
                tb_idx,
                tb_params_vec.size()));
    }
    return tb_params_vec[tb_idx];
}

CuphyPuschTvGnbParams
CuphyPuschTestVector::read_gnb_params_from_file(const std::string_view filename) {
    CuphyPuschTvGnbParams gnb_params{};

    try {
        // Open HDF5 file
        auto hdf_file = hdf5hpp::hdf5_file::open(filename.data(), H5F_ACC_RDONLY);

        // Read gnb_pars dataset (compound data type)
        if (!hdf_file.is_valid_dataset("gnb_pars")) {
            throw std::runtime_error("Dataset 'gnb_pars' not found in HDF5 file");
        }
        auto gnb_dataset = hdf_file.open_dataset("gnb_pars");

        // Get dataspace to check dimensions
        const auto dataspace = gnb_dataset.get_dataspace();
        const auto dims = dataspace.get_dimensions();

        if (dims.empty() || dims[0] == 0) {
            throw std::runtime_error("gnb_pars dataset is empty");
        }

        // For compound data types, access the first element and read fields using
        // compound member access
        const auto gnb_element = gnb_dataset[0];

        // Extract fields using compound member access
        gnb_params.n_user_groups = gnb_element["nUserGroups"].as<double>();
        gnb_params.mu = gnb_element["mu"].as<std::uint32_t>();
        gnb_params.n_rx = gnb_element["nRx"].as<std::uint32_t>();
        gnb_params.n_prb = gnb_element["nPrb"].as<std::uint32_t>();
        gnb_params.cell_id = gnb_element["cellId"].as<std::uint32_t>();
        gnb_params.slot_number = gnb_element["slotNumber"].as<std::uint32_t>();
        gnb_params.num_tb = gnb_element["numTb"].as<std::uint32_t>();

        // Boolean flags
        gnb_params.enable_early_harq = gnb_element["enableEarlyHarq"].as<std::uint8_t>();
        gnb_params.enable_cfo_correction = gnb_element["enableCfoCorrection"].as<std::uint8_t>();
        gnb_params.enable_cfo_estimation = gnb_element["enableCfoEstimation"].as<std::uint8_t>();
        gnb_params.enable_to_estimation = gnb_element["enableToEstimation"].as<std::uint8_t>();
        gnb_params.enable_to_correction = gnb_element["enableToCorrection"].as<std::uint8_t>();
        gnb_params.tdi_mode = gnb_element["TdiMode"].as<std::uint8_t>();
        gnb_params.enable_dft_s_ofdm = gnb_element["enableDftSOfdm"].as<std::uint8_t>();
        gnb_params.enable_rssi_measurement =
                gnb_element["enableRssiMeasurement"].as<std::uint8_t>();
        gnb_params.enable_sinr_measurement =
                gnb_element["enableSinrMeasurement"].as<std::uint8_t>();
        gnb_params.enable_static_dynamic_beamforming =
                gnb_element["enable_static_dynamic_beamforming"].as<std::uint8_t>();

        // LDPC parameters
        gnb_params.ldpc_early_termination = gnb_element["ldpcEarlyTermination"].as<std::uint32_t>();
        gnb_params.ldpc_algo_index = gnb_element["ldpcAlgoIndex"].as<std::uint32_t>();
        gnb_params.ldpc_flags = gnb_element["ldpcFlags"].as<std::uint32_t>();
        gnb_params.ldpc_use_half = gnb_element["ldpcUseHalf"].as<std::uint32_t>();
        gnb_params.num_bbu_layers = gnb_element["numBbuLayers"].as<std::uint32_t>();
        gnb_params.ldpc_max_num_itr = gnb_element["ldpcMaxNumItr"].as<std::uint8_t>();
        gnb_params.ldpc_max_num_itr_alg_idx = gnb_element["ldpcMaxNumItrAlgIdx"].as<std::uint8_t>();

        // Channel estimation parameters
        gnb_params.dmrs_ch_est_alg_idx = gnb_element["dmrsChEstAlgIdx"].as<std::uint8_t>();
        gnb_params.enable_per_prg_ch_est = gnb_element["enablePerPrgChEst"].as<std::uint8_t>();
        gnb_params.eq_coeff_algo_idx = gnb_element["eqCoeffAlgoIdx"].as<std::uint8_t>();
        gnb_params.list_length = gnb_element["listLength"].as<std::uint8_t>();

        // CSI parameters
        gnb_params.enable_csi_p2_fapiv3 = gnb_element["enableCsiP2Fapiv3"].as<std::uint8_t>();
        gnb_params.n_csi2_maps = gnb_element["nCsi2Maps"].as<std::uint16_t>();

        // Array fields
        gnb_params.csi2_maps_sum_of_prm_sizes =
                gnb_element["csi2Maps_sumOfPrmSizes"].as<std::array<std::uint8_t, 4>>();
        gnb_params.csi2_maps_buffer_start_idxs =
                gnb_element["csi2Maps_bufferStartIdxs"].as<std::array<std::uint32_t, 4>>();
        gnb_params.csi2_maps_buffer =
                gnb_element["csi2Maps_buffer"]
                        .as<std::array<std::uint16_t, MAX_CSI2_MAPS_BUFFER_SIZE>>();

    } catch (const hdf5hpp::hdf5_exception &e) {
        throw std::runtime_error(std::format(
                "Failed to load gNB parameters from '{}': HDF5 error: {}", filename, e.what()));
    } catch (const std::exception &e) {
        throw std::runtime_error(
                std::format("Failed to load gNB parameters from '{}': {}", filename, e.what()));
    }

    return gnb_params;
}

std::vector<CuphyPuschTvUeGrpParams>
CuphyPuschTestVector::read_ue_grp_params_from_file(const std::string_view filename) {
    std::vector<CuphyPuschTvUeGrpParams> ue_grp_params_vec{};

    try {
        // Open HDF5 file
        auto hdf_file = hdf5hpp::hdf5_file::open(filename.data(), H5F_ACC_RDONLY);

        // Read ueGrp_pars dataset (compound data type)
        if (!hdf_file.is_valid_dataset("ueGrp_pars")) {
            throw std::runtime_error("Dataset 'ueGrp_pars' not found in HDF5 file");
        }

        auto ue_grp_dataset = hdf_file.open_dataset("ueGrp_pars");

        // Get dataspace to check dimensions
        const auto ue_grp_dataspace = ue_grp_dataset.get_dataspace();
        const auto ue_grp_dims = ue_grp_dataspace.get_dimensions();

        if (ue_grp_dims.empty() || ue_grp_dims[0] == 0) {
            throw std::runtime_error("ueGrp_pars dataset is empty");
        }

        const std::size_t num_ue_groups = ue_grp_dims[0];
        ue_grp_params_vec.reserve(num_ue_groups);

        // Read all UE group entries
        for (std::size_t i = 0; i < num_ue_groups; ++i) {
            CuphyPuschTvUeGrpParams ue_grp_params{};

            // For compound data types, access element by index
            const auto ue_grp_element = ue_grp_dataset[static_cast<int>(i)];

            // Extract fields using compound member access
            ue_grp_params.n_ues = ue_grp_element["nUes"].as<std::uint16_t>();
            ue_grp_params.ue_prm_idxs =
                    ue_grp_element["UePrmIdxs"].as<std::vector<std::uint16_t>>();
            ue_grp_params.start_prb = ue_grp_element["startPrb"].as<std::uint16_t>();
            ue_grp_params.n_prb = ue_grp_element["nPrb"].as<std::uint16_t>();
            ue_grp_params.start_symbol_index =
                    ue_grp_element["StartSymbolIndex"].as<std::uint8_t>();
            ue_grp_params.nr_of_symbols = ue_grp_element["NrOfSymbols"].as<std::uint8_t>();
            ue_grp_params.prg_size = ue_grp_element["prgSize"].as<std::uint16_t>();
            ue_grp_params.dmrs_sym_loc_bmsk = ue_grp_element["dmrsSymLocBmsk"].as<std::uint16_t>();
            ue_grp_params.rssi_sym_loc_bmsk = ue_grp_element["rssiSymLocBmsk"].as<std::uint16_t>();
            ue_grp_params.n_uplink_streams = ue_grp_element["nUplinkStreams"].as<double>();

            ue_grp_params_vec.push_back(ue_grp_params);
        }

    } catch (const hdf5hpp::hdf5_exception &e) {
        throw std::runtime_error(std::format(
                "Failed to load UE group parameters from '{}': HDF5 error: {}",
                filename,
                e.what()));
    } catch (const std::exception &e) {
        throw std::runtime_error(std::format(
                "Failed to load UE group parameters from '{}': {}", filename, e.what()));
    }

    return ue_grp_params_vec;
}

std::vector<CuphyPuschTvTbParams>
CuphyPuschTestVector::read_tb_params_from_file(const std::string_view filename) {
    std::vector<CuphyPuschTvTbParams> tb_params_vec{};

    try {
        // Open HDF5 file
        auto hdf_file = hdf5hpp::hdf5_file::open(filename.data(), H5F_ACC_RDONLY);

        // Read tb_pars dataset (compound data type)
        if (!hdf_file.is_valid_dataset("tb_pars")) {
            throw std::runtime_error("Dataset 'tb_pars' not found in HDF5 file");
        }

        auto tb_dataset = hdf_file.open_dataset("tb_pars");

        // Get dataspace to check dimensions
        const auto tb_dataspace = tb_dataset.get_dataspace();
        const auto tb_dims = tb_dataspace.get_dimensions();

        if (tb_dims.empty() || tb_dims[0] == 0) {
            throw std::runtime_error("tb_pars dataset is empty");
        }

        const std::size_t num_tbs = tb_dims[0];
        tb_params_vec.reserve(num_tbs);

        // Read all TB entries
        for (std::size_t i = 0; i < num_tbs; ++i) {
            CuphyPuschTvTbParams tb_params{};

            // For compound data types, access element by index
            const auto tb_element = tb_dataset[static_cast<int>(i)];

            // Basic transport block parameters
            tb_params.n_rnti = tb_element["nRnti"].as<std::uint32_t>();
            tb_params.num_layers = tb_element["numLayers"].as<std::uint32_t>();
            tb_params.start_sym = tb_element["startSym"].as<std::uint32_t>();
            tb_params.num_sym = tb_element["numSym"].as<std::uint32_t>();
            tb_params.user_group_index = tb_element["userGroupIndex"].as<std::uint32_t>();
            tb_params.data_scram_id = tb_element["dataScramId"].as<std::uint32_t>();

            // MCS and coding parameters
            tb_params.mcs_table_index = tb_element["mcsTableIndex"].as<std::uint32_t>();
            tb_params.mcs_index = tb_element["mcsIndex"].as<std::uint32_t>();
            tb_params.rv = tb_element["rv"].as<std::uint32_t>();
            tb_params.ndi = tb_element["ndi"].as<std::uint32_t>();
            tb_params.n_tb_byte = tb_element["nTbByte"].as<std::uint32_t>();
            tb_params.n_cb = tb_element["nCb"].as<std::uint32_t>();

            // LBRM parameters
            tb_params.i_lbrm = tb_element["I_LBRM"].as<std::uint8_t>();
            tb_params.max_layers = tb_element["maxLayers"].as<std::uint8_t>();
            tb_params.max_qm = tb_element["maxQm"].as<std::uint8_t>();
            tb_params.n_prb_lbrm = tb_element["n_PRB_LBRM"].as<std::uint16_t>();

            // Modulation and coding parameters
            tb_params.qam_mod_order = tb_element["qamModOrder"].as<std::uint8_t>();
            tb_params.target_code_rate = tb_element["targetCodeRate"].as<std::uint16_t>();

            // HARQ and CSI parameters
            tb_params.n_bits_harq = tb_element["nBitsHarq"].as<std::uint16_t>();
            tb_params.n_bits_csi1 = tb_element["nBitsCsi1"].as<std::uint16_t>();
            tb_params.pdu_bitmap = tb_element["pduBitmap"].as<std::uint16_t>();
            tb_params.alpha_scaling = tb_element["alphaScaling"].as<std::uint8_t>();
            tb_params.beta_offset_harq_ack = tb_element["betaOffsetHarqAck"].as<std::uint8_t>();
            tb_params.beta_offset_csi1 = tb_element["betaOffsetCsi1"].as<std::uint8_t>();
            tb_params.beta_offset_csi2 = tb_element["betaOffsetCsi2"].as<std::uint8_t>();

            // Parameter arrays
            tb_params.n_part1_prms =
                    tb_element["nPart1Prms"]
                            .as<std::array<std::uint8_t, MAX_CSI2_REPORTS_PER_TB>>();
            tb_params.prm_sizes =
                    tb_element["prmSizes"]
                            .as<std::array<std::uint8_t, MAX_TOTAL_CSI1_PARAMETERS_PER_TB>>();
            tb_params.prm_offsets =
                    tb_element["prmOffsets"]
                            .as<std::array<std::uint16_t, MAX_TOTAL_CSI1_PARAMETERS_PER_TB>>();
            tb_params.csi2_size_map_idx =
                    tb_element["csi2sizeMapIdx"]
                            .as<std::array<std::uint16_t, MAX_CSI2_REPORTS_PER_TB>>();

            // CSI part 2 parameters
            tb_params.n_csi2_reports = tb_element["nCsi2Reports"].as<std::uint16_t>();
            tb_params.flag_csi_part2 = tb_element["flagCsiPart2"].as<std::uint16_t>();
            tb_params.rank_bit_offset = tb_element["rankBitOffset"].as<std::uint8_t>();
            tb_params.rank_bit_size = tb_element["rankBitSize"].as<std::uint8_t>();

            // DMRS parameters
            tb_params.dmrs_addl_position = tb_element["dmrsAddlPosition"].as<std::uint32_t>();
            tb_params.dmrs_max_length = tb_element["dmrsMaxLength"].as<std::uint32_t>();
            tb_params.dmrs_scram_id = tb_element["dmrsScramId"].as<std::uint32_t>();
            tb_params.n_scid = tb_element["nSCID"].as<std::uint32_t>();
            tb_params.dmrs_port_bmsk = tb_element["dmrsPortBmsk"].as<std::uint32_t>();
            tb_params.dmrs_sym_loc_bmsk = tb_element["dmrsSymLocBmsk"].as<std::uint32_t>();
            tb_params.rssi_sym_loc_bmsk = tb_element["rssiSymLocBmsk"].as<std::uint32_t>();
            tb_params.num_dmrs_cdm_grps_no_data =
                    tb_element["numDmrsCdmGrpsNoData"].as<std::uint8_t>();

            // DTX and transform precoding
            tb_params.dtx_threshold = tb_element["DTXthreshold"].as<float>();
            tb_params.enable_tf_prcd = tb_element["enableTfPrcd"].as<std::uint8_t>();

            // PUSCH and slot parameters
            tb_params.pusch_identity = tb_element["puschIdentity"].as<std::uint8_t>();
            tb_params.n_slot_frame = tb_element["N_slot_frame"].as<std::uint8_t>();
            tb_params.n_symb_slot = tb_element["N_symb_slot"].as<std::uint8_t>();

            // Low PAPR parameters
            tb_params.group_or_sequence_hopping =
                    tb_element["groupOrSequenceHopping"].as<std::uint8_t>();
            tb_params.low_papr_group_number = tb_element["lowPaprGroupNumber"].as<std::uint8_t>();
            tb_params.low_papr_sequence_number =
                    tb_element["lowPaprSequenceNumber"].as<std::uint16_t>();

            tb_params_vec.push_back(tb_params);
        }

    } catch (const hdf5hpp::hdf5_exception &e) {
        throw std::runtime_error(std::format(
                "Failed to load transport block parameters from '{}': HDF5 error: {}",
                filename,
                e.what()));
    } catch (const std::exception &e) {
        throw std::runtime_error(std::format(
                "Failed to load transport block parameters from '{}': {}", filename, e.what()));
    }

    return tb_params_vec;
}

template <Hdf5Compatible T>
T CuphyPuschTestVector::read_scalar(const std::string_view dataset_name) const {
    try {
        auto hdf_file = hdf5hpp::hdf5_file::open(filename_.data(), H5F_ACC_RDONLY);

        if (!hdf_file.is_valid_dataset(dataset_name.data())) {
            throw std::runtime_error(
                    std::format("Dataset '{}' not found in HDF5 file", dataset_name));
        }

        auto dataset = hdf_file.open_dataset(dataset_name.data());
        const auto dataspace = dataset.get_dataspace();
        const auto dims = dataspace.get_dimensions();

        // Check if dataset is scalar-compatible (single element)
        const std::size_t total_elements = std::accumulate(
                dims.begin(), dims.end(), std::size_t{1}, std::multiplies<std::size_t>{});

        if (total_elements != 1) {
            throw std::runtime_error(std::format(
                    "Dataset '{}' is not scalar-compatible: has {} elements",
                    dataset_name,
                    total_elements));
        }

        // Read the single value
        T value{};
        dataset.read(&value);
        return value;

    } catch (const hdf5hpp::hdf5_exception &e) {
        throw std::runtime_error(std::format(
                "Failed to read scalar '{}' from '{}': HDF5 error: {}",
                dataset_name,
                filename_,
                e.what()));
    } catch (const std::exception &e) {
        throw std::runtime_error(std::format(
                "Failed to read scalar '{}' from '{}': {}", dataset_name, filename_, e.what()));
    }
}

template <Hdf5Compatible T>
Hdf5Array<T> CuphyPuschTestVector::read_array(const std::string_view dataset_name) const {
    try {
        auto hdf_file = hdf5hpp::hdf5_file::open(filename_.data(), H5F_ACC_RDONLY);

        if (!hdf_file.is_valid_dataset(dataset_name.data())) {
            throw std::runtime_error(
                    std::format("Dataset '{}' not found in HDF5 file", dataset_name));
        }

        auto dataset = hdf_file.open_dataset(dataset_name.data());
        const auto dataspace = dataset.get_dataspace();
        const auto dims = dataspace.get_dimensions();

        // Calculate total number of elements
        const std::size_t total_elements = std::accumulate(
                dims.begin(), dims.end(), std::size_t{1}, std::multiplies<std::size_t>{});

        if (total_elements == 0) {
            throw std::runtime_error(std::format("Dataset '{}' is empty", dataset_name));
        }

        // Convert dimensions to size_t vector
        std::vector<std::size_t> dimensions(dims.begin(), dims.end());

        // Read all data into vector
        std::vector<T> data(total_elements);
        dataset.read(data.data());

        return Hdf5Array<T>{std::move(data), std::move(dimensions)};

    } catch (const hdf5hpp::hdf5_exception &e) {
        throw std::runtime_error(std::format(
                "Failed to read array '{}' from '{}': HDF5 error: {}",
                dataset_name,
                filename_,
                e.what()));
    } catch (const std::exception &e) {
        throw std::runtime_error(std::format(
                "Failed to read array '{}' from '{}': {}", dataset_name, filename_, e.what()));
    }
}

template <ComplexArrayType T>
Hdf5Array<T> CuphyPuschTestVector::read_complex_array(const std::string_view dataset_name) const {
    try {
        auto hdf_file = hdf5hpp::hdf5_file::open(filename_.data(), H5F_ACC_RDONLY);

        if (!hdf_file.is_valid_dataset(dataset_name.data())) {
            throw std::runtime_error(
                    std::format("Dataset '{}' not found in HDF5 file", dataset_name));
        }

        auto dataset = hdf_file.open_dataset(dataset_name.data());
        const auto dataspace = dataset.get_dataspace();
        const auto dims = dataspace.get_dimensions();

        if (dims.empty()) {
            throw std::runtime_error(std::format("Dataset '{}' has no dimensions", dataset_name));
        }

        // Calculate total number of complex elements
        const std::size_t total_complex_elements = std::accumulate(
                dims.begin(), dims.end(), std::size_t{1}, std::multiplies<std::size_t>{});

        if (total_complex_elements == 0) {
            throw std::runtime_error(std::format("Dataset '{}' is empty", dataset_name));
        }

        // Create temporary struct to match HDF5 compound type
        struct ComplexValue {
            T re;
            T im;
        };

        // Read compound data
        std::vector<ComplexValue> complex_data(total_complex_elements);
        dataset.read(complex_data.data());

        // Convert to interleaved real array: [re0, im0, re1, im1, ...]
        std::vector<T> real_data(total_complex_elements * 2);
        for (std::size_t i = 0; i < total_complex_elements; ++i) {
            real_data[i * 2 + 0] = complex_data[i].re;
            real_data[i * 2 + 1] = complex_data[i].im;
        }

        // Add dimension of 2 for real/imaginary parts
        std::vector<std::size_t> dimensions(dims.begin(), dims.end());
        dimensions.push_back(2);

        return Hdf5Array<T>{std::move(real_data), std::move(dimensions)};

    } catch (const hdf5hpp::hdf5_exception &e) {
        throw std::runtime_error(std::format(
                "Failed to read complex array '{}' from '{}': HDF5 error: {}",
                dataset_name,
                filename_,
                e.what()));
    } catch (const std::exception &e) {
        throw std::runtime_error(std::format(
                "Failed to read complex array '{}' from '{}': {}",
                dataset_name,
                filename_,
                e.what()));
    }
}

/**
 * Instantiates CuphyPuschTestVector templates for a type
 *
 * @param T Type to instantiate (e.g., float, double, std::uint8_t)
 */
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define INSTANTIATE_PUSCH_TV_TEMPLATES(T)                                                          \
    template T CuphyPuschTestVector::read_scalar<T>(const std::string_view) const;                 \
    template Hdf5Array<T> CuphyPuschTestVector::read_array<T>(const std::string_view) const;

/**
 * Instantiates complex array template for floating point types
 *
 * @param T Type to instantiate (float or double)
 */
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define INSTANTIATE_COMPLEX_ARRAY_TEMPLATE(T)                                                      \
    template Hdf5Array<T> CuphyPuschTestVector::read_complex_array<T>(const std::string_view) const;

// Explicit template instantiations for common types

// Signed integer types
INSTANTIATE_PUSCH_TV_TEMPLATES(std::int8_t)
INSTANTIATE_PUSCH_TV_TEMPLATES(std::int16_t)
INSTANTIATE_PUSCH_TV_TEMPLATES(std::int32_t)
INSTANTIATE_PUSCH_TV_TEMPLATES(std::int64_t)

// Unsigned integer types
INSTANTIATE_PUSCH_TV_TEMPLATES(std::uint8_t)
INSTANTIATE_PUSCH_TV_TEMPLATES(std::uint16_t)
INSTANTIATE_PUSCH_TV_TEMPLATES(std::uint32_t)
INSTANTIATE_PUSCH_TV_TEMPLATES(std::uint64_t)

// Floating point types
INSTANTIATE_PUSCH_TV_TEMPLATES(float)
INSTANTIATE_PUSCH_TV_TEMPLATES(double)

// Complex array support (for floating point and half-precision types)
INSTANTIATE_COMPLEX_ARRAY_TEMPLATE(float)
INSTANTIATE_COMPLEX_ARRAY_TEMPLATE(double)
INSTANTIATE_COMPLEX_ARRAY_TEMPLATE(__half)

#undef INSTANTIATE_PUSCH_TV_TEMPLATES
#undef INSTANTIATE_COMPLEX_ARRAY_TEMPLATE

} // namespace ran::aerial_tv
