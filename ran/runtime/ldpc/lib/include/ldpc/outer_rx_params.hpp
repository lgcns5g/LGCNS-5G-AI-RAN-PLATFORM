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

#ifndef RAN_LDPC_OUTER_RX_PARAMS_HPP
#define RAN_LDPC_OUTER_RX_PARAMS_HPP

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include <cuphy.h>
#include <driver_types.h>

#include "ldpc/derate_match_params.hpp"
#include "ldpc/ldpc_params.hpp"
#include "memory/buffer.hpp"
#include "memory/device_allocators.hpp"
#include "ran_common.hpp"

namespace ran::ldpc {

/**
 * Get the scrambling initialization value for a given RNTI and data scrambling ID.
 *
 * The scrambling initialization value is calculated per 3GPP TS 38.211 Section 6.3.1.1:
 * c_init = n_RNTI * 2^15 + n_ID
 *
 * Where:
 * - n_RNTI is the Radio Network Temporary Identifier
 * - n_ID is the data scrambling identity (data_scram_id)
 * - 2^15 = 32768 is the shift value defined in the specification
 *
 * The scrambling sequence generator uses a length-31 Gold sequence with
 * initialization value c_init.
 *
 * @param[in] rnti Radio Network Temporary Identifier (n_RNTI)
 * @param[in] data_scram_id Data scrambling ID (n_ID)
 * @return Scrambling initialization value (c_init)
 *
 * @see 3GPP TS 38.211 Section 6.3.1.1 (Scrambling)
 * @see 3GPP TS 38.211 Section 5.2.1 (Pseudo-random sequence generation)
 */
[[nodiscard]] inline std::uint32_t
get_scrambling_init(std::uint32_t rnti, std::uint32_t data_scram_id) {
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
    return (rnti << 15U) + data_scram_id;
}

/**
 * Get the rate matching length for a given number of PRBs, layers, modulation order, number of
 * symbols, number of DMRS CDM groups without data, and DMRS symbol location bitmask.
 *
 * @param[in] num_prbs Number of allocated PRBs.
 * @param[in] num_layers Number of layers allocated for the UE.
 * @param[in] mod_order Modulation order.
 * @param[in] num_symbols Number of symbols allocated for the transmission.
 * @param[in] num_dmrs_cdm_grps_no_data Number of DMRS CDM groups without data.
 * @param[in] dmrs_sym_loc_bmsk DMRS symbol location bitmask in SCF FAPI format, 0 = no DMRS, 1 =
 * DMRS.
 * @return Rate matching length
 *
 * @see 3GPP TS 38.214 Section 6.1.4.2.
 * @throws std::invalid_argument if num_prbs is 0.
 */
[[nodiscard]] inline std::uint32_t get_rate_matching_length(
        std::uint16_t num_prbs,
        std::uint8_t num_layers,
        ModulationOrder mod_order,
        // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
        std::uint8_t num_symbols,
        std::uint8_t num_dmrs_cdm_grps_no_data,
        std::uint16_t dmrs_sym_loc_bmsk) {

    if (num_prbs == 0) {
        throw std::invalid_argument("num_prbs must be greater than 0");
    }

    const auto n_dmrs_syms = static_cast<std::uint32_t>(std::popcount(dmrs_sym_loc_bmsk));
    const auto n_data_sym = static_cast<std::uint32_t>(num_symbols) - n_dmrs_syms;

    constexpr std::uint32_t NUM_SUBC_PER_PRB = 12;
    std::uint32_t n_re_prime = NUM_SUBC_PER_PRB * num_prbs * n_data_sym;
    if (num_dmrs_cdm_grps_no_data == 1) { // DMRS symbols contain data too.
        n_re_prime += ((NUM_SUBC_PER_PRB >> 1U) * num_prbs * n_dmrs_syms);
    }

    // By TS 38.214 Section 6.1.4.2, the number of RE's per PRB is set to a maximum of 156.
    const std::uint32_t n_re = std::min(156U, n_re_prime / num_prbs) * num_prbs;
    const auto rate_matching_length = n_re * static_cast<std::uint32_t>(mod_order) * num_layers;
    return rate_matching_length;
}

/**
 * Parameters needed by the PUSCH receiver pipeline outer_rx for processing
 * a single transport block.
 *
 * This class encapsulates all parameters required for processing a single
 * transport block by the PUSCH receiver pipeline outer_rx, which comprises
 * LDPC rate matching, decoding and CRC decoding. The parameters include:

 * - LDPC decoding parameters
 * - Rate matching parameters
 * - Layer mapping
 * - Scrambling configuration
 */
class SingleTbPuschOuterRxParams final {
public:
    /**
     * Construct PUSCH receiver pipeline outer_rx parameters for a single transport
     * block.
     *
     * Creates SingleTbPuschOuterRxParams using pre-computed LDPC parameters and
     * optional derate matching configuration.
     *
     * @param[in] ldpc_params Pre-computed LDPC encoding/decoding parameters
     * @param[in] de_rm_params Optional derate matching parameters. Needed if used for
     * derate matching.
     */
    explicit SingleTbPuschOuterRxParams(
            const LdpcParams &ldpc_params,
            const std::optional<DerateMatchParams> &de_rm_params = std::nullopt);

    /**
     * Get derate matching configuration parameters
     *
     * @return Const reference to derate matching parameters
     * @throws std::runtime_error if derate matching parameters have not been set
     */
    [[nodiscard]]
    const DerateMatchParams &de_rm_params() const;

    /**
     * Get LDPC parameters
     *
     * @return Const reference to LDPC parameters
     */
    [[nodiscard]]
    const LdpcParams &ldpc_params() const noexcept {
        return ldpc_params_;
    }

    /**
     * Whether DFT-S-OFDM transform precoding is enabled
     *
     * @return True if DFT-S-OFDM transform precoding is enabled
     */
    [[nodiscard]]
    bool enable_tf_prcd() const noexcept {
        return enable_tf_prcd_;
    }

    /**
     * Convert parameters to PerTbParams structure
     *
     * @param[out] tb_params PerTbParams structure to populate with converted parameters
     */
    void to_per_tb_params(PerTbParams &tb_params) const;

private:
    // Derate matching parameters
    DerateMatchParams de_rm_params_; //!< Derate matching parameters for this transport block

    // Advanced features
    bool enable_tf_prcd_{false}; //!< Enable transform precoding (currently not supported)

    // LDPC parameters
    LdpcParams ldpc_params_; //!< LDPC parameters for this transport block
};

/**
 * Parameters for PUSCH receiver pipeline outer_rx in a single slot.
 *
 * This class manages the parameters for PUSCH receiver pipeline outer_rx within a
 * single slot, including transport block parameters, LDPC parameters, and
 * derate matching parameters. It provides efficient GPU memory management and
 * conversion to cuPHY-compatible formats.
 *
 * The class automatically converts SingleTbPuschOuterRxParams to PerTbParams
 * format required by cuPHY and manages the related CPU and GPU memory buffers
 * for all transport blocks in the slot.
 */
class PuschOuterRxParams final {
public:
    /**
     * Construct parameters for PUSCH receiver pipeline outer_rx.
     *
     * Creates a PuschOuterRxParams object. Automatically
     * converts the SingleTbPuschOuterRxParams to cuPHY-compatible PerTbParams
     * format and allocates both CPU and GPU memory buffers.
     *
     * @param[in] pusch_outer_rx_params Vector of PUSCH receiver outer_rx
     * parameters for all TBs in the slot
     * @param[in] sch_user_idxs Vector of scheduled user indices corresponding to
     * each single transport block PUSCH receiver outer_rx parameters object.
     */
    explicit PuschOuterRxParams(
            const std::vector<SingleTbPuschOuterRxParams> &pusch_outer_rx_params,
            const std::vector<std::uint16_t> &sch_user_idxs);

    /**
     * Get pointer to CPU PerTbParams buffer
     *
     * @return Pointer to CPU memory containing PerTbParams for all transport
     * blocks
     */
    [[nodiscard]]
    PerTbParams *get_per_tb_params_cpu_ptr() noexcept {
        return tb_params_cpu_.addr();
    }

    /**
     * Get const pointer to CPU PerTbParams buffer
     *
     * @return Const pointer to CPU memory containing PerTbParams for all
     * transport blocks
     */
    [[nodiscard]]
    const PerTbParams *get_per_tb_params_cpu_ptr() const noexcept {
        return tb_params_cpu_.addr();
    }

    /**
     * Get pointer to GPU PerTbParams buffer
     *
     * @return Pointer to GPU memory containing PerTbParams for all transport
     * blocks
     */
    [[nodiscard]]
    PerTbParams *get_per_tb_params_gpu_ptr() noexcept {
        return tb_params_gpu_.addr();
    }

    /**
     * Get const pointer to GPU PerTbParams buffer
     *
     * @return Const pointer to GPU memory containing PerTbParams for all
     * transport blocks
     */
    [[nodiscard]]
    const PerTbParams *get_per_tb_params_gpu_ptr() const noexcept {
        return tb_params_gpu_.addr();
    }

    /**
     * Copy cuPHY transport block parameters from CPU to GPU memory
     *
     * Asynchronously copies the PerTbParams data from CPU buffer to GPU buffer
     * using the specified CUDA stream for optimal performance.
     *
     * @param[in] stream CUDA stream to use for the memory copy operation
     */
    void copy_tb_params_to_gpu(cudaStream_t stream);

    /**
     * Get the number of scheduled UEs in this slot
     *
     * @return Number of scheduled user equipments (UEs) for this slot
     */
    [[nodiscard]]
    std::size_t get_num_sch_ues() const noexcept {
        return sch_user_idxs_.size();
    }

    /**
     * Get the scheduled user indices for this slot
     *
     * @return Span of scheduled user indices
     */
    [[nodiscard]]
    std::span<const std::uint16_t> get_sch_user_idxs() const noexcept {
        return sch_user_idxs_;
    }

    /**
     * Get the number of transport blocks in this slot
     *
     * @return Number of transport blocks
     */
    [[nodiscard]]
    std::size_t num_tbs() const noexcept {
        return pusch_outer_rx_params_.size();
    }

    /**
     * Get SingleTbPuschOuterRxParams for a specific index
     *
     * @param[in] idx Index of the transport block (must be less than
     * number of transport blocks)
     * @return Const reference to SingleTbPuschOuterRxParams at the specified
     * index
     */
    [[nodiscard]]
    const SingleTbPuschOuterRxParams &get_pusch_outer_rx_params_single_tb(std::size_t idx) const {
        return pusch_outer_rx_params_.at(idx);
    }

    /**
     * Get SingleTbPuschOuterRxParams for a specific index
     *
     * @param[in] idx Index of the transport block (must be less than number of
     * transport blocks)
     * @return Reference to SingleTbPuschOuterRxParams at the specified
     * index
     */
    [[nodiscard]]
    SingleTbPuschOuterRxParams &operator[](std::size_t idx) {
        return pusch_outer_rx_params_.at(idx);
    }

    /**
     * Get SingleTbPuschOuterRxParams for a specific index
     *
     * @param[in] idx Index of the transport block (must be less than number of
     * transport blocks)
     * @return Const reference to SingleTbPuschOuterRxParams at the
     * specified index
     */
    [[nodiscard]]
    const SingleTbPuschOuterRxParams &operator[](std::size_t idx) const {
        return pusch_outer_rx_params_.at(idx);
    }

private:
    /**
     * Convert all SingleTbPuschOuterRxParams to PerTbParams and populate CPU
     * buffer
     *
     * Converts each stored SingleTbPuschOuterRxParams to PerTbParams using the
     * LDPC and rate matching parameters, and populates the CPU buffer for use
     * with cuPHY.
     */
    void to_per_tb_params();

    std::vector<SingleTbPuschOuterRxParams>
            pusch_outer_rx_params_; //!< PUSCH receiver outer_rx parameters for all
                                    //!< TBs in this slot

    framework::memory::Buffer<PerTbParams, framework::memory::PinnedAlloc>
            tb_params_cpu_; //!< Host pinned memory buffer for cuPHY PerTbParams
    framework::memory::Buffer<PerTbParams, framework::memory::DeviceAlloc>
            tb_params_gpu_; //!< GPU device memory buffer for cuPHY PerTbParams

    std::vector<std::uint16_t> sch_user_idxs_; //!< Scheduled user index for each transport block
};

} // namespace ran::ldpc

#endif // RAN_LDPC_OUTER_RX_PARAMS_HPP
