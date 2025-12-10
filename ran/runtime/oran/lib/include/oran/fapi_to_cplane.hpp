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

#ifndef RAN_ORAN_FAPI_TO_CPLANE_HPP
#define RAN_ORAN_FAPI_TO_CPLANE_HPP

#include <cstdint>
#include <system_error>
#include <vector>

#include <scf_5g_fapi.h>

#include "oran/cplane_types.hpp"
#include "oran/numerology.hpp"
#include "oran/oran_errors.hpp"
#include "oran/oran_export.hpp"

namespace ran::oran {

/**
 * PRB chunk representing a contiguous allocation in frequency and time
 */
struct ORAN_EXPORT PrbChunk final {
    std::uint16_t rb_start{};          //!< Starting resource block index
    std::uint16_t rb_size{};           //!< Number of resource blocks
    std::uint8_t start_symbol_index{}; //!< Starting OFDM symbol index
    std::uint8_t num_of_symbols{};     //!< Number of OFDM symbols
};

/// Maximum PRB chunks capacity
inline constexpr std::size_t MAX_PRB_CHUNKS = 255;

/**
 * Container for aggregated PRB chunks with pre-allocated capacity
 */
struct ORAN_EXPORT PrbChunks final {
    std::vector<PrbChunk> chunks; //!< Vector of PRB chunks

    PrbChunks() { chunks.reserve(MAX_PRB_CHUNKS); }

    /**
     * Clear all chunks
     */
    void clear() { chunks.clear(); }
};

/**
 * Aggregates contiguous PRBs from PUSCH PDUs in an UL_TTI.request
 *
 * PRBs are considered contiguous and merged into a single chunk if they:
 * 1. Have the same start symbol index (start_symbol_index)
 * 2. Have the same number of symbols (num_of_symbols)
 * 3. Are adjacent in frequency (rb_start + rb_size of one equals rb_start of
 * next)
 *
 * The function extracts PRB allocation information from all PUSCH PDUs in the
 * request, sorts them by start symbol and frequency, then merges adjacent
 * allocations with matching symbol ranges. Multiple chunks may share the same
 * start symbol if they have different num_of_symbols or are non-contiguous in
 * frequency.
 *
 * The function also validates that for each start_symbol_index, the total
 * number of sections needed does not exceed MAX_CPLANE_SECTIONS (64). If the
 * limit is exceeded, an error is logged and the function returns an error code.
 * The output chunks vector is cleared before processing.
 *
 * @param[in] request UL_TTI.request containing PUSCH PDUs
 * @param[in] body_len Length of message body (from FAPI body_hdr.length or equivalent)
 * @param[out] output Aggregated PRB chunks (cleared before use)
 * @return Error code indicating success or specific error condition
 * @retval OranErrc::Success Operation completed successfully
 * @retval OranErrc::InvalidPrbAllocationRbSizeZero PRB allocation has zero rb_size
 * @retval OranErrc::invalid_prb_allocation_num_symbols_zero PRB allocation has zero num_of_symbols
 * @retval OranErrc::invalid_prb_allocation_exceeds_slot Symbol allocation exceeds slot boundary
 * @retval OranErrc::too_many_sections_for_symbol Section limit exceeded for a start symbol
 * @retval OranErrc::PduPayloadOutOfBounds PDU parsing exceeded payload bounds
 */
ORAN_EXPORT [[nodiscard]] std::error_code find_contiguous_prb_chunks(
        const scf_fapi_ul_tti_req_t &request, std::size_t body_len, PrbChunks &output);

/**
 * Convert PRB chunks to C-plane message infos
 *
 * Converts PRB chunks to ORAN C-plane Section Type 1 messages. Chunks are
 * grouped by their start_symbol_index, and each group is converted to one
 * message per antenna port. Within a message, each chunk becomes one or more
 * sections:
 * - Multiple non-contiguous chunks with the same start_symbol_index become
 * multiple sections in the same message
 * - Each section within a message can have different num_of_symbols
 * - Large chunks (>255 PRBs) are split into multiple sections
 *
 * The input chunks should already be validated by find_contiguous_prb_chunks to
 * ensure section limits are not exceeded.
 *
 * Creates messages for antenna port 0 first, then duplicates for other antenna
 * ports.
 *
 * @param[in] chunks PRB chunks to convert (must be sorted by start_symbol_index
 * and validated)
 * @param[in] slot_timing Slot timing information (frame, subframe, slot)
 * @param[in] num_antenna_ports Number of antenna ports in system
 * @param[in] tx_windows Base transmission timing windows (for symbol 0)
 * @param[in] numerology ORAN numerology parameters
 * @param[in,out] msg_infos Vector to append C-plane messages to (not cleared)
 * @return Error code indicating success or specific error condition
 * @retval OranErrc::Success Operation completed successfully
 * @retval OranErrc::invalid_num_antenna_ports Number of antenna ports is zero
 * @retval OranErrc::InvalidPrbAllocationRbSizeZero PRB chunk has zero rb_size
 * @retval OranErrc::invalid_prb_allocation_num_symbols_zero PRB chunk has zero num_of_symbols
 * @retval OranErrc::invalid_prb_allocation_exceeds_slot Symbol allocation exceeds slot boundary
 */
ORAN_EXPORT [[nodiscard]] std::error_code convert_prb_chunks_to_cplane(
        const PrbChunks &chunks,
        const OranSlotTiming &slot_timing,
        std::uint16_t num_antenna_ports,
        const OranTxWindows &tx_windows,
        const OranNumerology &numerology,
        std::vector<OranCPlaneMsgInfo> &msg_infos);

/**
 * Convert FAPI UL_TTI.request to C-plane message infos
 *
 * Processes a complete UL_TTI.request message and converts all contained
 * PDUs (PUSCH, PUCCH, PRACH, SRS) to their corresponding C-plane messages.
 * The output vector is cleared before processing. Caller should pre-allocate
 * the vector with reserve() for best performance.
 *
 * The FAPI slot field represents the slot within the frame. The conversion
 * to ORAN subframe/slot depends on the numerology (SCS):
 * - 15 kHz (μ=0): 1 slot per subframe
 * - 30 kHz (μ=1): 2 slots per subframe
 * - 60 kHz (μ=2): 4 slots per subframe
 * - 120 kHz (μ=3): 8 slots per subframe
 *
 * @param[in] request FAPI UL_TTI.request structure
 * @param[in] body_len Length of message body (from FAPI body_hdr.length or equivalent)
 * @param[in] num_antenna_ports Number of antenna ports in system
 * @param[in] numerology ORAN numerology parameters
 * @param[in] tx_windows Base transmission timing windows (for first symbol)
 * @param[out] msg_infos Output vector of C-plane message structures (cleared
 * before use)
 * @return Error code indicating success or specific error condition
 * @retval OranErrc::Success Operation completed successfully
 * @retval OranErrc::invalid_num_antenna_ports Number of antenna ports is zero
 * @retval OranErrc::InvalidPrbAllocationRbSizeZero PRB allocation has zero rb_size
 * @retval OranErrc::invalid_prb_allocation_num_symbols_zero PRB allocation has zero num_of_symbols
 * @retval OranErrc::invalid_prb_allocation_exceeds_slot Symbol allocation exceeds slot boundary
 * @retval OranErrc::too_many_sections_for_symbol Section limit exceeded for a start symbol
 * @retval OranErrc::PduPayloadOutOfBounds PDU parsing exceeded payload bounds
 */
ORAN_EXPORT [[nodiscard]] std::error_code convert_ul_tti_request_to_cplane(
        const scf_fapi_ul_tti_req_t &request,
        std::size_t body_len,
        std::uint16_t num_antenna_ports,
        const OranNumerology &numerology,
        const OranTxWindows &tx_windows,
        std::vector<OranCPlaneMsgInfo> &msg_infos);

/**
 * Convert FAPI UL_TTI.request to C-plane message infos (with pre-allocated
 * buffer)
 *
 * Overload for real-time performance-critical operations where the caller
 * pre-allocates and reuses the PRB chunks buffer to avoid runtime allocations.
 *
 * @param[in] request FAPI UL_TTI.request structure
 * @param[in] body_len Length of message body (from FAPI body_hdr.length or equivalent)
 * @param[in] num_antenna_ports Number of antenna ports in system
 * @param[in] numerology ORAN numerology parameters
 * @param[in] tx_windows Base transmission timing windows (for first symbol)
 * @param[in,out] prb_chunks Pre-allocated PRB chunks buffer (cleared before
 * use)
 * @param[out] msg_infos Output vector of C-plane message structures (cleared
 * before use)
 * @return Error code indicating success or specific error condition
 * @retval OranErrc::Success Operation completed successfully
 * @retval OranErrc::invalid_num_antenna_ports Number of antenna ports is zero
 * @retval OranErrc::InvalidPrbAllocationRbSizeZero PRB allocation has zero rb_size
 * @retval OranErrc::invalid_prb_allocation_num_symbols_zero PRB allocation has zero num_of_symbols
 * @retval OranErrc::invalid_prb_allocation_exceeds_slot Symbol allocation exceeds slot boundary
 * @retval OranErrc::too_many_sections_for_symbol Section limit exceeded for a start symbol
 * @retval OranErrc::PduPayloadOutOfBounds PDU parsing exceeded payload bounds
 */
ORAN_EXPORT [[nodiscard]] std::error_code convert_ul_tti_request_to_cplane(
        const scf_fapi_ul_tti_req_t &request,
        std::size_t body_len,
        std::uint16_t num_antenna_ports,
        const OranNumerology &numerology,
        const OranTxWindows &tx_windows,
        PrbChunks &prb_chunks,
        std::vector<OranCPlaneMsgInfo> &msg_infos);

} // namespace ran::oran

#endif // RAN_ORAN_FAPI_TO_CPLANE_HPP
