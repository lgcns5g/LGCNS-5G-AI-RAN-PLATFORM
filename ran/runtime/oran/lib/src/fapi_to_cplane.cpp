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
#include <cstring>
#include <span>
#include <system_error>
#include <vector>

#include <aerial-fh-driver/oran.hpp>
#include <quill/LogMacros.h>
#include <scf_5g_fapi.h>

#include "fapi/fapi_buffer.hpp"
#include "log/rt_log_macros.hpp"
#include "oran/cplane_types.hpp"
#include "oran/fapi_to_cplane.hpp"
#include "oran/numerology.hpp"
#include "oran/oran_errors.hpp"
#include "oran/oran_log.hpp"

namespace ran::oran {

namespace {

/// Maximum symbols per slot (normal CP)
constexpr std::uint8_t MAX_SYMBOLS_PER_SLOT = 14;

/// Maximum PRBs per C-plane section
constexpr std::uint16_t MAX_PRBS_PER_SECTION = 255;

/// Maximum PUSCH PDUs in UL_TTI.request
constexpr std::size_t MAX_PUSCH_PDUS = 255;

/**
 * Helper structure for sorting PRB allocations
 */
struct PrbAllocation final {
    std::uint16_t rb_start{};
    std::uint16_t rb_size{};
    std::uint8_t start_symbol_index{};
    std::uint8_t num_of_symbols{};

    /**
     * Comparison operator for sorting
     *
     * Sorts by:
     * 1. start_symbol_index (primary)
     * 2. rb_start (secondary)
     */
    [[nodiscard]] bool operator<(const PrbAllocation &other) const {
        if (start_symbol_index != other.start_symbol_index) {
            return start_symbol_index < other.start_symbol_index;
        }
        return rb_start < other.rb_start;
    }

    /**
     * Check if two allocations have matching start symbol
     */
    [[nodiscard]] bool has_same_start_symbol(const PrbAllocation &other) const {
        return start_symbol_index == other.start_symbol_index;
    }

    /**
     * Check if this allocation is contiguous with another in frequency and has
     * same symbol range
     */
    [[nodiscard]] bool is_contiguous_with(const PrbAllocation &other) const {
        return (start_symbol_index == other.start_symbol_index) &&
               (num_of_symbols == other.num_of_symbols) && ((rb_start + rb_size) == other.rb_start);
    }
};

/**
 * Validate PRB allocation parameters
 *
 * @param[in] rb_size Number of PRBs
 * @param[in] num_of_symbols Number of symbols
 * @param[in] start_symbol_index Start symbol index
 * @return Error code indicating success or specific validation error
 */
[[nodiscard]] std::error_code validate_prb_allocation(
        const std::uint16_t rb_size,
        const std::uint8_t num_of_symbols,
        const std::uint8_t start_symbol_index) {
    if (rb_size == 0) {
        RT_LOGC_ERROR(Oran::OranCplane, "Invalid PRB allocation: rb_size is zero");
        return OranErrc::InvalidPrbAllocationRbSizeZero;
    }

    if (num_of_symbols == 0) {
        RT_LOGC_ERROR(Oran::OranCplane, "Invalid PRB allocation: num_of_symbols is zero");
        return OranErrc::InvalidPrbAllocationNumSymbolsZero;
    }

    if (start_symbol_index + num_of_symbols > MAX_SYMBOLS_PER_SLOT) {
        RT_LOGC_ERROR(
                Oran::OranCplane,
                "Invalid PRB allocation: symbol allocation exceeds slot boundary "
                "(start={}, num={}, exceeds {})",
                start_symbol_index,
                num_of_symbols,
                MAX_SYMBOLS_PER_SLOT);
        return OranErrc::InvalidPrbAllocationExceedsSlot;
    }

    return OranErrc::Success;
}

/**
 * Count sections needed for a PRB size
 *
 * @param[in] rb_size Number of PRBs
 * @return Number of sections needed
 */
[[nodiscard]] std::uint16_t count_sections_for_prbs(const std::uint16_t rb_size) {
    return (rb_size + MAX_PRBS_PER_SECTION - 1) / MAX_PRBS_PER_SECTION;
}

/**
 * Parse PDU header from payload
 *
 * @param[in] payload_ptr Pointer to payload data
 * @param[in] offset Current offset in payload
 * @param[out] pdu_type PDU type read from payload
 * @param[out] pdu_size PDU size read from payload
 * @return New offset after reading header
 */
[[nodiscard]] std::size_t parse_pdu_header(
        const std::uint8_t *payload_ptr,
        const std::size_t offset,
        scf_fapi_ul_tti_pdu_type_t &pdu_type,
        std::uint16_t &pdu_size) {
    std::size_t current_offset = offset;

    // Create span from payload to avoid pointer arithmetic
    static constexpr auto HEADER_SIZE = sizeof(std::uint16_t) * 2;
    auto payload_span = fapi::make_const_buffer_span(payload_ptr, offset + HEADER_SIZE);
    auto header_span = payload_span.subspan(current_offset);

    // Read PDU type (using memcpy for unaligned access)
    std::uint16_t pdu_type_raw{};
    std::memcpy(&pdu_type_raw, header_span.data(), sizeof(std::uint16_t));
    pdu_type = static_cast<scf_fapi_ul_tti_pdu_type_t>(pdu_type_raw);
    current_offset += sizeof(std::uint16_t);
    header_span = header_span.subspan(sizeof(std::uint16_t));

    // Read PDU size (using memcpy for unaligned access)
    std::memcpy(&pdu_size, header_span.data(), sizeof(std::uint16_t));
    current_offset += sizeof(std::uint16_t);

    return current_offset;
}

/**
 * Extract PUSCH PDU allocation
 *
 * @param[in] payload_ptr Pointer to payload data
 * @param[in] offset Current offset in payload
 * @param[out] alloc PRB allocation to fill
 * @return Error code indicating success or validation error
 */
[[nodiscard]] std::error_code extract_pusch_allocation(
        const std::uint8_t *payload_ptr, const std::size_t offset, PrbAllocation &alloc) {
    // Use span to access offset without pointer arithmetic
    auto payload_span =
            fapi::make_const_buffer_span(payload_ptr, offset + sizeof(scf_fapi_pusch_pdu_t));
    const auto *pusch_pdu =
            fapi::assume_cast<const scf_fapi_pusch_pdu_t>(payload_span.subspan(offset).data());

    // Validate PRB allocation
    const std::error_code ec = validate_prb_allocation(
            pusch_pdu->rb_size, pusch_pdu->num_of_symbols, pusch_pdu->start_symbol_index);
    if (ec != OranErrc::Success) {
        return ec;
    }

    // Extract allocation info
    alloc.rb_start = pusch_pdu->rb_start;
    alloc.rb_size = pusch_pdu->rb_size;
    alloc.start_symbol_index = pusch_pdu->start_symbol_index;
    alloc.num_of_symbols = pusch_pdu->num_of_symbols;

    return OranErrc::Success;
}

/**
 * Validate PRB chunk
 *
 * @param[in] chunk PRB chunk to validate
 * @return Error code indicating success or validation error
 */
[[nodiscard]] std::error_code validate_chunk(const PrbChunk &chunk) {
    return validate_prb_allocation(chunk.rb_size, chunk.num_of_symbols, chunk.start_symbol_index);
}

/**
 * Merge contiguous allocations into PRB chunks
 *
 * @param[in] allocations Sorted array of PRB allocations
 * @param[in] num_allocations Number of allocations
 * @param[out] output PRB chunks output
 */
void merge_allocations_into_chunks(
        const std::array<PrbAllocation, MAX_PUSCH_PDUS> &allocations,
        const std::size_t num_allocations,
        PrbChunks &output) {
    if (num_allocations == 0) {
        return;
    }

    PrbChunk current_chunk{};
    current_chunk.rb_start = allocations.at(0).rb_start;
    current_chunk.rb_size = allocations.at(0).rb_size;
    current_chunk.start_symbol_index = allocations.at(0).start_symbol_index;
    current_chunk.num_of_symbols = allocations.at(0).num_of_symbols;

    for (std::size_t i = 1; i < num_allocations; ++i) {
        const auto &alloc = allocations.at(i);

        if (current_chunk.start_symbol_index == alloc.start_symbol_index &&
            current_chunk.num_of_symbols == alloc.num_of_symbols &&
            (current_chunk.rb_start + current_chunk.rb_size) == alloc.rb_start) {
            current_chunk.rb_size += alloc.rb_size;
        } else {
            output.chunks.push_back(current_chunk);
            current_chunk.rb_start = alloc.rb_start;
            current_chunk.rb_size = alloc.rb_size;
            current_chunk.start_symbol_index = alloc.start_symbol_index;
            current_chunk.num_of_symbols = alloc.num_of_symbols;
        }
    }

    output.chunks.push_back(current_chunk);
}

/**
 * Setup radio application header for C-plane message
 *
 * @param[out] radio_hdr Radio header to configure
 * @param[in] slot_timing Slot timing information
 * @param[in] start_symbol_id Start symbol for this message
 */
void setup_radio_header(
        oran_cmsg_radio_app_hdr &radio_hdr,
        const OranSlotTiming &slot_timing,
        const std::uint8_t start_symbol_id) {
    // NOLINTBEGIN(cppcoreguidelines-pro-type-union-access)
    // Union access required for C API compatibility with aerial-fh-driver SDK
    radio_hdr.payloadVersion = ORAN_DEF_PAYLOAD_VERSION;
    radio_hdr.filterIndex = ORAN_DEF_FILTER_INDEX;
    radio_hdr.frameId = slot_timing.frame_id;
    radio_hdr.subframeId = slot_timing.subframe_id;
    radio_hdr.slotId = slot_timing.slot_id;
    radio_hdr.startSymbolId = start_symbol_id;
    radio_hdr.sectionType = ORAN_CMSG_SECTION_TYPE_1;
    radio_hdr.dataDirection = DIRECTION_UPLINK;
    // NOLINTEND(cppcoreguidelines-pro-type-union-access)
}

/**
 * Process chunks into sections for a message
 *
 * @param[in,out] msg_info Message info to populate with sections
 * @param[in] chunks Chunk container
 * @param[in] chunk_start Start index in chunks
 * @param[in] chunk_end End index in chunks
 * @return Number of sections created
 */
[[nodiscard]] std::uint8_t process_chunks_into_sections(
        OranCPlaneMsgInfo &msg_info,
        const PrbChunks &chunks,
        const std::size_t chunk_start,
        const std::size_t chunk_end) {
    // Default RE mask for PUSCH (all 12 REs)
    static constexpr std::uint16_t PUSCH_DEFAULT_RE_MASK = 0x0FFF;

    std::uint8_t section_idx{};

    for (std::size_t chunk_idx = chunk_start; chunk_idx < chunk_end; ++chunk_idx) {
        const auto &chunk = chunks.chunks.at(chunk_idx);

        std::uint16_t remaining_prbs = chunk.rb_size;
        std::uint16_t current_prb = chunk.rb_start;

        while (remaining_prbs > 0) {
            const std::uint16_t prbs_in_section =
                    (remaining_prbs > MAX_PRBS_PER_SECTION) ? MAX_PRBS_PER_SECTION : remaining_prbs;

            // NOLINTBEGIN(cppcoreguidelines-pro-type-union-access)
            // Union access and array indexing required for C API compatibility with
            // aerial-fh-driver SDK
            auto &section = msg_info.sections.at(section_idx).sect_1;
            section.sectionId = section_idx;
            section.rb = 0;     // All PRBs indicator (0 = use startPrbc/numPrbc)
            section.symInc = 0; // Symbol increment (0 for non-interleaved)
            section.startPrbc = current_prb;
            section.numPrbc = static_cast<std::uint8_t>(prbs_in_section);
            section.reMask = PUSCH_DEFAULT_RE_MASK;
            section.numSymbol = chunk.num_of_symbols;
            section.ef = 0;     // No extensions
            section.beamId = 0; // Non-beamformed case
            // NOLINTEND(cppcoreguidelines-pro-type-union-access)

            current_prb += prbs_in_section;
            remaining_prbs -= prbs_in_section;
            ++section_idx;
        }
    }

    return section_idx;
}

} // namespace

std::error_code find_contiguous_prb_chunks(
        const scf_fapi_ul_tti_req_t &request, const std::size_t body_len, PrbChunks &output) {
    output.clear();

    if (request.num_ulsch == 0) {
        return OranErrc::Success;
    }

    static constexpr std::size_t REQUEST_HEADER_SIZE =
            sizeof(scf_fapi_ul_tti_req_t) - offsetof(scf_fapi_ul_tti_req_t, payload);
    const std::size_t payload_size = body_len - REQUEST_HEADER_SIZE;

    std::array<PrbAllocation, MAX_PUSCH_PDUS> allocations{};
    std::size_t num_allocations{};

    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)
    const auto *payload_ptr = request.payload;

    // Parse PDUs and extract PUSCH allocations
    static constexpr std::size_t PDU_HEADER_SIZE = sizeof(std::uint16_t) * 2;
    std::size_t offset{};

    for (std::uint8_t pdu_idx = 0; pdu_idx < request.num_pdus; ++pdu_idx) {
        if (offset + PDU_HEADER_SIZE > payload_size) {
            RT_LOGC_ERROR(
                    Oran::OranCplane,
                    "PDU header at offset {} exceeds payload bounds (size: {})",
                    offset,
                    payload_size);
            return OranErrc::PduPayloadOutOfBounds;
        }

        scf_fapi_ul_tti_pdu_type_t pdu_type{};
        std::uint16_t pdu_size{};
        offset = parse_pdu_header(payload_ptr, offset, pdu_type, pdu_size);

        if (offset + pdu_size > payload_size) {
            RT_LOGC_ERROR(
                    Oran::OranCplane,
                    "PDU at offset {} with size {} exceeds payload bounds (size: {})",
                    offset,
                    pdu_size,
                    payload_size);
            return OranErrc::PduPayloadOutOfBounds;
        }

        if (pdu_type == UL_TTI_PDU_TYPE_PUSCH) {
            if (num_allocations >= MAX_PUSCH_PDUS) {
                RT_LOGC_ERROR(
                        Oran::OranCplane,
                        "Too many PDUs: {} exceeds maximum {}",
                        num_allocations + 1,
                        MAX_PUSCH_PDUS);
                return OranErrc::TooManyPdus;
            }
            const std::error_code ec =
                    extract_pusch_allocation(payload_ptr, offset, allocations.at(num_allocations));
            if (ec != OranErrc::Success) {
                return ec;
            }
            ++num_allocations;
        }

        offset += pdu_size;
    }

    if (num_allocations == 0) {
        return OranErrc::Success;
    }

    std::sort(allocations.begin(), allocations.begin() + num_allocations);
    merge_allocations_into_chunks(allocations, num_allocations, output);

    // Validate section limits per start symbol
    std::size_t chunk_idx{};
    while (chunk_idx < output.chunks.size()) {
        const std::uint8_t current_start_symbol = output.chunks.at(chunk_idx).start_symbol_index;

        std::uint16_t total_sections{};
        while (chunk_idx < output.chunks.size() &&
               output.chunks.at(chunk_idx).start_symbol_index == current_start_symbol) {
            total_sections += count_sections_for_prbs(output.chunks.at(chunk_idx).rb_size);
            ++chunk_idx;
        }

        if (total_sections > MAX_CPLANE_SECTIONS) {
            RT_LOGC_ERROR(
                    Oran::OranCplane,
                    "Too many sections for start symbol {}: need {} sections "
                    "but maximum supported is {}",
                    current_start_symbol,
                    total_sections,
                    MAX_CPLANE_SECTIONS);
            return OranErrc::TooManySectionsForSymbol;
        }
    }

    return OranErrc::Success;
}

std::error_code convert_prb_chunks_to_cplane(
        const PrbChunks &chunks,
        const OranSlotTiming &slot_timing,
        const std::uint16_t num_antenna_ports,
        const OranTxWindows &tx_windows,
        const OranNumerology &numerology,
        std::vector<OranCPlaneMsgInfo> &msg_infos) {

    // Input validation
    if (num_antenna_ports == 0) {
        RT_LOGC_ERROR(Oran::OranCplane, "Number of antenna ports cannot be zero");
        return OranErrc::InvalidNumAntennaPorts;
    }

    // Early exit if no chunks
    if (chunks.chunks.empty()) {
        return OranErrc::Success;
    }

    // Process chunks grouped by start_symbol_index
    std::size_t chunk_group_start{};
    while (chunk_group_start < chunks.chunks.size()) {
        const std::uint8_t current_start_symbol =
                chunks.chunks.at(chunk_group_start).start_symbol_index;

        // Find all chunks with the same start_symbol_index and validate them
        std::size_t chunk_group_end = chunk_group_start;
        while (chunk_group_end < chunks.chunks.size() &&
               chunks.chunks.at(chunk_group_end).start_symbol_index == current_start_symbol) {
            const std::error_code ec = validate_chunk(chunks.chunks.at(chunk_group_end));
            if (ec != OranErrc::Success) {
                return ec;
            }
            ++chunk_group_end;
        }

        // Create message for antenna port 0 first
        OranCPlaneMsgInfo ap0_msg_info{};

        // Union access required for C API compatibility with aerial-fh-driver SDK

        // Set up radio application header for Section Type 1
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-union-access)
        auto &radio_hdr = ap0_msg_info.section_common_hdr.sect_1_common_hdr.radioAppHdr;
        setup_radio_header(radio_hdr, slot_timing, current_start_symbol);

        // Set message-level properties
        ap0_msg_info.data_direction = DIRECTION_UPLINK;
        ap0_msg_info.has_section_ext = false;
        ap0_msg_info.ap_idx = 0;

        // Calculate timing windows for the start symbol
        const std::uint64_t symbol_time_offset =
                current_start_symbol * numerology.symbol_duration_ns;
        ap0_msg_info.tx_window_start = tx_windows.tx_window_start + symbol_time_offset;
        ap0_msg_info.tx_window_bfw_start = tx_windows.tx_window_bfw_start; // BFW is slot-relative
        ap0_msg_info.tx_window_end = tx_windows.tx_window_end + symbol_time_offset;

        // Fill section information for all chunks in this group
        const std::uint8_t num_sections = process_chunks_into_sections(
                ap0_msg_info, chunks, chunk_group_start, chunk_group_end);

        // Set number of sections now that we've created them all
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-union-access)
        radio_hdr.numberOfSections = num_sections;

        ap0_msg_info.num_sections = num_sections;

        // Add message for antenna port 0
        msg_infos.push_back(ap0_msg_info);

        // Duplicate for remaining antenna ports
        for (std::uint16_t ap_idx = 1; ap_idx < num_antenna_ports; ++ap_idx) {
            OranCPlaneMsgInfo ap_msg_info = ap0_msg_info;
            ap_msg_info.ap_idx = ap_idx;
            msg_infos.push_back(ap_msg_info);
        }

        // Move to next group
        chunk_group_start = chunk_group_end;
    }

    return OranErrc::Success;
}

std::error_code convert_ul_tti_request_to_cplane(
        const scf_fapi_ul_tti_req_t &request,
        const std::size_t body_len,
        const std::uint16_t num_antenna_ports,
        const OranNumerology &numerology,
        const OranTxWindows &tx_windows,
        std::vector<OranCPlaneMsgInfo> &msg_infos) {
    PrbChunks prb_chunks{};
    return convert_ul_tti_request_to_cplane(
            request, body_len, num_antenna_ports, numerology, tx_windows, prb_chunks, msg_infos);
}

std::error_code convert_ul_tti_request_to_cplane(
        const scf_fapi_ul_tti_req_t &request,
        const std::size_t body_len,
        const std::uint16_t num_antenna_ports,
        const OranNumerology &numerology,
        const OranTxWindows &tx_windows,
        PrbChunks &prb_chunks,
        std::vector<OranCPlaneMsgInfo> &msg_infos) {
    // Frame ID mask (8-bit field)
    static constexpr std::uint32_t FRAME_ID_MASK = 0xFFU;

    // Clear output vector
    msg_infos.clear();

    // Extract timing information from request
    // FAPI slot field is the slot within the frame
    // Convert to ORAN subframe (0-9) and slot within subframe
    OranSlotTiming slot_timing{};
    slot_timing.frame_id = static_cast<std::uint8_t>(request.sfn & FRAME_ID_MASK);
    slot_timing.subframe_id =
            static_cast<std::uint8_t>(request.slot / numerology.slots_per_subframe);
    slot_timing.slot_id = static_cast<std::uint8_t>(request.slot % numerology.slots_per_subframe);

    // Find contiguous PRB chunks from all PUSCH PDUs in the request
    const std::error_code chunk_ec = find_contiguous_prb_chunks(request, body_len, prb_chunks);
    if (chunk_ec != OranErrc::Success) {
        return chunk_ec;
    }

    // Convert PRB chunks to C-plane messages
    if (!prb_chunks.chunks.empty()) {
        const std::error_code convert_ec = convert_prb_chunks_to_cplane(
                prb_chunks, slot_timing, num_antenna_ports, tx_windows, numerology, msg_infos);
        if (convert_ec != OranErrc::Success) {
            return convert_ec;
        }
    }

    return OranErrc::Success;
}

} // namespace ran::oran
