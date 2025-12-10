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
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <optional>
#include <span>
#include <stdexcept>

#include <aerial-fh-driver/oran.hpp>
#include <quill/LogMacros.h>

#include "fapi/fapi_buffer.hpp"
#include "log/rt_log_macros.hpp"
#include "oran/cplane_message.hpp"
#include "oran/cplane_types.hpp"
#include "oran/cplane_utils.hpp"
#include "oran/dpdk_buf.hpp" // IWYU pragma: keep
#include "oran/oran_buf.hpp"
#include "oran/oran_log.hpp"

namespace ran::oran {

namespace {

// NOLINTBEGIN(cppcoreguidelines-pro-type-union-access)

struct SectionMetadata final {
    std::uint8_t section_type{};
    std::uint8_t number_of_sections{};
    std::uint16_t common_hdr_size{};
    std::uint16_t section_size{};
    std::uint16_t pkt_section_info_room{};
};

/**
 * Packet building state for sections with extensions
 */
struct PacketBuildState final {
    OranBuf *current_buffer{};
    std::uint8_t *common_hdr_ptr{};
    std::uint8_t *section_ptr{};
    std::uint8_t *current_ptr{};
    oran_cmsg_sect_ext_type_11 *current_ext11_ptr{};
    std::uint16_t packet_num{};
    std::uint16_t total_section_info_size{};
    std::uint16_t pkt_remaining_capacity{};
    std::uint16_t current_ext4_len{};
    std::uint16_t current_ext5_len{};
    std::uint16_t current_ext11_len{};
    std::uint8_t sections_generated{};
};

/**
 * Holds calculated sizes for Extension 11 processing
 */
struct Ext11SizeInfo final {
    std::size_t ext11_hdr_size{};
    std::uint16_t bundle_hdr_size{};
    std::uint16_t bfw_iq_size{};
    std::uint16_t bundle_size{};
    std::size_t total_ext_size{};
    bool disable_bfws{};
};

/**
 * Validates section type and extracts basic metadata
 *
 * @param[in] info C-Plane message info
 * @param[in] mtu Maximum transmission unit
 * @return Section metadata
 */
[[nodiscard]] SectionMetadata
extract_section_metadata(const OranCPlaneMsgInfo &info, const std::uint16_t mtu) {
    const auto &radio_app_hdr = info.section_common_hdr.sect_1_common_hdr.radioAppHdr;
    const std::uint8_t section_type = radio_app_hdr.sectionType;
    const std::uint8_t number_of_sections = radio_app_hdr.numberOfSections;

    // These functions validate section_type and throw if invalid/unsupported
    const std::uint16_t common_hdr_size = get_cmsg_common_hdr_size(section_type);
    const std::uint16_t section_size = get_cmsg_section_size(section_type);
    const std::uint16_t pkt_section_info_room = mtu - ORAN_CMSG_HDR_OFFSET - common_hdr_size;

    return {section_type, number_of_sections, common_hdr_size, section_size, pkt_section_info_room};
}

/**
 * Counts packets for messages without section extensions
 *
 * @param[in] metadata Section metadata
 * @return Number of packets needed
 */
[[nodiscard]] std::size_t count_packets_without_extensions(const SectionMetadata &metadata) {
    const auto total_section_info_size = metadata.section_size * metadata.number_of_sections;
    const auto num_packets = std::max(
            1UL,
            (static_cast<std::size_t>(total_section_info_size) +
             static_cast<std::size_t>(metadata.pkt_section_info_room) - 1) /
                    static_cast<std::size_t>(metadata.pkt_section_info_room));
    return num_packets;
}

/**
 * Validates extension configuration for a section
 *
 * @param[in] section_info Section information
 * @param[in] section_type Section type
 * @param[in] section_num Section number
 */
void validate_section_extensions(
        const OranCPlaneSectionInfo &section_info,
        const std::uint8_t section_type,
        const std::uint8_t section_num) {
    if (section_type != ORAN_CMSG_SECTION_TYPE_1) {
        if (section_info.ext4.has_value() || section_info.ext5.has_value() ||
            section_info.ext11.has_value()) {
            RT_LOGC_ERROR(
                    Oran::OranCplane,
                    "Section {} extensions are only supported for section type 1, "
                    "but extensions were requested for section type {}",
                    section_num,
                    section_type);
            throw std::invalid_argument("Section extensions not supported for this section type");
        }
    }

    int extension_count = 0;
    if (section_info.ext4.has_value()) {
        ++extension_count;
    }
    if (section_info.ext5.has_value()) {
        ++extension_count;
    }
    if (section_info.ext11.has_value()) {
        ++extension_count;
    }

    if (extension_count > 1) {
        RT_LOGC_ERROR(
                Oran::OranCplane,
                "Section {}: Multiple extensions in a single section are not yet supported",
                section_num);
        throw std::runtime_error("Multiple extensions in a single section are not yet supported");
    }
}

/**
 * Counts packets needed for extension type 11
 *
 * @param[in] section_info Section information
 * @param[in] metadata Section metadata
 * @param[in] mtu Maximum transmission unit
 * @param[in,out] total_section_info_size Current packet usage
 * @param[in,out] section_num_packets Packet counter
 */
void count_ext11_packets(
        OranCPlaneSectionInfo &section_info,
        const SectionMetadata &metadata,
        const std::uint16_t mtu,
        std::uint16_t &total_section_info_size,
        std::size_t &section_num_packets) {
    if (!section_info.ext11.has_value()) {
        return;
    }
    auto &ext11_info = section_info.ext11.value();
    const auto ext4_hdr_size = sizeof(oran_cmsg_ext_hdr) + sizeof(oran_cmsg_sect_ext_type_4);
    const auto ext5_hdr_size = sizeof(oran_cmsg_ext_hdr) + sizeof(oran_cmsg_sect_ext_type_5);

    const auto disable_bfws = oran_cmsg_get_ext_11_disableBFWs(&ext11_info.ext_11.ext_hdr);

    auto ext11_hdr_size = sizeof(oran_cmsg_ext_hdr) + sizeof(oran_cmsg_sect_ext_type_11);
    ext11_hdr_size +=
            disable_bfws ? 0 : sizeof(oran_cmsg_sect_ext_type_11_disableBFWs_0_bfwCompHdr);

    const auto bundle_hdr_size = ext11_info.ext_11.bundle_hdr_size;
    const auto bfw_iq_size = ext11_info.ext_11.bfw_iq_size;
    const auto bundle_size = bundle_hdr_size + (disable_bfws ? 0 : bfw_iq_size);
    const auto total_ext_size = (section_info.ext4.has_value() ? ext4_hdr_size : 0) +
                                (section_info.ext5.has_value() ? ext5_hdr_size : 0) +
                                ext11_hdr_size + static_cast<std::size_t>(bundle_size);

    RT_LOGC_DEBUG(
            Oran::OranCplane,
            "COUNT EXT11 SIZE CALC: MTU={}, pkt_section_info_room={}, "
            "section_size={}, "
            "ext11_hdr_size={}, bundle_size={}, total_ext_size={}, "
            "total_needed={}, disableBFWs={}",
            mtu,
            metadata.pkt_section_info_room,
            metadata.section_size,
            ext11_hdr_size,
            bundle_size,
            total_ext_size,
            metadata.section_size + total_ext_size,
            disable_bfws);

    if (metadata.pkt_section_info_room < metadata.section_size + total_ext_size) {
        throw std::invalid_argument(std::format(
                "MTU {} is too small to hold a section with a "
                "single extType 11 bundle, please increase it!",
                mtu));
    }

    if (total_section_info_size + total_ext_size > metadata.pkt_section_info_room) {
        ++section_num_packets;
        total_section_info_size = metadata.section_size;
    }

    total_section_info_size += (section_info.ext4.has_value() ? ext4_hdr_size : 0) +
                               (section_info.ext5.has_value() ? ext5_hdr_size : 0);
    total_section_info_size += ext11_hdr_size;

    auto fragmented_ext_len = ext11_hdr_size;
    const auto &num_bundles = ext11_info.ext_11.num_prb_bundles;

    for (int bundle_idx = 0; bundle_idx < num_bundles; ++bundle_idx) {
        const auto padding = oran_cmsg_se11_disableBFWs_0_padding_bytes(
                static_cast<std::uint32_t>(fragmented_ext_len));

        if (total_section_info_size + static_cast<std::uint16_t>(bundle_size) + padding >
            metadata.pkt_section_info_room) {
            ++section_num_packets;
            total_section_info_size = static_cast<std::uint16_t>(
                    metadata.section_size + (section_info.ext4.has_value() ? ext4_hdr_size : 0) +
                    (section_info.ext5.has_value() ? ext5_hdr_size : 0) + ext11_hdr_size);
            fragmented_ext_len = ext11_hdr_size;
        }

        total_section_info_size += static_cast<std::uint16_t>(bundle_size);
        fragmented_ext_len += static_cast<std::size_t>(bundle_size);
    }
}

/**
 * Counts packets for messages with section extensions
 *
 * @param[in] info C-Plane message info
 * @param[in] metadata Section metadata
 * @param[in] mtu Maximum transmission unit
 * @return Number of packets needed
 */
[[nodiscard]] std::size_t count_packets_with_extensions(
        OranCPlaneMsgInfo &info, const SectionMetadata &metadata, const std::uint16_t mtu) {
    if (metadata.pkt_section_info_room < metadata.section_size) {
        throw std::invalid_argument(std::format(
                "MTU {} is too small for {} section header size {}",
                mtu,
                metadata.section_type,
                metadata.section_size));
    }

    std::size_t section_num_packets = 1;
    std::uint16_t total_section_info_size{};
    const auto ext4_hdr_size = sizeof(oran_cmsg_ext_hdr) + sizeof(oran_cmsg_sect_ext_type_4);
    const auto ext5_hdr_size = sizeof(oran_cmsg_ext_hdr) + sizeof(oran_cmsg_sect_ext_type_5);

    for (std::uint8_t section_num = 0; section_num < metadata.number_of_sections; ++section_num) {
        auto &section_info = info.sections.at(section_num);

        validate_section_extensions(section_info, metadata.section_type, section_num);
        if (metadata.section_type != ORAN_CMSG_SECTION_TYPE_1) {
            continue;
        }

        total_section_info_size += metadata.section_size;

        if (total_section_info_size > metadata.pkt_section_info_room) {
            ++section_num_packets;
            total_section_info_size = metadata.section_size;
        }

        if (!oran_cmsg_get_section_1_ef(&section_info.sect_1)) {
            continue;
        }

        if (section_info.ext11.has_value()) {
            count_ext11_packets(
                    section_info, metadata, mtu, total_section_info_size, section_num_packets);
        } else if (section_info.ext4.has_value()) {
            if (total_section_info_size + ext4_hdr_size > metadata.pkt_section_info_room) {
                ++section_num_packets;
                total_section_info_size = metadata.section_size;
            }
            total_section_info_size += ext4_hdr_size;
        } else if (section_info.ext5.has_value()) {
            if (total_section_info_size + ext5_hdr_size > metadata.pkt_section_info_room) {
                ++section_num_packets;
                total_section_info_size = metadata.section_size;
            }
            total_section_info_size += ext5_hdr_size;
        }
    }

    return std::max(1UL, section_num_packets);
}

/**
 * Finalizes current packet by setting size, sequence ID, and metadata
 *
 * @param[in,out] buffer Current buffer to finalize
 * @param[in,out] flow Flow for sequence ID management
 * @param[in] common_hdr_ptr Pointer to common header
 * @param[in] data_len Total data length for this packet
 * @param[in] sections_in_packet Number of sections in this packet
 * @param[in] is_uplink Direction flag
 */
void finalize_packet(
        OranBuf &buffer,
        OranFlow &flow,
        std::uint8_t *common_hdr_ptr,
        const std::uint16_t data_len,
        const std::uint8_t sections_in_packet,
        const bool is_uplink) {
    buffer.set_size(data_len);

    auto *data = buffer.data_as<PacketHeaderTemplate>();
    data->ecpri.ecpriSeqid =
            is_uplink ? flow.next_sequence_id_uplink() : flow.next_sequence_id_downlink();
    data->ecpri.ecpriPayload = cpu_to_be_16(data_len - sizeof(PacketHeaderTemplate) + 4);

    // Use span to access numberOfSections field without pointer arithmetic
    static constexpr auto SECTIONS_OFFSET = offsetof(oran_cmsg_radio_app_hdr, numberOfSections);
    auto buffer_span =
            fapi::make_buffer_span(common_hdr_ptr, SECTIONS_OFFSET + sizeof(std::uint8_t));
    buffer_span[SECTIONS_OFFSET] = static_cast<std::byte>(sections_in_packet);
}

/**
 * Initializes a new packet buffer with header templates
 *
 * @param[in,out] buffer Buffer to initialize
 * @param[in] flow Flow for packet header template
 * @param[in] info Message info containing common header
 * @param[in] common_hdr_size Size of common header to copy
 * @return Pointer to the start of the common header in the buffer
 */
[[nodiscard]] std::uint8_t *initialize_new_packet(
        OranBuf &buffer,
        const OranFlow &flow,
        const OranCPlaneMsgInfo &info,
        const std::uint16_t common_hdr_size) {
    buffer.clear_flags();
    auto *data = buffer.data_as<PacketHeaderTemplate>();
    std::memcpy(data, &flow.get_packet_header_template(), sizeof(PacketHeaderTemplate));

    auto *common_hdr_ptr = buffer.data_at_offset<std::uint8_t>(sizeof(PacketHeaderTemplate));
    std::memcpy(common_hdr_ptr, &info.section_common_hdr.sect_1_common_hdr, common_hdr_size);

    return common_hdr_ptr;
}

/**
 * Copies extension type 4 data to buffer
 *
 * @param[in] ext_info Extension information
 * @param[in,out] buffer_span Span of remaining buffer (updated after copy)
 * @return Size of extension 4 data copied
 */
[[nodiscard]] std::uint16_t
copy_extension_4(const CPlaneSectionExtInfo &ext_info, std::span<std::byte> &buffer_span) {
    static constexpr auto EXT4_HDR_SIZE =
            sizeof(oran_cmsg_ext_hdr) + sizeof(oran_cmsg_sect_ext_type_4);

    const auto *ext4_hdr = &ext_info.sect_ext_common_hdr;
    std::memcpy(buffer_span.data(), ext4_hdr, sizeof(oran_cmsg_ext_hdr));
    buffer_span = buffer_span.subspan(sizeof(oran_cmsg_ext_hdr));

    const auto *ext_4_hdr = &ext_info.ext_4.ext_hdr;
    std::memcpy(buffer_span.data(), ext_4_hdr, sizeof(oran_cmsg_sect_ext_type_4));
    buffer_span = buffer_span.subspan(sizeof(oran_cmsg_sect_ext_type_4));

    return static_cast<std::uint16_t>(EXT4_HDR_SIZE);
}

/**
 * Copies extension type 5 data to buffer
 *
 * @param[in] ext_info Extension information
 * @param[in,out] buffer_span Span of remaining buffer (updated after copy)
 * @return Size of extension 5 data copied
 */
[[nodiscard]] std::uint16_t
copy_extension_5(const CPlaneSectionExtInfo &ext_info, std::span<std::byte> &buffer_span) {
    static constexpr auto EXT5_HDR_SIZE =
            sizeof(oran_cmsg_ext_hdr) + sizeof(oran_cmsg_sect_ext_type_5);

    const auto *ext5_hdr = &ext_info.sect_ext_common_hdr;
    std::memcpy(buffer_span.data(), ext5_hdr, sizeof(oran_cmsg_ext_hdr));
    buffer_span = buffer_span.subspan(sizeof(oran_cmsg_ext_hdr));

    const auto *ext_5_hdr = &ext_info.ext_5.ext_hdr;
    std::memcpy(buffer_span.data(), ext_5_hdr, sizeof(oran_cmsg_sect_ext_type_5));
    buffer_span = buffer_span.subspan(sizeof(oran_cmsg_sect_ext_type_5));

    return static_cast<std::uint16_t>(EXT5_HDR_SIZE);
}

/**
 * Validates extension type 11 bundle and BFW data
 *
 * @param[in] ext_info Extension information
 * @param[in] disable_bfws Whether BFWs are disabled
 * @param[in] section_idx Section index for error reporting
 */
void validate_ext11_data(
        const CPlaneSectionExtInfo &ext_info,
        const bool disable_bfws,
        const std::uint8_t section_idx) {
    if (ext_info.ext_11.num_prb_bundles == 0) {
        return;
    }

    if (ext_info.ext_11.bundles == nullptr) {
        RT_LOGC_ERROR(
                Oran::OranCplane,
                "Section {} extension type 11: bundles pointer is null but numPrbBundles={}",
                section_idx,
                ext_info.ext_11.num_prb_bundles);
        throw std::invalid_argument("Extension type 11 bundles data not initialized");
    }

    if (disable_bfws) {
        return;
    }

    if (ext_info.ext_11.bfw_iq == nullptr) {
        RT_LOGC_ERROR(
                Oran::OranCplane,
                "Section {} extension type 11: BFW IQ data pointer is null but BFWs are enabled",
                section_idx);
        throw std::invalid_argument("Extension type 11 BFW IQ data not initialized");
    }

    if (ext_info.ext_11.bfw_iq_size == 0) {
        RT_LOGC_ERROR(
                Oran::OranCplane,
                "Section {} extension type 11: BFW IQ size is zero but BFWs are enabled",
                section_idx);
        throw std::invalid_argument("Extension type 11 BFW IQ size not set");
    }

    // Create span to access bundles array without pointer arithmetic
    auto bundles_span = std::span{ext_info.ext_11.bundles, ext_info.ext_11.num_prb_bundles};
    for (std::uint16_t bundle_idx = 0; bundle_idx < ext_info.ext_11.num_prb_bundles; ++bundle_idx) {
        if (bundles_span[bundle_idx].bfw_iq == nullptr) {
            RT_LOGC_ERROR(
                    Oran::OranCplane,
                    "Section {} extension type 11 bundle {}: BFW IQ pointer is null",
                    section_idx,
                    bundle_idx);
            throw std::invalid_argument("Extension type 11 bundle BFW IQ data not initialized");
        }
    }
}

/**
 * Validates all extension data for sections with extensions
 *
 * @param[in] info C-Plane message info
 */
void validate_all_extensions(OranCPlaneMsgInfo &info) {
    if (!info.has_section_ext) {
        return;
    }

    for (std::uint8_t section_idx = 0; section_idx < info.num_sections; ++section_idx) {
        auto &section_info = info.sections.at(section_idx);

        if (section_info.ext11.has_value()) {
            auto &ext_info = section_info.ext11.value();
            const bool disable_bfws = oran_cmsg_get_ext_11_disableBFWs(&ext_info.ext_11.ext_hdr);

            validate_ext11_data(ext_info, disable_bfws, section_idx);
        }
    }
}

/**
 * Validates basic message parameters
 *
 * @tparam BufferSpanType Type of buffer span
 * @param[in] buffers Buffer span
 * @param[in] mtu Maximum transmission unit
 * @param[in] num_sections Number of sections
 * @param[in] pkt_section_info_room Available space for sections
 * @param[in] section_size Size of each section
 */
template <typename BufferSpanType>
void validate_message_parameters(
        BufferSpanType buffers,
        const std::uint16_t mtu,
        const std::uint8_t num_sections,
        const std::uint16_t pkt_section_info_room,
        const std::uint16_t section_size) {
    if (buffers.empty()) {
        RT_LOGC_ERROR(
                Oran::OranCplane,
                "prepare_cplane_message: No buffers provided - cannot create packets");
        throw std::invalid_argument("Invalid buffer array");
    }

    if (mtu == 0) {
        RT_LOGC_ERROR(Oran::OranCplane, "prepare_cplane_message: MTU cannot be zero");
        throw std::invalid_argument("MTU cannot be zero");
    }

    if (num_sections > 0 && pkt_section_info_room < section_size) {
        RT_LOGC_ERROR(
                Oran::OranCplane,
                "prepare_cplane_message: MTU too small - single section ({} bytes) doesn't fit in "
                "available space ({} bytes)",
                section_size,
                pkt_section_info_room);
        throw std::invalid_argument("MTU too small for even a single section");
    }
}

/**
 * Advances section pointer and resets extension lengths
 *
 * @param[in,out] state Packet building state
 * @param[in] metadata Section metadata
 */
void advance_section_ptr_and_reset_extensions(
        PacketBuildState &state, const SectionMetadata &metadata) {
    const std::size_t total_advance = static_cast<std::size_t>(metadata.section_size) +
                                      static_cast<std::size_t>(state.current_ext4_len) +
                                      static_cast<std::size_t>(state.current_ext5_len) +
                                      static_cast<std::size_t>(state.current_ext11_len);
    auto section_span = fapi::make_buffer_span(state.section_ptr, total_advance);
    state.section_ptr = fapi::assume_cast<std::uint8_t>(section_span.subspan(total_advance).data());
    state.current_ext4_len = 0;
    state.current_ext5_len = 0;
}

/**
 * Applies byte-order conversion to extension type 5 fields
 *
 * @param[in,out] ext5_info Extension 5 information
 */
void apply_ext5_byte_order_conversion(CPlaneSectionExtInfo &ext5_info) {
    auto *ext5_hdr_base = fapi::assume_cast<std::uint8_t>(&ext5_info.ext_5.ext_hdr);
    auto ext5_span = fapi::make_buffer_span(ext5_hdr_base, sizeof(std::uint64_t) + 1);
    auto *ext5_2sets_ptr = fapi::assume_cast<std::uint8_t>(ext5_span.subspan(1).data());
    std::uint64_t ext5_2sets{};
    std::memcpy(&ext5_2sets, ext5_2sets_ptr, sizeof(std::uint64_t));
    ext5_2sets = cpu_to_be_64(ext5_2sets);
    std::memcpy(ext5_2sets_ptr, &ext5_2sets, sizeof(std::uint64_t));
}

/**
 * Creates new packet when section doesn't fit in current packet
 *
 * @tparam BufferSpanType Type of buffer span
 * @param[in] info C-Plane message info
 * @param[in,out] flow Flow for sequence ID management
 * @param[in] buffers Buffer span
 * @param[in] section_num Current section number
 * @param[in] metadata Section metadata
 * @param[in] mtu Maximum transmission unit
 * @param[in] is_uplink Direction flag
 * @param[in,out] state Packet building state
 */
template <typename BufferSpanType>
void start_new_packet_for_section(
        const OranCPlaneMsgInfo &info,
        OranFlow &flow,
        BufferSpanType buffers,
        const std::uint8_t section_num,
        const SectionMetadata &metadata,
        const std::uint16_t mtu,
        const bool is_uplink,
        PacketBuildState &state) {
    const std::uint16_t data_len = mtu - state.pkt_remaining_capacity;
    finalize_packet(
            *state.current_buffer,
            flow,
            state.common_hdr_ptr,
            data_len,
            section_num - state.sections_generated,
            is_uplink);
    state.sections_generated = section_num;

    if (state.packet_num >= buffers.size()) {
        RT_LOGC_ERROR(
                Oran::OranCplane,
                "Insufficient buffers for fragmentation: need packet "
                "{} but only {} buffers provided",
                state.packet_num,
                buffers.size());
        throw std::runtime_error("Insufficient buffers for packet fragmentation");
    }

    state.current_buffer = &buffers[state.packet_num++];
    state.common_hdr_ptr =
            initialize_new_packet(*state.current_buffer, flow, info, metadata.common_hdr_size);

    auto state_hdr_span = fapi::make_buffer_span(
            state.common_hdr_ptr, metadata.common_hdr_size + metadata.section_size);
    state.section_ptr = fapi::assume_cast<std::uint8_t>(
            state_hdr_span.subspan(metadata.common_hdr_size).data());

    state.total_section_info_size = 0;
    state.pkt_remaining_capacity = metadata.pkt_section_info_room;
}

/**
 * Processes simple extension types (ext4 or ext5)
 *
 * @tparam BufferSpanType Type of buffer span
 * @param[in] info C-Plane message info
 * @param[in,out] flow Flow for sequence ID management
 * @param[in] buffers Buffer array
 * @param[in] section_num Current section number
 * @param[in] section_info Section information
 * @param[in] metadata Section metadata
 * @param[in] mtu Maximum transmission unit
 * @param[in] is_uplink Direction flag
 * @param[in] ext_info Extension information
 * @param[in] extension_size Size of extension header
 * @param[in] is_ext5 Whether this is extension type 5 (vs type 4)
 * @param[in,out] state Packet building state
 */
template <typename BufferSpanType>
void process_simple_extension(
        const OranCPlaneMsgInfo &info,
        OranFlow &flow,
        BufferSpanType buffers,
        const std::uint8_t section_num,
        const OranCPlaneSectionInfo &section_info,
        const SectionMetadata &metadata,
        const std::uint16_t mtu,
        const bool is_uplink,
        const CPlaneSectionExtInfo &ext_info,
        const std::size_t extension_size,
        const bool is_ext5,
        PacketBuildState &state) {
    if (state.total_section_info_size + extension_size > metadata.pkt_section_info_room) {
        const std::uint16_t data_len = mtu - state.pkt_remaining_capacity;
        finalize_packet(
                *state.current_buffer,
                flow,
                state.common_hdr_ptr,
                data_len,
                section_num + 1 - state.sections_generated,
                is_uplink);
        state.sections_generated = section_num;

        if (state.packet_num >= buffers.size()) {
            RT_LOGC_ERROR(
                    Oran::OranCplane,
                    "Insufficient buffers for fragmentation: need packet "
                    "{} but only {} buffers provided",
                    state.packet_num,
                    buffers.size());
            throw std::runtime_error("Insufficient buffers for packet fragmentation");
        }

        state.current_buffer = &buffers[state.packet_num++];
        state.common_hdr_ptr =
                initialize_new_packet(*state.current_buffer, flow, info, metadata.common_hdr_size);

        auto ext_hdr_span = fapi::make_buffer_span(
                state.common_hdr_ptr, metadata.common_hdr_size + metadata.section_size);
        state.section_ptr = fapi::assume_cast<std::uint8_t>(
                ext_hdr_span.subspan(metadata.common_hdr_size).data());

        state.total_section_info_size = 0;
        state.pkt_remaining_capacity = metadata.pkt_section_info_room;

        memcpy(state.section_ptr, &section_info, metadata.section_size);

        auto ext_section_span =
                fapi::make_buffer_span(state.section_ptr, state.pkt_remaining_capacity);
        state.current_ptr = fapi::assume_cast<std::uint8_t>(
                ext_section_span.subspan(metadata.section_size).data());
        state.total_section_info_size += metadata.section_size;
        state.pkt_remaining_capacity -= metadata.section_size;
        state.current_ext4_len = 0;
        state.current_ext5_len = 0;
    }

    auto ext_buffer_span = fapi::make_buffer_span(state.current_ptr, state.pkt_remaining_capacity);
    const auto copied_size = is_ext5 ? copy_extension_5(ext_info, ext_buffer_span)
                                     : copy_extension_4(ext_info, ext_buffer_span);

    state.current_ptr = fapi::assume_cast<std::uint8_t>(ext_buffer_span.data());
    state.total_section_info_size += copied_size;
    state.pkt_remaining_capacity -= copied_size;

    if (is_ext5) {
        state.current_ext5_len = copied_size;
    } else {
        state.current_ext4_len = copied_size;
    }
}

/**
 * Calculates sizes needed for Extension 11 processing
 *
 * @param[in] ext11_info Extension 11 information
 * @param[in] ext4_opt Optional extension 4 info
 * @param[in] ext5_opt Optional extension 5 info
 * @return Structure containing all calculated sizes
 */
[[nodiscard]] Ext11SizeInfo calculate_ext11_sizes(
        const CPlaneSectionExtInfo &ext11_info,
        const std::optional<CPlaneSectionExtInfo> &ext4_opt,
        const std::optional<CPlaneSectionExtInfo> &ext5_opt) {
    static constexpr auto EXT4_HDR_SIZE =
            sizeof(oran_cmsg_ext_hdr) + sizeof(oran_cmsg_sect_ext_type_4);
    static constexpr auto EXT5_HDR_SIZE =
            sizeof(oran_cmsg_ext_hdr) + sizeof(oran_cmsg_sect_ext_type_5);

    auto ext11_hdr_copy = ext11_info.ext_11.ext_hdr;
    const bool disable_bfws = oran_cmsg_get_ext_11_disableBFWs(&ext11_hdr_copy);

    std::size_t ext11_hdr_size = sizeof(oran_cmsg_ext_hdr) + sizeof(oran_cmsg_sect_ext_type_11);
    ext11_hdr_size +=
            disable_bfws ? 0 : sizeof(oran_cmsg_sect_ext_type_11_disableBFWs_0_bfwCompHdr);

    const auto bundle_hdr_size = ext11_info.ext_11.bundle_hdr_size;
    const auto bfw_iq_size = ext11_info.ext_11.bfw_iq_size;
    const auto bundle_size =
            static_cast<std::uint16_t>(bundle_hdr_size + (disable_bfws ? 0 : bfw_iq_size));

    const auto total_ext_size =
            (ext4_opt.has_value() ? EXT4_HDR_SIZE : 0) +
            (ext5_opt.has_value() ? EXT5_HDR_SIZE : 0) + ext11_hdr_size +
            static_cast<std::size_t>(bundle_size * ext11_info.ext_11.num_prb_bundles);

    return {ext11_hdr_size,
            bundle_hdr_size,
            bfw_iq_size,
            bundle_size,
            total_ext_size,
            disable_bfws};
}

/**
 * Validates that MTU is sufficient for Extension 11 processing
 *
 * @param[in] size_info Calculated size information
 * @param[in] metadata Section metadata
 * @param[in] mtu Maximum transmission unit
 */
void validate_ext11_mtu_sufficiency(
        const Ext11SizeInfo &size_info, const SectionMetadata &metadata, const std::uint16_t mtu) {
    RT_LOGC_DEBUG(
            Oran::OranCplane,
            "EXT11 SIZE CALC: MTU={}, pkt_section_info_room={}, "
            "section_size={}, "
            "ext11_hdr_size={}, bundle_size={}, "
            "total_ext_size={}, "
            "total_needed={}, disableBFWs={}",
            mtu,
            metadata.pkt_section_info_room,
            metadata.section_size,
            size_info.ext11_hdr_size,
            size_info.bundle_size,
            size_info.total_ext_size,
            metadata.section_size + size_info.total_ext_size,
            size_info.disable_bfws);

    if (metadata.pkt_section_info_room < metadata.section_size + size_info.total_ext_size) {
        RT_LOGC_ERROR(
                Oran::OranCplane,
                "MTU {} is too small to hold a section with a single "
                "extType 11 bundle, please increase it!",
                mtu);
        throw std::runtime_error(std::format(
                "MTU {} is too small to hold a section with a single "
                "extType 11 bundle, please increase it!",
                mtu));
    }
}

/**
 * Initializes Extension 11 headers in current packet buffer
 *
 * @param[in] ext11_info Extension 11 information
 * @param[in] size_info Calculated size information
 * @param[in,out] state Packet building state
 */
void initialize_ext11_headers_in_packet(
        const CPlaneSectionExtInfo &ext11_info,
        const Ext11SizeInfo &size_info,
        PacketBuildState &state) {
    state.total_section_info_size += size_info.ext11_hdr_size;
    state.pkt_remaining_capacity -= size_info.ext11_hdr_size;
    state.current_ext11_len = static_cast<std::uint16_t>(size_info.ext11_hdr_size);

    auto buffer_span = fapi::make_buffer_span(
            state.current_ptr, state.pkt_remaining_capacity + size_info.ext11_hdr_size);

    const auto *ext11_hdr = &ext11_info.sect_ext_common_hdr;
    std::memcpy(buffer_span.data(), ext11_hdr, sizeof(oran_cmsg_ext_hdr));
    buffer_span = buffer_span.subspan(sizeof(oran_cmsg_ext_hdr));

    const auto *ext_11_hdr = &ext11_info.ext_11.ext_hdr;
    std::memcpy(buffer_span.data(), ext_11_hdr, sizeof(oran_cmsg_sect_ext_type_11));
    state.current_ext11_ptr = fapi::assume_cast<oran_cmsg_sect_ext_type_11>(buffer_span.data());
    buffer_span = buffer_span.subspan(sizeof(oran_cmsg_sect_ext_type_11));

    if (!size_info.disable_bfws) {
        if (ext11_info.ext_11.bfw_iq == nullptr) {
            RT_LOGC_ERROR(Oran::OranCplane, "BFW IQ data pointer is null for extension type 11");
            throw std::invalid_argument("BFW IQ data not initialized");
        }
        if (ext11_info.ext_11.bundles == nullptr) {
            RT_LOGC_ERROR(Oran::OranCplane, "Bundle data pointer is null for extension type 11");
            throw std::invalid_argument("Bundle data not initialized");
        }

        const auto *ext_comp_ptr = &ext11_info.ext_11.ext_comp_hdr;
        std::memcpy(
                buffer_span.data(),
                ext_comp_ptr,
                sizeof(oran_cmsg_sect_ext_type_11_disableBFWs_0_bfwCompHdr));
        buffer_span =
                buffer_span.subspan(sizeof(oran_cmsg_sect_ext_type_11_disableBFWs_0_bfwCompHdr));
    }

    state.current_ptr = fapi::assume_cast<std::uint8_t>(buffer_span.data());
}

/**
 * Finalizes Extension 11 section with padding and updates section PRB info
 *
 * @param[in] ext11_info Extension 11 information
 * @param[in] bundles_included Number of bundles included in section
 * @param[in] curr_start_prbc Current starting PRB
 * @param[in] section_max_prbc Maximum PRB for section
 * @param[in,out] state Packet building state
 */
void finalize_ext11_section(
        const CPlaneSectionExtInfo &ext11_info,
        const int bundles_included,
        const std::uint16_t curr_start_prbc,
        const std::uint16_t section_max_prbc,
        PacketBuildState &state) {
    oran_cmsg_ext_hdr ext_hdr_copy = ext11_info.sect_ext_common_hdr;
    if (oran_cmsg_get_ext_ef(&ext_hdr_copy)) {
        RT_LOGC_ERROR(
                Oran::OranCplane,
                "Multiple section extensions in a single section "
                "is not yet supported!");
        throw std::runtime_error("Multiple section extensions in a "
                                 "single section is not yet supported!");
    }

    auto *sect_1_ptr = fapi::assume_cast<oran_cmsg_sect1>(state.section_ptr);
    const auto num_prbc = bundles_included * ext11_info.ext_11.ext_hdr.numBundPrb;
    const auto final_num_prbc = (curr_start_prbc + num_prbc > section_max_prbc)
                                        ? (section_max_prbc - curr_start_prbc)
                                        : num_prbc;
    sect_1_ptr->startPrbc = curr_start_prbc;
    sect_1_ptr->numPrbc = static_cast<std::uint16_t>(final_num_prbc);

    const auto padding = oran_cmsg_se11_disableBFWs_0_padding_bytes(state.current_ext11_len);
    memset(state.current_ptr, 0, padding);
    state.total_section_info_size += padding;
    state.pkt_remaining_capacity -= padding;
    state.current_ext11_len += padding;

    static constexpr auto MASK = 0xFFFFU;
    state.current_ext11_ptr->extLen = cpu_to_be_16(static_cast<std::uint16_t>(
            (static_cast<std::uint32_t>(state.current_ext11_len) >> 2U) & MASK));
}

/**
 * Copies a single ext11 bundle to the buffer
 *
 * @param[in] bundle_info Bundle information to copy
 * @param[in] disable_bfws Whether BFWs are disabled
 * @param[in] comp_meth Compression method
 * @param[in] bfw_iq_size Size of BFW IQ data
 * @param[in] bundle_idx Bundle index for error reporting
 * @param[in,out] buffer_span Span of remaining buffer (updated after copy)
 */
void copy_ext11_bundle(
        const CPlaneSectionExt11BundlesInfo &bundle_info,
        const bool disable_bfws,
        const UserDataBFWCompressionMethod comp_meth,
        const std::uint16_t bfw_iq_size,
        const int bundle_idx,
        std::span<std::byte> &buffer_span) {
    if (disable_bfws) {
        static constexpr auto BUNDLE_SIZE = sizeof(oran_cmsg_sect_ext_type_11_disableBFWs_1_bundle);
        auto *bundle_ptr = fapi::assume_cast<oran_cmsg_sect_ext_type_11_disableBFWs_1_bundle>(
                buffer_span.data());
        std::memcpy(bundle_ptr, &bundle_info.disable_bfws_1, BUNDLE_SIZE);
        bundle_ptr->beamId = bundle_info.disable_bfws_1.beamId.get();
        buffer_span = buffer_span.subspan(BUNDLE_SIZE);
        return;
    }

    if (comp_meth == UserDataBFWCompressionMethod::NO_COMPRESSION) {
        static constexpr auto BUNDLE_HDR_SIZE =
                sizeof(oran_cmsg_sect_ext_type_11_disableBFWs_0_bundle_uncompressed);
        auto *bundle_ptr =
                fapi::assume_cast<oran_cmsg_sect_ext_type_11_disableBFWs_0_bundle_uncompressed>(
                        buffer_span.data());
        std::memcpy(bundle_ptr, &bundle_info.disable_bfws_0_uncompressed, BUNDLE_HDR_SIZE);
        bundle_ptr->beamId = bundle_info.disable_bfws_0_uncompressed.beamId.get();

        if (bundle_info.bfw_iq == nullptr) {
            RT_LOGC_ERROR(Oran::OranCplane, "Bundle {} BFW IQ data is null", bundle_idx);
            throw std::invalid_argument("Bundle BFW IQ data not initialized");
        }

        // Copy BFW IQ data - bfw is a zero-length array, data goes after the header
        // Advance buffer_span past header, copy IQ data there
        auto iq_span = buffer_span.subspan(BUNDLE_HDR_SIZE);
        std::memcpy(iq_span.data(), bundle_info.bfw_iq, bfw_iq_size);
        buffer_span = buffer_span.subspan(BUNDLE_HDR_SIZE + bfw_iq_size);
    } else if (comp_meth == UserDataBFWCompressionMethod::BLOCK_FLOATING_POINT) {
        static constexpr auto BUNDLE_HDR_SIZE =
                sizeof(oran_cmsg_sect_ext_type_11_disableBFWs_0_bfp_compressed_bundle_hdr);
        auto *bundle_ptr = fapi::assume_cast<
                oran_cmsg_sect_ext_type_11_disableBFWs_0_bfp_compressed_bundle_hdr>(
                buffer_span.data());
        std::memcpy(bundle_ptr, &bundle_info.disable_bfws_0_compressed, BUNDLE_HDR_SIZE);
        bundle_ptr->beamId = bundle_info.disable_bfws_0_compressed.beamId.get();
        bundle_ptr->bfwCompParam.exponent =
                bundle_info.disable_bfws_0_compressed.bfwCompParam.exponent.get();

        if (bundle_info.bfw_iq == nullptr) {
            RT_LOGC_ERROR(Oran::OranCplane, "Bundle {} BFW IQ data is null", bundle_idx);
            throw std::invalid_argument("Bundle BFW IQ data not initialized");
        }

        // Copy BFW IQ data - bfw is a zero-length array, data goes after the header
        // Advance buffer_span past header, copy IQ data there
        auto iq_span = buffer_span.subspan(BUNDLE_HDR_SIZE);
        std::memcpy(iq_span.data(), bundle_info.bfw_iq, bfw_iq_size);
        buffer_span = buffer_span.subspan(BUNDLE_HDR_SIZE + bfw_iq_size);
    }
}

/**
 * Starts a new packet for ext11 continuation
 *
 * @tparam BufferSpanType Type of buffer span
 * @param[in] info C-Plane message info
 * @param[in] flow Flow for sequence ID management
 * @param[in] buffers Buffer span
 * @param[in] section_num Current section number
 * @param[in] metadata Section metadata with sizing info
 * @param[in] ext11_hdr_size Size of ext11 header
 * @param[in] ext4_opt Optional ext4 info
 * @param[in] ext5_opt Optional ext5 info
 * @param[in] ext11_info Extension 11 info
 * @param[in] disable_bfws Whether BFWs are disabled
 * @param[in,out] state Packet building state
 */
template <typename BufferSpanType>
void start_new_ext11_packet(
        const OranCPlaneMsgInfo &info,
        const OranFlow &flow,
        BufferSpanType buffers,
        const std::uint8_t section_num,
        const SectionMetadata &metadata,
        const std::size_t ext11_hdr_size,
        const std::optional<CPlaneSectionExtInfo> &ext4_opt,
        const std::optional<CPlaneSectionExtInfo> &ext5_opt,
        const CPlaneSectionExtInfo &ext11_info,
        const bool disable_bfws,
        PacketBuildState &state) {
    state.current_buffer = &buffers[state.packet_num++];
    state.common_hdr_ptr =
            initialize_new_packet(*state.current_buffer, flow, info, metadata.common_hdr_size);

    // Use span to calculate section_ptr without pointer arithmetic
    auto hdr_span = fapi::make_buffer_span(
            state.common_hdr_ptr, metadata.common_hdr_size + metadata.section_size);
    state.section_ptr =
            fapi::assume_cast<std::uint8_t>(hdr_span.subspan(metadata.common_hdr_size).data());

    std::memcpy(state.section_ptr, &info.sections.at(section_num), metadata.section_size);

    // Create span for remaining buffer starting after section
    const std::size_t remaining_capacity = metadata.pkt_section_info_room;
    auto buffer_span = fapi::make_buffer_span(state.section_ptr, remaining_capacity);
    buffer_span = buffer_span.subspan(metadata.section_size);

    if (ext4_opt.has_value()) {
        const auto &ext4_info = ext4_opt.value();
        state.current_ext4_len = copy_extension_4(ext4_info, buffer_span);
    } else if (ext5_opt.has_value()) {
        const auto &ext5_info = ext5_opt.value();
        state.current_ext5_len = copy_extension_5(ext5_info, buffer_span);
    }

    const auto *ext11_hdr = &ext11_info.sect_ext_common_hdr;
    std::memcpy(buffer_span.data(), ext11_hdr, sizeof(oran_cmsg_ext_hdr));
    buffer_span = buffer_span.subspan(sizeof(oran_cmsg_ext_hdr));

    const auto *ext_11_hdr = &ext11_info.ext_11.ext_hdr;
    std::memcpy(buffer_span.data(), ext_11_hdr, sizeof(oran_cmsg_sect_ext_type_11));
    state.current_ext11_ptr = fapi::assume_cast<oran_cmsg_sect_ext_type_11>(buffer_span.data());
    buffer_span = buffer_span.subspan(sizeof(oran_cmsg_sect_ext_type_11));

    if (!disable_bfws) {
        const auto *ext_comp_ptr = &ext11_info.ext_11.ext_comp_hdr;
        std::memcpy(
                buffer_span.data(),
                ext_comp_ptr,
                sizeof(oran_cmsg_sect_ext_type_11_disableBFWs_0_bfwCompHdr));
        buffer_span =
                buffer_span.subspan(sizeof(oran_cmsg_sect_ext_type_11_disableBFWs_0_bfwCompHdr));
    }

    state.current_ptr = fapi::assume_cast<std::uint8_t>(buffer_span.data());
    state.current_ext11_len = static_cast<std::uint16_t>(ext11_hdr_size);
}

/**
 * Process extension 11 handling
 *
 * @tparam BufferSpanType Type of buffer span
 * @param[in] info C-Plane message info
 * @param[in,out] flow Flow for sequence ID management
 * @param[in] buffers Buffer array
 * @param[in] section_num Current section number
 * @param[in] metadata Section metadata
 * @param[in] ext11_info Extension 11 information
 * @param[in] ext4_opt Optional extension 4 information
 * @param[in] ext5_opt Optional extension 5 information
 * @param[in] mtu Maximum transmission unit
 * @param[in] is_uplink Direction flag
 * @param[in,out] state Packet building state
 */
template <typename BufferSpanType>
void process_extension_11(
        OranCPlaneMsgInfo &info,
        OranFlow &flow,
        BufferSpanType buffers,
        const std::uint8_t section_num,
        const SectionMetadata &metadata,
        const std::optional<CPlaneSectionExtInfo> &ext11_opt,
        const std::optional<CPlaneSectionExtInfo> &ext4_opt,
        const std::optional<CPlaneSectionExtInfo> &ext5_opt,
        const std::uint16_t mtu,
        const bool is_uplink,
        PacketBuildState &state) {

    if (!ext11_opt.has_value()) {
        return;
    }
    const auto &ext11_info = ext11_opt.value();

    const auto size_info = calculate_ext11_sizes(ext11_info, ext4_opt, ext5_opt);
    validate_ext11_mtu_sufficiency(size_info, metadata, mtu);

    if (state.total_section_info_size + size_info.total_ext_size > metadata.pkt_section_info_room) {
        const std::uint16_t data_len = mtu - state.pkt_remaining_capacity;
        finalize_packet(
                *state.current_buffer,
                flow,
                state.common_hdr_ptr,
                data_len,
                section_num + 1 - state.sections_generated,
                is_uplink);
        state.sections_generated = section_num;

        if (state.packet_num >= buffers.size()) {
            RT_LOGC_ERROR(
                    Oran::OranCplane,
                    "Insufficient buffers for fragmentation: need packet "
                    "{} but only {} buffers provided",
                    state.packet_num,
                    buffers.size());
            throw std::runtime_error("Insufficient buffers for packet fragmentation");
        }

        state.current_buffer = &buffers[state.packet_num++];
        state.common_hdr_ptr =
                initialize_new_packet(*state.current_buffer, flow, info, metadata.common_hdr_size);

        auto hdr_span = fapi::make_buffer_span(
                state.common_hdr_ptr, metadata.common_hdr_size + metadata.section_size);
        state.section_ptr =
                fapi::assume_cast<std::uint8_t>(hdr_span.subspan(metadata.common_hdr_size).data());

        state.total_section_info_size = 0;
        state.pkt_remaining_capacity = metadata.pkt_section_info_room;

        memcpy(state.section_ptr, &info.sections.at(section_num), metadata.section_size);

        auto buffer_span = fapi::make_buffer_span(state.section_ptr, state.pkt_remaining_capacity);
        buffer_span = buffer_span.subspan(metadata.section_size);

        state.total_section_info_size += metadata.section_size;
        state.pkt_remaining_capacity -= metadata.section_size;
        state.current_ext4_len = 0;
        state.current_ext5_len = 0;
        state.current_ext11_len = 0;

        state.current_ptr = fapi::assume_cast<std::uint8_t>(buffer_span.data());
    }

    auto buffer_span = fapi::make_buffer_span(state.current_ptr, state.pkt_remaining_capacity);

    if (ext4_opt.has_value()) {
        const auto &ext4_info = ext4_opt.value();
        state.current_ext4_len = copy_extension_4(ext4_info, buffer_span);
        state.total_section_info_size += state.current_ext4_len;
        state.pkt_remaining_capacity -= state.current_ext4_len;
    } else if (ext5_opt.has_value()) {
        const auto &ext5_info = ext5_opt.value();
        state.current_ext5_len = copy_extension_5(ext5_info, buffer_span);
        state.total_section_info_size += state.current_ext5_len;
        state.pkt_remaining_capacity -= state.current_ext5_len;
    }

    initialize_ext11_headers_in_packet(ext11_info, size_info, state);

    const auto &num_bundles = ext11_info.ext_11.num_prb_bundles;
    int bundles_included = 0;
    RT_LOGC_DEBUG(
            Oran::OranCplane,
            "EXT11 BUNDLE LOOP: Starting with {} bundles, bundle_size={}, "
            "total_section_info_size={}, pkt_section_info_room={}",
            num_bundles,
            size_info.bundle_size,
            state.total_section_info_size,
            metadata.pkt_section_info_room);
    const auto &section_info = info.sections.at(section_num);

    const auto section_start_prbc = static_cast<std::uint16_t>(section_info.sect_1.startPrbc.get());
    const auto section_num_prbc =
            (section_info.sect_1.numPrbc.get() == 0)
                    ? static_cast<std::uint16_t>(ORAN_MAX_PRB_X_SLOT)
                    : static_cast<std::uint16_t>(section_info.sect_1.numPrbc.get());
    const std::uint16_t section_max_prbc = section_start_prbc + section_num_prbc;
    std::uint16_t curr_start_prbc = section_start_prbc;

    for (int bundle_idx = 0; bundle_idx < num_bundles; ++bundle_idx) {
        const auto padding = oran_cmsg_se11_disableBFWs_0_padding_bytes(state.current_ext11_len);
        const auto space_needed = state.total_section_info_size +
                                  static_cast<std::uint32_t>(size_info.bundle_size) + padding;
        RT_LOGC_DEBUG(
                Oran::OranCplane,
                "EXT11 BUNDLE {}: padding={}, space_needed={}, "
                "room={}, fits={}",
                bundle_idx,
                padding,
                space_needed,
                metadata.pkt_section_info_room,
                space_needed <= metadata.pkt_section_info_room);
        if (space_needed > metadata.pkt_section_info_room) {
            RT_LOGC_DEBUG(
                    Oran::OranCplane,
                    "EXT11 FRAGMENTATION: Bundle {} doesn't fit, "
                    "fragmenting packet. "
                    "Included {} bundles so far, starting new packet",
                    bundle_idx,
                    bundles_included);

            memset(state.current_ptr, 0, padding);
            state.current_ext11_ptr->extLen = cpu_to_be_16(
                    static_cast<std::uint16_t>((state.current_ext11_len + padding) >> 2U));

            auto *sect_1_ptr = fapi::assume_cast<oran_cmsg_sect1>(state.section_ptr);
            auto num_prbc = bundles_included * ext11_info.ext_11.ext_hdr.numBundPrb;
            num_prbc = (curr_start_prbc + num_prbc > section_max_prbc)
                               ? (section_max_prbc - curr_start_prbc)
                               : num_prbc;
            sect_1_ptr->startPrbc = curr_start_prbc;
            sect_1_ptr->numPrbc = static_cast<std::uint16_t>(num_prbc);
            curr_start_prbc += num_prbc;

            const std::uint16_t data_len = mtu - state.pkt_remaining_capacity;
            finalize_packet(
                    *state.current_buffer,
                    flow,
                    state.common_hdr_ptr,
                    data_len,
                    section_num + 1 - state.sections_generated,
                    is_uplink);
            state.sections_generated = section_num;

            state.total_section_info_size = 0;
            state.current_ext11_len = 0;
            state.pkt_remaining_capacity = metadata.pkt_section_info_room;

            start_new_ext11_packet(
                    info,
                    flow,
                    buffers,
                    section_num,
                    metadata,
                    size_info.ext11_hdr_size,
                    ext4_opt,
                    ext5_opt,
                    ext11_info,
                    size_info.disable_bfws,
                    state);

            static constexpr auto EXT4_SIZE =
                    sizeof(oran_cmsg_ext_hdr) + sizeof(oran_cmsg_sect_ext_type_4);
            static constexpr auto EXT5_SIZE =
                    sizeof(oran_cmsg_ext_hdr) + sizeof(oran_cmsg_sect_ext_type_5);

            const auto ext4_size = ext4_opt.has_value() ? EXT4_SIZE : 0;
            const auto ext5_size = ext5_opt.has_value() ? EXT5_SIZE : 0;

            state.total_section_info_size +=
                    metadata.section_size + ext4_size + ext5_size + size_info.ext11_hdr_size;
            state.pkt_remaining_capacity -=
                    (metadata.section_size + ext4_size + ext5_size + size_info.ext11_hdr_size);
            bundles_included = 0;
        }

        if (bundle_idx >= ext11_info.ext_11.num_prb_bundles) {
            RT_LOGC_ERROR(
                    Oran::OranCplane,
                    "Bundle index {} out of bounds (max {})",
                    bundle_idx,
                    ext11_info.ext_11.num_prb_bundles - 1);
            throw std::out_of_range("Bundle index out of bounds");
        }

        auto bundles_span = std::span{ext11_info.ext_11.bundles, ext11_info.ext_11.num_prb_bundles};
        const auto &bundle_info = bundles_span[static_cast<std::size_t>(bundle_idx)];
        const auto comp_meth = size_info.disable_bfws
                                       ? UserDataBFWCompressionMethod::NO_COMPRESSION
                                       : static_cast<UserDataBFWCompressionMethod>(
                                                 ext11_info.ext_11.ext_comp_hdr.bfwCompMeth.get());

        auto bundle_buffer_span =
                fapi::make_buffer_span(state.current_ptr, state.pkt_remaining_capacity);
        copy_ext11_bundle(
                bundle_info,
                size_info.disable_bfws,
                comp_meth,
                size_info.bfw_iq_size,
                bundle_idx,
                bundle_buffer_span);

        // Update current_ptr from the span after bundle is copied
        state.current_ptr = fapi::assume_cast<std::uint8_t>(bundle_buffer_span.data());

        ++bundles_included;
        state.total_section_info_size += size_info.bundle_size;
        state.pkt_remaining_capacity -= size_info.bundle_size;
        state.current_ext11_len += size_info.bundle_size;
        RT_LOGC_DEBUG(
                Oran::OranCplane,
                "EXT11 BUNDLE {}: Successfully added, bundles_included={}, "
                "total_section_info_size={}, pkt_remaining_capacity={}",
                bundle_idx,
                bundles_included,
                state.total_section_info_size,
                state.pkt_remaining_capacity);
    }

    finalize_ext11_section(ext11_info, bundles_included, curr_start_prbc, section_max_prbc, state);
}

/**
 * Processes sections without extensions
 *
 * @tparam BufferSpanType Type of buffer span
 * @tparam BufferPtrType Type of buffer pointer
 * @param[in] info C-Plane message info
 * @param[in,out] flow Flow for sequence ID management
 * @param[in] buffers Buffer span
 * @param[in] metadata Section metadata with sizing info
 * @param[in] mtu Maximum transmission unit
 * @param[in] is_uplink Direction flag
 * @param[in,out] packet_num Packet counter
 * @param[in,out] current_buffer Current buffer being filled
 * @param[in,out] common_hdr_ptr Pointer to common header in current buffer
 * @param[in,out] section_ptr Pointer to section data in current buffer
 * @param[in,out] pkt_remaining_capacity Remaining capacity in current packet
 */
template <typename BufferSpanType, typename BufferPtrType>
void process_sections_without_extensions(
        const OranCPlaneMsgInfo &info,
        OranFlow &flow,
        BufferSpanType buffers,
        const SectionMetadata &metadata,
        const std::uint16_t mtu,
        const bool is_uplink,
        std::uint16_t &packet_num,
        BufferPtrType &current_buffer,
        std::uint8_t *&common_hdr_ptr,
        std::uint8_t *&section_ptr,
        std::uint16_t &pkt_remaining_capacity) {
    std::uint8_t section_num{};
    std::uint8_t sections_generated{};

    while (section_num < metadata.number_of_sections) {
        const std::size_t extension_size{};
        const std::size_t section_size_with_extension = metadata.section_size + extension_size;

        if (section_size_with_extension > pkt_remaining_capacity) {
            RT_LOGC_DEBUG(
                    Oran::OranCplane,
                    "Fragmenting: section {} ({} bytes) > capacity {}, "
                    "finalizing pkt {} ({} sections) → pkt {}",
                    section_num,
                    section_size_with_extension,
                    pkt_remaining_capacity,
                    packet_num - 1,
                    section_num - sections_generated,
                    packet_num);
            const std::uint16_t data_len = mtu - pkt_remaining_capacity;
            finalize_packet(
                    *current_buffer,
                    flow,
                    common_hdr_ptr,
                    data_len,
                    section_num - sections_generated,
                    is_uplink);
            sections_generated = section_num;

            if (packet_num >= buffers.size()) {
                RT_LOGC_ERROR(
                        Oran::OranCplane,
                        "Insufficient buffers for fragmentation: need packet "
                        "{} but only {} buffers provided",
                        packet_num,
                        buffers.size());
                throw std::runtime_error("Insufficient buffers for packet fragmentation");
            }

            current_buffer = &buffers[packet_num++];
            common_hdr_ptr =
                    initialize_new_packet(*current_buffer, flow, info, metadata.common_hdr_size);

            // Use span to calculate section_ptr without pointer arithmetic
            auto hdr_span = fapi::make_buffer_span(
                    common_hdr_ptr, metadata.common_hdr_size + metadata.section_size);
            section_ptr = fapi::assume_cast<std::uint8_t>(
                    hdr_span.subspan(metadata.common_hdr_size).data());
            pkt_remaining_capacity = static_cast<std::uint16_t>(
                    metadata.pkt_section_info_room - section_size_with_extension);
        } else {
            pkt_remaining_capacity -= section_size_with_extension;
        }

        std::memcpy(section_ptr, &info.sections.at(section_num), metadata.section_size);
        // Advance section_ptr using span instead of pointer arithmetic
        auto section_span =
                fapi::make_buffer_span(section_ptr, pkt_remaining_capacity + metadata.section_size);
        section_ptr =
                fapi::assume_cast<std::uint8_t>(section_span.subspan(metadata.section_size).data());
        section_num++;
    }

    const std::uint16_t data_len = mtu - pkt_remaining_capacity;
    finalize_packet(
            *current_buffer,
            flow,
            common_hdr_ptr,
            data_len,
            section_num - sections_generated,
            is_uplink);
}

} // namespace

template <std::derived_from<OranBuf> BufferType, std::size_t EXTENT>
std::uint16_t prepare_cplane_message(
        OranCPlaneMsgInfo &info,
        OranFlow &flow,
        OranPeer &peer,
        std::span<BufferType, EXTENT> buffers,
        const std::uint16_t mtu) {
    const auto metadata = extract_section_metadata(info, mtu);

    validate_all_extensions(info);
    validate_message_parameters(
            buffers, mtu, info.num_sections, metadata.pkt_section_info_room, metadata.section_size);

    std::uint16_t packet_num{};

    auto *current_buffer = &buffers[packet_num++];
    auto *data = current_buffer->template data_as<PacketHeaderTemplate>();
    current_buffer->clear_flags();
    std::memcpy(data, &flow.get_packet_header_template(), sizeof(PacketHeaderTemplate));

    auto *common_hdr_ptr =
            current_buffer->template data_at_offset<std::uint8_t>(sizeof(PacketHeaderTemplate));
    std::memcpy(common_hdr_ptr, &info.section_common_hdr, metadata.common_hdr_size);
    const auto direction = static_cast<oran_pkt_dir>(
            info.section_common_hdr.sect_1_common_hdr.radioAppHdr.dataDirection.get());
    const bool is_uplink = (direction == DIRECTION_UPLINK);

    // Use span to calculate section_ptr without pointer arithmetic
    auto hdr_span = fapi::make_buffer_span(
            common_hdr_ptr, metadata.common_hdr_size + metadata.section_size);
    auto *section_ptr =
            fapi::assume_cast<std::uint8_t>(hdr_span.subspan(metadata.common_hdr_size).data());
    std::uint16_t pkt_remaining_capacity = metadata.pkt_section_info_room;
    auto &last_packet_ts = is_uplink ? peer.get_last_ul_timestamp() : peer.get_last_dl_timestamp();

    if (!info.has_section_ext) {
        process_sections_without_extensions(
                info,
                flow,
                buffers,
                metadata,
                mtu,
                is_uplink,
                packet_num,
                current_buffer,
                common_hdr_ptr,
                section_ptr,
                pkt_remaining_capacity);

        if (info.tx_window_start > last_packet_ts) {
            // Set timestamp on all packets (including fragments)
            for (std::uint16_t pkt_idx = 0; pkt_idx < packet_num; ++pkt_idx) {
                buffers[pkt_idx].set_timestamp(info.tx_window_start);
            }
            last_packet_ts = info.tx_window_start;
        }

        return packet_num;
    }

    std::uint8_t section_num{};
    PacketBuildState state{};
    state.current_buffer = current_buffer;
    state.common_hdr_ptr = common_hdr_ptr;
    state.section_ptr = section_ptr;
    state.current_ptr = nullptr;
    state.packet_num = packet_num;
    state.total_section_info_size = 0;
    state.pkt_remaining_capacity = pkt_remaining_capacity;
    state.sections_generated = 0;
    auto ext4_hdr_size = sizeof(oran_cmsg_ext_hdr) + sizeof(oran_cmsg_sect_ext_type_4);
    auto ext5_hdr_size = sizeof(oran_cmsg_ext_hdr) + sizeof(oran_cmsg_sect_ext_type_5);

    for (section_num = 0; section_num < metadata.number_of_sections; section_num++) {
        if (state.total_section_info_size + metadata.section_size >
            metadata.pkt_section_info_room) {
            start_new_packet_for_section(
                    info, flow, buffers, section_num, metadata, mtu, is_uplink, state);
        }
        state.total_section_info_size += metadata.section_size;
        state.pkt_remaining_capacity -= metadata.section_size;
        auto &section_info = info.sections.at(section_num);
        memcpy(state.section_ptr, &section_info, metadata.section_size);

        // Use span to calculate current_ptr without pointer arithmetic
        auto state_section_span = fapi::make_buffer_span(
                state.section_ptr, state.pkt_remaining_capacity + metadata.section_size);
        state.current_ptr = fapi::assume_cast<std::uint8_t>(
                state_section_span.subspan(metadata.section_size).data());
        state.current_ext11_len = 0;

        // Validate extension configuration
        validate_section_extensions(section_info, metadata.section_type, section_num);

        if (metadata.section_type != ORAN_CMSG_SECTION_TYPE_1) {
            advance_section_ptr_and_reset_extensions(state, metadata);
            continue;
        }

        // It's section type 1 so we can check the extension flag
        const auto extension_flag = oran_cmsg_get_section_1_ef(&section_info.sect_1);
        if (!extension_flag) {
            advance_section_ptr_and_reset_extensions(state, metadata);
            continue;
        }

        if (section_info.ext5.has_value()) {
            apply_ext5_byte_order_conversion(section_info.ext5.value());
        }

        if (section_info.ext11.has_value()) {
            const auto &ext4_opt = section_info.ext4;
            const auto &ext5_opt = section_info.ext5;
            const auto &ext11_opt = section_info.ext11;

            process_extension_11(
                    info,
                    flow,
                    buffers,
                    section_num,
                    metadata,
                    ext11_opt,
                    ext4_opt,
                    ext5_opt,
                    mtu,
                    is_uplink,
                    state);
        } else if (section_info.ext4.has_value()) {
            process_simple_extension(
                    info,
                    flow,
                    buffers,
                    section_num,
                    section_info,
                    metadata,
                    mtu,
                    is_uplink,
                    section_info.ext4.value(),
                    ext4_hdr_size,
                    false,
                    state);
        } else if (section_info.ext5.has_value()) {
            process_simple_extension(
                    info,
                    flow,
                    buffers,
                    section_num,
                    section_info,
                    metadata,
                    mtu,
                    is_uplink,
                    section_info.ext5.value(),
                    ext5_hdr_size,
                    true,
                    state);
        }
        advance_section_ptr_and_reset_extensions(state, metadata);
    }

    const std::uint16_t data_len = mtu - state.pkt_remaining_capacity;
    finalize_packet(
            *state.current_buffer,
            flow,
            state.common_hdr_ptr,
            data_len,
            section_num - state.sections_generated,
            is_uplink);

    if (info.tx_window_start > last_packet_ts) {
        // Set timestamp on all packets (including fragments)
        for (std::uint16_t pkt_idx = 0; pkt_idx < state.packet_num; ++pkt_idx) {
            buffers[pkt_idx].set_timestamp(info.tx_window_start);
        }
        last_packet_ts = info.tx_window_start;
    }

    return state.packet_num;
}

std::size_t count_cplane_packets(std::span<OranCPlaneMsgInfo> infos, const std::uint16_t mtu) {
    std::size_t num_packets{};

    for (auto &info : infos) {
        const auto metadata = extract_section_metadata(info, mtu);

        if (!info.has_section_ext) {
            num_packets += count_packets_without_extensions(metadata);
        } else {
            num_packets += count_packets_with_extensions(info, metadata, mtu);
        }
    }

    return num_packets;
}

// NOLINTEND(cppcoreguidelines-pro-type-union-access)

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define INSTANTIATE_PREPARE_CPLANE_MESSAGE(BufferType)                                             \
    template std::uint16_t prepare_cplane_message<BufferType, std::dynamic_extent>(                \
            OranCPlaneMsgInfo & info,                                                              \
            OranFlow & flow,                                                                       \
            OranPeer & peer,                                                                       \
            std::span<BufferType, std::dynamic_extent> buffers,                                    \
            std::uint16_t mtu);

// Explicit template instantiations for buffer types
INSTANTIATE_PREPARE_CPLANE_MESSAGE(VecBuf)
INSTANTIATE_PREPARE_CPLANE_MESSAGE(MBuf)

#undef INSTANTIATE_PREPARE_CPLANE_MESSAGE

} // namespace ran::oran
