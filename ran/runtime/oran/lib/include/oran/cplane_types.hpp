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

#ifndef RAN_ORAN_CPLANE_TYPES_HPP
#define RAN_ORAN_CPLANE_TYPES_HPP

#include <array>
#include <cstdint>
#include <optional>
#include <span>

#include <aerial-fh-driver/oran.hpp>

#include "oran/oran_buf.hpp"
#include "oran/oran_export.hpp"
#include "oran/vec_buf.hpp"

namespace ran::oran {

/// Maximum number of sections per C-plane message
inline constexpr std::size_t MAX_CPLANE_SECTIONS = 64;

/**
 * Slot timing information
 */
struct ORAN_EXPORT OranSlotTiming final {
    std::uint8_t frame_id{};    //!< Frame ID (0-255)
    std::uint8_t subframe_id{}; //!< Subframe ID (0-9)
    std::uint8_t slot_id{};     //!< Slot ID (depends on numerology)
};

/**
 * Transmission timing windows
 */
struct ORAN_EXPORT OranTxWindows final {
    std::uint64_t tx_window_start{};     //!< Transmission window start timestamp (ns)
    std::uint64_t tx_window_bfw_start{}; //!< Beamforming window start timestamp (ns)
    std::uint64_t tx_window_end{};       //!< Transmission window end timestamp (ns)
};

/**
 * Packet header template for Ethernet + VLAN + eCPRI headers
 */
// The __packed__ attribute is required to match exact network wire format without padding
// This is intentional for network protocol headers that must match byte-for-byte specifications
#ifdef __GNUC__
#ifndef __clang__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpacked-not-aligned"
#endif
#endif
struct ORAN_EXPORT PacketHeaderTemplate final {
    oran_ether_hdr eth;   //!< Ethernet header
    oran_vlan_hdr vlan;   //!< VLAN header
    oran_ecpri_hdr ecpri; //!< eCPRI header
} __attribute__((__packed__));
#ifdef __GNUC__
#ifndef __clang__
#pragma GCC diagnostic pop
#endif
#endif

/**
 * C-plane section common header union
 */
union CPlaneSectionCommonHdr {
    oran_cmsg_sect0_common_hdr sect_0_common_hdr{}; //!< Section type 0 common header
    oran_cmsg_sect1_common_hdr sect_1_common_hdr;   //!< Section type 1 common header
    oran_cmsg_sect3_common_hdr sect_3_common_hdr;   //!< Section type 3 common header
    oran_cmsg_sect5_common_hdr sect_5_common_hdr;   //!< Section type 5 common header
};

/**
 * C-plane Section Extension Type 4 Info
 */
struct ORAN_EXPORT CPlaneSectionExt4Info final {
    oran_cmsg_sect_ext_type_4 ext_hdr{}; //!< Extension type 4 header
};

/**
 * C-plane Section Extension Type 5 Info
 */
struct ORAN_EXPORT CPlaneSectionExt5Info final {
    oran_cmsg_sect_ext_type_5 ext_hdr{}; //!< Extension type 5 header
};

// Suppress pedantic warning: These structs embed types from aerial-fh-driver/oran.hpp
// that contain zero-size arrays (flexible array members), which are a C idiom but not
// standard C++. The types are used correctly (arrays not accessed directly) and
// suppressing this warning is necessary to maintain compatibility with the external SDK.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"

/**
 * C-plane Section Extension Type 11 bundle information
 */
struct ORAN_EXPORT CPlaneSectionExt11BundlesInfo final {
    /// @cond HIDE_FROM_DOXYGEN
    union {
        oran_cmsg_sect_ext_type_11_disableBFWs_0_bfp_compressed_bundle_hdr
                disable_bfws_0_compressed{}; //!< Compressed bundle header (disableBFWs=0)
        oran_cmsg_sect_ext_type_11_disableBFWs_0_bundle_uncompressed
                disable_bfws_0_uncompressed; //!< Uncompressed bundle (disableBFWs=0)
        oran_cmsg_sect_ext_type_11_disableBFWs_1_bundle
                disable_bfws_1; //!< Bundle with disableBFWs=1
    };
    /// @endcond

    std::uint8_t *bfw_iq{}; //!< Beamforming weight IQ data pointer
};

/**
 * C-plane Section Extension Type 11 information
 */
struct ORAN_EXPORT CPlaneSectionExt11Info final {
    oran_cmsg_sect_ext_type_11 ext_hdr{}; //!< Extension type 11 header
    oran_cmsg_sect_ext_type_11_disableBFWs_0_bfwCompHdr
            ext_comp_hdr{};                   //!< Beamforming compression header
    CPlaneSectionExt11BundlesInfo *bundles{}; //!< Array of bundle information
    std::uint16_t num_prb_bundles{};          //!< Number of PRB bundles
    std::uint16_t num_bund_prb{};             //!< Number of PRBs per bundle
    std::uint8_t bundle_hdr_size{};           //!< Bundle header size in bytes
    std::uint16_t bfw_iq_size{};              //!< Beamforming weight IQ data size
    std::uint8_t bundle_size{};               //!< Total bundle size in bytes
    bool static_bfw{};                        //!< Static beamforming weights flag
    std::uint8_t *bfw_iq{}; //!< Beamforming weight IQ buffer with offset for this eAxC
    std::uint8_t
            start_bundle_offset_in_bfw_buffer{}; //!< Start bundle offset due to PRB fragmentation
};

// NOLINTBEGIN(cppcoreguidelines-pro-type-union-access)
/**
 * C-plane Section Extension Info
 */
struct ORAN_EXPORT CPlaneSectionExtInfo final {
    oran_cmsg_ext_hdr sect_ext_common_hdr{}; //!< Section extension common header
    /// @cond HIDE_FROM_DOXYGEN
    union {
        CPlaneSectionExt4Info ext_4{}; //!< Extension type 4 information
        CPlaneSectionExt5Info ext_5;   //!< Extension type 5 information
        CPlaneSectionExt11Info ext_11; //!< Extension type 11 information
    };
    /// @endcond
};
// NOLINTEND(cppcoreguidelines-pro-type-union-access)

#pragma GCC diagnostic pop

/**
 * Cleaned up C-plane section info without pointers
 */
struct ORAN_EXPORT OranCPlaneSectionInfo final {
    /// @cond HIDE_FROM_DOXYGEN
    union {
        oran_cmsg_sect0 sect_0{}; //!< Section type 0
        oran_cmsg_sect1 sect_1;   //!< Section type 1
        oran_cmsg_sect3 sect_3;   //!< Section type 3
        oran_cmsg_sect5 sect_5;   //!< Section type 5
    };
    /// @endcond

    bool csirs_of_multiplex_pdsch_csirs{}; //!< CSI-RS multiplexed with PDSCH flag

    /// Optional section extensions (at most one of each type per section)
    std::optional<CPlaneSectionExtInfo> ext4; ///< Extension type 4
    std::optional<CPlaneSectionExtInfo> ext5; ///< Extension type 5
    std::optional<CPlaneSectionExtInfo>
            ext11; ///< Extension type 11 (has flexible arrays, use pointer)
};

/**
 * Abstract flow interface for packet template and sequence generation
 */
class ORAN_EXPORT OranFlow {
public:
    OranFlow() = default;
    virtual ~OranFlow() = default;

    /**
     * Copy constructor.
     */
    OranFlow(const OranFlow &) = default;

    /**
     * Copy assignment operator.
     * @return Reference to this object
     */
    OranFlow &operator=(const OranFlow &) = default;

    /**
     * Move constructor.
     */
    OranFlow(OranFlow &&) = default;

    /**
     * Move assignment operator.
     * @return Reference to this object
     */
    OranFlow &operator=(OranFlow &&) = default;

    /**
     * Get packet header template for this flow
     * @return Reference to packet header template
     */
    [[nodiscard]] virtual const PacketHeaderTemplate &get_packet_header_template() const = 0;

    /**
     * Generate next uplink sequence ID
     * @return Next sequence ID for uplink packets
     */
    [[nodiscard]] virtual std::uint8_t next_sequence_id_uplink() = 0;

    /**
     * Generate next downlink sequence ID
     * @return Next sequence ID for downlink packets
     */
    [[nodiscard]] virtual std::uint8_t next_sequence_id_downlink() = 0;
};

/**
 * Abstract peer interface for timestamp tracking
 */
class ORAN_EXPORT OranPeer {
public:
    OranPeer() = default;
    virtual ~OranPeer() = default;

    /**
     * Copy constructor.
     */
    OranPeer(const OranPeer &) = default;

    /**
     * Copy assignment operator.
     * @return Reference to this object
     */
    OranPeer &operator=(const OranPeer &) = default;

    /**
     * Move constructor.
     */
    OranPeer(OranPeer &&) = default;

    /**
     * Move assignment operator.
     * @return Reference to this object
     */
    OranPeer &operator=(OranPeer &&) = default;

    /**
     * Get reference to last downlink timestamp
     * @return Reference to last DL timestamp for modification
     */
    virtual std::uint64_t &get_last_dl_timestamp() = 0;

    /**
     * Get reference to last uplink timestamp
     * @return Reference to last UL timestamp for modification
     */
    virtual std::uint64_t &get_last_ul_timestamp() = 0;
};

/**
 * Minimal C-plane message info structure
 */
struct ORAN_EXPORT OranCPlaneMsgInfo final {
    CPlaneSectionCommonHdr section_common_hdr{};                       //!< Radio app header
    oran_pkt_dir data_direction{};                                     //!< UL/DL direction
    bool has_section_ext{};                                            //!< Has section extensions
    std::uint16_t ap_idx{};                                            //!< Antenna port index
    std::array<OranCPlaneSectionInfo, MAX_CPLANE_SECTIONS> sections{}; //!< Section data array
    std::uint8_t num_sections{};                                       //!< Number of sections used
    std::uint64_t tx_window_start{};     //!< Transmission window start timestamp
    std::uint64_t tx_window_bfw_start{}; //!< Beamforming window start timestamp
    std::uint64_t tx_window_end{};       //!< Transmission window end timestamp
};

/**
 * Simple implementation of OranFlow interface
 */
class ORAN_EXPORT SimpleOranFlow final : public OranFlow {
private:
    PacketHeaderTemplate header_template_{};
    std::uint8_t seq_id_ul_{};
    std::uint8_t seq_id_dl_{};

public:
    /**
     * Construct flow with packet header template
     * @param[in] template_hdr Packet header template to use
     */
    explicit SimpleOranFlow(const PacketHeaderTemplate &template_hdr)
            : header_template_(template_hdr) {}

    [[nodiscard]] const PacketHeaderTemplate &get_packet_header_template() const override {
        return header_template_;
    }

    [[nodiscard]] std::uint8_t next_sequence_id_uplink() override { return ++seq_id_ul_; }

    [[nodiscard]] std::uint8_t next_sequence_id_downlink() override { return ++seq_id_dl_; }
};

/**
 * Simple implementation of OranPeer interface
 */
class ORAN_EXPORT SimpleOranPeer final : public OranPeer {
private:
    std::uint64_t last_dl_ts_{};
    std::uint64_t last_ul_ts_{};

public:
    std::uint64_t &get_last_dl_timestamp() override { return last_dl_ts_; }

    std::uint64_t &get_last_ul_timestamp() override { return last_ul_ts_; }
};

} // namespace ran::oran

#endif // RAN_ORAN_CPLANE_TYPES_HPP
