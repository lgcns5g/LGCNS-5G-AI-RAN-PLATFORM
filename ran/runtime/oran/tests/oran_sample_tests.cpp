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

/**
 * @file oran_sample_tests.cpp
 * @brief Sample tests for ORAN library documentation
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include <aerial-fh-driver/oran.hpp>

#include <gtest/gtest.h>

#include "oran/cplane_message.hpp"
#include "oran/cplane_types.hpp"
#include "oran/numerology.hpp"
#include "oran/vec_buf.hpp"

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

namespace {

// NOLINTBEGIN(cppcoreguidelines-pro-type-union-access)

TEST(OranSampleTests, BasicNumerology) {
    // example-begin basic-numerology-1
    // Create numerology from subcarrier spacing
    const auto numerology = ran::oran::from_scs(ran::oran::SubcarrierSpacing::Scs30Khz);

    // Access timing parameters derived from SCS
    const auto scs = numerology.subcarrier_spacing;
    const auto slots = numerology.slots_per_subframe; // 30 kHz SCS = 2 slots per 1ms subframe
    // example-end basic-numerology-1

    EXPECT_EQ(scs, ran::oran::SubcarrierSpacing::Scs30Khz);
    EXPECT_EQ(slots, 2U);
}

TEST(OranSampleTests, SlotTimingCalculation) {
    // example-begin slot-timing-1
    const auto numerology = ran::oran::from_scs(ran::oran::SubcarrierSpacing::Scs30Khz);

    // Calculate timing for absolute slot number
    constexpr std::uint64_t ABSOLUTE_SLOT = 100;
    const auto timing = numerology.calculate_slot_timing(ABSOLUTE_SLOT);

    // Timing decomposed into frame, subframe, and slot
    const auto frame = timing.frame_id;
    const auto subframe = timing.subframe_id;
    const auto slot = timing.slot_id;
    // example-end slot-timing-1

    EXPECT_EQ(frame, 5U);
    EXPECT_EQ(subframe, 0U);
    EXPECT_EQ(slot, 0U);
}

TEST(OranSampleTests, SlotTimestamp) {
    // example-begin slot-timestamp-1
    const auto numerology = ran::oran::from_scs(ran::oran::SubcarrierSpacing::Scs30Khz);

    // Create timing for frame 10, subframe 5, slot 1
    ran::oran::OranSlotTiming timing{};
    timing.frame_id = 10;
    timing.subframe_id = 5;
    timing.slot_id = 1;

    // Calculate absolute timestamp in nanoseconds
    const auto timestamp_ns = numerology.calculate_slot_timestamp(timing);
    // example-end slot-timestamp-1

    EXPECT_GT(timestamp_ns, 0U);
}

TEST(OranSampleTests, SubcarrierSpacingConversion) {
    // example-begin scs-conversion-1
    // Convert from kHz value
    const auto scs = ran::oran::from_khz(60);

    // Convert to kHz value
    const auto khz = ran::oran::to_khz(ran::oran::SubcarrierSpacing::Scs60Khz);
    // example-end scs-conversion-1

    ASSERT_TRUE(scs.has_value());
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    EXPECT_EQ(*scs, ran::oran::SubcarrierSpacing::Scs60Khz);
    EXPECT_EQ(khz, 60U);
}

TEST(OranSampleTests, BasicCPlaneMessage) {
    // example-begin basic-cplane-1
    // Create message info structure
    ran::oran::OranCPlaneMsgInfo msg_info{};

    // Set up radio application header
    auto &radio_hdr = msg_info.section_common_hdr.sect_1_common_hdr.radioAppHdr;
    radio_hdr.frameId = 10;
    radio_hdr.subframeId = 5;
    radio_hdr.slotId = 3;
    radio_hdr.startSymbolId = 0;
    radio_hdr.numberOfSections = 1;
    radio_hdr.sectionType = ORAN_CMSG_SECTION_TYPE_1;
    radio_hdr.dataDirection = DIRECTION_DOWNLINK;

    // Configure message properties
    msg_info.data_direction = DIRECTION_DOWNLINK;
    msg_info.num_sections = 1;
    msg_info.tx_window_start = 1000000ULL;

    // Configure section
    auto &section = msg_info.sections.at(0);
    section.sect_1.sectionId = 1;
    section.sect_1.startPrbc = 0;
    section.sect_1.numPrbc = 10;
    section.sect_1.beamId = 0;
    // example-end basic-cplane-1

    EXPECT_EQ(msg_info.num_sections, 1);
}

TEST(OranSampleTests, CPlaneMessagePacketCreation) {
    // example-begin cplane-packets-1
    // Create message information
    ran::oran::OranCPlaneMsgInfo msg_info{};
    msg_info.section_common_hdr.sect_1_common_hdr.radioAppHdr.sectionType =
            ORAN_CMSG_SECTION_TYPE_1;
    msg_info.section_common_hdr.sect_1_common_hdr.radioAppHdr.numberOfSections = 1;
    msg_info.section_common_hdr.sect_1_common_hdr.radioAppHdr.dataDirection = DIRECTION_DOWNLINK;
    msg_info.data_direction = DIRECTION_DOWNLINK;
    msg_info.num_sections = 1;

    // Set up flow and peer for packet generation
    ran::oran::PacketHeaderTemplate header_template{};
    header_template.ecpri.ecpriVersion = ORAN_DEF_ECPRI_VERSION;
    header_template.ecpri.ecpriMessage = ECPRI_MSG_TYPE_RTC;

    ran::oran::SimpleOranFlow flow(header_template);
    ran::oran::SimpleOranPeer peer;

    // Allocate output buffers
    std::vector<ran::oran::VecBuf> buffers(5, ran::oran::VecBuf(1500));
    constexpr std::uint16_t MTU = 1500;

    // Create C-plane packets
    const auto packet_count = ran::oran::prepare_cplane_message(
            msg_info, flow, peer, std::span<ran::oran::VecBuf>{buffers}, MTU);
    // example-end cplane-packets-1

    EXPECT_EQ(packet_count, 1);
    EXPECT_GT(buffers[0].size(), 0);
}

TEST(OranSampleTests, CountCPlanePackets) {
    // example-begin count-packets-1
    // Create message information
    ran::oran::OranCPlaneMsgInfo msg_info{};
    msg_info.section_common_hdr.sect_1_common_hdr.radioAppHdr.sectionType =
            ORAN_CMSG_SECTION_TYPE_1;
    msg_info.section_common_hdr.sect_1_common_hdr.radioAppHdr.numberOfSections = 5;
    msg_info.num_sections = 5;

    // Predict packet count before allocation
    std::array<ran::oran::OranCPlaneMsgInfo, 1> messages = {msg_info};
    constexpr std::uint16_t MTU = 1500;

    const auto predicted_count = ran::oran::count_cplane_packets(messages, MTU);
    // example-end count-packets-1

    EXPECT_GE(predicted_count, 1);
}

TEST(OranSampleTests, VecBufUsage) {
    // example-begin vec-buf-1
    // Create a vector-based buffer
    constexpr std::size_t BUFFER_SIZE = 1500;
    ran::oran::VecBuf buffer(BUFFER_SIZE);

    // Access buffer properties
    const auto capacity = buffer.capacity();
    const auto initial_size = buffer.size();

    // Set data size
    buffer.set_size(100);
    const auto new_size = buffer.size();

    // Access buffer data
    auto *data_ptr = buffer.data();
    // example-end vec-buf-1

    EXPECT_EQ(capacity, BUFFER_SIZE);
    EXPECT_EQ(initial_size, 0);
    EXPECT_EQ(new_size, 100);
    EXPECT_NE(data_ptr, nullptr);
}

TEST(OranSampleTests, BufferTimestamp) {
    // example-begin buffer-timestamp-1
    ran::oran::VecBuf buffer(1500);

    // Set timestamp on buffer
    constexpr std::uint64_t TIMESTAMP = 1234567890ULL;
    buffer.set_timestamp(TIMESTAMP);

    // Check timestamp was set
    const auto has_timestamp = buffer.has_timestamp();
    const auto timestamp_value = buffer.get_timestamp();
    // example-end buffer-timestamp-1

    EXPECT_TRUE(has_timestamp);
    EXPECT_EQ(timestamp_value, TIMESTAMP);
}

// NOLINTEND(cppcoreguidelines-pro-type-union-access)

} // namespace

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
