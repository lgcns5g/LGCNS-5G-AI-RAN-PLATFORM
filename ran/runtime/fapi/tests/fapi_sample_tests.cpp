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
 * @file fapi_sample_tests.cpp
 * @brief Sample tests for FAPI library documentation
 */

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <vector>

#include <scf_5g_fapi.h>

#include <gtest/gtest.h>

#include "fapi/fapi_buffer.hpp"
#include "fapi/fapi_file_reader.hpp"
#include "fapi/fapi_file_replay.hpp"
#include "fapi/fapi_file_writer.hpp"
#include "fapi/fapi_state.hpp"
#include "fapi_test_utils.hpp"

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

namespace {

[[nodiscard]] inline std::filesystem::path get_test_output_dir() {
    return std::filesystem::path{"/tmp"};
}

TEST(FapiSampleTests, FileWriterBasic) {
    // example-begin file-writer-1
    const auto output_path = get_test_output_dir() / "fapi_capture_sample.fapi";

    // Create file writer
    ran::fapi::FapiFileWriter writer(output_path.string());

    // Create sample message data
    ran::fapi::FapiMessageData msg{};
    msg.cell_id = 0;
    msg.msg_id = SCF_FAPI_DL_TTI_REQUEST;

    std::vector<std::uint8_t> msg_buffer(128);
    msg.msg_buf = std::span<const std::uint8_t>{msg_buffer.data(), msg_buffer.size()};

    // Capture message
    writer.capture_message(msg);

    // Write to file
    writer.flush_to_file();
    // example-end file-writer-1

    EXPECT_EQ(writer.get_message_count(), 1);
}

TEST(FapiSampleTests, FileReaderBasic) {
    // Create a test file first
    const auto file_path = get_test_output_dir() / "fapi_reader_sample.fapi";
    {
        ran::fapi::FapiFileWriter writer(file_path.string());
        ran::fapi::FapiMessageData msg{};
        msg.cell_id = 0;
        msg.msg_id = SCF_FAPI_UL_TTI_REQUEST;
        std::vector<std::uint8_t> msg_buffer(64);
        msg.msg_buf = std::span<const std::uint8_t>{msg_buffer.data(), msg_buffer.size()};
        writer.capture_message(msg);
        writer.flush_to_file();
    }

    // example-begin file-reader-1
    // Open FAPI capture file
    ran::fapi::FapiFileReader reader(file_path.string());

    // Read messages
    std::size_t msg_count{};
    while (auto msg = reader.read_next()) {
        msg_count++;
    }
    // example-end file-reader-1

    EXPECT_EQ(msg_count, 1);
}

TEST(FapiSampleTests, FileReaderReadAll) {
    // Create test file
    const auto file_path = get_test_output_dir() / "fapi_read_all_sample.fapi";
    {
        ran::fapi::FapiFileWriter writer(file_path.string());
        for (int i = 0; i < 3; i++) {
            ran::fapi::FapiMessageData msg{};
            msg.cell_id = static_cast<std::uint16_t>(i);
            msg.msg_id = SCF_FAPI_SLOT_INDICATION;
            std::vector<std::uint8_t> msg_buffer(32);
            msg.msg_buf = std::span<const std::uint8_t>{msg_buffer.data(), msg_buffer.size()};
            writer.capture_message(msg);
        }
        writer.flush_to_file();
    }

    // example-begin file-reader-read-all-1
    ran::fapi::FapiFileReader reader(file_path.string());

    // Read all messages at once
    const auto all_messages = reader.read_all();
    // example-end file-reader-read-all-1

    EXPECT_EQ(all_messages.size(), 3);
}

TEST(FapiSampleTests, StateInitialization) {
    // example-begin state-init-1
    // Configure FAPI state machine
    ran::fapi::FapiState::InitParams params{};
    params.nvipc_config_string =
            ran::fapi::YamlConfigBuilder::create_primary_config("fapi_sample_init");
    params.max_cells = 4;
    params.max_sfn = 1024;
    params.max_slot = 20;

    // Initialize FAPI state
    const ran::fapi::FapiState fapi_state(params);
    // example-end state-init-1

    EXPECT_EQ(fapi_state.get_num_cells_configured(), 0);
}

TEST(FapiSampleTests, SlotManagement) {
    ran::fapi::FapiState::InitParams params{};
    params.nvipc_config_string =
            ran::fapi::YamlConfigBuilder::create_primary_config("fapi_slot_test");

    ran::fapi::FapiState fapi_state(params);

    // example-begin slot-management-1
    // Reset slot to (0, 0)
    fapi_state.reset_slot();

    // Get current slot
    const auto slot = fapi_state.get_current_slot();

    // Advance to next slot
    fapi_state.increment_slot();

    const auto next_slot = fapi_state.get_current_slot();
    // example-end slot-management-1

    EXPECT_EQ(slot.sfn, 0);
    EXPECT_EQ(slot.slot, 0);
    EXPECT_EQ(next_slot.sfn, 0);
    EXPECT_EQ(next_slot.slot, 1);
}

TEST(FapiSampleTests, MessageCallbacks) {
    ran::fapi::FapiState::InitParams params{};
    params.nvipc_config_string =
            ran::fapi::YamlConfigBuilder::create_primary_config("fapi_callback_test");

    ran::fapi::FapiState fapi_state(params);

    // example-begin message-callbacks-1
    std::size_t message_count{};

    // Set callback to capture all messages
    fapi_state.set_on_message(
            [&message_count](const ran::fapi::FapiMessageData & /*msg*/) { message_count++; });

    // Set callback for UL TTI requests
    fapi_state.set_on_ul_tti_request([](std::uint16_t /*cell_id*/,
                                        const scf_fapi_body_header_t & /*body_hdr*/,
                                        std::uint32_t /*body_len*/) {
        // Process UL TTI request
    });

    // Set callback for DL TTI requests
    fapi_state.set_on_dl_tti_request([](std::uint16_t /*cell_id*/,
                                        const scf_fapi_body_header_t & /*body_hdr*/,
                                        std::uint32_t /*body_len*/) {
        // Process DL TTI request
    });
    // example-end message-callbacks-1

    EXPECT_EQ(message_count, 0);
}

TEST(FapiSampleTests, ConfigCallbacks) {
    ran::fapi::FapiState::InitParams params{};
    params.nvipc_config_string =
            ran::fapi::YamlConfigBuilder::create_primary_config("fapi_config_callback_test");

    ran::fapi::FapiState fapi_state(params);

    // example-begin config-callbacks-1
    // Set callback for CONFIG.request
    fapi_state.set_on_config_request(
            [](std::uint16_t /*cell_id*/,
               const scf_fapi_body_header_t & /*body_hdr*/,
               std::uint32_t /*body_len*/) -> scf_fapi_error_codes_t {
                // Validate configuration
                return static_cast<scf_fapi_error_codes_t>(SCF_FAPI_MSG_OK);
            });

    // Set callback for START.request
    fapi_state.set_on_start_request(
            [](std::uint16_t /*cell_id*/,
               const scf_fapi_body_header_t & /*body_hdr*/,
               std::uint32_t /*body_len*/) -> scf_fapi_error_codes_t {
                // Initialize cell for operation
                return static_cast<scf_fapi_error_codes_t>(SCF_FAPI_MSG_OK);
            });

    // Set callback for STOP.request
    fapi_state.set_on_stop_request(
            [](std::uint16_t /*cell_id*/,
               const scf_fapi_body_header_t & /*body_hdr*/,
               std::uint32_t /*body_len*/) -> scf_fapi_error_codes_t {
                // Clean up cell resources
                return static_cast<scf_fapi_error_codes_t>(SCF_FAPI_MSG_OK);
            });
    // example-end config-callbacks-1
}

TEST(FapiSampleTests, FileReplayBasic) {
    // Create test file with UL_TTI_REQUEST messages
    const auto file_path = get_test_output_dir() / "fapi_replay_sample.fapi";
    {
        ran::fapi::FapiFileWriter writer(file_path.string());
        ran::fapi::FapiMessageData msg{};
        msg.cell_id = 0;
        msg.msg_id = SCF_FAPI_UL_TTI_REQUEST;

        // Create minimal UL_TTI_REQUEST
        std::vector<std::uint8_t> msg_buffer(sizeof(scf_fapi_ul_tti_req_t));
        auto *ul_req = ran::fapi::assume_cast<scf_fapi_ul_tti_req_t>(msg_buffer.data());
        ul_req->sfn = 0;
        ul_req->slot = 0;
        ul_req->num_pdus = 0;
        ul_req->num_ulsch = 0;
        ul_req->num_ulcch = 0;

        msg.msg_buf = std::span<const std::uint8_t>{msg_buffer.data(), msg_buffer.size()};
        writer.capture_message(msg);
        writer.flush_to_file();
    }

    // example-begin file-replay-1
    static constexpr std::uint8_t SLOTS_PER_SUBFRAME = 2;

    // Load FAPI capture file for replay
    ran::fapi::FapiFileReplay replay(file_path.string(), SLOTS_PER_SUBFRAME);

    // Get cell IDs
    const auto &cell_ids = replay.get_cell_ids();

    // Get request for current slot
    for (const auto cell_id : cell_ids) {
        auto req = replay.get_request_for_current_slot(cell_id);
        if (req) {
            // Process request (timing is updated automatically)
        }
    }

    // Advance to next slot
    const auto next_slot = replay.advance_slot();
    // example-end file-replay-1

    EXPECT_GT(cell_ids.size(), 0);

    EXPECT_GT(next_slot, 0);
    EXPECT_EQ(replay.get_cell_count(), 1);
}

} // namespace

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
