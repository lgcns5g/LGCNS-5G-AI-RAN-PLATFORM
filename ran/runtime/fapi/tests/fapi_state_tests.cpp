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

#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <nv_ipc.h>
#include <nv_ipc.hpp>
#include <scf_5g_fapi.h>
#include <yaml.hpp>

#include <gsl-lite/gsl-lite.hpp>
#include <gtest/gtest.h>

#include "fapi/fapi_buffer.hpp"
#include "fapi/fapi_state.hpp"
#include "fapi_test_utils.hpp"

namespace {

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

namespace rf = ran::fapi;

/**
 * Parameters for FapiMessageBuilder
 */
struct FapiMessageParams {
    uint16_t cell_id{};          //!< Cell identifier
    uint16_t msg_type{};         //!< FAPI message type
    std::size_t payload_size{0}; //!< Payload size in bytes
};

/**
 * Helper to create a FAPI message for testing
 *
 * Creates a properly formatted nv_ipc_msg_t with FAPI headers.
 * Follows SCF FAPI spec requirement for 4-byte alignment.
 */
class FapiMessageBuilder {
public:
    explicit FapiMessageBuilder(const FapiMessageParams &params)
            : buffer_(sizeof(scf_fapi_header_t) + sizeof(scf_fapi_body_header_t) +
                      align_to_4_bytes(params.payload_size)) {
        msg_.msg_buf = buffer_.data();
        msg_.cell_id = static_cast<int32_t>(params.cell_id);
        msg_.msg_id = static_cast<int32_t>(params.msg_type);
        msg_.data_len = static_cast<int32_t>(buffer_.size());

        const std::span<std::byte> buffer_span =
                rf::make_buffer_span(buffer_.data(), buffer_.size());

        auto *hdr = rf::assume_cast<scf_fapi_header_t>(buffer_span.data());
        hdr->message_count = 1;
        hdr->handle_id = static_cast<uint8_t>(params.cell_id);

        const std::span<std::byte> body_span = buffer_span.subspan(sizeof(scf_fapi_header_t));
        body_hdr_ = rf::assume_cast<scf_fapi_body_header_t>(body_span.data());
        body_hdr_->type_id = params.msg_type;
        body_hdr_->length = static_cast<uint32_t>(
                align_to_4_bytes(params.payload_size)); // Set aligned length per SCF spec

        msg_.msg_len = static_cast<int32_t>(buffer_.size());
    }

    nv_ipc_msg_t &get() { return msg_; }

    void *get_body_header() { return body_hdr_; }

    void *get_body_payload() {
        const std::span<std::byte> body_hdr_span =
                rf::make_buffer_span(body_hdr_, sizeof(scf_fapi_body_header_t) + body_hdr_->length);
        const std::span<std::byte> payload_span =
                body_hdr_span.subspan(sizeof(scf_fapi_body_header_t));
        return payload_span.data();
    }

private:
    /// Align size to 4-byte boundary per SCF FAPI spec
    static constexpr size_t align_to_4_bytes(const size_t size) { return (size + 3) & ~size_t(3); }

    std::vector<uint8_t> buffer_;
    nv_ipc_msg_t msg_{};
    scf_fapi_body_header_t *body_hdr_{};
};

/**
 * Unit tests for FapiState types and helpers
 *
 * Simple tests that don't require NVIPC initialization
 */

TEST(FapiStateTest, SlotInfoDefaultInitialization) {
    const rf::SlotInfo slot{};
    EXPECT_EQ(slot.sfn, 0);
    EXPECT_EQ(slot.slot, 0);
}

TEST(FapiStateTest, SlotInfoEquality) {
    const rf::SlotInfo s1{10, 5};
    const rf::SlotInfo s2{10, 5};
    const rf::SlotInfo s3{10, 6};
    const rf::SlotInfo s4{11, 5};

    EXPECT_EQ(s1, s2);
    EXPECT_NE(s1, s3);
    EXPECT_NE(s1, s4);
    EXPECT_NE(s3, s4);
}

/**
 * Parameters for create_fapi_state helper
 */
struct FapiStateParams {
    std::string prefix;
    uint32_t max_cells{4};
    uint32_t max_sfn{1024};
    uint32_t max_slot{20};
};

/**
 * Create a FapiState instance with specified parameters
 */
std::unique_ptr<rf::FapiState> create_fapi_state(const FapiStateParams &params) {
    rf::FapiState::InitParams init_params;
    init_params.nvipc_config_string = rf::YamlConfigBuilder::create_primary_config(params.prefix);
    init_params.max_cells = params.max_cells;
    init_params.max_sfn = static_cast<uint16_t>(params.max_sfn);
    init_params.max_slot = static_cast<uint16_t>(params.max_slot);

    return std::make_unique<rf::FapiState>(init_params);
}

/**
 * Test fixture for NVIPC integration tests
 *
 * Tests state machine transitions, message sending/receiving,
 * and full integration with NVIPC using string-based configuration
 */
class FapiStateNvIpcTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create unique prefix for this test
        test_prefix_ =
                "fapi_test_" +
                std::to_string(::testing::UnitTest::GetInstance()->current_test_info()->line());

        // Create FapiState with YAML string (no file needed)
        rf::FapiState::InitParams params;
        params.nvipc_config_string = rf::YamlConfigBuilder::create_primary_config(test_prefix_);
        params.max_cells = 4;
        params.max_sfn = 1024;
        params.max_slot = 20;

        fapi_state_ = std::make_unique<rf::FapiState>(params);
        ASSERT_NE(fapi_state_, nullptr) << "Failed to create FapiState";

        // Create secondary Ipc (MAC side)
        secondary_ipc_ = create_secondary_ipc();
        ASSERT_NE(secondary_ipc_, nullptr) << "Failed to create secondary Ipc";
    }

    void TearDown() override {
        if (secondary_ipc_ != nullptr) {
            secondary_ipc_->ipc_destroy(secondary_ipc_);
            secondary_ipc_ = nullptr;
        }

        // Destroy FapiState (will destroy primary Ipc)
        fapi_state_.reset();
    }

    nv_ipc_t *create_secondary_ipc() {
        try {
            // Write YAML to temp file (secondary still needs file for test infrastructure)
            const std::filesystem::path temp_dir = std::filesystem::temp_directory_path();
            secondary_config_file_ = temp_dir / ("fapi_test_secondary_" + test_prefix_ + ".yaml");

            // Setup cleanup guard to ensure temp file is removed
            const auto cleanup =
                    gsl_lite::finally([this] { std::filesystem::remove(secondary_config_file_); });

            const std::string yaml_content =
                    rf::YamlConfigBuilder::create_secondary_config(test_prefix_);
            std::ofstream ofs(secondary_config_file_);
            ofs << yaml_content;
            ofs.close();

            // Parse YAML file
            yaml::file_parser parser(secondary_config_file_.c_str());
            yaml::document doc = parser.next_document();
            const yaml::node root = doc.root();
            yaml::node transport = root["transport"];

            nv_ipc_config_t config{};
            const nv_ipc_module_t module = NV_IPC_MODULE_MAC;
            nv_ipc_parse_yaml_node(&config, &transport, module);

            return create_nv_ipc_interface(&config);
        } catch (const std::exception &e) {
            ADD_FAILURE() << "Failed to create secondary Ipc: " << e.what();
            return nullptr;
        }
    }

    std::string test_prefix_;
    std::filesystem::path secondary_config_file_;
    std::unique_ptr<rf::FapiState> fapi_state_;
    nv_ipc_t *secondary_ipc_{nullptr};

    // Helper to process CONFIG.request message
    [[nodiscard]] scf_fapi_error_codes_t process_config_request(const uint16_t cell_id) {
        constexpr size_t PAYLOAD_SIZE = sizeof(uint16_t); // num_tlvs
        FapiMessageBuilder builder({cell_id, SCF_FAPI_CONFIG_REQUEST, PAYLOAD_SIZE});
        auto *config_body =
                rf::assume_cast<scf_fapi_config_request_msg_t>(builder.get_body_header());
        config_body->msg_body.num_tlvs = 0;
        return fapi_state_->process_message(builder.get());
    }

    // Helper to process START.request message
    [[nodiscard]] scf_fapi_error_codes_t process_start_request(const uint16_t cell_id) {
        FapiMessageBuilder builder({cell_id, SCF_FAPI_START_REQUEST, 0});
        return fapi_state_->process_message(builder.get());
    }

    // Helper to process STOP.request message
    [[nodiscard]] scf_fapi_error_codes_t process_stop_request(const uint16_t cell_id) {
        FapiMessageBuilder builder({cell_id, SCF_FAPI_STOP_REQUEST, 0});
        return fapi_state_->process_message(builder.get());
    }

    // Helper to consume CONFIG.response from secondary Ipc
    void drain_config_response() {
        nv_ipc_msg_t msg{};
        const int ret = secondary_ipc_->rx_recv_msg(secondary_ipc_, &msg);
        ASSERT_GE(ret, 0) << "Failed to receive CONFIG.response";

        const std::span<const std::byte> msg_buffer =
                rf::make_const_buffer_span(msg.msg_buf, static_cast<std::size_t>(msg.msg_len));

        const std::span<const std::byte> body_span = msg_buffer.subspan(sizeof(scf_fapi_header_t));
        const auto *body_hdr = rf::assume_cast<scf_fapi_body_header_t>(body_span.data());

        // Verify it's actually a CONFIG.response
        EXPECT_EQ(body_hdr->type_id, SCF_FAPI_CONFIG_RESPONSE)
                << "Expected CONFIG.response but got type_id=" << body_hdr->type_id;

        secondary_ipc_->rx_release(secondary_ipc_, &msg);
    }

    // Helper to configure a cell and verify it reaches configured state
    void configure_cell(const uint16_t cell_id) {
        EXPECT_EQ(process_config_request(cell_id), SCF_ERROR_CODE_MSG_OK);
        drain_config_response();
        EXPECT_EQ(fapi_state_->get_cell_state(cell_id), rf::FapiStateT::FapiStateConfigured);
    }

    // Helper to configure and start a cell
    void configure_and_start_cell(const uint16_t cell_id) {
        configure_cell(cell_id);
        EXPECT_EQ(process_start_request(cell_id), SCF_ERROR_CODE_MSG_OK);
        EXPECT_EQ(fapi_state_->get_cell_state(cell_id), rf::FapiStateT::FapiStateRunning);
    }

    // Helper to configure, start, and stop a cell
    void configure_start_and_stop_cell(const uint16_t cell_id) {
        configure_and_start_cell(cell_id);
        EXPECT_EQ(process_stop_request(cell_id), SCF_ERROR_CODE_MSG_OK);
        EXPECT_EQ(fapi_state_->get_cell_state(cell_id), rf::FapiStateT::FapiStateStopped);
    }

    // Helper to set slot to a specific value
    void set_slot(const uint16_t sfn, const uint16_t slot, const uint16_t max_slot = 20) {
        fapi_state_->reset_slot();
        const int total_slots = sfn * max_slot + slot;
        for (int i = 0; i < total_slots; ++i) {
            fapi_state_->increment_slot();
        }

        const auto current = fapi_state_->get_current_slot();
        EXPECT_EQ(current.sfn, sfn);
        EXPECT_EQ(current.slot, slot);
    }

    // Helper to send a message from secondary Ipc
    struct SendMessageParams {
        uint16_t cell_id{};
        uint16_t msg_type{};
        uint32_t payload_size{0};
    };

    void send_message_from_secondary(const SendMessageParams &params) {
        nv_ipc_msg_t send_msg{};
        ASSERT_GE(secondary_ipc_->tx_allocate(secondary_ipc_, &send_msg, 0), 0);

        const std::size_t total_len =
                sizeof(scf_fapi_header_t) + sizeof(scf_fapi_body_header_t) + params.payload_size;
        const std::span<std::byte> send_buffer = rf::make_buffer_span(send_msg.msg_buf, total_len);

        auto *send_hdr = rf::assume_cast<scf_fapi_header_t>(send_buffer.data());
        send_hdr->message_count = 1;
        send_hdr->handle_id = static_cast<uint8_t>(params.cell_id);

        const std::span<std::byte> body_span = send_buffer.subspan(sizeof(scf_fapi_header_t));
        auto *send_body = rf::assume_cast<scf_fapi_body_header_t>(body_span.data());
        send_body->type_id = params.msg_type;
        send_body->length = params.payload_size;

        send_msg.msg_id = static_cast<int32_t>(params.msg_type);
        send_msg.cell_id = static_cast<int32_t>(params.cell_id);
        send_msg.msg_len = static_cast<int32_t>(total_len);
        send_msg.data_len = 0;

        ASSERT_GE(secondary_ipc_->tx_send_msg(secondary_ipc_, &send_msg), 0)
                << "tx_send_msg failed";
        secondary_ipc_->tx_tti_sem_post(secondary_ipc_);
    }

    // Helper struct for received messages
    struct ReceivedMessage {
        nv_ipc_msg_t msg{};
        scf_fapi_header_t *header{};
        scf_fapi_body_header_t *body_header{};
        void *body_payload{};
    };

    // Helper to receive and validate basic message structure
    [[nodiscard]] ReceivedMessage receive_and_validate_message(
            const uint16_t expected_cell_id, const uint16_t expected_msg_type) {
        ReceivedMessage result;
        const int ret = secondary_ipc_->rx_recv_msg(secondary_ipc_, &result.msg);
        EXPECT_GE(ret, 0) << "Failed to receive message type " << expected_msg_type;

        const std::span<std::byte> msg_buffer = rf::make_buffer_span(
                result.msg.msg_buf, static_cast<std::size_t>(result.msg.msg_len));

        result.header = rf::assume_cast<scf_fapi_header_t>(msg_buffer.data());
        EXPECT_EQ(result.header->message_count, 1);
        EXPECT_EQ(result.header->handle_id, expected_cell_id);

        const std::span<std::byte> body_span = msg_buffer.subspan(sizeof(scf_fapi_header_t));
        result.body_header = rf::assume_cast<scf_fapi_body_header_t>(body_span.data());
        EXPECT_EQ(result.body_header->type_id, expected_msg_type);

        result.body_payload = result.body_header;
        return result;
    }

    // Helper to release received message
    void release_received_message(ReceivedMessage &msg) {
        secondary_ipc_->rx_release(secondary_ipc_, &msg.msg);
    }

    // Helper to send an invalid message and expect it to be rejected
    template <typename ConfigureInvalidFn>
    void send_and_expect_invalid_message(
            const uint16_t expected_cell_id, ConfigureInvalidFn &&configure_invalid) {

        // Allocate message
        nv_ipc_msg_t send_msg{};
        ASSERT_GE(secondary_ipc_->tx_allocate(secondary_ipc_, &send_msg, 0), 0);

        const std::size_t msg_len = sizeof(scf_fapi_header_t) + sizeof(scf_fapi_body_header_t);
        const std::span<std::byte> send_buffer = rf::make_buffer_span(send_msg.msg_buf, msg_len);

        // Set up basic valid structure
        auto *send_hdr = rf::assume_cast<scf_fapi_header_t>(send_buffer.data());
        send_hdr->message_count = 1;
        send_hdr->handle_id = 0;

        const std::span<std::byte> body_span = send_buffer.subspan(sizeof(scf_fapi_header_t));
        auto *send_body = rf::assume_cast<scf_fapi_body_header_t>(body_span.data());
        send_body->type_id = SCF_FAPI_CONFIG_REQUEST;
        send_body->length = 0;

        send_msg.msg_id = SCF_FAPI_CONFIG_REQUEST;
        send_msg.cell_id = 0;
        send_msg.msg_len = sizeof(scf_fapi_header_t) + sizeof(scf_fapi_body_header_t);
        send_msg.data_len = 0;

        // Let the test customize what makes this message invalid
        std::forward<ConfigureInvalidFn>(configure_invalid)(*send_hdr, *send_body, send_msg);

        // Send the message
        ASSERT_GE(secondary_ipc_->tx_send_msg(secondary_ipc_, &send_msg), 0)
                << "tx_send_msg failed";
        secondary_ipc_->tx_tti_sem_post(secondary_ipc_);

        // Receive and process - should fail
        nv_ipc_msg_t recv_msg{};
        const int ret = fapi_state_->receive_message(recv_msg);
        ASSERT_GE(ret, 0);

        EXPECT_NE(fapi_state_->process_message(recv_msg), SCF_ERROR_CODE_MSG_OK);
        EXPECT_EQ(fapi_state_->get_cell_state(expected_cell_id), rf::FapiStateT::FapiStateIdle);

        fapi_state_->release_message(recv_msg);
    }
};

TEST_F(FapiStateNvIpcTest, BasicCommunication) {
    // Test basic slot info
    const auto slot = fapi_state_->get_current_slot();
    EXPECT_EQ(slot.sfn, 0);
    EXPECT_EQ(slot.slot, 0);

    // Test cell state queries
    EXPECT_EQ(fapi_state_->get_cell_state(0), rf::FapiStateT::FapiStateIdle);
    EXPECT_EQ(fapi_state_->get_num_cells_configured(), 0);
    EXPECT_EQ(fapi_state_->get_num_cells_running(), 0);
}

TEST_F(FapiStateNvIpcTest, SendSlotIndication) {
    constexpr uint16_t CELL_ID = 0;

    // Configure and start cell
    configure_and_start_cell(CELL_ID);

    // Set slot to a specific value
    set_slot(10, 5);

    // Send SLOT.indication using FapiState
    const bool send_result = fapi_state_->send_slot_indication();
    EXPECT_TRUE(send_result) << "Failed to send SLOT.indication";

    // Receive and validate on secondary side
    auto recv = receive_and_validate_message(CELL_ID, SCF_FAPI_SLOT_INDICATION);
    auto *recv_slot_ind = rf::assume_cast<scf_fapi_slot_ind_t>(recv.body_payload);
    EXPECT_EQ(recv_slot_ind->sfn, 10);
    EXPECT_EQ(recv_slot_ind->slot, 5);
    release_received_message(recv);
}

TEST_F(FapiStateNvIpcTest, ConfigResponseMessage) {
    constexpr uint16_t CELL_ID = 1;
    constexpr scf_fapi_error_codes_t ERROR_CODE = SCF_ERROR_CODE_MSG_OK;

    // Send a CONFIG.request which will trigger CONFIG.response
    EXPECT_EQ(process_config_request(CELL_ID), SCF_ERROR_CODE_MSG_OK);

    // Receive and validate CONFIG.response on secondary side
    auto recv = receive_and_validate_message(CELL_ID, SCF_FAPI_CONFIG_RESPONSE);
    auto *recv_cfg_rsp = rf::assume_cast<scf_fapi_config_response_msg_t>(recv.body_payload);
    EXPECT_EQ(recv_cfg_rsp->msg_body.error_code, ERROR_CODE);
    EXPECT_EQ(recv_cfg_rsp->msg_body.num_invalid_tlvs, 0);
    EXPECT_EQ(recv_cfg_rsp->msg_body.num_idle_only_tlvs, 0);
    EXPECT_EQ(recv_cfg_rsp->msg_body.num_running_only_tlvs, 0);
    EXPECT_EQ(recv_cfg_rsp->msg_body.num_missing_tlvs, 0);
    release_received_message(recv);
}

TEST(FapiStateFileInitTest, InitFromYamlFile) {
    // Test file-based initialization for backwards compatibility
    const std::filesystem::path temp_dir = std::filesystem::temp_directory_path();
    const std::filesystem::path config_file = temp_dir / "fapi_test_file_init.yaml";

    // Setup cleanup guard to ensure temp file is removed
    const auto cleanup =
            gsl_lite::finally([&config_file] { std::filesystem::remove(config_file); });

    // Write YAML to temp file
    const std::string yaml_content = rf::YamlConfigBuilder::create_primary_config("fapi_file_test");
    std::ofstream ofs(config_file);
    ofs << yaml_content;
    ofs.close();

    // Initialize from file
    rf::FapiState::InitParams params;
    params.nvipc_config_file = config_file.string();
    params.max_cells = 4;
    params.max_sfn = 1024;
    params.max_slot = 20;

    const rf::FapiState fapi_state(params);

    EXPECT_EQ(fapi_state.get_num_cells_configured(), 0);
    EXPECT_EQ(fapi_state.get_num_cells_running(), 0);
}

// ============================================================================
// SLOT COUNTER TESTS
// ============================================================================

TEST(FapiStateTest, SlotCounterIncrement) {
    auto fapi_state = create_fapi_state({"slot_test", 1});

    // Initial state
    auto slot = fapi_state->get_current_slot();
    EXPECT_EQ(slot.sfn, 0);
    EXPECT_EQ(slot.slot, 0);

    // Increment once
    fapi_state->increment_slot();
    slot = fapi_state->get_current_slot();
    EXPECT_EQ(slot.sfn, 0);
    EXPECT_EQ(slot.slot, 1);

    // Increment multiple times
    for (int i = 0; i < 10; ++i) {
        fapi_state->increment_slot();
    }
    slot = fapi_state->get_current_slot();
    EXPECT_EQ(slot.sfn, 0);
    EXPECT_EQ(slot.slot, 11);
}

TEST(FapiStateTest, SlotCounterSlotWraparound) {
    auto fapi_state = create_fapi_state({"slot_wrap_test", 1});

    // Advance to slot 19 (last slot)
    for (int i = 0; i < 19; ++i) {
        fapi_state->increment_slot();
    }

    auto slot = fapi_state->get_current_slot();
    EXPECT_EQ(slot.sfn, 0);
    EXPECT_EQ(slot.slot, 19);

    // Next increment should wrap slot to 0 and increment sfn
    fapi_state->increment_slot();
    slot = fapi_state->get_current_slot();
    EXPECT_EQ(slot.sfn, 1);
    EXPECT_EQ(slot.slot, 0);
}

TEST(FapiStateTest, SlotCounterSfnWraparound) {
    auto fapi_state = create_fapi_state({"sfn_wrap_test", 1});

    // Advance to sfn 1023, slot 19
    const int total_slots = 1023 * 20 + 19;
    for (int i = 0; i < total_slots; ++i) {
        fapi_state->increment_slot();
    }

    auto slot = fapi_state->get_current_slot();
    EXPECT_EQ(slot.sfn, 1023);
    EXPECT_EQ(slot.slot, 19);

    // Next increment should wrap both
    fapi_state->increment_slot();
    slot = fapi_state->get_current_slot();
    EXPECT_EQ(slot.sfn, 0);
    EXPECT_EQ(slot.slot, 0);
}

TEST(FapiStateTest, SlotCounterReset) {
    auto fapi_state = create_fapi_state({"reset_test", 1});

    // Advance to some arbitrary slot
    for (int i = 0; i < 55; ++i) {
        fapi_state->increment_slot();
    }

    auto slot = fapi_state->get_current_slot();
    EXPECT_NE(slot.sfn, 0);
    EXPECT_NE(slot.slot, 0);

    // Reset should bring back to (0, 0)
    fapi_state->reset_slot();
    slot = fapi_state->get_current_slot();
    EXPECT_EQ(slot.sfn, 0);
    EXPECT_EQ(slot.slot, 0);
}

TEST(FapiStateTest, SlotCounterCustomMaxValues) {
    auto fapi_state = create_fapi_state({"custom_max_test", 1, 512, 10});

    // Advance to slot 9 (last slot with max_slot = 10)
    for (int i = 0; i < 9; ++i) {
        fapi_state->increment_slot();
    }

    auto slot = fapi_state->get_current_slot();
    EXPECT_EQ(slot.sfn, 0);
    EXPECT_EQ(slot.slot, 9);

    // Next increment should wrap
    fapi_state->increment_slot();
    slot = fapi_state->get_current_slot();
    EXPECT_EQ(slot.sfn, 1);
    EXPECT_EQ(slot.slot, 0);

    // Test SFN wraparound at 512
    const int total_slots = 511 * 10 + 9;
    fapi_state->reset_slot();
    for (int i = 0; i < total_slots; ++i) {
        fapi_state->increment_slot();
    }

    slot = fapi_state->get_current_slot();
    EXPECT_EQ(slot.sfn, 511);
    EXPECT_EQ(slot.slot, 9);

    fapi_state->increment_slot();
    slot = fapi_state->get_current_slot();
    EXPECT_EQ(slot.sfn, 0);
    EXPECT_EQ(slot.slot, 0);
}

// ============================================================================
// STATE MACHINE INVALID TRANSITION TESTS
// ============================================================================

TEST_F(FapiStateNvIpcTest, InvalidTransition_StartWithoutConfig) {
    constexpr uint16_t CELL_ID = 0;

    // Cell is in idle state
    EXPECT_EQ(fapi_state_->get_cell_state(CELL_ID), rf::FapiStateT::FapiStateIdle);

    // Try to start without configuring (should fail)
    EXPECT_NE(process_start_request(CELL_ID), SCF_ERROR_CODE_MSG_OK);
    EXPECT_EQ(fapi_state_->get_cell_state(CELL_ID), rf::FapiStateT::FapiStateIdle);
    EXPECT_EQ(fapi_state_->get_num_cells_running(), 0);
}

TEST_F(FapiStateNvIpcTest, InvalidTransition_StopWithoutStart) {
    constexpr uint16_t CELL_ID = 0;

    // Configure cell
    configure_cell(CELL_ID);

    // Try to stop without starting (should fail)
    EXPECT_NE(process_stop_request(CELL_ID), SCF_ERROR_CODE_MSG_OK);
    EXPECT_EQ(fapi_state_->get_cell_state(CELL_ID), rf::FapiStateT::FapiStateConfigured);
    EXPECT_EQ(fapi_state_->get_num_cells_running(), 0);
}

TEST_F(FapiStateNvIpcTest, InvalidTransition_StopOnIdle) {
    constexpr uint16_t CELL_ID = 0;

    // Cell is in idle state
    EXPECT_EQ(fapi_state_->get_cell_state(CELL_ID), rf::FapiStateT::FapiStateIdle);

    // Try to stop idle cell (should fail)
    EXPECT_NE(process_stop_request(CELL_ID), SCF_ERROR_CODE_MSG_OK);
    EXPECT_EQ(fapi_state_->get_cell_state(CELL_ID), rf::FapiStateT::FapiStateIdle);
}

TEST_F(FapiStateNvIpcTest, InvalidTransition_ReconfigRunning) {
    constexpr uint16_t CELL_ID = 0;

    // Configure and start cell
    configure_and_start_cell(CELL_ID);

    // Try to reconfigure while running (should fail)
    EXPECT_NE(process_config_request(CELL_ID), SCF_ERROR_CODE_MSG_OK);
    EXPECT_EQ(fapi_state_->get_cell_state(CELL_ID), rf::FapiStateT::FapiStateRunning);
    EXPECT_EQ(fapi_state_->get_num_cells_configured(), 1);
}

TEST_F(FapiStateNvIpcTest, InvalidTransition_ReconfigConfigured) {
    constexpr uint16_t CELL_ID = 0;

    // Configure cell
    configure_cell(CELL_ID);
    EXPECT_EQ(fapi_state_->get_num_cells_configured(), 1);

    // Try to reconfigure again (should be idempotent)
    EXPECT_EQ(process_config_request(CELL_ID), SCF_ERROR_CODE_MSG_OK);
    drain_config_response();
    EXPECT_EQ(fapi_state_->get_cell_state(CELL_ID), rf::FapiStateT::FapiStateConfigured);
    EXPECT_EQ(fapi_state_->get_num_cells_configured(), 1); // Should not increment
}

TEST_F(FapiStateNvIpcTest, InvalidTransition_DoubleStart) {
    constexpr uint16_t CELL_ID = 0;

    // Configure and start cell
    configure_and_start_cell(CELL_ID);
    EXPECT_EQ(fapi_state_->get_num_cells_running(), 1);

    // Try to start again (should fail - strict state machine)
    EXPECT_NE(process_start_request(CELL_ID), SCF_ERROR_CODE_MSG_OK);
    EXPECT_EQ(fapi_state_->get_cell_state(CELL_ID), rf::FapiStateT::FapiStateRunning);
    EXPECT_EQ(fapi_state_->get_num_cells_running(), 1); // Should not increment
}

TEST_F(FapiStateNvIpcTest, InvalidTransition_DoubleStop) {
    constexpr uint16_t CELL_ID = 0;

    // Configure, start, and stop cell
    configure_start_and_stop_cell(CELL_ID);

    // Try to stop again (should fail - strict state machine)
    EXPECT_NE(process_stop_request(CELL_ID), SCF_ERROR_CODE_MSG_OK);
    EXPECT_EQ(fapi_state_->get_cell_state(CELL_ID), rf::FapiStateT::FapiStateStopped);
}

TEST_F(FapiStateNvIpcTest, StateTransition_FullCycle) {
    constexpr uint16_t CELL_ID = 0;

    // Initial: IDLE
    EXPECT_EQ(fapi_state_->get_cell_state(CELL_ID), rf::FapiStateT::FapiStateIdle);
    EXPECT_EQ(fapi_state_->get_num_cells_configured(), 0);
    EXPECT_EQ(fapi_state_->get_num_cells_running(), 0);

    // IDLE → CONFIGURED
    configure_cell(CELL_ID);
    EXPECT_EQ(fapi_state_->get_num_cells_configured(), 1);
    EXPECT_EQ(fapi_state_->get_num_cells_running(), 0);

    // CONFIGURED → RUNNING
    EXPECT_EQ(process_start_request(CELL_ID), SCF_ERROR_CODE_MSG_OK);
    EXPECT_EQ(fapi_state_->get_cell_state(CELL_ID), rf::FapiStateT::FapiStateRunning);
    EXPECT_EQ(fapi_state_->get_num_cells_configured(), 1);
    EXPECT_EQ(fapi_state_->get_num_cells_running(), 1);

    // RUNNING → STOPPED
    EXPECT_EQ(process_stop_request(CELL_ID), SCF_ERROR_CODE_MSG_OK);
    EXPECT_EQ(fapi_state_->get_cell_state(CELL_ID), rf::FapiStateT::FapiStateStopped);
    EXPECT_EQ(fapi_state_->get_num_cells_configured(), 1); // Still configured
    EXPECT_EQ(fapi_state_->get_num_cells_running(), 0);    // No longer running
}

TEST_F(FapiStateNvIpcTest, StateTransition_MultipleStopStartCycles) {
    constexpr uint16_t CELL_ID = 0;

    // Initial config and first start/stop cycle
    configure_start_and_stop_cell(CELL_ID);
    EXPECT_EQ(fapi_state_->get_num_cells_running(), 0);

    // Note: After STOP, cell is in STOPPED state, not CONFIGURED
    // Cannot restart from STOPPED without reconfiguring
    EXPECT_EQ(fapi_state_->get_cell_state(CELL_ID), rf::FapiStateT::FapiStateStopped);
}

// ============================================================================
// MULTI-CELL TESTS
// ============================================================================

TEST_F(FapiStateNvIpcTest, MultiCell_IndependentStateMachines) {
    // Cell 0: IDLE
    EXPECT_EQ(fapi_state_->get_cell_state(0), rf::FapiStateT::FapiStateIdle);

    // Cell 1: CONFIGURED
    configure_cell(1);

    // Cell 2: RUNNING
    configure_and_start_cell(2);

    // Cell 3: STOPPED
    configure_start_and_stop_cell(3);

    // Verify each cell maintained independent state
    EXPECT_EQ(fapi_state_->get_cell_state(0), rf::FapiStateT::FapiStateIdle);
    EXPECT_EQ(fapi_state_->get_cell_state(1), rf::FapiStateT::FapiStateConfigured);
    EXPECT_EQ(fapi_state_->get_cell_state(2), rf::FapiStateT::FapiStateRunning);
    EXPECT_EQ(fapi_state_->get_cell_state(3), rf::FapiStateT::FapiStateStopped);

    // Verify counters
    EXPECT_EQ(fapi_state_->get_num_cells_configured(), 3); // Cells 1, 2, 3
    EXPECT_EQ(fapi_state_->get_num_cells_running(), 1);    // Only cell 2
}

TEST_F(FapiStateNvIpcTest, MultiCell_SlotIndicationOnlyToRunning) {
    // Configure and start cells 0 and 2
    configure_and_start_cell(0);
    configure_and_start_cell(2);

    // Cell 1 is configured but not started
    configure_cell(1);

    // Set slot to known value
    set_slot(1, 5);

    // Send slot indication
    const bool send_result = fapi_state_->send_slot_indication();
    EXPECT_TRUE(send_result);

    // SLOT.indication is sent once per slot to the first running cell (cell 0)
    // Not sent individually to each running cell
    auto recv = receive_and_validate_message(0, SCF_FAPI_SLOT_INDICATION);
    auto *recv_slot_ind = rf::assume_cast<scf_fapi_slot_ind_t>(recv.body_payload);
    EXPECT_EQ(recv_slot_ind->sfn, 1);
    EXPECT_EQ(recv_slot_ind->slot, 5);
    release_received_message(recv);
}

TEST_F(FapiStateNvIpcTest, MultiCell_CapacityLimit) {
    // Configure all 4 cells (max_cells = 4)
    for (uint16_t cell_id = 0; cell_id < 4; ++cell_id) {
        configure_cell(cell_id);
    }

    EXPECT_EQ(fapi_state_->get_num_cells_configured(), 4);

    // Try to configure beyond capacity (should fail or be rejected)
    // Note: Current implementation doesn't track "active" cells separately from max_cells
    // but validates cell_id bounds
}

TEST_F(FapiStateNvIpcTest, MultiCell_InvalidCellIdBeyondMax) {
    constexpr uint16_t INVALID_CELL_ID = 100; // Way beyond max_cells = 4

    // Try operations with invalid cell ID
    // These should fail for invalid cell IDs
    EXPECT_NE(process_config_request(INVALID_CELL_ID), SCF_ERROR_CODE_MSG_OK);
    EXPECT_NE(process_start_request(INVALID_CELL_ID), SCF_ERROR_CODE_MSG_OK);
    EXPECT_NE(process_stop_request(INVALID_CELL_ID), SCF_ERROR_CODE_MSG_OK);

    // get_cell_state should return safe default
    EXPECT_EQ(fapi_state_->get_cell_state(INVALID_CELL_ID), rf::FapiStateT::FapiStateIdle);
}

TEST_F(FapiStateNvIpcTest, MultiCell_BoundaryValidation) {
    // Test at boundary: cell_id = max_cells - 1 (should succeed)
    constexpr uint16_t LAST_VALID_CELL = 3; // max_cells = 4, so last valid is 3
    configure_cell(LAST_VALID_CELL);

    // Test at boundary: cell_id = max_cells (should fail)
    constexpr uint16_t FIRST_INVALID_CELL = 4;
    EXPECT_NE(process_config_request(FIRST_INVALID_CELL), SCF_ERROR_CODE_MSG_OK);
    // Invalid cell should remain idle
    EXPECT_EQ(fapi_state_->get_cell_state(FIRST_INVALID_CELL), rf::FapiStateT::FapiStateIdle);
}

TEST_F(FapiStateNvIpcTest, MultiCell_CounterTracking) {
    EXPECT_EQ(fapi_state_->get_num_cells_configured(), 0);
    EXPECT_EQ(fapi_state_->get_num_cells_running(), 0);

    // Configure cells 0 and 1
    configure_cell(0);
    EXPECT_EQ(fapi_state_->get_num_cells_configured(), 1);

    configure_cell(1);
    EXPECT_EQ(fapi_state_->get_num_cells_configured(), 2);
    EXPECT_EQ(fapi_state_->get_num_cells_running(), 0);

    // Start cell 0
    EXPECT_EQ(process_start_request(0), SCF_ERROR_CODE_MSG_OK);
    EXPECT_EQ(fapi_state_->get_num_cells_configured(), 2);
    EXPECT_EQ(fapi_state_->get_num_cells_running(), 1);

    // Start cell 1
    EXPECT_EQ(process_start_request(1), SCF_ERROR_CODE_MSG_OK);
    EXPECT_EQ(fapi_state_->get_num_cells_configured(), 2);
    EXPECT_EQ(fapi_state_->get_num_cells_running(), 2);

    // Stop cell 0
    EXPECT_EQ(process_stop_request(0), SCF_ERROR_CODE_MSG_OK);
    EXPECT_EQ(fapi_state_->get_num_cells_configured(), 2); // Still configured
    EXPECT_EQ(fapi_state_->get_num_cells_running(), 1);    // One less running

    // Stop cell 1
    EXPECT_EQ(process_stop_request(1), SCF_ERROR_CODE_MSG_OK);
    EXPECT_EQ(fapi_state_->get_num_cells_configured(), 2);
    EXPECT_EQ(fapi_state_->get_num_cells_running(), 0);
}

TEST_F(FapiStateNvIpcTest, MultiCell_InterleavedOperations) {
    // Interleaved config and start operations
    configure_and_start_cell(0);

    configure_cell(1);

    configure_and_start_cell(2);

    EXPECT_EQ(process_start_request(1), SCF_ERROR_CODE_MSG_OK);
    EXPECT_EQ(fapi_state_->get_cell_state(1), rf::FapiStateT::FapiStateRunning);

    // Verify all cells are in expected state
    EXPECT_EQ(fapi_state_->get_num_cells_configured(), 3);
    EXPECT_EQ(fapi_state_->get_num_cells_running(), 3);
}

// ============================================================================
// CALLBACK TESTS
// ============================================================================

TEST_F(FapiStateNvIpcTest, Callback_NotSetNoCallback) {
    // Process message without setting callbacks (should not crash)
    // This is tested implicitly in other tests, but explicit test for clarity

    // Create and send a UL_TTI_REQUEST from secondary
    send_message_from_secondary({0, SCF_FAPI_UL_TTI_REQUEST});

    // Receive and process on primary (FapiState side)
    nv_ipc_msg_t recv_msg{};
    const int ret = fapi_state_->receive_message(recv_msg);
    ASSERT_GE(ret, 0);

    // Should process without crashing even though callback not set
    EXPECT_EQ(fapi_state_->process_message(recv_msg), SCF_ERROR_CODE_MSG_OK);

    fapi_state_->release_message(recv_msg);
}

TEST_F(FapiStateNvIpcTest, Callback_UlTtiRequest) {
    bool callback_invoked = false;
    uint16_t callback_cell_id = 0xFFFF;
    uint16_t callback_type_id = 0;

    // Set callback
    fapi_state_->set_on_ul_tti_request([&callback_invoked, &callback_cell_id, &callback_type_id](
                                               const uint16_t cell_id,
                                               const scf_fapi_body_header_t &body_hdr,
                                               [[maybe_unused]] const uint32_t body_len) {
        callback_invoked = true;
        callback_cell_id = cell_id;
        callback_type_id = body_hdr.type_id;
    });

    // Create and send UL_TTI_REQUEST from secondary
    send_message_from_secondary({0, SCF_FAPI_UL_TTI_REQUEST});

    // Receive and process
    nv_ipc_msg_t recv_msg{};
    const int ret = fapi_state_->receive_message(recv_msg);
    ASSERT_GE(ret, 0);

    EXPECT_EQ(fapi_state_->process_message(recv_msg), SCF_ERROR_CODE_MSG_OK);

    // Verify callback was invoked
    EXPECT_TRUE(callback_invoked);
    EXPECT_EQ(callback_cell_id, 0);
    EXPECT_EQ(callback_type_id, SCF_FAPI_UL_TTI_REQUEST);

    fapi_state_->release_message(recv_msg);
}

TEST_F(FapiStateNvIpcTest, Callback_DlTtiRequest) {
    bool callback_invoked = false;
    uint16_t callback_cell_id = 0xFFFF;

    // Set callback
    fapi_state_->set_on_dl_tti_request(
            [&callback_invoked, &callback_cell_id](
                    const uint16_t cell_id,
                    [[maybe_unused]] const scf_fapi_body_header_t &body_hdr,
                    [[maybe_unused]] const uint32_t body_len) {
                callback_invoked = true;
                callback_cell_id = cell_id;
            });

    // Create and send DL_TTI_REQUEST from secondary
    send_message_from_secondary({1, SCF_FAPI_DL_TTI_REQUEST});

    // Receive and process
    nv_ipc_msg_t recv_msg{};
    const int ret = fapi_state_->receive_message(recv_msg);
    ASSERT_GE(ret, 0);

    EXPECT_EQ(fapi_state_->process_message(recv_msg), SCF_ERROR_CODE_MSG_OK);

    // Verify callback was invoked
    EXPECT_TRUE(callback_invoked);
    EXPECT_EQ(callback_cell_id, 1);

    fapi_state_->release_message(recv_msg);
}

TEST_F(FapiStateNvIpcTest, Callback_SlotResponse) {
    int callback_count = 0;

    // Set callback
    fapi_state_->set_on_slot_response(
            [&callback_count](
                    [[maybe_unused]] const uint16_t cell_id,
                    [[maybe_unused]] const scf_fapi_body_header_t &body_hdr,
                    [[maybe_unused]] const uint32_t body_len) { callback_count++; });

    // Create and send SLOT_RESPONSE from secondary
    nv_ipc_msg_t send_msg{};
    ASSERT_GE(secondary_ipc_->tx_allocate(secondary_ipc_, &send_msg, 0), 0);

    const std::size_t total_len = sizeof(scf_fapi_header_t) + sizeof(scf_fapi_slot_rsp_t);
    const std::span<std::byte> send_buffer = rf::make_buffer_span(send_msg.msg_buf, total_len);

    auto *send_hdr = rf::assume_cast<scf_fapi_header_t>(send_buffer.data());
    send_hdr->message_count = 1;
    send_hdr->handle_id = 0;

    const std::span<std::byte> body_span = send_buffer.subspan(sizeof(scf_fapi_header_t));
    auto *send_body_hdr = rf::assume_cast<scf_fapi_body_header_t>(body_span.data());
    send_body_hdr->type_id = SCF_FAPI_SLOT_RESPONSE;
    send_body_hdr->length = sizeof(scf_fapi_slot_rsp_t) - sizeof(scf_fapi_body_header_t);

    auto *send_body = rf::assume_cast<scf_fapi_slot_rsp_t>(body_span.data());
    send_body->sfn = 10;
    send_body->slot = 5;

    send_msg.msg_id = SCF_FAPI_SLOT_RESPONSE;
    send_msg.cell_id = 0;
    send_msg.msg_len = sizeof(scf_fapi_header_t) + sizeof(scf_fapi_slot_rsp_t);
    send_msg.data_len = 0;

    ASSERT_GE(secondary_ipc_->tx_send_msg(secondary_ipc_, &send_msg), 0) << "tx_send_msg failed";
    secondary_ipc_->tx_tti_sem_post(secondary_ipc_);

    // Receive and process
    nv_ipc_msg_t recv_msg{};
    const int ret = fapi_state_->receive_message(recv_msg);
    ASSERT_GE(ret, 0);

    EXPECT_EQ(callback_count, 0);
    EXPECT_EQ(fapi_state_->process_message(recv_msg), SCF_ERROR_CODE_MSG_OK);
    EXPECT_EQ(callback_count, 1);

    fapi_state_->release_message(recv_msg);
}

TEST_F(FapiStateNvIpcTest, Callback_SetMultipleTimes) {
    int callback_count_a = 0;
    int callback_count_b = 0;

    // Set first callback
    fapi_state_->set_on_ul_tti_request(
            [&callback_count_a](
                    [[maybe_unused]] const uint16_t cell_id,
                    [[maybe_unused]] const scf_fapi_body_header_t &body_hdr,
                    [[maybe_unused]] const uint32_t body_len) { callback_count_a++; });

    // Override with second callback
    fapi_state_->set_on_ul_tti_request(
            [&callback_count_b](
                    [[maybe_unused]] const uint16_t cell_id,
                    [[maybe_unused]] const scf_fapi_body_header_t &body_hdr,
                    [[maybe_unused]] const uint32_t body_len) { callback_count_b++; });

    // Send UL_TTI_REQUEST
    send_message_from_secondary({0, SCF_FAPI_UL_TTI_REQUEST});

    // Receive and process
    nv_ipc_msg_t recv_msg{};
    const int ret = fapi_state_->receive_message(recv_msg);
    ASSERT_GE(ret, 0);

    EXPECT_EQ(fapi_state_->process_message(recv_msg), SCF_ERROR_CODE_MSG_OK);

    // Only second callback should be invoked
    EXPECT_EQ(callback_count_a, 0);
    EXPECT_EQ(callback_count_b, 1);

    fapi_state_->release_message(recv_msg);
}

TEST_F(FapiStateNvIpcTest, Callback_StateRequestCallbacks) {
    constexpr uint16_t CELL_ID = 0;

    bool config_called = false;
    bool start_called = false;
    bool stop_called = false;
    uint16_t config_cell_id = 0xFFFF;
    uint16_t start_cell_id = 0xFFFF;
    uint16_t stop_cell_id = 0xFFFF;
    uint16_t config_msg_type = 0;
    uint16_t start_msg_type = 0;
    uint16_t stop_msg_type = 0;

    // Set all three callbacks - they return OK to allow state transitions
    fapi_state_->set_on_config_request([&config_called, &config_cell_id, &config_msg_type](
                                               const uint16_t cell_id,
                                               const scf_fapi_body_header_t &body_hdr,
                                               [[maybe_unused]] const uint32_t body_len) {
        config_called = true;
        config_cell_id = cell_id;
        config_msg_type = body_hdr.type_id;
        return SCF_ERROR_CODE_MSG_OK;
    });

    fapi_state_->set_on_start_request([&start_called, &start_cell_id, &start_msg_type](
                                              const uint16_t cell_id,
                                              const scf_fapi_body_header_t &body_hdr,
                                              [[maybe_unused]] const uint32_t body_len) {
        start_called = true;
        start_cell_id = cell_id;
        start_msg_type = body_hdr.type_id;
        return SCF_ERROR_CODE_MSG_OK;
    });

    fapi_state_->set_on_stop_request([&stop_called, &stop_cell_id, &stop_msg_type](
                                             const uint16_t cell_id,
                                             const scf_fapi_body_header_t &body_hdr,
                                             [[maybe_unused]] const uint32_t body_len) {
        stop_called = true;
        stop_cell_id = cell_id;
        stop_msg_type = body_hdr.type_id;
        return SCF_ERROR_CODE_MSG_OK;
    });

    // Test CONFIG.request callback
    EXPECT_FALSE(config_called);
    EXPECT_EQ(process_config_request(CELL_ID), SCF_ERROR_CODE_MSG_OK);
    drain_config_response();
    EXPECT_TRUE(config_called);
    EXPECT_EQ(config_cell_id, CELL_ID);
    EXPECT_EQ(config_msg_type, SCF_FAPI_CONFIG_REQUEST);
    EXPECT_EQ(fapi_state_->get_cell_state(CELL_ID), rf::FapiStateT::FapiStateConfigured);

    // Test START.request callback
    EXPECT_FALSE(start_called);
    EXPECT_EQ(process_start_request(CELL_ID), SCF_ERROR_CODE_MSG_OK);
    EXPECT_TRUE(start_called);
    EXPECT_EQ(start_cell_id, CELL_ID);
    EXPECT_EQ(start_msg_type, SCF_FAPI_START_REQUEST);
    EXPECT_EQ(fapi_state_->get_cell_state(CELL_ID), rf::FapiStateT::FapiStateRunning);

    // Test STOP.request callback
    EXPECT_FALSE(stop_called);
    EXPECT_EQ(process_stop_request(CELL_ID), SCF_ERROR_CODE_MSG_OK);
    EXPECT_TRUE(stop_called);
    EXPECT_EQ(stop_cell_id, CELL_ID);
    EXPECT_EQ(stop_msg_type, SCF_FAPI_STOP_REQUEST);
    EXPECT_EQ(fapi_state_->get_cell_state(CELL_ID), rf::FapiStateT::FapiStateStopped);
}

TEST_F(FapiStateNvIpcTest, Callback_StateRequestCallbacksRejectRequests) {
    constexpr uint16_t CELL_ID = 1;

    // Set callbacks that reject the requests with error codes
    fapi_state_->set_on_config_request([]([[maybe_unused]] const uint16_t cell_id,
                                          [[maybe_unused]] const scf_fapi_body_header_t &body_hdr,
                                          [[maybe_unused]] const uint32_t body_len) {
        return SCF_ERROR_CODE_MSG_INVALID_CONFIG;
    });

    fapi_state_->set_on_start_request([]([[maybe_unused]] const uint16_t cell_id,
                                         [[maybe_unused]] const scf_fapi_body_header_t &body_hdr,
                                         [[maybe_unused]] const uint32_t body_len) {
        return SCF_ERROR_CODE_MSG_INVALID_STATE;
    });

    fapi_state_->set_on_stop_request([]([[maybe_unused]] const uint16_t cell_id,
                                        [[maybe_unused]] const scf_fapi_body_header_t &body_hdr,
                                        [[maybe_unused]] const uint32_t body_len) {
        return SCF_ERROR_CODE_MSG_INVALID_STATE;
    });

    // CONFIG.request should be rejected by callback
    EXPECT_EQ(process_config_request(CELL_ID), SCF_ERROR_CODE_MSG_INVALID_CONFIG);
    EXPECT_EQ(fapi_state_->get_cell_state(CELL_ID), rf::FapiStateT::FapiStateIdle);

    // Configure without callback rejection (temporarily clear callback)
    fapi_state_->set_on_config_request(
            []([[maybe_unused]] const uint16_t cell_id,
               [[maybe_unused]] const scf_fapi_body_header_t &body_hdr,
               [[maybe_unused]] const uint32_t body_len) { return SCF_ERROR_CODE_MSG_OK; });
    configure_cell(CELL_ID);

    // Restore rejecting callback for START
    fapi_state_->set_on_start_request([]([[maybe_unused]] const uint16_t cell_id,
                                         [[maybe_unused]] const scf_fapi_body_header_t &body_hdr,
                                         [[maybe_unused]] const uint32_t body_len) {
        return SCF_ERROR_CODE_MSG_INVALID_STATE;
    });

    // START.request should be rejected by callback
    EXPECT_EQ(process_start_request(CELL_ID), SCF_ERROR_CODE_MSG_INVALID_STATE);
    EXPECT_EQ(fapi_state_->get_cell_state(CELL_ID), rf::FapiStateT::FapiStateConfigured);

    // Start without callback rejection
    fapi_state_->set_on_start_request(
            []([[maybe_unused]] const uint16_t cell_id,
               [[maybe_unused]] const scf_fapi_body_header_t &body_hdr,
               [[maybe_unused]] const uint32_t body_len) { return SCF_ERROR_CODE_MSG_OK; });
    EXPECT_EQ(process_start_request(CELL_ID), SCF_ERROR_CODE_MSG_OK);
    EXPECT_EQ(fapi_state_->get_cell_state(CELL_ID), rf::FapiStateT::FapiStateRunning);

    // Restore rejecting callback for STOP
    fapi_state_->set_on_stop_request([]([[maybe_unused]] const uint16_t cell_id,
                                        [[maybe_unused]] const scf_fapi_body_header_t &body_hdr,
                                        [[maybe_unused]] const uint32_t body_len) {
        return SCF_ERROR_CODE_MSG_INVALID_STATE;
    });

    // STOP.request should be rejected by callback
    EXPECT_EQ(process_stop_request(CELL_ID), SCF_ERROR_CODE_MSG_INVALID_STATE);
    EXPECT_EQ(fapi_state_->get_cell_state(CELL_ID), rf::FapiStateT::FapiStateRunning);
}

// ============================================================================
// ERROR INDICATION TESTS
// ============================================================================

TEST_F(FapiStateNvIpcTest, ErrorIndication_InvalidStartState) {
    constexpr uint16_t CELL_ID = 0;

    // Cell is in idle state, START should fail
    EXPECT_EQ(fapi_state_->get_cell_state(CELL_ID), rf::FapiStateT::FapiStateIdle);

    // Try to start without configuring (will fail internally)
    EXPECT_NE(process_start_request(CELL_ID), SCF_ERROR_CODE_MSG_OK);

    // Send ERROR.indication for testing
    const bool send_result = fapi_state_->send_error_indication(
            CELL_ID, SCF_FAPI_START_REQUEST, SCF_ERROR_CODE_MSG_INVALID_STATE);
    EXPECT_TRUE(send_result);

    // Receive and validate ERROR.indication
    auto recv = receive_and_validate_message(CELL_ID, SCF_FAPI_ERROR_INDICATION);
    auto *err_ind = rf::assume_cast<scf_fapi_error_ind_t>(recv.body_payload);
    EXPECT_EQ(err_ind->msg_id, SCF_FAPI_START_REQUEST);
    EXPECT_EQ(err_ind->err_code, SCF_ERROR_CODE_MSG_INVALID_STATE);

    // Verify current slot is included
    const auto current_slot = fapi_state_->get_current_slot();
    EXPECT_EQ(err_ind->sfn, current_slot.sfn);
    EXPECT_EQ(err_ind->slot, current_slot.slot);

    release_received_message(recv);
}

TEST_F(FapiStateNvIpcTest, ErrorIndication_InvalidStopState) {
    constexpr uint16_t CELL_ID = 1;

    // Configure cell (but don't start)
    configure_cell(CELL_ID);

    // Try to stop without starting (will fail internally)
    EXPECT_NE(process_stop_request(CELL_ID), SCF_ERROR_CODE_MSG_OK);

    // Send ERROR.indication for testing
    const bool send_result = fapi_state_->send_error_indication(
            CELL_ID, SCF_FAPI_STOP_REQUEST, SCF_ERROR_CODE_MSG_INVALID_STATE);
    EXPECT_TRUE(send_result);

    // Receive and validate
    auto recv = receive_and_validate_message(CELL_ID, SCF_FAPI_ERROR_INDICATION);
    auto *err_ind = rf::assume_cast<scf_fapi_error_ind_t>(recv.body_payload);
    EXPECT_EQ(err_ind->msg_id, SCF_FAPI_STOP_REQUEST);
    EXPECT_EQ(err_ind->err_code, SCF_ERROR_CODE_MSG_INVALID_STATE);

    release_received_message(recv);
}

TEST_F(FapiStateNvIpcTest, ErrorIndication_MessageFormat) {
    constexpr uint16_t CELL_ID = 2;

    // Set slot to specific value
    set_slot(6, 3); // 123 = 6 * 20 + 3
    const auto expected_slot = fapi_state_->get_current_slot();

    // Send ERROR.indication with specific parameters
    const bool send_result = fapi_state_->send_error_indication(
            CELL_ID, SCF_FAPI_CONFIG_REQUEST, SCF_ERROR_CODE_MSG_INVALID_CONFIG);
    EXPECT_TRUE(send_result);

    // Receive and validate all fields
    auto recv = receive_and_validate_message(CELL_ID, SCF_FAPI_ERROR_INDICATION);

    const std::size_t expected_body_len =
            sizeof(scf_fapi_error_ind_t) - sizeof(scf_fapi_body_header_t);
    EXPECT_EQ(recv.body_header->length, expected_body_len);

    auto *err_ind = rf::assume_cast<scf_fapi_error_ind_t>(recv.body_payload);
    EXPECT_EQ(err_ind->sfn, expected_slot.sfn);
    EXPECT_EQ(err_ind->slot, expected_slot.slot);
    EXPECT_EQ(err_ind->msg_id, SCF_FAPI_CONFIG_REQUEST);
    EXPECT_EQ(err_ind->err_code, SCF_ERROR_CODE_MSG_INVALID_CONFIG);

    release_received_message(recv);
}

// ============================================================================
// STOP INDICATION TESTS
// ============================================================================

TEST_F(FapiStateNvIpcTest, StopIndication_MessageFormat) {
    constexpr uint16_t CELL_ID = 0;

    // Send STOP.indication
    const bool send_result = fapi_state_->send_stop_indication(CELL_ID);
    EXPECT_TRUE(send_result);

    // Receive and validate
    auto recv = receive_and_validate_message(CELL_ID, SCF_FAPI_STOP_INDICATION);
    EXPECT_EQ(recv.body_header->length, 0); // STOP.indication has no body
    release_received_message(recv);
}

TEST_F(FapiStateNvIpcTest, StopIndication_AfterValidStop) {
    constexpr uint16_t CELL_ID = 0;

    // Configure, start, then stop cell
    configure_start_and_stop_cell(CELL_ID);

    // Send STOP.indication
    const bool send_result = fapi_state_->send_stop_indication(CELL_ID);
    EXPECT_TRUE(send_result);

    // Verify it was sent
    auto recv = receive_and_validate_message(CELL_ID, SCF_FAPI_STOP_INDICATION);
    release_received_message(recv);
}

// ============================================================================
// MALFORMED/INVALID PACKET TESTS
// ============================================================================

TEST_F(FapiStateNvIpcTest, InvalidPacket_MessageCountZero) {
    send_and_expect_invalid_message(
            0, [](scf_fapi_header_t &hdr, scf_fapi_body_header_t &, nv_ipc_msg_t &) {
                hdr.message_count = 0; // Invalid - must be 1
            });
}

TEST_F(FapiStateNvIpcTest, InvalidPacket_MessageCountMultiple) {
    send_and_expect_invalid_message(
            0, [](scf_fapi_header_t &hdr, scf_fapi_body_header_t &, nv_ipc_msg_t &) {
                hdr.message_count = 5; // Invalid - should be 1
            });
}

TEST_F(FapiStateNvIpcTest, InvalidPacket_HandleIdMismatch) {
    send_and_expect_invalid_message(
            0, [](scf_fapi_header_t &hdr, scf_fapi_body_header_t &, nv_ipc_msg_t &) {
                hdr.handle_id = 5; // Mismatched with msg.cell_id = 0
            });
}

TEST_F(FapiStateNvIpcTest, InvalidPacket_LengthMismatch) {
    send_and_expect_invalid_message(
            0, [](scf_fapi_header_t &, scf_fapi_body_header_t &body, nv_ipc_msg_t &msg) {
                body.length = 100; // Claims 100 bytes
                msg.msg_len = sizeof(scf_fapi_header_t) + sizeof(scf_fapi_body_header_t) +
                              50; // Mismatch!
            });
}

TEST_F(FapiStateNvIpcTest, InvalidPacket_UnknownMessageType) {
    send_and_expect_invalid_message(
            0, [](scf_fapi_header_t &, scf_fapi_body_header_t &body, nv_ipc_msg_t &msg) {
                constexpr uint16_t UNKNOWN_TYPE = 9999;
                body.type_id = UNKNOWN_TYPE;
                msg.msg_id = static_cast<int32_t>(UNKNOWN_TYPE);
            });
}

// ============================================================================
// CONFIG TLV PARSING TESTS
// ============================================================================

TEST_F(FapiStateNvIpcTest, ConfigTlv_EmptyConfig) {
    constexpr uint16_t CELL_ID = 0;

    // Create CONFIG.request with no TLVs
    EXPECT_EQ(process_config_request(CELL_ID), SCF_ERROR_CODE_MSG_OK);
    drain_config_response();
    EXPECT_EQ(fapi_state_->get_cell_state(CELL_ID), rf::FapiStateT::FapiStateConfigured);
}

TEST_F(FapiStateNvIpcTest, StateQuery_InvalidCellId) {
    // Query state for cell beyond max_cells
    const auto state = fapi_state_->get_cell_state(100);

    // Should return safe default (Idle)
    EXPECT_EQ(state, rf::FapiStateT::FapiStateIdle);
}

TEST_F(FapiStateNvIpcTest, StateQuery_AllStates) {
    // Cell 0: IDLE
    EXPECT_EQ(fapi_state_->get_cell_state(0), rf::FapiStateT::FapiStateIdle);

    // Cell 1: CONFIGURED
    configure_cell(1);

    // Cell 2: RUNNING
    configure_and_start_cell(2);

    // Cell 3: STOPPED
    configure_start_and_stop_cell(3);
}

// ============================================================================
// ON_MESSAGE_ CALLBACK TESTS (Generic message capture)
// ============================================================================

TEST_F(FapiStateNvIpcTest, Callback_OnMessageBasic) {
    bool callback_invoked = false;
    uint16_t captured_cell_id = 0xFFFF;
    uint16_t captured_msg_id = 0xFFFF;
    std::size_t captured_msg_buf_size = 0;
    std::size_t captured_data_buf_size = 0;

    // Set on_message_ callback
    fapi_state_->set_on_message([&callback_invoked,
                                 &captured_cell_id,
                                 &captured_msg_id,
                                 &captured_msg_buf_size,
                                 &captured_data_buf_size](const rf::FapiMessageData &msg) {
        callback_invoked = true;
        captured_cell_id = msg.cell_id;
        captured_msg_id = msg.msg_id;
        captured_msg_buf_size = msg.msg_buf.size();
        captured_data_buf_size = msg.data_buf.size();
    });

    // Send CONFIG.request
    EXPECT_EQ(process_config_request(0), SCF_ERROR_CODE_MSG_OK);

    // Verify callback was invoked with correct data
    EXPECT_TRUE(callback_invoked);
    EXPECT_EQ(captured_cell_id, 0);
    EXPECT_EQ(captured_msg_id, SCF_FAPI_CONFIG_REQUEST);
    EXPECT_GT(captured_msg_buf_size, 0);
    EXPECT_EQ(captured_data_buf_size, 0); // CONFIG.request has no data buffer

    drain_config_response();
}

TEST_F(FapiStateNvIpcTest, Callback_OnMessageCalledBeforeSpecificCallbacks) {
    std::vector<std::string> execution_order;

    // Set on_message_ callback
    fapi_state_->set_on_message([&execution_order](const rf::FapiMessageData &) {
        execution_order.emplace_back("on_message");
    });

    // Set on_ul_tti_request_ callback
    fapi_state_->set_on_ul_tti_request(
            [&execution_order](const uint16_t, const scf_fapi_body_header_t &, const uint32_t) {
                execution_order.emplace_back("on_ul_tti_request");
            });

    // Send UL_TTI_REQUEST
    send_message_from_secondary({0, SCF_FAPI_UL_TTI_REQUEST});

    nv_ipc_msg_t recv_msg{};
    const int ret = fapi_state_->receive_message(recv_msg);
    ASSERT_GE(ret, 0);

    EXPECT_EQ(fapi_state_->process_message(recv_msg), SCF_ERROR_CODE_MSG_OK);
    fapi_state_->release_message(recv_msg);

    // Verify execution order
    ASSERT_EQ(execution_order.size(), 2);
    EXPECT_EQ(execution_order.at(0), "on_message");
    EXPECT_EQ(execution_order.at(1), "on_ul_tti_request");
}

TEST_F(FapiStateNvIpcTest, Callback_OnMessageWithDataBuffer) {
    std::size_t captured_data_buf_size = 0;

    // Set on_message_ callback
    fapi_state_->set_on_message([&captured_data_buf_size](const rf::FapiMessageData &msg) {
        captured_data_buf_size = msg.data_buf.size();
    });

    // Send UL_TTI_REQUEST (may not have data buffer in test environment)
    send_message_from_secondary({0, SCF_FAPI_UL_TTI_REQUEST});

    nv_ipc_msg_t recv_msg{};
    const int ret = fapi_state_->receive_message(recv_msg);
    ASSERT_GE(ret, 0);

    EXPECT_EQ(fapi_state_->process_message(recv_msg), SCF_ERROR_CODE_MSG_OK);
    fapi_state_->release_message(recv_msg);

    // In test environment, data_buf is typically empty (data_len = 0)
    // This test verifies the span is correctly set (empty when no data buffer)
    EXPECT_EQ(captured_data_buf_size, 0);
}

TEST_F(FapiStateNvIpcTest, Callback_OnMessageWithoutDataBuffer) {
    std::size_t captured_data_buf_size = 0xFFFF;

    // Set on_message_ callback
    fapi_state_->set_on_message([&captured_data_buf_size](const rf::FapiMessageData &msg) {
        captured_data_buf_size = msg.data_buf.size();
    });

    // Send CONFIG.request (known to have no data buffer)
    EXPECT_EQ(process_config_request(0), SCF_ERROR_CODE_MSG_OK);

    // Verify data_buf span is empty
    EXPECT_EQ(captured_data_buf_size, 0);

    drain_config_response();
}

TEST_F(FapiStateNvIpcTest, Callback_OnMessageForAllMessageTypes) {
    int callback_count = 0;
    std::vector<uint16_t> captured_msg_ids;

    // Set on_message_ callback with counter
    fapi_state_->set_on_message(
            [&callback_count, &captured_msg_ids](const rf::FapiMessageData &msg) {
                callback_count++;
                captured_msg_ids.push_back(msg.msg_id);
            });

    // Send different message types
    EXPECT_EQ(process_config_request(0), SCF_ERROR_CODE_MSG_OK);
    drain_config_response();

    EXPECT_EQ(process_start_request(0), SCF_ERROR_CODE_MSG_OK);

    send_message_from_secondary({0, SCF_FAPI_UL_TTI_REQUEST});
    nv_ipc_msg_t recv_msg1{};
    ASSERT_GE(fapi_state_->receive_message(recv_msg1), 0);
    EXPECT_EQ(fapi_state_->process_message(recv_msg1), SCF_ERROR_CODE_MSG_OK);
    fapi_state_->release_message(recv_msg1);

    send_message_from_secondary({0, SCF_FAPI_DL_TTI_REQUEST});
    nv_ipc_msg_t recv_msg2{};
    ASSERT_GE(fapi_state_->receive_message(recv_msg2), 0);
    EXPECT_EQ(fapi_state_->process_message(recv_msg2), SCF_ERROR_CODE_MSG_OK);
    fapi_state_->release_message(recv_msg2);

    // Verify callback was invoked for all message types
    EXPECT_EQ(callback_count, 4); // CONFIG, START, UL_TTI, DL_TTI
    EXPECT_EQ(captured_msg_ids.size(), 4);
}

TEST_F(FapiStateNvIpcTest, Callback_OnMessageBufferContentsValid) {
    bool callback_invoked = false;
    uint16_t parsed_type_id = 0;
    uint32_t parsed_length = 0;

    // Set on_message_ callback that parses msg_buf
    fapi_state_->set_on_message([&callback_invoked, &parsed_type_id, &parsed_length](
                                        const rf::FapiMessageData &msg) {
        callback_invoked = true;

        // Parse msg_buf as body_header
        if (msg.msg_buf.size() >= sizeof(scf_fapi_body_header_t)) {
            const auto *body_hdr = rf::assume_cast<scf_fapi_body_header_t>(msg.msg_buf.data());
            parsed_type_id = body_hdr->type_id;
            parsed_length = body_hdr->length;
        }
    });

    // Send CONFIG.request
    EXPECT_EQ(process_config_request(0), SCF_ERROR_CODE_MSG_OK);

    // Verify msg_buf contains valid FAPI structure
    EXPECT_TRUE(callback_invoked);
    EXPECT_EQ(parsed_type_id, SCF_FAPI_CONFIG_REQUEST);
    EXPECT_GE(parsed_length, 0);

    drain_config_response();
}

TEST_F(FapiStateNvIpcTest, Callback_OnMessageNotSetNoCallback) {
    // Do NOT set on_message_ callback

    // Send various messages - should work normally without callback
    EXPECT_EQ(process_config_request(0), SCF_ERROR_CODE_MSG_OK);
    drain_config_response();

    EXPECT_EQ(process_start_request(0), SCF_ERROR_CODE_MSG_OK);
    EXPECT_EQ(fapi_state_->get_cell_state(0), rf::FapiStateT::FapiStateRunning);

    send_message_from_secondary({0, SCF_FAPI_UL_TTI_REQUEST});
    nv_ipc_msg_t recv_msg{};
    ASSERT_GE(fapi_state_->receive_message(recv_msg), 0);
    EXPECT_EQ(fapi_state_->process_message(recv_msg), SCF_ERROR_CODE_MSG_OK);
    fapi_state_->release_message(recv_msg);

    // Verify system works correctly without callback
    EXPECT_EQ(fapi_state_->get_cell_state(0), rf::FapiStateT::FapiStateRunning);
}

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

} // anonymous namespace
