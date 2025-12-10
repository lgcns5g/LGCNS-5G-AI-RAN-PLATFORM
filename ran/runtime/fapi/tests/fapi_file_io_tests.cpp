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

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <scf_5g_fapi.h>

#include <gtest/gtest.h>

#include "fapi/fapi_file_reader.hpp"
#include "fapi/fapi_file_writer.hpp"
#include "fapi/fapi_state.hpp"

namespace {

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

namespace rf = ran::fapi;

/**
 * Test message storage that owns its data
 */
struct TestMessage {
    std::vector<uint8_t> msg_storage;
    std::vector<uint8_t> data_storage;
    rf::FapiMessageData msg_data;

    /**
     * Create test message with specified parameters
     *
     * @param[in] cell_id Cell identifier
     * @param[in] msg_id FAPI message type
     * @param[in] msg_size Size of message buffer in bytes
     * @param[in] data_size Size of data buffer in bytes
     */
    explicit TestMessage(
            // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
            const uint16_t cell_id,
            const uint16_t msg_id,
            const std::size_t msg_size,
            const std::size_t data_size)
            : msg_storage(msg_size), data_storage(data_size) {

        // Fill with incrementing pattern
        for (std::size_t i = 0; i < msg_size; ++i) {
            msg_storage.at(i) = static_cast<uint8_t>(i % 256);
        }
        for (std::size_t i = 0; i < data_size; ++i) {
            data_storage.at(i) = static_cast<uint8_t>((i + 100) % 256);
        }

        msg_data.cell_id = cell_id;
        msg_data.msg_id = static_cast<scf_fapi_message_id_e>(msg_id);
        msg_data.msg_buf = std::span<const uint8_t>(msg_storage.data(), msg_size);
        msg_data.data_buf = std::span<const uint8_t>(data_storage.data(), data_size);
    }
};

/**
 * Create unique temporary file path
 *
 * @param[in] prefix File name prefix
 * @return Unique file path in temp directory
 */
std::filesystem::path create_temp_file_path(const std::string &prefix) {
    const std::filesystem::path temp_dir = std::filesystem::temp_directory_path();
    const std::string filename =
            prefix + "_" +
            std::to_string(::testing::UnitTest::GetInstance()->current_test_info()->line()) +
            ".fapi";
    return temp_dir / filename;
}

/**
 * Write multiple messages to file using FapiFileWriter
 *
 * @param[in] path Output file path
 * @param[in] test_messages Vector of test messages to write
 * @return Same path for convenience
 */
std::filesystem::path
write_test_file(const std::filesystem::path &path, const std::vector<TestMessage> &test_messages) {
    rf::FapiFileWriter writer(path.string());
    for (const auto &test_msg : test_messages) {
        writer.capture_message(test_msg.msg_data);
    }
    writer.flush_to_file();
    return path;
}

/**
 * Read all messages from file
 *
 * @param[in] path Input file path
 * @return Vector of messages read from file
 */
std::vector<rf::CapturedFapiMessage> read_all_messages(const std::filesystem::path &path) {
    std::vector<rf::CapturedFapiMessage> messages;
    rf::FapiFileReader reader(path.string());

    while (auto msg = reader.read_next()) {
        messages.push_back(std::move(*msg));
    }

    return messages;
}

/**
 * Test fixture for FAPI file I/O tests
 */
class FapiFileIoTest : public ::testing::Test {
protected:
    void SetUp() override { test_file_path_ = create_temp_file_path("fapi_io_test"); }

    void TearDown() override {
        if (std::filesystem::exists(test_file_path_)) {
            std::filesystem::remove(test_file_path_);
        }
    }

    std::filesystem::path test_file_path_;
};

// ============================================================================
// FAPI FILE WRITER TESTS
// ============================================================================

TEST_F(FapiFileIoTest, FileWriter_BasicCapture) {
    rf::FapiFileWriter writer(test_file_path_.string());

    // Create test message
    const TestMessage test_msg(0, SCF_FAPI_CONFIG_REQUEST, 100, 0);

    // Capture message
    writer.capture_message(test_msg.msg_data);

    // Verify count
    EXPECT_EQ(writer.get_message_count(), 1);
}

TEST_F(FapiFileIoTest, FileWriter_MultipleMessages) {
    rf::FapiFileWriter writer(test_file_path_.string());

    // Capture 5 messages with different parameters
    const TestMessage msg1(0, SCF_FAPI_CONFIG_REQUEST, 50, 0);
    writer.capture_message(msg1.msg_data);
    EXPECT_EQ(writer.get_message_count(), 1);

    const TestMessage msg2(1, SCF_FAPI_START_REQUEST, 60, 0);
    writer.capture_message(msg2.msg_data);
    EXPECT_EQ(writer.get_message_count(), 2);

    const TestMessage msg3(0, SCF_FAPI_UL_TTI_REQUEST, 70, 0);
    writer.capture_message(msg3.msg_data);
    EXPECT_EQ(writer.get_message_count(), 3);

    const TestMessage msg4(1, SCF_FAPI_DL_TTI_REQUEST, 80, 0);
    writer.capture_message(msg4.msg_data);
    EXPECT_EQ(writer.get_message_count(), 4);

    const TestMessage msg5(2, SCF_FAPI_SLOT_RESPONSE, 90, 0);
    writer.capture_message(msg5.msg_data);
    EXPECT_EQ(writer.get_message_count(), 5);
}

TEST_F(FapiFileIoTest, FileWriter_EmptyCapture) {
    rf::FapiFileWriter writer(test_file_path_.string());

    // Don't capture any messages
    EXPECT_EQ(writer.get_message_count(), 0);

    // Flush to file
    writer.flush_to_file();

    // Verify file exists
    EXPECT_TRUE(std::filesystem::exists(test_file_path_));

    // Read header to verify message count is 0
    std::ifstream file(test_file_path_, std::ios::binary);
    ASSERT_TRUE(file.is_open());

    rf::FapiFileHeader header{};
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    file.read(reinterpret_cast<char *>(&header), sizeof(header));
    ASSERT_TRUE(file);

    EXPECT_EQ(header.message_count, 0);
}

TEST_F(FapiFileIoTest, FileWriter_BufferSizeCalculation) {
    rf::FapiFileWriter writer(test_file_path_.string());

    // Message 1: msg_buf = 100 bytes, no data_buf
    const TestMessage msg1(0, SCF_FAPI_CONFIG_REQUEST, 100, 0);
    writer.capture_message(msg1.msg_data);

    // Message 2: msg_buf = 200 bytes, data_buf = 50 bytes
    const TestMessage msg2(1, SCF_FAPI_UL_TTI_REQUEST, 200, 50);
    writer.capture_message(msg2.msg_data);

    // Calculate expected size
    const std::size_t expected_size =
            sizeof(rf::FapiFileHeader) + 2 * sizeof(rf::FapiMessageRecordHeader) + 100 + 200 + 50;

    EXPECT_EQ(writer.get_buffer_size_bytes(), expected_size);
}

TEST_F(FapiFileIoTest, FileWriter_WithDataBuffer) {
    rf::FapiFileWriter writer(test_file_path_.string());

    // Create message with both msg_buf and data_buf
    const TestMessage test_msg(0, SCF_FAPI_UL_TTI_REQUEST, 100, 200);

    writer.capture_message(test_msg.msg_data);

    EXPECT_EQ(writer.get_message_count(), 1);

    // Verify buffer size includes both buffers
    const std::size_t expected_size =
            sizeof(rf::FapiFileHeader) + sizeof(rf::FapiMessageRecordHeader) + 100 + 200;

    EXPECT_EQ(writer.get_buffer_size_bytes(), expected_size);
}

TEST_F(FapiFileIoTest, FileWriter_FlushCreatesValidFile) {
    rf::FapiFileWriter writer(test_file_path_.string());

    // Capture 2 messages
    const TestMessage msg1(0, SCF_FAPI_CONFIG_REQUEST, 50, 0);
    writer.capture_message(msg1.msg_data);
    const TestMessage msg2(1, SCF_FAPI_START_REQUEST, 60, 0);
    writer.capture_message(msg2.msg_data);

    // Flush to file
    writer.flush_to_file();

    // Verify file exists
    EXPECT_TRUE(std::filesystem::exists(test_file_path_));

    // Read and verify header
    std::ifstream file(test_file_path_, std::ios::binary);
    ASSERT_TRUE(file.is_open());

    rf::FapiFileHeader header{};
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    file.read(reinterpret_cast<char *>(&header), sizeof(header));
    ASSERT_TRUE(file);

    EXPECT_EQ(header.magic.at(0), 'F');
    EXPECT_EQ(header.magic.at(1), 'A');
    EXPECT_EQ(header.magic.at(2), 'P');
    EXPECT_EQ(header.magic.at(3), 'I');
    EXPECT_EQ(header.version, rf::FapiFileHeader::CURRENT_VERSION);
    EXPECT_EQ(header.message_count, 2);
}

TEST_F(FapiFileIoTest, FileWriter_FlushMultipleTimes) {
    rf::FapiFileWriter writer(test_file_path_.string());

    // First flush with 1 message
    const TestMessage msg1(0, SCF_FAPI_CONFIG_REQUEST, 50, 0);
    writer.capture_message(msg1.msg_data);
    writer.flush_to_file();

    // Verify file has 1 message
    {
        const rf::FapiFileReader reader(test_file_path_.string());
        EXPECT_EQ(reader.get_total_message_count(), 1);
    }

    // Second flush with another message (overwrites due to std::ios::trunc)
    const TestMessage msg2(1, SCF_FAPI_START_REQUEST, 60, 0);
    writer.capture_message(msg2.msg_data);
    writer.flush_to_file();

    // Verify file now has 2 messages total (both msg1 and msg2 from internal buffer,
    // flush_to_file() uses std::ios::trunc to overwrite but writes all accumulated messages)
    {
        const rf::FapiFileReader reader(test_file_path_.string());
        EXPECT_EQ(reader.get_total_message_count(), 2);
    }
}

// ============================================================================
// FAPI FILE READER TESTS
// ============================================================================

TEST_F(FapiFileIoTest, FileReader_ReadSingleMessage) {
    // Create test file with 1 message
    std::vector<TestMessage> messages;
    messages.emplace_back(0, SCF_FAPI_CONFIG_REQUEST, 100, 0);
    write_test_file(test_file_path_, messages);

    // Read message
    rf::FapiFileReader reader(test_file_path_.string());
    const auto read_msg = reader.read_next();

    ASSERT_TRUE(read_msg.has_value());
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    const auto &msg = *read_msg;
    EXPECT_EQ(msg.cell_id, 0);
    EXPECT_EQ(msg.msg_id, SCF_FAPI_CONFIG_REQUEST);
    EXPECT_EQ(msg.msg_data.size(), 100);
    EXPECT_EQ(msg.data_buf.size(), 0);

    // Verify data contents match
    for (std::size_t i = 0; i < 100; ++i) {
        EXPECT_EQ(msg.msg_data.at(i), static_cast<uint8_t>(i % 256));
    }
}

TEST_F(FapiFileIoTest, FileReader_ReadMultipleMessages) {
    // Create test file with 5 messages
    std::vector<TestMessage> messages;
    messages.emplace_back(0, SCF_FAPI_CONFIG_REQUEST, 50, 0);
    messages.emplace_back(1, SCF_FAPI_START_REQUEST, 60, 0);
    messages.emplace_back(0, SCF_FAPI_UL_TTI_REQUEST, 70, 0);
    messages.emplace_back(1, SCF_FAPI_DL_TTI_REQUEST, 80, 0);
    messages.emplace_back(2, SCF_FAPI_SLOT_RESPONSE, 90, 0);
    write_test_file(test_file_path_, messages);

    // Read all messages
    rf::FapiFileReader reader(test_file_path_.string());
    EXPECT_EQ(reader.get_total_message_count(), 5);

    for (int i = 0; i < 5; ++i) {
        const auto msg = reader.read_next();
        ASSERT_TRUE(msg.has_value()) << "Failed to read message " << i;
    }

    // Next read should return nullopt
    const auto eof_msg = reader.read_next();
    EXPECT_FALSE(eof_msg.has_value());
    EXPECT_TRUE(reader.is_eof());
}

TEST_F(FapiFileIoTest, FileReader_Reset) {
    // Create test file with 3 messages
    std::vector<TestMessage> messages;
    messages.emplace_back(0, SCF_FAPI_CONFIG_REQUEST, 50, 0);
    messages.emplace_back(1, SCF_FAPI_START_REQUEST, 60, 0);
    messages.emplace_back(2, SCF_FAPI_UL_TTI_REQUEST, 70, 0);
    write_test_file(test_file_path_, messages);

    rf::FapiFileReader reader(test_file_path_.string());

    // Read first 2 messages
    const auto msg1 = reader.read_next();
    ASSERT_TRUE(msg1.has_value());
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    EXPECT_EQ(msg1->cell_id, 0);

    const auto msg2 = reader.read_next();
    ASSERT_TRUE(msg2.has_value());
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    EXPECT_EQ(msg2->cell_id, 1);

    EXPECT_EQ(reader.get_messages_read(), 2);

    // Reset
    reader.reset();
    EXPECT_EQ(reader.get_messages_read(), 0);
    EXPECT_FALSE(reader.is_eof());

    // Read first message again
    const auto msg = reader.read_next();
    ASSERT_TRUE(msg.has_value());
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    const auto &first_msg = *msg;
    EXPECT_EQ(first_msg.cell_id, 0);
    EXPECT_EQ(first_msg.msg_id, SCF_FAPI_CONFIG_REQUEST);
}

TEST_F(FapiFileIoTest, FileReader_MessageCounts) {
    // Create file with 7 messages
    std::vector<TestMessage> messages;
    messages.reserve(7);
    for (int i = 0; i < 7; ++i) {
        messages.emplace_back(
                static_cast<uint16_t>(i % 3), SCF_FAPI_CONFIG_REQUEST, 50 + i * 10, 0);
    }
    write_test_file(test_file_path_, messages);

    rf::FapiFileReader reader(test_file_path_.string());

    // Before reading
    EXPECT_EQ(reader.get_total_message_count(), 7);
    EXPECT_EQ(reader.get_messages_read(), 0);
    EXPECT_FALSE(reader.is_eof());

    // Read 3 messages
    for (int i = 0; i < 3; ++i) {
        const auto msg = reader.read_next();
        ASSERT_TRUE(msg.has_value()) << "Failed to read message " << i;
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        EXPECT_EQ(msg->msg_id, SCF_FAPI_CONFIG_REQUEST);
    }
    EXPECT_EQ(reader.get_messages_read(), 3);
    EXPECT_FALSE(reader.is_eof());

    // Read remaining 4 messages
    for (int i = 0; i < 4; ++i) {
        const auto msg = reader.read_next();
        ASSERT_TRUE(msg.has_value()) << "Failed to read message " << (i + 3);
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        EXPECT_EQ(msg->msg_id, SCF_FAPI_CONFIG_REQUEST);
    }
    EXPECT_EQ(reader.get_messages_read(), 7);
    EXPECT_TRUE(reader.is_eof());
}

TEST_F(FapiFileIoTest, FileReader_InvalidFile) {
    const std::filesystem::path non_existent = test_file_path_.parent_path() / "non_existent.fapi";

    // Try to construct reader with non-existent file
    EXPECT_THROW({ const rf::FapiFileReader reader(non_existent.string()); }, std::runtime_error);
}

TEST_F(FapiFileIoTest, FileReader_CorruptedMagic) {
    // Create file with invalid magic bytes
    std::ofstream file(test_file_path_, std::ios::binary);
    ASSERT_TRUE(file.is_open());

    rf::FapiFileHeader header{};
    header.magic.at(0) = 'X';
    header.magic.at(1) = 'X';
    header.magic.at(2) = 'X';
    header.magic.at(3) = 'X';
    header.version = rf::FapiFileHeader::CURRENT_VERSION;
    header.message_count = 0;
    header.reserved = 0;

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    file.write(reinterpret_cast<const char *>(&header), sizeof(header));
    file.close();

    // Try to construct reader
    EXPECT_THROW(
            { const rf::FapiFileReader reader(test_file_path_.string()); }, std::runtime_error);
}

TEST_F(FapiFileIoTest, FileReader_UnsupportedVersion) {
    // Create file with invalid version
    std::ofstream file(test_file_path_, std::ios::binary);
    ASSERT_TRUE(file.is_open());

    rf::FapiFileHeader header{};
    header.magic.at(0) = 'F';
    header.magic.at(1) = 'A';
    header.magic.at(2) = 'P';
    header.magic.at(3) = 'I';
    header.version = 999; // Invalid version
    header.message_count = 0;
    header.reserved = 0;

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    file.write(reinterpret_cast<const char *>(&header), sizeof(header));
    file.close();

    // Try to construct reader
    EXPECT_THROW(
            { const rf::FapiFileReader reader(test_file_path_.string()); }, std::runtime_error);
}

TEST_F(FapiFileIoTest, FileReader_TruncatedFile) {
    // Create valid file with 2 messages
    std::vector<TestMessage> messages;
    messages.emplace_back(0, SCF_FAPI_CONFIG_REQUEST, 50, 0);
    messages.emplace_back(1, SCF_FAPI_START_REQUEST, 60, 0);
    write_test_file(test_file_path_, messages);

    // Truncate file (remove last half of second message)
    const auto file_size = std::filesystem::file_size(test_file_path_);
    std::filesystem::resize_file(test_file_path_, file_size - 30);

    // Try to read
    rf::FapiFileReader reader(test_file_path_.string());

    // First message should read successfully
    const auto msg1 = reader.read_next();
    ASSERT_TRUE(msg1.has_value());
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    EXPECT_EQ(msg1->cell_id, 0);

    // Second message should throw exception due to truncation
    EXPECT_THROW({ const auto msg2 = reader.read_next(); }, std::runtime_error);
}

// ============================================================================
// ROUND-TRIP TESTS
// ============================================================================

TEST_F(FapiFileIoTest, RoundTrip_SingleMessage) {
    // Create original message with known values
    const TestMessage original(5, SCF_FAPI_CONFIG_REQUEST, 150, 0);

    // Write to file
    rf::FapiFileWriter writer(test_file_path_.string());
    writer.capture_message(original.msg_data);
    writer.flush_to_file();

    // Read back
    rf::FapiFileReader reader(test_file_path_.string());
    const auto read_msg = reader.read_next();

    ASSERT_TRUE(read_msg.has_value());
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    const auto &msg = *read_msg;

    // Verify all fields match
    EXPECT_EQ(msg.cell_id, original.msg_data.cell_id);
    EXPECT_EQ(msg.msg_id, original.msg_data.msg_id);
    EXPECT_EQ(msg.msg_data.size(), original.msg_data.msg_buf.size());
    EXPECT_EQ(msg.data_buf.size(), original.msg_data.data_buf.size());

    // Byte-by-byte comparison
    for (std::size_t i = 0; i < original.msg_data.msg_buf.size(); ++i) {
        EXPECT_EQ(msg.msg_data.at(i), original.msg_data.msg_buf[i]) << "Mismatch at byte " << i;
    }
}

TEST_F(FapiFileIoTest, RoundTrip_MessageWithDataBuffer) {
    // Create message with both msg_buf and data_buf with custom pattern
    TestMessage original(3, SCF_FAPI_UL_TTI_REQUEST, 100, 500);

    // Override with custom pattern (multiply by 3 instead of add 100)
    for (std::size_t i = 0; i < 500; ++i) {
        original.data_storage.at(i) = static_cast<uint8_t>((i * 3) % 256);
    }
    original.msg_data.data_buf = std::span<const uint8_t>(original.data_storage.data(), 500);

    // Write and read back
    rf::FapiFileWriter writer(test_file_path_.string());
    writer.capture_message(original.msg_data);
    writer.flush_to_file();

    rf::FapiFileReader reader(test_file_path_.string());
    const auto read_msg = reader.read_next();

    ASSERT_TRUE(read_msg.has_value());
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    const auto &msg = *read_msg;

    // Verify both buffers are identical
    EXPECT_EQ(msg.msg_data.size(), 100);
    EXPECT_EQ(msg.data_buf.size(), 500);

    for (std::size_t i = 0; i < 100; ++i) {
        EXPECT_EQ(msg.msg_data.at(i), original.msg_storage.at(i));
    }
    for (std::size_t i = 0; i < 500; ++i) {
        EXPECT_EQ(msg.data_buf.at(i), original.data_storage.at(i));
    }
}

TEST_F(FapiFileIoTest, RoundTrip_MultipleMessageTypes) {
    // Create 5 messages simulating different FAPI types with custom patterns
    std::vector<TestMessage> original_messages;
    original_messages.emplace_back(0, SCF_FAPI_CONFIG_REQUEST, 50, 0);
    original_messages.emplace_back(0, SCF_FAPI_START_REQUEST, 60, 0);
    original_messages.emplace_back(1, SCF_FAPI_UL_TTI_REQUEST, 70, 0);
    original_messages.emplace_back(1, SCF_FAPI_DL_TTI_REQUEST, 80, 0);
    original_messages.emplace_back(2, SCF_FAPI_SLOT_RESPONSE, 90, 0);

    // Override with custom patterns
    for (std::size_t msg_idx = 1; msg_idx < original_messages.size(); ++msg_idx) {
        for (std::size_t i = 0; i < original_messages.at(msg_idx).msg_storage.size(); ++i) {
            original_messages.at(msg_idx).msg_storage.at(i) =
                    static_cast<uint8_t>(i * (msg_idx + 1));
        }
    }

    // Update spans to point to modified data
    for (auto &msg : original_messages) {
        msg.msg_data.msg_buf =
                std::span<const uint8_t>(msg.msg_storage.data(), msg.msg_storage.size());
    }

    // Write all messages
    write_test_file(test_file_path_, original_messages);

    // Read all back
    const auto read_messages = read_all_messages(test_file_path_);

    ASSERT_EQ(read_messages.size(), 5);

    // Verify each message
    const std::vector<uint16_t> expected_cell_ids = {0, 0, 1, 1, 2};
    const std::vector<uint16_t> expected_msg_ids = {
            SCF_FAPI_CONFIG_REQUEST,
            SCF_FAPI_START_REQUEST,
            SCF_FAPI_UL_TTI_REQUEST,
            SCF_FAPI_DL_TTI_REQUEST,
            SCF_FAPI_SLOT_RESPONSE};
    const std::vector<std::size_t> expected_sizes = {50, 60, 70, 80, 90};

    for (std::size_t i = 0; i < 5; ++i) {
        EXPECT_EQ(read_messages.at(i).cell_id, expected_cell_ids.at(i));
        EXPECT_EQ(read_messages.at(i).msg_id, expected_msg_ids.at(i));
        EXPECT_EQ(read_messages.at(i).msg_data.size(), expected_sizes.at(i));
    }
}

TEST_F(FapiFileIoTest, FileReader_ReadAll) {
    struct ExpectedMessage {
        uint16_t cell_id;
        uint16_t msg_id;
        std::size_t msg_data_size;
        std::size_t data_buf_size;
    };

    const std::array<ExpectedMessage, 5> expected_messages = {
            {{0, SCF_FAPI_CONFIG_REQUEST, 50, 0},
             {1, SCF_FAPI_START_REQUEST, 60, 0},
             {2, SCF_FAPI_UL_TTI_REQUEST, 70, 100},
             {0, SCF_FAPI_DL_TTI_REQUEST, 80, 0},
             {1, SCF_FAPI_SLOT_RESPONSE, 90, 200}}};

    // Create test file with 5 messages
    std::vector<TestMessage> messages;
    messages.reserve(expected_messages.size());
    for (const auto &expected : expected_messages) {
        messages.emplace_back(
                expected.cell_id, expected.msg_id, expected.msg_data_size, expected.data_buf_size);
    }
    write_test_file(test_file_path_, messages);

    // Use read_all to load all messages
    rf::FapiFileReader reader(test_file_path_.string());
    const auto all_messages = reader.read_all();

    // Verify count
    ASSERT_EQ(all_messages.size(), expected_messages.size());
    EXPECT_EQ(reader.get_messages_read(), expected_messages.size());
    EXPECT_TRUE(reader.is_eof());

    // Verify each message
    for (std::size_t i = 0; i < expected_messages.size(); ++i) {
        EXPECT_EQ(all_messages.at(i).cell_id, expected_messages.at(i).cell_id);
        EXPECT_EQ(all_messages.at(i).msg_id, expected_messages.at(i).msg_id);
        EXPECT_EQ(all_messages.at(i).msg_data.size(), expected_messages.at(i).msg_data_size);
        EXPECT_EQ(all_messages.at(i).data_buf.size(), expected_messages.at(i).data_buf_size);
    }

    // Verify data contents for first message
    for (std::size_t i = 0; i < 50; ++i) {
        EXPECT_EQ(all_messages.at(0).msg_data.at(i), static_cast<uint8_t>(i % 256));
    }
}

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

} // namespace
