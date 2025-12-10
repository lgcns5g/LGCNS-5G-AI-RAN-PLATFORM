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
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "fapi/fapi_file_reader.hpp"
#include "fapi/fapi_file_writer.hpp" // For header structs and CapturedFapiMessage

namespace ran::fapi {

FapiFileReader::FapiFileReader(std::string input_path) : input_path_{std::move(input_path)} {
    open_and_validate();
}

std::optional<CapturedFapiMessage> FapiFileReader::read_next() {
    if (!file_ || file_.eof() || messages_read_ >= total_message_count_) {
        return std::nullopt;
    }

    // Read record header
    FapiMessageRecordHeader rec_hdr{};
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    file_.read(reinterpret_cast<char *>(&rec_hdr), sizeof(rec_hdr));
    if (!file_ || file_.gcount() != sizeof(rec_hdr)) {
        if (file_.eof()) {
            return std::nullopt;
        }
        throw std::runtime_error("Failed to read message record header");
    }

    // Validate lengths (sanity check against corrupted files)
    static constexpr uint32_t MAX_MSG_LEN = 64 * 1024;         // 64KB
    static constexpr uint32_t MAX_DATA_LEN = 10 * 1024 * 1024; // 10MB
    if (rec_hdr.msg_len > MAX_MSG_LEN) {
        throw std::runtime_error(
                "Invalid msg_len in file (possible corruption): " +
                std::to_string(rec_hdr.msg_len));
    }
    if (rec_hdr.data_len > MAX_DATA_LEN) {
        throw std::runtime_error(
                "Invalid data_len in file (possible corruption): " +
                std::to_string(rec_hdr.data_len));
    }

    CapturedFapiMessage msg;
    msg.cell_id = rec_hdr.cell_id;
    msg.msg_id = rec_hdr.msg_id;

    // Read msg_data
    if (rec_hdr.msg_len > 0) {
        msg.msg_data.resize(rec_hdr.msg_len);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        file_.read(reinterpret_cast<char *>(msg.msg_data.data()), rec_hdr.msg_len);
        if (!file_ || static_cast<uint32_t>(file_.gcount()) != rec_hdr.msg_len) {
            throw std::runtime_error("Failed to read msg_data");
        }
    }

    // Read data_buf
    if (rec_hdr.data_len > 0) {
        msg.data_buf.resize(rec_hdr.data_len);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        file_.read(reinterpret_cast<char *>(msg.data_buf.data()), rec_hdr.data_len);
        if (!file_ || static_cast<uint32_t>(file_.gcount()) != rec_hdr.data_len) {
            throw std::runtime_error("Failed to read data_buf");
        }
    }

    messages_read_++;
    return msg;
}

std::vector<CapturedFapiMessage> FapiFileReader::read_all() {
    std::vector<CapturedFapiMessage> messages;

    // Reserve space if we know the total count
    if (total_message_count_ > messages_read_) {
        messages.reserve(total_message_count_ - messages_read_);
    }

    // Read all remaining messages
    while (auto msg = read_next()) {
        messages.push_back(std::move(*msg));
    }

    return messages;
}

void FapiFileReader::reset() {
    file_.clear();
    file_.seekg(sizeof(FapiFileHeader), std::ios::beg);
    if (!file_) {
        throw std::runtime_error("Failed to seek to beginning of messages");
    }
    messages_read_ = 0;
}

std::size_t FapiFileReader::get_total_message_count() const noexcept {
    return total_message_count_;
}

std::size_t FapiFileReader::get_messages_read() const noexcept { return messages_read_; }

bool FapiFileReader::is_eof() const noexcept {
    return file_.eof() || messages_read_ >= total_message_count_;
}

void FapiFileReader::open_and_validate() {
    file_.open(input_path_, std::ios::binary);
    if (!file_) {
        throw std::runtime_error("Failed to open file: " + input_path_);
    }

    // Read and validate file header
    FapiFileHeader header{};
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    file_.read(reinterpret_cast<char *>(&header), sizeof(header));
    if (!file_ || file_.gcount() != sizeof(header)) {
        throw std::runtime_error("Failed to read file header from: " + input_path_);
    }

    // Validate magic
    if (header.magic != FapiFileHeader::MAGIC_CHARS) {
        throw std::runtime_error("Invalid file format (bad magic): " + input_path_);
    }

    // Validate version
    if (header.version != FapiFileHeader::CURRENT_VERSION) {
        throw std::runtime_error(
                "Unsupported file format version " + std::to_string(header.version) +
                " (expected " + std::to_string(FapiFileHeader::CURRENT_VERSION) + ")");
    }

    total_message_count_ = header.message_count;
}

} // namespace ran::fapi
