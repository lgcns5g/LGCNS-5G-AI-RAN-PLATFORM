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
#include <fstream>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "fapi/fapi_file_writer.hpp"
#include "fapi/fapi_state.hpp"

namespace ran::fapi {

namespace {

/**
 * Write a single message record to file
 *
 * @param[in,out] file Output file stream
 * @param[in] msg Message to write
 * @throw std::runtime_error if write fails
 */
void write_message_record(std::ofstream &file, const CapturedFapiMessage &msg) {
    // Write record header
    const FapiMessageRecordHeader rec_hdr{
            .cell_id = msg.cell_id,
            .msg_id = msg.msg_id,
            .msg_len = static_cast<uint32_t>(msg.msg_data.size()),
            .data_len = static_cast<uint32_t>(msg.data_buf.size())};

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    file.write(reinterpret_cast<const char *>(&rec_hdr), sizeof(rec_hdr));

    // Write msg_data
    if (rec_hdr.msg_len > 0) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        file.write(reinterpret_cast<const char *>(msg.msg_data.data()), rec_hdr.msg_len);
    }

    // Write data_buf
    if (rec_hdr.data_len > 0) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        file.write(reinterpret_cast<const char *>(msg.data_buf.data()), rec_hdr.data_len);
    }

    if (!file) {
        throw std::runtime_error("Failed to write message record");
    }
}

} // anonymous namespace

FapiFileWriter::FapiFileWriter(std::string output_path, const std::size_t initial_capacity)
        : output_path_{std::move(output_path)} {
    messages_.reserve(initial_capacity); // Reserve space for typical capture session
}

void FapiFileWriter::capture_message(const FapiMessageData &msg) {
    CapturedFapiMessage captured;
    captured.cell_id = msg.cell_id;
    captured.msg_id = msg.msg_id;

    // Copy msg_buf
    if (!msg.msg_buf.empty()) {
        captured.msg_data.assign(msg.msg_buf.begin(), msg.msg_buf.end());
    }

    // Copy data_buf if present
    if (!msg.data_buf.empty()) {
        captured.data_buf.assign(msg.data_buf.begin(), msg.data_buf.end());
    }

    messages_.push_back(std::move(captured));
}

void FapiFileWriter::flush_to_file() {
    std::ofstream file(output_path_, std::ios::binary | std::ios::trunc);
    if (!file) {
        throw std::runtime_error("Failed to open file for writing: " + output_path_);
    }

    // Write file header
    write_file_header(file);

    // Write all messages
    for (const auto &msg : messages_) {
        write_message_record(file, msg);
    }

    file.close();
    if (!file.good()) {
        throw std::runtime_error("Failed to close file: " + output_path_);
    }
}

std::size_t FapiFileWriter::get_message_count() const noexcept { return messages_.size(); }

std::size_t FapiFileWriter::get_buffer_size_bytes() const noexcept {
    std::size_t total = sizeof(FapiFileHeader);
    for (const auto &msg : messages_) {
        total += sizeof(FapiMessageRecordHeader);
        total += msg.msg_data.size();
        total += msg.data_buf.size();
    }
    return total;
}

void FapiFileWriter::write_file_header(std::ofstream &file) const {
    const FapiFileHeader header{
            .magic = FapiFileHeader::MAGIC_CHARS,
            .version = FapiFileHeader::CURRENT_VERSION,
            .message_count = static_cast<uint32_t>(messages_.size()),
            .reserved = 0};

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    file.write(reinterpret_cast<const char *>(&header), sizeof(header));
    if (!file) {
        throw std::runtime_error("Failed to write file header");
    }
}

} // namespace ran::fapi
