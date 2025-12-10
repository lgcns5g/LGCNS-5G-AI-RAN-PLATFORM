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
 * @file fapi_file_writer.hpp
 * @brief FAPI message file writer for capture and replay
 *
 * Provides FapiFileWriter class for capturing FAPI messages to a binary file.
 * Messages are buffered in memory during capture and written to file on flush.
 */

#ifndef RAN_FAPI_FILE_WRITER_HPP
#define RAN_FAPI_FILE_WRITER_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <span>
#include <string>
#include <vector>

#include <scf_5g_fapi.h>

#include "fapi/fapi_state.hpp"

namespace ran::fapi {

/**
 * File format header (16 bytes)
 */
struct FapiFileHeader {
    static constexpr uint32_t CURRENT_VERSION = 1; //!< File format version
    static constexpr std::array<char, 4> MAGIC_CHARS = {
            'F', 'A', 'P', 'I'}; //!< Magic identifier characters

    std::array<char, 4> magic{}; //!< Magic identifier "FAPI"
    uint32_t version{};          //!< File format version
    uint32_t message_count{};    //!< Total number of messages
    uint32_t reserved{};         //!< Reserved for future use
};

/**
 * Message record header (12 bytes)
 */
struct FapiMessageRecordHeader {
    uint16_t cell_id;  //!< Cell identifier
    uint16_t msg_id;   //!< FAPI message type
    uint32_t msg_len;  //!< Length of message data
    uint32_t data_len; //!< Length of data buffer (0 if none)
};

/**
 * Captured FAPI message
 *
 * Used by both FapiFileWriter and FapiFileReader for storing
 * FAPI message data with optional data buffer.
 */
struct CapturedFapiMessage {
    uint16_t cell_id{};            //!< Cell identifier
    uint16_t msg_id{};             //!< FAPI message type
    std::vector<uint8_t> msg_data; //!< body_hdr + body
    std::vector<uint8_t> data_buf; //!< optional data buffer
};

/**
 * FAPI message file writer
 *
 * Captures FAPI messages to memory during execution and writes them
 * to a binary file on flush. Designed for use with FapiState callbacks
 * to record messages for later replay without NVIPC dependencies.
 *
 * Usage:
 * @code
 * FapiFileWriter writer("/tmp/capture.fapi");
 * fapi_state.set_on_message([&writer](const FapiMessageData& msg) {
 *     writer.capture_message(msg);
 * });
 * // ... run test ...
 * writer.flush_to_file();
 * @endcode
 *
 * Thread safety: Not thread-safe. External synchronization required if
 * called from multiple threads.
 */
class FapiFileWriter {
public:
    static constexpr std::size_t DEFAULT_INITIAL_CAPACITY = 1000; //!< Default message capacity

    /**
     * Construct file writer
     *
     * @param[in] output_path Path to output file (will be created/overwritten)
     * @param[in] initial_capacity Initial message capacity to reserve
     */
    explicit FapiFileWriter(
            std::string output_path, std::size_t initial_capacity = DEFAULT_INITIAL_CAPACITY);

    /**
     * Capture a message to memory buffer
     *
     * Called from FapiState on_message callback. Copies message data
     * to internal buffer for later writing.
     *
     * @param[in] msg Message data to capture
     */
    void capture_message(const FapiMessageData &msg);

    /**
     * Write all buffered messages to file
     *
     * Writes file header followed by all captured message records.
     * Should be called once at end of capture session.
     *
     * @throw std::runtime_error if file operations fail
     */
    void flush_to_file();

    /**
     * Get number of captured messages
     *
     * @return Count of messages in buffer
     */
    [[nodiscard]] std::size_t get_message_count() const noexcept;

    /**
     * Get total buffer size in bytes
     *
     * @return Total size of all captured data including headers
     */
    [[nodiscard]] std::size_t get_buffer_size_bytes() const noexcept;

private:
    void write_file_header(std::ofstream &file) const;

    std::string output_path_;
    std::vector<CapturedFapiMessage> messages_;
};

} // namespace ran::fapi

#endif // RAN_FAPI_FILE_WRITER_HPP
