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
 * @file fapi_file_reader.hpp
 * @brief FAPI message file reader for replay
 *
 * Provides FapiFileReader class for reading FAPI messages from binary files
 * created by FapiFileWriter.
 */

#ifndef RAN_FAPI_FILE_READER_HPP
#define RAN_FAPI_FILE_READER_HPP

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

namespace ran::fapi {

// Forward declaration
struct CapturedFapiMessage;

/**
 * FAPI message file reader
 *
 * Reads FAPI messages from binary files created by FapiFileWriter.
 * Provides sequential access to captured messages for replay or analysis.
 *
 * Usage:
 * @code
 * FapiFileReader reader("/tmp/capture.fapi");
 * std::map<std::uint16_t, std::vector<std::vector<std::uint8_t>>> fapi_messages;
 * while (auto msg = reader.read_next()) {
 *     if (msg->msg_data.empty()) {
 *         continue;
 *     }
 *     const std::uint16_t cell_id = msg->cell_id;
 *     fapi_messages[cell_id].emplace_back(msg->msg_data.begin(), msg->msg_data.end());
 * }
 * @endcode
 *
 * Thread safety: Not thread-safe. External synchronization required.
 */
class FapiFileReader {
public:
    /**
     * Open and validate FAPI file
     *
     * @param[in] input_path Path to input file
     * @throw std::runtime_error if file cannot be opened or is invalid
     */
    explicit FapiFileReader(std::string input_path);

    /**
     * Read next message from file
     *
     * @return Message if available, std::nullopt if EOF or error
     * @throw std::runtime_error if file is corrupted
     */
    [[nodiscard]] std::optional<CapturedFapiMessage> read_next();

    /**
     * Read all remaining messages from file
     *
     * Reads all messages from current position to end of file.
     * Useful for loading entire capture files into memory.
     *
     * @return Vector of all remaining messages
     * @throw std::runtime_error if file is corrupted
     */
    [[nodiscard]] std::vector<CapturedFapiMessage> read_all();

    /**
     * Reset reader to beginning of file
     *
     * Seeks back to first message for re-reading.
     */
    void reset();

    /**
     * Get total message count from file header
     *
     * @return Total number of messages in file
     */
    [[nodiscard]] std::size_t get_total_message_count() const noexcept;

    /**
     * Get number of messages read so far
     *
     * @return Count of messages read
     */
    [[nodiscard]] std::size_t get_messages_read() const noexcept;

    /**
     * Check if at end of file
     *
     * @return true if EOF or all messages read
     */
    [[nodiscard]] bool is_eof() const noexcept;

private:
    void open_and_validate();

    std::string input_path_;
    std::ifstream file_;
    std::size_t total_message_count_{};
    std::size_t messages_read_{};
};

} // namespace ran::fapi

#endif // RAN_FAPI_FILE_READER_HPP
