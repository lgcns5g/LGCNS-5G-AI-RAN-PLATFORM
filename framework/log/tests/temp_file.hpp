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
 * @file temp_file.hpp
 * @brief Temporary file management utility for testing
 */

#ifndef FRAMEWORK_LOG_TESTS_TEMP_FILE_HPP
#define FRAMEWORK_LOG_TESTS_TEMP_FILE_HPP

#include <atomic>
#include <filesystem>
#include <string>
#include <vector>

namespace framework::log {

/**
 * @brief RAII utility for managing temporary files in tests
 *
 * Creates temporary files with unique names and automatically cleans them up
 * when the object is destroyed. Each instance is independent with its own
 * naming. Thread-safe for concurrent calls to get_temp_file().
 */
class TempFileManager {
public:
    /**
     * @brief Create a temporary file manager
     * @param prefix Prefix for temporary file names
     */
    explicit TempFileManager(std::string prefix = "rt_log_test");

    /**
     * @brief Destructor - cleans up all temporary files
     */
    ~TempFileManager();

    // Delete copy constructor and copy assignment operator
    TempFileManager(const TempFileManager &) = delete;
    TempFileManager &operator=(const TempFileManager &) = delete;

    // Delete move constructor and move assignment operator
    TempFileManager(TempFileManager &&) = delete;
    TempFileManager &operator=(TempFileManager &&) = delete;

    /**
     * @brief Get a unique temporary file path
     * @param suffix Optional suffix to append to the file name
     * @return Full path to the temporary file
     */
    std::string get_temp_file(const std::string &suffix = "");

private:
    std::string prefix_;
    std::vector<std::string> temp_files_;
    std::filesystem::path temp_dir_;
    std::atomic<int> instance_counter_{0};
};

// Free utility functions
/**
 * @brief Read the contents of a file
 * @param filepath Path to the file to read
 * @return File contents as a string
 */
std::string read_file_contents(const std::string &filepath);

/**
 * @brief Check if a file contains the specified text
 * @param filepath Path to the file to check
 * @param search_text Text to search for
 * @return true if the text is found, false otherwise
 */
bool file_contains(const std::string &filepath, const std::string &search_text);

/**
 * @brief Check if a file contains all specified texts
 * @param filepath Path to the file to check
 * @param search_texts Vector of texts to search for
 * @return true if all texts are found, false otherwise
 */
bool file_contains_all(const std::string &filepath, const std::vector<std::string> &search_texts);

} // namespace framework::log

#endif // FRAMEWORK_LOG_TESTS_TEMP_FILE_HPP
