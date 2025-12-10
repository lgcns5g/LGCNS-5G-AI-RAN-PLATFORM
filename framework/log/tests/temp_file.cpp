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
 * @file temp_file.cpp
 * @brief Implementation of temporary file management utility
 */

#include <algorithm> // for std::all_of
#include <chrono>    // for duration_cast, duration, high_resolution_clock
#include <exception> // for exception
#include <fstream>   // for basic_ostream, operator<<, basic_ifstream
#include <iostream>  // for cerr
#include <sstream>   // for basic_stringstream
#include <utility>   // for std::move

#include "temp_file.hpp"

namespace framework::log {

TempFileManager::TempFileManager(std::string prefix)
        : prefix_(std::move(prefix)), temp_dir_(std::filesystem::temp_directory_path()) {}

TempFileManager::~TempFileManager() {
    for (const auto &file : temp_files_) {
        try {
            if (std::filesystem::exists(file)) {
                std::filesystem::remove(file);
            }
        } catch (const std::exception &e) {
            std::cerr << "Warning: Failed to remove temporary file " << file << ": " << e.what()
                      << '\n';
        }
    }
}

std::string TempFileManager::get_temp_file(const std::string &suffix) {
    auto now = std::chrono::high_resolution_clock::now();
    auto timestamp =
            std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();

    const std::string filename = prefix_ + "_" + std::to_string(timestamp) + "_" +
                                 std::to_string(instance_counter_.fetch_add(1) + 1) + suffix;

    auto filepath = (temp_dir_ / filename).string();
    temp_files_.push_back(filepath);

    return filepath;
}

// Free utility functions
std::string read_file_contents(const std::string &filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        return "";
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
bool file_contains(const std::string &filepath, const std::string &search_text) {
    const std::string contents = read_file_contents(filepath);
    return contents.find(search_text) != std::string::npos;
}

bool file_contains_all(const std::string &filepath, const std::vector<std::string> &search_texts) {
    const std::string contents = read_file_contents(filepath);

    return std::all_of(
            search_texts.begin(), search_texts.end(), [&contents](const std::string &search_text) {
                return contents.find(search_text) != std::string::npos;
            });
}

} // namespace framework::log
