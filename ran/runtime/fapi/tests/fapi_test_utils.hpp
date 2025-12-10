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

#ifndef RAN_FAPI_TEST_UTILS_HPP
#define RAN_FAPI_TEST_UTILS_HPP

#include <filesystem>
#include <string>

#include <tl/expected.hpp>

namespace ran::fapi {

/**
 * Get FAPI capture file path from environment variables
 *
 * Reads FAPI_CAPTURE_DIR and TEST_CELLS environment variables,
 * validates their values, and constructs the path to the FAPI capture file.
 *
 * @return Path to FAPI capture file on success, error message on failure
 */
[[nodiscard]] tl::expected<std::filesystem::path, std::string> get_fapi_capture_file_path();

/**
 * Helper class for building NVIPC YAML configurations for tests
 */
class YamlConfigBuilder {
public:
    /**
     * Create primary NVIPC configuration
     *
     * @param[in] prefix NVIPC prefix for shared memory
     * @return YAML configuration string
     */
    static std::string create_primary_config(const std::string &prefix = "fapi_test");

    /**
     * Create secondary NVIPC configuration
     *
     * @param[in] prefix NVIPC prefix for shared memory
     * @return YAML configuration string
     */
    static std::string create_secondary_config(const std::string &prefix = "fapi_test");
};

} // namespace ran::fapi

#endif // RAN_FAPI_TEST_UTILS_HPP
