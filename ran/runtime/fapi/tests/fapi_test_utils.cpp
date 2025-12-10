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

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <functional>
#include <string>

#include "fapi_test_utils.hpp"

namespace ran::fapi {

tl::expected<std::filesystem::path, std::string> get_fapi_capture_file_path() {
    // NOLINTNEXTLINE(concurrency-mt-unsafe)
    const char *env_dir = std::getenv("FAPI_CAPTURE_DIR");
    if (env_dir == nullptr) {
        return tl::unexpected<std::string>(
                "FAPI_CAPTURE_DIR environment variable not set. "
                "Ensure test is run via CTest with fapi_capture_file fixture.");
    }

    // NOLINTNEXTLINE(concurrency-mt-unsafe)
    const char *env_cell_count = std::getenv("TEST_CELLS");
    const std::string cell_count_str = env_cell_count != nullptr ? env_cell_count : "1";

    if (cell_count_str.empty()) {
        return tl::unexpected<std::string>("TEST_CELLS cannot be empty");
    }

    if (!std::ranges::all_of(
                cell_count_str, [](const unsigned char c) { return std::isdigit(c); })) {
        return tl::unexpected<std::string>(std::format(
                "TEST_CELLS must contain only numeric digits, got: {}", cell_count_str));
    }

    // Construct filename dynamically: fapi_capture_fapi_sample_{TEST_CELLS}C.fapi
    const std::string filename = std::format("fapi_capture_fapi_sample_{}C.fapi", cell_count_str);
    std::filesystem::path fapi_capture_file = std::filesystem::path(env_dir) / filename;

    if (!std::filesystem::exists(fapi_capture_file)) {
        return tl::unexpected<std::string>(std::format(
                "FAPI capture file does not exist: {}. "
                "Ensure fapi_sample.integration_test has completed successfully.",
                fapi_capture_file.string()));
    }

    return fapi_capture_file;
}

std::string YamlConfigBuilder::create_primary_config(const std::string &prefix) {
    return R"(
transport:
  type: shm
  shm_config:
    primary: 1
    prefix: )" +
           prefix + R"(
    cuda_device_id: -1
    ring_len: 1024
    mempool_size:
      cpu_msg: {buf_size: 15000, pool_len: 256}
      cpu_data: {buf_size: 576000, pool_len: 64}
      cpu_large: {buf_size: 4096000, pool_len: 16}
      cuda_data: {buf_size: 307200, pool_len: 0}
      gpu_data: {buf_size: 576000, pool_len: 0}
  app_config:
    grpc_forward: 0
    debug_timing: 0
    pcap_enable: 0
    pcap_shm_caching_cpu_core: 13
    pcap_file_saving_cpu_core: 13
    pcap_cache_size_bits: 29
    pcap_file_size_bits: 31
    pcap_max_data_size: 8000
    msg_filter: []
    cell_filter: []
)";
}

std::string YamlConfigBuilder::create_secondary_config(const std::string &prefix) {
    return R"(
transport:
  type: shm
  shm_config:
    primary: 0
    prefix: )" +
           prefix + R"(
    cuda_device_id: -1
    ring_len: 1024
    mempool_size:
      cpu_msg: {buf_size: 15000, pool_len: 256}
      cpu_data: {buf_size: 576000, pool_len: 64}
      cpu_large: {buf_size: 4096000, pool_len: 16}
      cuda_data: {buf_size: 307200, pool_len: 0}
      gpu_data: {buf_size: 576000, pool_len: 0}
  app_config:
    grpc_forward: 0
    debug_timing: 0
    pcap_enable: 0
    pcap_shm_caching_cpu_core: 13
    pcap_file_saving_cpu_core: 13
    pcap_cache_size_bits: 29
    pcap_file_size_bits: 31
    pcap_max_data_size: 8000
    msg_filter: []
    cell_filter: []
)";
}

} // namespace ran::fapi
