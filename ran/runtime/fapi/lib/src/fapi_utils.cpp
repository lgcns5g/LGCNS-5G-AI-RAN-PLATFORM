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
 * @file fapi_utils.cpp
 * @brief Implementation of FAPI utility functions
 */

#include <string>
#include <string_view>

#include <fmt/core.h>

#include "fapi/fapi_utils.hpp"

namespace ran::fapi {

std::string create_default_nvipc_config(const std::string_view prefix) {
    // Note: Double braces {{ }} in format string are escaped to produce single braces in YAML
    // clang-format off
    return fmt::format(R"(
transport:
  type: shm
  shm_config:
    primary: 1
    prefix: {}
    cuda_device_id: -1
    ring_len: 1024
    mempool_size:
      cpu_msg: {{buf_size: 15000, pool_len: 256}}
      cpu_data: {{buf_size: 576000, pool_len: 64}}
      cpu_large: {{buf_size: 4096000, pool_len: 16}}
      cuda_data: {{buf_size: 307200, pool_len: 0}}
      gpu_data: {{buf_size: 576000, pool_len: 0}}
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
)", prefix);
    // clang-format on
}

} // namespace ran::fapi
