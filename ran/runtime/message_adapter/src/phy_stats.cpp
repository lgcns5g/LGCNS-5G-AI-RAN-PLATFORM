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
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <vector>

#include "message_adapter/phy_stats.hpp"

namespace ran::message_adapter {

PhyStats::PhyStats(std::size_t max_cells) : per_cell_crc_failures_count_(max_cells) {
    // C++20: std::atomic default constructor guarantees zero-initialization
}

void PhyStats::record_crc_failure(const std::uint32_t cell_id) noexcept {
    // Bounds check to prevent out-of-range access
    if (cell_id < per_cell_crc_failures_count_.size()) {
        per_cell_crc_failures_count_[cell_id].fetch_add(1, std::memory_order_relaxed);
    }
}

std::vector<std::uint64_t> PhyStats::get_stats() const {
    std::vector<std::uint64_t> crc_failures;
    crc_failures.reserve(per_cell_crc_failures_count_.size());
    std::ranges::transform(
            per_cell_crc_failures_count_,
            std::back_inserter(crc_failures),
            [](const auto &counter) { return counter.load(std::memory_order_relaxed); });
    return crc_failures;
}

} // namespace ran::message_adapter
