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

#ifndef RAN_MESSAGE_ADAPTER_PHY_STATS_HPP
#define RAN_MESSAGE_ADAPTER_PHY_STATS_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

namespace ran::message_adapter {

/**
 * @brief PHY statistics collector
 *
 * Thread-safe statistics collection using lock-free atomics with relaxed memory ordering.
 * Per-cell granularity for multi-cell debugging and validation.
 *
 * Performance: ~1-2 CPU cycles per counter increment using relaxed atomics.
 * Memory: 8 bytes per cell (std::atomic<uint64_t>)
 *
 * Thread Safety: All methods are thread-safe and lock-free.
 */
class PhyStats final {
public:
    /**
     * @brief Construct statistics collector for specified number of cells
     *
     * @param[in] max_cells Maximum number of cells to track
     */
    explicit PhyStats(std::size_t max_cells);

    /**
     * @brief Record a CRC failure for the specified cell
     *
     * @param[in] cell_id Cell identifier (must be < max_cells)
     */
    void record_crc_failure(std::uint32_t cell_id) noexcept;

    /**
     * @brief Get the statistics
     *
     * @return Statistics
     */
    [[nodiscard]] std::vector<std::uint64_t> get_stats() const;

private:
    std::vector<std::atomic_uint64_t> per_cell_crc_failures_count_; //!< Per-cell CRC failure counts
};

} // namespace ran::message_adapter

#endif // RAN_MESSAGE_ADAPTER_PHY_STATS_HPP
