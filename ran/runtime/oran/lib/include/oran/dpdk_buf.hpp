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

#ifndef RAN_ORAN_MBUF_BUFFER_HPP
#define RAN_ORAN_MBUF_BUFFER_HPP

#include <cstddef>
#include <cstdint>
#include <format>
#include <mutex>
#include <optional>
#include <stdexcept>

#include "oran/oran_buf.hpp"
#include "oran/oran_export.hpp"

// Forward declaration
struct rte_mbuf;

namespace ran::oran {

/**
 * DPDK mbuf buffer implementation
 *
 * Wraps an rte_mbuf to provide the OranBuf interface.
 * This allows DPDK mbufs to be used with ORAN packet preparation functions.
 */
class ORAN_EXPORT MBuf final : public OranBuf {
private:
    rte_mbuf *mbuf_{};

    // Static DPDK timestamp configuration (initialized once, shared by all
    // instances)
    inline static std::optional<std::int32_t> timestamp_offset;
    inline static std::uint64_t timestamp_mask;

    /**
     * Initialize DPDK timestamp offsets (called once via std::call_once)
     */
    static void initialize_timestamp_offsets();

    /**
     * Check if index is within bounds
     * @param[in] index Element index
     * @throws std::out_of_range if index >= size()
     */
    void check_bounds(const std::size_t index) const {
        const std::size_t buffer_size = size();
        if (index >= buffer_size) {
            throw std::out_of_range(
                    std::format("MBuf::at: index {} out of range (size: {})", index, buffer_size));
        }
    }

public:
    /**
     * Construct buffer wrapping an rte_mbuf
     * @param[in] mbuf DPDK mbuf pointer
     */
    explicit MBuf(rte_mbuf *mbuf);

    [[nodiscard]] std::uint8_t *data() override;
    [[nodiscard]] const std::uint8_t *data() const override;
    [[nodiscard]] std::size_t capacity() const override;
    [[nodiscard]] std::size_t size() const override;

    /**
     * Set buffer size
     *
     * @param[in] new_size New buffer size in bytes
     * @throws std::length_error if new_size exceeds buffer capacity
     * @throws std::length_error if new_size exceeds maximum mbuf data length (65535 bytes)
     */
    void set_size(std::size_t new_size) override;
    void set_timestamp(std::uint64_t timestamp) override;
    void clear_flags() override;

    /**
     * Get underlying mbuf pointer
     * @return DPDK mbuf pointer
     */
    [[nodiscard]] rte_mbuf *get_mbuf() const { return mbuf_; }

    /**
     * Access element with bounds checking
     * @param[in] index Element index
     * @return Reference to element at specified index
     * @throws std::out_of_range if index >= size()
     */
    [[nodiscard]] std::uint8_t &at(const std::size_t index) {
        check_bounds(index);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        return data()[index];
    }

    /**
     * Access element with bounds checking (const version)
     * @param[in] index Element index
     * @return Const reference to element at specified index
     * @throws std::out_of_range if index >= size()
     */
    [[nodiscard]] const std::uint8_t &at(const std::size_t index) const {
        check_bounds(index);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        return data()[index];
    }
};

} // namespace ran::oran

#endif // RAN_ORAN_MBUF_BUFFER_HPP
