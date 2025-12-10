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

#ifndef RAN_ORAN_ORAN_BUF_HPP
#define RAN_ORAN_ORAN_BUF_HPP

#include <cstddef>
#include <cstdint>

#include "fapi/fapi_buffer.hpp"
#include "oran/oran_export.hpp"

namespace ran::oran {

/**
 * Abstract buffer interface for packet data
 */
class ORAN_EXPORT OranBuf {
public:
    OranBuf() = default;
    virtual ~OranBuf() = default;

    /**
     * Copy constructor.
     */
    OranBuf(const OranBuf &) = default;

    /**
     * Copy assignment operator.
     * @return Reference to this object
     */
    OranBuf &operator=(const OranBuf &) = default;

    /**
     * Move constructor.
     */
    OranBuf(OranBuf &&) = default;

    /**
     * Move assignment operator.
     * @return Reference to this object
     */
    OranBuf &operator=(OranBuf &&) = default;

    /**
     * Get buffer data pointer
     * @return Pointer to buffer data
     */
    [[nodiscard]] virtual std::uint8_t *data() = 0;

    /**
     * Get buffer data pointer (const)
     * @return Const pointer to buffer data
     */
    [[nodiscard]] virtual const std::uint8_t *data() const = 0;

    /**
     * Get buffer capacity
     * @return Maximum buffer size in bytes
     */
    [[nodiscard]] virtual std::size_t capacity() const = 0;

    /**
     * Get current data size
     * @return Current data size in bytes
     */
    [[nodiscard]] virtual std::size_t size() const = 0;

    /**
     * Set data size
     * @param[in] new_size New data size in bytes
     */
    virtual void set_size(std::size_t new_size) = 0;

    /**
     * Set timestamp value (for DPDK mbuf compatibility)
     *
     * This automatically sets the timestamp flag when needed.
     * @param[in] timestamp Timestamp value
     */
    virtual void set_timestamp([[maybe_unused]] std::uint64_t timestamp) {}

    /**
     * Clear all buffer flags (for DPDK mbuf compatibility)
     */
    virtual void clear_flags() {}

    /**
     * Template method for typed access to buffer data
     * @tparam T Type to cast buffer data to
     * @return Pointer to buffer data cast as type T
     */
    template <typename T> [[nodiscard]] T *data_as() { return ran::fapi::assume_cast<T>(data()); }

    /**
     * Template method for typed access at offset
     * @tparam T Type to cast buffer data to
     * @param[in] offset Byte offset into buffer
     * @return Pointer to buffer data at offset cast as type T
     */
    template <typename T> [[nodiscard]] T *data_at_offset(std::size_t offset) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        return ran::fapi::assume_cast<T>(data() + offset);
    }
};

} // namespace ran::oran

#endif // RAN_ORAN_ORAN_BUF_HPP
