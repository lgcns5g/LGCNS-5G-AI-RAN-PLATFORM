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

#ifndef RAN_ORAN_VEC_BUF_HPP
#define RAN_ORAN_VEC_BUF_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include "oran/oran_buf.hpp"
#include "oran/oran_export.hpp"

namespace ran::oran {

/**
 * Simple memory buffer implementation using std::vector
 */
class ORAN_EXPORT VecBuf final : public OranBuf {
private:
    std::vector<std::uint8_t> buffer_;
    std::size_t data_size_{};
    bool timestamp_flag_{};
    std::uint64_t timestamp_{};

public:
    /**
     * Construct buffer with specified capacity
     * @param[in] capacity Buffer capacity in bytes
     */
    explicit VecBuf(std::size_t capacity) : buffer_(capacity) {}

    [[nodiscard]] std::uint8_t *data() override { return buffer_.data(); }

    [[nodiscard]] const std::uint8_t *data() const override { return buffer_.data(); }

    [[nodiscard]] std::size_t capacity() const override { return buffer_.size(); }

    [[nodiscard]] std::size_t size() const override { return data_size_; }

    void set_size(std::size_t new_size) override {
        if (new_size <= capacity()) {
            data_size_ = new_size;
        }
    }

    void set_timestamp(const std::uint64_t timestamp) override {
        timestamp_ = timestamp;
        timestamp_flag_ = true;
    }

    void clear_flags() override { timestamp_flag_ = false; }

    /**
     * Get timestamp flag (for testing)
     * @return True if timestamp flag is set
     */
    [[nodiscard]] bool has_timestamp() const { return timestamp_flag_; }

    /**
     * Get timestamp value (for testing)
     * @return Timestamp value
     */
    [[nodiscard]] std::uint64_t get_timestamp() const { return timestamp_; }
};

} // namespace ran::oran

#endif // RAN_ORAN_VEC_BUF_HPP
