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

#ifndef FRAMEWORK_CORE_TENSOR_INFO_HPP
#define FRAMEWORK_CORE_TENSOR_INFO_HPP

#include <vector>

#include "tensor/data_types.hpp"

namespace framework::tensor {

/**
 * @class TensorInfo
 * @brief Describes tensor properties for ABI validation between modules.
 *
 * This class encapsulates all the necessary information about a tensor,
 * including its data type, dimensions, and other metadata to ensure
 * compatibility between modules.
 */
class TensorInfo final {
public:
    using DataType = NvDataType; //!< Data type alias for tensor elements

    /**
     * Default constructor.
     */
    TensorInfo() = default;

    /**
     * Constructor with data type and dimensions.
     *
     * @param[in] type The data type of the tensor
     * @param[in] dimensions The dimensions of the tensor
     * @throws std::invalid_argument if any dimension is zero
     */
    TensorInfo(DataType type, std::vector<std::size_t> dimensions);

    /**
     * Get the data type of the tensor.
     *
     * @return The data type
     */
    [[nodiscard]] DataType get_type() const noexcept;

    /**
     * Get the dimensions of the tensor.
     *
     * @return A const reference to the dimensions vector
     */
    [[nodiscard]] const std::vector<std::size_t> &get_dimensions() const noexcept;

    /**
     * Check if this TensorInfo is compatible with another.
     *
     * @param[in] other The TensorInfo to check compatibility with
     * @return true if compatible, false otherwise
     */
    [[nodiscard]] bool is_compatible_with(const TensorInfo &other) const noexcept;

    /**
     * Get the total number of elements in the tensor.
     *
     * @return The total number of elements
     */
    [[nodiscard]] std::size_t get_total_elements() const;

    /**
     * Set the total size in bytes for this tensor.
     * This is typically called after calculating the size based on data type
     * and shape.
     *
     * @param[in] size_bytes The total size in bytes
     */
    void set_size_bytes(std::size_t size_bytes);

    /**
     * Get the total size in bytes for this tensor.
     *
     * @return The total size in bytes (0 if not set)
     */
    [[nodiscard]] std::size_t get_total_size_in_bytes() const;

private:
    DataType type_{DataType::TensorC32F}; //!< The data type of the tensor
    std::vector<std::size_t> dimensions_; //!< The dimensions of the tensor
    std::size_t size_bytes_{0};           //!< Total size in bytes (cached)
};

} // namespace framework::tensor

#endif // FRAMEWORK_CORE_TENSOR_INFO_HPP
