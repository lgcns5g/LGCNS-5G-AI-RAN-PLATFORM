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

#include <algorithm>  // for equal, __any_of_fn, any_of
#include <cstddef>    // for size_t
#include <functional> // for identity, multiplies
#include <numeric>    // for accumulate
#include <stdexcept>  // for invalid_argument
#include <utility>    // for move
#include <vector>     // for vector, operator==

#include <quill/LogMacros.h> // for QUILL_LOG_ERROR

#include "tensor/tensor_info.hpp" // for TensorInfo
#include "utils/error_macros.hpp" // for FRAMEWORK_NV_THROW_IF

namespace framework::tensor {

TensorInfo::TensorInfo(const DataType type, std::vector<std::size_t> dimensions)
        : type_(type), dimensions_(std::move(dimensions)) {
    FRAMEWORK_NV_THROW_IF(
            std::ranges::any_of(dimensions_, [](std::size_t dim) { return dim == 0; }),
            std::invalid_argument,
            "Tensor dimensions cannot contain zero");
}

TensorInfo::DataType TensorInfo::get_type() const noexcept { return type_; }

const std::vector<std::size_t> &TensorInfo::get_dimensions() const noexcept { return dimensions_; }

bool TensorInfo::is_compatible_with(const TensorInfo &other) const noexcept {
    return type_ == other.type_ && dimensions_ == other.dimensions_;
}

std::size_t TensorInfo::get_total_elements() const {
    if (dimensions_.empty()) {
        return 0;
    }

    const std::size_t total = std::accumulate(
            dimensions_.begin(), dimensions_.end(), std::size_t{1}, std::multiplies<std::size_t>{});
    return total;
}

void TensorInfo::set_size_bytes(std::size_t size_bytes) { size_bytes_ = size_bytes; }

std::size_t TensorInfo::get_total_size_in_bytes() const { return size_bytes_; }

} // namespace framework::tensor
