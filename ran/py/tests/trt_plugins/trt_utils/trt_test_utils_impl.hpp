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
 * @file trt_test_utils_impl.hpp
 * @brief Template implementations for TensorRT test utilities
 *
 * This file contains the template method implementations for CudaTensor
 * that must be included in the header.
 */

#ifndef RAN_TRT_UTILS_IMPL_HPP
#define RAN_TRT_UTILS_IMPL_HPP

#include <algorithm>
#include <cmath>
#include <format>
#include <span>
#include <utility>

#include <cuda_runtime_api.h>

namespace ran::trt_utils {

// ============================================================================
// CudaTensor template implementations
// ============================================================================

// Private helper methods

template <typename T>
std::size_t CudaTensor<T>::compute_flat_index(const std::initializer_list<int64_t> indices) const {
    if (static_cast<int32_t>(indices.size()) != spec_.shape.nbDims) {
        throw std::invalid_argument("Number of indices doesn't match tensor dimensions");
    }

    // Calculate strides for row-major layout
    const std::span<const int64_t> shape_span(
            static_cast<const int64_t *>(spec_.shape.d), spec_.shape.nbDims);
    std::vector<std::size_t> strides(spec_.shape.nbDims);
    std::size_t stride = 1;
    for (int32_t i = spec_.shape.nbDims - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= static_cast<std::size_t>(shape_span[i]);
    }

    // Compute flat index: idx = i0*stride[0] + i1*stride[1] + ...
    std::size_t flat_idx = 0;
    int32_t i = 0;
    for (const auto idx : indices) {
        if (idx < 0 || idx >= shape_span[i]) {
            throw std::out_of_range("Index out of bounds");
        }
        flat_idx += static_cast<std::size_t>(idx) * strides[i];
        ++i;
    }
    return flat_idx;
}

template <typename T>
std::size_t CudaTensor<T>::compute_flat_index(const std::vector<int64_t> &indices) const {
    if (static_cast<int32_t>(indices.size()) != spec_.shape.nbDims) {
        throw std::invalid_argument("Number of indices doesn't match tensor dimensions");
    }

    // Calculate strides for row-major layout
    const std::span<const int64_t> shape_span(
            static_cast<const int64_t *>(spec_.shape.d), spec_.shape.nbDims);
    std::vector<std::size_t> strides(spec_.shape.nbDims);
    std::size_t stride = 1;
    for (int32_t i = spec_.shape.nbDims - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= static_cast<std::size_t>(shape_span[i]);
    }

    // Compute flat index: idx = i0*stride[0] + i1*stride[1] + ...
    std::size_t flat_idx = 0;
    for (int32_t i = 0; i < spec_.shape.nbDims; ++i) {
        const int64_t idx = indices[i];
        if (idx < 0 || idx >= shape_span[i]) {
            throw std::out_of_range("Index out of bounds");
        }
        flat_idx += static_cast<std::size_t>(idx) * strides[i];
    }
    return flat_idx;
}

// Validation helpers

template <typename T> bool CudaTensor<T>::has_non_zero() const {
    return std::any_of(host_data_.begin(), host_data_.end(), [](const T &val) {
        if constexpr (std::is_floating_point_v<T>) {
            return std::abs(val) > std::numeric_limits<T>::epsilon();
        } else if constexpr (std::is_same_v<T, __half>) {
            const float val_f = __half2float(val);
            return std::abs(val_f) > std::numeric_limits<float>::epsilon();
        } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            const float val_f = __bfloat162float(val);
            return std::abs(val_f) > std::numeric_limits<float>::epsilon();
        } else {
            return val != T{0};
        }
    });
}

template <typename T> bool CudaTensor<T>::all_non_zero() const {
    return std::all_of(host_data_.begin(), host_data_.end(), [](const T &val) {
        if constexpr (std::is_floating_point_v<T>) {
            return std::abs(val) > std::numeric_limits<T>::epsilon();
        } else if constexpr (std::is_same_v<T, __half>) {
            const float val_f = __half2float(val);
            return std::abs(val_f) > std::numeric_limits<float>::epsilon();
        } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            const float val_f = __bfloat162float(val);
            return std::abs(val_f) > std::numeric_limits<float>::epsilon();
        } else {
            return val != T{0};
        }
    });
}

template <typename T> std::size_t CudaTensor<T>::count_non_zero() const {
    return std::count_if(host_data_.begin(), host_data_.end(), [](const T &val) {
        if constexpr (std::is_floating_point_v<T>) {
            return std::abs(val) > std::numeric_limits<T>::epsilon();
        } else if constexpr (std::is_same_v<T, __half>) {
            const float val_f = __half2float(val);
            return std::abs(val_f) > std::numeric_limits<float>::epsilon();
        } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            const float val_f = __bfloat162float(val);
            return std::abs(val_f) > std::numeric_limits<float>::epsilon();
        } else {
            return val != T{0};
        }
    });
}

// Formatting helpers

/**
 * Format a value for display
 *
 * @tparam T Value type
 * @param[in] value Value to format
 * @return Formatted string representation
 */
template <typename T> inline std::string format_value(const T &value) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::format("{:.4f}", value);
    } else if constexpr (std::is_same_v<T, __half>) {
        return std::format("{:.4f}", __half2float(value));
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return std::format("{:.4f}", __bfloat162float(value));
    } else {
        return std::format("{}", value);
    }
}

/**
 * Format a complex value for display
 *
 * @tparam T Component type
 * @param[in] real_val Real component
 * @param[in] imag_val Imaginary component
 * @return Formatted string in "a + bj" format
 */
template <typename T>
inline std::string format_complex_value(const T &real_val, const T &imag_val) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::format("{:.4f} + {:.4f}j", real_val, imag_val);
    } else if constexpr (std::is_same_v<T, __half>) {
        return std::format("{:.4f} + {:.4f}j", __half2float(real_val), __half2float(imag_val));
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return std::format(
                "{:.4f} + {:.4f}j", __bfloat162float(real_val), __bfloat162float(imag_val));
    } else {
        return std::format("{} + {}j", real_val, imag_val);
    }
}

// Formatting methods

template <typename T>
std::string CudaTensor<T>::format(
        const std::initializer_list<std::variant<int64_t, Range>> indices,
        const int values_per_line) const {
    // Bounds check: verify indices size matches tensor dimensions
    if (static_cast<int32_t>(indices.size()) != spec_.shape.nbDims) {
        throw std::invalid_argument(std::format(
                "Number of indices ({}) doesn't match tensor dimensions ({})",
                indices.size(),
                spec_.shape.nbDims));
    }

    // Find the Range and validate there's exactly one
    int64_t slice_dim = -1;
    int64_t range_start = 0;
    int64_t range_end = 0;
    std::vector<int64_t> fixed_indices;
    fixed_indices.reserve(indices.size());

    // Build header string
    std::string header;
    if (!name_.empty()) {
        header = name_;
    } else {
        header = "Tensor";
    }
    header += "[";

    int64_t current_dim = 0;
    for (const auto &idx : indices) {
        if (std::holds_alternative<Range>(idx)) {
            if (slice_dim != -1) {
                throw std::invalid_argument("Exactly one Range must be specified");
            }
            slice_dim = current_dim;
            const auto &range = std::get<Range>(idx);
            range_start = range.start;
            range_end = range.end;
            fixed_indices.push_back(0); // Placeholder
            header += std::format("{}:{}", range_start, range_end);
        } else {
            const int64_t fixed_idx = std::get<int64_t>(idx);
            fixed_indices.push_back(fixed_idx);

            const std::span<const int64_t> shape_span(
                    static_cast<const int64_t *>(spec_.shape.d), spec_.shape.nbDims);
            // Bounds check: fixed index must be in range
            if (fixed_idx < 0 || fixed_idx >= shape_span[current_dim]) {
                throw std::out_of_range(std::format(
                        "Index {} at dimension {} is out of bounds [0, {})",
                        fixed_idx,
                        current_dim,
                        shape_span[current_dim]));
            }
            header += std::format("{}", fixed_idx);
        }

        if (current_dim < static_cast<int64_t>(indices.size()) - 1) {
            header += ", ";
        }
        ++current_dim;
    }
    header += "] = {\n";

    // Verify exactly one Range was found
    if (slice_dim == -1) {
        throw std::invalid_argument("Exactly one Range must be specified");
    }

    const std::span<const int64_t> shape_span(
            static_cast<const int64_t *>(spec_.shape.d), spec_.shape.nbDims);
    // Bounds check: Range must be valid
    if (range_start < 0 || range_start >= range_end || range_end > shape_span[slice_dim]) {
        throw std::out_of_range(std::format(
                "Range [{}:{}) at dimension {} is invalid (dimension size: {})",
                range_start,
                range_end,
                slice_dim,
                shape_span[slice_dim]));
    }

    // Build the formatted string
    std::string result = header;
    const int64_t num_values = range_end - range_start;
    static constexpr std::size_t BYTES_PER_VALUE = 15;
    static constexpr std::size_t BYTES_PER_NEWLINE = 3;
    static constexpr std::size_t EXTRA_BYTES = 10;
    result.reserve(
            header.size() + static_cast<std::size_t>(num_values) * BYTES_PER_VALUE +
            static_cast<std::size_t>(num_values / values_per_line) * BYTES_PER_NEWLINE +
            EXTRA_BYTES);
    result += "\t";

    for (int64_t i = range_start; i < range_end; ++i) {
        // Add line break and indent
        if (i > range_start && (i - range_start) % values_per_line == 0) {
            result += "\n\t";
        }

        // Build full index for this iteration
        fixed_indices[slice_dim] = i;

        // Format the value
        result += format_value((*this)(fixed_indices));

        // Add comma separator
        if (i < range_end - 1) {
            result += ", ";
        }
    }

    result += "\n}";
    return result;
}

/**
 * Validate range bounds for tensor slicing
 *
 * @param[in] range_start Start index
 * @param[in] range_end End index
 * @param[in] spec Tensor specification
 * @param[in] slice_dim Dimension being sliced
 * @throws std::out_of_range if range is invalid
 */
inline void validate_range(
        const int64_t range_start,
        const int64_t range_end,
        const TensorSpec &spec,
        const int64_t slice_dim) {
    const std::span<const int64_t> dims(&spec.shape.d[0], spec.shape.nbDims);
    if (range_start < 0 || range_start >= range_end || range_end > dims[slice_dim]) {
        throw std::out_of_range(std::format(
                "Range [{}:{}) at dimension {} is invalid (dimension size: {})",
                range_start,
                range_end,
                slice_dim,
                dims[slice_dim]));
    }
}

/**
 * Validate indices match complex tensor spec
 *
 * @param[in] indices Indices for tensor access
 * @param[in] spec Tensor specification
 * @throws std::invalid_argument if indices don't match spec or tensor is not complex
 */
inline void validate_indices_and_spec(
        const std::initializer_list<std::variant<int64_t, Range>> indices, const TensorSpec &spec) {
    // Bounds check: verify this is a complex tensor (dim 0 should be 2)
    if (spec.shape.nbDims < 2 || spec.shape.d[0] != 2) {
        throw std::invalid_argument(
                "format_complex requires dimension 0 to be size 2 [real, imag]");
    }

    // Bounds check: verify indices size matches remaining dimensions
    if (static_cast<int32_t>(indices.size()) != spec.shape.nbDims - 1) {
        throw std::invalid_argument(std::format(
                "Number of indices ({}) doesn't match tensor dimensions - 1 ({})",
                indices.size(),
                spec.shape.nbDims - 1));
    }
}

template <typename T>
std::string CudaTensor<T>::format_complex(
        const std::initializer_list<std::variant<int64_t, Range>> indices,
        const int values_per_line) const {
    validate_indices_and_spec(indices, spec_);

    const std::span<const int64_t> dims(&spec_.shape.d[0], spec_.shape.nbDims);

    // Build header string with tensor name
    std::string header = !name_.empty() ? name_ : "Tensor";
    header += "[";

    // Find the Range and validate there's exactly one
    int64_t slice_dim = -1; // Dimension in the user indices (0-based in user space)
    int64_t range_start = 0;
    int64_t range_end = 0;
    std::vector<int64_t> fixed_indices;
    fixed_indices.reserve(indices.size());

    int64_t current_dim = 0;
    for (const auto &idx : indices) {
        if (std::holds_alternative<Range>(idx)) {
            if (slice_dim != -1) {
                throw std::invalid_argument("Exactly one Range must be specified");
            }
            slice_dim = current_dim;
            const auto &range = std::get<Range>(idx);
            range_start = range.start;
            range_end = range.end;
            fixed_indices.push_back(0); // Placeholder
            header += std::format("{}:{}", range_start, range_end);
        } else {
            const int64_t fixed_idx = std::get<int64_t>(idx);
            fixed_indices.push_back(fixed_idx);

            // Bounds check: fixed index must be in range (current_dim+1 because dim 0 is real/imag)
            const int64_t actual_dim = current_dim + 1;
            if (fixed_idx < 0 || fixed_idx >= dims[actual_dim]) {
                throw std::out_of_range(std::format(
                        "Index {} at dimension {} is out of bounds [0, {})",
                        fixed_idx,
                        current_dim,
                        dims[actual_dim]));
            }
            header += std::format("{}", fixed_idx);
        }

        if (current_dim < static_cast<int64_t>(indices.size()) - 1) {
            header += ", ";
        }
        ++current_dim;
    }
    header += "] = {\n";

    // Verify exactly one Range was found
    if (slice_dim == -1) {
        throw std::invalid_argument("Exactly one Range must be specified");
    }

    // Bounds check: Range must be valid (slice_dim+1 because dim 0 is real/imag)
    const int64_t actual_slice_dim = slice_dim + 1;
    validate_range(range_start, range_end, spec_, actual_slice_dim);

    // Build the formatted string
    std::string result = header;
    const int64_t num_values = range_end - range_start;
    // Estimated bytes per complex value for string formatting
    static constexpr std::size_t BYTES_PER_COMPLEX_VALUE = 30;
    result.reserve(
            header.size() + num_values * BYTES_PER_COMPLEX_VALUE +
            num_values / values_per_line * 3); // Estimate for complex
    result += "\t";

    // Prepare indices for real and imag parts
    std::vector<int64_t> real_indices;
    std::vector<int64_t> imag_indices;
    real_indices.reserve(spec_.shape.nbDims);
    imag_indices.reserve(spec_.shape.nbDims);
    real_indices.push_back(0); // Real part (dim 0 = 0)
    imag_indices.push_back(1); // Imag part (dim 0 = 1)

    for (int64_t i = range_start; i < range_end; ++i) {
        // Add line break and indent
        if (i > range_start && (i - range_start) % values_per_line == 0) {
            result += "\n\t";
        }

        // Build full indices for this iteration
        fixed_indices[slice_dim] = i;

        // Create full dimensional indices (prepend real/imag dimension)
        real_indices.resize(1);
        imag_indices.resize(1);
        for (const auto &idx : fixed_indices) {
            real_indices.push_back(idx);
            imag_indices.push_back(idx);
        }

        // Get real and imag values
        const T real_val = (*this)(real_indices);
        const T imag_val = (*this)(imag_indices);

        // Format as complex number
        result += format_complex_value(real_val, imag_val);

        // Add comma separator
        if (i < range_end - 1) {
            result += ", ";
        }
    }

    result += "\n}";
    return result;
}

} // namespace ran::trt_utils

#endif // RAN_TRT_UTILS_IMPL_HPP
