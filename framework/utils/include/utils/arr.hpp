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

#ifndef FRAMEWORK_ARR_HPP
#define FRAMEWORK_ARR_HPP

#include <array>
#include <stdexcept>

#include "cuda_defines.hpp"

namespace framework::utils {

/**
 * Fixed-size array container for mathematical operations
 *
 * This class provides a lightweight, fixed-size array container optimized for
 * mathematical operations in CUDA environments. It supports both host and
 * device execution contexts and provides STL-compatible iterators.
 *
 * @note The array uses std::array for internal storage, which is fully
 * compatible with CUDA device code when compiled with --expt-relaxed-constexpr.
 * This is because std::array is a POD type and can be used in constexpr
 * functions.
 *
 * @tparam T The element type (must be default constructible)
 * @tparam Dim The number of elements in the array (must be > 0)
 */
template <typename T, std::size_t DIM> class Arr final {
    static_assert(DIM > 0, "Arr dimension must be greater than zero");

public:
    /**
     * Default constructor - zero-initializes all elements
     *
     * Creates an array with all elements initialized to their default value
     * (zero for numeric types, false for bool, etc.).
     */
    CUDA_BOTH_INLINE constexpr Arr() noexcept = default;

    /**
     * Array constructor - initializes from C-style array
     *
     * Constructs the array by copying elements from the provided C-style array.
     * The array size must exactly match the array dimension.
     *
     * @tparam N The size of the input array (must equal DIM)
     * @param[in] arr The source C-style array reference to copy from
     *
     * @note This constructor will fail to compile if N != DIM due to
     * static_assert
     */
    // clang-format off
  template <std::size_t N>
  CUDA_BOTH_INLINE explicit constexpr Arr(const T (&arr)[N]) { // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays)
    static_assert(N == DIM, "Input array size must equal DIM");
    initialize_from_indexed_container(arr);
  }
    // clang-format on

    /**
     * Array constructor - initializes from std::array
     *
     * Constructs the array by copying elements from the provided std::array.
     * The array size must exactly match the array dimension.
     *
     * @tparam N The size of the input array (must equal DIM)
     * @param[in] arr The source std::array to copy from
     *
     * @note This constructor will fail to compile if N != DIM due to
     * static_assert
     */
    template <std::size_t N> CUDA_BOTH_INLINE explicit Arr(const std::array<T, N> &arr) {
        static_assert(N == DIM, "Array size does not match array dimension");
        initialize_from_indexed_container(arr);
    }

    /**
     * Fill all elements with the same value
     *
     * Sets all elements of the array to the specified value.
     *
     * @param[in] val The value to assign to all elements
     */
    CUDA_BOTH_INLINE
    void fill(T val) {
        for (std::size_t i = 0; i < DIM; ++i) {
            // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
            elem_[i] = val;
        }
    }

    /**
     * Access element by index (mutable)
     *
     * Provides direct access to the element at the specified index.
     * No bounds checking is performed.
     *
     * @param[in] idx The index of the element to access
     * @return Reference to the element at the specified index
     *
     * @note No bounds checking is performed for performance reasons
     */
    [[nodiscard]] CUDA_BOTH_INLINE T &operator[](std::size_t idx) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
        return elem_[idx];
    }

    /**
     * Access element by index (immutable)
     *
     * Provides read-only access to the element at the specified index.
     * No bounds checking is performed.
     *
     * @param[in] idx The index of the element to access
     * @return Const reference to the element at the specified index
     *
     * @note No bounds checking is performed for performance reasons
     */
    [[nodiscard]] CUDA_BOTH_INLINE const T &operator[](std::size_t idx) const {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
        return elem_[idx];
    }

    /**
     * Get mutable iterator to beginning
     *
     * Returns a pointer to the first element, enabling STL-style iteration.
     *
     * @return Pointer to the first element
     */
    [[nodiscard]] CUDA_BOTH_INLINE constexpr T *begin() noexcept { return elem_.data(); }

    /**
     * Get mutable iterator to end
     *
     * Returns a pointer to one past the last element, enabling STL-style
     * iteration.
     *
     * @return Pointer to one past the last element
     */
    [[nodiscard]] CUDA_BOTH_INLINE constexpr T *end() noexcept { return elem_.data() + DIM; }

    /**
     * Get const iterator to beginning
     *
     * Returns a const pointer to the first element, enabling STL-style iteration
     * for const arrays.
     *
     * @return Const pointer to the first element
     */
    [[nodiscard]] CUDA_BOTH_INLINE constexpr const T *begin() const noexcept {
        return elem_.data();
    }

    /**
     * Get const iterator to end
     *
     * Returns a const pointer to one past the last element, enabling STL-style
     * iteration for const arrays.
     *
     * @return Const pointer to one past the last element
     */
    [[nodiscard]] CUDA_BOTH_INLINE constexpr const T *end() const noexcept {
        return elem_.data() + DIM;
    }

    /**
     * Get the size of the array
     *
     * Returns the number of elements in the array.
     *
     * @return The number of elements in the array
     */
    [[nodiscard]] static constexpr std::size_t size() noexcept { return DIM; }

    /**
     * Get a pointer to the data of the array
     *
     * Returns a pointer to the first element of the array.
     *
     * @return A pointer to the first element of the array
     */
    [[nodiscard]] CUDA_BOTH_INLINE constexpr T *data() noexcept { return elem_.data(); }

    /**
     * Get a const pointer to the data of the array
     *
     * Returns a const pointer to the first element of the array.
     *
     * @return A const pointer to the first element of the array
     */
    [[nodiscard]] CUDA_BOTH_INLINE constexpr const T *data() const noexcept { return elem_.data(); }

    /**
     * Equality comparison operator
     *
     * Compares two arrays element-wise for equality. Uses index-based comparison
     * for CUDA-compatible operation.
     *
     * @param[in] lhs The first array to compare
     * @param[in] rhs The second array to compare
     * @return True if all corresponding elements are equal, false otherwise
     */
    [[nodiscard]] CUDA_BOTH_INLINE friend bool operator==(const Arr &lhs, const Arr &rhs) {
        for (std::size_t i = 0; i < DIM; ++i) {
            // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
            if (lhs.elem_[i] != rhs.elem_[i]) {
                return false;
            }
        }
        return true;
    }

    /**
     * Inequality comparison operator
     *
     * Compares two arrays element-wise for inequality.
     *
     * @param[in] lhs The first array to compare
     * @param[in] rhs The second array to compare
     * @return True if any corresponding elements are not equal, false otherwise
     */
    [[nodiscard]] CUDA_BOTH_INLINE friend bool operator!=(const Arr &lhs, const Arr &rhs) {
        return !(lhs == rhs);
    }

    /**
     * Calculate the product of all elements
     *
     * Computes the product of all elements in the array (a₀ × a₁ × ... × aₙ₋₁).
     *
     * @return The product of all elements
     */
    [[nodiscard]] CUDA_BOTH_INLINE T product() const {
        T result = elem_[0];
        for (std::size_t i = 1; i < DIM; ++i) {
            // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
            result *= elem_[i];
        }
        return result;
    }

private:
    std::array<T, DIM> elem_{}; //!< Internal storage array, zero-initialized

    /**
     * Helper method to initialize the array from an indexed container (C-array
     * or std::array)
     *
     * This method is used by both the C-array and std::array constructors to
     * copy elements from the provided container into the internal array.
     *
     * @tparam Container The type of the container (C-array or std::array)
     * @param[in] container The container to copy from
     */
    template <typename Container>
    CUDA_BOTH_INLINE void initialize_from_indexed_container(const Container &container) {
        for (std::size_t i = 0; i < DIM; ++i) {
            // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
            elem_[i] = container[i];
        }
    }
};

} // namespace framework::utils

#endif // FRAMEWORK_ARR_HPP
