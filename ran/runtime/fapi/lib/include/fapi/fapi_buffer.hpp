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

#ifndef RAN_FAPI_BUFFER_HPP
#define RAN_FAPI_BUFFER_HPP

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <type_traits>

#include <gsl-lite/gsl-lite.hpp>

namespace ran::fapi {

/**
 * Cast a pointer assuming it's safe
 *
 * Use this for pointers where the caller has already ensured the pointer
 * points to sufficient memory for type T (e.g., pointers returned from
 * allocation or message construction functions). Centralizes reinterpret_cast
 * with linter suppression.
 *
 * Performs compile-time and runtime alignment checks:
 * - Compile-time: Verifies type alignment is reasonable
 * - Runtime: Validates pointer is properly aligned for type T
 *
 * @note For __packed__ structs, alignof(T) == 1, so alignment checks
 *       verify byte-alignment only. The compiler handles unaligned field
 *       access, but this may cause performance penalties on some architectures.
 *
 * @note This cast is still technically UB per C++ standard (violates strict
 *       aliasing and object lifetime rules), but is widely used in systems/
 *       networking code with proper precautions.
 *
 * @tparam T Type to cast to
 * @param[in] ptr Pointer assumed to be safe
 * @return Pointer cast to type T
 */
template <typename T>
// cppcheck-suppress constParameterPointer  // Intentionally non-const for mutable overload
[[nodiscard]] inline T *assume_cast(void *ptr) {
    static_assert(
            alignof(T) <= alignof(std::max_align_t),
            "Type T has alignment requirements that may not be satisfied by arbitrary pointers");
    static_assert(std::is_standard_layout_v<T>, "Type T must be standard layout for safe casting");
    static_assert(std::is_trivial_v<T>, "Type T must be trivial for safe casting");

    // Runtime alignment check
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    const auto ptr_value = reinterpret_cast<std::uintptr_t>(ptr);
    gsl_Expects(ptr_value % alignof(T) == 0);

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    return reinterpret_cast<T *>(ptr);
}

/**
 * Cast a const pointer assuming it's safe
 *
 * Use this for pointers where the caller has already ensured the pointer
 * points to sufficient memory for type T (e.g., pointers from validated messages).
 * Centralizes reinterpret_cast with linter suppression.
 *
 * Performs compile-time and runtime alignment checks:
 * - Compile-time: Verifies type alignment is reasonable
 * - Runtime: Validates pointer is properly aligned for type T
 *
 * @note For __packed__ structs, alignof(T) == 1, so alignment checks
 *       verify byte-alignment only. The compiler handles unaligned field
 *       access, but this may cause performance penalties on some architectures.
 *
 * @note This cast is still technically UB per C++ standard (violates strict
 *       aliasing and object lifetime rules), but is widely used in systems/
 *       networking code with proper precautions.
 *
 * @tparam T Type to cast to
 * @param[in] ptr Pointer assumed to be safe
 * @return Const pointer cast to type T
 */
template <typename T> [[nodiscard]] inline const T *assume_cast(const void *ptr) {
    static_assert(
            alignof(T) <= alignof(std::max_align_t),
            "Type T has alignment requirements that may not be satisfied by arbitrary pointers");
    static_assert(std::is_standard_layout_v<T>, "Type T must be standard layout for safe casting");
    static_assert(std::is_trivial_v<T>, "Type T must be trivial for safe casting");

    // Runtime alignment check
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    const auto ptr_value = reinterpret_cast<std::uintptr_t>(ptr);
    gsl_Expects(ptr_value % alignof(T) == 0);

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    return reinterpret_cast<const T *>(ptr);
}

/**
 * Cast a reference assuming it's safe
 *
 * Use this for references where the caller has already ensured the reference
 * is properly typed (e.g., message body headers). Centralizes reinterpret_cast
 * with linter suppression.
 *
 * Performs compile-time and runtime alignment checks:
 * - Compile-time: Verifies type alignment is reasonable
 * - Runtime: Validates reference address is properly aligned for type T
 *
 * @note For __packed__ structs, alignof(T) == 1, so alignment checks
 *       verify byte-alignment only. The compiler handles unaligned field
 *       access, but this may cause performance penalties on some architectures.
 *
 * @note This cast is still technically UB per C++ standard (violates strict
 *       aliasing and object lifetime rules), but is widely used in systems/
 *       networking code with proper precautions.
 *
 * @tparam T Type to cast to
 * @tparam U Source type of reference
 * @param[in,out] ref Reference assumed to be safe
 * @return Reference cast to type T
 */
template <typename T, typename U> [[nodiscard]] inline T &assume_cast_ref(U &ref) {
    static_assert(
            alignof(T) <= alignof(std::max_align_t),
            "Type T has alignment requirements that may not be satisfied by arbitrary pointers");
    static_assert(std::is_standard_layout_v<T>, "Type T must be standard layout for safe casting");
    static_assert(std::is_trivial_v<T>, "Type T must be trivial for safe casting");

    // Runtime alignment check
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    const auto ptr_value = reinterpret_cast<std::uintptr_t>(&ref);
    gsl_Expects(ptr_value % alignof(T) == 0);

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    return reinterpret_cast<T &>(ref);
}

/**
 * Cast a const reference assuming it's safe
 *
 * Use this for references where the caller has already ensured the reference
 * is properly typed (e.g., message body headers). Centralizes reinterpret_cast
 * with linter suppression.
 *
 * Performs compile-time and runtime alignment checks:
 * - Compile-time: Verifies type alignment is reasonable
 * - Runtime: Validates reference address is properly aligned for type T
 *
 * @note For __packed__ structs, alignof(T) == 1, so alignment checks
 *       verify byte-alignment only. The compiler handles unaligned field
 *       access, but this may cause performance penalties on some architectures.
 *
 * @note This cast is still technically UB per C++ standard (violates strict
 *       aliasing and object lifetime rules), but is widely used in systems/
 *       networking code with proper precautions.
 *
 * @tparam T Type to cast to
 * @tparam U Source type of reference
 * @param[in] ref Reference assumed to be safe
 * @return Const reference cast to type T
 */
template <typename T, typename U> [[nodiscard]] inline const T &assume_cast_ref(const U &ref) {
    static_assert(
            alignof(T) <= alignof(std::max_align_t),
            "Type T has alignment requirements that may not be satisfied by arbitrary pointers");
    static_assert(std::is_standard_layout_v<T>, "Type T must be standard layout for safe casting");
    static_assert(std::is_trivial_v<T>, "Type T must be trivial for safe casting");

    // Runtime alignment check
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    const auto ptr_value = reinterpret_cast<std::uintptr_t>(&ref);
    gsl_Expects(ptr_value % alignof(T) == 0);

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    return reinterpret_cast<const T &>(ref);
}

/**
 * Create a byte span from a raw pointer and length
 *
 * @param[in] ptr Raw pointer to data
 * @param[in] length Length of the data in bytes
 * @return Byte span over the data
 */
[[nodiscard]] inline std::span<std::byte> make_buffer_span(void *ptr, const std::size_t length) {
    gsl_Expects(ptr != nullptr || length == 0);
    return std::span<std::byte>{assume_cast<std::byte>(ptr), length};
}

/**
 * Create a const byte span from a raw pointer and length
 *
 * @param[in] ptr Raw pointer to data
 * @param[in] length Length of the data in bytes
 * @return Const byte span over the data
 */
[[nodiscard]] inline std::span<const std::byte>
make_const_buffer_span(const void *ptr, const std::size_t length) {
    gsl_Expects(ptr != nullptr || length == 0);
    return std::span<const std::byte>{assume_cast<const std::byte>(ptr), length};
}

} // namespace ran::fapi

#endif // RAN_FAPI_BUFFER_HPP
