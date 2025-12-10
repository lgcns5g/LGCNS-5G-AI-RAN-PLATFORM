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

#ifndef FRAMEWORK_CORE_STRING_HASH_HPP
#define FRAMEWORK_CORE_STRING_HASH_HPP

#include <cstddef>
#include <functional>
#include <string_view>

namespace framework::utils {

/**
 * Transparent hash functor for string types that enables heterogeneous lookup.
 *
 * This allows std::unordered_map<std::string, T> to perform lookups with
 * string_view without constructing temporary strings, improving performance
 * by avoiding allocations.
 *
 * Background:
 * - C++14 introduced heterogeneous lookup for ordered containers (std::map,
 *   std::set) using transparent comparators like std::less<>
 * - C++20 extended this to unordered containers (std::unordered_map,
 *   std::unordered_set) requiring both transparent hash and comparator
 * - The is_transparent tag enables the container to accept different key types
 *
 * Example usage:
 * @code
 * std::unordered_map<std::string, Module, TransparentStringHash,
 * std::equal_to<>> modules;
 *
 * // Zero allocations - uses string_view directly
 * modules.find("key"sv);
 *
 * // Zero allocations - uses const char* directly
 * modules.contains("literal");
 *
 * // Works with std::string too
 * std::string key = "dynamic";
 * modules.find(key);
 * @endcode
 *
 * Performance benefits:
 * - Eliminates temporary std::string allocations during lookups
 * - Especially beneficial when using string literals or string_view keys
 * - No overhead compared to std::hash<std::string> for std::string keys
 *
 * Requirements:
 * - C++20 for heterogeneous lookup support in unordered containers
 * - Must be used with transparent comparator (e.g., std::equal_to<>)
 */
struct TransparentStringHash {
    /**
     * Tag enabling heterogeneous lookup.
     *
     * This is a standard library trait that must be named exactly this way for
     * C++20 heterogeneous lookup to work.
     *
     * is_transparent is a standard library trait that must be named exactly this
     * way for C++20 heterogeneous lookup to work.
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    using is_transparent = void; //!< Tag enabling heterogeneous lookup

    /**
     * Hash function for string_view and string-like types.
     *
     * Uses std::hash<std::string_view> which works efficiently with all
     * string-like types including std::string, std::string_view, and const char*.
     *
     * @param[in] sv String view to hash
     * @return Hash value
     */
    [[nodiscard]] std::size_t operator()(std::string_view sv) const noexcept {
        return std::hash<std::string_view>{}(sv);
    }
};

} // namespace framework::utils

#endif // FRAMEWORK_CORE_STRING_HASH_HPP
