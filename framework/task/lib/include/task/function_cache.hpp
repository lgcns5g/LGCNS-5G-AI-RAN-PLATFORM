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

#ifndef FRAMEWORK_TASK_FUNCTION_CACHE_HPP
#define FRAMEWORK_TASK_FUNCTION_CACHE_HPP

#include <algorithm>
#include <array>
#include <atomic>
#include <compare>
#include <concepts>
#include <cstring>
#include <string_view>
#include <type_traits>

#include "task/flat_map.hpp"

namespace framework::task {

// Define concept for valid FixedString size
template <std::size_t N>
// NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
concept ValidFixedStringSize = N > 0 && N <= 65536; // reasonable upper limit

/**
 * Fixed-size string to avoid heap allocations
 *
 * Uses std::array for storage with automatic null-termination.
 */
template <std::size_t N>
    requires ValidFixedStringSize<N>
class FixedString final {
private:
    std::array<char, N> data_{};

    void copy_from(std::string_view str) noexcept {
        auto dest_span = std::span{data_.data(), N - 1}; // Reserve space for null terminator
        const auto copy_size = std::min(str.size(), dest_span.size());

        std::copy_n(str.data(), copy_size, dest_span.data());
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
        data_[copy_size] = '\0'; // Null terminate
    }

public:
    /// Default constructor - creates empty string
    explicit FixedString() { data_[0] = '\0'; }

    /**
     * Constructor from string view
     * @param[in] str Source string to copy
     */
    explicit FixedString(std::string_view str) { copy_from(str); }

    /**
     * Constructor from C string (for nullptr compatibility)
     * @param[in] str Source string to copy (nullptr creates empty string)
     */
    explicit FixedString(const char *str) {
        if (str != nullptr) {
            copy_from(std::string_view{str});
        } else {
            data_[0] = '\0';
        }
    }

    /**
     * Assignment from string view
     * @param[in] str Source string to copy
     * @return Reference to this object
     */
    FixedString &operator=(std::string_view str) {
        copy_from(str);
        return *this;
    }

    /**
     * Assignment from C string (for nullptr compatibility)
     * @param[in] str Source string to copy (nullptr creates empty string)
     * @return Reference to this object
     */
    FixedString &operator=(const char *str) {
        if (str != nullptr) {
            copy_from(std::string_view{str});
        } else {
            data_[0] = '\0';
        }
        return *this;
    }

    /**
     * Get null-terminated string (const version)
     * @return Pointer to null-terminated C-string
     */
    [[nodiscard]] const char *c_str() const noexcept { return data_.data(); }

    /**
     * Get null-terminated string (non-const version)
     * @return Pointer to null-terminated C-string
     */
    [[nodiscard]] char *c_str() noexcept { return data_.data(); }

    /**
     * Get string capacity (maximum size)
     * @return Maximum number of characters that can be stored
     */
    [[nodiscard]] constexpr std::size_t
    capacity() const noexcept { // cppcheck-suppress functionStatic
        return N;
    }

    /**
     * Get current string length
     * @return Number of characters in the string (excluding null terminator)
     */
    [[nodiscard]] std::size_t size() const noexcept { return std::strlen(data_.data()); }

    /**
     * Check if string is empty
     * @return True if string is empty
     */
    [[nodiscard]] bool empty() const noexcept { return data_[0] == '\0'; }

    /**
     * Get underlying array for direct access (non-const version)
     * @return Reference to underlying character array
     */
    [[nodiscard]] std::array<char, N> &data() noexcept { return data_; }

    /**
     * Get underlying array for direct access (const version)
     * @return Const reference to underlying character array
     */
    [[nodiscard]] const std::array<char, N> &data() const noexcept { return data_; }

    /**
     * Three-way comparison with another FixedString
     * @param[in] other String to compare with
     * @return Comparison result (strong ordering)
     */
    [[nodiscard]] std::strong_ordering operator<=>(const FixedString &other) const noexcept {
        const int result = std::strcmp(data_.data(), other.data_.data());
        if (result < 0) {
            return std::strong_ordering::less;
        }
        if (result > 0) {
            return std::strong_ordering::greater;
        }
        return std::strong_ordering::equal;
    }

    /**
     * Equality comparison with another FixedString
     * @param[in] other String to compare with
     * @return True if strings are equal
     */
    [[nodiscard]] bool operator==(const FixedString &other) const noexcept {
        return std::strcmp(data_.data(), other.data_.data()) == 0;
    }

    /**
     * Three-way comparison with string view
     * @param[in] str String to compare with
     * @return Comparison result (strong ordering)
     */
    [[nodiscard]] std::strong_ordering operator<=>(std::string_view str) const noexcept {
        return std::string_view{data_.data()} <=> str;
    }

    /**
     * Equality comparison with string view
     * @param[in] str String to compare with
     * @return True if strings are equal
     */
    [[nodiscard]] bool operator==(std::string_view str) const noexcept {
        return std::string_view{data_.data()} == str;
    }

    /**
     * Equality comparison with C string (for nullptr compatibility)
     * @param[in] str String to compare with (nullptr treated as empty string)
     * @return True if strings are equal
     */
    [[nodiscard]] bool operator==(const char *str) const noexcept {
        if (str != nullptr) {
            return std::string_view{data_.data()} == std::string_view{str};
        }
        return data_[0] == '\0';
    }
};

/**
 * High-performance function name cache with automatic eviction
 *
 * Caches function names by address using fixed-size strings to avoid
 * allocations. Supports automatic C++ name demangling and provides cache
 * statistics.
 */
class FunctionCache final {
private:
    static constexpr std::size_t NAME_SIZE = 256; //!< Size for demangled C++ names
    using NameString = FixedString<NAME_SIZE>;    //!< Type alias for fixed-size
                                                  //!< function name strings
    using CacheMap = FlatMap<void *, NameString>; //!< Type alias for address-to-name cache map

    // Default cache configuration constants
    static constexpr std::size_t DEFAULT_MAX_SIZE = 16384; //!< Default maximum cache entries
    static constexpr std::size_t DEFAULT_FULL_PERCENTAGE =
            90; //!< Default eviction trigger percentage
    static constexpr std::size_t DEFAULT_EVICT_PERCENTAGE = 25; //!< Default eviction percentage

    CacheMap cache_; //!< Function address to name mapping cache

    // Cache statistics (per-instance)
    std::atomic<std::uint64_t> cache_hits_{0};     //!< Number of successful cache lookups
    std::atomic<std::uint64_t> cache_misses_{0};   //!< Number of failed cache lookups
    std::atomic<std::uint64_t> cache_attempts_{0}; //!< Total number of cache access attempts

public:
    /**
     * Constructor
     *
     * @param[in] max_size Maximum number of cached entries
     * @param[in] full_percentage Percentage full at which eviction triggers
     * @param[in] evict_percentage Percentage of entries to evict
     */
    explicit FunctionCache(
            std::size_t max_size = DEFAULT_MAX_SIZE,
            std::size_t full_percentage = DEFAULT_FULL_PERCENTAGE,
            std::size_t evict_percentage = DEFAULT_EVICT_PERCENTAGE);

    /**
     * Get a function name from the cache
     *
     * @param[in] addr Function address
     * @return Cached function name or nullptr if not found
     */
    [[nodiscard]] const char *get(void *addr);

    /**
     * Add a function name to the cache with automatic demangling
     *
     * @param[in] addr Function address
     */
    void add_with_demangling(void *addr);

    /**
     * Add a function name to the cache
     *
     * @param[in] addr Function address
     * @param[in] name_str Function name string
     */
    void add(void *addr, std::string_view name_str);

    /**
     * Add a function name to the cache (C string overload for nullptr
     * compatibility)
     *
     * @param[in] addr Function address
     * @param[in] name_str Function name string (nullptr creates empty string)
     */
    void add(void *addr, const char *name_str);

    /**
     * Clear all cached entries
     */
    void clear();

    /**
     * Get number of cached entries
     * @return Number of cached function entries
     */
    [[nodiscard]] std::size_t size() const;

    /**
     * Get cache hit count
     * @return Number of cache hits
     */
    [[nodiscard]] std::uint64_t get_cache_hits() const;

    /**
     * Get cache miss count
     * @return Number of cache misses
     */
    [[nodiscard]] std::uint64_t get_cache_misses() const;

    /**
     * Get total cache access attempts
     * @return Total number of cache access attempts
     */
    [[nodiscard]] std::uint64_t get_cache_attempts() const;

    /**
     * Manual eviction control
     * @param[in] percentage Percentage of entries to evict
     */
    void evict_percentage(std::size_t percentage);

    /**
     * Get cache hit ratio as percentage
     * @return Cache hit ratio as percentage (0.0 to 100.0)
     */
    [[nodiscard]] double get_hit_ratio() const;
};

} // namespace framework::task

#endif // FRAMEWORK_TASK_FUNCTION_CACHE_HPP
