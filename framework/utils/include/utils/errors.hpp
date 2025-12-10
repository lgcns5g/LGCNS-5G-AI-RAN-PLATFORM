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

#ifndef FRAMEWORK_ERRORS_HPP
#define FRAMEWORK_ERRORS_HPP

#include <array>
#include <concepts>
#include <cstdint>
#include <limits>
#include <source_location> // NOLINT(unused-includes)
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>

#include <wise_enum.h>

#include "log/rt_log_macros.hpp"
#include "utils/core_log.hpp"
#include "utils/details/underlying.hpp"

namespace framework::utils {

/**
 * NV error codes compatible with std::error_code
 *
 * This enum class provides a type-safe wrapper around nvStatus_t values
 * that integrates seamlessly with the standard C++ error handling framework.
 */
// clang-format off
enum class NvErrc : std::uint8_t {
    Success,             //!< The API call returned with no errors
    InternalError,       //!< An unexpected, internal error occurred
    NotSupported,        //!< The requested function is not currently supported
    InvalidArgument,     //!< One or more of the arguments provided to the function was invalid
    ArchMismatch,        //!< The requested operation is not supported on the current architecture
    AllocFailed,         //!< A memory allocation failed
    SizeMismatch,        //!< The size of the operands provided to the function do not match
    MemcpyError,         //!< An error occurred during a memcpy operation
    InvalidConversion,   //!< An invalid conversion operation was requested
    UnsupportedType,     //!< An operation was requested on an unsupported type
    UnsupportedLayout,   //!< An operation was requested on an unsupported layout
    UnsupportedRank,     //!< An operation was requested on an unsupported rank
    UnsupportedConfig,   //!< An operation was requested on an unsupported configuration
    UnsupportedAlignment, //!< One or more API arguments don't have the required alignment
    ValueOutOfRange,     //!< Data conversion could not occur because an input value was out of range
    RefMismatch          //!< Mismatch found when comparing to TV
};
// clang-format on

static_assert(
        static_cast<std::uint32_t>(NvErrc::RefMismatch) <= std::numeric_limits<std::uint8_t>::max(),
        "NvErrc enumerator values must fit in std::uint8_t");

} // namespace framework::utils

// WISE_ENUM_ADAPT must be called at global namespace scope for ADL to work
WISE_ENUM_ADAPT(
        framework::utils::NvErrc,
        Success,
        InternalError,
        NotSupported,
        InvalidArgument,
        ArchMismatch,
        AllocFailed,
        SizeMismatch,
        MemcpyError,
        InvalidConversion,
        UnsupportedType,
        UnsupportedLayout,
        UnsupportedRank,
        UnsupportedConfig,
        UnsupportedAlignment,
        ValueOutOfRange,
        RefMismatch)

namespace framework::utils {

/**
 * Custom error category for NV errors
 *
 * This class provides human-readable error messages and integrates NV errors
 * with the standard C++ error handling system.
 */
class NvErrorCategory final : public std::error_category {
private:
    // Compile-time table indexed by the enum's underlying value.
    static constexpr std::array<std::string_view, 16> KMESSAGES{
            "Success: The API call returned with no errors",
            "Internal error: An unexpected, internal error occurred",
            "Not supported: The requested function is not currently supported",
            "Invalid argument: One or more of the arguments provided to the function "
            "was invalid",
            "Architecture mismatch: The requested operation is not supported on the "
            "current architecture",
            "Allocation failed: A memory allocation failed",
            "Size mismatch: The size of the operands provided to the function do not "
            "match",
            "Memory copy error: An error occurred during a memcpy operation",
            "Invalid conversion: An invalid conversion operation was requested",
            "Unsupported type: An operation was requested on an unsupported type",
            "Unsupported layout: An operation was requested on an unsupported layout",
            "Unsupported rank: An operation was requested on an unsupported rank",
            "Unsupported config: An operation was requested on an unsupported "
            "configuration",
            "Unsupported alignment: One or more API arguments don't have the "
            "required alignment",
            "Value out of range: Data conversion could not occur because an input "
            "value was out of range",
            "Reference mismatch: Mismatch found when comparing to TV"};

    // Ensure KMESSAGES array size matches the number of enum values
    static_assert(
            KMESSAGES.size() == ::wise_enum::size<NvErrc>,
            "KMESSAGES array size must match the number of NvErrc enum values");

public:
    /**
     * Get the name of this error category
     *
     * @return The category name as a C-style string
     */
    [[nodiscard]] const char *name() const noexcept override { return "adsp::core"; }

    /**
     * Get a descriptive message for the given error code – O(1) lookup, no heap
     * allocation except for the implicit std::string construction required by
     * the std::error_category interface.
     *
     * @param[in] condition The error code value
     * @return A descriptive error message
     */
    [[nodiscard]] std::string message(const int condition) const override {
        const auto idx = static_cast<std::size_t>(condition);
        if (idx < KMESSAGES.size()) {
            return std::string{*std::next(KMESSAGES.begin(), static_cast<std::ptrdiff_t>(idx))};
        }
        return "Unknown NV error: " + std::to_string(condition);
    }

    /**
     * Map NV errors to standard error conditions where applicable
     *
     * @param[in] condition The error code value
     * @return The equivalent standard error condition, or a default-constructed
     * condition
     */
    [[nodiscard]] std::error_condition
    default_error_condition(const int condition) const noexcept override {
        switch (static_cast<NvErrc>(condition)) {
        case NvErrc::Success:
            return {};
        case NvErrc::InvalidArgument:
            return std::errc::invalid_argument;
        case NvErrc::NotSupported:
            return std::errc::operation_not_supported;
        case NvErrc::AllocFailed:
            return std::errc::not_enough_memory;
        case NvErrc::ValueOutOfRange:
            return std::errc::result_out_of_range;
        default:
            // For NV-specific errors that don't map to standard conditions
            return std::error_condition{condition, *this};
        }
    }

    /**
     * Get the name of the error code enum value
     *
     * @param[in] condition The error code value
     * @return The enum name as a string (e.g., "success", "internal_error")
     */
    [[nodiscard]] static const char *name(const int condition) {
        const auto errc = static_cast<NvErrc>(condition);
        return ::wise_enum::to_string(errc).data();
    }
};

/**
 * Get the singleton instance of the NV error category
 *
 * @return Reference to the NV error category
 */
[[nodiscard]] inline const NvErrorCategory &nv_category() noexcept {
    static const NvErrorCategory instance{};
    return instance;
}

/**
 * Create an error_code from a NvErrc value
 *
 * @param[in] errc The NV error code
 * @return A std::error_code representing the NV error
 */
[[nodiscard]] inline std::error_code make_error_code(const NvErrc errc) noexcept {
    return {to_underlying(errc), nv_category()};
}

/**
 * Convert a raw nvStatus_t value to NvErrc
 *
 * @param[in] status Raw nvStatus_t value
 * @return Equivalent NvErrc value
 *
 * @note This function performs a static_cast and assumes the input is valid
 */
[[nodiscard]] constexpr NvErrc from_nv_status(const int status) noexcept {
    return static_cast<NvErrc>(status);
}

/**
 * Create an error_code from a raw nvStatus_t value
 *
 * @param[in] status Raw nvStatus_t value
 * @return A std::error_code representing the NV error
 */
[[nodiscard]] inline std::error_code make_error_code(const int status) noexcept {
    return make_error_code(from_nv_status(status));
}

/**
 * Check if a NvErrc represents success
 *
 * @param[in] errc The error code to check
 * @return true if the error code represents success, false otherwise
 */
[[nodiscard]] constexpr bool is_success(const NvErrc errc) noexcept {
    return errc == NvErrc::Success;
}

/**
 * Check if an error_code represents NV success
 *
 * @param[in] errc The error code to check
 * @return true if the error code represents NV success, false otherwise
 */
[[nodiscard]] inline bool is_nv_success(const std::error_code &errc) noexcept {
    return errc.category() == nv_category() && errc.value() == 0;
}

/**
 * Get the name of a NvErrc enum value
 *
 * @param[in] errc The error code
 * @return The enum name as a string
 */
[[nodiscard]] inline const char *get_error_name(const NvErrc errc) noexcept {
    return ::wise_enum::to_string(errc).data();
}

} // namespace framework::utils

// Register NvErrc as an error code enum to enable implicit conversion to
// std::error_code
namespace std {
template <> struct is_error_code_enum<framework::utils::NvErrc> : true_type {};
} // namespace std

#endif // FRAMEWORK_ERRORS_HPP
