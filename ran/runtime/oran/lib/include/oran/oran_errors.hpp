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
 * @file oran_errors.hpp
 * @brief Error codes for ORAN C-plane operations
 *
 * Provides type-safe error codes compatible with std::error_code
 * for ORAN C-plane message preparation and FAPI conversion operations.
 */

#ifndef RAN_ORAN_ERRORS_HPP
#define RAN_ORAN_ERRORS_HPP

#include <array>
#include <cstdint>
#include <format>
#include <limits>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>

#include <wise_enum.h>

#include "oran/oran_export.hpp"

namespace ran::oran {

/**
 * ORAN framework error codes compatible with std::error_code
 *
 * This enum class provides type-safe error codes for ORAN operations
 * that integrate seamlessly with the standard C++ error handling framework.
 */
// clang-format off
enum class OranErrc : std::uint8_t {
    Success,                              //!< Operation succeeded
    UnsupportedSectionType,               //!< Section type is not supported
    SectionExtensionsNotSupported,        //!< Section extensions not supported for this section type
    MultipleExtensionsNotSupported,       //!< Multiple extensions in a single section are not yet supported
    MtuTooSmallForExt11,                  //!< MTU too small to hold a section with extType 11 bundle
    MtuTooSmallForSection,                //!< MTU too small for section header
    Ext11BundlesNotInitialized,           //!< Extension type 11 bundles data not initialized
    Ext11BfwIqNotInitialized,             //!< Extension type 11 BFW IQ data not initialized
    Ext11BfwIqSizeNotSet,                 //!< Extension type 11 BFW IQ size not set
    Ext11BundleBfwIqNotInitialized,       //!< Extension type 11 bundle BFW IQ data not initialized
    InvalidBufferArray,                   //!< Invalid buffer array provided
    MtuCannotBeZero,                      //!< MTU cannot be zero
    MtuTooSmallForSingleSection,          //!< MTU too small for even a single section
    InsufficientBuffers,                  //!< Insufficient buffers for packet fragmentation
    BundleIndexOutOfBounds,               //!< Bundle index out of bounds
    InvalidPrbAllocationRbSizeZero,       //!< Invalid PRB allocation: rb_size is zero
    InvalidPrbAllocationNumSymbolsZero,   //!< Invalid PRB allocation: num_of_symbols is zero
    InvalidPrbAllocationExceedsSlot,      //!< Invalid PRB allocation: symbol allocation exceeds slot boundary
    TooManySectionsForSymbol,             //!< Too many sections for start symbol (exceeds MAX_CPLANE_SECTIONS)
    TooManyPdus,                          //!< Too many PDUs (exceeds maximum)
    InvalidNumAntennaPorts,               //!< Number of antenna ports cannot be zero
    PduPayloadOutOfBounds                 //!< PDU parsing exceeded payload bounds
};
// clang-format on

static_assert(
        static_cast<std::uint32_t>(OranErrc::PduPayloadOutOfBounds) <=
                std::numeric_limits<std::uint8_t>::max(),
        "OranErrc enumerator values must fit in std::uint8_t");

} // namespace ran::oran

// WISE_ENUM_ADAPT must be called at global namespace scope for ADL to work
WISE_ENUM_ADAPT(
        ran::oran::OranErrc,
        Success,
        UnsupportedSectionType,
        SectionExtensionsNotSupported,
        MultipleExtensionsNotSupported,
        MtuTooSmallForExt11,
        MtuTooSmallForSection,
        Ext11BundlesNotInitialized,
        Ext11BfwIqNotInitialized,
        Ext11BfwIqSizeNotSet,
        Ext11BundleBfwIqNotInitialized,
        InvalidBufferArray,
        MtuCannotBeZero,
        MtuTooSmallForSingleSection,
        InsufficientBuffers,
        BundleIndexOutOfBounds,
        InvalidPrbAllocationRbSizeZero,
        InvalidPrbAllocationNumSymbolsZero,
        InvalidPrbAllocationExceedsSlot,
        TooManySectionsForSymbol,
        TooManyPdus,
        InvalidNumAntennaPorts,
        PduPayloadOutOfBounds)

// Register OranErrc as an error code enum to enable implicit conversion to
// std::error_code
// NOTE: This MUST come before any functions that use OranErrc with
// std::error_code
namespace std {
template <> struct is_error_code_enum<ran::oran::OranErrc> : true_type {};
} // namespace std

namespace ran::oran {

/**
 * Custom error category for ORAN errors
 *
 * This class provides human-readable error messages and integrates ORAN errors
 * with the standard C++ error handling system.
 */
class OranErrorCategory final : public std::error_category {
private:
    // Compile-time table indexed by the enum's underlying value
    static constexpr std::array<std::string_view, ::wise_enum::size<OranErrc>> KMESSAGES{
            "Success: Operation completed successfully",
            "Unsupported section type: Section type is not supported",
            "Section extensions not supported: Section extensions are only supported for section "
            "type 1",
            "Multiple extensions not supported: Multiple extensions in a single section are not "
            "yet supported",
            "MTU too small for ext11: MTU too small to hold a section with extType 11 bundle",
            "MTU too small for section: MTU too small for section header",
            "Ext11 bundles not initialized: Extension type 11 bundles data not initialized",
            "Ext11 BFW IQ not initialized: Extension type 11 BFW IQ data not initialized",
            "Ext11 BFW IQ size not set: Extension type 11 BFW IQ size not set",
            "Ext11 bundle BFW IQ not initialized: Extension type 11 bundle BFW IQ data not "
            "initialized",
            "Invalid buffer array: Invalid buffer array provided",
            "MTU cannot be zero: MTU must be non-zero, please provide a valid MTU value",
            "MTU too small for single section: MTU too small for even a single section",
            "Insufficient buffers: Insufficient buffers for packet fragmentation",
            "Bundle index out of bounds: Bundle index out of bounds",
            "Invalid PRB allocation rb_size zero: Invalid PRB allocation: rb_size is zero",
            "Invalid PRB allocation num_symbols zero: Invalid PRB allocation: num_of_symbols is "
            "zero",
            "Invalid PRB allocation exceeds slot: Invalid PRB allocation: symbol allocation "
            "exceeds slot boundary",
            "Too many sections for symbol: Too many sections for start symbol (exceeds "
            "MAX_CPLANE_SECTIONS)",
            "Too many PDUs: Number of PDUs exceeds maximum",
            "Invalid num antenna ports: Number of antenna ports cannot be zero",
            "PDU payload out of bounds: PDU parsing exceeded payload bounds"};

    // Ensure KMESSAGES array size matches the number of enum values
    static_assert(
            KMESSAGES.size() == ::wise_enum::size<OranErrc>,
            "KMESSAGES array size must match the number of OranErrc enum values");

public:
    /**
     * Get the name of this error category
     *
     * @return The category name as a C-style string
     */
    [[nodiscard]] const char *name() const noexcept override { return "ran::oran"; }

    /**
     * Get a descriptive message for the given error code
     *
     * @param[in] condition The error code value
     * @return A descriptive error message
     */
    [[nodiscard]] std::string message(const int condition) const override {
        const auto idx = static_cast<std::size_t>(condition);
        if (idx < KMESSAGES.size()) {
            return std::string{KMESSAGES.at(idx)};
        }
        return std::format("Unknown ORAN error: {}", condition);
    }

    /**
     * Map ORAN errors to standard error conditions where applicable
     *
     * @param[in] condition The error code value
     * @return The equivalent standard error condition, or a default-constructed
     * condition
     */
    [[nodiscard]] std::error_condition
    default_error_condition(const int condition) const noexcept override {
        switch (static_cast<OranErrc>(condition)) {
        case OranErrc::Success:
            return {};
        case OranErrc::InvalidBufferArray:
        case OranErrc::InvalidPrbAllocationRbSizeZero:
        case OranErrc::InvalidPrbAllocationNumSymbolsZero:
        case OranErrc::InvalidPrbAllocationExceedsSlot:
        case OranErrc::InvalidNumAntennaPorts:
        case OranErrc::PduPayloadOutOfBounds:
            return std::errc::invalid_argument;
        case OranErrc::InsufficientBuffers:
            return std::errc::no_buffer_space;
        case OranErrc::BundleIndexOutOfBounds:
            return std::errc::result_out_of_range;
        case OranErrc::UnsupportedSectionType:
        case OranErrc::SectionExtensionsNotSupported:
        case OranErrc::MultipleExtensionsNotSupported:
            return std::errc::not_supported;
        default:
            // For ORAN-specific errors that don't map to standard conditions
            return std::error_condition{condition, *this};
        }
    }

    /**
     * Get the name of the error code enum value
     *
     * @param[in] condition The error code value
     * @return The enum name as a string
     */
    [[nodiscard]] static const char *name(const int condition) {
        const auto errc = static_cast<OranErrc>(condition);
        return ::wise_enum::to_string(errc).data();
    }
};

/**
 * Get the singleton instance of the ORAN error category
 *
 * @return Reference to the ORAN error category
 */
[[nodiscard]] inline const OranErrorCategory &oran_category() noexcept {
    static const OranErrorCategory instance{};
    return instance;
}

/**
 * Create an error_code from an OranErrc value
 *
 * @param[in] errc The ORAN error code
 * @return A std::error_code representing the ORAN error
 */
[[nodiscard]] inline std::error_code make_error_code(const OranErrc errc) noexcept {
    return {static_cast<int>(errc), oran_category()};
}

/**
 * Check if an OranErrc represents success
 *
 * @param[in] errc The error code to check
 * @return true if the error code represents success, false otherwise
 */
[[nodiscard]] constexpr bool is_success(const OranErrc errc) noexcept {
    return errc == OranErrc::Success;
}

/**
 * Check if an error_code represents ORAN success
 *
 * An error code represents success if it evaluates to false (i.e., value is 0).
 * This works regardless of the category (system, task, etc.).
 *
 * @param[in] errc The error code to check
 * @return true if the error code represents success, false otherwise
 */
[[nodiscard]] inline bool is_oran_success(const std::error_code &errc) noexcept {
    return !errc; // std::error_code evaluates to false when value is 0 (success)
}

/**
 * Get the name of an OranErrc enum value
 *
 * @param[in] errc The error code
 * @return The enum name as a string
 */
[[nodiscard]] inline const char *get_error_name(const OranErrc errc) noexcept {
    return ::wise_enum::to_string(errc).data();
}

/**
 * Get the name of an OranErrc from a std::error_code
 *
 * @param[in] ec The error code
 * @return The enum name as a string, or "unknown" if not an ORAN error
 */
[[nodiscard]] inline const char *get_error_name(const std::error_code &ec) noexcept {
    if (ec.category() != oran_category()) {
        return "unknown";
    }
    return get_error_name(static_cast<OranErrc>(ec.value()));
}

} // namespace ran::oran

#endif // RAN_ORAN_ERRORS_HPP
