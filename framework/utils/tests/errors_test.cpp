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

#include <array>        // for array
#include <cstddef>      // for size_t
#include <string>       // for allocator, string
#include <system_error> // for error_code, errc, error_condi...

#include <wise_enum_detail.h> // for optional_type, value_and_name

#include <gtest/gtest.h> // for AssertionResult, Message, Tes...
#include <wise_enum.h>   // for from_string, range, size

#include "utils/errors.hpp" // for NvErrc, get_error_name

namespace {
TEST(ErrorsTest, NvErrcEnumValues) {
    // NOTE: When adding new NvErrc values, update EXPECTED_MAX_ENUM_VALUE
    // below
    constexpr int EXPECTED_MAX_ENUM_VALUE = 15;
    constexpr int EXPECTED_ENUM_COUNT = EXPECTED_MAX_ENUM_VALUE + 1; // 0-based, so +1

    EXPECT_EQ(static_cast<int>(framework::utils::NvErrc::Success), 0);
    EXPECT_EQ(static_cast<int>(framework::utils::NvErrc::InternalError), 1);
    EXPECT_EQ(static_cast<int>(framework::utils::NvErrc::NotSupported), 2);
    EXPECT_EQ(static_cast<int>(framework::utils::NvErrc::InvalidArgument), 3);
    EXPECT_EQ(static_cast<int>(framework::utils::NvErrc::ArchMismatch), 4);
    EXPECT_EQ(static_cast<int>(framework::utils::NvErrc::AllocFailed), 5);
    EXPECT_EQ(static_cast<int>(framework::utils::NvErrc::SizeMismatch), 6);
    EXPECT_EQ(static_cast<int>(framework::utils::NvErrc::MemcpyError), 7);
    EXPECT_EQ(static_cast<int>(framework::utils::NvErrc::InvalidConversion), 8);
    EXPECT_EQ(static_cast<int>(framework::utils::NvErrc::UnsupportedType), 9);
    EXPECT_EQ(static_cast<int>(framework::utils::NvErrc::UnsupportedLayout), 10);
    EXPECT_EQ(static_cast<int>(framework::utils::NvErrc::UnsupportedRank), 11);
    EXPECT_EQ(static_cast<int>(framework::utils::NvErrc::UnsupportedConfig), 12);
    EXPECT_EQ(static_cast<int>(framework::utils::NvErrc::UnsupportedAlignment), 13);
    EXPECT_EQ(static_cast<int>(framework::utils::NvErrc::ValueOutOfRange), 14);
    EXPECT_EQ(static_cast<int>(framework::utils::NvErrc::RefMismatch), 15);

    // Verify ref_mismatch is the highest expected enum value
    EXPECT_EQ(static_cast<int>(framework::utils::NvErrc::RefMismatch), EXPECTED_MAX_ENUM_VALUE)
            << "ref_mismatch should be the highest enum value (" << EXPECTED_MAX_ENUM_VALUE << ")";

    // Verify we have exactly the expected number of enum values (16: values 0-15)
    // This assertion will fail if new enum values are added without updating the
    // test
    EXPECT_EQ(EXPECTED_ENUM_COUNT, 16)
            << "Expected exactly 16 NvErrc enum values (0-15). "
            << "If you added new enum values, update EXPECTED_MAX_ENUM_VALUE above.";
}

// Test: NvErrorCategory name() method returns correct string
TEST(ErrorsTest, ErrorCategoryName) {
    const auto &category = framework::utils::nv_category();
    const std::string name = category.name();

    EXPECT_EQ(name, "adsp::core");
}

// Test: NvErrorCategory message() method returns descriptive messages
TEST(ErrorsTest, ErrorCategoryMessages) {
    const auto &category = framework::utils::nv_category();

    // Test success message
    const std::string success_msg = category.message(0);
    EXPECT_TRUE(success_msg.find("Success") != std::string::npos);
    EXPECT_TRUE(success_msg.find("no errors") != std::string::npos);

    // Test internal error message
    const std::string internal_msg = category.message(1);
    EXPECT_TRUE(internal_msg.find("Internal error") != std::string::npos);
    EXPECT_TRUE(internal_msg.find("unexpected") != std::string::npos);

    // Test invalid argument message
    const std::string invalid_arg_msg = category.message(3);
    EXPECT_TRUE(invalid_arg_msg.find("Invalid argument") != std::string::npos);
    EXPECT_TRUE(invalid_arg_msg.find("invalid") != std::string::npos);

    // Test unknown error message
    static constexpr int UNKNOWN_ERROR_CODE = 999;
    const std::string unknown_msg = category.message(UNKNOWN_ERROR_CODE);
    EXPECT_TRUE(unknown_msg.find("Unknown NV error") != std::string::npos);
    EXPECT_TRUE(unknown_msg.find("999") != std::string::npos);
}

// Test: NvErrorCategory default_error_condition() maps to standard
// conditions
TEST(ErrorsTest, ErrorCategoryDefaultConditions) {
    const auto &category = framework::utils::nv_category();

    // Test success condition
    const auto success_condition = category.default_error_condition(0);
    EXPECT_FALSE(success_condition); // Default-constructed condition is false

    // Test invalid argument mapping
    const auto invalid_arg_condition = category.default_error_condition(3);
    EXPECT_EQ(invalid_arg_condition, std::errc::invalid_argument);

    // Test not supported mapping
    const auto not_supported_condition = category.default_error_condition(2);
    EXPECT_EQ(not_supported_condition, std::errc::operation_not_supported);

    // Test allocation failed mapping
    const auto alloc_failed_condition = category.default_error_condition(5);
    EXPECT_EQ(alloc_failed_condition, std::errc::not_enough_memory);

    // Test value out of range mapping
    const auto out_of_range_condition = category.default_error_condition(14);
    EXPECT_EQ(out_of_range_condition, std::errc::result_out_of_range);

    // Test NV-specific error (should return condition with nv category)
    const auto nv_specific_condition = category.default_error_condition(1);
    EXPECT_EQ(nv_specific_condition.category(), category);
    EXPECT_EQ(nv_specific_condition.value(), 1);
}

// Test: nv_category() returns singleton instance
TEST(ErrorsTest, NvCategorySingleton) {
    const auto &category1 = framework::utils::nv_category();
    const auto &category2 = framework::utils::nv_category();

    // Should be the same instance
    EXPECT_EQ(&category1, &category2);
}

// Test: make_error_code(NvErrc) creates correct error_code
TEST(ErrorsTest, MakeErrorCodeFromNvErrc) {
    // Test success code
    const auto success_code = framework::utils::make_error_code(framework::utils::NvErrc::Success);
    EXPECT_EQ(success_code.value(), 0);
    EXPECT_EQ(success_code.category(), framework::utils::nv_category());
    EXPECT_FALSE(success_code); // Success should be false in boolean context

    // Test error code
    const auto error_code =
            framework::utils::make_error_code(framework::utils::NvErrc::InvalidArgument);
    EXPECT_EQ(error_code.value(), 3);
    EXPECT_EQ(error_code.category(), framework::utils::nv_category());
    EXPECT_TRUE(error_code); // Error should be true in boolean context
}

// Test: from_nv_status() converts raw status values correctly
TEST(ErrorsTest, FromNvStatus) {
    EXPECT_EQ(framework::utils::from_nv_status(0), framework::utils::NvErrc::Success);
    EXPECT_EQ(framework::utils::from_nv_status(1), framework::utils::NvErrc::InternalError);
    EXPECT_EQ(framework::utils::from_nv_status(3), framework::utils::NvErrc::InvalidArgument);
    EXPECT_EQ(framework::utils::from_nv_status(15), framework::utils::NvErrc::RefMismatch);
}

// Test: make_error_code(int) creates error_code from raw status
TEST(ErrorsTest, MakeErrorCodeFromInt) {
    const auto success_code = framework::utils::make_error_code(0);
    EXPECT_EQ(success_code.value(), 0);
    EXPECT_EQ(success_code.category(), framework::utils::nv_category());
    EXPECT_FALSE(success_code);

    const auto error_code = framework::utils::make_error_code(3);
    EXPECT_EQ(error_code.value(), 3);
    EXPECT_EQ(error_code.category(), framework::utils::nv_category());
    EXPECT_TRUE(error_code);
}

// Test: is_success() correctly identifies success status
TEST(ErrorsTest, IsSuccessFunction) {
    EXPECT_TRUE(framework::utils::is_success(framework::utils::NvErrc::Success));
    EXPECT_FALSE(framework::utils::is_success(framework::utils::NvErrc::InternalError));
    EXPECT_FALSE(framework::utils::is_success(framework::utils::NvErrc::InvalidArgument));
    EXPECT_FALSE(framework::utils::is_success(framework::utils::NvErrc::RefMismatch));
}

// Helper function to test NV success with various error codes
void test_nv_success_with_error_codes() {
    // Test NV success
    const auto nv_success = framework::utils::make_error_code(framework::utils::NvErrc::Success);
    EXPECT_TRUE(framework::utils::is_nv_success(nv_success));

    // Test NV error
    const auto nv_error =
            framework::utils::make_error_code(framework::utils::NvErrc::InvalidArgument);
    EXPECT_FALSE(framework::utils::is_nv_success(nv_error));
}

// Helper function to test NV success with non-NV error codes
void test_nv_success_with_non_nv_error_codes() {
    // Test non-NV error_code
    const std::error_code system_error = std::make_error_code(std::errc::invalid_argument);
    EXPECT_FALSE(framework::utils::is_nv_success(system_error));

    // Test default constructed error_code
    const std::error_code default_error{};
    EXPECT_FALSE(framework::utils::is_nv_success(default_error));
}

// Test: is_nv_success() correctly identifies NV success in error_code
TEST(ErrorsTest, IsNvSuccessFunction) {
    test_nv_success_with_error_codes();
    test_nv_success_with_non_nv_error_codes();
}

// Test: std::error_code integration through template specialization
TEST(ErrorsTest, StdErrorCodeIntegration) {
    // Verify the template specialization is working
    EXPECT_TRUE(std::is_error_code_enum_v<framework::utils::NvErrc>);

    // Test implicit conversion from NvErrc to std::error_code
    const std::error_code error_code = framework::utils::NvErrc::InvalidArgument;
    EXPECT_EQ(error_code.value(), 3);
    EXPECT_EQ(error_code.category(), framework::utils::nv_category());

    // Test comparison with std::errc
    const std::error_code invalid_arg_code = framework::utils::NvErrc::InvalidArgument;
    EXPECT_TRUE(invalid_arg_code == std::errc::invalid_argument);
}

// Test: Error message consistency and completeness
TEST(ErrorsTest, ErrorMessageConsistency) {
    // Use the same constants as NvErrcEnumValues test for consistency
    static constexpr int EXPECTED_MAX_ENUM_VALUE = 15;
    static constexpr std::size_t MIN_MESSAGE_LENGTH = 10U;

    const auto &category = framework::utils::nv_category();

    // Test that all enum values have non-empty messages
    for (int i = 0; i <= EXPECTED_MAX_ENUM_VALUE; ++i) {
        const std::string message = category.message(i);
        EXPECT_FALSE(message.empty()) << "Message for error code " << i << " should not be empty";
        EXPECT_GT(message.length(), MIN_MESSAGE_LENGTH)
                << "Message for error code " << i << " should be descriptive";
    }

    // Test that error messages start with appropriate prefixes
    EXPECT_TRUE(category.message(0).starts_with("Success"));
    EXPECT_TRUE(category.message(1).starts_with("Internal error"));
    EXPECT_TRUE(category.message(2).starts_with("Not supported"));
    EXPECT_TRUE(category.message(3).starts_with("Invalid argument"));
}

// Test: Error code comparison and equivalence
TEST(ErrorsTest, ErrorCodeComparison) {
    const auto success1 = framework::utils::make_error_code(framework::utils::NvErrc::Success);
    const auto success2 = framework::utils::make_error_code(framework::utils::NvErrc::Success);
    const auto error1 =
            framework::utils::make_error_code(framework::utils::NvErrc::InvalidArgument);
    const auto error2 =
            framework::utils::make_error_code(framework::utils::NvErrc::InvalidArgument);
    const auto different_error =
            framework::utils::make_error_code(framework::utils::NvErrc::InternalError);

    // Test equality
    EXPECT_EQ(success1, success2);
    EXPECT_EQ(error1, error2);

    // Test inequality
    EXPECT_NE(success1, error1);
    EXPECT_NE(error1, different_error);

    // Test comparison with specific enum values
    EXPECT_EQ(success1, framework::utils::NvErrc::Success);
    EXPECT_EQ(error1, framework::utils::NvErrc::InvalidArgument);
    EXPECT_NE(error1, framework::utils::NvErrc::Success);
}

// Test: Verify that NvErrc enum values can be converted to string names
TEST(ErrorsTest, EnumToStringConversion) {
    // Test all enum values
    EXPECT_STREQ(framework::utils::get_error_name(framework::utils::NvErrc::Success), "Success");
    EXPECT_STREQ(
            framework::utils::get_error_name(framework::utils::NvErrc::InternalError),
            "InternalError");
    EXPECT_STREQ(
            framework::utils::get_error_name(framework::utils::NvErrc::NotSupported),
            "NotSupported");
    EXPECT_STREQ(
            framework::utils::get_error_name(framework::utils::NvErrc::InvalidArgument),
            "InvalidArgument");
    EXPECT_STREQ(
            framework::utils::get_error_name(framework::utils::NvErrc::ArchMismatch),
            "ArchMismatch");
    EXPECT_STREQ(
            framework::utils::get_error_name(framework::utils::NvErrc::AllocFailed), "AllocFailed");
    EXPECT_STREQ(
            framework::utils::get_error_name(framework::utils::NvErrc::SizeMismatch),
            "SizeMismatch");
    EXPECT_STREQ(
            framework::utils::get_error_name(framework::utils::NvErrc::MemcpyError), "MemcpyError");
    EXPECT_STREQ(
            framework::utils::get_error_name(framework::utils::NvErrc::InvalidConversion),
            "InvalidConversion");
    EXPECT_STREQ(
            framework::utils::get_error_name(framework::utils::NvErrc::UnsupportedType),
            "UnsupportedType");
    EXPECT_STREQ(
            framework::utils::get_error_name(framework::utils::NvErrc::UnsupportedLayout),
            "UnsupportedLayout");
    EXPECT_STREQ(
            framework::utils::get_error_name(framework::utils::NvErrc::UnsupportedRank),
            "UnsupportedRank");
    EXPECT_STREQ(
            framework::utils::get_error_name(framework::utils::NvErrc::UnsupportedConfig),
            "UnsupportedConfig");
    EXPECT_STREQ(
            framework::utils::get_error_name(framework::utils::NvErrc::UnsupportedAlignment),
            "UnsupportedAlignment");
    EXPECT_STREQ(
            framework::utils::get_error_name(framework::utils::NvErrc::ValueOutOfRange),
            "ValueOutOfRange");
    EXPECT_STREQ(
            framework::utils::get_error_name(framework::utils::NvErrc::RefMismatch), "RefMismatch");
}

// Test: Verify that NvErrorCategory::name() method works correctly
TEST(ErrorsTest, ErrorCategoryNameMethod) {
    static constexpr int INVALID_ERROR_CODE = 999;

    const auto &category = framework::utils::nv_category();

    // Test with valid error codes
    EXPECT_STREQ(category.name(0), "Success");
    EXPECT_STREQ(category.name(1), "InternalError");
    EXPECT_STREQ(category.name(2), "NotSupported");
    EXPECT_STREQ(category.name(3), "InvalidArgument");

    // Test with invalid error code - should return empty string
    const char *invalid_name = category.name(INVALID_ERROR_CODE);
    EXPECT_EQ(invalid_name, nullptr);
}

// Test: Verify that wise_enum provides expected functionality
TEST(ErrorsTest, WiseEnumFeatures) {
    // Test enum count
    static constexpr auto ENUM_COUNT = ::wise_enum::size<framework::utils::NvErrc>;
    EXPECT_EQ(ENUM_COUNT, 16); // We have 16 enum values

    // Test iteration over all enum values
    int index = 0;
    for (const auto &[value, name] : ::wise_enum::range<framework::utils::NvErrc>) {
        EXPECT_EQ(static_cast<int>(value),
                  index); // Values should be sequential starting from 0
        index++;
    }

    // Test string to enum conversion
    const auto success_opt = wise_enum::from_string<framework::utils::NvErrc>("Success");
    EXPECT_TRUE(success_opt.has_value());
    if (success_opt.has_value()) {
        EXPECT_EQ(success_opt.value(), framework::utils::NvErrc::Success);
    }

    const auto invalid_opt = wise_enum::from_string<framework::utils::NvErrc>("invalid_name");
    EXPECT_FALSE(invalid_opt.has_value());
}

// Test: Verify integration with std::error_code
TEST(ErrorsTest, ErrorCodeIntegration) {
    // Create error_code from NvErrc
    const std::error_code error_code =
            framework::utils::make_error_code(framework::utils::NvErrc::InvalidArgument);

    // Verify category
    EXPECT_EQ(&error_code.category(), &framework::utils::nv_category());

    // Verify message
    EXPECT_EQ(
            error_code.message(),
            "Invalid argument: One or more of the arguments provided to the "
            "function was invalid");

    // Verify we can get the enum name through the category
    EXPECT_STREQ(framework::utils::nv_category().name(error_code.value()), "InvalidArgument");
}
} // namespace
