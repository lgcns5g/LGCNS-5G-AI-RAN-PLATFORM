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

#include <cstddef>     // for size_t
#include <exception>   // for exception
#include <format>      // for format
#include <functional>  // for reference_wrapper, cref
#include <iostream>    // for cout
#include <string>      // for allocator, string, basic_string, char...
#include <string_view> // for string_view
#include <type_traits> // for is_final_v
#include <utility>     // for pair, move
#include <vector>      // for vector

#include <driver_types.h> // for cudaError, cudaError_t

#include <gtest/gtest.h> // for AssertionResult, Message, TestPartResult

#include <cuda.h> // for cudaError_enum, CUresult

#include "utils/errors.hpp"     // for NvErrc
#include "utils/exceptions.hpp" // for CudaDriverException, NvFnException

namespace {

class ExceptionsTest : public ::testing::Test {
protected:
    void SetUp() override { std::cout << "Setting up ExceptionsTest...\n"; }

    void TearDown() override { std::cout << "Tearing down ExceptionsTest...\n"; }
};

// Test: CudaRuntimeException construction and what() message
TEST_F(ExceptionsTest, CudaRuntimeExceptionBasicFunctionality) {
    // Test with cudaSuccess
    const framework::utils::CudaRuntimeException success_ex(cudaSuccess);
    const std::string success_msg = success_ex.what();
    EXPECT_FALSE(success_msg.empty());
    std::cout << std::format("  Success message: '{}'\n", success_msg);

    // Test with a typical error code
    const framework::utils::CudaRuntimeException error_ex(cudaErrorInvalidValue);
    const std::string error_msg = error_ex.what();
    EXPECT_FALSE(error_msg.empty());
    std::cout << std::format("  Error message: '{}'\n", error_msg);

    // Verify it's a proper std::exception
    const std::exception &base_ex = error_ex;
    EXPECT_STREQ(base_ex.what(), error_ex.what());
}

// Test: CudaRuntimeException noexcept guarantee
TEST_F(ExceptionsTest, CudaRuntimeExceptionNoexceptGuarantee) {
    framework::utils::CudaRuntimeException cuda_ex(cudaErrorInvalidValue);

    // what() should be noexcept
    EXPECT_NO_THROW([[maybe_unused]] const auto msg = cuda_ex.what());

    // Store expected message before moving
    const std::string expected_message = cuda_ex.what();

    // Verify it can be caught as std::exception
    try {
        throw std::move(cuda_ex);
    } catch (const std::exception &caught) {
        EXPECT_STREQ(caught.what(), expected_message.c_str());
        std::cout << "  Successfully caught as std::exception\n";
    } catch (...) {
        FAIL() << "Should have been caught as std::exception";
    }
}

// Test: CudaDriverException construction with CUresult only
TEST_F(ExceptionsTest, CudaDriverExceptionBasicConstruction) {
    // Test with CUDA_SUCCESS
    const framework::utils::CudaDriverException success_ex(CUDA_SUCCESS);
    const std::string success_msg = success_ex.what();
    EXPECT_FALSE(success_msg.empty());
    EXPECT_TRUE(success_msg.find("CUDA driver error") != std::string::npos);
    std::cout << std::format("  Success message: '{}'\n", success_msg);

    // Test with a typical error code
    const framework::utils::CudaDriverException error_ex(CUDA_ERROR_INVALID_VALUE);
    const std::string error_msg = error_ex.what();
    EXPECT_FALSE(error_msg.empty());
    EXPECT_TRUE(error_msg.find("CUDA driver error") != std::string::npos);
    std::cout << std::format("  Error message: '{}'\n", error_msg);

    // Verify it's a proper std::exception
    const std::exception &base_ex = error_ex;
    EXPECT_STREQ(base_ex.what(), error_ex.what());
}

// Test: CudaDriverException construction with CUresult and user string
TEST_F(ExceptionsTest, CudaDriverExceptionWithUserString) {
    const char *user_msg = "Custom error context";
    const framework::utils::CudaDriverException cuda_driver_ex(CUDA_ERROR_INVALID_VALUE, user_msg);
    const std::string message = cuda_driver_ex.what();

    EXPECT_FALSE(message.empty());
    EXPECT_TRUE(message.find("CUDA driver error") != std::string::npos);
    EXPECT_TRUE(message.find(user_msg) != std::string::npos);
    std::cout << std::format("  Message with user string: '{}'\n", message);
}

// Test: CudaDriverException construction with empty string user string
TEST_F(ExceptionsTest, CudaDriverExceptionEmptyUserString) {
    const framework::utils::CudaDriverException cuda_driver_ex(CUDA_ERROR_INVALID_VALUE, "");
    const std::string message = cuda_driver_ex.what();

    EXPECT_FALSE(message.empty());
    EXPECT_TRUE(message.find("CUDA driver error") != std::string::npos);
    // Should not contain extra comma or user string when empty string is passed
    std::cout << std::format("  Message without user string: '{}'\n", message);
}

// Test: CudaDriverException noexcept guarantee
TEST_F(ExceptionsTest, CudaDriverExceptionNoexceptGuarantee) {
    framework::utils::CudaDriverException cuda_driver_ex(CUDA_ERROR_INVALID_VALUE, "test");

    // what() should be noexcept
    EXPECT_NO_THROW([[maybe_unused]] const auto msg = cuda_driver_ex.what());

    // Store expected message before moving
    const std::string expected_message = cuda_driver_ex.what();

    // Verify it can be caught as std::exception
    try {
        throw std::move(cuda_driver_ex);
    } catch (const std::exception &caught) {
        EXPECT_STREQ(caught.what(), expected_message.c_str());
        std::cout << "  Successfully caught as std::exception\n";
    } catch (...) {
        FAIL() << "Should have been caught as std::exception";
    }
}

// Test: NvException construction and what() message
TEST_F(ExceptionsTest, NvExceptionBasicFunctionality) {
    // Test with success status
    const framework::utils::NvException success_ex(framework::utils::NvErrc::Success);
    const std::string success_msg = success_ex.what();
    EXPECT_FALSE(success_msg.empty());
    EXPECT_TRUE(success_msg.find("Success") != std::string::npos);
    std::cout << std::format("  Success message: '{}'\n", success_msg);

    // Test with error status
    const framework::utils::NvException error_ex(framework::utils::NvErrc::InvalidArgument);
    const std::string error_msg = error_ex.what();
    EXPECT_FALSE(error_msg.empty());
    EXPECT_TRUE(error_msg.find("Invalid argument") != std::string::npos);
    std::cout << std::format("  Error message: '{}'\n", error_msg);

    // Verify it's a proper std::exception
    const std::exception &base_ex = error_ex;
    EXPECT_STREQ(base_ex.what(), error_ex.what());
}

// Test: NvException with various error codes
TEST_F(ExceptionsTest, NvExceptionVariousErrorCodes) {
    // Test a few different error codes to ensure proper message retrieval
    const std::vector test_codes = {
            framework::utils::NvErrc::Success,
            framework::utils::NvErrc::InternalError,
            framework::utils::NvErrc::NotSupported,
            framework::utils::NvErrc::InvalidArgument,
            framework::utils::NvErrc::AllocFailed,
            framework::utils::NvErrc::SizeMismatch,
            framework::utils::NvErrc::RefMismatch};

    for (const auto &code : test_codes) {
        const framework::utils::NvException nv_ex(code);
        const std::string message = nv_ex.what();

        EXPECT_FALSE(message.empty())
                << "Message should not be empty for error code: " << static_cast<int>(code);
        static constexpr std::size_t MIN_MESSAGE_LENGTH = 5U;
        EXPECT_GT(message.length(), MIN_MESSAGE_LENGTH)
                << "Message should be descriptive for error code: " << static_cast<int>(code);

        std::cout << std::format("  Code {}: '{}'\n", static_cast<int>(code), message);
    }
}

// Test: NvException noexcept guarantee
TEST_F(ExceptionsTest, NvExceptionNoexceptGuarantee) {
    framework::utils::NvException nv_ex(framework::utils::NvErrc::InvalidArgument);

    // what() should be noexcept
    EXPECT_NO_THROW([[maybe_unused]] const auto msg = nv_ex.what());

    // Store expected message before moving
    const std::string expected_message = nv_ex.what();

    // Verify it can be caught as std::exception
    try {
        throw std::move(nv_ex);
    } catch (const std::exception &caught) {
        EXPECT_STREQ(caught.what(), expected_message.c_str());
        std::cout << "  Successfully caught as std::exception\n";
    } catch (...) {
        FAIL() << "Should have been caught as std::exception";
    }
}

// Test: NvFnException construction and what() message
TEST_F(ExceptionsTest, NvFnExceptionBasicFunctionality) {
    static constexpr std::string_view FUNCTION_NAME = "testFunction";
    const framework::utils::NvFnException nv_fn_ex(
            framework::utils::NvErrc::InvalidArgument, FUNCTION_NAME);
    const std::string message = nv_fn_ex.what();

    EXPECT_FALSE(message.empty());
    EXPECT_TRUE(message.find("Function") != std::string::npos);
    EXPECT_TRUE(message.find(FUNCTION_NAME) != std::string::npos);
    EXPECT_TRUE(message.find("returned") != std::string::npos);
    EXPECT_TRUE(message.find("InvalidArgument") != std::string::npos);

    std::cout << std::format("  Function exception message: '{}'\n", message);

    // Verify it's a proper std::exception
    const std::exception &base_ex = nv_fn_ex;
    EXPECT_STREQ(base_ex.what(), nv_fn_ex.what());
}

// Test: NvFnException with various function names and error codes
TEST_F(ExceptionsTest, NvFnExceptionVariousCombinations) {
    const std::vector<std::pair<framework::utils::NvErrc, const char *>> test_cases = {
            {framework::utils::NvErrc::Success, "successFunction"},
            {framework::utils::NvErrc::InternalError, "internalFunction"},
            {framework::utils::NvErrc::NotSupported, "unsupportedFunction"},
            {framework::utils::NvErrc::AllocFailed, "allocFunction"},
            {framework::utils::NvErrc::RefMismatch, "validateFunction"}};

    for (const auto &[code, func_name] : test_cases) {
        const framework::utils::NvFnException nv_fn_ex(code, func_name);
        const std::string message = nv_fn_ex.what();

        EXPECT_FALSE(message.empty());
        EXPECT_TRUE(message.find("Function") != std::string::npos);
        EXPECT_TRUE(message.find(func_name) != std::string::npos);
        EXPECT_TRUE(message.find("returned") != std::string::npos);

        std::cout << std::format("  {}: '{}'\n", func_name, message);
    }
}

// Test: NvFnException noexcept guarantee
TEST_F(ExceptionsTest, NvFnExceptionNoexceptGuarantee) {
    framework::utils::NvFnException nv_fn_ex(framework::utils::NvErrc::InvalidArgument, "testFunc");

    // what() should be noexcept
    EXPECT_NO_THROW([[maybe_unused]] const auto msg = nv_fn_ex.what());

    // Store expected message before moving
    const std::string expected_message = nv_fn_ex.what();

    // Verify it can be caught as std::exception
    try {
        throw std::move(nv_fn_ex);
    } catch (const std::exception &caught) {
        EXPECT_STREQ(caught.what(), expected_message.c_str());
        std::cout << "  Successfully caught as std::exception\n";
    } catch (...) {
        FAIL() << "Should have been caught as std::exception";
    }
}

// Test: Exception hierarchy and polymorphism
TEST_F(ExceptionsTest, ExceptionHierarchyAndPolymorphism) {
    // Create different exception types
    const framework::utils::CudaRuntimeException cuda_ex(cudaErrorInvalidValue);
    const framework::utils::CudaDriverException driver_ex(CUDA_ERROR_INVALID_VALUE, "test");
    const framework::utils::NvException nv_ex(framework::utils::NvErrc::InvalidArgument);
    const framework::utils::NvFnException nv_fn_ex(
            framework::utils::NvErrc::NotSupported, "testFunc");

    // Test that all can be caught as std::exception
    const std::vector<std::reference_wrapper<const std::exception>> exceptions = {
            std::cref(cuda_ex), std::cref(driver_ex), std::cref(nv_ex), std::cref(nv_fn_ex)};

    for (std::size_t i = 0; i < exceptions.size(); ++i) {
        const std::exception &ex = exceptions[i].get();
        const std::string message = ex.what();
        EXPECT_FALSE(message.empty()) << "Exception " << i << " should have non-empty message";
        std::cout << std::format("  Exception {}: '{}'\n", i, message);
    }
}

// Test: Exception final classes cannot be inherited
TEST_F(ExceptionsTest, ExceptionFinalClasses) {
    // These are compile-time checks - the classes should be marked as final
    // If they weren't final, this would be a design issue

    // Verify the classes are declared as final using compile-time checks
    static_assert(
            std::is_final_v<framework::utils::CudaRuntimeException>,
            "CudaRuntimeException must be declared final");
    static_assert(
            std::is_final_v<framework::utils::CudaDriverException>,
            "CudaDriverException must be declared final");
    static_assert(
            std::is_final_v<framework::utils::NvException>, "NvException must be declared final");
    static_assert(
            std::is_final_v<framework::utils::NvFnException>,
            "NvFnException must be declared final");
}

// Test: Exception message consistency and format
TEST_F(ExceptionsTest, ExceptionMessageConsistency) {
    // Test CudaDriverException message format
    const framework::utils::CudaDriverException driver_ex(CUDA_ERROR_INVALID_VALUE, "context");
    const std::string driver_msg = driver_ex.what();
    EXPECT_TRUE(driver_msg.find("CUDA driver error:") != std::string::npos);
    EXPECT_TRUE(driver_msg.find("context") != std::string::npos);

    // Test NvFnException message format
    const framework::utils::NvFnException nv_fn_ex(
            framework::utils::NvErrc::InvalidArgument, "myFunction");
    const std::string nv_fn_msg = nv_fn_ex.what();
    EXPECT_TRUE(nv_fn_msg.find("Function myFunction returned") != std::string::npos);
    EXPECT_TRUE(nv_fn_msg.find("InvalidArgument") != std::string::npos);

    // Test NvException uses error category messages
    const framework::utils::NvException nv_ex(framework::utils::NvErrc::AllocFailed);
    const std::string nv_msg = nv_ex.what();
    EXPECT_TRUE(nv_msg.find("Allocation failed") != std::string::npos);

    std::cout << std::format("  Driver exception format: '{}'\n", driver_msg);
    std::cout << std::format("  Function exception format: '{}'\n", nv_fn_msg);
    std::cout << std::format("  NV exception format: '{}'\n", nv_msg);
}

// Test: Exception construction with edge cases
TEST_F(ExceptionsTest, ExceptionConstructionEdgeCases) {
    // Test with extreme error code values
    const framework::utils::CudaRuntimeException extreme_cuda_ex(static_cast<cudaError_t>(999));
    EXPECT_NO_THROW([[maybe_unused]] const auto msg1 = extreme_cuda_ex.what());

    const framework::utils::CudaDriverException extreme_driver_ex(static_cast<CUresult>(999));
    EXPECT_NO_THROW([[maybe_unused]] const auto msg2 = extreme_driver_ex.what());

    // Test NvFnException with empty function name
    const framework::utils::NvFnException empty_nv_fn_ex(framework::utils::NvErrc::Success, "");
    const std::string empty_nv_fn_msg = empty_nv_fn_ex.what();
    EXPECT_TRUE(empty_nv_fn_msg.find("Function  returned") != std::string::npos);

    // Test NvFnException with very long function name
    const std::string long_func_name(1000, 'a');
    const framework::utils::NvFnException long_nv_fn_ex(
            framework::utils::NvErrc::Success, long_func_name);
    const std::string long_nv_fn_msg = long_nv_fn_ex.what();
    EXPECT_TRUE(long_nv_fn_msg.find(long_func_name) != std::string::npos);
}

} // namespace
