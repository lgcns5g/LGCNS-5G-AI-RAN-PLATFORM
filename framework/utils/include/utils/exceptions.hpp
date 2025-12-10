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

#ifndef FRAMEWORK_EXCEPTIONS_HPP
#define FRAMEWORK_EXCEPTIONS_HPP

#include <exception>
#include <format>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>

#include "utils/errors.hpp"

namespace framework::utils {

/**
 * Exception class for CUDA runtime API errors
 *
 * This exception wraps cudaError_t values and provides human-readable
 * error messages through the CUDA runtime API.
 */
class CudaRuntimeException final : public std::exception {
public:
    /**
     * Construct a CUDA exception from a CUDA runtime error code
     *
     * @param[in] status The CUDA runtime error code that caused the exception
     */
    explicit CudaRuntimeException(const cudaError_t status) : status_(status) {}

    /**
     * Get the error message for this exception
     *
     * @return Human-readable error message from cudaGetErrorString
     */
    [[nodiscard]]
    const char *what() const noexcept override {
        return cudaGetErrorString(status_);
    }

private:
    cudaError_t status_{}; //!< The CUDA runtime error code
};

/**
 * Exception class for CUDA driver API errors
 *
 * This exception wraps CUresult values and provides detailed error
 * information including error names, descriptions, and optional user context.
 */
class CudaDriverException final : public std::exception {
public:
    /**
     * Construct a CUDA driver exception from a driver API result code
     *
     * Creates a detailed error message that includes the error name and
     * description obtained from the CUDA driver API, along with optional
     * user-provided context.
     *
     * @param[in] result The CUDA driver API result code that caused the exception
     * @param[in] user_str Optional user-provided context string to include in the
     * error message
     */
    explicit CudaDriverException(const CUresult result, const std::string_view user_str = "")
            : result_(result) {
        const char *res_name_str{};
        const CUresult error_name_result = cuGetErrorName(result_, &res_name_str);
        const char *res_description_str{};
        const CUresult error_description_result = cuGetErrorString(result_, &res_description_str);

        const std::string error_name = (error_name_result == CUDA_SUCCESS)
                                               ? std::string{res_name_str}
                                               : std::to_string(result_);
        const std::string error_desc = (error_description_result == CUDA_SUCCESS)
                                               ? std::string{res_description_str}
                                               : std::to_string(result_);

        // Only append user string if it's not null and not empty to avoid trailing
        // comma
        disp_str_ = user_str.empty()
                            ? std::format("CUDA driver error: {} - {}", error_name, error_desc)
                            : std::format(
                                      "CUDA driver error: {} - {}, {}",
                                      error_name,
                                      error_desc,
                                      user_str);
    }

    /**
     * Get the error message for this exception
     *
     * @return Formatted error message containing error name, description, and
     * optional user context
     */
    [[nodiscard]]
    const char *what() const noexcept override {
        return disp_str_.c_str();
    }

private:
    std::string disp_str_; //!< Formatted error message string
    CUresult result_{};    //!< The CUDA driver API result code
};

/**
 * Exception class for NV library errors
 *
 * This exception wraps NvErrc error codes and provides human-readable
 * error messages through the NV error category system.
 */
class NvException final : public std::exception {
public:
    /**
     * Construct a NV exception from an error code
     *
     * @param[in] status The NV error code that caused the exception
     */
    explicit NvException(const NvErrc status) : status_(status) {
        message_ = std::format(
                "NV Error: {} ({})",
                NvErrorCategory::name(to_underlying(status_)),
                nv_category().message(to_underlying(status_)));
    }

    /**
     * Get the error message for this exception
     *
     * @return Human-readable error message from the NV error category
     */
    [[nodiscard]]
    const char *what() const noexcept override {
        return message_.c_str();
    }

private:
    [[maybe_unused]] NvErrc status_{}; //!< The NV error code
    std::string message_;              //!< Cached error message string
};

/**
 * Exception class for NV function-specific errors
 *
 * This exception provides detailed error information including the function
 * name that failed, along with the error code and description from the NV error
 * system.
 */
class NvFnException final : public std::exception {
public:
    /**
     * Construct a NV function exception with function context
     *
     * Creates a detailed error message that includes the function name,
     * error code message, and error code name for debugging purposes.
     *
     * @param[in] status The NV error code that caused the exception
     * @param[in] function_name_str The name of the function that failed
     */
    NvFnException(const NvErrc status, const std::string_view function_name_str) : status_(status) {
        message_ = std::format(
                "Function {} returned {}: {}",
                function_name_str,
                nv_category().message(to_underlying(status_)),
                NvErrorCategory::name(to_underlying(status_)));
    }

    /**
     * Get the error message for this exception
     *
     * @return Formatted error message containing function name, error message,
     * and error name
     */
    [[nodiscard]]
    const char *what() const noexcept override {
        return message_.c_str();
    }

private:
    NvErrc status_{};     //!< The NV error code
    std::string message_; //!< Formatted error description string
};
} // namespace framework::utils

#endif // FRAMEWORK_EXCEPTIONS_HPP
