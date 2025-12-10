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

#ifndef FRAMEWORK_ERROR_MACROS_HPP
#define FRAMEWORK_ERROR_MACROS_HPP

#include "utils/errors.hpp"
#include "utils/exceptions.hpp"

namespace framework::utils {

// GSL contract violation behavior configuration (from CMake GSL_CONTRACT_VIOLATION_THROWS option)
#ifdef gsl_CONFIG_CONTRACT_VIOLATION_THROWS
//! GSL contract violation behavior flag (true if violations throw exceptions)
inline constexpr bool GSL_CONTRACT_THROWS = true;
#else
//! GSL contract violation behavior flag (true if violations throw exceptions)
inline constexpr bool GSL_CONTRACT_THROWS = false;
#endif

} // namespace framework::utils

// NOLINTBEGIN(cppcoreguidelines-avoid-do-while,cppcoreguidelines-macro-usage,readability-function-cognitive-complexity)

// Compiler-specific pragma helpers
#if defined(__GNUC__) && !defined(__clang__)
// GCC-specific pragmas
#define FRAMEWORK_PRAGMA_DIAGNOSTIC_PUSH _Pragma("GCC diagnostic push")
#define FRAMEWORK_PRAGMA_DIAGNOSTIC_POP _Pragma("GCC diagnostic pop")
#define FRAMEWORK_PRAGMA_IGNORE_UNKNOWN_PRAGMAS                                                    \
    _Pragma("GCC diagnostic ignored \"-Wunknown-pragmas\"")
#elif defined(__clang__)
// Clang-specific pragmas
#define FRAMEWORK_PRAGMA_DIAGNOSTIC_PUSH _Pragma("clang diagnostic push")
#define FRAMEWORK_PRAGMA_DIAGNOSTIC_POP _Pragma("clang diagnostic pop")
#define FRAMEWORK_PRAGMA_IGNORE_UNKNOWN_PRAGMAS                                                    \
    _Pragma("clang diagnostic ignored \"-Wunknown-pragmas\"")
#else
// Fallback for other compilers
#define FRAMEWORK_PRAGMA_DIAGNOSTIC_PUSH
#define FRAMEWORK_PRAGMA_DIAGNOSTIC_POP
#define FRAMEWORK_PRAGMA_IGNORE_UNKNOWN_PRAGMAS
#endif

// VectorCAST instrumentation pragmas (conditionally defined)
#if defined(__VECTORCAST__) || defined(VECTORCAST)
#define FRAMEWORK_PRAGMA_VCAST_START _Pragma("vcast_dont_instrument_start")
#define FRAMEWORK_PRAGMA_VCAST_END _Pragma("vcast_dont_instrument_end")
#else
#define FRAMEWORK_PRAGMA_VCAST_START
#define FRAMEWORK_PRAGMA_VCAST_END
#endif

// Self-move warning suppression pragmas
#if defined(__clang__)
#define FRAMEWORK_PRAGMA_IGNORE_SELF_MOVE                                                          \
    _Pragma("clang diagnostic push") _Pragma("clang diagnostic ignored \"-Wself-move\"")
#define FRAMEWORK_PRAGMA_RESTORE_SELF_MOVE _Pragma("clang diagnostic pop")
#elif defined(__GNUC__)
#define FRAMEWORK_PRAGMA_IGNORE_SELF_MOVE                                                          \
    _Pragma("GCC diagnostic push") _Pragma("GCC diagnostic ignored \"-Wself-move\"")
#define FRAMEWORK_PRAGMA_RESTORE_SELF_MOVE _Pragma("GCC diagnostic pop")
#else
#define FRAMEWORK_PRAGMA_IGNORE_SELF_MOVE
#define FRAMEWORK_PRAGMA_RESTORE_SELF_MOVE
#endif

/**
 * Check NV API call and throw nv_exception on error
 *
 * This macro wraps NV API calls to automatically check for errors.
 * On error, it logs the error information and throws a
 * framework::utils::NvException.
 *
 * @param[in] nv_result NV API expression that returns nvStatus_t
 *
 * @throws framework::utils::NvException If the NV call returns
 * non-success status
 *
 * @see FRAMEWORK_CUDA_RUNTIME_CHECK_THROW For CUDA runtime version
 * @see FRAMEWORK_CUDA_DRIVER_CHECK_THROW For CUDA driver API version
 */
#define FRAMEWORK_NV_CHECK_THROW(nv_result)                                                        \
    do {                                                                                           \
        if (const framework::utils::NvErrc status = nv_result;                                     \
            status != framework::utils::NvErrc::Success) {                                         \
            RT_LOGC_ERROR(                                                                         \
                    framework::utils::Core::CoreNvApi,                                             \
                    "NV_ERROR: {} ({})",                                                           \
                    std::source_location::current().file_name(),                                   \
                    std::source_location::current().line());                                       \
            throw framework::utils::NvException(status);                                           \
        }                                                                                          \
    } while (0)

/**
 * Concept to check if an exception type can be constructed with a string
 * message
 *
 * This concept verifies that an exception type T can be constructed using
 * a const std::string& parameter, which is required for exception macros
 * that pass error messages to exception constructors.
 *
 * @tparam T The exception type to check
 */
template <typename T>
concept StringConstructibleException =
        std::is_base_of_v<std::exception, T> && std::is_constructible_v<T, const std::string &>;

/**
 * Throw an exception with a message
 *
 * This macro throws an exception with a message. It logs the error information
 * and throws the specified exception with the provided message.
 *
 * @param[in] exception The exception type to throw
 * @param[in] message The message to throw the exception with
 */
#define FRAMEWORK_NV_THROW(exception, message)                                                     \
    do {                                                                                           \
        RT_LOGC_ERROR(                                                                             \
                framework::utils::Core::CoreNvApi,                                                 \
                "NV_ERROR: {}:{} {}",                                                              \
                std::source_location::current().file_name(),                                       \
                std::source_location::current().line(),                                            \
                message);                                                                          \
        throw exception(message);                                                                  \
    } while (0)

/**
 * Check condition and throw exception on error (constrained version)
 *
 * This macro checks a condition and throws an exception if the condition is
 * true. It logs the error information and throws the specified exception
 * with the provided message.
 *
 * The exception type must satisfy the StringConstructibleException concept,
 * meaning it must:
 * - Inherit from std::exception
 * - Have a constructor that accepts const std::string& parameter
 *
 * @param[in] condition The condition to check
 * @param[in] exception The exception type to throw (must be
 * StringConstructibleException)
 * @param[in] message The message to pass to exception constructor and log
 *
 * @throws exception If the condition is true
 *
 * @note This macro uses C++20 concepts to ensure type safety. Common standard
 * exceptions that work include: std::runtime_error, std::logic_error,
 * std::invalid_argument, std::out_of_range, etc.
 *
 * @note Exceptions like std::bad_alloc (default constructor only) will not
 * compile with this macro. Use FRAMEWORK_NV_THROW_IF_DEFAULT for such types.
 */
#define FRAMEWORK_NV_THROW_IF(condition, exception, message)                                       \
    do {                                                                                           \
        static_assert(                                                                             \
                ::StringConstructibleException<exception>,                                         \
                #exception " must inherit from std::exception and have a "                         \
                           "constructor accepting const std::string&");                            \
        if (condition) {                                                                           \
            RT_LOGC_ERROR(                                                                         \
                    framework::utils::Core::CoreNvApi,                                             \
                    "NV_ERROR: {}:{} {}",                                                          \
                    std::source_location::current().file_name(),                                   \
                    std::source_location::current().line(),                                        \
                    message);                                                                      \
            throw exception(message);                                                              \
        }                                                                                          \
    } while (0)

/**
 * Check condition and throw exception on error (default constructor version)
 *
 * This macro checks a condition and throws an exception using its default
 * constructor if the condition is true. It logs the error information but
 * cannot pass the message to the exception constructor.
 *
 * Use this macro for exception types that only have default constructors
 * (e.g., std::bad_alloc, std::bad_cast).
 *
 * @param[in] condition The condition to check
 * @param[in] exception The exception type to throw (must have default
 * constructor)
 * @param[in] message The message to log (not passed to exception constructor)
 *
 * @throws exception If the condition is true
 *
 * @note The message is only used for logging; it is not passed to the
 * exception constructor since this macro is for exceptions with default
 * constructors only.
 */
#define FRAMEWORK_NV_THROW_IF_DEFAULT(condition, exception, message)                               \
    do {                                                                                           \
        static_assert(                                                                             \
                std::is_base_of_v<std::exception, exception> &&                                    \
                        std::is_default_constructible_v<exception>,                                \
                #exception " must inherit from std::exception and be "                             \
                           "default constructible");                                               \
        if (condition) {                                                                           \
            RT_LOGC_ERROR(                                                                         \
                    framework::utils::Core::CoreNvApi,                                             \
                    "NV_ERROR: {}:{} {}",                                                          \
                    std::source_location::current().file_name(),                                   \
                    std::source_location::current().line(),                                        \
                    message);                                                                      \
            throw exception();                                                                     \
        }                                                                                          \
    } while (0)

/**
 * Check CUDA runtime API call without throwing exceptions
 *
 * This macro wraps CUDA runtime API calls to automatically check for errors.
 * On error, it logs the error information but does not throw exceptions.
 * The macro is instrumentation-aware and excludes itself from VectorCAST
 * coverage.
 *
 * @param[in] expr_to_check CUDA runtime API expression to evaluate and check
 *
 * @note This macro clears any pending CUDA errors with cudaGetLastError()
 * @note Does not throw exceptions - only logs errors
 *
 * @see FRAMEWORK_CUDA_RUNTIME_EXPR_CHECK_THROW For exception-based version
 * with framework::utils::CudaRuntimeException
 */
#define FRAMEWORK_CUDA_RUNTIME_EXPR_CHECK_NO_THROW(expr_to_check)                                  \
    do {                                                                                           \
        FRAMEWORK_PRAGMA_DIAGNOSTIC_PUSH                                                           \
        FRAMEWORK_PRAGMA_IGNORE_UNKNOWN_PRAGMAS                                                    \
        FRAMEWORK_PRAGMA_VCAST_START                                                               \
        if (const cudaError_t result = expr_to_check; result != cudaSuccess) {                     \
            const cudaError_t last_error = cudaGetLastError();                                     \
            RT_LOGC_ERROR(                                                                         \
                    framework::utils::Core::CoreCudaRuntime,                                       \
                    "CUDA Runtime Error: {}:{}:{} (last "                                          \
                    "error {})",                                                                   \
                    std::source_location::current().file_name(),                                   \
                    std::source_location::current().line(),                                        \
                    cudaGetErrorString(result),                                                    \
                    cudaGetErrorString(last_error));                                               \
        }                                                                                          \
        FRAMEWORK_PRAGMA_VCAST_END                                                                 \
        FRAMEWORK_PRAGMA_DIAGNOSTIC_POP                                                            \
    } while (0)

/**
 * Check CUDA driver API call without throwing exceptions
 *
 * This macro wraps CUDA driver API calls to automatically check for errors.
 * On error, it retrieves the error string, logs the error information, but
 * does not throw exceptions. The macro is instrumentation-aware and excludes
 * itself from VectorCAST coverage.
 *
 * @param[in] cu_result CUDA driver API expression that returns CUresult
 *
 * @note Does not throw exceptions - only logs errors
 * @note Uses cuGetErrorString to retrieve human-readable error messages
 * @note Handles cases where cuGetErrorString might return nullptr
 *
 * @see FRAMEWORK_CUDA_DRIVER_CHECK_THROW For exception-based version
 * @see FRAMEWORK_CUDA_RUNTIME_EXPR_CHECK_NO_THROW For CUDA runtime API version
 */
#define FRAMEWORK_CUDA_DRIVER_CHECK_NO_THROW(cu_result)                                            \
    do {                                                                                           \
        FRAMEWORK_PRAGMA_DIAGNOSTIC_PUSH                                                           \
        FRAMEWORK_PRAGMA_IGNORE_UNKNOWN_PRAGMAS                                                    \
        FRAMEWORK_PRAGMA_VCAST_START                                                               \
        if (const CUresult aerial_cu_result = cu_result; aerial_cu_result != CUDA_SUCCESS) {       \
            const char *pErrStr{};                                                                 \
            cuGetErrorString(aerial_cu_result, &pErrStr);                                          \
            RT_LOGC_ERROR(                                                                         \
                    framework::utils::Core::CoreCudaDriver,                                        \
                    "[{}:{}] CUDA driver error {}",                                                \
                    std::source_location::current().file_name(),                                   \
                    std::source_location::current().line(),                                        \
                    pErrStr != nullptr ? pErrStr : "Unknown error");                               \
        }                                                                                          \
        FRAMEWORK_PRAGMA_VCAST_END                                                                 \
        FRAMEWORK_PRAGMA_DIAGNOSTIC_POP                                                            \
    } while (0)

/**
 * Check CUDA runtime API call and throw exception on error
 *
 * This macro wraps CUDA runtime API calls to automatically check for errors.
 * On error, it logs the error information and throws a
 * framework::utils::CudaRuntimeException. The macro is instrumentation-aware
 * and excludes itself from VectorCAST coverage.
 *
 * @param[in] expr_to_check CUDA runtime API expression to evaluate and check
 *
 * @throws framework::utils::CudaRuntimeException If the CUDA call returns an
 * error
 *
 * @note This macro clears any pending CUDA errors with cudaGetLastError()
 * @note Always throws when the current expression fails, regardless of previous
 * error state
 *
 * @see FRAMEWORK_CUDA_RUNTIME_EXPR_CHECK_NO_THROW For non-throwing version
 * @see FRAMEWORK_CUDA_RUNTIME_CHECK_THROW For simpler exception-based version
 */
#define FRAMEWORK_CUDA_RUNTIME_EXPR_CHECK_THROW(expr_to_check)                                     \
    do {                                                                                           \
        FRAMEWORK_PRAGMA_DIAGNOSTIC_PUSH                                                           \
        FRAMEWORK_PRAGMA_IGNORE_UNKNOWN_PRAGMAS                                                    \
        FRAMEWORK_PRAGMA_VCAST_START                                                               \
        if (const cudaError_t result = expr_to_check; result != cudaSuccess) {                     \
            const cudaError_t last_error = cudaGetLastError();                                     \
            RT_LOGC_ERROR(                                                                         \
                    framework::utils::Core::CoreCudaRuntime,                                       \
                    "CUDA Runtime Error: {}:{}:{} (last "                                          \
                    "error {})",                                                                   \
                    std::source_location::current().file_name(),                                   \
                    std::source_location::current().line(),                                        \
                    cudaGetErrorString(result),                                                    \
                    cudaGetErrorString(last_error));                                               \
            throw framework::utils::CudaRuntimeException(result);                                  \
        }                                                                                          \
        FRAMEWORK_PRAGMA_VCAST_END                                                                 \
        FRAMEWORK_PRAGMA_DIAGNOSTIC_POP                                                            \
    } while (0)

/**
 * Check CUDA runtime API call and throw CudaRuntimeException on error
 *
 * This macro wraps CUDA runtime API calls to automatically check for errors.
 * On error, it logs the error information and throws a
 * framework::utils::CudaRuntimeException.
 *
 * @param[in] cuda_result CUDA runtime API expression that returns cudaError_t
 *
 * @throws framework::utils::CudaRuntimeException If the CUDA call returns
 * non-success status
 *
 * @see FRAMEWORK_CUDA_RUNTIME_EXPR_CHECK_THROW For more complex error handling
 * version
 */
#define FRAMEWORK_CUDA_RUNTIME_CHECK_THROW(cuda_result)                                            \
    do {                                                                                           \
        if (const cudaError_t aerial_cuda_result = cuda_result;                                    \
            aerial_cuda_result != cudaSuccess) {                                                   \
            RT_LOGC_ERROR(                                                                         \
                    framework::utils::Core::CoreCudaRuntime,                                       \
                    "[{}:{}] CUDA runtime error {}",                                               \
                    std::source_location::current().file_name(),                                   \
                    std::source_location::current().line(),                                        \
                    cudaGetErrorString(aerial_cuda_result));                                       \
            throw framework::utils::CudaRuntimeException(aerial_cuda_result);                      \
        }                                                                                          \
    } while (0)

/**
 * Check CUDA driver API call and throw cuda_driver_exception on error
 *
 * This macro wraps CUDA driver API calls to automatically check for errors.
 * On error, it retrieves the error string, logs the error information, and
 * throws a framework::utils::CudaDriverException.
 *
 * @param[in] cu_result CUDA driver API expression that returns CUresult
 *
 * @throws framework::utils::CudaDriverException If the CUDA driver call
 * returns non-success status
 *
 * @note Uses cuGetErrorString to retrieve human-readable error messages
 * @note Handles cases where cuGetErrorString might return nullptr
 *
 * @see FRAMEWORK_CUDA_RUNTIME_EXPR_CHECK_THROW For CUDA runtime API version
 * @see FRAMEWORK_NV_CHECK_THROW For NV API version
 */
#define FRAMEWORK_CUDA_DRIVER_CHECK_THROW(cu_result)                                               \
    do {                                                                                           \
        if (const CUresult aerial_cu_result = cu_result; aerial_cu_result != CUDA_SUCCESS) {       \
            const char *pErrStr{};                                                                 \
            cuGetErrorString(aerial_cu_result, &pErrStr);                                          \
            RT_LOGC_ERROR(                                                                         \
                    framework::utils::Core::CoreCudaDriver,                                        \
                    "[{}:{}] CUDA driver error {}",                                                \
                    std::source_location::current().file_name(),                                   \
                    std::source_location::current().line(),                                        \
                    pErrStr != nullptr ? pErrStr : "Unknown error");                               \
            throw framework::utils::CudaDriverException(aerial_cu_result);                         \
        }                                                                                          \
    } while (0)
// NOLINTEND(cppcoreguidelines-avoid-do-while,cppcoreguidelines-macro-usage,readability-function-cognitive-complexity)

#endif // FRAMEWORK_ERROR_MACROS_HPP
