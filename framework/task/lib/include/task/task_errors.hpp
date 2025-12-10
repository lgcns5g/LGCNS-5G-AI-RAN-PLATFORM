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
 * @file task_errors.hpp
 * @brief Error codes for task framework operations
 *
 * Provides type-safe error codes compatible with std::error_code
 * for task management, monitoring, and thread configuration operations.
 */

#ifndef FRAMEWORK_TASK_TASK_ERRORS_HPP
#define FRAMEWORK_TASK_TASK_ERRORS_HPP

#include <array>
#include <cstdint>
#include <format>
#include <limits>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>

#include <wise_enum.h>

#include "task/task_export.hpp"

namespace framework::task {

/**
 * Task framework error codes compatible with std::error_code
 *
 * This enum class provides type-safe error codes for task framework operations
 * that integrate seamlessly with the standard C++ error handling framework.
 */
// clang-format off
enum class TaskErrc : std::uint8_t {
    Success,                    //!< Operation succeeded
    AlreadyRunning,             //!< Operation failed: already running
    NotStarted,                 //!< Operation failed: not started
    QueueFull,                  //!< Operation failed: queue is full
    InvalidParameter,           //!< Invalid parameter provided
    TaskNotFound,               //!< Task not found in registry
    ThreadConfigFailed,         //!< Thread configuration failed
    ThreadPinFailed,            //!< Thread core pinning failed
    ThreadPriorityFailed,       //!< Thread priority setting failed
    FileOpenFailed,             //!< File open operation failed
    FileWriteFailed,            //!< File write operation failed
    PthreadGetaffinityFailed,   //!< pthread_getaffinity_np failed
    PthreadSetaffinityFailed,   //!< pthread_setaffinity_np failed
    PthreadGetschedparamFailed, //!< pthread_getschedparam failed
    PthreadSetschedparamFailed  //!< pthread_setschedparam failed
};
// clang-format on

static_assert(
        static_cast<std::uint32_t>(TaskErrc::PthreadSetschedparamFailed) <=
                std::numeric_limits<std::uint8_t>::max(),
        "TaskErrc enumerator values must fit in std::uint8_t");

} // namespace framework::task

// WISE_ENUM_ADAPT must be called at global namespace scope for ADL to work
WISE_ENUM_ADAPT(
        framework::task::TaskErrc,
        Success,
        AlreadyRunning,
        NotStarted,
        QueueFull,
        InvalidParameter,
        TaskNotFound,
        ThreadConfigFailed,
        ThreadPinFailed,
        ThreadPriorityFailed,
        FileOpenFailed,
        FileWriteFailed,
        PthreadGetaffinityFailed,
        PthreadSetaffinityFailed,
        PthreadGetschedparamFailed,
        PthreadSetschedparamFailed)

// Register TaskErrc as an error code enum to enable implicit conversion to
// std::error_code
// NOTE: This MUST come before any functions that use TaskErrc with
// std::error_code
namespace std {
template <> struct is_error_code_enum<framework::task::TaskErrc> : true_type {};
} // namespace std

namespace framework::task {

/**
 * Custom error category for task framework errors
 *
 * This class provides human-readable error messages and integrates task errors
 * with the standard C++ error handling system.
 */
class TaskErrorCategory final : public std::error_category {
private:
    // Compile-time table indexed by the enum's underlying value
    static constexpr std::array<std::string_view, 15> KMESSAGES{
            "Success: Operation completed successfully",
            "Already running: Operation cannot be started because it is already "
            "running",
            "Not started: Operation cannot be performed because system is not "
            "started",
            "Queue full: Cannot enqueue event because queue is full",
            "Invalid parameter: Parameter value is invalid or out of range",
            "Task not found: Task ID not found in monitoring registry",
            "Thread configuration failed: Unable to configure thread parameters",
            "Thread pin failed: Unable to pin thread to CPU core",
            "Thread priority failed: Unable to set thread priority",
            "File open failed: Unable to open file for writing",
            "File write failed: Unable to write data to file",
            "pthread_getaffinity_np failed: Unable to get thread CPU affinity",
            "pthread_setaffinity_np failed: Unable to set thread CPU affinity",
            "pthread_getschedparam failed: Unable to get thread scheduling "
            "parameters",
            "pthread_setschedparam failed: Unable to set thread scheduling "
            "parameters"};

    // Ensure KMESSAGES array size matches the number of enum values
    static_assert(
            KMESSAGES.size() == ::wise_enum::size<TaskErrc>,
            "KMESSAGES array size must match the number of TaskErrc enum values");

public:
    /**
     * Get the name of this error category
     *
     * @return The category name as a C-style string
     */
    [[nodiscard]] const char *name() const noexcept override { return "adsp::task"; }

    /**
     * Get a descriptive message for the given error code
     *
     * @param[in] condition The error code value
     * @return A descriptive error message
     */
    [[nodiscard]] std::string message(const int condition) const override {
        const auto idx = static_cast<std::size_t>(condition);
        if (idx < KMESSAGES.size()) {
            return std::string{*std::next(KMESSAGES.begin(), static_cast<std::ptrdiff_t>(idx))};
        }
        return std::format("Unknown task error: {}", condition);
    }

    /**
     * Map task errors to standard error conditions where applicable
     *
     * @param[in] condition The error code value
     * @return The equivalent standard error condition, or a default-constructed
     * condition
     */
    [[nodiscard]] std::error_condition
    default_error_condition(const int condition) const noexcept override {
        switch (static_cast<TaskErrc>(condition)) {
        case TaskErrc::Success:
            return {};
        case TaskErrc::InvalidParameter:
            return std::errc::invalid_argument;
        case TaskErrc::QueueFull:
            return std::errc::no_buffer_space;
        case TaskErrc::TaskNotFound:
            return std::errc::no_such_file_or_directory;
        case TaskErrc::FileOpenFailed:
        case TaskErrc::FileWriteFailed:
            return std::errc::io_error;
        default:
            // For task-specific errors that don't map to standard conditions
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
        const auto errc = static_cast<TaskErrc>(condition);
        return ::wise_enum::to_string(errc).data();
    }
};

/**
 * Get the singleton instance of the task error category
 *
 * @return Reference to the task error category
 */
[[nodiscard]] inline const TaskErrorCategory &task_category() noexcept {
    static const TaskErrorCategory instance{};
    return instance;
}

/**
 * Create an error_code from a TaskErrc value
 *
 * @param[in] errc The task error code
 * @return A std::error_code representing the task error
 */
[[nodiscard]] inline std::error_code make_error_code(const TaskErrc errc) noexcept {
    return {static_cast<int>(errc), task_category()};
}

/**
 * Check if a TaskErrc represents success
 *
 * @param[in] errc The error code to check
 * @return true if the error code represents success, false otherwise
 */
[[nodiscard]] constexpr bool is_success(const TaskErrc errc) noexcept {
    return errc == TaskErrc::Success;
}

/**
 * Check if an error_code represents task success
 *
 * An error code represents success if it evaluates to false (i.e., value is 0).
 * This works regardless of the category (system, task, etc.).
 *
 * @param[in] errc The error code to check
 * @return true if the error code represents success, false otherwise
 */
[[nodiscard]] inline bool is_task_success(const std::error_code &errc) noexcept {
    return !errc; // std::error_code evaluates to false when value is 0 (success)
}

/**
 * Get the name of a TaskErrc enum value
 *
 * @param[in] errc The error code
 * @return The enum name as a string
 */
[[nodiscard]] inline const char *get_error_name(const TaskErrc errc) noexcept {
    return ::wise_enum::to_string(errc).data();
}

/**
 * Get the name of a TaskErrc from a std::error_code
 *
 * @param[in] ec The error code
 * @return The enum name as a string, or "unknown" if not a task error
 */
[[nodiscard]] inline const char *get_error_name(const std::error_code &ec) noexcept {
    if (ec.category() != task_category()) {
        return "unknown";
    }
    return get_error_name(static_cast<TaskErrc>(ec.value()));
}

} // namespace framework::task

#endif // FRAMEWORK_TASK_TASK_ERRORS_HPP
