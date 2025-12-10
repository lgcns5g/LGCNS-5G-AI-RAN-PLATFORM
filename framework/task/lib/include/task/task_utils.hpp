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

#ifndef FRAMEWORK_TASK_TASK_UTILS_HPP
#define FRAMEWORK_TASK_TASK_UTILS_HPP

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

#include <pthread.h>

#include <wise_enum.h>

#include "task/task_export.hpp"

namespace framework::task {

/// File write mode for trace output functions
enum class TraceWriteMode {
    Overwrite, //!< Overwrite existing file (default)
    Append     //!< Append to existing file
};

} // namespace framework::task

// WISE_ENUM_ADAPT must be called at global namespace scope for ADL to work
WISE_ENUM_ADAPT(framework::task::TraceWriteMode, Overwrite, Append)

namespace framework::task {

/**
 * Parse core list from kernel parameter value
 * Supports ranges (4-64) and individual cores (1,2,3)
 * @param[in] param_value Parameter value string
 * @return Set of core IDs
 */
[[nodiscard]] TASK_EXPORT std::set<std::uint32_t>
parse_core_list(const std::string_view param_value);

/**
 * Validate that cores are properly configured for RT workloads
 * Checks if cores are in isolcpus, nohz_full, and rcu_nocbs lists
 * @param[in] cmdline Kernel command line string
 * @param[in] cores Vector of core IDs to validate
 * @return true if all cores are properly configured for RT, false otherwise
 */
[[nodiscard]] TASK_EXPORT bool
validate_rt_core_config(const std::string_view cmdline, const std::vector<std::uint32_t> &cores);

// ============================================================================
// Statistics Utilities - Common functions for task_monitor and timed_trigger
// ============================================================================

/**
 * Timing statistics result structure
 * Contains comprehensive timing statistics in microseconds
 */
struct TASK_EXPORT TimingStatistics final {
    double min_us{};     //!< Minimum value in microseconds
    double max_us{};     //!< Maximum value in microseconds
    double avg_us{};     //!< Average value in microseconds
    double median_us{};  //!< Median value in microseconds
    double p95_us{};     //!< 95th percentile in microseconds
    double p99_us{};     //!< 99th percentile in microseconds
    double std_us{};     //!< Standard deviation in microseconds
    std::size_t count{}; //!< Number of values
};

/**
 * Calculate standard deviation for a collection of int64_t values
 * @param[in] values Vector of int64_t values to analyze
 * @param[in] mean Pre-calculated mean of the values
 * @return Standard deviation
 */
[[nodiscard]] TASK_EXPORT double
calculate_standard_deviation(const std::vector<std::int64_t> &values, double mean);

/**
 * Calculate standard deviation for a collection of double values
 * @param[in] values Vector of double values to analyze
 * @param[in] mean Pre-calculated mean of the values
 * @return Standard deviation
 */
[[nodiscard]] TASK_EXPORT double
calculate_standard_deviation(const std::vector<double> &values, double mean);

/**
 * Calculate percentile from sorted vector
 * @param[in] sorted_values Sorted values vector
 * @param[in] percentile Percentile to calculate (0.0 to 1.0)
 * @return Percentile value
 */
[[nodiscard]] TASK_EXPORT double
calculate_percentile(const std::vector<double> &sorted_values, double percentile);

/**
 * Convert nanoseconds to microseconds (int64_t version)
 * @param[in] nanos_count Nanoseconds value as int64_t
 * @return Microseconds as double
 */
[[nodiscard]] TASK_EXPORT double nanos_to_micros_int64(std::int64_t nanos_count);

/**
 * Convert nanoseconds to microseconds (double version)
 * @param[in] nanos_count Nanoseconds value as double
 * @return Microseconds as double
 */
[[nodiscard]] TASK_EXPORT double nanos_to_micros_double(double nanos_count);

/**
 * Calculate comprehensive timing statistics from int64_t nanosecond values
 * @param[in] values_ns Vector of timing values in nanoseconds (int64_t)
 * @return TimingStatistics structure with all calculated metrics in
 * microseconds
 */
[[nodiscard]] TASK_EXPORT TimingStatistics
calculate_timing_statistics(const std::vector<std::int64_t> &values_ns);

/**
 * Calculate comprehensive timing statistics from double nanosecond values
 * @param[in] values_ns Vector of timing values in nanoseconds (double)
 * @return TimingStatistics structure with all calculated metrics in
 * microseconds
 */
[[nodiscard]] TASK_EXPORT TimingStatistics
calculate_timing_statistics(const std::vector<double> &values_ns);

// ============================================================================
// Record Management Utilities
// ============================================================================

/**
 * Calculate maximum number of records for a given memory budget
 * @tparam RecordType The type of record to calculate for
 * @param[in] max_bytes Maximum bytes to allocate for records
 * @return Maximum number of records that fit in the given byte limit
 */
template <typename RecordType>
[[nodiscard]] constexpr std::size_t
calculate_max_records_for_bytes(const std::size_t max_bytes) noexcept {
    if (max_bytes == 0 || sizeof(RecordType) == 0) {
        return 0;
    }
    return max_bytes / sizeof(RecordType);
}

// ============================================================================
// GPS/TAI Timing Utilities
// ============================================================================

/**
 * Parameters for calculating start time for next period boundary
 */
struct TASK_EXPORT StartTimeParams final {
    std::uint64_t current_time_ns{}; //!< Current time in nanoseconds since epoch
    std::uint64_t period_ns{};       //!< Period in nanoseconds for alignment
    std::int64_t gps_alpha{};        //!< GPS alpha parameter for frequency adjustment (default 0)
    std::int64_t gps_beta{};         //!< GPS beta parameter for phase adjustment (default 0)
};

/**
 * Calculate current TAI offset
 * Retrieves TAI (International Atomic Time) offset from system clock using
 * adjtimex()
 * @return TAI offset as std::chrono::nanoseconds, or 0 if retrieval fails
 */
[[nodiscard]] TASK_EXPORT std::chrono::nanoseconds calculate_tai_offset();

/**
 * Calculate start time for next period boundary
 * Computes the next aligned time boundary based on GPS parameters and period
 *
 * @param[in] params Parameters for start time calculation
 * @param[in] tai_offset TAI offset as std::chrono::nanoseconds
 * @return Next aligned start time in nanoseconds since epoch
 */
[[nodiscard]] TASK_EXPORT std::uint64_t calculate_start_time_for_next_period(
        const StartTimeParams &params, std::chrono::nanoseconds tai_offset);

/**
 * Calculate start time for next period boundary using current system TAI offset
 * Convenience wrapper that retrieves current TAI offset and calculates start
 * time
 *
 * @param[in] params Parameters for start time calculation
 * @return Next aligned start time in nanoseconds since epoch
 */
[[nodiscard]] TASK_EXPORT std::uint64_t
calculate_start_time_for_next_period(const StartTimeParams &params);

// ============================================================================
// Thread Configuration Utilities
// ============================================================================

/**
 * Thread configuration parameters
 */
struct ThreadConfig {
    std::optional<std::uint32_t> core_id; //!< CPU core ID to pin thread to (nullopt = no pinning)
    std::optional<std::uint32_t>
            priority; //!< Thread priority level (1-99, nullopt = normal scheduling)
};

/**
 * Pin current thread to specified CPU core
 * @param[in] core_id CPU core ID to pin to
 * @return Error code indicating success or failure
 */
[[nodiscard]] TASK_EXPORT std::error_code pin_current_thread_to_core(std::uint32_t core_id);

/**
 * Pin thread to specified CPU core using thread handle
 * @param[in] thread_handle Native pthread handle
 * @param[in] core_id CPU core ID to pin to
 * @return Error code indicating success or failure
 */
[[nodiscard]] TASK_EXPORT std::error_code
pin_thread_to_core(pthread_t thread_handle, std::uint32_t core_id);

/**
 * Set real-time priority for current thread
 * @param[in] priority Real-time priority (1-99)
 * @return Error code indicating success or failure
 */
[[nodiscard]] TASK_EXPORT std::error_code set_current_thread_priority(std::uint32_t priority);

/**
 * Set real-time priority for thread using thread handle
 * @param[in] thread_handle Native pthread handle
 * @param[in] priority Real-time priority (1-99)
 * @return Error code indicating success or failure
 */
[[nodiscard]] TASK_EXPORT std::error_code
set_thread_priority(pthread_t thread_handle, std::uint32_t priority);

/**
 * Configure thread with optional core pinning and priority
 * Applies both pinning and priority settings for current thread
 * @param[in] config Thread configuration parameters
 * @return Error code indicating success or failure
 */
[[nodiscard]] TASK_EXPORT std::error_code configure_current_thread(ThreadConfig config);

/**
 * Configure thread with optional core pinning and priority using std::thread
 * Applies both pinning and priority settings for specified thread
 * @param[in] thread std::thread reference
 * @param[in] config Thread configuration parameters
 * @return Error code indicating success or failure
 */
[[nodiscard]] TASK_EXPORT std::error_code
configure_thread(std::thread &thread, ThreadConfig config);

/**
 * Enable sanitizer compatibility for processes with elevated capabilities
 *
 * When a process has CAP_SYS_NICE (for real-time scheduling), it becomes
 * non-dumpable by default for security. This prevents LeakSanitizer and other
 * debugging tools from working. This function makes the process dumpable again
 * when sanitizers are enabled.
 */
TASK_EXPORT void enable_sanitizer_compatibility();

} // namespace framework::task

#endif // FRAMEWORK_TASK_TASK_UTILS_HPP
