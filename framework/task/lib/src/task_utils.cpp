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

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

#include <pthread.h>
#include <quill/LogMacros.h>
#include <sched.h>
#include <sys/timex.h>

// Cross-compiler sanitizer detection
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_LEAK__)
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LEAK_SANITIZER_ENABLED 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(leak_sanitizer)
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LEAK_SANITIZER_ENABLED 1
#endif // __has_feature(address_sanitizer) || __has_feature(leak_sanitizer)
#endif // defined(__has_feature)

#if LEAK_SANITIZER_ENABLED
#include <sys/prctl.h>
#endif // LEAK_SANITIZER_ENABLED

#include "log/rt_log_macros.hpp"
#include "task/task_errors.hpp"
#include "task/task_log.hpp"
#include "task/task_utils.hpp"

namespace framework::task {

namespace {

/**
 * Parse a single core number from a token string
 * @param[in] token String containing core number
 * @return Core number, or nullopt if parsing failed
 */
[[nodiscard]] std::optional<std::uint32_t> parse_single_core(const std::string_view token) {
    try {
        return static_cast<std::uint32_t>(std::stoul(std::string(token)));
    } catch (const std::exception &) {
        return std::nullopt;
    }
}

/**
 * Parse a core range from a token string (e.g., "1-4")
 * @param[in] token String containing range specification
 * @param[out] cores Set to add parsed cores to
 * @return true if parsing succeeded, false otherwise
 */
bool parse_core_range(const std::string_view token, std::set<std::uint32_t> &cores) {
    const std::size_t dash_pos = token.find('-');
    if (dash_pos == std::string::npos || dash_pos == 0 || dash_pos == token.length() - 1) {
        return false;
    }

    try {
        const auto range_start =
                static_cast<std::uint32_t>(std::stoul(std::string(token.substr(0, dash_pos))));
        const auto range_end =
                static_cast<std::uint32_t>(std::stoul(std::string(token.substr(dash_pos + 1))));

        if (range_start > range_end) {
            return false;
        }

        for (std::uint32_t core = range_start; core <= range_end; ++core) {
            cores.insert(core);
        }
        return true;
    } catch (const std::exception &) {
        return false;
    }
}

/**
 * Process a single token (can be either a single core or a range)
 * @param[in] token String token to process
 * @param[out] cores Set to add parsed cores to
 */
void process_token(const std::string_view token, std::set<std::uint32_t> &cores) {
    if (token.empty()) {
        return;
    }

    // Try parsing as range first
    if (token.find('-') != std::string::npos) {
        if (!parse_core_range(token, cores)) {
            // Invalid range format - skip this token
        }
    } else {
        // Try parsing as single core
        const auto core = parse_single_core(token);
        if (core.has_value()) {
            cores.insert(core.value());
        }
        // Invalid core number - skip this token
    }
}

} // anonymous namespace

std::set<std::uint32_t> parse_core_list(const std::string_view param_value) {
    std::set<std::uint32_t> cores;

    // Split by commas and process each token
    std::size_t start = 0;
    std::size_t pos = 0;

    while ((pos = param_value.find(',', start)) != std::string::npos) {
        const std::string token = std::string(param_value.substr(start, pos - start));
        process_token(token, cores);
        start = pos + 1;
    }

    // Handle last token
    if (start < param_value.length()) {
        const std::string token = std::string(param_value.substr(start));
        process_token(token, cores);
    }

    return cores;
}

bool validate_rt_core_config(
        const std::string_view cmdline, const std::vector<std::uint32_t> &cores) {
    try {
        bool all_cores_valid = true;

        // Parse isolcpus parameter
        std::set<std::uint32_t> isolated_cores;
        const std::size_t isolcpus_pos = cmdline.find("isolcpus=");
        if (isolcpus_pos != std::string::npos) {
            static constexpr std::size_t ISOLCPUS_PREFIX_LEN = 9; // Skip "isolcpus="
            const std::size_t start = isolcpus_pos + ISOLCPUS_PREFIX_LEN;
            std::size_t end = cmdline.find(' ', start);
            if (end == std::string::npos) {
                end = cmdline.length();
            }

            const std::string isolcpus_value = std::string(cmdline.substr(start, end - start));
            // Skip domain/managed_irq prefixes, find core list
            const std::size_t core_start = isolcpus_value.find_last_of(',');
            if (core_start != std::string::npos) {
                isolated_cores = parse_core_list(isolcpus_value.substr(core_start + 1));
            } else {
                isolated_cores = parse_core_list(isolcpus_value);
            }
        }

        // Parse nohz_full parameter
        std::set<std::uint32_t> nohz_cores;
        const std::size_t nohz_pos = cmdline.find("nohz_full=");
        if (nohz_pos != std::string::npos) {
            static constexpr std::size_t NOHZ_PREFIX_LEN = 10; // Skip "nohz_full="
            const std::size_t start = nohz_pos + NOHZ_PREFIX_LEN;
            std::size_t end = cmdline.find(' ', start);
            if (end == std::string::npos) {
                end = cmdline.length();
            }
            nohz_cores = parse_core_list(cmdline.substr(start, end - start));
        }

        // Parse rcu_nocbs parameter
        std::set<std::uint32_t> nocb_cores;
        const std::size_t nocbs_pos = cmdline.find("rcu_nocbs=");
        if (nocbs_pos != std::string::npos) {
            static constexpr std::size_t NOCBS_PREFIX_LEN = 10; // Skip "rcu_nocbs="
            const std::size_t start = nocbs_pos + NOCBS_PREFIX_LEN;
            std::size_t end = cmdline.find(' ', start);
            if (end == std::string::npos) {
                end = cmdline.length();
            }
            nocb_cores = parse_core_list(cmdline.substr(start, end - start));
        }

        // Validate each core
        for (const std::uint32_t core_id : cores) {
            const bool in_isolcpus = isolated_cores.contains(core_id);
            const bool in_nohz = nohz_cores.contains(core_id);
            const bool in_nocbs = nocb_cores.contains(core_id);

            if (!in_isolcpus || !in_nohz || !in_nocbs) {
                RT_LOGC_WARN(
                        TaskLog::TaskScheduler,
                        "Core {} assigned RT priority but not optimally configured: "
                        "isolcpus={}, nohz_full={}, rcu_nocbs={}",
                        core_id,
                        in_isolcpus,
                        in_nohz,
                        in_nocbs);
                all_cores_valid = false;
            } else {
                RT_LOGC_DEBUG(
                        TaskLog::TaskScheduler,
                        "Core {} properly configured for RT workloads",
                        core_id);
            }
        }

        return all_cores_valid;

    } catch (const std::exception &e) {
        RT_LOGC_WARN(
                TaskLog::TaskScheduler, "Failed to validate RT core configuration: {}", e.what());
        return false;
    }
}

// ============================================================================
// Statistics Utilities Implementation
// ============================================================================

namespace {

/**
 * Template implementation for calculating standard deviation
 * @tparam T Value type (int64_t or double)
 * @param[in] values Vector of values to analyze
 * @param[in] mean Pre-calculated mean of the values
 * @return Standard deviation
 */
template <typename T>
[[nodiscard]] double
calculate_standard_deviation_impl(const std::vector<T> &values, const double mean) {
    if (values.size() <= 1) {
        return 0.0;
    }

    double sum_squared_diff = 0.0;
    for (const auto &value : values) {
        const double diff = static_cast<double>(value) - mean;
        sum_squared_diff += diff * diff;
    }

    return std::sqrt(sum_squared_diff / static_cast<double>(values.size() - 1));
}

/**
 * Template implementation for nanoseconds to microseconds conversion
 * @tparam T Value type (int64_t or double)
 * @param[in] nanos_count Value in nanoseconds
 * @return Value in microseconds as double
 */
template <typename T> [[nodiscard]] double nanos_to_micros_impl(const T nanos_count) {
    static constexpr double NANOS_TO_MICROS_DIVISOR = 1000.0;
    return static_cast<double>(nanos_count) / NANOS_TO_MICROS_DIVISOR;
}

/**
 * Template implementation for calculating timing statistics
 * @tparam T Value type (int64_t or double)
 * @param[in] values_ns Vector of timing values in nanoseconds
 * @return TimingStatistics structure with all calculated metrics in
 * microseconds
 */
template <typename T>
[[nodiscard]] TimingStatistics calculate_timing_statistics_impl(const std::vector<T> &values_ns) {
    TimingStatistics stats{};

    if (values_ns.empty()) {
        return stats;
    }

    stats.count = values_ns.size();

    // Convert to microseconds and calculate basic metrics
    std::vector<double> values_us;
    values_us.reserve(values_ns.size());

    double total_us = 0.0;
    double min_val_us = std::numeric_limits<double>::max();
    double max_val_us = std::numeric_limits<double>::lowest();

    static constexpr double NANOS_TO_MICROS = 1000.0;
    for (const auto &value_ns : values_ns) {
        const double value_us = static_cast<double>(value_ns) / NANOS_TO_MICROS;
        values_us.push_back(value_us);
        total_us += value_us;
        min_val_us = std::min(min_val_us, value_us);
        max_val_us = std::max(max_val_us, value_us);
    }

    stats.min_us = min_val_us;
    stats.max_us = max_val_us;
    stats.avg_us = total_us / static_cast<double>(values_us.size());

    // Sort for percentiles
    std::sort(values_us.begin(), values_us.end());

    // NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
    stats.median_us = calculate_percentile(values_us, 0.5);
    stats.p95_us = calculate_percentile(values_us, 0.95);
    stats.p99_us = calculate_percentile(values_us, 0.99);
    // NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

    // Calculate standard deviation
    stats.std_us = calculate_standard_deviation_impl(values_us, stats.avg_us);

    return stats;
}

} // anonymous namespace

double calculate_standard_deviation(const std::vector<std::int64_t> &values, const double mean) {
    return calculate_standard_deviation_impl(values, mean);
}

double calculate_standard_deviation(const std::vector<double> &values, const double mean) {
    return calculate_standard_deviation_impl(values, mean);
}

double calculate_percentile(const std::vector<double> &sorted_values, const double percentile) {
    if (sorted_values.empty()) {
        return 0.0;
    }

    if (sorted_values.size() == 1) {
        return sorted_values[0];
    }

    // Calculate the exact position (0-based indexing)
    const double exact_position = static_cast<double>(sorted_values.size() - 1) * percentile;
    const auto lower_index = static_cast<std::size_t>(std::floor(exact_position));
    const auto upper_index = static_cast<std::size_t>(std::ceil(exact_position));

    // If exact_position is an integer, return that value
    if (lower_index == upper_index) {
        return sorted_values[lower_index];
    }

    // Otherwise, interpolate between the two values
    const double lower_value = sorted_values[lower_index];
    const double upper_value = sorted_values[upper_index];
    const double weight = exact_position - static_cast<double>(lower_index);

    return lower_value + weight * (upper_value - lower_value);
}

double nanos_to_micros_int64(const std::int64_t nanos_count) {
    return nanos_to_micros_impl(nanos_count);
}

double nanos_to_micros_double(const double nanos_count) {
    return nanos_to_micros_impl(nanos_count);
}

TimingStatistics calculate_timing_statistics(const std::vector<std::int64_t> &values_ns) {
    return calculate_timing_statistics_impl(values_ns);
}

TimingStatistics calculate_timing_statistics(const std::vector<double> &values_ns) {
    return calculate_timing_statistics_impl(values_ns);
}

// ============================================================================
// GPS/TAI Timing Utilities Implementation
// ============================================================================

std::chrono::nanoseconds calculate_tai_offset() {
    timex tmx{};

    if (adjtimex(&tmx) == -1) {
        RT_LOGC_ERROR(TaskLog::TaskScheduler, "adjtimex failed: error code {}", errno);
        return std::chrono::nanoseconds{0};
    }

    const std::chrono::nanoseconds tai_offset_ns{
            static_cast<std::int64_t>(tmx.tai) * 1'000'000'000LL};
    RT_LOGC_DEBUG(TaskLog::TaskScheduler, "Current TAI offset: {}s", tmx.tai);

    return tai_offset_ns;
}

std::uint64_t calculate_start_time_for_next_period(
        const StartTimeParams &params, const std::chrono::nanoseconds tai_offset) {

    // Validate input parameters
    if (params.period_ns == 0) {
        throw std::invalid_argument("period_ns cannot be zero");
    }

    // GPS/TAI timing constants
    static constexpr std::int64_t TAI_GPS_EPOCH_DELTA =
            315'964'800ULL; // Jan 6th 1980(GPS epoch) - Jan 1st 1970 (TAI epoch)
    static constexpr std::int64_t GPS_TO_TAI_LAG = 19ULL; // GPS lags TAI by 19s

    // Convert to GPS time scale
    const std::uint64_t tai_to_gps_offset_ns =
            static_cast<std::uint64_t>(TAI_GPS_EPOCH_DELTA + GPS_TO_TAI_LAG) * 1'000'000'000ULL;

    const auto tai_offset_ns = static_cast<std::uint64_t>(tai_offset.count());
    RT_LOGC_DEBUG(TaskLog::TaskScheduler, "TAI offset: {}ns", tai_offset_ns);

    // Convert current time to GPS time scale
    const std::uint64_t gps_current = params.current_time_ns + tai_offset_ns - tai_to_gps_offset_ns;

    // Calculate GPS offset from alpha and beta parameters
    const std::int64_t gps_offset =
            (params.gps_beta * 10'000'000LL) + ((params.gps_alpha * 10'000LL) / 12'288LL);

    RT_LOGC_DEBUG(
            TaskLog::TaskScheduler,
            "GPS parameters: alpha={}, beta={}, calculated_offset={}ns",
            params.gps_alpha,
            params.gps_beta,
            gps_offset);

    // Find next period boundary in GPS time scale
    const std::uint64_t gps_next_boundary = (gps_current / params.period_ns + 1) * params.period_ns;

    // Calculate offset to next boundary
    auto gps_next_boundary_offset = static_cast<std::int64_t>(gps_next_boundary - gps_current);

    // Add GPS offset that will be reverted by the PHY module to maintain
    // alignment
    gps_next_boundary_offset += gps_offset % static_cast<std::int64_t>(params.period_ns);

    // Ensure offset is positive by adding full periods if necessary
    while (gps_next_boundary_offset < 0) {
        gps_next_boundary_offset += static_cast<std::int64_t>(params.period_ns);
    }

    const std::uint64_t next_start_time =
            params.current_time_ns +
            static_cast<std::uint64_t>(
                    gps_next_boundary_offset % static_cast<std::int64_t>(params.period_ns));

    RT_LOGC_DEBUG(
            TaskLog::TaskScheduler,
            "Current time: {}ns, calculated start time: {}ns, period: {}ns",
            params.current_time_ns,
            next_start_time,
            params.period_ns);

    return next_start_time;
}

std::uint64_t calculate_start_time_for_next_period(const StartTimeParams &params) {

    const std::chrono::nanoseconds tai_offset = calculate_tai_offset();
    return calculate_start_time_for_next_period(params, tai_offset);
}

// ============================================================================
// Thread Configuration Utilities Implementation
// ============================================================================

std::error_code pin_current_thread_to_core(const std::uint32_t core_id) {
    return pin_thread_to_core(pthread_self(), core_id);
}

std::error_code pin_thread_to_core(const pthread_t thread_handle, const std::uint32_t core_id) {
    cpu_set_t cpuset{};
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);

    const int result = pthread_setaffinity_np(thread_handle, sizeof(cpu_set_t), &cpuset);
    if (result != 0) {
        RT_LOGC_WARN(
                TaskLog::TaskScheduler,
                "Failed to pin thread to core {}: error code {}",
                core_id,
                result);
        return make_error_code(TaskErrc::PthreadSetaffinityFailed);
    }

    RT_LOGC_DEBUG(TaskLog::TaskScheduler, "Successfully pinned thread to core {}", core_id);
    return make_error_code(TaskErrc::Success);
}

std::error_code set_current_thread_priority(const std::uint32_t priority) {
    return set_thread_priority(pthread_self(), priority);
}

std::error_code set_thread_priority(const pthread_t thread_handle, const std::uint32_t priority) {
    struct sched_param param {};
    param.sched_priority = static_cast<int>(priority);

    const int result = pthread_setschedparam(thread_handle, SCHED_FIFO, &param);
    if (result != 0) {
        RT_LOGC_WARN(
                TaskLog::TaskScheduler,
                "Failed to set thread priority to {}: error code {}",
                priority,
                result);
        return make_error_code(TaskErrc::PthreadSetschedparamFailed);
    }

    RT_LOGC_DEBUG(TaskLog::TaskScheduler, "Successfully set thread priority to {}", priority);
    return make_error_code(TaskErrc::Success);
}

std::error_code configure_current_thread(const ThreadConfig config) {
    const pthread_t thread_handle = pthread_self();
    std::error_code result{make_error_code(TaskErrc::Success)};

    // Set thread priority if specified
    if (config.priority.has_value()) {
        const std::error_code priority_result =
                set_thread_priority(thread_handle, config.priority.value());
        if (priority_result) {
            result = priority_result;
        }
    }

    // Pin to CPU core if specified
    if (config.core_id.has_value()) {
        const std::error_code pin_result =
                pin_thread_to_core(thread_handle, config.core_id.value());
        if (pin_result) {
            result = !result ? pin_result : result; // Preserve first error
        }
    }

    return result;
}

std::error_code configure_thread(std::thread &thread, const ThreadConfig config) {
    const pthread_t thread_handle = thread.native_handle();
    std::error_code result{make_error_code(TaskErrc::Success)};

    // Set thread priority if specified
    if (config.priority.has_value()) {
        const std::error_code priority_result =
                set_thread_priority(thread_handle, config.priority.value());
        if (priority_result) {
            result = priority_result;
        }
    }

    // Pin to CPU core if specified
    if (config.core_id.has_value()) {
        const std::error_code pin_result =
                pin_thread_to_core(thread_handle, config.core_id.value());
        if (pin_result) {
            result = !result ? pin_result : result; // Preserve first error
        }
    }

    return result;
}

void enable_sanitizer_compatibility() {
#if LEAK_SANITIZER_ENABLED
    // Enable process dumpability to allow both leak sanitizer and real-time
    // scheduling When CAP_SYS_NICE is set, the process becomes non-dumpable by
    // default for security This prevents ptrace attachment needed by sanitizers
    // and debugging tools
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg,hicpp-vararg)
    if (prctl(PR_SET_DUMPABLE, 1) != 0) {
        const std::error_code ec(errno, std::generic_category());
        RT_LOG_WARN("Failed to set process as dumpable: {}", ec.message());
    }
#endif // LEAK_SANITIZER_ENABLED
}

} // namespace framework::task
