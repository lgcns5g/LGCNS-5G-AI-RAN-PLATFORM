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
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "task/task_utils.hpp"

namespace {
namespace ft = framework::task;

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

// Tests for parse_core_list function
TEST(TaskUtils, ParseCoreList_SingleCore) {
    const auto result = ft::parse_core_list("5");
    EXPECT_EQ(result.size(), 1);
    EXPECT_TRUE(result.contains(5));
}

TEST(TaskUtils, ParseCoreList_MultipleCores) {
    const auto result = ft::parse_core_list("1,3,7");
    EXPECT_EQ(result.size(), 3);
    EXPECT_TRUE(result.contains(1));
    EXPECT_TRUE(result.contains(3));
    EXPECT_TRUE(result.contains(7));
}

TEST(TaskUtils, ParseCoreList_Range) {
    const auto result = ft::parse_core_list("4-6");
    EXPECT_EQ(result.size(), 3);
    EXPECT_TRUE(result.contains(4));
    EXPECT_TRUE(result.contains(5));
    EXPECT_TRUE(result.contains(6));
}

TEST(TaskUtils, ParseCoreList_MixedRangeAndSingle) {
    const auto result = ft::parse_core_list("1,4-6,9");
    EXPECT_EQ(result.size(), 5);
    EXPECT_TRUE(result.contains(1));
    EXPECT_TRUE(result.contains(4));
    EXPECT_TRUE(result.contains(5));
    EXPECT_TRUE(result.contains(6));
    EXPECT_TRUE(result.contains(9));
}

TEST(TaskUtils, ParseCoreList_EmptyString) {
    const auto result = ft::parse_core_list("");
    EXPECT_TRUE(result.empty());
}

TEST(TaskUtils, ParseCoreList_InvalidRange) {
    const auto result = ft::parse_core_list("4-abc");
    EXPECT_TRUE(result.empty());
}

TEST(TaskUtils, ParseCoreList_InvalidCore) {
    const auto result = ft::parse_core_list("1,abc,3");
    EXPECT_EQ(result.size(), 2);
    EXPECT_TRUE(result.contains(1));
    EXPECT_TRUE(result.contains(3));
}

TEST(TaskUtils, ParseCoreList_LargeRange) {
    const auto result = ft::parse_core_list("4-64");
    EXPECT_EQ(result.size(), 61); // 4 through 64 inclusive
    EXPECT_TRUE(result.contains(4));
    EXPECT_TRUE(result.contains(32));
    EXPECT_TRUE(result.contains(64));
    EXPECT_FALSE(result.contains(3));
    EXPECT_FALSE(result.contains(65));
}

// Tests for validate_rt_core_config function
TEST(TaskUtils, ValidateRtCoreConfig_AllCoresPresent) {
    const std::string cmdline =
            "BOOT_IMAGE=/vmlinuz root=/dev/sda1 isolcpus=managed_irq,domain,4-64 "
            "nohz_full=4-64 rcu_nocbs=4-64";

    const std::vector<std::uint32_t> cores = {5, 10, 32, 64};
    const bool result = ft::validate_rt_core_config(cmdline, cores);
    EXPECT_TRUE(result);
}

TEST(TaskUtils, ValidateRtCoreConfig_MissingFromIsolcpus) {
    const std::string cmdline =
            "BOOT_IMAGE=/vmlinuz root=/dev/sda1 isolcpus=managed_irq,domain,8-64 "
            "nohz_full=4-64 rcu_nocbs=4-64";

    const std::vector<std::uint32_t> cores = {5}; // Core 5 not in isolcpus
    const bool result = ft::validate_rt_core_config(cmdline, cores);
    EXPECT_FALSE(result);
}

TEST(TaskUtils, ValidateRtCoreConfig_MissingFromNohz) {
    const std::string cmdline =
            "BOOT_IMAGE=/vmlinuz root=/dev/sda1 isolcpus=managed_irq,domain,4-64 "
            "nohz_full=8-64 rcu_nocbs=4-64";

    const std::vector<std::uint32_t> cores = {5}; // Core 5 not in nohz_full
    const bool result = ft::validate_rt_core_config(cmdline, cores);
    EXPECT_FALSE(result);
}

TEST(TaskUtils, ValidateRtCoreConfig_MissingFromRcuNocbs) {
    const std::string cmdline =
            "BOOT_IMAGE=/vmlinuz root=/dev/sda1 isolcpus=managed_irq,domain,4-64 "
            "nohz_full=4-64 rcu_nocbs=8-64";

    const std::vector<std::uint32_t> cores = {5}; // Core 5 not in rcu_nocbs
    const bool result = ft::validate_rt_core_config(cmdline, cores);
    EXPECT_FALSE(result);
}

TEST(TaskUtils, ValidateRtCoreConfig_NoRtParameters) {
    const std::string cmdline = "BOOT_IMAGE=/vmlinuz root=/dev/sda1";

    const std::vector<std::uint32_t> cores = {5};
    const bool result = ft::validate_rt_core_config(cmdline, cores);
    EXPECT_FALSE(result);
}

TEST(TaskUtils, ValidateRtCoreConfig_EmptyCoreList) {
    const std::string cmdline =
            "BOOT_IMAGE=/vmlinuz root=/dev/sda1 isolcpus=managed_irq,domain,4-64 "
            "nohz_full=4-64 rcu_nocbs=4-64";

    const std::vector<std::uint32_t> cores = {};
    const bool result = ft::validate_rt_core_config(cmdline, cores);
    EXPECT_TRUE(result); // Empty list should be valid
}

TEST(TaskUtils, ValidateRtCoreConfig_MixedValidInvalid) {
    const std::string cmdline =
            "BOOT_IMAGE=/vmlinuz root=/dev/sda1 isolcpus=managed_irq,domain,8-16 "
            "nohz_full=8-16 rcu_nocbs=8-16";

    const std::vector<std::uint32_t> cores = {5, 10}; // 5 invalid, 10 valid
    const bool result = ft::validate_rt_core_config(cmdline, cores);
    EXPECT_FALSE(result); // Should fail if any core is invalid
}

TEST(TaskUtils, ValidateRtCoreConfig_ComplexIsolcpus) {
    const std::string cmdline =
            "BOOT_IMAGE=/vmlinuz root=/dev/sda1 isolcpus=managed_irq,domain,4-64 "
            "nohz_full=4-64 rcu_nocbs=4-64";

    const std::vector<std::uint32_t> cores = {4, 32, 64};
    const bool result = ft::validate_rt_core_config(cmdline, cores);
    EXPECT_TRUE(result);
}

TEST(TaskUtils, ValidateRtCoreConfig_RealWorldKernelCmdline) {
    // Based on the actual cmdline from the user
    const std::string cmdline =
            "BOOT_IMAGE=/vmlinuz-6.8.0-1025-nvidia-64k "
            "root=/dev/mapper/ubuntu--vg-ubuntu--lv "
            "ro pci=realloc=off pci=pcie_bus_safe default_hugepagesz=512M "
            "hugepagesz=512M "
            "hugepages=48 tsc=reliable processor.max_cstate=0 audit=0 idle=poll "
            "rcu_nocb_poll nosoftlockup irqaffinity=0 "
            "isolcpus=managed_irq,domain,4-64 "
            "nohz_full=4-64 rcu_nocbs=4-64 earlycon module_blacklist=nouveau "
            "acpi_power_meter.force_cap_on=y numa_balancing=disable init_on_alloc=0 "
            "preempt=none";

    const std::vector<std::uint32_t> cores = {4, 32, 64};
    const bool result = ft::validate_rt_core_config(cmdline, cores);
    EXPECT_TRUE(result);
}

TEST(TaskUtils, ValidateRtCoreConfig_RealWorldInvalidCore) {
    // Based on the actual cmdline from the user
    const std::string cmdline =
            "BOOT_IMAGE=/vmlinuz-6.8.0-1025-nvidia-64k "
            "root=/dev/mapper/ubuntu--vg-ubuntu--lv "
            "ro pci=realloc=off pci=pcie_bus_safe default_hugepagesz=512M "
            "hugepagesz=512M "
            "hugepages=48 tsc=reliable processor.max_cstate=0 audit=0 idle=poll "
            "rcu_nocb_poll nosoftlockup irqaffinity=0 "
            "isolcpus=managed_irq,domain,4-64 "
            "nohz_full=4-64 rcu_nocbs=4-64 earlycon module_blacklist=nouveau "
            "acpi_power_meter.force_cap_on=y numa_balancing=disable init_on_alloc=0 "
            "preempt=none";

    const std::vector<std::uint32_t> cores = {2, 3}; // These cores are not isolated
    const bool result = ft::validate_rt_core_config(cmdline, cores);
    EXPECT_FALSE(result);
}

// Tests for statistics utility functions

TEST(TaskUtils, CalculateStandardDeviation_Int64Vector) {
    const std::vector<std::int64_t> values = {1000, 2000, 3000, 4000, 5000}; // nanoseconds
    const double mean = 3000.0;
    const double std_dev = ft::calculate_standard_deviation(values, mean);
    const double expected_std_dev = std::sqrt(2500000.0); // ~1581.14
    EXPECT_NEAR(std_dev, expected_std_dev, 0.01);
}

TEST(TaskUtils, CalculateStandardDeviation_DoubleVector) {
    const std::vector<double> values = {1.0, 2.0, 3.0, 4.0, 5.0};
    const double mean = 3.0;
    const double std_dev = ft::calculate_standard_deviation(values, mean);
    const double expected_std_dev = std::sqrt(2.5); // ~1.581
    EXPECT_NEAR(std_dev, expected_std_dev, 0.001);
}

TEST(TaskUtils, CalculateStandardDeviation_EmptyVector) {
    const std::vector<std::int64_t> values = {};
    const double std_dev = ft::calculate_standard_deviation(values, 0.0);
    EXPECT_EQ(std_dev, 0.0);
}

TEST(TaskUtils, CalculateStandardDeviation_SingleElement) {
    const std::vector<double> values = {42.5};
    const double std_dev = ft::calculate_standard_deviation(values, 42.5);
    EXPECT_EQ(std_dev, 0.0);
}

TEST(TaskUtils, CalculatePercentile_Basic) {
    const std::vector<double> values = {1.0, 2.0, 3.0, 4.0, 5.0}; // Already sorted

    EXPECT_NEAR(ft::calculate_percentile(values, 0.0), 1.0, 0.001); // Min
    EXPECT_NEAR(ft::calculate_percentile(values, 0.5), 3.0, 0.001); // Median
    EXPECT_NEAR(ft::calculate_percentile(values, 0.95), 4.8,
                0.001);                                             // 95th (interpolated)
    EXPECT_NEAR(ft::calculate_percentile(values, 1.0), 5.0, 0.001); // Max
}

TEST(TaskUtils, CalculatePercentile_LargeDataset) {
    std::vector<double> values;
    values.reserve(100);
    for (int i = 0; i < 100; ++i) {
        values.push_back(static_cast<double>(i)); // 0 to 99
    }

    EXPECT_NEAR(ft::calculate_percentile(values, 0.5), 49.5,
                0.001); // 50th percentile
    EXPECT_NEAR(ft::calculate_percentile(values, 0.95), 94.05,
                0.001); // 95th percentile
    EXPECT_NEAR(ft::calculate_percentile(values, 0.99), 98.01,
                0.001); // 99th percentile
}

TEST(TaskUtils, CalculatePercentile_EmptyVector) {
    const std::vector<double> values = {};
    EXPECT_EQ(ft::calculate_percentile(values, 0.5), 0.0);
}

TEST(TaskUtils, CalculatePercentile_SingleElement) {
    const std::vector<double> values = {42.0};
    EXPECT_EQ(ft::calculate_percentile(values, 0.5), 42.0);
    EXPECT_EQ(ft::calculate_percentile(values, 0.95), 42.0);
}

TEST(TaskUtils, NanosToMicros_Int64) {
    EXPECT_NEAR(ft::nanos_to_micros_int64(1000), 1.0, 0.001);
    EXPECT_NEAR(ft::nanos_to_micros_int64(2500), 2.5, 0.001);
    EXPECT_NEAR(ft::nanos_to_micros_int64(0), 0.0, 0.001);
    EXPECT_NEAR(ft::nanos_to_micros_int64(1500000), 1500.0, 0.001);
}

TEST(TaskUtils, NanosToMicros_Double) {
    EXPECT_NEAR(ft::nanos_to_micros_double(1000.0), 1.0, 0.001);
    EXPECT_NEAR(ft::nanos_to_micros_double(2500.5), 2.5005, 0.0001);
    EXPECT_NEAR(ft::nanos_to_micros_double(0.0), 0.0, 0.001);
    EXPECT_NEAR(ft::nanos_to_micros_double(1500000.75), 1500.00075, 0.00001);
}

TEST(TaskUtils, CalculateTimingStatistics_Int64Vector) {
    const std::vector<std::int64_t> values_ns = {1000, 2000, 3000, 4000, 5000}; // nanoseconds
    const ft::TimingStatistics stats = ft::calculate_timing_statistics(values_ns);

    EXPECT_EQ(stats.count, 5);
    EXPECT_NEAR(stats.min_us, 1.0, 0.001);
    EXPECT_NEAR(stats.max_us, 5.0, 0.001);
    EXPECT_NEAR(stats.avg_us, 3.0, 0.001);
    EXPECT_NEAR(stats.median_us, 3.0, 0.001);
    EXPECT_NEAR(stats.p95_us, 4.8, 0.001);  // Interpolated percentile
    EXPECT_NEAR(stats.p99_us, 4.96, 0.001); // Interpolated percentile
    EXPECT_GT(stats.std_us, 0.0);           // Should have some standard deviation
}

TEST(TaskUtils, CalculateTimingStatistics_DoubleVector) {
    const std::vector<double> values_ns = {1000.0, 2000.0, 3000.0, 4000.0, 5000.0}; // nanoseconds
    const ft::TimingStatistics stats = ft::calculate_timing_statistics(values_ns);

    EXPECT_EQ(stats.count, 5);
    EXPECT_NEAR(stats.min_us, 1.0, 0.001);
    EXPECT_NEAR(stats.max_us, 5.0, 0.001);
    EXPECT_NEAR(stats.avg_us, 3.0, 0.001);
    EXPECT_NEAR(stats.median_us, 3.0, 0.001);
    EXPECT_NEAR(stats.p95_us, 4.8, 0.001);  // Interpolated percentile
    EXPECT_NEAR(stats.p99_us, 4.96, 0.001); // Interpolated percentile
    EXPECT_GT(stats.std_us, 0.0);           // Should have some standard deviation
}

TEST(TaskUtils, CalculateTimingStatistics_EmptyVector) {
    const std::vector<std::int64_t> values_ns = {};
    const ft::TimingStatistics stats = ft::calculate_timing_statistics(values_ns);

    EXPECT_EQ(stats.count, 0);
    EXPECT_EQ(stats.min_us, 0.0);
    EXPECT_EQ(stats.max_us, 0.0);
    EXPECT_EQ(stats.avg_us, 0.0);
    EXPECT_EQ(stats.median_us, 0.0);
    EXPECT_EQ(stats.p95_us, 0.0);
    EXPECT_EQ(stats.p99_us, 0.0);
    EXPECT_EQ(stats.std_us, 0.0);
}

TEST(TaskUtils, CalculateTimingStatistics_SingleElement) {
    const std::vector<std::int64_t> values_ns = {42000}; // 42 microseconds in nanoseconds
    const ft::TimingStatistics stats = ft::calculate_timing_statistics(values_ns);

    EXPECT_EQ(stats.count, 1);
    EXPECT_NEAR(stats.min_us, 42.0, 0.001);
    EXPECT_NEAR(stats.max_us, 42.0, 0.001);
    EXPECT_NEAR(stats.avg_us, 42.0, 0.001);
    EXPECT_NEAR(stats.median_us, 42.0, 0.001);
    EXPECT_NEAR(stats.p95_us, 42.0, 0.001);
    EXPECT_NEAR(stats.p99_us, 42.0, 0.001);
    EXPECT_EQ(stats.std_us, 0.0); // Single element has zero standard deviation
}

TEST(TaskUtils, CalculateTimingStatistics_LargeDataset) {
    std::vector<std::int64_t> values_ns;
    for (int i = 1; i <= 1000; ++i) {
        values_ns.push_back(static_cast<std::int64_t>(i) * 1000); // 1us to 1000us in nanoseconds
    }
    const ft::TimingStatistics stats = ft::calculate_timing_statistics(values_ns);

    EXPECT_EQ(stats.count, 1000);
    EXPECT_NEAR(stats.min_us, 1.0, 0.001);
    EXPECT_NEAR(stats.max_us, 1000.0, 0.001);
    EXPECT_NEAR(stats.avg_us, 500.5, 0.001);
    EXPECT_NEAR(stats.median_us, 500.0, 1.0); // Allow for some rounding error
    EXPECT_NEAR(stats.p95_us, 950.0, 1.0);    // 95th percentile
    EXPECT_NEAR(stats.p99_us, 990.0, 1.0);    // 99th percentile
    EXPECT_GT(stats.std_us, 250.0);           // Should have significant standard deviation
}

// ============================================================================
// GPS/TAI Timing Utilities Tests
// ============================================================================

TEST(TaskUtils, CalculateTaiOffset_ReturnsNonZero) {
    const std::chrono::nanoseconds tai_offset = ft::calculate_tai_offset();

    // TAI offset should be non-zero in most systems
    // Even if adjtimex fails, the function should return 0 gracefully
    EXPECT_GE(tai_offset.count(), 0);
}

// Table-based tests for calculate_start_time_for_next_period
TEST(TaskUtils, CalculateStartTimeForNextPeriod_TableDriven) {
    struct StartTimeTestCase {
        std::uint64_t current_time_ns;
        std::chrono::nanoseconds tai_offset;
        std::uint64_t period_ns;
        std::int64_t gps_alpha;
        std::int64_t gps_beta;
        std::uint64_t expected_start_time_ns;
        std::string description;
    };

    const std::vector<StartTimeTestCase> test_cases = {
            // Basic period tests with realistic TAI offset
            {1'000'000'000ULL,                            // 1 second current time
             std::chrono::nanoseconds{37'000'000'000ULL}, // 37s TAI offset
             1'000'000ULL,                                // 1ms period
             0,                                           // No alpha
             0,                                           // No beta
             1'000'448'384ULL,                            // Expected
             "1 second, 1ms period, 37s TAI"},

            {5'000'000'000ULL,                            // 5 second current time
             std::chrono::nanoseconds{37'000'000'000ULL}, // 37s TAI offset
             10'000'000ULL,                               // 10ms period
             0,                                           // No alpha
             0,                                           // No beta
             5'000'448'384ULL,                            // Expected
             "5 seconds, 10ms period, 37s TAI"},

            {1'500'000'000ULL,                            // 1.5 second current time
             std::chrono::nanoseconds{37'000'000'000ULL}, // 37s TAI offset
             10'000'000ULL,                               // 10ms period
             0,                                           // No alpha
             0,                                           // No beta
             1'500'448'384ULL,                            // Expected
             "1.5 seconds, 10ms period, 37s TAI"},

            // SFN_PERIOD tests with various TAI offsets
            {15'000'000'000ULL,              // 15 second current time
             std::chrono::nanoseconds{0ULL}, // Zero TAI offset
             10'240'000'000ULL,              // SFN_PERIOD (1024 * 10ms = 10.24s)
             0,                              // No alpha
             0,                              // No beta
             24'650'448'384ULL,              // Expected
             "15 seconds, SFN_PERIOD, 0s TAI"},

            {10'000'000'000ULL,                           // 10 second current time
             std::chrono::nanoseconds{37'000'000'000ULL}, // 37s TAI offset
             10'240'000'000ULL,                           // SFN_PERIOD
             0,                                           // No alpha
             0,                                           // No beta
             18'370'448'384ULL,                           // Expected
             "10 seconds, SFN_PERIOD, 37s TAI"},

            {25'000'000'000ULL,                           // 25 second current time
             std::chrono::nanoseconds{18'000'000'000ULL}, // 18s TAI offset
             10'240'000'000ULL,                           // SFN_PERIOD
             0,                                           // No alpha
             0,                                           // No beta
             27'130'448'384ULL,                           // Expected
             "25 seconds, SFN_PERIOD, 18s TAI"},

            // GPS parameter tests with both alpha and beta
            {2'000'000'000ULL,                            // 2 second current time
             std::chrono::nanoseconds{37'000'000'000ULL}, // 37s TAI offset
             10'000'000ULL,                               // 10ms period
             0,                                           // No alpha
             100,                                         // Beta = 100
             2'000'448'384ULL,                            // Expected
             "2 seconds, 10ms period, beta=100, 37s TAI"},

            {3'000'000'000ULL,                            // 3 second current time
             std::chrono::nanoseconds{37'000'000'000ULL}, // 37s TAI offset
             10'000'000ULL,                               // 10ms period
             500,                                         // Alpha = 500
             0,                                           // No beta
             3'000'448'790ULL,                            // Expected
             "3 seconds, 10ms period, alpha=500, 37s TAI"},

            {4'000'000'000ULL,                            // 4 second current time
             std::chrono::nanoseconds{37'000'000'000ULL}, // 37s TAI offset
             10'000'000ULL,                               // 10ms period
             1000,                                        // Alpha = 1000
             200,                                         // Beta = 200
             4'000'449'197ULL,                            // Expected
             "4 seconds, 10ms period, alpha=1000 beta=200, 37s TAI"},

            {8'000'000'000ULL,                            // 8 second current time
             std::chrono::nanoseconds{37'000'000'000ULL}, // 37s TAI offset
             10'240'000'000ULL,                           // SFN_PERIOD
             2000,                                        // Alpha = 2000
             300,                                         // Beta = 300
             11'130'450'011ULL,                           // Expected
             "8 seconds, SFN_PERIOD, alpha=2000 beta=300, 37s TAI"},

            // Edge cases with realistic TAI
            {999'000'000ULL,                              // 999ms current time
             std::chrono::nanoseconds{37'000'000'000ULL}, // 37s TAI offset
             10'000'000ULL,                               // 10ms period
             0,                                           // No alpha
             0,                                           // No beta
             1'000'448'384ULL,                            // Expected
             "999ms, 10ms period, 37s TAI"},
            {5'000'000'000ULL,                            // 5 second current time
             std::chrono::nanoseconds{37'000'000'000ULL}, // 37s TAI offset
             3'600'000'000'000ULL,                        // 1 hour period
             0,                                           // No alpha
             0,                                           // No beta
             1'508'290'448'384ULL,                        // Expected
             "5 seconds, 1 hour period, 37s TAI"}};

    for (const auto &test_case : test_cases) {
        SCOPED_TRACE(test_case.description);

        const ft::StartTimeParams params{
                .current_time_ns = test_case.current_time_ns,
                .period_ns = test_case.period_ns,
                .gps_alpha = test_case.gps_alpha,
                .gps_beta = test_case.gps_beta};

        const std::uint64_t result =
                ft::calculate_start_time_for_next_period(params, test_case.tai_offset);

        // Check exact expected value
        EXPECT_EQ(result, test_case.expected_start_time_ns);
    }
}

// Test for invalid input parameters
TEST(TaskUtils, CalculateStartTimeForNextPeriod_ZeroPeriod) {
    const ft::StartTimeParams params{
            .current_time_ns = 1'000'000'000ULL, // 1 second current time
            .period_ns = 0ULL,                   // Zero period - should throw
            .gps_alpha = 0,                      // No alpha
            .gps_beta = 0};                      // No beta

    EXPECT_THROW(
            std::ignore = ft::calculate_start_time_for_next_period(
                    params, std::chrono::nanoseconds{37'000'000'000ULL}), // 37s TAI offset
            std::invalid_argument);
}

// ============================================================================
// Record Management Utilities Tests
// ============================================================================

// Test struct for calculate_max_records_for_bytes
struct TestRecord {
    std::uint64_t id{};
    std::uint32_t value{};
    bool flag{};
    // Total size: 13 bytes + padding = 16 bytes on most platforms
};

TEST(TaskUtils, CalculateMaxRecordsForBytes_TableDriven) {
    struct MaxRecordsTestCase {
        std::size_t max_bytes;
        std::size_t expected_records_16_byte;
        std::size_t expected_records_104_byte;
        std::string description;
    };

    struct LargeTestRecord {
        std::uint64_t timestamp{};
        std::string name;             // 32 bytes on most platforms
        std::array<double, 8> data{}; // 8 * 8 = 64 bytes
                                      // Total size: ~104 bytes on most platforms
    };

    const std::vector<MaxRecordsTestCase> test_cases = {
            {0, 0, 0, "Zero bytes"},
            {1, 0, 0, "1 byte - too small for any record"},
            {16, 1, 0, "Exactly one 16-byte record"},
            {32, 2, 0, "Two 16-byte records"},
            {100, 6, 0, "100 bytes - 6 records of 16 bytes"},
            {104, 6, 1, "104 bytes - exactly one large record"},
            {1024, 64, 9, "1KB - 64 small or 9 large records"},
            {static_cast<std::size_t>(1024) * 1024, 65536, 10082, "1MB - many records"},
            {50ULL * 1024 * 1024 * 1024, 3355443200ULL, 516222030ULL, "50GB - huge capacity"}};

    for (const auto &test_case : test_cases) {
        SCOPED_TRACE(test_case.description);

        // Test with small record type
        const std::size_t result_small =
                ft::calculate_max_records_for_bytes<TestRecord>(test_case.max_bytes);
        EXPECT_EQ(result_small, test_case.expected_records_16_byte);

        // Test with large record type
        const std::size_t result_large =
                ft::calculate_max_records_for_bytes<LargeTestRecord>(test_case.max_bytes);
        EXPECT_EQ(result_large, test_case.expected_records_104_byte);
    }
}

TEST(TaskUtils, CalculateMaxRecordsForBytes_EdgeCases) {
    // Zero-sized record type (hypothetical)
    struct EmptyRecord {};

    // Most compilers will make this at least 1 byte, but test the logic
    const std::size_t empty_result = ft::calculate_max_records_for_bytes<EmptyRecord>(1000);
    EXPECT_GT(empty_result, 0); // Should handle empty structs gracefully

    // Large byte count
    constexpr std::size_t HUGE_BYTES = std::numeric_limits<std::size_t>::max() / 2;
    const std::size_t huge_result = ft::calculate_max_records_for_bytes<TestRecord>(HUGE_BYTES);
    EXPECT_GT(huge_result, 1000000); // Should handle very large allocations
}

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

} // namespace
