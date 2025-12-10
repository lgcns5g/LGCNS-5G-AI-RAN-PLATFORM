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
 * @file rt_log_custom_tests.cpp
 * @brief Unit tests for RT_LOGGER* macros with custom frontend options
 */
#include <cstddef>    // for size_t
#include <cstdint>    // for uint32_t
#include <filesystem> // for exists, path
#include <functional> // for function
#include <string>     // for allocator, string
#include <utility>    // for move

#include <quill/Backend.h>                // for Backend
#include <quill/Frontend.h>               // for FrontendImpl
#include <quill/LogMacros.h>              // for QUILL_LOG_DEBUG, QUILL_LOG...
#include <quill/Logger.h>                 // for LoggerImpl
#include <quill/backend/BackendOptions.h> // for BackendOptions
#include <quill/core/Common.h>            // for HugePagesPolicy, QueueType
#include <quill/core/Filesystem.h>        // for fs
#include <quill/sinks/FileSink.h>         // for FileSink, FileSinkConfig
#include <quill/sinks/StreamSink.h>       // for FileEventNotifier
#include <wise_enum_detail.h>             // for WISE_ENUM_IMPL_IIF_0
#include <wise_enum_generated.h>          // for WISE_ENUM_IMPL_LOOP_6

#include <gtest/gtest.h> // for AssertionResult, Message

#include "log/components.hpp"    // for register_component, DECLAR...
#include "log/rt_log_macros.hpp" // for RT_LOGGERC_DEBUG, RT_LOGGE...
#include "temp_file.hpp"         // for file_contains, TempFileMan...

namespace {

namespace fl = ::framework::log;

// Define custom test components and events for this test file
DECLARE_LOG_COMPONENT(TestComponent, Core, Network, Database, Security, MemoryManager, FileSystem);

DECLARE_LOG_EVENT(
        TestSystemEvent,
        AppStart,
        AppStop,
        ConfigLoaded,
        ConfigError,
        ShutdownRequest,
        HealthCheckOk);

DECLARE_LOG_EVENT(
        TestErrorEvent,
        InvalidParam,
        NetworkError,
        OPERATION_FAILED,
        OPERATION_TIMEOUT,
        CONNECTION_FAILED,
        RESOURCE_UNAVAILABLE);

// Custom frontend options

/**
 * Custom frontend options with UnboundedBlocking queue configuration
 *
 * Provides configuration for high-throughput logging scenarios where
 * blocking behavior is acceptable to prevent message loss.
 */
// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
struct UnboundedBlockingFrontendOptions {
    // NOLINTBEGIN(readability-identifier-naming)
    static constexpr quill::QueueType queue_type = //!< Queue type for message buffering
            quill::QueueType::UnboundedBlocking;
    static constexpr uint32_t initial_queue_capacity =
            64 * 1024; //!< Initial queue capacity in bytes
    static constexpr uint32_t blocking_queue_retry_interval_ns =
            1000; //!< Retry interval for blocking operations
    static constexpr std::size_t unbounded_queue_max_capacity = //!< Maximum queue capacity in bytes
            static_cast<std::size_t>(1024) * 1024;
    static constexpr quill::HugePagesPolicy huge_pages_policy = //!< Memory allocation policy
            quill::HugePagesPolicy::Never;
    // NOLINTEND(readability-identifier-naming)
};
// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

/**
 * Custom frontend implementation using UnboundedBlocking configuration
 */
using CustomFrontend = quill::FrontendImpl<UnboundedBlockingFrontendOptions>;

/**
 * Custom logger implementation using UnboundedBlocking configuration
 */
using CustomLogger = quill::LoggerImpl<UnboundedBlockingFrontendOptions>;

// Custom logger tests

// Test: Verifies all RT_LOGGER* macros work with custom UnboundedBlocking
// logger
TEST(CustomLogger, AllMacrosWithUnboundedBlocking) {
    fl::TempFileManager temp_manager{"custom_all_macros"};
    const std::string log_file = temp_manager.get_temp_file(".log");

    // Start the custom backend for our custom frontend
    const quill::BackendOptions backend_options;
    quill::Backend::start(backend_options);

    // Set component levels to DEBUG for this test
    fl::register_component<TestComponent>(fl::LogLevel::Debug);

    // Create custom logger with file sink
    auto file_sink = CustomFrontend::create_or_get_sink<quill::FileSink>(
            log_file,
            []() {
                quill::FileSinkConfig cfg;
                cfg.set_open_mode('w');
                return cfg;
            }(),
            quill::FileEventNotifier{});
    CustomLogger *custom_logger =
            CustomFrontend::create_or_get_logger("custom_all_logger", std::move(file_sink));

    // Test basic RT_LOGGER macros
    RT_LOGGER_DEBUG(
            custom_logger,
            "Custom debug: {}",
            42); // NOLINT(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
    RT_LOGGER_INFO(custom_logger, "Custom info: {}", "test");
    RT_LOGGER_WARN(custom_logger, "Custom warning");
    RT_LOGGER_ERROR(custom_logger, "Custom error");
    RT_LOGGER_CRITICAL(custom_logger, "Custom critical");

    // Test new log levels
    RT_LOGGER_TRACE_L1(
            custom_logger,
            "Custom trace L1: {}",
            1); // NOLINT(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
    RT_LOGGER_NOTICE(
            custom_logger,
            "Custom notice: {}",
            2); // NOLINT(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

    // Test JSON macros
    const int json_value =
            999; // NOLINT(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
    const std::string json_string = "custom_json";
    RT_LOGGERJ_WARN(custom_logger, "Custom JSON debug", json_value);
    RT_LOGGERJ_INFO(custom_logger, "Custom JSON info", json_string);

    // Test component macros
    RT_LOGGERC_DEBUG(
            custom_logger,
            TestComponent::Core,
            "Custom core: {}",
            123); // NOLINT(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
    RT_LOGGERC_WARN(custom_logger, TestComponent::Network, "Custom network warning");

    // Test event macros
    RT_LOGGERE_INFO(custom_logger, TestSystemEvent::AppStart, "Custom app start: {}", "v1.0");
    RT_LOGGERE_ERROR(custom_logger, TestErrorEvent::NetworkError, "Custom network error");
    RT_LOGGERE_CRITICAL(
            custom_logger, TestErrorEvent::RESOURCE_UNAVAILABLE, "Custom critical resource error");

    // Test combined event+component macros
    RT_LOGGEREC_INFO(
            custom_logger,
            TestComponent::Security,
            TestSystemEvent::ConfigLoaded,
            "Custom security config: {}",
            "loaded");
    RT_LOGGEREC_ERROR(
            custom_logger,
            TestComponent::Database,
            TestErrorEvent::OPERATION_FAILED,
            "Custom database operation failed");

    custom_logger->flush_log();

    quill::Backend::stop();

    EXPECT_TRUE(std::filesystem::exists(log_file));

    EXPECT_FALSE(fl::file_contains(log_file,
                                   "Custom debug: 42")); // log level too low
    EXPECT_TRUE(fl::file_contains(log_file, "Custom info: test"));
    EXPECT_TRUE(fl::file_contains(log_file, "Custom warning"));
    EXPECT_TRUE(fl::file_contains(log_file, "Custom error"));
    EXPECT_TRUE(fl::file_contains(log_file, "Custom critical"));

    EXPECT_FALSE(fl::file_contains(log_file,
                                   "Custom trace L1: 1")); // log level too low
    EXPECT_TRUE(fl::file_contains(log_file, "Custom notice: 2"));

    EXPECT_TRUE(fl::file_contains(log_file, "999"));
    EXPECT_TRUE(fl::file_contains(log_file, "custom_json"));

    EXPECT_FALSE(fl::file_contains(log_file, "[Core]")); // log level too low
    EXPECT_FALSE(fl::file_contains(log_file,
                                   "Custom core: 123")); // log level too low
    EXPECT_TRUE(fl::file_contains(log_file, "[Network]"));
    EXPECT_TRUE(fl::file_contains(log_file, "Custom network warning"));

    EXPECT_TRUE(fl::file_contains(log_file, "EVENT [AppStart]"));
    EXPECT_TRUE(fl::file_contains(log_file, "Custom app start: v1.0"));
    EXPECT_TRUE(fl::file_contains(log_file, "EVENT [NetworkError]"));
    EXPECT_TRUE(fl::file_contains(log_file, "Custom network error"));
    EXPECT_TRUE(fl::file_contains(log_file, "EVENT [RESOURCE_UNAVAILABLE]"));
    EXPECT_TRUE(fl::file_contains(log_file, "Custom critical resource error"));

    EXPECT_TRUE(fl::file_contains(log_file, "[Security] EVENT [ConfigLoaded]"));
    EXPECT_TRUE(fl::file_contains(log_file, "Custom security config: loaded"));
    EXPECT_TRUE(fl::file_contains(log_file, "[Database] EVENT [OPERATION_FAILED]"));
    EXPECT_TRUE(fl::file_contains(log_file, "Custom database operation failed"));
}

} // namespace
