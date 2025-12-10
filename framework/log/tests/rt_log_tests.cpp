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
 * @file rt_log_tests.cpp
 * @brief Unit tests for the rt_log library using static singleton API
 */

#include <array>         // for array
#include <chrono>        // for nanoseconds, milliseconds
#include <cstdint>       // for uint64_t, uint16_t
#include <filesystem>    // for exists, path
#include <span>          // for span
#include <stdexcept>     // for invalid_argument
#include <string>        // for allocator, string, to_string
#include <unordered_map> // for unordered_map
#include <utility>       // for pair, move
#include <vector>        // for vector

#include <quill/LogMacros.h>     // for QUILL_LOG_INFO, QUILL_LOG_DEBUG
#include <wise_enum_detail.h>    // for WISE_ENUM_IMPL_IIF_0
#include <wise_enum_generated.h> // for WISE_ENUM_IMPL_LOOP_12, WISE_ENUM_I...

#include <gtest/gtest.h> // for AssertionResult, Message, TestPartR...

#include "log/components.hpp"    // for LogLevel, get_component_level, regi...
#include "log/rt_log.hpp"        // for Logger, LoggerConfig, SinkType
#include "log/rt_log_macros.hpp" // for RT_LOG_INFO, RT_LOGE_ERROR, RT_LOGC...
#include "temp_file.hpp"         // for file_contains, TempFileManager

namespace {

namespace fl = ::framework::log;

DECLARE_LOG_COMPONENT(
        SystemComponent,
        Core,
        Config,
        Network,
        Database,
        Security,
        Performance,
        ThreadPool,
        MemoryManager,
        FileSystem,
        Ipc,
        Scheduler,
        Monitor);

DECLARE_LOG_EVENT(
        SystemEvent,
        AppStart,
        AppStop,
        ConfigLoaded,
        ConfigError,
        ShutdownRequest,
        SHUTDOWN_COMPLETE,
        HealthCheckOk,
        HEALTH_CHECK_FAIL,
        THREAD_START,
        THREAD_STOP,
        RESOURCE_ALLOC,
        RESOURCE_FREE,
        TIMEOUT,
        Retry,
        CriticalError);

DECLARE_LOG_EVENT(
        ErrorEvent,
        InvalidParam,
        OutOfMemory,
        FILE_NOT_FOUND,
        PERMISSION_DENIED,
        NetworkError,
        OPERATION_FAILED,
        OPERATION_TIMEOUT,
        InvalidState,
        BUFFER_OVERFLOW,
        RESOURCE_EXHAUSTED,
        AuthenticationFailed,
        CONFIGURATION_ERROR,
        PROTOCOL_ERROR,
        SERIALIZATION_ERROR,
        DESERIALIZATION_ERROR,
        CONNECTION_FAILED,
        RESOURCE_UNAVAILABLE);

// Test: Verifies basic console logging functionality with default logger
TEST(RTLog, BasicConsoleLogging) {
    // Use default console logger (no configuration needed)
    RT_LOG_DEBUG("Debug message: {}", 42);
    RT_LOG_INFO("Info message: {}", "test");
    RT_LOG_WARN("Warning message");
    RT_LOG_ERROR("Error message");
}

// Test: Verifies file logging works correctly when logger is configured for
// file output
TEST(RTLog, ConfiguredFileLogging) {
    fl::TempFileManager temp_manager{"configured_file"};
    const std::string log_file = temp_manager.get_temp_file(".log");

    fl::Logger::configure(fl::LoggerConfig::file(log_file.c_str(), fl::LogLevel::Debug));

    const std::string actual_log_file = fl::Logger::get_actual_log_file();

    // Use RT_LOG macros (they automatically use the configured logger)
    RT_LOG_DEBUG("Debug file message: {}", 123);
    RT_LOG_INFO("Info file message: {}", "testing");
    RT_LOG_WARN("Warning file message");
    RT_LOG_ERROR("Error file message");

    fl::Logger::flush();

    // Verify file contents
    EXPECT_TRUE(std::filesystem::exists(actual_log_file));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Debug file message: 123"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Info file message: testing"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Warning file message"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Error file message"));
}

// Test: Verifies rotating file logging with file/line information enabled
TEST(RTLog, RotatingFileLogging) {
    fl::TempFileManager temp_manager{"rotating_file"};
    const std::string log_file = temp_manager.get_temp_file(".log");

    fl::Logger::configure(fl::LoggerConfig::rotating_file(log_file.c_str(), fl::LogLevel::Debug)
                                  .with_file_line(true));

    const std::string actual_log_file = fl::Logger::get_actual_log_file();

    // Use RT_LOG macros
    RT_LOG_DEBUG("Rotating debug message: {}", 789);
    RT_LOG_INFO("Rotating info message: {}", "rotation");
    RT_LOG_WARN("Rotating warning message");
    RT_LOG_ERROR("Rotating error message");

    fl::Logger::flush();

    // Verify file contents
    EXPECT_TRUE(std::filesystem::exists(actual_log_file));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Rotating debug message: 789"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Rotating info message: rotation"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Rotating warning message"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Rotating error message"));
}

// Test: Verifies JSON logging format with caller information enabled
TEST(RTLog, JSONLogging) {
    fl::TempFileManager temp_manager{"json_logging"};
    const std::string log_file = temp_manager.get_temp_file(".json");

    fl::Logger::configure(
            fl::LoggerConfig::json_file(log_file.c_str(), fl::LogLevel::Debug).with_caller(true));

    const std::string actual_log_file = fl::Logger::get_actual_log_file();

    // CORRECT: Use LOGJ macros with variable names (no {} placeholders)
    const int debug_value =
            999; // NOLINT(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
    const std::string info_value = "json";
    const std::string warn_msg = "JSON warning message";
    const std::string error_msg = "JSON error message";

    RT_LOGJ_DEBUG("JSON debug message", debug_value);
    RT_LOGJ_INFO("JSON info message", info_value);
    RT_LOGJ_WARN("JSON warning message", warn_msg);
    RT_LOGJ_ERROR("JSON error message", error_msg);

    fl::Logger::flush();

    // Verify file contents (check for the actual variable names and values)
    EXPECT_TRUE(std::filesystem::exists(actual_log_file));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "debug_value"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "999"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "info_value"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "json"));
}

// Test: Verifies log level filtering works correctly (only logs at or above
// configured level)
TEST(RTLog, LogLevelFiltering) {
    fl::TempFileManager temp_manager{"level_filtering"};
    const std::string log_file = temp_manager.get_temp_file(".log");

    // Configure logger with WARNING level using factory method
    fl::Logger::configure(fl::LoggerConfig::file(log_file.c_str(), fl::LogLevel::Warn));

    const std::string actual_log_file = fl::Logger::get_actual_log_file();

    // Log at different levels
    RT_LOG_DEBUG("This debug should NOT appear");
    RT_LOG_INFO("This info should NOT appear");
    RT_LOG_WARN("This warning should appear");
    RT_LOG_ERROR("This error should appear");

    fl::Logger::flush();

    // Verify only WARNING and ERROR appear
    EXPECT_TRUE(std::filesystem::exists(actual_log_file));
    EXPECT_FALSE(fl::file_contains(actual_log_file, "This debug should NOT appear"));
    EXPECT_FALSE(fl::file_contains(actual_log_file, "This info should NOT appear"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "This warning should appear"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "This error should appear"));
}

// Test: Verifies component-based logging functionality
TEST(RTLog, ComponentLogging) {
    fl::TempFileManager temp_manager{"component_logging"};
    const std::string log_file = temp_manager.get_temp_file(".log");

    fl::Logger::configure(fl::LoggerConfig::file(log_file.c_str(), fl::LogLevel::Debug));
    fl::register_component<SystemComponent>(fl::LogLevel::Debug);

    const std::string actual_log_file = fl::Logger::get_actual_log_file();

    // Use component logging macros
    RT_LOGC_DEBUG(SystemComponent::Core, "Core debug message: {}", 42);
    RT_LOGC_INFO(SystemComponent::Network, "Network info message: {}", "test");
    RT_LOGC_WARN(SystemComponent::Database, "Database warning message");
    RT_LOGC_ERROR(SystemComponent::Security, "Security error message");

    fl::Logger::flush();

    // Verify file contents include component names
    EXPECT_TRUE(std::filesystem::exists(actual_log_file));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "[Core]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Core debug message: 42"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "[Network]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Network info message: test"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "[Database]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "[Security]"));
}

// Test: Verifies event-based logging functionality
TEST(RTLog, EventLogging) {
    fl::TempFileManager temp_manager{"event_logging"};
    const std::string log_file = temp_manager.get_temp_file(".log");

    fl::Logger::configure(fl::LoggerConfig::file(log_file.c_str(), fl::LogLevel::Debug));

    const std::string actual_log_file = fl::Logger::get_actual_log_file();

    // Use event logging macros
    RT_LOGE_DEBUG(SystemEvent::AppStart, "Application starting: {}", "v1.0");
    RT_LOGE_INFO(SystemEvent::ConfigLoaded, "Configuration loaded: {}", "success");
    RT_LOGE_WARN(SystemEvent::TIMEOUT, "Operation timeout warning");
    RT_LOGE_ERROR(SystemEvent::CriticalError, "Critical system error");

    fl::Logger::flush();

    // Verify file contents include event names
    EXPECT_TRUE(std::filesystem::exists(actual_log_file));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "EVENT [AppStart]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Application starting: v1.0"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "EVENT [ConfigLoaded]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "EVENT [TIMEOUT]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "EVENT [CriticalError]"));
}

// Test: Shows off the flexible configuration with designated initializers
TEST(RTLog, CustomConfiguration) {
    fl::TempFileManager temp_manager{"custom_config"};
    const std::string log_file = temp_manager.get_temp_file(".log");

    // C++20 designated initializers - very clean!
    fl::Logger::configure(
            {.sink_type = fl::SinkType::File,
             .log_file = log_file.c_str(),
             .min_level = fl::LogLevel::Debug,
             .enable_colors = false,
             .enable_file_line = true,
             .enable_caller = true,
             .backend_cpu_affinity = 2});

    const std::string actual_log_file = fl::Logger::get_actual_log_file();

    RT_LOG_DEBUG("Custom config message: {}", 42);
    fl::Logger::flush();

    EXPECT_TRUE(std::filesystem::exists(actual_log_file));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Custom config message: 42"));
}

// Test: Verifies console configuration with custom settings
TEST(RTLog, ConsoleConfiguration) {
    // Configure console with DEBUG level and no colors
    fl::Logger::configure(fl::LoggerConfig::console(fl::LogLevel::Debug, false));

    // These should all appear in console output (no file to check)
    RT_LOG_DEBUG("Console debug: {}", 1);
    RT_LOG_INFO("Console info: {}", 2);
    RT_LOG_WARN("Console warn: {}", 3);
    RT_LOG_ERROR("Console error: {}", 4);

    fl::Logger::flush();

    // Verify logger is configured correctly
    EXPECT_EQ(fl::Logger::get_sink_type(), fl::SinkType::Console);
    EXPECT_EQ(fl::Logger::get_current_level(), fl::LogLevel::Debug);
}

// Test: Verifies chaining API for configuration
TEST(RTLog, ConfigChaining) {
    fl::TempFileManager temp_manager{"config_chaining"};
    const std::string log_file = temp_manager.get_temp_file(".log");

    // Show off the chaining API
    fl::Logger::configure(fl::LoggerConfig::file(log_file.c_str())
                                  .with_file_line(true)
                                  .with_caller(true)
                                  .with_cpu_affinity(1));

    const std::string actual_log_file = fl::Logger::get_actual_log_file();

    RT_LOG_INFO("Chained config test: {}", "success");
    fl::Logger::flush();

    EXPECT_TRUE(std::filesystem::exists(actual_log_file));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Chained config test: success"));
}

// Additional sink type tests

// Test: Verifies JSON console logging functionality
TEST(RTLog, JsonConsoleLogging) {
    // Configure JSON console logger
    fl::Logger::configure(fl::LoggerConfig::json_console(fl::LogLevel::Debug));

    // These should appear as JSON in console output
    RT_LOGJ_DEBUG("JSON console debug", 42);
    RT_LOGJ_INFO("JSON console info", "test");
    RT_LOGJ_WARN("JSON console warn", true);
    RT_LOGJ_ERROR(
            "JSON console error",
            3.14); // NOLINT(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

    fl::Logger::flush();

    // Verify logger configuration
    EXPECT_EQ(fl::Logger::get_sink_type(), fl::SinkType::JsonConsole);
    EXPECT_EQ(fl::Logger::get_current_level(), fl::LogLevel::Debug);
    EXPECT_EQ(fl::Logger::get_actual_log_file(), ""); // No file for console
}

// Test: Verifies rotating JSON file logging functionality
TEST(RTLog, RotatingJsonFileLogging) {
    fl::TempFileManager temp_manager{"rotating_json"};
    const std::string log_file = temp_manager.get_temp_file(".json");

    fl::Logger::configure(
            fl::LoggerConfig::rotating_json_file(log_file.c_str(), fl::LogLevel::Debug));

    const std::string actual_log_file = fl::Logger::get_actual_log_file();

    // Log JSON data
    const int value1 =
            100; // NOLINT(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
    const std::string value2 = "rotating";
    const bool value3 = true;

    RT_LOGJ_DEBUG("Rotating JSON debug", value1);
    RT_LOGJ_INFO("Rotating JSON info", value2);
    RT_LOGJ_WARN("Rotating JSON warn", value3);
    RT_LOGJ_ERROR("Rotating JSON error", value1, value2);

    fl::Logger::flush();

    // Verify file exists and contains JSON data
    EXPECT_TRUE(std::filesystem::exists(actual_log_file));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "value1"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "100"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "value2"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "rotating"));
}

// Runtime logger operations

// Test: Verifies runtime log level changes work correctly
TEST(RTLog, RuntimeLevelChanges) {
    fl::TempFileManager temp_manager{"runtime_level"};
    const std::string log_file = temp_manager.get_temp_file(".log");

    // Start with DEBUG level
    fl::Logger::configure(fl::LoggerConfig::file(log_file.c_str(), fl::LogLevel::Debug));

    const std::string actual_log_file = fl::Logger::get_actual_log_file();

    // Verify initial level
    EXPECT_EQ(fl::Logger::get_current_level(), fl::LogLevel::Debug);

    // Log a debug message (should appear)
    RT_LOG_DEBUG("Debug before level change");

    // Change level to WARNING at runtime
    fl::Logger::set_level(fl::LogLevel::Warn);
    EXPECT_EQ(fl::Logger::get_current_level(), fl::LogLevel::Warn);

    // Log messages at different levels
    RT_LOG_DEBUG("Debug after level change (should NOT appear)");
    RT_LOG_INFO("Info after level change (should NOT appear)");
    RT_LOG_WARN("Warning after level change (should appear)");
    RT_LOG_ERROR("Error after level change (should appear)");

    fl::Logger::flush();

    // Verify level filtering worked
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Debug before level change"));
    EXPECT_FALSE(fl::file_contains(actual_log_file, "Debug after level change"));
    EXPECT_FALSE(fl::file_contains(actual_log_file, "Info after level change"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Warning after level change"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Error after level change"));
}

// Test: Verifies multiple reconfigurations work correctly
TEST(RTLog, MultipleReconfigurations) {
    fl::TempFileManager temp_manager{"reconfig"};

    // First configuration: Console
    fl::Logger::configure(fl::LoggerConfig::console(fl::LogLevel::Info));
    EXPECT_EQ(fl::Logger::get_sink_type(), fl::SinkType::Console);
    RT_LOG_INFO("Console message 1");

    // Second configuration: File
    const std::string log_file1 = temp_manager.get_temp_file("_first.log");
    fl::Logger::configure(fl::LoggerConfig::file(log_file1.c_str(), fl::LogLevel::Debug));
    EXPECT_EQ(fl::Logger::get_sink_type(), fl::SinkType::File);
    RT_LOG_DEBUG("File message 1");

    // Third configuration: JSON File
    const std::string log_file2 = temp_manager.get_temp_file("_second.json");
    fl::Logger::configure(fl::LoggerConfig::json_file(log_file2.c_str(), fl::LogLevel::Warn));
    EXPECT_EQ(fl::Logger::get_sink_type(), fl::SinkType::JsonFile);

    const int json_value =
            42; // NOLINT(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
    RT_LOGJ_WARN("JSON warning", json_value);

    fl::Logger::flush();

    // Verify each configuration worked
    const std::string actual_file2 = fl::Logger::get_actual_log_file();

    EXPECT_TRUE(fl::file_contains(actual_file2, "json_value"));
}

// Component system tests

// Test: Verifies component registration with level map
TEST(RTLog, ComponentRegistrationWithMap) {
    fl::TempFileManager temp_manager{"component_map"};
    const std::string log_file = temp_manager.get_temp_file(".log");

    fl::Logger::configure(fl::LoggerConfig::file(log_file.c_str(), fl::LogLevel::Debug));

    // Register components with different levels
    const std::unordered_map<SystemComponent, fl::LogLevel> component_levels = {
            {SystemComponent::Core, fl::LogLevel::Debug},
            {SystemComponent::Network, fl::LogLevel::Info},
            {SystemComponent::Database, fl::LogLevel::Warn},
            {SystemComponent::Security, fl::LogLevel::Error}};

    fl::register_component(component_levels);

    // Verify component levels were set correctly
    EXPECT_EQ(fl::get_component_level(SystemComponent::Core), fl::LogLevel::Debug);
    EXPECT_EQ(fl::get_component_level(SystemComponent::Network), fl::LogLevel::Info);
    EXPECT_EQ(fl::get_component_level(SystemComponent::Database), fl::LogLevel::Warn);
    EXPECT_EQ(fl::get_component_level(SystemComponent::Security), fl::LogLevel::Error);

    const std::string actual_log_file = fl::Logger::get_actual_log_file();

    // Test component filtering - only messages at or above component level should
    // appear
    RT_LOGC_DEBUG(SystemComponent::Core, "Core debug (should appear)");
    RT_LOGC_DEBUG(SystemComponent::Network, "Network debug (should NOT appear)");
    RT_LOGC_INFO(SystemComponent::Network, "Network info (should appear)");
    RT_LOGC_INFO(SystemComponent::Database, "Database info (should NOT appear)");
    RT_LOGC_WARN(SystemComponent::Database, "Database warn (should appear)");
    RT_LOGC_WARN(SystemComponent::Security, "Security warn (should NOT appear)");
    RT_LOGC_ERROR(SystemComponent::Security, "Security error (should appear)");

    fl::Logger::flush();

    // Verify filtering worked correctly
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Core debug (should appear)"));
    EXPECT_FALSE(fl::file_contains(actual_log_file, "Network debug (should NOT appear)"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Network info (should appear)"));
    EXPECT_FALSE(fl::file_contains(actual_log_file, "Database info (should NOT appear)"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Database warn (should appear)"));
    EXPECT_FALSE(fl::file_contains(actual_log_file, "Security warn (should NOT appear)"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Security error (should appear)"));
}

// Test: Verifies component registration with single level
TEST(RTLog, ComponentRegistrationWithSingleLevel) {
    fl::TempFileManager temp_manager{"component_single"};
    const std::string log_file = temp_manager.get_temp_file(".log");

    fl::Logger::configure(fl::LoggerConfig::file(log_file.c_str(), fl::LogLevel::Debug));

    // Register all components with ERROR level
    fl::register_component<SystemComponent>(fl::LogLevel::Error);

    // Verify all components have ERROR level
    EXPECT_EQ(fl::get_component_level(SystemComponent::Core), fl::LogLevel::Error);
    EXPECT_EQ(fl::get_component_level(SystemComponent::Network), fl::LogLevel::Error);
    EXPECT_EQ(fl::get_component_level(SystemComponent::Database), fl::LogLevel::Error);
    EXPECT_EQ(fl::get_component_level(SystemComponent::Security), fl::LogLevel::Error);

    const std::string actual_log_file = fl::Logger::get_actual_log_file();

    // Only ERROR messages should appear
    RT_LOGC_DEBUG(SystemComponent::Core, "Core debug (should NOT appear)");
    RT_LOGC_INFO(SystemComponent::Network, "Network info (should NOT appear)");
    RT_LOGC_WARN(SystemComponent::Database, "Database warn (should NOT appear)");
    RT_LOGC_ERROR(SystemComponent::Security, "Security error (should appear)");

    fl::Logger::flush();

    // Verify only ERROR appeared
    EXPECT_FALSE(fl::file_contains(actual_log_file, "Core debug (should NOT appear)"));
    EXPECT_FALSE(fl::file_contains(actual_log_file, "Network info (should NOT appear)"));
    EXPECT_FALSE(fl::file_contains(actual_log_file, "Database warn (should NOT appear)"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Security error (should appear)"));
}

// Event + component combined macros

// Test: Verifies combined event and component logging
TEST(RTLog, EventComponentCombinedLogging) {
    fl::TempFileManager temp_manager{"event_component"};
    const std::string log_file = temp_manager.get_temp_file(".log");

    fl::Logger::configure(fl::LoggerConfig::file(log_file.c_str(), fl::LogLevel::Debug));

    // RESET component levels to DEBUG for this test
    fl::register_component<SystemComponent>(fl::LogLevel::Debug);

    const std::string actual_log_file = fl::Logger::get_actual_log_file();

    // Use combined event + component macros
    RT_LOGEC_DEBUG(
            SystemComponent::Core,
            SystemEvent::AppStart,
            "Application starting in core: {}",
            "v1.0");
    RT_LOGEC_INFO(
            SystemComponent::Config,
            SystemEvent::ConfigLoaded,
            "Configuration loaded successfully");
    RT_LOGEC_WARN(
            SystemComponent::Network,
            SystemEvent::TIMEOUT,
            "Network timeout occurred: {} ms",
            5000);
    RT_LOGEC_ERROR(
            SystemComponent::Security, SystemEvent::CriticalError, "Security breach detected");

    fl::Logger::flush();

    // Verify combined format includes both component and event
    EXPECT_TRUE(fl::file_contains(actual_log_file, "[Core] EVENT [AppStart]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Application starting in core: v1.0"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "[Config] EVENT [ConfigLoaded]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "[Network] EVENT [TIMEOUT]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Network timeout occurred: 5000"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "[Security] EVENT [CriticalError]"));
}

// Predefined enums comprehensive testing

// Test: Verifies all SystemComponent values work correctly
TEST(RTLog, AllSystemComponents) {
    fl::TempFileManager temp_manager{"all_components"};
    const std::string log_file = temp_manager.get_temp_file(".log");

    fl::Logger::configure(fl::LoggerConfig::file(log_file.c_str(), fl::LogLevel::Debug));

    // RESET component levels to DEBUG for this test
    fl::register_component<SystemComponent>(fl::LogLevel::Debug);

    const std::string actual_log_file = fl::Logger::get_actual_log_file();

    // Test every SystemComponent value
    RT_LOGC_INFO(SystemComponent::Core, "Core component test");
    RT_LOGC_INFO(SystemComponent::Config, "Config component test");
    RT_LOGC_INFO(SystemComponent::Network, "Network component test");
    RT_LOGC_INFO(SystemComponent::Database, "Database component test");
    RT_LOGC_INFO(SystemComponent::Security, "Security component test");
    RT_LOGC_INFO(SystemComponent::Performance, "Performance component test");
    RT_LOGC_INFO(SystemComponent::ThreadPool, "Thread pool component test");
    RT_LOGC_INFO(SystemComponent::MemoryManager, "Memory manager component test");
    RT_LOGC_INFO(SystemComponent::FileSystem, "File system component test");
    RT_LOGC_INFO(SystemComponent::Ipc, "Ipc component test");
    RT_LOGC_INFO(SystemComponent::Scheduler, "Scheduler component test");
    RT_LOGC_INFO(SystemComponent::Monitor, "Monitor component test");

    fl::Logger::flush();

    // Verify all component names appear
    EXPECT_TRUE(fl::file_contains(actual_log_file, "[Core]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "[Config]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "[Network]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "[Database]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "[Security]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "[Performance]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "[ThreadPool]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "[MemoryManager]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "[FileSystem]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "[Ipc]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "[Scheduler]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "[Monitor]"));
}

// Test: Verifies all SystemEvent values work correctly
TEST(RTLog, AllSystemEvents) {
    fl::TempFileManager temp_manager{"all_events"};
    const std::string log_file = temp_manager.get_temp_file(".log");

    fl::Logger::configure(fl::LoggerConfig::file(log_file.c_str(), fl::LogLevel::Debug));

    const std::string actual_log_file = fl::Logger::get_actual_log_file();

    // Test every SystemEvent value
    RT_LOGE_INFO(SystemEvent::AppStart, "App start event");
    RT_LOGE_INFO(SystemEvent::AppStop, "App stop event");
    RT_LOGE_INFO(SystemEvent::ConfigLoaded, "Config loaded event");
    RT_LOGE_INFO(SystemEvent::ConfigError, "Config error event");
    RT_LOGE_INFO(SystemEvent::ShutdownRequest, "Shutdown request event");
    RT_LOGE_INFO(SystemEvent::SHUTDOWN_COMPLETE, "Shutdown complete event");
    RT_LOGE_INFO(SystemEvent::HealthCheckOk, "Health check OK event");
    RT_LOGE_INFO(SystemEvent::HEALTH_CHECK_FAIL, "Health check fail event");
    RT_LOGE_INFO(SystemEvent::THREAD_START, "Thread start event");
    RT_LOGE_INFO(SystemEvent::THREAD_STOP, "Thread stop event");
    RT_LOGE_INFO(SystemEvent::RESOURCE_ALLOC, "Resource alloc event");
    RT_LOGE_INFO(SystemEvent::RESOURCE_FREE, "Resource free event");
    RT_LOGE_INFO(SystemEvent::TIMEOUT, "Timeout event");
    RT_LOGE_INFO(SystemEvent::Retry, "Retry event");
    RT_LOGE_INFO(SystemEvent::CriticalError, "Critical error event");

    fl::Logger::flush();

    // Verify all event names appear
    EXPECT_TRUE(fl::file_contains(actual_log_file, "EVENT [AppStart]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "EVENT [AppStop]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "EVENT [ConfigLoaded]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "EVENT [ConfigError]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "EVENT [ShutdownRequest]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "EVENT [SHUTDOWN_COMPLETE]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "EVENT [HealthCheckOk]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "EVENT [HEALTH_CHECK_FAIL]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "EVENT [THREAD_START]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "EVENT [THREAD_STOP]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "EVENT [RESOURCE_ALLOC]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "EVENT [RESOURCE_FREE]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "EVENT [TIMEOUT]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "EVENT [Retry]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "EVENT [CriticalError]"));
}

// Test: Verifies all ErrorEvent values work correctly
TEST(RTLog, AllErrorEvents) {
    fl::TempFileManager temp_manager{"all_error_events"};
    const std::string log_file = temp_manager.get_temp_file(".log");

    fl::Logger::configure(fl::LoggerConfig::file(log_file.c_str(), fl::LogLevel::Debug));

    const std::string actual_log_file = fl::Logger::get_actual_log_file();

    // Test every ErrorEvent value
    RT_LOGE_ERROR(ErrorEvent::InvalidParam, "Invalid parameter error");
    RT_LOGE_ERROR(ErrorEvent::OutOfMemory, "Out of memory error");
    RT_LOGE_ERROR(ErrorEvent::FILE_NOT_FOUND, "File not found error");
    RT_LOGE_ERROR(ErrorEvent::PERMISSION_DENIED, "Permission denied error");
    RT_LOGE_ERROR(ErrorEvent::NetworkError, "Network error");
    RT_LOGE_ERROR(ErrorEvent::OPERATION_FAILED, "Operation failed error");
    RT_LOGE_ERROR(ErrorEvent::OPERATION_TIMEOUT, "Operation timeout error");
    RT_LOGE_ERROR(ErrorEvent::InvalidState, "Invalid state error");
    RT_LOGE_ERROR(ErrorEvent::BUFFER_OVERFLOW, "Buffer overflow error");
    RT_LOGE_ERROR(ErrorEvent::RESOURCE_EXHAUSTED, "Resource exhausted error");
    RT_LOGE_ERROR(ErrorEvent::AuthenticationFailed, "Authentication failed error");
    RT_LOGE_ERROR(ErrorEvent::CONFIGURATION_ERROR, "Configuration error");
    RT_LOGE_ERROR(ErrorEvent::PROTOCOL_ERROR, "Protocol error");
    RT_LOGE_ERROR(ErrorEvent::SERIALIZATION_ERROR, "Serialization error");
    RT_LOGE_ERROR(ErrorEvent::DESERIALIZATION_ERROR, "Deserialization error");
    RT_LOGE_ERROR(ErrorEvent::CONNECTION_FAILED, "Connection failed error");
    RT_LOGE_ERROR(ErrorEvent::RESOURCE_UNAVAILABLE, "Resource unavailable error");

    fl::Logger::flush();

    // Verify all error event names appear
    EXPECT_TRUE(fl::file_contains(actual_log_file, "EVENT [InvalidParam]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "EVENT [OutOfMemory]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "EVENT [FILE_NOT_FOUND]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "EVENT [PERMISSION_DENIED]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "EVENT [NetworkError]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "EVENT [OPERATION_FAILED]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "EVENT [OPERATION_TIMEOUT]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "EVENT [InvalidState]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "EVENT [BUFFER_OVERFLOW]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "EVENT [RESOURCE_EXHAUSTED]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "EVENT [AuthenticationFailed]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "EVENT [CONFIGURATION_ERROR]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "EVENT [PROTOCOL_ERROR]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "EVENT [SERIALIZATION_ERROR]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "EVENT [DESERIALIZATION_ERROR]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "EVENT [CONNECTION_FAILED]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "EVENT [RESOURCE_UNAVAILABLE]"));
}

// Utility functions and registry tests

// Test: Verifies utility functions work correctly
TEST(RTLog, UtilityFunctions) {
    // Test format_component_name - Fixed: convert string_view to const char*
    EXPECT_STREQ(fl::format_component_name(SystemComponent::Core).data(), "Core");
    EXPECT_STREQ(fl::format_component_name(SystemComponent::Network).data(), "Network");
    EXPECT_STREQ(fl::format_component_name(SystemComponent::Security).data(), "Security");

    // Test format_event_name - Fixed: convert string_view to const char*
    EXPECT_STREQ(fl::format_event_name(SystemEvent::AppStart).data(), "AppStart");
    EXPECT_STREQ(fl::format_event_name(ErrorEvent::InvalidParam).data(), "InvalidParam");

    // Test is_valid_component
    EXPECT_TRUE(fl::is_valid_component(SystemComponent::Core));
    EXPECT_TRUE(fl::is_valid_component(SystemComponent::Monitor));

    // Test is_valid_event
    EXPECT_TRUE(fl::is_valid_event(SystemEvent::AppStart));
    EXPECT_TRUE(fl::is_valid_event(ErrorEvent::NetworkError));
}

// Test: Verifies registry functionality
TEST(RTLog, RegistryFunctionality) {
    // Test ComponentRegistry
    using CoreRegistry = fl::ComponentRegistry<SystemComponent>;

    EXPECT_GT(CoreRegistry::get_table_size(), 0);
    EXPECT_STREQ(CoreRegistry::get_name(SystemComponent::Core).data(), "Core");
    EXPECT_TRUE(CoreRegistry::is_valid(SystemComponent::Core));
    EXPECT_EQ(CoreRegistry::get_index(SystemComponent::Core), 0);

    // Test EventRegistry
    using EventReg = fl::EventRegistry<SystemEvent>;

    EXPECT_GT(EventReg::get_table_size(), 0);
    EXPECT_STREQ(EventReg::get_name(SystemEvent::AppStart).data(), "AppStart");
    EXPECT_TRUE(EventReg::is_valid(SystemEvent::AppStart));
    EXPECT_EQ(EventReg::get_index(SystemEvent::AppStart), 0);
}

// Error handling tests

// Test: Verifies error handling for missing log files
TEST(RTLog, ErrorHandlingMissingLogFile) {
    // These should throw for file-based sinks without log_file
    EXPECT_THROW(
            { fl::Logger::configure(fl::LoggerConfig::file(nullptr, fl::LogLevel::Info)); },
            std::invalid_argument);

    EXPECT_THROW(
            { fl::Logger::configure(fl::LoggerConfig::file("", fl::LogLevel::Info)); },
            std::invalid_argument);

    EXPECT_THROW(
            {
                fl::Logger::configure(fl::LoggerConfig::rotating_file(nullptr, fl::LogLevel::Info));
            },
            std::invalid_argument);

    EXPECT_THROW(
            { fl::Logger::configure(fl::LoggerConfig::json_file(nullptr, fl::LogLevel::Info)); },
            std::invalid_argument);

    EXPECT_THROW(
            {
                fl::Logger::configure(fl::LoggerConfig::rotating_json_file("", fl::LogLevel::Info));
            },
            std::invalid_argument);
}

// Log level comprehensive tests

// Test: Verifies all log levels work correctly
TEST(RTLog, AllLogLevels) {
    fl::TempFileManager temp_manager{"all_levels"};
    const std::string log_file = temp_manager.get_temp_file(".log");

    fl::Logger::configure(fl::LoggerConfig::file(log_file.c_str(), fl::LogLevel::TraceL3));

    const std::string actual_log_file = fl::Logger::get_actual_log_file();

    // Test all log levels
    RT_LOG_TRACE_L3("Trace L3 level test: {}", static_cast<int>(fl::LogLevel::TraceL3));
    RT_LOG_TRACE_L2("Trace L2 level test: {}", static_cast<int>(fl::LogLevel::TraceL2));
    RT_LOG_TRACE_L1("Trace L1 level test: {}", static_cast<int>(fl::LogLevel::TraceL1));
    RT_LOG_DEBUG("Debug level test: {}", static_cast<int>(fl::LogLevel::Debug));
    RT_LOG_INFO("Info level test: {}", static_cast<int>(fl::LogLevel::Info));
    RT_LOG_NOTICE("Notice level test: {}", static_cast<int>(fl::LogLevel::Notice));
    RT_LOG_WARN("Warning level test: {}", static_cast<int>(fl::LogLevel::Warn));
    RT_LOG_ERROR("Error level test: {}", static_cast<int>(fl::LogLevel::Error));
    RT_LOG_CRITICAL("Critical level test: {}", static_cast<int>(fl::LogLevel::Critical));

    fl::Logger::flush();

    // Verify all levels appear with their numeric values
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Trace L3 level test: 0"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Trace L2 level test: 1"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Trace L1 level test: 2"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Debug level test: 3"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Info level test: 4"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Notice level test: 5"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Warning level test: 6"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Error level test: 7"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Critical level test: 8"));

    // Verify log level markers appear in output
    EXPECT_TRUE(fl::file_contains(actual_log_file, "[TRACE_L3]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "[TRACE_L2]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "[TRACE_L1]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "[DEBUG]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "[INFO]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "[NOTICE]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "[WARNING]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "[ERROR]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "[CRITICAL]"));
}

// Test: Verifies new log levels work with filtering (extreme levels)
TEST(RTLog, NewLogLevelsFiltering) {
    fl::TempFileManager temp_manager{"new_levels_filtering"};
    const std::string log_file = temp_manager.get_temp_file(".log");

    // Configure logger with NOTICE level - should filter out TRACE and DEBUG
    fl::Logger::configure(fl::LoggerConfig::file(log_file.c_str(), fl::LogLevel::Notice));

    const std::string actual_log_file = fl::Logger::get_actual_log_file();

    // Log at different levels
    RT_LOG_TRACE_L3("This trace L3 should NOT appear");
    RT_LOG_TRACE_L2("This trace L2 should NOT appear");
    RT_LOG_TRACE_L1("This trace L1 should NOT appear");
    RT_LOG_DEBUG("This debug should NOT appear");
    RT_LOG_INFO("This info should NOT appear");
    RT_LOG_NOTICE("This notice should appear");
    RT_LOG_WARN("This warning should appear");
    RT_LOG_ERROR("This error should appear");
    RT_LOG_CRITICAL("This critical should appear");

    fl::Logger::flush();

    // Verify filtering worked correctly
    EXPECT_FALSE(fl::file_contains(actual_log_file, "This trace L3 should NOT appear"));
    EXPECT_FALSE(fl::file_contains(actual_log_file, "This trace L2 should NOT appear"));
    EXPECT_FALSE(fl::file_contains(actual_log_file, "This trace L1 should NOT appear"));
    EXPECT_FALSE(fl::file_contains(actual_log_file, "This debug should NOT appear"));
    EXPECT_FALSE(fl::file_contains(actual_log_file, "This info should NOT appear"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "This notice should appear"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "This warning should appear"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "This error should appear"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "This critical should appear"));
}

// Test: Verifies component logging with new log levels
TEST(RTLog, ComponentLoggingNewLevels) {
    fl::TempFileManager temp_manager{"component_new_levels"};
    const std::string log_file = temp_manager.get_temp_file(".log");

    fl::Logger::configure(fl::LoggerConfig::file(log_file.c_str(), fl::LogLevel::TraceL3));

    // Reset component levels to TRACE_L3 for this test
    fl::register_component<SystemComponent>(fl::LogLevel::TraceL3);

    const std::string actual_log_file = fl::Logger::get_actual_log_file();

    // Test component logging with new levels
    RT_LOGC_TRACE_L3(SystemComponent::Core, "Core trace L3 message: {}", 1);
    RT_LOGC_TRACE_L2(SystemComponent::Network, "Network trace L2 message: {}", 2);
    RT_LOGC_TRACE_L1(SystemComponent::Database, "Database trace L1 message: {}", 3);
    RT_LOGC_NOTICE(SystemComponent::Security, "Security notice message: {}", 4);
    RT_LOGC_CRITICAL(SystemComponent::Performance, "Performance critical message: {}", 5);

    fl::Logger::flush();

    // Verify component messages with new levels appear
    EXPECT_TRUE(fl::file_contains(actual_log_file, "[Core]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Core trace L3 message: 1"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "[Network]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Network trace L2 message: 2"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "[Database]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Database trace L1 message: 3"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "[Security]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Security notice message: 4"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "[Performance]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Performance critical message: 5"));
}

// Test: Verifies event logging with new log levels
TEST(RTLog, EventLoggingNewLevels) {
    fl::TempFileManager temp_manager{"event_new_levels"};
    const std::string log_file = temp_manager.get_temp_file(".log");

    fl::Logger::configure(fl::LoggerConfig::file(log_file.c_str(), fl::LogLevel::TraceL3));

    const std::string actual_log_file = fl::Logger::get_actual_log_file();

    // Test event logging with new levels
    RT_LOGE_TRACE_L3(SystemEvent::AppStart, "Application trace start: {}", "v2.0");
    RT_LOGE_TRACE_L2(SystemEvent::ConfigLoaded, "Configuration trace loaded: {}", "detailed");
    RT_LOGE_TRACE_L1(SystemEvent::THREAD_START, "Thread trace start: {}", "worker-1");
    RT_LOGE_NOTICE(SystemEvent::HealthCheckOk, "Health check notice: {}", "system ok");
    RT_LOGE_CRITICAL(SystemEvent::CriticalError, "Critical system failure: {}", "disk full");

    fl::Logger::flush();

    // Verify event messages with new levels appear
    EXPECT_TRUE(fl::file_contains(actual_log_file, "EVENT [AppStart]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Application trace start: v2.0"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "EVENT [ConfigLoaded]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Configuration trace loaded: detailed"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "EVENT [THREAD_START]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Thread trace start: worker-1"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "EVENT [HealthCheckOk]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Health check notice: system ok"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "EVENT [CriticalError]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Critical system failure: disk full"));
}

// Test: Verifies JSON logging with new log levels
TEST(RTLog, JSONLoggingNewLevels) {
    fl::TempFileManager temp_manager{"json_new_levels"};
    const std::string log_file = temp_manager.get_temp_file(".json");

    fl::Logger::configure(fl::LoggerConfig::json_file(log_file.c_str(), fl::LogLevel::TraceL3));

    const std::string actual_log_file = fl::Logger::get_actual_log_file();

    // Test JSON logging with new levels
    const int trace_value =
            123; // NOLINT(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
    const std::string notice_msg = "system notice";
    const std::string critical_msg = "critical failure";

    RT_LOGJ_TRACE_L3("JSON trace L3", trace_value);
    RT_LOGJ_NOTICE("JSON notice message", notice_msg);
    RT_LOGJ_CRITICAL("JSON critical message", critical_msg);

    fl::Logger::flush();

    // Verify JSON file contains the variable names and values
    EXPECT_TRUE(fl::file_contains(actual_log_file, "trace_value"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "123"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "notice_msg"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "system notice"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "critical_msg"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "critical failure"));
}

// Default level function test

// Test: Verifies get_logger_default_level, get_current_level function
TEST(RTLog, GetLoggerLevel) {
    // Configure logger with specific level
    fl::Logger::configure(fl::LoggerConfig::console(fl::LogLevel::Warn));

    // Verify get_logger_default_level matches current level
    EXPECT_EQ(fl::Logger::get_current_level(), fl::LogLevel::Warn);

    // Change level and verify again
    fl::Logger::set_level(fl::LogLevel::Error);
    EXPECT_EQ(fl::Logger::get_current_level(), fl::LogLevel::Error);

    EXPECT_EQ(fl::get_logger_default_level(), fl::LogLevel::Info);
}

// Advanced configuration tests

// Test: Verifies CPU affinity configuration
TEST(RTLog, CpuAffinityConfiguration) {
    constexpr uint16_t TEST_CPU_AFFINITY = 3;

    fl::Logger::configure(fl::LoggerConfig::console().with_cpu_affinity(TEST_CPU_AFFINITY));

    // Just verify configuration doesn't crash
    RT_LOG_INFO("CPU affinity test: {}", TEST_CPU_AFFINITY);
    fl::Logger::flush();
}

// Test: Verifies file line and caller information formatting
TEST(RTLog, FileLineAndCallerFormatting) {
    fl::TempFileManager temp_manager{"file_line_caller"};
    const std::string log_file = temp_manager.get_temp_file(".log");

    // Configure with both file/line and caller info
    fl::Logger::configure(fl::LoggerConfig::file(log_file.c_str(), fl::LogLevel::Debug)
                                  .with_file_line(true)
                                  .with_caller(true));

    const std::string actual_log_file = fl::Logger::get_actual_log_file();

    const int log_line = __LINE__ + 1; // Capture the exact line number
    RT_LOG_INFO("Test file line and caller formatting");
    fl::Logger::flush();

    // Verify file exists and contains the message
    EXPECT_TRUE(std::filesystem::exists(actual_log_file));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Test file line and caller formatting"));

    // Verify file name appears in the log
    EXPECT_TRUE(fl::file_contains(actual_log_file, "rt_log_tests.cpp"));

    // Verify line number appears in the log
    EXPECT_TRUE(fl::file_contains(actual_log_file, std::to_string(log_line)));

    // Verify caller function name appears (GTest uses TestBody for TEST macros)
    EXPECT_TRUE(fl::file_contains(actual_log_file, "TestBody"));
}

// Pattern configuration testsx

// Test: Verifies timestamps can be explicitly enabled (default behavior)
TEST(RTLog, EnableTimestamps) {
    fl::TempFileManager temp_manager{"enable_timestamps"};
    const std::string log_file = temp_manager.get_temp_file(".log");

    // Configure with timestamps explicitly enabled
    fl::Logger::configure(fl::LoggerConfig::file(log_file.c_str(), fl::LogLevel::Debug)
                                  .with_timestamps(true)
                                  .with_thread_name(true));

    const std::string actual_log_file = fl::Logger::get_actual_log_file();

    RT_LOG_INFO("Message with timestamp");
    fl::Logger::flush();

    // Verify file exists and contains the message
    EXPECT_TRUE(std::filesystem::exists(actual_log_file));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Message with timestamp"));

    // Verify timestamp pattern appears (contains time format with colons)
    EXPECT_TRUE(fl::file_contains(actual_log_file, ":"));

    // And log level should also appear
    EXPECT_TRUE(fl::file_contains(actual_log_file, "[INFO]"));
}

// Test: Verifies log level display can be disabled
TEST(RTLog, DisableLogLevel) {
    fl::TempFileManager temp_manager{"disable_log_level"};
    const std::string log_file = temp_manager.get_temp_file(".log");

    // Configure with log level display disabled
    fl::Logger::configure(fl::LoggerConfig::file(log_file.c_str(), fl::LogLevel::Debug)
                                  .with_log_level(false)
                                  .with_thread_name(true));

    const std::string actual_log_file = fl::Logger::get_actual_log_file();

    RT_LOG_INFO("Message without log level");
    RT_LOG_ERROR("Error without log level");
    fl::Logger::flush();

    // Verify file exists and contains the messages
    EXPECT_TRUE(std::filesystem::exists(actual_log_file));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Message without log level"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Error without log level"));

    // Verify log level markers do NOT appear
    EXPECT_FALSE(fl::file_contains(actual_log_file, "[INFO]"));
    EXPECT_FALSE(fl::file_contains(actual_log_file, "[ERROR]"));

    // But timestamps should still appear
    EXPECT_TRUE(fl::file_contains(actual_log_file, ":")); // Timestamp
}

// Test: Verifies log level display can be explicitly enabled (default behavior)
TEST(RTLog, EnableLogLevel) {
    fl::TempFileManager temp_manager{"enable_log_level"};
    const std::string log_file = temp_manager.get_temp_file(".log");

    // Configure with log level display explicitly enabled
    fl::Logger::configure(fl::LoggerConfig::file(log_file.c_str(), fl::LogLevel::Debug)
                                  .with_log_level(true)
                                  .with_thread_name(true));

    const std::string actual_log_file = fl::Logger::get_actual_log_file();

    RT_LOG_INFO("Message with log level");
    RT_LOG_ERROR("Error with log level");
    fl::Logger::flush();

    // Verify file exists and contains the messages
    EXPECT_TRUE(std::filesystem::exists(actual_log_file));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Message with log level"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Error with log level"));

    // Verify log level markers appear
    EXPECT_TRUE(fl::file_contains(actual_log_file, "[INFO]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "[ERROR]"));

    // And timestamps should also appear
    EXPECT_TRUE(fl::file_contains(actual_log_file, ":")); // Timestamp
}

// Test: Verifies thread name display can be disabled
TEST(RTLog, DisableThreadName) {
    fl::TempFileManager temp_manager{"disable_thread_name"};
    const std::string log_file = temp_manager.get_temp_file(".log");

    // Configure with thread name display disabled
    fl::Logger::configure(
            fl::LoggerConfig::file(log_file.c_str(), fl::LogLevel::Debug).with_thread_name(false));

    const std::string actual_log_file = fl::Logger::get_actual_log_file();

    RT_LOG_INFO("Message without thread name");
    fl::Logger::flush();

    // Verify file exists and contains the message
    EXPECT_TRUE(std::filesystem::exists(actual_log_file));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Message without thread name"));

    // But timestamps and log level should still appear
    EXPECT_TRUE(fl::file_contains(actual_log_file, ":")); // Timestamp
    EXPECT_TRUE(fl::file_contains(actual_log_file, "[INFO]"));
}

// Test: Verifies thread name display can be explicitly enabled (default
// behavior)
TEST(RTLog, EnableThreadName) {
    fl::TempFileManager temp_manager{"enable_thread_name"};
    const std::string log_file = temp_manager.get_temp_file(".log");

    // Configure with thread name display explicitly enabled
    fl::Logger::configure(
            fl::LoggerConfig::file(log_file.c_str(), fl::LogLevel::Debug).with_thread_name(true));

    const std::string actual_log_file = fl::Logger::get_actual_log_file();

    RT_LOG_INFO("Message with thread name");
    fl::Logger::flush();

    // Verify file exists and contains the message
    EXPECT_TRUE(std::filesystem::exists(actual_log_file));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Message with thread name"));

    // And timestamps and log level should also appear
    EXPECT_TRUE(fl::file_contains(actual_log_file, ":")); // Timestamp
    EXPECT_TRUE(fl::file_contains(actual_log_file, "[INFO]"));
}

// Test: Verifies backend sleep duration can be configured
TEST(RTLog, BackendSleepDurationConfiguration) {
    fl::TempFileManager temp_manager{"backend_sleep"};
    const std::string log_file = temp_manager.get_temp_file(".log");

    // Configure with custom backend sleep duration
    constexpr auto CUSTOM_SLEEP_DURATION = std::chrono::nanoseconds{500};
    fl::Logger::configure(fl::LoggerConfig::file(log_file.c_str(), fl::LogLevel::Debug)
                                  .with_backend_sleep_duration(CUSTOM_SLEEP_DURATION));

    const std::string actual_log_file = fl::Logger::get_actual_log_file();

    RT_LOG_INFO("Message with custom backend sleep duration: {}ns", CUSTOM_SLEEP_DURATION.count());
    fl::Logger::flush();

    // Verify file exists and contains the message
    EXPECT_TRUE(std::filesystem::exists(actual_log_file));
    EXPECT_TRUE(fl::file_contains(
            actual_log_file, "Message with custom backend sleep duration: 500ns"));

    // The configuration should also log the backend sleep duration in the setup
    // message
    EXPECT_TRUE(fl::file_contains(actual_log_file, "BackendSleep: 500ns"));
}

// Test: Verifies combination of pattern options with file/line and caller info
TEST(RTLog, PatternCombinationWithFileLineAndCaller) {
    fl::TempFileManager temp_manager{"pattern_combo"};
    const std::string log_file = temp_manager.get_temp_file(".log");

    // Configure with timestamps and thread name disabled, but file/line and
    // caller enabled
    fl::Logger::configure(fl::LoggerConfig::file(log_file.c_str(), fl::LogLevel::Debug)
                                  .with_timestamps(false)
                                  .with_thread_name(false)
                                  .with_file_line(true)
                                  .with_caller(true));

    const std::string actual_log_file = fl::Logger::get_actual_log_file();

    const int log_line = __LINE__ + 1; // Capture the exact line number
    RT_LOG_INFO("Pattern combination test");
    fl::Logger::flush();

    // Verify file exists and contains the message
    EXPECT_TRUE(std::filesystem::exists(actual_log_file));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Pattern combination test"));

    // Verify log level still appears (not disabled)
    EXPECT_TRUE(fl::file_contains(actual_log_file, "[INFO]"));

    // Verify file/line and caller info appear
    EXPECT_TRUE(fl::file_contains(actual_log_file, "rt_log_tests.cpp"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, std::to_string(log_line)));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "TestBody"));
}

// Test: Verifies fluent API chaining with all new options
TEST(RTLog, FluentAPIChaining) {
    fl::TempFileManager temp_manager{"fluent_api"};
    const std::string log_file = temp_manager.get_temp_file(".log");

    // Test chaining all new options together
    constexpr auto CUSTOM_SLEEP = std::chrono::nanoseconds{250};
    fl::Logger::configure(fl::LoggerConfig::file(log_file.c_str(), fl::LogLevel::Debug)
                                  .with_timestamps(true)
                                  .with_log_level(true)
                                  .with_thread_name(true)
                                  .with_file_line(false)
                                  .with_caller(false)
                                  .with_backend_sleep_duration(CUSTOM_SLEEP)
                                  .with_cpu_affinity(1));

    const std::string actual_log_file = fl::Logger::get_actual_log_file();

    RT_LOG_INFO("Fluent API chaining test: all options configured");
    fl::Logger::flush();

    // Verify file exists and contains the message
    EXPECT_TRUE(std::filesystem::exists(actual_log_file));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Fluent API chaining test"));

    // Verify enabled pattern components appear
    EXPECT_TRUE(fl::file_contains(actual_log_file, ":")); // Timestamp
    EXPECT_TRUE(fl::file_contains(actual_log_file, "[INFO]"));

    // Verify configuration was logged with our custom sleep duration
    EXPECT_TRUE(fl::file_contains(actual_log_file, "BackendSleep: 250ns"));
}

// Test: Verifies pattern options work with JSON logging
TEST(RTLog, PatternOptionsWithJSON) {
    fl::TempFileManager temp_manager{"json_pattern"};
    const std::string log_file = temp_manager.get_temp_file(".json");

    // Configure JSON with some pattern options disabled
    fl::Logger::configure(fl::LoggerConfig::json_file(log_file.c_str(), fl::LogLevel::Debug)
                                  .with_timestamps(false)
                                  .with_log_level(true)
                                  .with_thread_name(false));

    const std::string actual_log_file = fl::Logger::get_actual_log_file();

    const int test_value =
            123; // NOLINT(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
    const std::string test_string = "json_test";

    RT_LOGJ_INFO("JSON pattern test", test_value, test_string);
    fl::Logger::flush();

    // Verify file exists and contains JSON data
    EXPECT_TRUE(std::filesystem::exists(actual_log_file));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "test_value"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "123"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "test_string"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "json_test"));
}

// Test: Verifies pattern options work with component logging
TEST(RTLog, PatternOptionsWithComponentLogging) {
    fl::TempFileManager temp_manager{"component_pattern"};
    const std::string log_file = temp_manager.get_temp_file(".log");

    // Configure with thread name disabled but everything else enabled
    fl::Logger::configure(
            fl::LoggerConfig::file(log_file.c_str(), fl::LogLevel::Debug).with_thread_name(false));

    // Reset component levels for this test
    fl::register_component<SystemComponent>(fl::LogLevel::Debug);

    const std::string actual_log_file = fl::Logger::get_actual_log_file();

    RT_LOGC_INFO(SystemComponent::Core, "Component logging with custom pattern: {}", "test");
    fl::Logger::flush();

    // Verify file exists and contains component message
    EXPECT_TRUE(std::filesystem::exists(actual_log_file));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "[Core]"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Component logging with custom pattern: test"));

    // Verify timestamps and log level appear
    EXPECT_TRUE(fl::file_contains(actual_log_file, ":")); // Timestamp
    EXPECT_TRUE(fl::file_contains(actual_log_file, "[INFO]"));
}
} // namespace

// ==================== USER DEFINED TYPES (GLOBAL SCOPE) ====================
// These must be at global scope for template specialization to work

// User class with unsafe pointer member - requires direct formatting

/**
 * Test class demonstrating unsafe type logging with pointer members
 *
 * This class contains a pointer member making it unsafe for deferred
 * formatting across threads. It must use RT_LOGGABLE_DIRECT_FORMAT
 * to ensure immediate formatting in the calling thread.
 */
class User {
public:
    std::string name;             //!< User's display name
    uint64_t *value_ptr{nullptr}; //!< Pointer to external value (unsafe for copying)
    std::array<int, 3> arr{
            1, 2, 3}; // NOLINT(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
                      // //!< Fixed-size array for testing

    /**
     * Create a new User instance
     *
     * @param[in] user_name Display name for the user
     * @param[in] ptr Pointer to external uint64_t value (can be nullptr)
     */
    User(std::string user_name, uint64_t *ptr) : name(std::move(user_name)), value_ptr(ptr) {}
};

// Mark as unsafe type - will be formatted immediately
RT_LOGGABLE_DIRECT_FORMAT(
        User,
        "name: {}, value: {}, arr: {}",
        obj.name,
        obj.value_ptr ? *obj.value_ptr : 0,
        obj.arr);

// Product class with only value types - safe for deferred formatting

/**
 * Test class demonstrating safe type logging with value-only members
 *
 * This class contains only value types making it safe for deferred
 * formatting. It can use RT_LOGGABLE_DEFERRED_FORMAT to allow
 * asynchronous formatting in the background thread for better performance.
 */
class Product {
public:
    std::string name;                             //!< Product name
    double price{0.0};                            //!< Product price in dollars
    int quantity{0};                              //!< Available quantity in stock
    std::vector<std::pair<int, std::string>> vec; //!< Feature list with ID and description pairs

    /**
     * Create a new Product instance
     *
     * @param[in] product_name Name of the product
     * @param[in] product_vec Vector of feature pairs (ID, description)
     * @param[in] product_price Price in dollars
     * @param[in] product_quantity Available quantity
     */
    Product(std::string product_name,
            std::vector<std::pair<int, std::string>> product_vec,
            double product_price, // NOLINT(bugprone-easily-swappable-parameters)
            int product_quantity)
            : name(std::move(product_name)), price(product_price), quantity(product_quantity),
              vec(std::move(product_vec)) {}
};

// Mark as safe type - can be formatted asynchronously
RT_LOGGABLE_DEFERRED_FORMAT(
        Product,
        "name: {}, price: {}, quantity: {}, vec: {}",
        obj.name,
        obj.price,
        obj.quantity,
        obj.vec);

namespace {
// Test: Verifies user-defined types work with Quill formatting macros
TEST(RTLog, UserDefinedTypes) {
    fl::TempFileManager temp_manager{"user_defined_types"};
    const std::string log_file = temp_manager.get_temp_file(".log");

    fl::Logger::configure(fl::LoggerConfig::file(log_file.c_str(), fl::LogLevel::Debug));

    const std::string actual_log_file = fl::Logger::get_actual_log_file();

    // Create test data
    uint64_t user_value =
            12345; // NOLINT(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
    const User unsafe_user{"Alice", &user_value};
    const User safe_user{"Bob", nullptr};

    const Product product1{
            "Laptop",
            {{1, "feature1"}, {2, "feature2"}},
            999.99,
            5}; // NOLINT(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
    const Product product2{
            "Mouse",
            {{1, "wireless"}, {2, "ergonomic"}},
            29.99,
            50}; // NOLINT(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

    // Test direct formatting (unsafe types - formatted immediately)
    RT_LOG_INFO("Logging unsafe user with pointer: {}", unsafe_user);
    RT_LOG_WARN("Logging safe user without pointer: {}", safe_user);

    // Test deferred formatting (safe types - can be copied and formatted later)
    RT_LOG_INFO("Logging product 1: {}", product1);
    RT_LOG_DEBUG("Logging product 2: {}", product2);

    fl::Logger::flush();

    // Verify file exists and contains formatted output
    EXPECT_TRUE(std::filesystem::exists(actual_log_file));

    // Verify User formatting (direct format)
    EXPECT_TRUE(
            fl::file_contains(actual_log_file, "User(name: Alice, value: 12345, arr: [1, 2, 3])"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "User(name: Bob, value: 0, arr: [1, 2, 3])"));

    // Verify Product formatting (deferred format)
    EXPECT_TRUE(fl::file_contains(
            actual_log_file,
            R"(Product(name: Laptop, price: 999.99, quantity: 5, vec: [(1, "feature1"), (2, "feature2")]))"));
    EXPECT_TRUE(fl::file_contains(
            actual_log_file,
            R"(Product(name: Mouse, price: 29.99, quantity: 50, vec: [(1, "wireless"), (2, "ergonomic")]))"));
}

TEST(RTLog, SpanLogging) {
    fl::TempFileManager temp_manager{"span_logging"};
    const std::string log_file = temp_manager.get_temp_file(".log");

    fl::Logger::configure(fl::LoggerConfig::file(log_file.c_str(), fl::LogLevel::Debug));

    const std::string actual_log_file = fl::Logger::get_actual_log_file();

    // Test spans wrapping different source types

    // 1. std::array (stack-allocated)
    const std::array<int64_t, 4> array_data{16, 3, 224, 224};
    const std::span<const int64_t> span_from_array(array_data);
    RT_LOG_INFO("Span from std::array: {}", span_from_array);

    // 2. std::vector (heap-allocated)
    const std::vector<int32_t> vector_data{1024, 768, 512};
    const std::span<const int32_t> span_from_vector(vector_data);
    RT_LOG_INFO("Span from std::vector: {}", span_from_vector);

    // 3. C array (stack-allocated)
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays)
    const int64_t c_array[] = {8, 224, 224};
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)
    const std::span<const int64_t> span_from_c_array(c_array, 3);
    RT_LOG_INFO("Span from C array: {}", span_from_c_array);

    // 4. Raw pointer + size (common in APIs)
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays)
    const float raw_data[] = {1.0F, 2.5F, 3.14F, 4.2F};
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)
    const std::span<const float> span_from_pointer(raw_data, 4);
    RT_LOG_DEBUG("Span from raw pointer: {}", span_from_pointer);

    // 5. Subspan (view into existing data)
    const std::array<int64_t, 6> full_dims{1, 3, 224, 224, 512, 256};
    const std::span<const int64_t> full_span(full_dims);
    const std::span<const int64_t> sub_span = full_span.subspan(0, 4);
    RT_LOG_INFO("Subspan (first 4 elements): {}", sub_span);

    // 6. Empty span
    const std::span<const int64_t> empty_span;
    RT_LOG_INFO("Empty span: {}", empty_span);

    // 7. Different numeric types
    const std::array<uint32_t, 3> uint_data{100, 200, 300};
    const std::span<const uint32_t> span_uint(uint_data);
    RT_LOG_DEBUG("Span of uint32_t: {}", span_uint);

    // NOLINTNEXTLINE(modernize-use-std-numbers)
    const std::array<double, 2> double_data{3.14159, 2.71828};
    const std::span<const double> span_double(double_data);
    RT_LOG_DEBUG("Span of double: {}", span_double);

    // 8. Component logging with span
    RT_LOGC_INFO(SystemComponent::Core, "Tensor dimensions: {}", span_from_array);

    fl::Logger::flush();

    // Verify file exists
    EXPECT_TRUE(std::filesystem::exists(actual_log_file));

    // Verify span formatting for different source types
    EXPECT_TRUE(fl::file_contains(actual_log_file, "[16, 3, 224, 224]"));   // std::array
    EXPECT_TRUE(fl::file_contains(actual_log_file, "[1024, 768, 512]"));    // std::vector
    EXPECT_TRUE(fl::file_contains(actual_log_file, "[8, 224, 224]"));       // C array
    EXPECT_TRUE(fl::file_contains(actual_log_file, "[1, 2.5, 3.14, 4.2]")); // float pointer
    EXPECT_TRUE(fl::file_contains(actual_log_file, "[1, 3, 224, 224]"));    // subspan
    EXPECT_TRUE(fl::file_contains(actual_log_file, "[]"));                  // empty span
    EXPECT_TRUE(fl::file_contains(actual_log_file, "[100, 200, 300]"));     // uint32_t
    EXPECT_TRUE(fl::file_contains(actual_log_file, "[3.14159, 2.71828]"));  // double
}

} // namespace
