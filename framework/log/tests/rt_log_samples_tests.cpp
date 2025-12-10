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
 * @file rt_log_samples_tests.cpp
 * @brief Documentation sample tests for the rt_log library
 *
 * This file contains tests that serve as executable documentation examples.
 * Code snippets from these tests are extracted and included in the Sphinx documentation.
 */

#include <array>         // for array
#include <chrono>        // for nanoseconds, milliseconds
#include <cstddef>       // for ptrdiff_t, size_t
#include <cstdint>       // for uint64_t, uint16_t
#include <filesystem>    // for exists, path
#include <functional>    // for bit_xor
#include <map>           // for map
#include <numeric>       // for accumulate
#include <optional>      // for optional
#include <span>          // for span
#include <stdexcept>     // for invalid_argument
#include <string>        // for allocator, string, to_string
#include <unordered_map> // for unordered_map
#include <utility>       // for pair, move
#include <vector>        // for vector

#include <quill/Frontend.h>             // for Frontend
#include <quill/LogMacros.h>            // for QUILL_LOG_INFO, QUILL_LOG_DEBUG
#include <quill/core/FrontendOptions.h> // for FrontendOptions
#include <quill/sinks/FileSink.h>       // for FileSink, FileSinkConfig
#include <wise_enum_detail.h>           // for WISE_ENUM_IMPL_IIF_0
#include <wise_enum_generated.h>        // for WISE_ENUM_IMPL_LOOP_12, WISE_ENUM_I...

#include <gtest/gtest.h> // for AssertionResult, Message, TestPartR...

#include "log/components.hpp"    // for LogLevel, get_component_level, regi...
#include "log/rt_log.hpp"        // for Logger, LoggerConfig, SinkType
#include "log/rt_log_macros.hpp" // for RT_LOG_INFO, RT_LOGE_ERROR, RT_LOGC...
#include "temp_file.hpp"         // for file_contains, TempFileManager
// Custom types for documentation - must be at global scope for RT_LOGGABLE_* macros
// example-begin performance-deferred-direct-formatting-1
struct SafeType {
    int value;
};
RT_LOGGABLE_DEFERRED_FORMAT(SafeType, "value: {}", obj.value);

struct UnsafeType {
    int *ptr;
};
RT_LOGGABLE_DIRECT_FORMAT(UnsafeType, "ptr: {}", obj.ptr ? *obj.ptr : 0);
// example-end performance-deferred-direct-formatting-1

// Advanced custom type examples for documentation
// example-begin custom-type-user-struct-1
struct DocUser {
    std::string name;
    uint64_t *session_ptr; // Contains pointer - not safe for async
    std::vector<std::string> roles;
    std::chrono::time_point<std::chrono::system_clock> last_login;
};

// For types with pointers/references (immediate formatting)
RT_LOGGABLE_DIRECT_FORMAT(
        DocUser,
        "name: '{}', session_id: {}, roles: {}, last_login: {}",
        obj.name,
        obj.session_ptr ? *obj.session_ptr : 0,           // Conditional pointer access
        obj.roles.size(),                                 // Function call
        std::chrono::duration_cast<std::chrono::seconds>( // Complex computation
                obj.last_login.time_since_epoch())
                .count());
// example-end custom-type-user-struct-1

// example-begin custom-type-network-buffer-1
struct NetworkConnection {
    std::string host;
    uint16_t port;
    bool is_encrypted;
    std::optional<std::string> proxy;
    std::chrono::milliseconds latency;
};

RT_LOGGABLE_DEFERRED_FORMAT(
        NetworkConnection,
        "{}://{}:{} (latency: {}ms, proxy: {})",
        obj.is_encrypted ? "https" : "http", // Conditional protocol
        obj.host,
        obj.port,
        obj.latency.count(),         // Method call
        obj.proxy.value_or("none")); // Optional handling

struct DataBuffer {
    std::vector<uint8_t> data;
    std::size_t valid_bytes;
    bool is_compressed;
};

RT_LOGGABLE_DEFERRED_FORMAT(
        DataBuffer,
        "size: {}/{} bytes, compressed: {}, checksum: 0x{:x}",
        obj.valid_bytes,
        obj.data.size(),
        obj.is_compressed,
        std::accumulate(
                obj.data.begin(), // Complex computation
                obj.data.begin() + static_cast<std::ptrdiff_t>(obj.valid_bytes),
                0u,
                std::bit_xor<uint8_t>{})); // Custom checksum
// example-end custom-type-network-buffer-1

// example-begin custom-type-http-request-1
struct HttpRequest {
    std::string method;
    std::string url;
    std::map<std::string, std::string> headers;
    std::vector<uint8_t> body;
    std::chrono::steady_clock::time_point received_at;
    std::optional<std::string> user_agent;
};

RT_LOGGABLE_DEFERRED_FORMAT(
        HttpRequest,
        "{} {} | headers: {} | body: {} bytes | user-agent: {} | age: {}ms",
        obj.method,
        obj.url,
        obj.headers.size(),
        obj.body.size(),
        obj.user_agent.value_or("unknown"),
        std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - obj.received_at)
                .count());
// example-end custom-type-http-request-1

namespace {

namespace fl = ::framework::log;

DECLARE_LOG_COMPONENT(
        SystemComponent,
        CORE,
        CONFIG,
        NETWORK,
        DATABASE,
        SECURITY,
        PERFORMANCE,
        THREAD_POOL,
        MEMORY_MANAGER,
        FILE_SYSTEM,
        IPC,
        SCHEDULER,
        MONITOR);

DECLARE_LOG_EVENT(
        SystemEvent,
        APP_START,
        APP_STOP,
        CONFIG_LOADED,
        CONFIG_ERROR,
        SHUTDOWN_REQUEST,
        SHUTDOWN_COMPLETE,
        HEALTH_CHECK_OK,
        HEALTH_CHECK_FAIL,
        THREAD_START,
        THREAD_STOP,
        RESOURCE_ALLOC,
        RESOURCE_FREE,
        TIMEOUT,
        RETRY,
        CRITICAL_ERROR);

DECLARE_LOG_EVENT(
        ErrorEvent,
        INVALID_PARAM,
        OUT_OF_MEMORY,
        FILE_NOT_FOUND,
        PERMISSION_DENIED,
        NETWORK_ERROR,
        OPERATION_FAILED,
        OPERATION_TIMEOUT,
        INVALID_STATE,
        BUFFER_OVERFLOW,
        RESOURCE_EXHAUSTED,
        AUTHENTICATION_FAILED,
        CONFIGURATION_ERROR,
        PROTOCOL_ERROR,
        SERIALIZATION_ERROR,
        DESERIALIZATION_ERROR,
        CONNECTION_FAILED,
        RESOURCE_UNAVAILABLE);

// Simple components for documentation examples
// example-begin simple-component-event-1
// Define application components
DECLARE_LOG_COMPONENT(MyComponent, CORE, NETWORK, DATABASE, UI);

// Define system events
DECLARE_LOG_EVENT(MyEvent, APP_START, CONFIG_LOADED, USER_ACTION);
// example-end simple-component-event-1

// Test: Demonstrates simple logger configuration patterns for documentation
TEST(RTLog, SimpleLoggerConfiguration) {
    // example-begin simple-logger-configuration-1
    using namespace framework::log;

    // Console logging with INFO level
    Logger::configure(LoggerConfig::console(LogLevel::INFO));

    // Or file logging
    Logger::configure(LoggerConfig::file("app.log", LogLevel::DEBUG));
    // example-end simple-logger-configuration-1

    // Verify the configuration worked
    EXPECT_EQ(fl::Logger::get_sink_type(), fl::SinkType::File);
}

// Test: Demonstrates simple component registration patterns for documentation
TEST(RTLog, SimpleComponentRegistration) {
    fl::TempFileManager temp_manager{"simple_registration"};
    const std::string log_file = temp_manager.get_temp_file(".log");
    fl::Logger::configure(fl::LoggerConfig::file(log_file.c_str(), fl::LogLevel::DEBUG));

    // example-begin simple-component-registration-1
    using namespace framework::log;

    // Set all components to DEBUG level
    register_component<MyComponent>(LogLevel::DEBUG);

    // Or set individual component levels
    register_component<MyComponent>(
            {{MyComponent::CORE, LogLevel::INFO},
             {MyComponent::NETWORK, LogLevel::DEBUG},
             {MyComponent::DATABASE, LogLevel::WARN}});
    // example-end simple-component-registration-1

    // Verify the registration worked by checking component levels
    EXPECT_EQ(fl::get_component_level<MyComponent>(MyComponent::CORE), fl::LogLevel::INFO);
    EXPECT_EQ(fl::get_component_level<MyComponent>(MyComponent::NETWORK), fl::LogLevel::DEBUG);
    EXPECT_EQ(fl::get_component_level<MyComponent>(MyComponent::DATABASE), fl::LogLevel::WARN);
}

// Test: Demonstrates basic logging patterns for Quick Start documentation
TEST(RTLog, SimpleStartLogging) {
    fl::TempFileManager temp_manager{"simple_start_logging"};
    const std::string log_file = temp_manager.get_temp_file(".log");
    fl::Logger::configure(fl::LoggerConfig::file(log_file.c_str(), fl::LogLevel::DEBUG));
    fl::register_component<MyComponent>(fl::LogLevel::DEBUG);

    // example-begin simple-start-logging-1
    // Basic logging
    RT_LOG_INFO("Application started");

    // Component logging
    RT_LOGC_INFO(MyComponent::CORE, "Core system initialized");

    // Event logging
    RT_LOGE_INFO(MyEvent::APP_START, "Application startup completed");

    // Combined component and event logging
    RT_LOGEC_INFO(MyComponent::CORE, MyEvent::CONFIG_LOADED, "Configuration loaded successfully");
    // example-end simple-start-logging-1

    fl::Logger::flush();

    // Verify messages were logged
    const std::string actual_log_file = fl::Logger::get_actual_log_file();
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Application started"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Core system initialized"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Application startup completed"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Configuration loaded successfully"));
}

// Test: Demonstrates runtime component level management for documentation
TEST(RTLog, SimpleRuntimeComponentLevel) {
    fl::TempFileManager temp_manager{"simple_runtime_level"};
    const std::string log_file = temp_manager.get_temp_file(".log");
    fl::Logger::configure(fl::LoggerConfig::file(log_file.c_str(), fl::LogLevel::DEBUG));
    fl::register_component<MyComponent>(fl::LogLevel::INFO);

    // example-begin simple-runtime-component-level-1
    using namespace framework::log;

    // Query current component level
    LogLevel current_level = get_component_level<MyComponent>(MyComponent::NETWORK);

    // Dynamically change component levels
    register_component<MyComponent>(
            {{MyComponent::NETWORK, LogLevel::TRACE_L1}}); // Increase verbosity for debugging
    // example-end simple-runtime-component-level-1

    // Verify the level was changed
    (void)current_level; // Suppress unused variable warning
    EXPECT_EQ(fl::get_component_level<MyComponent>(MyComponent::NETWORK), fl::LogLevel::TRACE_L1);
}

// Test: Demonstrates log level filtering for documentation
TEST(RTLog, SimpleLogLevelFiltering) {
    fl::TempFileManager temp_manager{"simple_filtering"};
    const std::string log_file = temp_manager.get_temp_file(".log");

    // example-begin simple-log-level-filtering-1
    using namespace framework::log;

    // Configure logger at INFO level
    Logger::configure(LoggerConfig::file("app.log", LogLevel::INFO));

    RT_LOG_DEBUG("This won't appear"); // Below INFO level
    RT_LOG_INFO("This will appear");   // INFO level
    RT_LOG_WARN("This will appear");   // Above INFO level
    // example-end simple-log-level-filtering-1

    fl::Logger::flush();

    // Verify filtering behavior
    const std::string actual_log_file = fl::Logger::get_actual_log_file();
    EXPECT_FALSE(fl::file_contains(actual_log_file, "This won't appear"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "This will appear"));
}

// Test: Demonstrates standard logging macros for documentation
TEST(RTLog, SimpleStandardLogging) {
    fl::TempFileManager temp_manager{"simple_standard_logging"};
    const std::string log_file = temp_manager.get_temp_file(".log");
    fl::Logger::configure(fl::LoggerConfig::file(log_file.c_str(), fl::LogLevel::TRACE_L3));

    // example-begin simple-standard-logging-1
    // Basic logging without components or events
    RT_LOG_TRACE_L3("Entering function with param: {}", 42);
    RT_LOG_TRACE_L2("Processing step {} of {}", 1, 10);
    RT_LOG_TRACE_L1("Intermediate result: {}", "success");
    RT_LOG_DEBUG("Debug information: {}", "debugging");
    RT_LOG_INFO("Operation completed successfully");
    RT_LOG_NOTICE("Important milestone reached");
    RT_LOG_WARN("Deprecated function called: {}", "oldFunc");
    RT_LOG_ERROR("Operation failed: {}", "connection timeout");
    RT_LOG_CRITICAL("System in critical state: {}", "low memory");
    // example-end simple-standard-logging-1

    fl::Logger::flush();

    // Verify all levels were logged
    const std::string actual_log_file = fl::Logger::get_actual_log_file();
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Entering function with param"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Operation completed successfully"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "System in critical state"));
}

// Test: Demonstrates format string support for documentation
TEST(RTLog, SimpleFormatStrings) {
    fl::TempFileManager temp_manager{"simple_format_strings"};
    const std::string log_file = temp_manager.get_temp_file(".log");
    fl::Logger::configure(fl::LoggerConfig::file(log_file.c_str(), fl::LogLevel::INFO));

    // example-begin simple-format-strings-1
    RT_LOG_INFO("User {} has {} credits remaining", "alice", 100);
    RT_LOG_WARN("Memory usage at {:.1f}%", 85.7);
    RT_LOG_ERROR("Failed after {} attempts in {:.2f}s", 3, 1.25);
    // example-end simple-format-strings-1

    fl::Logger::flush();

    // Verify format strings worked
    const std::string actual_log_file = fl::Logger::get_actual_log_file();
    EXPECT_TRUE(fl::file_contains(actual_log_file, "User alice has 100 credits remaining"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Memory usage at 85.7%"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Failed after 3 attempts in 1.25s"));
}

// Test: Demonstrates JSON logging for documentation
TEST(RTLog, SimpleJsonLogging) {
    fl::TempFileManager temp_manager{"simple_json_logging"};
    const std::string log_file = temp_manager.get_temp_file(".log");
    fl::Logger::configure(fl::LoggerConfig::json_file(log_file.c_str(), fl::LogLevel::INFO));
    fl::register_component<MyComponent>(fl::LogLevel::INFO);

    // example-begin simple-json-logging-1
    // Basic JSON logging
    RT_LOGJ_INFO("user_id", 12345, "action", "login", "timestamp", 1234567890);

    // Component JSON logging with component as a key
    RT_LOGJ_INFO(
            "component", "CORE", "event", "auth_success", "user", "alice", "ip", "192.168.1.1");
    // example-end simple-json-logging-1

    fl::Logger::flush();

    // Verify JSON format in output
    const std::string actual_log_file = fl::Logger::get_actual_log_file();
    EXPECT_TRUE(fl::file_contains(actual_log_file, "user_id"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "12345"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "auth_success"));
}

// Detailed component example for documentation
// example-begin detailed-component-declaration-1
DECLARE_LOG_COMPONENT(
        AppComponent,
        CORE,        // Core functionality
        NETWORK,     // Network operations
        DATABASE,    // Database interactions
        SECURITY,    // Security/authentication
        UI,          // User interface
        FILE_SYSTEM, // File operations
        PERFORMANCE  // Performance monitoring
);
// example-end detailed-component-declaration-1

// Additional components for event-based logging documentation
// example-begin event-examples-declarations-1
// System lifecycle events
DECLARE_LOG_EVENT(
        SystemEventDoc,
        APP_START,
        APP_STOP,
        CONFIG_LOADED,
        CONFIG_ERROR,
        SHUTDOWN_REQUEST,
        HEALTH_CHECK_OK,
        RESOURCE_ALLOC);

// Error events
DECLARE_LOG_EVENT(
        ErrorEventDoc,
        INVALID_PARAM,
        NETWORK_ERROR,
        DATABASE_ERROR,
        AUTHENTICATION_FAILED,
        OPERATION_TIMEOUT);
// example-end event-examples-declarations-1

// Test: Demonstrates detailed component registration for documentation
TEST(RTLog, DetailedComponentRegistration) {
    fl::TempFileManager temp_manager{"detailed_registration"};
    const std::string log_file = temp_manager.get_temp_file(".log");
    fl::Logger::configure(fl::LoggerConfig::file(log_file.c_str(), fl::LogLevel::DEBUG));

    // example-begin detailed-component-registration-1
    using namespace framework::log;

    // Option 1: Set same level for all components
    register_component<AppComponent>(LogLevel::INFO);

    // Option 2: Set individual levels
    register_component<AppComponent>(
            {{AppComponent::CORE, LogLevel::INFO},
             {AppComponent::NETWORK, LogLevel::DEBUG},
             {AppComponent::DATABASE, LogLevel::WARN},
             {AppComponent::SECURITY, LogLevel::ERROR}});
    // example-end detailed-component-registration-1

    // Verify registration
    EXPECT_EQ(fl::get_component_level<AppComponent>(AppComponent::CORE), fl::LogLevel::INFO);
    EXPECT_EQ(fl::get_component_level<AppComponent>(AppComponent::NETWORK), fl::LogLevel::DEBUG);
    EXPECT_EQ(fl::get_component_level<AppComponent>(AppComponent::DATABASE), fl::LogLevel::WARN);
}

// Test: Demonstrates component-based logging for documentation
TEST(RTLog, SimpleComponentLogging) {
    fl::TempFileManager temp_manager{"simple_component_logging"};
    const std::string log_file = temp_manager.get_temp_file(".log");
    fl::Logger::configure(fl::LoggerConfig::file(log_file.c_str(), fl::LogLevel::DEBUG));
    fl::register_component<MyComponent>(fl::LogLevel::DEBUG);

    // example-begin simple-component-logging-1
    // Log to specific components
    RT_LOGC_DEBUG(MyComponent::CORE, "Core subsystem initializing");
    RT_LOGC_INFO(MyComponent::NETWORK, "Connection established to server");
    RT_LOGC_WARN(MyComponent::DATABASE, "Query took longer than expected: {}ms", 500);
    RT_LOGC_ERROR(MyComponent::UI, "Failed to render component: {}", "NavBar");
    // example-end simple-component-logging-1

    fl::Logger::flush();

    // Verify component logging
    const std::string actual_log_file = fl::Logger::get_actual_log_file();
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Core subsystem initializing"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Connection established"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Query took longer"));
}

// Test: Demonstrates event-based logging for documentation
TEST(RTLog, SimpleEventLogging) {
    fl::TempFileManager temp_manager{"simple_event_logging"};
    const std::string log_file = temp_manager.get_temp_file(".log");
    fl::Logger::configure(fl::LoggerConfig::file(log_file.c_str(), fl::LogLevel::DEBUG));

    // example-begin simple-event-logging-1
    // Log specific events
    RT_LOGE_INFO(SystemEventDoc::APP_START, "Application version 1.0.0 starting");
    RT_LOGE_INFO(SystemEventDoc::CONFIG_LOADED, "Loaded configuration from config.yaml");
    RT_LOGE_WARN(ErrorEventDoc::NETWORK_ERROR, "Retrying connection, attempt {}", 2);
    RT_LOGE_ERROR(ErrorEventDoc::OPERATION_TIMEOUT, "Operation timed out after {}s", 30);
    // example-end simple-event-logging-1

    fl::Logger::flush();

    // Verify event logging
    const std::string actual_log_file = fl::Logger::get_actual_log_file();
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Application version 1.0.0 starting"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Loaded configuration"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Retrying connection"));
}

// Test: Demonstrates combined component and event logging for documentation
TEST(RTLog, SimpleCombinedLogging) {
    fl::TempFileManager temp_manager{"simple_combined_logging"};
    const std::string log_file = temp_manager.get_temp_file(".log");
    fl::Logger::configure(fl::LoggerConfig::file(log_file.c_str(), fl::LogLevel::DEBUG));
    fl::register_component<MyComponent>(fl::LogLevel::DEBUG);

    // example-begin simple-combined-logging-1
    // Combine component and event for maximum context
    RT_LOGEC_INFO(
            MyComponent::CORE, SystemEventDoc::APP_START, "Core module initialized successfully");
    RT_LOGEC_INFO(MyComponent::NETWORK, SystemEventDoc::CONFIG_LOADED, "Network settings applied");
    RT_LOGEC_WARN(
            MyComponent::DATABASE,
            ErrorEventDoc::OPERATION_TIMEOUT,
            "Database query timeout on table: {}",
            "users");
    RT_LOGEC_ERROR(
            MyComponent::CORE,
            ErrorEventDoc::AUTHENTICATION_FAILED,
            "Authentication failed for user: {}",
            "admin");
    // example-end simple-combined-logging-1

    fl::Logger::flush();

    // Verify combined logging
    const std::string actual_log_file = fl::Logger::get_actual_log_file();
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Core module initialized"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Network settings applied"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Database query timeout"));
}

// Test: Demonstrates console logging configuration for documentation
TEST(RTLog, ConsoleLoggingConfig) {
    // example-begin console-logging-config-1
    using namespace framework::log;

    // Basic console logging
    Logger::configure(LoggerConfig::console(LogLevel::INFO));

    // Console with colors enabled
    Logger::configure(LoggerConfig::console(LogLevel::DEBUG).with_colors(true));

    // Console with timestamps
    Logger::configure(LoggerConfig::console(LogLevel::INFO).with_timestamps(true));
    // example-end console-logging-config-1

    // Verify the last configuration
    EXPECT_EQ(fl::Logger::get_sink_type(), fl::SinkType::Console);
}

// Test: Demonstrates file logging configuration for documentation
TEST(RTLog, FileLoggingConfig) {
    fl::TempFileManager temp_manager{"file_logging_config"};
    const std::string log_file = temp_manager.get_temp_file(".log");

    // example-begin file-logging-config-1
    using namespace framework::log;

    // Basic file logging
    Logger::configure(LoggerConfig::file("app.log", LogLevel::DEBUG));

    // File logging with timestamps
    Logger::configure(LoggerConfig::file("app.log", LogLevel::INFO).with_timestamps(true));

    // File logging with file/line info
    Logger::configure(LoggerConfig::file("app.log", LogLevel::DEBUG).with_file_line(true));
    // example-end file-logging-config-1

    // Verify the configuration
    EXPECT_EQ(fl::Logger::get_sink_type(), fl::SinkType::File);
}

// Test: Demonstrates rotating file logging for documentation
TEST(RTLog, RotatingFileLogging) {
    fl::TempFileManager temp_manager{"rotating_file_logging"};
    const std::string log_file = temp_manager.get_temp_file(".log");

    // example-begin rotating-file-logging-1
    using namespace framework::log;

    // Rotating files (good for long-running applications)
    Logger::configure(LoggerConfig::rotating_file("app.log", LogLevel::INFO)
                              .with_timestamps(true)
                              .with_file_line(false));
    // example-end rotating-file-logging-1

    // Verify the configuration
    EXPECT_EQ(fl::Logger::get_sink_type(), fl::SinkType::RotatingFile);
}

// Test: Demonstrates log flushing for documentation
TEST(RTLog, LogFlushing) {
    fl::TempFileManager temp_manager{"log_flushing"};
    const std::string log_file = temp_manager.get_temp_file(".log");
    fl::Logger::configure(fl::LoggerConfig::file(log_file.c_str(), fl::LogLevel::INFO));

    RT_LOG_INFO("Test message");

    // example-begin log-flushing-1
    using namespace framework::log;

    // Flush all pending log messages
    Logger::flush();

    // Get actual log file path (useful for rotating logs)
    std::string actual_file = Logger::get_actual_log_file();
    // example-end log-flushing-1

    // Verify the flush worked
    EXPECT_FALSE(actual_file.empty());
    EXPECT_TRUE(std::filesystem::exists(actual_file));
}

// Test: Demonstrates component level management best practices
TEST(RTLog, BestPracticeComponentLevelManagement) {
    fl::TempFileManager temp_manager{"bp_component_level"};
    const std::string log_file = temp_manager.get_temp_file(".log");
    fl::Logger::configure(fl::LoggerConfig::file(log_file.c_str(), fl::LogLevel::TRACE_L3));

    // example-begin best-practice-component-level-1
    using namespace framework::log;

    // Production: Conservative levels
    register_component<MyComponent>(
            {{MyComponent::CORE, LogLevel::INFO},
             {MyComponent::DATABASE, LogLevel::WARN},
             {MyComponent::UI, LogLevel::NOTICE}});

    // Development: Verbose levels
    register_component<MyComponent>(LogLevel::DEBUG);

    // Debugging specific issues
    register_component<MyComponent>({
            {MyComponent::NETWORK, LogLevel::TRACE_L1} // Focus on network issues
    });
    // example-end best-practice-component-level-1

    // Verify the configuration
    EXPECT_EQ(fl::get_component_level<MyComponent>(MyComponent::NETWORK), fl::LogLevel::TRACE_L1);
}

// Test: Demonstrates meaningful log messages best practices
TEST(RTLog, BestPracticeMeaningfulMessages) {
    fl::TempFileManager temp_manager{"bp_meaningful_messages"};
    const std::string log_file = temp_manager.get_temp_file(".log");
    fl::Logger::configure(fl::LoggerConfig::file(log_file.c_str(), fl::LogLevel::DEBUG));
    fl::register_component<MyComponent>(fl::LogLevel::DEBUG);

    const int retry_count = 3;
    const std::string host = "db.example.com";
    const int port = 5432;
    const std::string key = "timeout_ms";
    const int default_value = 5000;

    // example-begin best-practice-meaningful-messages-1
    // Good: Informative and actionable
    RT_LOGC_ERROR(
            MyComponent::DATABASE,
            "Connection failed after {} retries to {}:{}",
            retry_count,
            host,
            port);

    RT_LOGE_WARN(
            MyEvent::CONFIG_LOADED,
            "Missing config key '{}', using default: {}",
            key,
            default_value);

    // Avoid: Vague or uninformative
    // RT_LOG_ERROR("Something went wrong");  // What? Where? Why?
    // RT_LOG_INFO("Done");                   // Done with what?
    // example-end best-practice-meaningful-messages-1

    fl::Logger::flush();

    // Verify good messages were logged
    const std::string actual_log_file = fl::Logger::get_actual_log_file();
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Connection failed after 3 retries"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Missing config key"));
}

// Test: Demonstrates performance-critical path logging
TEST(RTLog, BestPracticePerformanceCriticalPaths) {
    fl::TempFileManager temp_manager{"bp_performance"};
    const std::string log_file = temp_manager.get_temp_file(".log");
    fl::Logger::configure(fl::LoggerConfig::file(log_file.c_str(), fl::LogLevel::INFO));
    fl::register_component<MyComponent>(fl::LogLevel::DEBUG);

    // example-begin best-practice-performance-critical-1
    using namespace framework::log;

    // Check log level before expensive operations
    if (ComponentLevelStorage<MyComponent>::should_log(MyComponent::CORE, LogLevel::DEBUG)) {
        std::string expensive_debug_info = "Computed expensive data";
        RT_LOGC_DEBUG(MyComponent::CORE, "Debug info: {}", expensive_debug_info);
    }
    // example-end best-practice-performance-critical-1

    fl::Logger::flush();

    // Since we configured at INFO level, DEBUG shouldn't appear
    const std::string actual_log_file = fl::Logger::get_actual_log_file();
    EXPECT_FALSE(fl::file_contains(actual_log_file, "Debug info"));
}

// Test: Demonstrates error context best practices
TEST(RTLog, BestPracticeErrorContext) {
    fl::TempFileManager temp_manager{"bp_error_context"};
    const std::string log_file = temp_manager.get_temp_file(".log");
    fl::Logger::configure(fl::LoggerConfig::file(log_file.c_str(), fl::LogLevel::ERROR));
    fl::register_component<MyComponent>(fl::LogLevel::ERROR);

    const std::string host = "api.example.com";
    const int port = 443;
    const int elapsed_ms = 5000;
    const int attempt = 3;
    const int max_attempts = 5;

    // example-begin best-practice-error-context-1
    // Include relevant context in error messages
    RT_LOGEC_ERROR(
            MyComponent::NETWORK,
            ErrorEvent::NETWORK_ERROR,
            "Failed to connect to {}:{} after {}ms (attempt {} of {})",
            host,
            port,
            elapsed_ms,
            attempt,
            max_attempts);
    // example-end best-practice-error-context-1

    fl::Logger::flush();

    // Verify comprehensive error context
    const std::string actual_log_file = fl::Logger::get_actual_log_file();
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Failed to connect to api.example.com:443"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "attempt 3 of 5"));
}

// Test: Demonstrates component level filtering optimization
TEST(RTLog, PerformanceComponentLevelFiltering) {
    fl::TempFileManager temp_manager{"perf_filtering"};
    const std::string log_file = temp_manager.get_temp_file(".log");
    fl::Logger::configure(fl::LoggerConfig::file(log_file.c_str(), fl::LogLevel::INFO));
    fl::register_component<MyComponent>(fl::LogLevel::INFO);

    // example-begin performance-component-filtering-1
    using namespace framework::log;

    // This check is very fast (direct array access)
    if (ComponentLevelStorage<MyComponent>::should_log(MyComponent::CORE, LogLevel::INFO)) {
        // Expensive formatting only happens if needed
        RT_LOGC_INFO(MyComponent::CORE, "This will be logged");
    }
    // example-end performance-component-filtering-1

    fl::Logger::flush();

    // Verify the conditional logging worked
    const std::string actual_log_file = fl::Logger::get_actual_log_file();
    EXPECT_TRUE(fl::file_contains(actual_log_file, "This will be logged"));
}

// Test: Demonstrates deferred vs direct formatting performance
TEST(RTLog, PerformanceDeferredVsDirectFormatting) {
    fl::TempFileManager temp_manager{"perf_formatting"};
    const std::string log_file = temp_manager.get_temp_file(".log");
    fl::Logger::configure(fl::LoggerConfig::file(log_file.c_str(), fl::LogLevel::INFO));

    SafeType safe_obj{42};
    int value = 100;
    UnsafeType unsafe_obj{&value};

    RT_LOG_INFO("SafeType: {}", safe_obj);
    RT_LOG_INFO("UnsafeType: {}", unsafe_obj);

    fl::Logger::flush();

    // Verify both types logged correctly
    const std::string actual_log_file = fl::Logger::get_actual_log_file();
    EXPECT_TRUE(fl::file_contains(actual_log_file, "SafeType"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "UnsafeType"));
}

// Test: Demonstrates custom type with pointer (User struct)
TEST(RTLog, CustomTypeUserStruct) {
    fl::TempFileManager temp_manager{"custom_type_user"};
    const std::string log_file = temp_manager.get_temp_file(".log");
    fl::Logger::configure(fl::LoggerConfig::file(log_file.c_str(), fl::LogLevel::INFO));

    uint64_t session_id = 123456789;
    DocUser user{"alice", &session_id, {"admin", "user"}, std::chrono::system_clock::now()};

    RT_LOG_INFO("User logged in: {}", user);
    fl::Logger::flush();

    const std::string actual_log_file = fl::Logger::get_actual_log_file();
    EXPECT_TRUE(fl::file_contains(actual_log_file, "alice"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "123456789"));
}

// Test: Demonstrates advanced custom types (NetworkConnection and DataBuffer)
TEST(RTLog, CustomTypeAdvancedExamples) {
    fl::TempFileManager temp_manager{"custom_type_advanced"};
    const std::string log_file = temp_manager.get_temp_file(".log");
    fl::Logger::configure(fl::LoggerConfig::file(log_file.c_str(), fl::LogLevel::INFO));

    NetworkConnection conn{
            "api.example.com", 443, true, "proxy.corp.com", std::chrono::milliseconds(125)};
    RT_LOG_INFO("Connection: {}", conn);

    DataBuffer buffer{{0x01, 0x02, 0x03, 0x04, 0x05}, 5, true};
    RT_LOG_INFO("Buffer: {}", buffer);

    fl::Logger::flush();

    const std::string actual_log_file = fl::Logger::get_actual_log_file();
    EXPECT_TRUE(fl::file_contains(actual_log_file, "https://api.example.com:443"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "proxy.corp.com"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "5/5 bytes"));
}

// Test: Demonstrates complex HttpRequest custom type
TEST(RTLog, CustomTypeHttpRequest) {
    fl::TempFileManager temp_manager{"custom_type_http"};
    const std::string log_file = temp_manager.get_temp_file(".log");
    fl::Logger::configure(fl::LoggerConfig::file(log_file.c_str(), fl::LogLevel::INFO));

    HttpRequest req{
            "GET",
            "/api/users/123",
            {{"Authorization", "Bearer token"}, {"Content-Type", "application/json"}},
            {},
            std::chrono::steady_clock::now(),
            "Mozilla/5.0"};

    RT_LOG_INFO("HTTP Request: {}", req);
    fl::Logger::flush();

    const std::string actual_log_file = fl::Logger::get_actual_log_file();
    EXPECT_TRUE(fl::file_contains(actual_log_file, "GET /api/users/123"));
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Mozilla/5.0"));
}

// Test: Demonstrates common pitfalls to avoid
TEST(RTLog, AvoidingCommonPitfalls) {
    fl::TempFileManager temp_manager{"avoid_pitfalls"};
    const std::string log_file = temp_manager.get_temp_file(".log");
    fl::Logger::configure(fl::LoggerConfig::file(log_file.c_str(), fl::LogLevel::INFO));
    fl::register_component<MyComponent>(fl::LogLevel::INFO);

    const int item_id = 12345;
    const int bytes_sent = 1024;

    // example-begin avoiding-common-pitfalls-1
    // Good: Let the logger handle formatting
    RT_LOG_INFO("Processing item {}", item_id);

    // Avoid: Pre-formatting strings unnecessarily
    // std::string msg = std::format("Processing item {}", item_id);  // Waste!
    // RT_LOG_INFO("{}", msg);

    // Good: Use direct values
    RT_LOGC_DEBUG(MyComponent::NETWORK, "Bytes sent: {}", bytes_sent);

    // Avoid: Expensive conversions
    // RT_LOGC_DEBUG(MyComponent::NETWORK, "Bytes sent: {}", std::to_string(bytes_sent));
    // example-end avoiding-common-pitfalls-1

    fl::Logger::flush();

    // Verify good patterns were used
    const std::string actual_log_file = fl::Logger::get_actual_log_file();
    EXPECT_TRUE(fl::file_contains(actual_log_file, "Processing item 12345"));
}

// Test: Demonstrates creating custom loggers for documentation
TEST(RTLog, CreatingCustomLoggers) {
    fl::TempFileManager temp_manager{"creating_custom_loggers"};
    const std::string log_file = temp_manager.get_temp_file(".log");

    // example-begin creating-custom-loggers-1
    // #include <quill/Frontend.h>
    // #include <quill/sinks/FileSink.h>

    // Create custom logger
    auto file_sink = quill::Frontend::create_or_get_sink<quill::FileSink>("custom.log");
    auto *custom_logger =
            quill::Frontend::create_or_get_logger("custom_logger", std::move(file_sink));
    // example-end creating-custom-loggers-1

    // Verify the logger was created
    EXPECT_NE(custom_logger, nullptr);
}

// Test: Demonstrates custom logger macros for documentation
TEST(RTLog, CustomLoggerMacros) {
    fl::TempFileManager temp_manager{"custom_logger_macros"};
    const std::string log_file = temp_manager.get_temp_file(".log");

    // Create a custom logger using Quill directly
    auto file_sink = quill::Frontend::create_or_get_sink<quill::FileSink>(log_file.c_str(), []() {
        quill::FileSinkConfig cfg{};
        return cfg;
    }());
    auto *custom_logger =
            quill::Frontend::create_or_get_logger("custom_logger", std::move(file_sink));

    fl::register_component<AppComponent>(fl::LogLevel::DEBUG);

    // example-begin custom-logger-macros-1
    // Basic logging with custom logger
    RT_LOGGER_INFO(custom_logger, "Message to custom logger");
    RT_LOGGER_ERROR(custom_logger, "Error in custom logger: {}", 404);

    // Component logging with custom logger
    RT_LOGGERC_DEBUG(custom_logger, AppComponent::NETWORK, "Network debug info");

    // Event logging with custom logger
    RT_LOGGERE_WARN(custom_logger, ErrorEvent::OPERATION_TIMEOUT, "Operation timed out");

    // Combined component and event with custom logger
    RT_LOGGEREC_ERROR(
            custom_logger,
            AppComponent::DATABASE,
            ErrorEvent::NETWORK_ERROR,
            "Database connection failed");

    // JSON logging with custom logger
    RT_LOGGERJ_INFO(custom_logger, "event", "custom_event", "data", 12345);
    // example-end custom-logger-macros-1

    fl::Logger::flush();

    // Verify messages were logged
    EXPECT_TRUE(fl::file_contains(log_file, "Message to custom logger"));
    EXPECT_TRUE(fl::file_contains(log_file, "Error in custom logger"));
    EXPECT_TRUE(fl::file_contains(log_file, "Network debug info"));
}

} // namespace
