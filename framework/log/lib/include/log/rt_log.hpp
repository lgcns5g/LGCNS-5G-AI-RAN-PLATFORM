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

#ifndef FRAMEWORK_LOG_RT_LOG_HPP
#define FRAMEWORK_LOG_RT_LOG_HPP

#include <array>
#include <filesystem>
#include <limits>
#include <memory>
#include <string_view>
#include <unordered_map>

// Disable Quill's non-prefixed macros to avoid conflicts
#define QUILL_DISABLE_NON_PREFIXED_MACROS

#include <quill/Backend.h>
#include <quill/DeferredFormatCodec.h>
#include <quill/DirectFormatCodec.h>
#include <quill/Frontend.h>
#include <quill/HelperMacros.h>
#include <quill/LogMacros.h>
#include <quill/Logger.h>
#include <quill/sinks/ConsoleSink.h>
#include <quill/sinks/FileSink.h>
#include <quill/sinks/JsonSink.h>
#include <quill/sinks/RotatingFileSink.h>
#include <quill/sinks/RotatingJsonFileSink.h>
#include <quill/std/Pair.h>
#include <quill/std/Vector.h>

#include <wise_enum.h>

// Include component and event system
#include "log/components.hpp"

namespace framework::log {

/**
 * Custom frontend options optimized for real-time logging
 *
 * Configures the Quill frontend with settings optimized for high-throughput
 * logging scenarios with bounded dropping queues and larger initial capacity.
 */
struct RealTimeFrontendOptions final {
    static constexpr quill::QueueType queue_type =     // NOLINT(readability-identifier-naming)
            quill::QueueType::BoundedDropping;         //!< Use bounded dropping queue for
                                                       //!< high performance
    static constexpr uint32_t initial_queue_capacity = // NOLINT(readability-identifier-naming)
            8 * 1024 * 1024;                           //!< 8 MB initial capacity (vs default 128KB)
    static constexpr uint32_t
            blocking_queue_retry_interval_ns = // NOLINT(readability-identifier-naming)
            0;                                 //!< No retry interval for blocking operations
    static constexpr size_t unbounded_queue_max_capacity = // NOLINT(readability-identifier-naming)
            0;                                             //!< No unbounded queue limit
    static constexpr quill::HugePagesPolicy
            huge_pages_policy =            // NOLINT(readability-identifier-naming)
            quill::HugePagesPolicy::Never; //!< Disable huge pages for compatibility
};

/**
 * Real-time frontend implementation alias
 *
 * Specialized frontend implementation using the real-time options
 * for optimal logging throughput in production environments.
 */
using RealTimeFrontend = quill::FrontendImpl<RealTimeFrontendOptions>;

/**
 * Real-time logger implementation alias
 *
 * Specialized logger implementation using the real-time frontend
 * options for optimal logging performance.
 */
using RealTimeLogger = quill::LoggerImpl<RealTimeFrontendOptions>;

/**
 * Supported sink types for log output destinations
 *
 * Defines the various output destinations available for log messages,
 * including console, file-based, and JSON format options.
 */
enum class SinkType {
    Console,         //!< Console output
    File,            //!< File output
    RotatingFile,    //!< Rotating file output
    JsonFile,        //!< JSON file output
    JsonConsole,     //!< JSON console output
    RotatingJsonFile //!< Rotating JSON file output
};

} // namespace framework::log

// WISE_ENUM_ADAPT must be called at global namespace scope for ADL to work
WISE_ENUM_ADAPT(
        framework::log::SinkType,
        Console,
        File,
        RotatingFile,
        JsonFile,
        JsonConsole,
        RotatingJsonFile)

namespace framework::log {

/**
 * Configuration structure for logger initialization
 *
 * Contains all configurable options for setting up the logging system,
 * including sink type, output formatting, performance tuning, and
 * backend thread configuration.
 */
struct LoggerConfig final {
    SinkType sink_type{SinkType::Console}; //!< Type of log output sink to use
    const char *log_file{nullptr};         //!< Path to log file (for file-based sinks)
    LogLevel min_level{LogLevel::Info};    //!< Minimum log level to process
    bool enable_colors{true};              //!< Enable color output for console sinks
    bool enable_file_line{true};           //!< Include file and line number in log output
    bool enable_caller{false};             //!< Include calling function name in log output
    bool enable_timestamps{true};          //!< Include timestamps in log output
    bool enable_log_level{true};           //!< Include log level in log output
    bool enable_thread_name{true};         //!< Include thread name in log output
    static constexpr int DEFAULT_BACKEND_SLEEP_NS =
            100; //!< Default backend thread sleep duration in nanoseconds
    std::chrono::nanoseconds backend_sleep_duration{
            std::chrono::nanoseconds{DEFAULT_BACKEND_SLEEP_NS}}; //!< Backend thread sleep duration
                                                                 //!< for performance tuning
    uint16_t backend_cpu_affinity{std::numeric_limits<uint16_t>::max()}; //!< CPU affinity for
                                                                         //!< backend thread

    /**
     * Create console logger configuration (default)
     *
     * @param[in] level Minimum log level to process
     * @param[in] colors Enable color output
     * @return LoggerConfig configured for console output
     */
    static LoggerConfig console(LogLevel level = LogLevel::Info, bool colors = true);

    /**
     * Create file logger configuration
     *
     * @param[in] path Path to the log file
     * @param[in] level Minimum log level to process
     * @return LoggerConfig configured for file output
     */
    static LoggerConfig file(const char *path, LogLevel level = LogLevel::Info);

    /**
     * Create rotating file logger configuration
     *
     * @param[in] path Path to the log file
     * @param[in] level Minimum log level to process
     * @return LoggerConfig configured for rotating file output
     */
    static LoggerConfig rotating_file(const char *path, LogLevel level = LogLevel::Info);

    /**
     * Create JSON file logger configuration
     *
     * @param[in] path Path to the JSON log file
     * @param[in] level Minimum log level to process
     * @return LoggerConfig configured for JSON file output
     */
    static LoggerConfig json_file(const char *path, LogLevel level = LogLevel::Info);

    /**
     * Create JSON console logger configuration
     *
     * @param[in] level Minimum log level to process
     * @return LoggerConfig configured for JSON console output
     */
    static LoggerConfig json_console(LogLevel level = LogLevel::Info);

    /**
     * Create rotating JSON file logger configuration
     *
     * @param[in] path Path to the rotating JSON log file
     * @param[in] level Minimum log level to process
     * @return LoggerConfig configured for rotating JSON file output
     */
    static LoggerConfig rotating_json_file(const char *path, LogLevel level = LogLevel::Info);

    /**
     * Enable file and line information in logs
     *
     * @param[in] enable Whether to enable file and line information
     * @return Reference to this config for method chaining
     */
    LoggerConfig &with_file_line(bool enable = true);

    /**
     * Enable caller function information in logs
     *
     * @param[in] enable Whether to enable caller function information
     * @return Reference to this config for method chaining
     */
    LoggerConfig &with_caller(bool enable = true);

    /**
     * Enable or disable timestamps in log output
     *
     * @param[in] enable Whether to enable timestamps
     * @return Reference to this config for method chaining
     */
    LoggerConfig &with_timestamps(bool enable = true);

    /**
     * Enable or disable log level display in log output
     *
     * @param[in] enable Whether to enable log level display
     * @return Reference to this config for method chaining
     */
    LoggerConfig &with_log_level(bool enable = true);

    /**
     * Enable or disable thread name display in log output
     *
     * @param[in] enable Whether to enable thread name display
     * @return Reference to this config for method chaining
     */
    LoggerConfig &with_thread_name(bool enable = true);

    /**
     * Enable or disable colors in console output
     *
     * @param[in] enable Whether to enable color output
     * @return Reference to this config for method chaining
     */
    LoggerConfig &with_colors(bool enable = true);

    /**
     * Set backend thread sleep duration for performance tuning
     *
     * @param[in] duration Sleep duration for the backend thread
     * @return Reference to this config for method chaining
     */
    LoggerConfig &with_backend_sleep_duration(std::chrono::nanoseconds duration);

    /**
     * Set CPU affinity for backend thread
     *
     * @param[in] cpu CPU core to bind the backend thread to
     * @return Reference to this config for method chaining
     */
    LoggerConfig &with_cpu_affinity(uint16_t cpu);
};

/**
 * Detail namespace for internal logger implementation
 */
namespace detail {
/**
 * Get the internal Quill logger instance
 *
 * @return Pointer to the real-time logger instance
 */
RealTimeLogger *get_quill_logger();
} // namespace detail

/**
 * Main logger class providing real-time logging functionality
 *
 * Singleton logger implementation that wraps Quill's real-time
 * logging library with a simplified interface and component-based
 * log level management.
 */
class Logger final {
public:
    /**
     * Configure the default logger with clean config struct
     *
     * @warning This method is NOT real-time safe. It performs internal locking
     * to ensure thread-safe Logger configuration and should only be called during
     * application initialization, never from real-time threads.
     *
     * @param[in] config Logger configuration containing all setup options
     */
    static void configure(const LoggerConfig &config);

    /**
     * Set the global log level for the default logger
     *
     * @param[in] level New log level to set
     */
    static void set_level(LogLevel level);

    /**
     * Flush all pending log messages immediately
     */
    static void flush();

    /**
     * Get the current sink type of the logger
     *
     * @return Currently configured sink type
     */
    [[nodiscard]] static SinkType get_sink_type();

    /**
     * Get the current global log level
     *
     * @return Current global log level
     */
    [[nodiscard]] static LogLevel get_current_level();

    /**
     * Get the actual log file path being used
     *
     * @return Path to the log file, or empty string if not using file output
     */
    [[nodiscard]] static std::string get_actual_log_file();

    /**
     * Destructor - public so unique_ptr can call it
     */
    ~Logger() noexcept;

    // Non-copyable, non-movable
    Logger(const Logger &) = delete;
    Logger &operator=(const Logger &) = delete;
    Logger(Logger &&) = delete;
    Logger &operator=(Logger &&) = delete;

private:
    /**
     * Private constructor - use configure() instead
     *
     * @param[in] config Configuration for logger initialization
     */
    explicit Logger(const LoggerConfig &config);

    /**
     * Get the singleton logger instance
     *
     * @return Reference to the singleton logger instance
     */
    [[nodiscard]] static std::unique_ptr<Logger> &get_instance();

    /**
     * Implementation for setting log level
     *
     * @param[in] level New log level to set
     */
    void set_level_impl(LogLevel level);

    /**
     * Implementation for flushing log messages
     */
    void flush_impl();

    /**
     * Implementation for getting current log level
     *
     * @return Current log level
     */
    [[nodiscard]] LogLevel get_current_level_impl() const;

    /**
     * Convert framework log level to Quill log level
     *
     * @param[in] level Framework log level
     * @return Corresponding Quill log level
     */
    static quill::LogLevel to_quill_level(LogLevel level);

    /**
     * Convert Quill log level to framework log level
     *
     * @param[in] level Quill log level
     * @return Corresponding framework log level
     */
    static LogLevel from_quill_level(quill::LogLevel level);

    /**
     * Create a log sink based on configuration
     *
     * @param[in] sink_type Type of sink to create
     * @param[in] log_file Path to log file (for file-based sinks)
     * @return Shared pointer to the created sink
     */
    [[nodiscard]] std::shared_ptr<quill::Sink>
    create_sink(SinkType sink_type, const char *log_file);

    SinkType sink_type_{SinkType::Console}; //!< Currently configured sink type
    std::string actual_log_file_;           //!< Actual path to the log file being used
    RealTimeLogger *quill_logger_{nullptr}; //!< Pointer to the underlying Quill logger
    bool enable_colors_{true};              //!< Whether color output is enabled

    // Allow detail namespace to access internal methods
    friend RealTimeLogger *detail::get_quill_logger();
};

} // namespace framework::log

#endif // FRAMEWORK_LOG_RT_LOG_HPP
