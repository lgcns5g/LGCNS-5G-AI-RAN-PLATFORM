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

#include <algorithm>  // for max
#include <atomic>     // for atomic
#include <chrono>     // for microseconds, nanoseconds
#include <cstdint>    // for uint16_t
#include <cstring>    // for strlen, size_t
#include <filesystem> // for path
#include <functional> // for function
#include <limits>     // for numeric_limits
#include <memory>     // for allocator, shared_ptr
#include <mutex>      // for mutex
#include <stdexcept>  // for invalid_argument
#include <string>     // for string, to_string
#include <utility>    // for move
#include <vector>     // for vector

#include <quill/Backend.h>                      // for Backend
#include <quill/LogMacros.h>                    // for QUILL_LOG_INFO
#include <quill/backend/BackendOptions.h>       // for BackendOptions
#include <quill/core/Common.h>                  // for Timezone, ClockSourc...
#include <quill/core/Filesystem.h>              // for fs
#include <quill/core/LogLevel.h>                // for LogLevel
#include <quill/core/PatternFormatterOptions.h> // for PatternFormatterOptions
#include <quill/sinks/ConsoleSink.h>            // for ConsoleSink, Console...
#include <quill/sinks/FileSink.h>               // for FileSink, FileSinkCo...
#include <quill/sinks/JsonSink.h>               // for JsonFileSink, JsonCo...
#include <quill/sinks/RotatingFileSink.h>       // for RotatingFileSink
#include <quill/sinks/RotatingJsonFileSink.h>   // for RotatingJsonFileSink
#include <quill/sinks/RotatingSink.h>           // for RotatingFileSinkConfig
#include <quill/sinks/Sink.h>                   // for Sink
#include <quill/sinks/StreamSink.h>             // for FileEventNotifier

#include <wise_enum.h> // for to_string

#include "log/components.hpp" // for LogLevel, get_logger...
#include "log/rt_log.hpp"     // for LoggerConfig, Logger

namespace framework::log {

namespace {
/**
 * Create a LoggerConfig with specified parameters
 *
 * @param[in] sink_type Type of log output sink to use
 * @param[in] level Minimum log level to process
 * @param[in] enable_colors Enable color output
 * @param[in] log_file Path to log file (nullptr for console sinks)
 * @return LoggerConfig configured with the specified parameters
 */
LoggerConfig create_logger_config(
        SinkType sink_type, LogLevel level, bool enable_colors, const char *log_file = nullptr) {
    LoggerConfig config{};
    config.sink_type = sink_type;
    config.min_level = level;
    config.enable_colors = enable_colors;
    config.log_file = log_file;
    return config;
}
} // anonymous namespace

// LoggerConfig static factory methods
LoggerConfig LoggerConfig::console(LogLevel level, bool colors) {
    return create_logger_config(SinkType::Console, level, colors);
}

LoggerConfig LoggerConfig::file(const char *path, LogLevel level) {
    return create_logger_config(SinkType::File, level, false, path);
}

LoggerConfig LoggerConfig::rotating_file(const char *path, LogLevel level) {
    return create_logger_config(SinkType::RotatingFile, level, false, path);
}

LoggerConfig LoggerConfig::json_file(const char *path, LogLevel level) {
    return create_logger_config(SinkType::JsonFile, level, false, path);
}

LoggerConfig LoggerConfig::json_console(LogLevel level) {
    return create_logger_config(SinkType::JsonConsole, level, false);
}

LoggerConfig LoggerConfig::rotating_json_file(const char *path, LogLevel level) {
    return create_logger_config(SinkType::RotatingJsonFile, level, false, path);
}

// LoggerConfig fluent API methods
LoggerConfig &LoggerConfig::with_file_line(bool enable) {
    enable_file_line = enable;
    return *this;
}

LoggerConfig &LoggerConfig::with_caller(bool enable) {
    enable_caller = enable;
    return *this;
}

LoggerConfig &LoggerConfig::with_timestamps(bool enable) {
    enable_timestamps = enable;
    return *this;
}

LoggerConfig &LoggerConfig::with_log_level(bool enable) {
    enable_log_level = enable;
    return *this;
}

LoggerConfig &LoggerConfig::with_thread_name(bool enable) {
    enable_thread_name = enable;
    return *this;
}

LoggerConfig &LoggerConfig::with_colors(bool enable) {
    enable_colors = enable;
    return *this;
}

LoggerConfig &LoggerConfig::with_backend_sleep_duration(std::chrono::nanoseconds duration) {
    backend_sleep_duration = duration;
    return *this;
}

LoggerConfig &LoggerConfig::with_cpu_affinity(uint16_t cpu) {
    backend_cpu_affinity = cpu;
    return *this;
}

Logger::Logger(const LoggerConfig &config)
        : sink_type_{config.sink_type}, enable_colors_{config.enable_colors} {

    // ENFORCE SINGLETON: Stop any existing backend first
    if (quill::Backend::is_running()) {
        quill::Backend::stop();
    }

    // Configure backend for high performance
    quill::BackendOptions backend_options;

    // Pin backend to specific CPU core if specified
    backend_options.cpu_affinity = config.backend_cpu_affinity;

    // Use configurable backend sleep duration
    backend_options.thread_name = "QuillBackend";
    backend_options.enable_yield_when_idle = true;
    backend_options.sleep_duration = config.backend_sleep_duration;

    backend_options.log_timestamp_ordering_grace_period = std::chrono::microseconds{0};

    // Start fresh backend for this logger
    quill::Backend::start(backend_options);

    RealTimeFrontend::preallocate();

    // Create the appropriate sink using our real-time frontend
    auto sink = create_sink(config.sink_type, config.log_file);

    // Build pattern based on configuration options
    std::string pattern;

    // Build the base pattern components
    std::vector<std::string> components;

    if (config.enable_timestamps) {
        components.emplace_back("%(time)");
    }

    if (config.enable_log_level) {
        components.emplace_back("[%(log_level)]");
    }

    if (config.enable_thread_name) {
        components.emplace_back("[%(thread_name)]");
    }

    // Join components with spaces
    for (size_t i = 0; i < components.size(); ++i) {
        if (i > 0) {
            pattern += " ";
        }
        pattern += components[i];
    }

    // If no base components, provide minimal pattern
    if (components.empty()) {
        pattern = "%(message)";
    }

    if (config.enable_file_line && config.enable_caller) {
        // Combined: filename:line function
        pattern += " [%(short_source_location) %(caller_function)]";
    } else if (config.enable_file_line) {
        // Just filename:line
        pattern += " [%(short_source_location)]";
    } else if (config.enable_caller) {
        // Just function name
        pattern += " [%(caller_function)]";
    }

    // Always end with message if we have other components
    if (!components.empty()) {
        pattern += " %(message)";
    }

    quill::PatternFormatterOptions formatter_opts{
            pattern, "%H:%M:%S.%Qns", quill::Timezone::LocalTime};

    if (config.sink_type == SinkType::JsonFile || config.sink_type == SinkType::JsonConsole ||
        config.sink_type == SinkType::RotatingJsonFile) {
        formatter_opts = quill::PatternFormatterOptions{
                pattern, "%H:%M:%S.%Qns", quill::Timezone::LocalTime};
    }

    // Clean up existing logger if any
    if (quill_logger_ != nullptr) {
        RealTimeFrontend::remove_logger_blocking(quill_logger_);
    }

    static std::atomic<int> logger_counter{0};
    const std::string logger_name = "rt_logger_" + std::to_string(logger_counter.fetch_add(1));

    quill_logger_ = RealTimeFrontend::create_or_get_logger(
            logger_name, std::move(sink), formatter_opts, quill::ClockSourceType::Tsc);

    quill_logger_->set_log_level(to_quill_level(config.min_level));

    const auto sink_name = ::wise_enum::to_string(config.sink_type);
    const auto level_name = ::wise_enum::to_string(config.min_level);

    QUILL_LOG_INFO(
            quill_logger_,
            "Real-time logger configured - Sink: {}, Level: {}, "
            "Pattern: '{}', File: '{}'",
            sink_name,
            level_name,
            pattern,
            actual_log_file_.empty() ? "none" : actual_log_file_);

    QUILL_LOG_INFO(
            quill_logger_,
            "Logger features - Timestamps: {}, LogLevel: {}, ThreadName: {}, "
            "File/Line: {}, Caller: {}, Colors: {}, Thread: '{}', CPU: {}, "
            "BackendSleep: {}ns, InitialQueueCapacity: {}MB",
            config.enable_timestamps ? "enabled" : "disabled",
            config.enable_log_level ? "enabled" : "disabled",
            config.enable_thread_name ? "enabled" : "disabled",
            config.enable_file_line ? "enabled" : "disabled",
            config.enable_caller ? "enabled" : "disabled",
            config.enable_colors ? "enabled" : "disabled",
            backend_options.thread_name,
            config.backend_cpu_affinity == std::numeric_limits<uint16_t>::max()
                    ? "any"
                    : std::to_string(config.backend_cpu_affinity),
            config.backend_sleep_duration.count(),
            RealTimeFrontendOptions::initial_queue_capacity / (1024 * 1024));
}

Logger::~Logger() noexcept {
    if (quill::Backend::is_running()) {
        quill::Backend::stop();
    }
}

std::unique_ptr<Logger> &Logger::get_instance() {
    static std::unique_ptr<Logger> instance =
            std::unique_ptr<Logger>(new Logger(LoggerConfig::console()));
    return instance;
}

void Logger::configure(const LoggerConfig &config) {
    static std::mutex configure_mutex;
    const std::lock_guard<std::mutex> lock(configure_mutex);
    auto &instance = get_instance();
    instance.reset();
    instance = std::unique_ptr<Logger>(new Logger(config));
}

// Static interface methods
void Logger::set_level(LogLevel level) { get_instance()->set_level_impl(level); }

void Logger::flush() { get_instance()->flush_impl(); }

SinkType Logger::get_sink_type() { return get_instance()->sink_type_; }

LogLevel Logger::get_current_level() { return get_instance()->get_current_level_impl(); }

std::string Logger::get_actual_log_file() { return get_instance()->actual_log_file_; }

// Instance methods (called by static interface)
void Logger::set_level_impl(LogLevel level) {
    if (quill_logger_ != nullptr) {
        quill_logger_->set_log_level(to_quill_level(level));
    }
}

void Logger::flush_impl() {
    if (quill_logger_ != nullptr) {
        quill_logger_->flush_log();
    }
}

LogLevel Logger::get_current_level_impl() const {
    if (quill_logger_ != nullptr) {
        return from_quill_level(quill_logger_->get_log_level());
    }
    return LogLevel::Info;
}

std::shared_ptr<quill::Sink> Logger::create_sink(SinkType sink_type, const char *log_file) {
    std::shared_ptr<quill::Sink> sink;

    switch (sink_type) {
    case SinkType::Console: {
        quill::ConsoleSinkConfig console_config;
        // Set color mode based on option
        if (enable_colors_) {
            console_config.set_colour_mode(quill::ConsoleSinkConfig::ColourMode::Always);
        } else {
            console_config.set_colour_mode(quill::ConsoleSinkConfig::ColourMode::Never);
        }
        sink = std::make_shared<quill::ConsoleSink>(console_config);
        actual_log_file_ = ""; // No file for console
        break;
    }

    case SinkType::File: {
        if (log_file == nullptr || strlen(log_file) == 0) {
            throw std::invalid_argument("File sink requires a log file path");
        }

        // Create sink directly to avoid type conflicts from caching
        quill::FileSinkConfig cfg;
        cfg.set_open_mode('w');
        cfg.set_filename_append_option(quill::FilenameAppendOption::StartDateTime);
        auto file_sink =
                std::make_shared<quill::FileSink>(log_file, cfg, quill::FileEventNotifier{});
        sink = file_sink;

        actual_log_file_ = file_sink->get_filename().string();
        break;
    }

    case SinkType::RotatingFile: {
        if (log_file == nullptr || strlen(log_file) == 0) {
            throw std::invalid_argument("Rotating file sink requires a log file path");
        }

        // Create sink directly to avoid type conflicts from caching
        constexpr auto LOG_ROTATION_MAX_FILE_SIZE_BYTES =
                static_cast<std::size_t>(10 * 1024 * 1024);

        quill::RotatingFileSinkConfig cfg;
        cfg.set_open_mode('w');
        cfg.set_filename_append_option(quill::FilenameAppendOption::StartDateTime);
        cfg.set_rotation_time_daily("23:59");
        cfg.set_rotation_max_file_size(LOG_ROTATION_MAX_FILE_SIZE_BYTES);
        auto rotating_sink = std::make_shared<quill::RotatingFileSink>(log_file, cfg);
        sink = rotating_sink;

        actual_log_file_ = rotating_sink->get_filename().string();
        break;
    }

    case SinkType::JsonFile: {
        if (log_file == nullptr || strlen(log_file) == 0) {
            throw std::invalid_argument("JSON file sink requires a log file path");
        }

        // Create sink directly to avoid type conflicts from caching
        quill::FileSinkConfig config;
        config.set_open_mode('w');
        config.set_filename_append_option(quill::FilenameAppendOption::StartDateTime);
        auto json_sink = std::make_shared<quill::JsonFileSink>(log_file, config);
        sink = json_sink;

        actual_log_file_ = json_sink->get_filename().string();
        break;
    }

    case SinkType::JsonConsole: {
        auto json_console_sink =
                RealTimeFrontend::create_or_get_sink<quill::JsonConsoleSink>("json_console_sink");
        sink = json_console_sink;
        actual_log_file_ = ""; // No file for console
        break;
    }

    case SinkType::RotatingJsonFile: {
        if (log_file == nullptr || strlen(log_file) == 0) {
            throw std::invalid_argument("Rotating JSON file sink requires a log file path");
        }

        // Create sink directly to avoid type conflicts from caching
        constexpr auto LOG_ROTATION_MAX_FILE_SIZE_BYTES =
                static_cast<std::size_t>(10 * 1024 * 1024);

        quill::RotatingFileSinkConfig cfg;
        cfg.set_open_mode('w');
        cfg.set_filename_append_option(quill::FilenameAppendOption::StartDateTime);
        cfg.set_rotation_time_daily("23:59");
        cfg.set_rotation_max_file_size(LOG_ROTATION_MAX_FILE_SIZE_BYTES);
        auto rotating_json_sink = std::make_shared<quill::RotatingJsonFileSink>(log_file, cfg);
        sink = rotating_json_sink;

        actual_log_file_ = rotating_json_sink->get_filename().string();
        break;
    }

    default:
        throw std::invalid_argument("Unknown sink type");
    }

    return sink;
}

quill::LogLevel Logger::to_quill_level(LogLevel level) {
    switch (level) {
    case LogLevel::TraceL3:
        return quill::LogLevel::TraceL3;
    case LogLevel::TraceL2:
        return quill::LogLevel::TraceL2;
    case LogLevel::TraceL1:
        return quill::LogLevel::TraceL1;
    case LogLevel::Debug:
        return quill::LogLevel::Debug;
    case LogLevel::Info:
        return quill::LogLevel::Info;
    case LogLevel::Notice:
        return quill::LogLevel::Notice;
    case LogLevel::Warn:
        return quill::LogLevel::Warning;
    case LogLevel::Error:
        return quill::LogLevel::Error;
    case LogLevel::Critical:
        return quill::LogLevel::Critical;
    default:
        return quill::LogLevel::Info;
    }
}

LogLevel Logger::from_quill_level(quill::LogLevel level) {
    switch (level) {
    case quill::LogLevel::TraceL3:
        return LogLevel::TraceL3;
    case quill::LogLevel::TraceL2:
        return LogLevel::TraceL2;
    case quill::LogLevel::TraceL1:
        return LogLevel::TraceL1;
    case quill::LogLevel::Debug:
        return LogLevel::Debug;
    case quill::LogLevel::Info:
        return LogLevel::Info;
    case quill::LogLevel::Notice:
        return LogLevel::Notice;
    case quill::LogLevel::Warning:
        return LogLevel::Warn;
    case quill::LogLevel::Error:
        return LogLevel::Error;
    case quill::LogLevel::Critical:
        return LogLevel::Critical;
    default:
        return LogLevel::Info;
    }
}

LogLevel get_logger_default_level() { return LogLevel::Info; }

// Detail namespace implementation
namespace detail {
RealTimeLogger *get_quill_logger() { return Logger::get_instance()->quill_logger_; }
} // namespace detail

} // namespace framework::log
