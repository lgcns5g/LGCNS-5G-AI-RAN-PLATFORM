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

#ifndef FRAMEWORK_LOG_RT_LOG_MACROS_HPP
#define FRAMEWORK_LOG_RT_LOG_MACROS_HPP

#include <cstddef>
#include <span>

#include <quill/LogMacros.h>

#include "log/rt_log.hpp"

// std::span codec for Quill logging
namespace quill {

/**
 * Generic Quill codec for std::span<T>
 *
 * Enables logging of spans without heap allocation.
 * Uses DirectFormatCodec since span contains a pointer (unsafe to copy across threads).
 * The span data is formatted immediately in the calling thread.
 *
 * @tparam T Element type of the span
 * @tparam EXTENT Extent of the span (defaults to dynamic extent)
 */
template <typename T, std::size_t EXTENT>
struct Codec<std::span<T, EXTENT>> : DirectFormatCodec<std::span<T, EXTENT>> {};

} // namespace quill

namespace framework::log {

// NOLINTBEGIN(cppcoreguidelines-macro-usage,cppcoreguidelines-avoid-do-while,clang-diagnostic-gnu-zero-variadic-macro-arguments)

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#endif

/**
 * Get the default Quill logger instance
 *
 * Returns the singleton logger configured via Logger::get_instance().
 *
 * @return Reference to the default Quill logger
 */
#define RT_GET_LOGGER() ::framework::log::detail::get_quill_logger()

/**
 * Helper macro for component logging
 *
 * Checks component log level before logging to avoid unnecessary work.
 *
 * @param level_enum RT framework log level enum
 * @param quill_level Corresponding Quill log level
 * @param component Component to log for
 * @param message Log message format string
 * @param ... Format arguments
 */
#define RT_LOGC_HELPER(level_enum, quill_level, component, message, ...)                           \
    do {                                                                                           \
        if (::framework::log::ComponentLevelStorage<decltype(component)>::should_log(              \
                    component, ::framework::log::LogLevel::level_enum)) {                          \
            QUILL_LOG_##quill_level(                                                               \
                    RT_GET_LOGGER(),                                                               \
                    "[{}] " message,                                                               \
                    ::framework::log::format_component_name(component),                            \
                    ##__VA_ARGS__);                                                                \
        }                                                                                          \
    } while (0)

/**
 * Helper macro for event+component logging
 *
 * Checks component log level before logging and includes event information.
 *
 * @param level_enum RT framework log level enum
 * @param quill_level Corresponding Quill log level
 * @param component Component to log for
 * @param event Event being logged
 * @param message Log message format string
 * @param ... Format arguments
 */
#define RT_LOGEC_HELPER(level_enum, quill_level, component, event, message, ...)                   \
    do {                                                                                           \
        if (::framework::log::ComponentLevelStorage<decltype(component)>::should_log(              \
                    component, ::framework::log::LogLevel::level_enum)) {                          \
            QUILL_LOG_##quill_level(                                                               \
                    RT_GET_LOGGER(),                                                               \
                    "[{}] EVENT [{}] " message,                                                    \
                    ::framework::log::format_component_name(component),                            \
                    ::framework::log::format_event_name(event),                                    \
                    ##__VA_ARGS__);                                                                \
        }                                                                                          \
    } while (0)

/**
 * Helper macro for component logging with custom logger
 *
 * @param level_enum RT framework log level enum
 * @param quill_level Corresponding Quill log level
 * @param logger Custom logger instance to use
 * @param component Component to log for
 * @param message Log message format string
 * @param ... Format arguments
 */
#define RT_LOGGERC_HELPER(level_enum, quill_level, logger, component, message, ...)                \
    do {                                                                                           \
        if (::framework::log::ComponentLevelStorage<decltype(component)>::should_log(              \
                    component, ::framework::log::LogLevel::level_enum)) {                          \
            QUILL_LOG_##quill_level(                                                               \
                    logger,                                                                        \
                    "[{}] " message,                                                               \
                    ::framework::log::format_component_name(component),                            \
                    ##__VA_ARGS__);                                                                \
        }                                                                                          \
    } while (0)

/**
 * Helper macro for event+component logging with custom logger
 *
 * @param level_enum RT framework log level enum
 * @param quill_level Corresponding Quill log level
 * @param logger Custom logger instance to use
 * @param component Component to log for
 * @param event Event being logged
 * @param message Log message format string
 * @param ... Format arguments
 */
#define RT_LOGGEREC_HELPER(level_enum, quill_level, logger, component, event, message, ...)        \
    do {                                                                                           \
        if (::framework::log::ComponentLevelStorage<decltype(component)>::should_log(              \
                    component, ::framework::log::LogLevel::level_enum)) {                          \
            QUILL_LOG_##quill_level(                                                               \
                    logger,                                                                        \
                    "[{}] EVENT [{}] " message,                                                    \
                    ::framework::log::format_component_name(component),                            \
                    ::framework::log::format_event_name(event),                                    \
                    ##__VA_ARGS__);                                                                \
        }                                                                                          \
    } while (0)

/**
 * Base macro for creating formatter specialization for user-defined types
 *
 * This macro creates a fmtquill::formatter specialization that allows custom
 * types to be logged with a specific format string. The format includes the
 * type name as a prefix for clear identification in logs.
 *
 * @param type The user-defined type to make loggable
 * @param format_str The format string for the type's data members
 * @param ... Variable arguments corresponding to the format string placeholders
 */
#define RT_LOGGABLE_FORMATTER(type, format_str, ...)                                               \
    template <> struct fmtquill::formatter<type> {                                                 \
        constexpr auto parse(format_parse_context &ctx) -> decltype(ctx.begin()) {                 \
            return ctx.begin();                                                                    \
        }                                                                                          \
                                                                                                   \
        template <typename FormatContext>                                                          \
        auto format(const type &obj, FormatContext &ctx) const -> decltype(ctx.out()) {            \
            return fmtquill::format_to(ctx.out(), #type "(" format_str ")", __VA_ARGS__);          \
        }                                                                                          \
    };

/**
 * General macro for making user-defined types loggable with specific codec
 *
 * This macro creates both the formatter specialization and the Quill codec
 * specialization. The codec determines whether the type is formatted
 * immediately (DirectFormatCodec) or can be safely copied and formatted
 * asynchronously (DeferredFormatCodec).
 *
 * @param type The user-defined type to make loggable
 * @param codec_type The Quill codec type (quill::DirectFormatCodec or
 * quill::DeferredFormatCodec)
 * @param format_str The format string for the type's data members
 * @param ... Variable arguments corresponding to the format string placeholders
 */
#define RT_LOGGABLE_FORMAT(type, codec_type, format_str, ...)                                      \
    RT_LOGGABLE_FORMATTER(type, format_str, __VA_ARGS__)                                           \
                                                                                                   \
    template <> struct quill::Codec<type> : codec_type<type> {};

/**
 * Make a user-defined type loggable with deferred formatting (safe for async)
 *
 * Use this macro for types that contain only value types (no pointers,
 * references, or other unsafe members). The type will be copied and formatted
 * asynchronously in the background thread for better performance.
 *
 * Example:
 * @code
 * class Product {
 * public:
 *     std::string name;
 *     double price;
 *     int quantity;
 * };
 *
 * RT_LOGGABLE_DEFERRED_FORMAT(Product, "name: {}, price: {}, quantity: {}",
 *                              obj.name, obj.price, obj.quantity)
 * @endcode
 *
 * @param type The user-defined type to make loggable
 * @param format_str The format string for the type's data members
 * @param ... Variable arguments corresponding to the format string placeholders
 */
#define RT_LOGGABLE_DEFERRED_FORMAT(type, format_str, ...)                                         \
    RT_LOGGABLE_FORMAT(type, quill::DeferredFormatCodec, format_str, __VA_ARGS__)

/**
 * Make a user-defined type loggable with direct formatting (for unsafe types)
 *
 * Use this macro for types that contain pointers, references, or other members
 * that are unsafe to copy across threads. The type will be formatted
 * immediately in the calling thread to ensure safety.
 *
 * Example:
 * @code
 * class User {
 * public:
 *     std::string name;
 *     uint64_t* value_ptr;  // Pointer - unsafe to copy
 * };
 *
 * RT_LOGGABLE_DIRECT_FORMAT(User, "name: {}, value: {}",
 *                            obj.name, obj.value_ptr ? *obj.value_ptr : 0)
 * @endcode
 *
 * @param type The user-defined type to make loggable
 * @param format_str The format string for the type's data members
 * @param ... Variable arguments corresponding to the format string placeholders
 */
#define RT_LOGGABLE_DIRECT_FORMAT(type, format_str, ...)                                           \
    RT_LOGGABLE_FORMAT(type, quill::DirectFormatCodec, format_str, __VA_ARGS__)

// All logging macros (grouped by level)

/**
 * Log a TRACE_L3 level message using the default logger
 *
 * @param fmt Format string for the log message
 * @param ... Format arguments
 */
#define RT_LOG_TRACE_L3(fmt, ...) QUILL_LOG_TRACE_L3(RT_GET_LOGGER(), fmt, ##__VA_ARGS__)

/**
 * Log a TRACE_L3 level JSON message using the default logger
 *
 * @param ... JSON key-value pairs
 */
#define RT_LOGJ_TRACE_L3(...) QUILL_LOGJ_TRACE_L3(RT_GET_LOGGER(), ##__VA_ARGS__)

/**
 * Log a TRACE_L3 level message with component information
 *
 * @param c Component enum value
 * @param m Message format string
 * @param ... Format arguments
 */
#define RT_LOGC_TRACE_L3(c, m, ...) RT_LOGC_HELPER(TraceL3, TRACE_L3, c, m, ##__VA_ARGS__)

/**
 * Log a TRACE_L3 level message with event information
 *
 * @param e Event enum value
 * @param m Message format string
 * @param ... Format arguments
 */
#define RT_LOGE_TRACE_L3(e, m, ...)                                                                \
    RT_LOG_TRACE_L3("EVENT [{}] " m, ::framework::log::format_event_name(e), ##__VA_ARGS__)

/**
 * Log a TRACE_L3 level message with component and event information
 *
 * @param c Component enum value
 * @param e Event enum value
 * @param m Message format string
 * @param ... Format arguments
 */
#define RT_LOGEC_TRACE_L3(c, e, m, ...) RT_LOGEC_HELPER(TraceL3, TRACE_L3, c, e, m, ##__VA_ARGS__)

/**
 * Log a TRACE_L3 level message using a custom logger
 *
 * @param logger Custom logger instance
 * @param fmt Format string for the log message
 * @param ... Format arguments
 */
#define RT_LOGGER_TRACE_L3(logger, fmt, ...) QUILL_LOG_TRACE_L3(logger, fmt, ##__VA_ARGS__)

/**
 * Log a TRACE_L3 level JSON message using a custom logger
 *
 * @param logger Custom logger instance
 * @param ... JSON key-value pairs
 */
#define RT_LOGGERJ_TRACE_L3(logger, ...) QUILL_LOGJ_TRACE_L3(logger, ##__VA_ARGS__)

/**
 * Log a TRACE_L3 level message with component information using a custom logger
 *
 * @param logger Custom logger instance
 * @param c Component enum value
 * @param m Message format string
 * @param ... Format arguments
 */
#define RT_LOGGERC_TRACE_L3(logger, c, m, ...)                                                     \
    RT_LOGGERC_HELPER(TraceL3, TRACE_L3, logger, c, m, ##__VA_ARGS__)

/**
 * Log a TRACE_L3 level message with event information using a custom logger
 *
 * @param logger Custom logger instance
 * @param e Event enum value
 * @param m Message format string
 * @param ... Format arguments
 */
#define RT_LOGGERE_TRACE_L3(logger, e, m, ...)                                                     \
    QUILL_LOG_TRACE_L3(                                                                            \
            logger, "EVENT [{}] " m, ::framework::log::format_event_name(e), ##__VA_ARGS__)

/**
 * Log a TRACE_L3 level message with component and event information using a
 * custom logger
 *
 * @param logger Custom logger instance
 * @param c Component enum value
 * @param e Event enum value
 * @param m Message format string
 * @param ... Format arguments
 */
#define RT_LOGGEREC_TRACE_L3(logger, c, e, m, ...)                                                 \
    RT_LOGGEREC_HELPER(TraceL3, TRACE_L3, logger, c, e, m, ##__VA_ARGS__)

// TRACE_L2 logging macros
#define RT_LOG_TRACE_L2(fmt, ...) QUILL_LOG_TRACE_L2(RT_GET_LOGGER(), fmt, ##__VA_ARGS__)
#define RT_LOGJ_TRACE_L2(...) QUILL_LOGJ_TRACE_L2(RT_GET_LOGGER(), ##__VA_ARGS__)
#define RT_LOGC_TRACE_L2(c, m, ...) RT_LOGC_HELPER(TraceL2, TRACE_L2, c, m, ##__VA_ARGS__)
#define RT_LOGE_TRACE_L2(e, m, ...)                                                                \
    RT_LOG_TRACE_L2("EVENT [{}] " m, ::framework::log::format_event_name(e), ##__VA_ARGS__)
#define RT_LOGEC_TRACE_L2(c, e, m, ...) RT_LOGEC_HELPER(TraceL2, TRACE_L2, c, e, m, ##__VA_ARGS__)

#define RT_LOGGER_TRACE_L2(logger, fmt, ...) QUILL_LOG_TRACE_L2(logger, fmt, ##__VA_ARGS__)
#define RT_LOGGERJ_TRACE_L2(logger, ...) QUILL_LOGJ_TRACE_L2(logger, ##__VA_ARGS__)
#define RT_LOGGERC_TRACE_L2(logger, c, m, ...)                                                     \
    RT_LOGGERC_HELPER(TraceL2, TRACE_L2, logger, c, m, ##__VA_ARGS__)
#define RT_LOGGERE_TRACE_L2(logger, e, m, ...)                                                     \
    QUILL_LOG_TRACE_L2(                                                                            \
            logger, "EVENT [{}] " m, ::framework::log::format_event_name(e), ##__VA_ARGS__)
#define RT_LOGGEREC_TRACE_L2(logger, c, e, m, ...)                                                 \
    RT_LOGGEREC_HELPER(TraceL2, TRACE_L2, logger, c, e, m, ##__VA_ARGS__)

// TRACE_L1 logging macros
#define RT_LOG_TRACE_L1(fmt, ...) QUILL_LOG_TRACE_L1(RT_GET_LOGGER(), fmt, ##__VA_ARGS__)
#define RT_LOGJ_TRACE_L1(...) QUILL_LOGJ_TRACE_L1(RT_GET_LOGGER(), ##__VA_ARGS__)
#define RT_LOGC_TRACE_L1(c, m, ...) RT_LOGC_HELPER(TraceL1, TRACE_L1, c, m, ##__VA_ARGS__)
#define RT_LOGE_TRACE_L1(e, m, ...)                                                                \
    RT_LOG_TRACE_L1("EVENT [{}] " m, ::framework::log::format_event_name(e), ##__VA_ARGS__)
#define RT_LOGEC_TRACE_L1(c, e, m, ...) RT_LOGEC_HELPER(TraceL1, TRACE_L1, c, e, m, ##__VA_ARGS__)

#define RT_LOGGER_TRACE_L1(logger, fmt, ...) QUILL_LOG_TRACE_L1(logger, fmt, ##__VA_ARGS__)
#define RT_LOGGERJ_TRACE_L1(logger, ...) QUILL_LOGJ_TRACE_L1(logger, ##__VA_ARGS__)
#define RT_LOGGERC_TRACE_L1(logger, c, m, ...)                                                     \
    RT_LOGGERC_HELPER(TraceL1, TRACE_L1, logger, c, m, ##__VA_ARGS__)
#define RT_LOGGERE_TRACE_L1(logger, e, m, ...)                                                     \
    QUILL_LOG_TRACE_L1(                                                                            \
            logger, "EVENT [{}] " m, ::framework::log::format_event_name(e), ##__VA_ARGS__)
#define RT_LOGGEREC_TRACE_L1(logger, c, e, m, ...)                                                 \
    RT_LOGGEREC_HELPER(TraceL1, TRACE_L1, logger, c, e, m, ##__VA_ARGS__)

// DEBUG logging macros
#define RT_LOG_DEBUG(fmt, ...) QUILL_LOG_DEBUG(RT_GET_LOGGER(), fmt, ##__VA_ARGS__)
#define RT_LOGJ_DEBUG(...) QUILL_LOGJ_DEBUG(RT_GET_LOGGER(), ##__VA_ARGS__)
#define RT_LOGC_DEBUG(c, m, ...) RT_LOGC_HELPER(Debug, DEBUG, c, m, ##__VA_ARGS__)
#define RT_LOGE_DEBUG(e, m, ...)                                                                   \
    RT_LOG_DEBUG("EVENT [{}] " m, ::framework::log::format_event_name(e), ##__VA_ARGS__)
#define RT_LOGEC_DEBUG(c, e, m, ...) RT_LOGEC_HELPER(Debug, DEBUG, c, e, m, ##__VA_ARGS__)

#define RT_LOGGER_DEBUG(logger, fmt, ...) QUILL_LOG_DEBUG(logger, fmt, ##__VA_ARGS__)
#define RT_LOGGERJ_DEBUG(logger, ...) QUILL_LOGJ_DEBUG(logger, ##__VA_ARGS__)
#define RT_LOGGERC_DEBUG(logger, c, m, ...)                                                        \
    RT_LOGGERC_HELPER(Debug, DEBUG, logger, c, m, ##__VA_ARGS__)
#define RT_LOGGERE_DEBUG(logger, e, m, ...)                                                        \
    QUILL_LOG_DEBUG(logger, "EVENT [{}] " m, ::framework::log::format_event_name(e), ##__VA_ARGS__)
#define RT_LOGGEREC_DEBUG(logger, c, e, m, ...)                                                    \
    RT_LOGGEREC_HELPER(Debug, DEBUG, logger, c, e, m, ##__VA_ARGS__)

// INFO logging macros
#define RT_LOG_INFO(fmt, ...) QUILL_LOG_INFO(RT_GET_LOGGER(), fmt, ##__VA_ARGS__)
#define RT_LOGJ_INFO(...) QUILL_LOGJ_INFO(RT_GET_LOGGER(), ##__VA_ARGS__)
#define RT_LOGC_INFO(c, m, ...) RT_LOGC_HELPER(Info, INFO, c, m, ##__VA_ARGS__)
#define RT_LOGE_INFO(e, m, ...)                                                                    \
    RT_LOG_INFO("EVENT [{}] " m, ::framework::log::format_event_name(e), ##__VA_ARGS__)
#define RT_LOGEC_INFO(c, e, m, ...) RT_LOGEC_HELPER(Info, INFO, c, e, m, ##__VA_ARGS__)

#define RT_LOGGER_INFO(logger, fmt, ...) QUILL_LOG_INFO(logger, fmt, ##__VA_ARGS__)
#define RT_LOGGERJ_INFO(logger, ...) QUILL_LOGJ_INFO(logger, ##__VA_ARGS__)
#define RT_LOGGERC_INFO(logger, c, m, ...)                                                         \
    RT_LOGGERC_HELPER(Info, INFO, logger, c, m, ##__VA_ARGS__)
#define RT_LOGGERE_INFO(logger, e, m, ...)                                                         \
    QUILL_LOG_INFO(logger, "EVENT [{}] " m, ::framework::log::format_event_name(e), ##__VA_ARGS__)
#define RT_LOGGEREC_INFO(logger, c, e, m, ...)                                                     \
    RT_LOGGEREC_HELPER(Info, INFO, logger, c, e, m, ##__VA_ARGS__)

// NOTICE logging macros
#define RT_LOG_NOTICE(fmt, ...) QUILL_LOG_NOTICE(RT_GET_LOGGER(), fmt, ##__VA_ARGS__)
#define RT_LOGJ_NOTICE(...) QUILL_LOGJ_NOTICE(RT_GET_LOGGER(), ##__VA_ARGS__)
#define RT_LOGC_NOTICE(c, m, ...) RT_LOGC_HELPER(Notice, NOTICE, c, m, ##__VA_ARGS__)
#define RT_LOGE_NOTICE(e, m, ...)                                                                  \
    RT_LOG_NOTICE("EVENT [{}] " m, ::framework::log::format_event_name(e), ##__VA_ARGS__)
#define RT_LOGEC_NOTICE(c, e, m, ...) RT_LOGEC_HELPER(Notice, NOTICE, c, e, m, ##__VA_ARGS__)

#define RT_LOGGER_NOTICE(logger, fmt, ...) QUILL_LOG_NOTICE(logger, fmt, ##__VA_ARGS__)
#define RT_LOGGERJ_NOTICE(logger, ...) QUILL_LOGJ_NOTICE(logger, ##__VA_ARGS__)
#define RT_LOGGERC_NOTICE(logger, c, m, ...)                                                       \
    RT_LOGGERC_HELPER(Notice, NOTICE, logger, c, m, ##__VA_ARGS__)
#define RT_LOGGERE_NOTICE(logger, e, m, ...)                                                       \
    QUILL_LOG_NOTICE(logger, "EVENT [{}] " m, ::framework::log::format_event_name(e), ##__VA_ARGS__)
#define RT_LOGGEREC_NOTICE(logger, c, e, m, ...)                                                   \
    RT_LOGGEREC_HELPER(Notice, NOTICE, logger, c, e, m, ##__VA_ARGS__)

// WARN logging macros
#define RT_LOG_WARN(fmt, ...) QUILL_LOG_WARNING(RT_GET_LOGGER(), fmt, ##__VA_ARGS__)
#define RT_LOGJ_WARN(...) QUILL_LOGJ_WARNING(RT_GET_LOGGER(), ##__VA_ARGS__)
#define RT_LOGC_WARN(c, m, ...) RT_LOGC_HELPER(Warn, WARNING, c, m, ##__VA_ARGS__)
#define RT_LOGE_WARN(e, m, ...)                                                                    \
    RT_LOG_WARN("EVENT [{}] " m, ::framework::log::format_event_name(e), ##__VA_ARGS__)
#define RT_LOGEC_WARN(c, e, m, ...) RT_LOGEC_HELPER(Warn, WARNING, c, e, m, ##__VA_ARGS__)

#define RT_LOGGER_WARN(logger, fmt, ...) QUILL_LOG_WARNING(logger, fmt, ##__VA_ARGS__)
#define RT_LOGGERJ_WARN(logger, ...) QUILL_LOGJ_WARNING(logger, ##__VA_ARGS__)
#define RT_LOGGERC_WARN(logger, c, m, ...)                                                         \
    RT_LOGGERC_HELPER(Warn, WARNING, logger, c, m, ##__VA_ARGS__)
#define RT_LOGGERE_WARN(logger, e, m, ...)                                                         \
    QUILL_LOG_WARNING(                                                                             \
            logger, "EVENT [{}] " m, ::framework::log::format_event_name(e), ##__VA_ARGS__)
#define RT_LOGGEREC_WARN(logger, c, e, m, ...)                                                     \
    RT_LOGGEREC_HELPER(Warn, WARNING, logger, c, e, m, ##__VA_ARGS__)

// ERROR logging macros
#define RT_LOG_ERROR(fmt, ...) QUILL_LOG_ERROR(RT_GET_LOGGER(), fmt, ##__VA_ARGS__)
#define RT_LOGJ_ERROR(...) QUILL_LOGJ_ERROR(RT_GET_LOGGER(), ##__VA_ARGS__)
#define RT_LOGC_ERROR(c, m, ...) RT_LOGC_HELPER(Error, ERROR, c, m, ##__VA_ARGS__)
#define RT_LOGE_ERROR(e, m, ...)                                                                   \
    RT_LOG_ERROR("EVENT [{}] " m, ::framework::log::format_event_name(e), ##__VA_ARGS__)
#define RT_LOGEC_ERROR(c, e, m, ...) RT_LOGEC_HELPER(Error, ERROR, c, e, m, ##__VA_ARGS__)

#define RT_LOGGER_ERROR(logger, fmt, ...) QUILL_LOG_ERROR(logger, fmt, ##__VA_ARGS__)
#define RT_LOGGERJ_ERROR(logger, ...) QUILL_LOGJ_ERROR(logger, ##__VA_ARGS__)
#define RT_LOGGERC_ERROR(logger, c, m, ...)                                                        \
    RT_LOGGERC_HELPER(Error, ERROR, logger, c, m, ##__VA_ARGS__)
#define RT_LOGGERE_ERROR(logger, e, m, ...)                                                        \
    QUILL_LOG_ERROR(logger, "EVENT [{}] " m, ::framework::log::format_event_name(e), ##__VA_ARGS__)
#define RT_LOGGEREC_ERROR(logger, c, e, m, ...)                                                    \
    RT_LOGGEREC_HELPER(Error, ERROR, logger, c, e, m, ##__VA_ARGS__)

// CRITICAL logging macros
#define RT_LOG_CRITICAL(fmt, ...) QUILL_LOG_CRITICAL(RT_GET_LOGGER(), fmt, ##__VA_ARGS__)
#define RT_LOGJ_CRITICAL(...) QUILL_LOGJ_CRITICAL(RT_GET_LOGGER(), ##__VA_ARGS__)
#define RT_LOGC_CRITICAL(c, m, ...) RT_LOGC_HELPER(Critical, CRITICAL, c, m, ##__VA_ARGS__)
#define RT_LOGE_CRITICAL(e, m, ...)                                                                \
    RT_LOG_CRITICAL("EVENT [{}] " m, ::framework::log::format_event_name(e), ##__VA_ARGS__)
#define RT_LOGEC_CRITICAL(c, e, m, ...) RT_LOGEC_HELPER(Critical, CRITICAL, c, e, m, ##__VA_ARGS__)

#define RT_LOGGER_CRITICAL(logger, fmt, ...) QUILL_LOG_CRITICAL(logger, fmt, ##__VA_ARGS__)
#define RT_LOGGERJ_CRITICAL(logger, ...) QUILL_LOGJ_CRITICAL(logger, ##__VA_ARGS__)
#define RT_LOGGERC_CRITICAL(logger, c, m, ...)                                                     \
    RT_LOGGERC_HELPER(Critical, CRITICAL, logger, c, m, ##__VA_ARGS__)
#define RT_LOGGERE_CRITICAL(logger, e, m, ...)                                                     \
    QUILL_LOG_CRITICAL(                                                                            \
            logger, "EVENT [{}] " m, ::framework::log::format_event_name(e), ##__VA_ARGS__)
#define RT_LOGGEREC_CRITICAL(logger, c, e, m, ...)                                                 \
    RT_LOGGEREC_HELPER(Critical, CRITICAL, logger, c, e, m, ##__VA_ARGS__)

#ifdef __clang__
#pragma clang diagnostic pop
#endif

// NOLINTEND(cppcoreguidelines-macro-usage,cppcoreguidelines-avoid-do-while,clang-diagnostic-gnu-zero-variadic-macro-arguments)

} // namespace framework::log

#endif // FRAMEWORK_LOG_RT_LOG_MACROS_HPP
