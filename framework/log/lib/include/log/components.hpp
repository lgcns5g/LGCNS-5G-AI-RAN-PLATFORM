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

#ifndef FRAMEWORK_LOG_COMPONENTS_HPP
#define FRAMEWORK_LOG_COMPONENTS_HPP

#include <array>
#include <mutex>
#include <string_view>
#include <unordered_map>

#include <quill/LogMacros.h>

#include <wise_enum.h>

namespace framework::log {

/**
 * Core log levels for the logging framework
 *
 * Defines severity levels from most verbose (TRACE_L3) to most critical
 * (CRITICAL). Used by both logger and component systems for filtering log
 * messages.
 */
enum class LogLevel {
    TraceL3, //!< Most verbose trace level
    TraceL2, //!< Medium trace level
    TraceL1, //!< Least verbose trace level
    Debug,   //!< Debug messages
    Info,    //!< Informational messages
    Notice,  //!< Notice messages
    Warn,    //!< Warning messages
    Error,   //!< Error messages
    Critical //!< Critical error messages
};

} // namespace framework::log

// WISE_ENUM_ADAPT must be called at global namespace scope for ADL to work
WISE_ENUM_ADAPT(
        framework::log::LogLevel,
        TraceL3,
        TraceL2,
        TraceL1,
        Debug,
        Info,
        Notice,
        Warn,
        Error,
        Critical)

namespace framework::log {

/**
 * Get the default log level for new components
 *
 * @return Default log level (INFO)
 */
LogLevel get_logger_default_level();

/**
 * Generic registry template for contiguous enum types
 *
 * Provides O(1) lookup operations for enum name resolution and validation
 * using compile-time generated lookup tables from wise_enum data.
 *
 * @note Requires enum values to be contiguous starting from 0
 * @tparam EnumType The enum type to create registry for
 */
template <typename EnumType> struct EnumRegistry final {
private:
    static constexpr size_t NUM_VALUES = ::wise_enum::size<EnumType>; //!< Number of enum values

    /**
     * Get the static lookup table for enum names
     *
     * @return Reference to the name lookup table
     */
    static const std::array<std::string_view, NUM_VALUES> &get_name_table() {
        static const std::array<std::string_view, NUM_VALUES> table = build_name_table();
        return table;
    }

    /**
     * Build the lookup table from wise_enum data
     *
     * @return Array of string views for enum names
     */
    static std::array<std::string_view, NUM_VALUES> build_name_table() {
        std::array<std::string_view, NUM_VALUES> table;

        // Simple sequential fill since enums are contiguous 0, 1, 2, ...
        size_t idx = 0;
        for (auto value_and_name : ::wise_enum::range<EnumType>) {
            // wise_enum provides compile-time string literals, safe for string_view
            // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
            table[idx] = std::string_view{value_and_name.name.data(), value_and_name.name.size()};
            ++idx;
        }
        return table;
    }

public:
    /**
     * Get the number of enum values
     *
     * @return Number of enum values in the registry
     */
    static constexpr size_t get_table_size() { return NUM_VALUES; }

    /**
     * Get string name for enum value
     *
     * @param[in] value Enum value to get name for
     * @return String view of enum name, or "UNKNOWN" if invalid
     */
    static constexpr std::string_view get_name(const EnumType value) {
        const auto idx = static_cast<size_t>(value);
        return (idx < NUM_VALUES)
                       // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
                       ? get_name_table()[idx]
                       : std::string_view{"UNKNOWN"};
    }

    /**
     * Check if enum value is valid
     *
     * @param[in] value Enum value to validate
     * @return true if value is within valid range
     */
    static constexpr bool is_valid(const EnumType value) {
        const auto idx = static_cast<size_t>(value);
        return idx < NUM_VALUES; // Simple bounds check
    }

    /**
     * Get array idx for enum value
     *
     * @param[in] value Enum value
     * @return Array idx for the enum value
     */
    static constexpr size_t get_index(const EnumType value) {
        return static_cast<size_t>(value); // Direct cast for contiguous enums
    }
};

/**
 * Component registry type alias for enum-based components
 *
 * @tparam ComponentType Component enum type
 */
template <typename ComponentType> using ComponentRegistry = EnumRegistry<ComponentType>;

/**
 * Event registry type alias for enum-based events
 *
 * @tparam EventType Event enum type
 */
template <typename EventType> using EventRegistry = EnumRegistry<EventType>;

// NOLINTBEGIN(cppcoreguidelines-macro-usage)

/**
 * Declare a log component enum with specified values
 *
 * Creates a wise_enum-based component enumeration that can be used
 * with component-based logging macros.
 *
 * @param ComponentType Name of the component enum type
 * @param ... List of component values
 */
#define DECLARE_LOG_COMPONENT(ComponentType, ...) WISE_ENUM_CLASS(ComponentType, __VA_ARGS__)

/**
 * Declare a log event enum with specified values
 *
 * Creates a wise_enum-based event enumeration that can be used
 * with event-based logging macros.
 *
 * @param EventType Name of the event enum type
 * @param ... List of event values
 */
#define DECLARE_LOG_EVENT(EventType, ...) WISE_ENUM_CLASS(EventType, __VA_ARGS__)

// NOLINTEND(cppcoreguidelines-macro-usage)

/**
 * Component level storage with efficient access patterns
 *
 * Manages per-component log levels using direct array indexing for
 * maximum performance in logging hot paths.
 *
 * @tparam ComponentType The component enum type
 */
template <typename ComponentType> class ComponentLevelStorage final {
private:
    static constexpr size_t NUM_COMPONENTS =
            ::wise_enum::size<ComponentType>;           //!< Number of components in the enum
    static std::array<LogLevel, NUM_COMPONENTS> levels; //!< Per-component log level storage array
    static std::once_flag init_flag;                    //!< Thread-safe initialization flag

public:
    /**
     * Initialize the component level storage with default levels
     */
    static void initialize() {
        std::call_once(init_flag, []() {
            const LogLevel default_level = get_logger_default_level();
            levels.fill(default_level);
        });
    }

    /**
     * Get the current log level for a component
     *
     * @param[in] component Component to query
     * @return Current log level for the component
     */
    static LogLevel get_level(ComponentType component) {
        initialize();
        const auto idx = static_cast<size_t>(
                component); // Direct cast for contiguous enums
                            // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
        return levels[idx];
    }

    /**
     * Set log level for a specific component
     *
     * @param[in] component Component to configure
     * @param[in] level New log level for the component
     */
    static void set_level(ComponentType component, LogLevel level) {
        initialize();
        const auto idx = static_cast<size_t>(component); // Direct cast for contiguous enums
        if (idx < NUM_COMPONENTS) {
            // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
            levels[idx] = level;
        }
    }

    /**
     * Check if a message should be logged for a component
     *
     * @param[in] component Component being logged to
     * @param[in] message_level Log level of the message
     * @return true if message should be logged
     */
    static bool should_log(ComponentType component, LogLevel message_level) {
        initialize();
        const auto idx = static_cast<size_t>(component);
        if (idx >= NUM_COMPONENTS) {
            return message_level >= get_logger_default_level();
        }
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
        return message_level >= levels[idx];
    }

    /**
     * Set the same log level for all components
     *
     * @param[in] level Log level to apply to all components
     */
    static void set_all_levels(LogLevel level) {
        initialize();
        levels.fill(level); // Fill entire array
    }
};

// Static member definitions
template <typename ComponentType>
std::array<LogLevel, ComponentLevelStorage<ComponentType>::NUM_COMPONENTS>
        ComponentLevelStorage<ComponentType>::levels;

template <typename ComponentType> std::once_flag ComponentLevelStorage<ComponentType>::init_flag;

/**
 * Get string representation of component name
 *
 * @tparam ComponentType Component enum type
 * @param[in] component Component enum value
 * @return String view of component name
 */
template <typename ComponentType>
constexpr std::string_view format_component_name(ComponentType component) {
    return ComponentRegistry<ComponentType>::get_name(component);
}

/**
 * Check if component value is valid
 *
 * @tparam ComponentType Component enum type
 * @param[in] component Component enum value to validate
 * @return true if component is within valid range
 */
template <typename ComponentType> constexpr bool is_valid_component(ComponentType component) {
    const auto idx = static_cast<size_t>(component);
    return idx < ::wise_enum::size<ComponentType>; // Simple bounds check
}

/**
 * Register components with individual log levels
 *
 * @tparam ComponentType Component enum type
 * @param[in] component_levels Map of components to their log levels
 */
template <typename ComponentType>
void register_component(const std::unordered_map<ComponentType, LogLevel> &component_levels) {
    ComponentLevelStorage<ComponentType>::initialize();
    for (const auto &[component, level] : component_levels) {
        ComponentLevelStorage<ComponentType>::set_level(component, level);
    }
}

/**
 * Register all components with the same log level
 *
 * @tparam ComponentType Component enum type
 * @param[in] level Log level to assign to all components
 */
template <typename ComponentType> void register_component(LogLevel level) {
    ComponentLevelStorage<ComponentType>::initialize();
    ComponentLevelStorage<ComponentType>::set_all_levels(level);
}

/**
 * Get the current log level for a specific component
 *
 * @tparam ComponentType Component enum type
 * @param[in] component Component to query
 * @return Current log level for the component
 */
template <typename ComponentType>
[[nodiscard]] LogLevel get_component_level(ComponentType component) {
    return ComponentLevelStorage<ComponentType>::get_level(component);
}

/**
 * Get string representation of event name
 *
 * @tparam EventType Event enum type
 * @param[in] event Event enum value
 * @return String view of event name
 */
template <typename EventType> constexpr std::string_view format_event_name(EventType event) {
    return EventRegistry<EventType>::get_name(event);
}

/**
 * Check if event value is valid
 *
 * @tparam EventType Event enum type
 * @param[in] event Event enum value to validate
 * @return true if event is within valid range
 */
template <typename EventType> constexpr bool is_valid_event(EventType event) {
    const auto idx = static_cast<size_t>(event);
    return idx < ::wise_enum::size<EventType>;
}

} // namespace framework::log

#endif // FRAMEWORK_LOG_COMPONENTS_HPP
