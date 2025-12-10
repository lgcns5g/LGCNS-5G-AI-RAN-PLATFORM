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
 * @file task_category.hpp
 * @brief Extensible task categorization system for organizing and prioritizing
 * work
 *
 * Provides a TaskCategory system that supports both built-in
 * categories and user-defined categories via simple macro declarations. Uses
 * wise_enum for reflection and string conversion capabilities.
 */

#ifndef FRAMEWORK_TASK_TASK_CATEGORY_HPP
#define FRAMEWORK_TASK_TASK_CATEGORY_HPP

#include <cstdint>
#include <format>
#include <functional>
#include <string_view>
#include <typeinfo>

#include <wise_enum.h>

namespace framework::task {

/**
 * Declare task categories using wise_enum
 *
 * Creates a wise_enum-based category enumeration that can be used with the
 * TaskCategory system for type-safe, extensible task categorization.
 *
 * @param CategoryType Name of the category enum type
 * @param ... List of category names
 *
 * Example:
 * @code
 * DECLARE_TASK_CATEGORIES(MyCategories, DataProcessing, NetworkIO, Rendering);
 *
 * // Use with TaskCategory:
 * Task task{func, "work", MyCategories::DataProcessing};
 * @endcode
 */
// NOLINTBEGIN(cppcoreguidelines-macro-usage)
#define DECLARE_TASK_CATEGORIES(CategoryType, ...)                                                 \
    WISE_ENUM_CLASS(CategoryType, __VA_ARGS__)                                                     \
    [[maybe_unused]] constexpr std::string_view get_category_type_name(CategoryType *) {           \
        return #CategoryType;                                                                      \
    }                                                                                              \
    [[maybe_unused]] static constexpr auto CategoryType##_suppress_unused_ =                       \
            ::wise_enum::size<CategoryType>
// NOLINTEND(cppcoreguidelines-macro-usage)

/**
 * Built-in task categories for common use cases
 */
DECLARE_TASK_CATEGORIES(
        BuiltinTaskCategory,
        Default,      //!< General purpose tasks
        HighPriority, //!< High priority tasks that need immediate attention
        LowPriority,  //!< Low priority tasks that can be delayed
        IO,           //!< Input/Output bound tasks
        Compute,      //!< CPU intensive computational tasks
        Network,      //!< Network communication tasks
        Message       //!< Message related tasks
);

/**
 * Type-erased task category wrapper
 *
 * Provides a unified interface for both built-in and user-defined task
 * categories. Can hold any category type declared with DECLARE_TASK_CATEGORIES
 * macro and provides efficient comparison and string conversion.
 */
class TaskCategory {
private:
    std::uint64_t id_{};    //!< Unique category identifier
    std::string_view name_; //!< Category name for debugging/logging

    /**
     * Create full category name string combining type and value names
     * @tparam EnumType The enum type
     * @param[in] value The enum value
     * @return String like "TypeName::ValueName"
     */
    template <typename EnumType>
        requires std::is_enum_v<EnumType>
    static std::string create_full_name(const EnumType value) {
        static constexpr auto TYPE_NAME = get_category_type_name(static_cast<EnumType *>(nullptr));
        const auto value_name = ::wise_enum::to_string(value);
        return std::format("{}::{}", TYPE_NAME, value_name);
    }

public:
    /**
     * Constructor from any wise_enum category type
     * @param[in] value Category enum value
     */
    template <typename EnumType>
        requires std::is_enum_v<EnumType>
    explicit TaskCategory(const EnumType value)
            : id_{std::hash<std::string>{}(create_full_name(value))},
              name_{::wise_enum::to_string(value)} {}

    /**
     * Get category ID
     * @return Unique category identifier
     */
    [[nodiscard]] constexpr std::uint64_t id() const noexcept { return id_; }

    /**
     * Get category name
     * @return Category name as string view
     */
    [[nodiscard]] constexpr std::string_view name() const noexcept { return name_; }

    /**
     * Three-way comparison operator
     * @param[in] other TaskCategory to compare with
     * @return Comparison result based on category IDs
     */
    [[nodiscard]] constexpr auto operator<=>(const TaskCategory &other) const noexcept {
        return id_ <=> other.id_;
    }

    /**
     * Equality comparison
     * @param[in] other TaskCategory to compare with
     * @return true if categories have the same ID
     */
    [[nodiscard]] constexpr bool operator==(const TaskCategory &other) const noexcept = default;
};

} // namespace framework::task

/// @cond HIDE_FROM_DOXYGEN
/// Hash support for std::unordered_map and similar containers
template <> struct std::hash<framework::task::TaskCategory> {
    /// Hash function for TaskCategory
    [[nodiscard]] std::size_t
    operator()(const framework::task::TaskCategory &category) const noexcept {
        return std::hash<std::uint64_t>{}(category.id());
    }
};
/// @endcond

#endif // FRAMEWORK_TASK_TASK_CATEGORY_HPP
