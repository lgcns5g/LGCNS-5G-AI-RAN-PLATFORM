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

#ifndef RAN_TRT_PLUGIN_CREATOR_BASE_HPP
#define RAN_TRT_PLUGIN_CREATOR_BASE_HPP

#include <cstdint>
#include <format>
#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <NvInfer.h>

namespace ran::trt_plugin {

/**
 * Extracts a field value from a PluginFieldCollection with optional default
 *
 * This template function provides a clean interface for parsing plugin fields
 * with automatic error logging when required fields are missing.
 *
 * @tparam T The type of the field value to extract
 * @tparam Default The type of the default value (deduced, typically T or std::nullopt_t)
 *
 * @param[in] fc Plugin field collection to search
 * @param[in] field_name Name of the field to find
 * @param[in] expected_type Expected TensorRT plugin field type
 * @param[in] default_value Optional default value; if std::nullopt, logs error when field is
 * missing
 *
 * @return The field value if found, otherwise the default value or default-constructed T
 *
 * @note If default_value is std::nullopt and field is not found, logs error and returns T{}
 * @note If field data is nullptr or type mismatches, logs error and returns default/T{}
 *
 * Usage examples:
 * @code
 * // With default value - no error if missing
 * auto size = get_plugin_field<std::int32_t>(fc, "fft_size", PluginFieldType::kINT32, 128);
 *
 * // Without default - logs error if missing
 * auto size = get_plugin_field<std::int32_t>(fc, "fft_size", PluginFieldType::kINT32);
 * @endcode
 */
template <typename T, typename Default = std::nullopt_t>
T get_plugin_field(
        const nvinfer1::PluginFieldCollection *fc,
        const std::string_view field_name,
        const nvinfer1::PluginFieldType expected_type,
        Default &&default_value = std::nullopt) {

    // Handle null or empty field collection
    if (fc == nullptr || fc->nbFields == 0) {
        if constexpr (std::is_same_v<std::decay_t<Default>, std::nullopt_t>) {
            std::cerr << std::format(
                    "PluginField '{}': field collection is null or empty, no default provided\n",
                    field_name);
            return T{};
        } else {
            return std::forward<Default>(default_value);
        }
    }

    // Search for the field
    const std::span<const nvinfer1::PluginField> fields(fc->fields, fc->nbFields);
    for (const auto &field : fields) {
        if (std::string_view{field.name} != field_name) {
            continue;
        }
        // Validate field type
        if (field.type != expected_type) {
            std::cerr << std::format(
                    "PluginField '{}': type mismatch (expected {}, got {})\n",
                    field_name,
                    static_cast<int>(expected_type),
                    static_cast<int>(field.type));
            if constexpr (std::is_same_v<std::decay_t<Default>, std::nullopt_t>) {
                return T{};
            } else {
                return std::forward<Default>(default_value);
            }
        }

        // Validate field data pointer
        if (field.data == nullptr) {
            std::cerr << std::format("PluginField '{}': field data pointer is null\n", field_name);
            if constexpr (std::is_same_v<std::decay_t<Default>, std::nullopt_t>) {
                return T{};
            } else {
                return std::forward<Default>(default_value);
            }
        }

        // Return the field value
        return *static_cast<const T *>(field.data);
    }

    // Field not found
    if constexpr (std::is_same_v<std::decay_t<Default>, std::nullopt_t>) {
        std::cerr << std::format(
                "PluginField '{}': required field not found in PluginFieldCollection\n",
                field_name);
        return T{};
    } else {
        return std::forward<Default>(default_value);
    }
}

/**
 * @class TrtPluginCreatorBase
 * CRTP base class for TensorRT V3 plugin creators
 *
 * Provides common implementations for IPluginCreatorV3One interface methods
 * that are identical across all plugin creators.
 *
 * Derived class requirements:
 * - Must inherit from TrtPluginCreatorBase<PluginType>
 * - PluginType must have static members: PLUGIN_TYPE and PLUGIN_VERSION
 * - Must implement createPlugin() method
 *
 * @tparam PluginType The plugin class this creator creates
 *
 * @example
 * class MyPluginCreator : public TrtPluginCreatorBase<MyPlugin> {
 * public:
 *     MyPluginCreator() {
 *         // Setup fields using addField() helper
 *     }
 *
 *     nvinfer1::IPluginV3 *createPlugin(...) override {
 *         // Create plugin instance
 *     }
 * };
 */
template <typename PluginType> class TrtPluginCreatorBase : public nvinfer1::IPluginCreatorV3One {
public:
    /**
     * Constructor with required namespace
     *
     * @param[in] name_space Plugin namespace
     */
    explicit TrtPluginCreatorBase(const std::string_view name_space) : m_namespace_(name_space) {
        m_field_collection_.nbFields = 0;
        m_field_collection_.fields = nullptr;
    }

    /**
     * Virtual destructor for proper cleanup
     */
    ~TrtPluginCreatorBase() override = default;

    // Delete copy and move operations
    TrtPluginCreatorBase(const TrtPluginCreatorBase &) = delete;
    TrtPluginCreatorBase &operator=(const TrtPluginCreatorBase &) = delete;
    TrtPluginCreatorBase(TrtPluginCreatorBase &&) = delete;
    TrtPluginCreatorBase &operator=(TrtPluginCreatorBase &&) = delete;

    // ========================================================================
    // IPluginCreatorV3One Interface - Common implementation
    // ========================================================================

    /**
     * Returns plugin type name
     *
     * Uses the PluginType's static PLUGIN_TYPE member.
     *
     * @return C-string containing the plugin type name
     */
    [[nodiscard]] nvinfer1::AsciiChar const *getPluginName() const noexcept override {
        return PluginType::PLUGIN_TYPE;
    }

    /**
     * Returns plugin version string
     *
     * Uses the PluginType's static PLUGIN_VERSION member.
     *
     * @return C-string containing the plugin version
     */
    [[nodiscard]] nvinfer1::AsciiChar const *getPluginVersion() const noexcept override {
        return PluginType::PLUGIN_VERSION;
    }

    /**
     * Returns plugin namespace
     *
     * @return C-string containing the plugin namespace
     */
    [[nodiscard]] nvinfer1::AsciiChar const *getPluginNamespace() const noexcept override {
        if (m_namespace_.empty()) {
            return "";
        }
        return m_namespace_.c_str();
    }

    /**
     * Returns the field collection
     *
     * @return Pointer to the plugin field collection
     */
    [[nodiscard]] nvinfer1::PluginFieldCollection const *getFieldNames() noexcept override {
        return &m_field_collection_;
    }

protected:
    nvinfer1::PluginFieldCollection m_field_collection_{}; //!< Field collection
    std::vector<nvinfer1::PluginField> m_plugin_fields_;   //!< Vector of fields
    std::string m_namespace_;                              //!< Plugin namespace
};

} // namespace ran::trt_plugin

#endif // RAN_TRT_PLUGIN_CREATOR_BASE_HPP
