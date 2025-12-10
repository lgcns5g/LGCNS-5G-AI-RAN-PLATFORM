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

#ifndef RAN_TRT_PLUGIN_BASE_HPP
#define RAN_TRT_PLUGIN_BASE_HPP

#include <cstddef>
#include <cstdint>
#include <exception>
#include <format>
#include <iostream>
#include <string>
#include <string_view>

#include <NvInfer.h>

namespace ran::trt_plugin {

/**
 * @class TrtPluginBase
 * CRTP base class for TensorRT V3 plugins
 *
 * Provides common implementations for IPluginV3 interface methods that are
 * identical across all plugins. Uses CRTP pattern for zero runtime overhead.
 *
 * Derived class requirements:
 * - Must inherit from TrtPluginBase<DerivedClass>
 * - Must define static constexpr members: PLUGIN_TYPE and PLUGIN_VERSION
 * - Must implement plugin-specific methods (enqueue, getOutputShapes, etc.)
 *
 * @tparam Derived The derived plugin class (CRTP pattern)
 *
 * @example
 * class MyPlugin : public TrtPluginBase<MyPlugin> {
 * public:
 *     static constexpr const char* PLUGIN_TYPE = "MyPlugin";
 *     static constexpr const char* PLUGIN_VERSION = "1";
 *     // ... implement plugin-specific methods
 * };
 */
template <typename Derived>
class TrtPluginBase : public nvinfer1::IPluginV3,
                      public nvinfer1::IPluginV3OneCore,
                      public nvinfer1::IPluginV3OneBuild,
                      public nvinfer1::IPluginV3OneRuntime {
public:
    // Prevent accidental copying/moving (plugins should use clone())
    TrtPluginBase(const TrtPluginBase &) = delete;
    TrtPluginBase &operator=(const TrtPluginBase &) = delete;
    TrtPluginBase(TrtPluginBase &&) = delete;
    TrtPluginBase &operator=(TrtPluginBase &&) = delete;

    /**
     * Virtual destructor for proper cleanup
     */
    ~TrtPluginBase() override = default;

    // ========================================================================
    // IPluginV3 Interface - Common implementation
    // ========================================================================

    /**
     * Returns capability interface for the requested type
     *
     * This implementation is identical for all plugins and routes to the
     * appropriate interface based on the capability type.
     *
     * @param[in] type The capability type being requested
     * @return Pointer to the capability interface, or nullptr if unsupported
     */
    [[nodiscard]] nvinfer1::IPluginCapability *
    getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override {
        try {
            switch (type) {
            case nvinfer1::PluginCapabilityType::kBUILD:
                return static_cast<nvinfer1::IPluginV3OneBuild *>(this);
            case nvinfer1::PluginCapabilityType::kRUNTIME:
                return static_cast<nvinfer1::IPluginV3OneRuntime *>(this);
            case nvinfer1::PluginCapabilityType::kCORE:
                return static_cast<nvinfer1::IPluginV3OneCore *>(this);
            }
        } catch (const std::exception &e) {
            std::cerr << std::format("Exception in getCapabilityInterface: {}\n", e.what());
        } catch (...) {
            std::cerr << std::format("Unknown exception in getCapabilityInterface\n");
        }
        return nullptr;
    }

    // ========================================================================
    // IPluginV3OneCore Interface - Common implementation
    // ========================================================================

    /**
     * Returns plugin type name
     *
     * Uses CRTP to access the derived class's static PLUGIN_TYPE member.
     *
     * @return C-string containing the plugin type name
     */
    [[nodiscard]] nvinfer1::AsciiChar const *getPluginName() const noexcept override {
        return Derived::PLUGIN_TYPE;
    }

    /**
     * Returns plugin version string
     *
     * Uses CRTP to access the derived class's static PLUGIN_VERSION member.
     *
     * @return C-string containing the plugin version
     */
    [[nodiscard]] nvinfer1::AsciiChar const *getPluginVersion() const noexcept override {
        return Derived::PLUGIN_VERSION;
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

    // ========================================================================
    // IPluginV3OneBuild Interface - Default implementations
    // ========================================================================

    /**
     * Configures the plugin for the given input/output configuration
     *
     * Default implementation performs no configuration. Derived classes can
     * override if they need custom configuration logic.
     *
     * @param[in] in Array of input tensor descriptions
     * @param[in] nb_inputs Number of input tensors
     * @param[in] out Array of output tensor descriptions
     * @param[in] nb_outputs Number of output tensors
     * @return 0 on success
     */
    int32_t configurePlugin(
            [[maybe_unused]] nvinfer1::DynamicPluginTensorDesc const *in,
            [[maybe_unused]] int32_t nb_inputs,
            [[maybe_unused]] nvinfer1::DynamicPluginTensorDesc const *out,
            [[maybe_unused]] int32_t nb_outputs) noexcept override {
        return 0;
    }

    /**
     * Returns the workspace size required by the plugin
     *
     * Default implementation returns 0 (no workspace needed). Derived classes
     * can override if they require workspace memory.
     *
     * @param[in] inputs Array of input tensor descriptions
     * @param[in] nb_inputs Number of input tensors
     * @param[in] outputs Array of output tensor descriptions
     * @param[in] nb_outputs Number of output tensors
     * @return Required workspace size in bytes (default: 0)
     */
    [[nodiscard]] size_t getWorkspaceSize(
            [[maybe_unused]] nvinfer1::DynamicPluginTensorDesc const *inputs,
            [[maybe_unused]] int32_t nb_inputs,
            [[maybe_unused]] nvinfer1::DynamicPluginTensorDesc const *outputs,
            [[maybe_unused]] int32_t nb_outputs) const noexcept override {
        return 0;
    }

    // ========================================================================
    // IPluginV3OneRuntime Interface - Default implementations
    // ========================================================================

    /**
     * Handles dynamic shape changes during runtime
     *
     * Default implementation performs no special handling. Derived classes can
     * override if they need custom shape change logic.
     *
     * @param[in] in Array of input tensor descriptions
     * @param[in] nb_inputs Number of input tensors
     * @param[in] out Array of output tensor descriptions
     * @param[in] nb_outputs Number of output tensors
     * @return 0 on success
     */
    int32_t onShapeChange(
            [[maybe_unused]] nvinfer1::PluginTensorDesc const *in,
            [[maybe_unused]] int32_t nb_inputs,
            [[maybe_unused]] nvinfer1::PluginTensorDesc const *out,
            [[maybe_unused]] int32_t nb_outputs) noexcept override {
        return 0;
    }

    /**
     * Attaches the plugin to a resource context
     *
     * Default implementation creates a clone of the plugin for the new context.
     * Derived classes can override if they need custom context handling.
     *
     * @param[in] context Resource context provided by TensorRT
     * @return Pointer to a new plugin instance for the context
     */
    [[nodiscard]] nvinfer1::IPluginV3 *
    attachToContext([[maybe_unused]] nvinfer1::IPluginResourceContext *context) noexcept override {
        return clone();
    }

protected:
    /**
     * Protected constructor with plugin name and namespace
     *
     * @param[in] name Plugin instance name
     * @param[in] name_space Plugin namespace (defaults to empty)
     */
    explicit TrtPluginBase(const std::string_view name, const std::string_view name_space = "")
            : m_plugin_name_(name), m_namespace_(name_space) {}

    const std::string m_plugin_name_; //!< Plugin instance name (immutable)
    const std::string m_namespace_;   //!< Plugin namespace (immutable)
};

} // namespace ran::trt_plugin

#endif // RAN_TRT_PLUGIN_BASE_HPP
