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

#ifndef FRAMEWORK_CORE_MODULE_FACTORY_HPP
#define FRAMEWORK_CORE_MODULE_FACTORY_HPP

#include <any>
#include <concepts>
#include <format>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

#include "pipeline/imodule.hpp"
#include "pipeline/imodule_factory.hpp"
#include "utils/string_hash.hpp"

namespace framework::pipeline {

/**
 * Concept for module creator callables.
 * Constrains types that can create module instances from ID and parameters.
 */
template <typename F>
concept ModuleCreator = std::invocable<F, const std::string &, const std::any &> &&
                        std::same_as<
                                std::invoke_result_t<F, const std::string &, const std::any &>,
                                std::unique_ptr<IModule>>;

/**
 * @class ModuleFactory
 * @brief Concrete implementation of IModuleFactory with runtime registration
 *
 * This factory uses a registry pattern allowing module types to be registered
 * at runtime. This provides flexibility for different applications to register
 * their specific module types without modifying the factory implementation.
 *
 * Example usage:
 * @code
 * ModuleFactory factory;
 * factory.register_module_type("gemm",
 *    [](const std::string& id, const std::any& params) {
 *      return std::make_unique<GemmModule>(id,
 *                                           std::any_cast<GemmConfig>(params));
 *     });
 *
 * auto module = factory.create_module("gemm", "gemm_0", gemm_config);
 * @endcode
 */
class ModuleFactory final : public IModuleFactory {
public:
    /**
     * Default constructor.
     */
    ModuleFactory() = default;

    /**
     * Destructor.
     */
    ~ModuleFactory() override = default;

    // Non-copyable, movable
    ModuleFactory(const ModuleFactory &) = delete;
    ModuleFactory &operator=(const ModuleFactory &) = delete;

    /**
     * Move constructor.
     */
    ModuleFactory(ModuleFactory &&) = default;

    /**
     * Move assignment operator.
     *
     * @return Reference to this object
     */
    ModuleFactory &operator=(ModuleFactory &&) = default;

    /**
     * Register a module type with its creator function.
     *
     * @param[in] module_type Type identifier for the module
     * @param[in] creator Function that creates instances of this module type
     * @throws std::invalid_argument if module_type is already registered
     */
    template <ModuleCreator Creator>
    void register_module_type(std::string_view module_type, Creator &&creator) {
        if (module_creators_.contains(module_type)) {
            throw std::invalid_argument(
                    std::format("Module type '{}' is already registered", module_type));
        }
        module_creators_[std::string(module_type)] = std::forward<Creator>(creator);
    }

    /**
     * Create a module of the specified type.
     *
     * @param[in] module_type The type of module to create
     * @param[in] instance_id The unique instance identifier for this module
     * @param[in] static_params Type-erased static parameters for module
     * initialization
     * @return Unique pointer to the created module
     * @throws std::invalid_argument if module_type is not supported
     * @throws std::bad_any_cast if static_params type doesn't match module
     * requirements
     */
    [[nodiscard]] std::unique_ptr<IModule> create_module(
            std::string_view module_type,
            const std::string &instance_id,
            const std::any &static_params) override;

    /**
     * Check if a module type is supported by this factory.
     *
     * @param[in] module_type The type of module to check
     * @return true if the module type is supported, false otherwise
     */
    [[nodiscard]] bool supports_module_type(std::string_view module_type) const override;

private:
    using StoredCreator =
            std::function<std::unique_ptr<IModule>(const std::string &, const std::any &)>;

    std::unordered_map<
            std::string,
            StoredCreator,
            utils::TransparentStringHash,
            std::equal_to<>>
            module_creators_; //!< Registry of module creators by type
};

} // namespace framework::pipeline

#endif // FRAMEWORK_CORE_MODULE_FACTORY_HPP
