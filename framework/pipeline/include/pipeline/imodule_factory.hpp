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

#ifndef FRAMEWORK_CORE_IMODULE_FACTORY_HPP
#define FRAMEWORK_CORE_IMODULE_FACTORY_HPP

#include <any>
#include <memory>
#include <string>
#include <string_view>

#include "pipeline/imodule.hpp"

namespace framework::pipeline {

/**
 * @class IModuleFactory
 * @brief Interface for creating modules dynamically.
 *
 * This interface defines the contract for factories that can create different
 * types of modules based on type identifiers and parameters.
 */
class IModuleFactory {
public:
    /**
     * Default constructor.
     */
    IModuleFactory() = default;

    /**
     * Virtual destructor.
     */
    virtual ~IModuleFactory() = default;

    /**
     * Move constructor.
     */
    IModuleFactory(IModuleFactory &&) = default;

    /**
     * Move assignment operator.
     * @return Reference to this object
     */
    IModuleFactory &operator=(IModuleFactory &&) = default;

    IModuleFactory(const IModuleFactory &) = delete;
    IModuleFactory &operator=(const IModuleFactory &) = delete;

    /**
     * Create a module of the specified type.
     *
     * @param[in] module_type The type of module to create (e.g., "gemm")
     * @param[in] instance_id The unique instance identifier for this module
     * @param[in] static_params Type-erased static parameters for module
     * initialization
     * @return Unique pointer to the created module
     * @throws std::invalid_argument if module_type is not supported
     * @throws std::bad_any_cast if static_params type doesn't match module
     * requirements
     */
    [[nodiscard]] virtual std::unique_ptr<IModule> create_module(
            std::string_view module_type,
            const std::string &instance_id,
            const std::any &static_params) = 0;

    /**
     * Check if a module type is supported by this factory.
     *
     * @param[in] module_type The type of module to check
     * @return true if the module type is supported, false otherwise
     */
    [[nodiscard]] virtual bool supports_module_type(std::string_view module_type) const = 0;
};

} // namespace framework::pipeline

#endif // FRAMEWORK_CORE_IMODULE_FACTORY_HPP
