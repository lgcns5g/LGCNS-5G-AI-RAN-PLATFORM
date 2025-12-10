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

#ifndef FRAMEWORK_PIPELINES_SAMPLE_MODULE_FACTORIES_HPP
#define FRAMEWORK_PIPELINES_SAMPLE_MODULE_FACTORIES_HPP

#include <any>
#include <memory>
#include <string>
#include <string_view>

#include "pipeline/imodule.hpp"
#include "pipeline/imodule_factory.hpp"

namespace framework::pipelines::samples {

// Namespace alias for compatibility with framework reorganization
namespace pipeline = ::framework::pipeline;

/**
 * Factory for creating SampleModuleA instances
 *
 * Supports module type: "sample_module_a"
 * Accepts SampleModuleA::StaticParams via std::any in create_module()
 */
class SampleModuleAFactory final : public pipeline::IModuleFactory {
public:
    /**
     * Default constructor
     */
    SampleModuleAFactory() = default;

    /**
     * Destructor
     */
    ~SampleModuleAFactory() override = default;

    // Non-copyable, non-movable
    SampleModuleAFactory(const SampleModuleAFactory &) = delete;
    SampleModuleAFactory &operator=(const SampleModuleAFactory &) = delete;
    SampleModuleAFactory(SampleModuleAFactory &&) = delete;
    SampleModuleAFactory &operator=(SampleModuleAFactory &&) = delete;

    /**
     * Create a SampleModuleA instance
     *
     * @param[in] module_type The type of module to create (must be
     * "sample_module_a")
     * @param[in] instance_id Unique identifier for this module instance
     * @param[in] static_params Type-erased SampleModuleA::StaticParams
     * @return Unique pointer to the created module
     * @throws std::invalid_argument if module_type is not "sample_module_a"
     * @throws std::bad_any_cast if static_params type doesn't match
     */
    [[nodiscard]] std::unique_ptr<pipeline::IModule> create_module(
            std::string_view module_type,
            const std::string &instance_id,
            const std::any &static_params) override;

    /**
     * Check if a module type is supported
     *
     * @param[in] module_type The type of module to check
     * @return true if module_type is "sample_module_a", false otherwise
     */
    [[nodiscard]] bool supports_module_type(std::string_view module_type) const override;
};

/**
 * Factory for creating SampleModuleB instances
 *
 * Supports module type: "sample_module_b"
 * Accepts SampleModuleB::StaticParams via std::any in create_module()
 */
class SampleModuleBFactory final : public pipeline::IModuleFactory {
public:
    /**
     * Default constructor
     */
    SampleModuleBFactory() = default;

    /**
     * Destructor
     */
    ~SampleModuleBFactory() override = default;

    // Non-copyable, non-movable
    SampleModuleBFactory(const SampleModuleBFactory &) = delete;
    SampleModuleBFactory &operator=(const SampleModuleBFactory &) = delete;
    SampleModuleBFactory(SampleModuleBFactory &&) = delete;
    SampleModuleBFactory &operator=(SampleModuleBFactory &&) = delete;

    /**
     * Create a SampleModuleB instance
     *
     * @param[in] module_type The type of module to create (must be
     * "sample_module_b")
     * @param[in] instance_id Unique identifier for this module instance
     * @param[in] static_params Type-erased SampleModuleB::StaticParams
     * @return Unique pointer to the created module
     * @throws std::invalid_argument if module_type is not "sample_module_b"
     * @throws std::bad_any_cast if static_params type doesn't match
     */
    [[nodiscard]] std::unique_ptr<pipeline::IModule> create_module(
            std::string_view module_type,
            const std::string &instance_id,
            const std::any &static_params) override;

    /**
     * Check if a module type is supported
     *
     * @param[in] module_type The type of module to check
     * @return true if module_type is "sample_module_b", false otherwise
     */
    [[nodiscard]] bool supports_module_type(std::string_view module_type) const override;
};

/**
 * Combined factory for all sample pipeline modules
 *
 * Aggregates SampleModuleAFactory and SampleModuleBFactory.
 * Supports module types: "sample_module_a", "sample_module_b"
 *
 * This factory delegates to the appropriate sub-factory based on module type.
 */
class SampleModuleFactory final : public pipeline::IModuleFactory {
public:
    /**
     * Constructor - initializes sub-factories
     */
    explicit SampleModuleFactory();

    /**
     * Destructor
     */
    ~SampleModuleFactory() override = default;

    // Non-copyable, non-movable
    SampleModuleFactory(const SampleModuleFactory &) = delete;
    SampleModuleFactory &operator=(const SampleModuleFactory &) = delete;
    SampleModuleFactory(SampleModuleFactory &&) = delete;
    SampleModuleFactory &operator=(SampleModuleFactory &&) = delete;

    /**
     * Create a module instance
     *
     * Delegates to the appropriate sub-factory based on module_type.
     *
     * @param[in] module_type The type of module to create
     *   ("sample_module_a" or "sample_module_b")
     * @param[in] instance_id Unique identifier for this module instance
     * @param[in] static_params Type-erased module static parameters
     * @return Unique pointer to the created module
     * @throws std::invalid_argument if module_type is not supported
     * @throws std::bad_any_cast if static_params type doesn't match module
     * requirements
     */
    [[nodiscard]] std::unique_ptr<pipeline::IModule> create_module(
            std::string_view module_type,
            const std::string &instance_id,
            const std::any &static_params) override;

    /**
     * Check if a module type is supported
     *
     * @param[in] module_type The type of module to check
     * @return true if supported by any sub-factory, false otherwise
     */
    [[nodiscard]] bool supports_module_type(std::string_view module_type) const override;

private:
    std::unique_ptr<SampleModuleAFactory> module_a_factory_; //!< Factory for SampleModuleA
    std::unique_ptr<SampleModuleBFactory> module_b_factory_; //!< Factory for SampleModuleB
};

} // namespace framework::pipelines::samples

#endif // FRAMEWORK_PIPELINES_SAMPLE_MODULE_FACTORIES_HPP
