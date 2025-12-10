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

#ifndef RAN_PUSCH_MODULE_FACTORIES_HPP
#define RAN_PUSCH_MODULE_FACTORIES_HPP

#include <any>
#include <memory>
#include <string>
#include <string_view>

#include "pipeline/imodule.hpp"
#include "pipeline/imodule_factory.hpp"

namespace ran::pusch {

/**
 * Factory for creating InnerRxModule instances
 *
 * Supports module type: "inner_rx_module"
 * Accepts InnerRxModule::StaticParams via std::any in create_module()
 */
class InnerRxModuleFactory final : public framework::pipeline::IModuleFactory {
public:
    /**
     * Default constructor
     */
    InnerRxModuleFactory() = default;

    /**
     * Destructor
     */
    ~InnerRxModuleFactory() override = default;

    // Non-copyable, movable (consistent with IModuleFactory interface)
    InnerRxModuleFactory(const InnerRxModuleFactory &) = delete;
    InnerRxModuleFactory &operator=(const InnerRxModuleFactory &) = delete;

    /**
     * Move constructor
     * @param[in,out] other Source factory to move from
     */
    InnerRxModuleFactory(InnerRxModuleFactory &&other) = default;

    /**
     * Move assignment operator
     * @param[in,out] other Source factory to move from
     * @return Reference to this object
     */
    InnerRxModuleFactory &operator=(InnerRxModuleFactory &&other) = default;

    /**
     * Create a InnerRxModule instance
     *
     * @param[in] module_type The type of module to create (must be "inner_rx_module")
     * @param[in] instance_id Unique identifier for this module instance
     * @param[in] static_params Type-erased InnerRxModule::StaticParams
     * @return Unique pointer to the created module
     * @throws std::invalid_argument if module_type is not "inner_rx_module"
     * @throws std::bad_any_cast if static_params type doesn't match
     */
    [[nodiscard]] std::unique_ptr<framework::pipeline::IModule> create_module(
            std::string_view module_type,
            const std::string &instance_id,
            const std::any &static_params) override;

    /**
     * Check if a module type is supported
     *
     * @param[in] module_type The type of module to check
     * @return true if module_type is "inner_rx_module", false otherwise
     */
    [[nodiscard]] bool supports_module_type(std::string_view module_type) const override;
};

/**
 * Combined factory for all PUSCH pipeline modules
 *
 * Aggregates InnerRxModuleFactory and LDPC module factories.
 * Supports module types: "inner_rx_module", "ldpc_decoder_module", "ldpc_derate_match_module",
 * "crc_decoder_module"
 *
 * This factory delegates to the appropriate sub-factory based on module type.
 */
class PuschModuleFactory final : public framework::pipeline::IModuleFactory {
public:
    /**
     * Constructor - initializes sub-factories
     */
    PuschModuleFactory();

    /**
     * Destructor
     */
    ~PuschModuleFactory() override = default;

    // Non-copyable, movable (consistent with IModuleFactory interface)
    PuschModuleFactory(const PuschModuleFactory &) = delete;
    PuschModuleFactory &operator=(const PuschModuleFactory &) = delete;

    /**
     * Move constructor
     * @param[in,out] other Source factory to move from
     */
    PuschModuleFactory(PuschModuleFactory &&other) = default;

    /**
     * Move assignment operator
     * @param[in,out] other Source factory to move from
     * @return Reference to this object
     */
    PuschModuleFactory &operator=(PuschModuleFactory &&other) = default;

    /**
     * Create a module instance
     *
     * Delegates to the appropriate sub-factory based on module_type.
     *
     * @param[in] module_type The type of module to create
     * @param[in] instance_id Unique identifier for this module instance
     * @param[in] static_params Type-erased module static parameters
     * @return Unique pointer to the created module
     * @throws std::invalid_argument if module_type is not supported
     * @throws std::bad_any_cast if static_params type doesn't match module requirements
     */
    [[nodiscard]] std::unique_ptr<framework::pipeline::IModule> create_module(
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
    std::unique_ptr<InnerRxModuleFactory> inner_rx_module_factory_; //!< Factory for InnerRxModule
    std::unique_ptr<framework::pipeline::IModuleFactory>
            ldpc_decoder_module_factory_; //!< Factory for LdpcDecoderModule
    std::unique_ptr<framework::pipeline::IModuleFactory>
            ldpc_derate_match_module_factory_; //!< Factory for LdpcDerateMatchModule
    std::unique_ptr<framework::pipeline::IModuleFactory>
            crc_decoder_module_factory_; //!< Factory for CrcDecoderModule
};

} // namespace ran::pusch

#endif // RAN_PUSCH_MODULE_FACTORIES_HPP
