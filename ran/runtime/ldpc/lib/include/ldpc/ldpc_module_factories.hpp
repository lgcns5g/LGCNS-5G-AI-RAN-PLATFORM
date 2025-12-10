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

#ifndef RAN_LDPC_MODULE_FACTORIES_HPP
#define RAN_LDPC_MODULE_FACTORIES_HPP

#include <any>
#include <memory>
#include <string>
#include <string_view>

#include "pipeline/imodule.hpp"
#include "pipeline/imodule_factory.hpp"

namespace ran::ldpc {

/**
 * Factory for creating LdpcDecoderModule instances
 *
 * Supports module type: "ldpc_decoder_module"
 * Accepts LdpcDecoderModule::StaticParams via std::any in create_module()
 */
class LdpcDecoderModuleFactory final : public framework::pipeline::IModuleFactory {
public:
    /**
     * Default constructor
     */
    LdpcDecoderModuleFactory() = default;

    /**
     * Destructor
     */
    ~LdpcDecoderModuleFactory() override = default;

    // Non-copyable, movable
    LdpcDecoderModuleFactory(const LdpcDecoderModuleFactory &) = delete;
    LdpcDecoderModuleFactory &operator=(const LdpcDecoderModuleFactory &) = delete;

    /**
     * Move constructor
     * @param[in,out] other Source factory to move from
     */
    LdpcDecoderModuleFactory(LdpcDecoderModuleFactory &&other) = default;

    /**
     * Move assignment operator
     * @param[in,out] other Source factory to move from
     * @return Reference to this object
     */
    LdpcDecoderModuleFactory &operator=(LdpcDecoderModuleFactory &&other) = default;

    /**
     * Create an LdpcDecoderModule instance
     *
     * @param[in] module_type The type of module to create (must be "ldpc_decoder_module")
     * @param[in] instance_id Unique identifier for this module instance
     * @param[in] static_params Type-erased LdpcDecoderModule::StaticParams
     * @return Unique pointer to the created module
     * @throws std::invalid_argument if module_type is not "ldpc_decoder_module"
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
     * @return true if module_type is "ldpc_decoder_module", false otherwise
     */
    [[nodiscard]] bool supports_module_type(std::string_view module_type) const override;
};

/**
 * Factory for creating LdpcDerateMatchModule instances
 *
 * Supports module type: "ldpc_derate_match_module"
 * Accepts LdpcDerateMatchModule::StaticParams via std::any in create_module()
 */
class LdpcDerateMatchModuleFactory final : public framework::pipeline::IModuleFactory {
public:
    /**
     * Default constructor
     */
    LdpcDerateMatchModuleFactory() = default;

    /**
     * Destructor
     */
    ~LdpcDerateMatchModuleFactory() override = default;

    // Non-copyable, movable
    LdpcDerateMatchModuleFactory(const LdpcDerateMatchModuleFactory &) = delete;
    LdpcDerateMatchModuleFactory &operator=(const LdpcDerateMatchModuleFactory &) = delete;

    /**
     * Move constructor
     * @param[in,out] other Source factory to move from
     */
    LdpcDerateMatchModuleFactory(LdpcDerateMatchModuleFactory &&other) = default;

    /**
     * Move assignment operator
     * @param[in,out] other Source factory to move from
     * @return Reference to this object
     */
    LdpcDerateMatchModuleFactory &operator=(LdpcDerateMatchModuleFactory &&other) = default;

    /**
     * Create an LdpcDerateMatchModule instance
     *
     * @param[in] module_type The type of module to create (must be "ldpc_derate_match_module")
     * @param[in] instance_id Unique identifier for this module instance
     * @param[in] static_params Type-erased LdpcDerateMatchModule::StaticParams
     * @return Unique pointer to the created module
     * @throws std::invalid_argument if module_type is not "ldpc_derate_match_module"
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
     * @return true if module_type is "ldpc_derate_match_module", false otherwise
     */
    [[nodiscard]] bool supports_module_type(std::string_view module_type) const override;
};

/**
 * Factory for creating CrcDecoderModule instances
 *
 * Supports module type: "crc_decoder_module"
 * Accepts CrcDecoderModule::StaticParams via std::any in create_module()
 */
class CrcDecoderModuleFactory final : public framework::pipeline::IModuleFactory {
public:
    /**
     * Default constructor
     */
    CrcDecoderModuleFactory() = default;

    /**
     * Destructor
     */
    ~CrcDecoderModuleFactory() override = default;

    // Non-copyable, movable
    CrcDecoderModuleFactory(const CrcDecoderModuleFactory &) = delete;
    CrcDecoderModuleFactory &operator=(const CrcDecoderModuleFactory &) = delete;

    /**
     * Move constructor
     * @param[in,out] other Source factory to move from
     */
    CrcDecoderModuleFactory(CrcDecoderModuleFactory &&other) = default;

    /**
     * Move assignment operator
     * @param[in,out] other Source factory to move from
     * @return Reference to this object
     */
    CrcDecoderModuleFactory &operator=(CrcDecoderModuleFactory &&other) = default;

    /**
     * Create a CrcDecoderModule instance
     *
     * @param[in] module_type The type of module to create (must be "crc_decoder_module")
     * @param[in] instance_id Unique identifier for this module instance
     * @param[in] static_params Type-erased CrcDecoderModule::StaticParams
     * @return Unique pointer to the created module
     * @throws std::invalid_argument if module_type is not "crc_decoder_module"
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
     * @return true if module_type is "crc_decoder_module", false otherwise
     */
    [[nodiscard]] bool supports_module_type(std::string_view module_type) const override;
};

} // namespace ran::ldpc

#endif // RAN_LDPC_MODULE_FACTORIES_HPP
