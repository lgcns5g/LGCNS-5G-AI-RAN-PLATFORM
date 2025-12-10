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

#include <any>
#include <format>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

#include <quill/LogMacros.h> // for QUILL_LOG_ERROR, QUILL_LOG_DEBUG

#include "ldpc/ldpc_module_factories.hpp"
#include "log/rt_log_macros.hpp"
#include "pipeline/imodule.hpp"
#include "pipeline/imodule_factory.hpp"
#include "pusch/inner_rx_module.hpp"
#include "pusch/pusch_log.hpp"
#include "pusch/pusch_module_factories.hpp"

namespace ran::pusch {

namespace pipeline = framework::pipeline;

// =============================================================================
// InnerRxModuleFactory Implementation
// =============================================================================

std::unique_ptr<pipeline::IModule> InnerRxModuleFactory::create_module(
        std::string_view module_type,
        const std::string &instance_id,
        const std::any &static_params) {

    if (!supports_module_type(module_type)) {
        const std::string error_msg =
                std::format("InnerRxModuleFactory: Unsupported module type '{}'", module_type);
        RT_LOGC_ERROR(PuschComponent::InnerRxModuleFactory, "{}", error_msg);
        throw std::invalid_argument(error_msg);
    }

    // Extract static parameters from std::any
    try {
        const auto &params = std::any_cast<const InnerRxModule::StaticParams &>(static_params);

        RT_LOGC_INFO(
                PuschComponent::InnerRxModuleFactory,
                "Creating InnerRxModule instance '{}'",
                instance_id);

        return std::make_unique<InnerRxModule>(instance_id, params);

    } catch (const std::bad_any_cast &e) {
        const std::string error_msg = std::format(
                "Invalid static_params type for module '{}'. "
                "Expected InnerRxModule::StaticParams. Error: {}",
                instance_id,
                e.what());
        RT_LOGC_ERROR(PuschComponent::InnerRxModuleFactory, "{}", error_msg);
        throw;
    }
}

bool InnerRxModuleFactory::supports_module_type(std::string_view module_type) const {
    return module_type == "inner_rx_module";
}

// =============================================================================
// PuschModuleFactory Implementation
// =============================================================================

PuschModuleFactory::PuschModuleFactory()
        : inner_rx_module_factory_(std::make_unique<InnerRxModuleFactory>()),
          ldpc_decoder_module_factory_(std::make_unique<ran::ldpc::LdpcDecoderModuleFactory>()),
          ldpc_derate_match_module_factory_(
                  std::make_unique<ran::ldpc::LdpcDerateMatchModuleFactory>()),
          crc_decoder_module_factory_(std::make_unique<ran::ldpc::CrcDecoderModuleFactory>()) {
    RT_LOGC_DEBUG(PuschComponent::PuschModuleFactory, "Initialized with sub-factories");
}

std::unique_ptr<pipeline::IModule> PuschModuleFactory::create_module(
        std::string_view module_type,
        const std::string &instance_id,
        const std::any &static_params) {

    // Delegate to appropriate sub-factory
    if (inner_rx_module_factory_->supports_module_type(module_type)) {
        return inner_rx_module_factory_->create_module(module_type, instance_id, static_params);
    } else if (ldpc_decoder_module_factory_->supports_module_type(module_type)) {
        return ldpc_decoder_module_factory_->create_module(module_type, instance_id, static_params);
    } else if (ldpc_derate_match_module_factory_->supports_module_type(module_type)) {
        return ldpc_derate_match_module_factory_->create_module(
                module_type, instance_id, static_params);
    } else if (crc_decoder_module_factory_->supports_module_type(module_type)) {
        return crc_decoder_module_factory_->create_module(module_type, instance_id, static_params);
    }

    // No sub-factory supports this module type
    const std::string error_msg = std::format(
            "PuschModuleFactory: Unsupported module type '{}'. "
            "Supported types: 'inner_rx_module', 'ldpc_decoder_module', "
            "'ldpc_derate_match_module', 'crc_decoder_module'",
            module_type);
    RT_LOGC_ERROR(PuschComponent::PuschModuleFactory, "{}", error_msg);
    throw std::invalid_argument(error_msg);
}

bool PuschModuleFactory::supports_module_type(std::string_view module_type) const {
    return (inner_rx_module_factory_->supports_module_type(module_type) ||
            ldpc_decoder_module_factory_->supports_module_type(module_type) ||
            ldpc_derate_match_module_factory_->supports_module_type(module_type) ||
            crc_decoder_module_factory_->supports_module_type(module_type));
}

} // namespace ran::pusch
