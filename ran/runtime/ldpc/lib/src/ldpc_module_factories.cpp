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

#include "ldpc/crc_decoder_module.hpp"
#include "ldpc/ldpc_decoder_module.hpp"
#include "ldpc/ldpc_derate_match_module.hpp"
#include "ldpc/ldpc_log.hpp"
#include "ldpc/ldpc_module_factories.hpp"
#include "log/rt_log_macros.hpp"
#include "pipeline/imodule.hpp"

namespace ran::ldpc {

namespace pipeline = framework::pipeline;

// =============================================================================
// LdpcDecoderModuleFactory Implementation
// =============================================================================

std::unique_ptr<pipeline::IModule> LdpcDecoderModuleFactory::create_module(
        std::string_view module_type,
        const std::string &instance_id,
        const std::any &static_params) {

    if (!supports_module_type(module_type)) {
        const std::string error_msg = std::format("Unsupported module type '{}'", module_type);
        RT_LOGC_ERROR(LdpcComponent::LdpcDecoderModuleFactory, "{}", error_msg);
        throw std::invalid_argument(error_msg);
    }

    // Extract static parameters from std::any
    try {
        const auto &params =
                std::any_cast<const ran::ldpc::LdpcDecoderModule::StaticParams &>(static_params);

        RT_LOGC_INFO(
                LdpcComponent::LdpcDecoderModuleFactory,
                "Creating LdpcDecoderModule instance '{}' with "
                "clamp_value={}, max_num_iterations={}, "
                "max_num_cbs_per_tb={}, max_num_tbs={}, normalization_factor={}, "
                "max_iterations_method={}, max_num_ldpc_het_configs={}",
                instance_id,
                params.clamp_value,
                params.max_num_iterations,
                params.max_num_cbs_per_tb,
                params.max_num_tbs,
                params.normalization_factor,
                static_cast<int>(params.max_iterations_method),
                params.max_num_ldpc_het_configs);

        return std::make_unique<ran::ldpc::LdpcDecoderModule>(instance_id, params);

    } catch (const std::bad_any_cast &e) {
        const std::string error_msg = std::format(
                "Invalid static_params type for module '{}'. "
                "Expected LdpcDecoderModule::StaticParams. Error: {}",
                instance_id,
                e.what());
        RT_LOGC_ERROR(LdpcComponent::LdpcDecoderModuleFactory, "{}", error_msg);
        throw;
    }
}

bool LdpcDecoderModuleFactory::supports_module_type(std::string_view module_type) const {
    return module_type == "ldpc_decoder_module";
}

// =============================================================================
// LdpcDerateMatchModuleFactory Implementation
// =============================================================================

std::unique_ptr<pipeline::IModule> LdpcDerateMatchModuleFactory::create_module(
        std::string_view module_type,
        const std::string &instance_id,
        const std::any &static_params) {

    if (!supports_module_type(module_type)) {
        const std::string error_msg = std::format("Unsupported module type '{}'", module_type);
        RT_LOGC_ERROR(LdpcComponent::LdpcDerateMatchModuleFactory, "{}", error_msg);
        throw std::invalid_argument(error_msg);
    }

    // Extract static parameters from std::any
    try {
        const auto &params = std::any_cast<const ran::ldpc::LdpcDerateMatchModule::StaticParams &>(
                static_params);

        RT_LOGC_INFO(
                LdpcComponent::LdpcDerateMatchModuleFactory,
                "Creating LdpcDerateMatchModule instance '{}' with "
                "enable_scrambling={}, max_num_tbs={}, max_num_cbs_per_tb={}, "
                "max_num_rm_llrs_per_cb={}, max_num_ue_grps={}",
                instance_id,
                params.enable_scrambling,
                params.max_num_tbs,
                params.max_num_cbs_per_tb,
                params.max_num_rm_llrs_per_cb,
                params.max_num_ue_grps);

        return std::make_unique<ran::ldpc::LdpcDerateMatchModule>(instance_id, params);

    } catch (const std::bad_any_cast &e) {
        const std::string error_msg = std::format(
                "Invalid static_params type for module '{}'. "
                "Expected LdpcDerateMatchModule::StaticParams. Error: {}",
                instance_id,
                e.what());
        RT_LOGC_ERROR(LdpcComponent::LdpcDerateMatchModuleFactory, "{}", error_msg);
        throw;
    }
}

bool LdpcDerateMatchModuleFactory::supports_module_type(std::string_view module_type) const {
    return module_type == "ldpc_derate_match_module";
}

// =============================================================================
// CrcDecoderModuleFactory Implementation
// =============================================================================

std::unique_ptr<pipeline::IModule> CrcDecoderModuleFactory::create_module(
        std::string_view module_type,
        const std::string &instance_id,
        const std::any &static_params) {

    if (!supports_module_type(module_type)) {
        const std::string error_msg = std::format("Unsupported module type '{}'", module_type);
        RT_LOGC_ERROR(LdpcComponent::CrcDecoderModuleFactory, "{}", error_msg);
        throw std::invalid_argument(error_msg);
    }

    // Extract static parameters from std::any
    try {
        const auto &params =
                std::any_cast<const ran::ldpc::CrcDecoderModule::StaticParams &>(static_params);

        RT_LOGC_INFO(
                LdpcComponent::CrcDecoderModuleFactory,
                "Creating CrcDecoderModule instance '{}' with "
                "reverse_bytes={}, max_num_cbs_per_tb={}, max_num_tbs={}",
                instance_id,
                params.reverse_bytes,
                params.max_num_cbs_per_tb,
                params.max_num_tbs);

        return std::make_unique<ran::ldpc::CrcDecoderModule>(instance_id, params);

    } catch (const std::bad_any_cast &e) {
        const std::string error_msg = std::format(
                "Invalid static_params type for module '{}'. "
                "Expected CrcDecoderModule::StaticParams. Error: {}",
                instance_id,
                e.what());
        RT_LOGC_ERROR(LdpcComponent::CrcDecoderModuleFactory, "{}", error_msg);
        throw;
    }
}

bool CrcDecoderModuleFactory::supports_module_type(std::string_view module_type) const {
    return module_type == "crc_decoder_module";
}

} // namespace ran::ldpc
