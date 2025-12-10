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

#include "log/rt_log_macros.hpp"
#include "pipeline/imodule.hpp"
#include "sample_module_a.hpp"
#include "sample_module_b.hpp"
#include "sample_module_factories.hpp"

namespace framework::pipelines::samples {

// Namespace alias for compatibility with framework reorganization
namespace pipeline = ::framework::pipeline;

// =============================================================================
// SampleModuleAFactory Implementation
// =============================================================================

std::unique_ptr<pipeline::IModule> SampleModuleAFactory::create_module(
        std::string_view module_type,
        const std::string &instance_id,
        const std::any &static_params) {

    if (!supports_module_type(module_type)) {
        const std::string error_msg =
                std::format("SampleModuleAFactory: Unsupported module type '{}'", module_type);
        RT_LOG_ERROR("{}", error_msg);
        throw std::invalid_argument(error_msg);
    }

    // Extract static parameters from std::any
    try {
        const auto &params = std::any_cast<const SampleModuleA::StaticParams &>(static_params);

        RT_LOG_INFO(
                "SampleModuleAFactory: Creating SampleModuleA instance '{}' with "
                "tensor_size={}, trt_engine_path='{}'",
                instance_id,
                params.tensor_size,
                params.trt_engine_path);

        return std::make_unique<SampleModuleA>(instance_id, params);

    } catch (const std::bad_any_cast &e) {
        const std::string error_msg = std::format(
                "SampleModuleAFactory: Invalid static_params type for module '{}'. "
                "Expected SampleModuleA::StaticParams. Error: {}",
                instance_id,
                e.what());
        RT_LOG_ERROR("{}", error_msg);
        throw;
    }
}

bool SampleModuleAFactory::supports_module_type(std::string_view module_type) const {
    return module_type == "sample_module_a";
}

// =============================================================================
// SampleModuleBFactory Implementation
// =============================================================================

std::unique_ptr<pipeline::IModule> SampleModuleBFactory::create_module(
        std::string_view module_type,
        const std::string &instance_id,
        const std::any &static_params) {

    if (!supports_module_type(module_type)) {
        const std::string error_msg =
                std::format("SampleModuleBFactory: Unsupported module type '{}'", module_type);
        RT_LOG_ERROR("{}", error_msg);
        throw std::invalid_argument(error_msg);
    }

    // Extract static parameters from std::any
    try {
        const auto &params = std::any_cast<const SampleModuleB::StaticParams &>(static_params);

        RT_LOG_INFO(
                "SampleModuleBFactory: Creating SampleModuleB instance '{}' with "
                "tensor_size={}",
                instance_id,
                params.tensor_size);

        return std::make_unique<SampleModuleB>(instance_id, params);

    } catch (const std::bad_any_cast &e) {
        const std::string error_msg = std::format(
                "SampleModuleBFactory: Invalid static_params type for module '{}'. "
                "Expected SampleModuleB::StaticParams. Error: {}",
                instance_id,
                e.what());
        RT_LOG_ERROR("{}", error_msg);
        throw;
    }
}

bool SampleModuleBFactory::supports_module_type(std::string_view module_type) const {
    return module_type == "sample_module_b";
}

// =============================================================================
// SampleModuleFactory Implementation
// =============================================================================

SampleModuleFactory::SampleModuleFactory()
        : module_a_factory_(std::make_unique<SampleModuleAFactory>()),
          module_b_factory_(std::make_unique<SampleModuleBFactory>()) {
    RT_LOG_DEBUG("SampleModuleFactory: Initialized with sub-factories");
}

std::unique_ptr<pipeline::IModule> SampleModuleFactory::create_module(
        std::string_view module_type,
        const std::string &instance_id,
        const std::any &static_params) {

    // Delegate to appropriate sub-factory
    if (module_a_factory_->supports_module_type(module_type)) {
        return module_a_factory_->create_module(module_type, instance_id, static_params);
    }

    if (module_b_factory_->supports_module_type(module_type)) {
        return module_b_factory_->create_module(module_type, instance_id, static_params);
    }

    // No sub-factory supports this module type
    const std::string error_msg = std::format(
            "SampleModuleFactory: Unsupported module type '{}'. "
            "Supported types: 'sample_module_a', 'sample_module_b'",
            module_type);
    RT_LOG_ERROR("{}", error_msg);
    throw std::invalid_argument(error_msg);
}

bool SampleModuleFactory::supports_module_type(std::string_view module_type) const {
    return module_a_factory_->supports_module_type(module_type) ||
           module_b_factory_->supports_module_type(module_type);
}

} // namespace framework::pipelines::samples
