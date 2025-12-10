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

// Factory implementations for OrderKernelModule and OrderKernelPipeline

#include <any>
#include <format>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <quill/LogMacros.h>

#include "fronthaul/fronthaul_log.hpp"
#include "fronthaul/order_kernel_factories.hpp"
#include "fronthaul/order_kernel_module.hpp"
#include "fronthaul/order_kernel_pipeline.hpp"
#include "log/rt_log_macros.hpp"
#include "pipeline/imodule.hpp"
#include "pipeline/ipipeline.hpp"
#include "pipeline/types.hpp"

namespace ran::fronthaul {

// Namespace alias for cleaner code
namespace pipeline = framework::pipeline;

// =============================================================================
// OrderKernelModuleFactory Implementation
// =============================================================================

std::unique_ptr<pipeline::IModule> OrderKernelModuleFactory::create_module(
        std::string_view module_type,
        const std::string &instance_id,
        const std::any &static_params) {

    if (!supports_module_type(module_type)) {
        const std::string error_msg =
                std::format("OrderKernelModuleFactory: Unsupported module type '{}'", module_type);
        RT_LOGC_ERROR(FronthaulKernels::OrderFactory, "{}", error_msg);
        throw std::invalid_argument(error_msg);
    }

    // Extract static parameters from std::any
    try {
        const auto &params = std::any_cast<const OrderKernelModule::StaticParams &>(static_params);

        RT_LOGC_INFO(
                FronthaulKernels::OrderFactory,
                "OrderKernelModuleFactory: Creating OrderKernelModule instance '{}' with "
                "execution_mode={}, gdr_handle={}",
                instance_id,
                params.execution_mode == pipeline::ExecutionMode::Graph ? "Graph" : "Stream",
                // NOLINTNEXTLINE(bugprone-multi-level-implicit-pointer-conversion)
                static_cast<void *>(params.gdr_handle));

        return std::make_unique<OrderKernelModule>(instance_id, params);

    } catch (const std::bad_any_cast &e) {
        const std::string error_msg = std::format(
                "OrderKernelModuleFactory: Invalid static_params type for module '{}'. "
                "Expected OrderKernelModule::StaticParams. Error: {}",
                instance_id,
                e.what());
        RT_LOGC_ERROR(FronthaulKernels::OrderFactory, "{}", error_msg);
        throw;
    }
}

bool OrderKernelModuleFactory::supports_module_type(std::string_view module_type) const {
    return module_type == "order_kernel_module";
}

std::unique_ptr<OrderKernelModule> OrderKernelModuleFactory::create_order_kernel_module(
        const std::string &instance_id, const std::any &static_params) {

    // Delegate to virtual method
    auto base_module = create_module("order_kernel_module", instance_id, static_params);

    // Safe static_cast since we control the factory and know it returns OrderKernelModule
    return std::unique_ptr<OrderKernelModule>(
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
            static_cast<OrderKernelModule *>(base_module.release()));
}

// =============================================================================
// OrderKernelPipelineFactory Implementation
// =============================================================================

std::unique_ptr<pipeline::IPipeline> OrderKernelPipelineFactory::create_pipeline(
        std::string_view pipeline_type,
        const std::string &pipeline_id,
        const pipeline::PipelineSpec &spec) {

    if (!is_pipeline_type_supported(pipeline_type)) {
        const std::string error_msg = std::format(
                "OrderKernelPipelineFactory: Unsupported pipeline type '{}'", pipeline_type);
        RT_LOGC_ERROR(FronthaulKernels::OrderFactory, "{}", error_msg);
        throw std::invalid_argument(error_msg);
    }

    if (doca_rxq_params_ == nullptr) {
        const std::string error_msg =
                std::format("OrderKernelPipelineFactory: DOCA RX queue parameters not set. "
                            "Call set_doca_params() before create_pipeline()");
        RT_LOGC_ERROR(FronthaulKernels::OrderFactory, "{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    RT_LOGC_INFO(
            FronthaulKernels::OrderFactory,
            "OrderKernelPipelineFactory: Creating OrderKernelPipeline '{}' with dedicated module "
            "factory",
            pipeline_id);

    // Create a fresh module factory for this pipeline (pipeline takes ownership)
    auto pipeline_module_factory = std::make_unique<OrderKernelModuleFactory>();

    return std::make_unique<OrderKernelPipeline>(
            pipeline_id, std::move(pipeline_module_factory), spec, doca_rxq_params_);
}

bool OrderKernelPipelineFactory::is_pipeline_type_supported(std::string_view pipeline_type) const {
    return pipeline_type == "order_kernel_pipeline";
}

std::vector<std::string> OrderKernelPipelineFactory::get_supported_pipeline_types() const {
    return {"order_kernel_pipeline"};
}

std::unique_ptr<OrderKernelPipeline> OrderKernelPipelineFactory::create_order_kernel_pipeline(
        const std::string &pipeline_id, const pipeline::PipelineSpec &spec) {

    // Delegate to virtual method
    auto base_pipeline = create_pipeline("order_kernel_pipeline", pipeline_id, spec);

    // Safe static_cast since we control the factory and know it returns OrderKernelPipeline
    return std::unique_ptr<OrderKernelPipeline>(
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
            static_cast<OrderKernelPipeline *>(base_pipeline.release()));
}

} // namespace ran::fronthaul
