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

#include <format>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "log/rt_log_macros.hpp"
#include "pipeline/ipipeline.hpp"
#include "pipeline/types.hpp"
#include "sample_pipeline.hpp"
#include "sample_pipeline_factory.hpp"

namespace framework::pipelines::samples {

// Namespace alias for compatibility with framework reorganization
namespace pipeline = ::framework::pipeline;

SamplePipelineFactory::SamplePipelineFactory(
        gsl_lite::not_null<pipeline::IModuleFactory *> module_factory)
        : module_factory_(std::move(module_factory)) {
    RT_LOG_DEBUG("SamplePipelineFactory: Initialized with module factory");
}

std::unique_ptr<pipeline::IPipeline> SamplePipelineFactory::create_pipeline(
        std::string_view pipeline_type,
        const std::string &pipeline_id,
        const pipeline::PipelineSpec &spec) {

    if (!is_pipeline_type_supported(pipeline_type)) {
        const std::string error_msg =
                std::format("SamplePipelineFactory: Unsupported pipeline type '{}'", pipeline_type);
        RT_LOG_ERROR("{}", error_msg);
        throw std::invalid_argument(error_msg);
    }

    RT_LOG_INFO(
            "SamplePipelineFactory: Creating pipeline '{}' (type: '{}') with {} "
            "modules",
            pipeline_id,
            pipeline_type,
            spec.modules.size());

    // Create pipeline - it will use module_factory_ to create modules
    auto pipeline = std::make_unique<SamplePipeline>(
            pipeline_id, gsl_lite::not_null<pipeline::IModuleFactory *>(module_factory_), spec);

    RT_LOG_INFO("SamplePipelineFactory: Successfully created pipeline '{}'", pipeline_id);

    return pipeline;
}

bool SamplePipelineFactory::is_pipeline_type_supported(std::string_view pipeline_type) const {
    return pipeline_type == "sample";
}

std::vector<std::string> SamplePipelineFactory::get_supported_pipeline_types() const {
    return {"sample"};
}

} // namespace framework::pipelines::samples
