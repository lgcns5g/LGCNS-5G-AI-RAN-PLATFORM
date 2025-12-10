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
#include <exception>
#include <format>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <quill/LogMacros.h>

#include "log/rt_log_macros.hpp"
#include "pipeline/imodule_factory.hpp"
#include "pipeline/ipipeline.hpp"
#include "pipeline/pipeline_factory.hpp"
#include "pipeline/types.hpp"
#include "utils/core_log.hpp"
#include "utils/string_hash.hpp"

namespace framework::pipeline {

PipelineFactory::PipelineFactory(IModuleFactory &module_factory) : module_factory_(module_factory) {
    RT_LOGC_DEBUG(utils::Core::CoreFactory, "PipelineFactory created");
}

void PipelineFactory::register_pipeline_type(
        std::string_view pipeline_type, PipelineCreator creator) {
    RT_LOGC_DEBUG(utils::Core::CoreFactory, "Registering pipeline type: {}", pipeline_type);

    if (pipeline_creators_.contains(pipeline_type)) {
        const std::string error_msg =
                std::format("Pipeline type '{}' is already registered", pipeline_type);
        RT_LOGC_ERROR(utils::Core::CoreFactory, "{}", error_msg);
        throw std::invalid_argument(error_msg);
    }

    pipeline_creators_[std::string(pipeline_type)] = std::move(creator);
    RT_LOGC_INFO(
            utils::Core::CoreFactory, "Pipeline type '{}' registered successfully", pipeline_type);
}

std::unique_ptr<IPipeline> PipelineFactory::create_pipeline(
        std::string_view pipeline_type, const std::string &pipeline_id, const PipelineSpec &spec) {

    RT_LOGC_DEBUG(
            utils::Core::CoreFactory,
            "Creating pipeline: type='{}', id='{}'",
            pipeline_type,
            pipeline_id);

    const auto it = pipeline_creators_.find(pipeline_type);
    if (it == pipeline_creators_.end()) {
        const std::string error_msg = std::format("Unsupported pipeline type: '{}'", pipeline_type);
        RT_LOGC_ERROR(utils::Core::CoreFactory, "{}", error_msg);
        throw std::invalid_argument(error_msg);
    }

    try {
        auto pipeline = it->second(module_factory_, pipeline_id, spec);
        RT_LOGC_INFO(
                utils::Core::CoreFactory,
                "Pipeline created: type='{}', id='{}'",
                pipeline_type,
                pipeline_id);
        return pipeline;
    } catch (const std::bad_any_cast &ex) {
        const std::string error_msg = std::format(
                "Invalid parameters for pipeline type '{}': {}", pipeline_type, ex.what());
        RT_LOGC_ERROR(utils::Core::CoreFactory, "{}", error_msg);
        throw;
    } catch (const std::exception &ex) {
        const std::string error_msg =
                std::format("Failed to create pipeline type '{}': {}", pipeline_type, ex.what());
        RT_LOGC_ERROR(utils::Core::CoreFactory, "{}", error_msg);
        throw;
    }
}

bool PipelineFactory::is_pipeline_type_supported(std::string_view pipeline_type) const {
    return pipeline_creators_.contains(pipeline_type);
}

std::vector<std::string> PipelineFactory::get_supported_pipeline_types() const {
    std::vector<std::string> types;
    types.reserve(pipeline_creators_.size());

    for (const auto &[type, creator] : pipeline_creators_) {
        std::ignore = creator; // Suppress unused variable warning
        types.push_back(type);
    }

    return types;
}

} // namespace framework::pipeline
