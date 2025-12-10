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
#include <unordered_map>
#include <utility>

#include <quill/LogMacros.h>

#include "log/rt_log_macros.hpp"
#include "pipeline/imodule.hpp"
#include "pipeline/module_factory.hpp"
#include "utils/core_log.hpp"
#include "utils/string_hash.hpp"

namespace framework::pipeline {

std::unique_ptr<IModule> ModuleFactory::create_module(
        std::string_view module_type,
        const std::string &instance_id,
        const std::any &static_params) {

    RT_LOGC_DEBUG(
            utils::Core::CoreFactory,
            "Creating module: type='{}', instance_id='{}'",
            module_type,
            instance_id);

    const auto it = module_creators_.find(module_type);
    if (it == module_creators_.end()) {
        const std::string error_msg = std::format("Unsupported module type: '{}'", module_type);
        RT_LOGC_ERROR(utils::Core::CoreFactory, "{}", error_msg);
        throw std::invalid_argument(error_msg);
    }

    try {
        auto module = it->second(instance_id, static_params);
        RT_LOGC_INFO(
                utils::Core::CoreFactory,
                "Module created: type='{}', instance_id='{}'",
                module_type,
                instance_id);
        return module;
    } catch (const std::bad_any_cast &ex) {
        const std::string error_msg =
                std::format("Invalid parameters for module type '{}': {}", module_type, ex.what());
        RT_LOGC_ERROR(utils::Core::CoreFactory, "{}", error_msg);
        throw;
    } catch (const std::exception &ex) {
        const std::string error_msg =
                std::format("Failed to create module type '{}': {}", module_type, ex.what());
        RT_LOGC_ERROR(utils::Core::CoreFactory, "{}", error_msg);
        throw;
    }
}

bool ModuleFactory::supports_module_type(std::string_view module_type) const {
    return module_creators_.contains(module_type);
}

} // namespace framework::pipeline
