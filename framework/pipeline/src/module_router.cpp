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
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <quill/LogMacros.h>

#include "log/rt_log_macros.hpp"
#include "pipeline/module_router.hpp"
#include "pipeline/types.hpp"
#include "utils/core_log.hpp"
#include "utils/string_hash.hpp"

namespace framework::pipeline {

void ModuleRouter::add_connection(const PortConnection &connection) {
    // Check for duplicate connection
    for (const auto &[source_module, source_port, target_module, target_port] : connections_) {
        if (source_module == connection.source_module && source_port == connection.source_port &&
            target_module == connection.target_module && target_port == connection.target_port) {
            const std::string error_msg = std::format(
                    "Duplicate connection: {}.{} -> {}.{} already exists",
                    connection.source_module,
                    connection.source_port,
                    connection.target_module,
                    connection.target_port);
            RT_LOGC_ERROR(utils::Core::CoreModule, "{}", error_msg);
            throw std::runtime_error(error_msg);
        }
    }

    connections_.push_back(connection);

    // Update indices for fast lookup
    const auto idx = connections_.size() - 1;
    module_to_connection_indices_[connection.source_module].push_back(idx);
    module_to_connection_indices_[connection.target_module].push_back(idx);

    RT_LOGC_DEBUG(
            utils::Core::CoreModule,
            "Added connection: {}.{} -> {}.{}",
            connection.source_module,
            connection.source_port,
            connection.target_module,
            connection.target_port);
}

std::vector<PortConnection>
ModuleRouter::get_module_connections(const std::string_view module_id) const {
    std::vector<PortConnection> result;

    const auto it = module_to_connection_indices_.find(module_id);
    if (it != module_to_connection_indices_.end()) {
        result.reserve(it->second.size());
        // Simple index-based access is clearer than std::transform
        for (const auto idx : it->second) {
            // cppcheck-suppress useStlAlgorithm
            result.push_back(connections_[idx]);
        }
    }

    return result;
}

std::vector<PortConnection>
ModuleRouter::get_input_connections(const std::string_view module_id) const {
    std::vector<PortConnection> result;

    const auto it = module_to_connection_indices_.find(module_id);
    if (it != module_to_connection_indices_.end()) {
        for (const auto idx : it->second) {
            if (connections_[idx].target_module == module_id) {
                result.push_back(connections_[idx]);
            }
        }
    }

    return result;
}

std::vector<PortConnection>
ModuleRouter::get_output_connections(const std::string_view module_id) const {
    std::vector<PortConnection> result;

    const auto it = module_to_connection_indices_.find(module_id);
    if (it != module_to_connection_indices_.end()) {
        for (const auto idx : it->second) {
            if (connections_[idx].source_module == module_id) {
                result.push_back(connections_[idx]);
            }
        }
    }

    return result;
}

bool ModuleRouter::has_connections(const std::string_view module_id) const {
    return module_to_connection_indices_.contains(module_id);
}

std::unordered_set<std::string> ModuleRouter::get_all_module_ids() const {
    std::unordered_set<std::string> module_ids;

    for (const auto &conn : connections_) {
        module_ids.insert(conn.source_module);
        module_ids.insert(conn.target_module);
    }

    return module_ids;
}

void ModuleRouter::validate() const {
    RT_LOGC_DEBUG(
            utils::Core::CoreModule, "Validating router with {} connections", connections_.size());

    // Check for duplicate input connections (multiple sources to same input port)
    // Use unordered_set for O(n) validation instead of O(n²) nested loops
    std::unordered_set<std::string> seen_targets;
    seen_targets.reserve(connections_.size());

    for (const auto &[source_module, source_port, target_module, target_port] : connections_) {
        const std::string target_key = std::format("{}.{}", target_module, target_port);

        if (!seen_targets.insert(target_key).second) {
            // Find the first connection with this target for error message
            for (const auto
                         &[first_source_module,
                           first_source_port,
                           first_target_module,
                           first_target_port] : connections_) {
                if (first_target_module == target_module && first_target_port == target_port &&
                    (first_source_module != source_module || first_source_port != source_port)) {
                    const std::string error_msg = std::format(
                            "Duplicate input connection: {}.{} is connected to "
                            "both {}.{} and "
                            "{}.{}",
                            target_module,
                            target_port,
                            first_source_module,
                            first_source_port,
                            source_module,
                            source_port);
                    RT_LOGC_ERROR(utils::Core::CoreModule, "{}", error_msg);
                    throw std::runtime_error(error_msg);
                }
            }
        }
    }

    RT_LOGC_INFO(utils::Core::CoreModule, "Router validation passed");

    // Additional validation can be added here:
    // - Check for cycles
    // - Validate port names against module definitions
    // - Check for disconnected modules
}

} // namespace framework::pipeline
