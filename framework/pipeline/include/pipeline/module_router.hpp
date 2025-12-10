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

#ifndef FRAMEWORK_CORE_MODULE_ROUTER_HPP
#define FRAMEWORK_CORE_MODULE_ROUTER_HPP

#include <cstddef>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "pipeline/types.hpp"
#include "utils/string_hash.hpp"

namespace framework::pipeline {

/**
 * @class ModuleRouter
 * @brief Manages routing configuration between modules in a pipeline
 *
 * This class manages how modules are connected via their input/output ports,
 * providing efficient lookup of connections involving specific modules.
 *
 * @details Internal Architecture:
 * The router uses an index-based lookup optimization for efficient connection
 * queries.
 * - connections_: Vector storing all connections sequentially
 * - module_to_connection_indices_: Maps each module ID to indices in
 * connections_
 *
 * When a connection A->B is added:
 * 1. Connection is appended to connections_ at index N
 * 2. Index N is added to both moduleA's and moduleB's index lists
 *
 * This allows O(1) lookup of all connections involving a module, rather than
 * O(n) scanning of all connections.
 *
 * Example after adding A->B, B->C, A->C:
 * @code
 * connections_ = [
 *     0: {A, output0, B, input0},
 *     1: {B, output0, C, input0},
 *     2: {A, output1, C, input1}
 * ]
 * module_to_connection_indices_ = {
 *     "A": [0, 2],  // A is involved in connections 0 and 2
 *     "B": [0, 1],  // B is involved in connections 0 and 1
 *     "C": [1, 2]   // C is involved in connections 1 and 2
 * }
 * @endcode
 */
class ModuleRouter final {
public:
    /**
     * Default constructor.
     */
    ModuleRouter() = default;

    /**
     * Destructor.
     */
    ~ModuleRouter() = default;

    // Non-copyable, movable
    ModuleRouter(const ModuleRouter &) = delete;
    ModuleRouter &operator=(const ModuleRouter &) = delete;

    /**
     * Move constructor.
     */
    ModuleRouter(ModuleRouter &&) = default;

    /**
     * Move assignment operator.
     *
     * @return Reference to this object
     */
    ModuleRouter &operator=(ModuleRouter &&) = default;

    /**
     * Add a connection between two module ports.
     *
     * @param[in] connection The port connection to add
     * @throws std::runtime_error if duplicate connection already exists
     */
    void add_connection(const PortConnection &connection);

    /**
     * Get all connections for a specific module.
     *
     * Returns connections where the module is either source or target.
     *
     * @param[in] module_id The module ID to query
     * @return Vector of connections involving this module
     */
    [[nodiscard]] std::vector<PortConnection>
    get_module_connections(std::string_view module_id) const;

    /**
     * Get input connections for a specific module.
     *
     * Returns connections where the module is the target (receiving input).
     *
     * @param[in] module_id The module ID to query
     * @return Vector of connections where this module is the target
     */
    [[nodiscard]] std::vector<PortConnection>
    get_input_connections(std::string_view module_id) const;

    /**
     * Get output connections for a specific module.
     *
     * Returns connections where the module is the source (providing output).
     *
     * @param[in] module_id The module ID to query
     * @return Vector of connections where this module is the source
     */
    [[nodiscard]] std::vector<PortConnection>
    get_output_connections(std::string_view module_id) const;

    /**
     * Check if a module has any connections.
     *
     * @param[in] module_id The module ID to check
     * @return true if the module has connections, false otherwise
     */
    [[nodiscard]] bool has_connections(std::string_view module_id) const;

    /**
     * Get all unique module IDs in the routing configuration.
     *
     * @return Set of module IDs
     */
    [[nodiscard]] std::unordered_set<std::string> get_all_module_ids() const;

    /**
     * Validate the routing configuration.
     *
     * Checks for duplicate connections and invalid port configurations.
     *
     * @throws std::runtime_error if configuration is invalid
     */
    void validate() const;

private:
    std::vector<PortConnection> connections_; //!< All port connections
    std::unordered_map<
            std::string,
            std::vector<std::size_t>,
            utils::TransparentStringHash,
            std::equal_to<>>
            module_to_connection_indices_; //!< Module ID to connection indices
};

} // namespace framework::pipeline

#endif // FRAMEWORK_CORE_MODULE_ROUTER_HPP
