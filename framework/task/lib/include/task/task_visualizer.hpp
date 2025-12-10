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

/**
 * @file task_visualizer.hpp
 * @brief Task visualization system for generating ASCII art representations of
 * task dependencies
 *
 * Provides functionality to visualize task graphs as ASCII art. Tasks are
 * represented as nodes with category information and dependencies are shown
 * in a hierarchical tree structure.
 */

#ifndef FRAMEWORK_TASK_TASK_VISUALIZER_HPP
#define FRAMEWORK_TASK_TASK_VISUALIZER_HPP

#include <cstddef>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <wise_enum.h>

#include "task/task_category.hpp"

namespace framework::task {

/**
 * Node information structure
 * Contains metadata about a task node in the graph
 */
struct NodeInfo final {
    std::string name;                                    //!< Task name
    TaskCategory category{BuiltinTaskCategory::Default}; //!< Task category for coloring
    bool is_completed{};                                 //!< Whether task has completed
    bool has_failed{};                                   //!< Whether task failed
    std::string tooltip;                                 //!< Optional tooltip text
};

/**
 * Edge information structure
 * Represents a dependency relationship between tasks
 */
struct Edge final {
    std::string from;  //!< Source task name
    std::string to;    //!< Destination task name
    std::string label; //!< Optional edge label
};

/**
 * Task visualization class for generating ASCII art representations of task
 * graphs
 *
 * Creates ASCII art visualization to display task dependencies and categories
 * in a hierarchical tree structure.
 */
class TaskVisualizer final {
private:
    std::unordered_map<std::string, NodeInfo> nodes_; //!< Task nodes by name
    std::vector<Edge> edges_;                         //!< Dependency edges

    std::string graph_title_{"Task Graph"}; //!< Graph title

public:
    /**
     * Add a task node to the graph
     * @param[in] name Task name
     * @param[in] category Task category for styling
     * @param[in] tooltip Optional tooltip text
     */
    void
    add_node(const std::string &name, const TaskCategory category, const std::string &tooltip = "");

    /**
     * Add a dependency edge between tasks
     * @param[in] from Source task name (dependency)
     * @param[in] to Destination task name (dependent)
     * @param[in] label Optional edge label
     */
    void add_edge(const std::string &from, const std::string &to, const std::string &label = "");

    /**
     * Set graph title
     * @param[in] title Graph title text
     */
    void set_title(const std::string &title) { graph_title_ = title; }

    /**
     * Clear all nodes and edges
     * Resets the graph to empty state
     */
    void clear() noexcept;

    /**
     * Get number of nodes in the graph
     * @return Number of task nodes
     */
    [[nodiscard]] std::size_t get_node_count() const noexcept { return nodes_.size(); }

    /**
     * Get number of edges in the graph
     * @return Number of dependency edges
     */
    [[nodiscard]] std::size_t get_edge_count() const noexcept { return edges_.size(); }

    /**
     * Generate string visualization of the task graph
     * Creates an ASCII representation of the task graph structure
     * @return String representation of the graph
     */
    [[nodiscard]] std::string to_string() const;

private:
    /**
     * Calculate dependency generation for each node
     * @return Map of node name to generation level (0 = root)
     */
    [[nodiscard]] std::unordered_map<std::string, int> calculate_node_generations() const;

    /**
     * Render hybrid generation + tree ASCII art style
     * @param[in,out] result Output stream for the ASCII art
     * @param[in] nodes_by_generation Nodes grouped by generation level
     */
    void render_hybrid_tree(
            std::ostringstream &result,
            const std::map<int, std::vector<std::string>> &nodes_by_generation) const;

    /**
     * Recursively render a node and its children in ASCII art
     * @param[in,out] result Output stream for the ASCII art
     * @param[in] node_name Current node being rendered
     * @param[in] prefix Current indentation prefix
     * @param[in] is_last Whether this is the last child at this level
     * @param[in,out] visited Set of already visited nodes to avoid cycles
     */
    void render_node_tree(
            std::ostringstream &result,
            const std::string &node_name,
            const std::string &prefix,
            bool is_last,
            std::unordered_set<std::string> &visited) const;
};

} // namespace framework::task

#endif // FRAMEWORK_TASK_TASK_VISUALIZER_HPP
