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

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "task/task_category.hpp"
#include "task/task_visualizer.hpp"

namespace framework::task {

namespace {

/**
 * Get display name for a task category
 * @param[in] category Task category
 * @return Category display name
 */
[[nodiscard]] std::string get_category_name(const TaskCategory category) noexcept {
    return std::string{category.name()};
}

/**
 * Truncate task name if too long
 * @param[in] name Original task name
 * @param[in] max_length Maximum allowed length
 * @return Truncated name with ellipsis if needed
 */
[[nodiscard]] std::string
truncate_name(const std::string_view name, const std::size_t max_length = 15) {
    if (name.length() <= max_length) {
        return std::string(name);
    }

    if (max_length <= 3) {
        return std::string(name.substr(0, max_length));
    }

    return std::string(name.substr(0, max_length - 3)) + "...";
}

/**
 * Get parent nodes of a given node
 * @param[in] node_name Name of the node
 * @param[in] edges Vector of dependency edges
 * @return Vector of parent node names
 */
[[nodiscard]] std::vector<std::string>
get_parent_nodes(const std::string_view node_name, const std::vector<Edge> &edges) {
    std::vector<std::string> parents;
    for (const auto &edge : edges) {
        if (edge.to == node_name) {
            parents.push_back(edge.from);
        }
    }
    std::sort(parents.begin(), parents.end());
    return parents;
}

/**
 * Render a single node with status indicators
 * @param[in,out] result Output stream for the ASCII art
 * @param[in] node_name Name of the node to render
 * @param[in] connector Connector string to use for this node
 * @param[in] nodes Map of node information
 */
void render_single_node(
        std::ostringstream &result,
        const std::string_view node_name,
        const std::string_view connector,
        const std::unordered_map<std::string, NodeInfo> &nodes) {
    const std::string truncated_name = truncate_name(node_name);
    const auto node_it = nodes.find(std::string(node_name));

    if (node_it != nodes.end()) {
        result << "    " << connector << truncated_name << " ["
               << get_category_name(node_it->second.category) << "]";

        // Add status indicators if present
        if (node_it->second.has_failed) {
            result << " X";
        } else if (node_it->second.is_completed) {
            result << " *";
        }
        result << "\n";
    }
}

/**
 * Render dependency information for a node
 * @param[in,out] result Output stream for the ASCII art
 * @param[in] node_name Name of the node
 * @param[in] dep_prefix Prefix to use for dependency lines
 * @param[in] edges Vector of dependency edges
 */
void render_node_dependencies(
        std::ostringstream &result,
        const std::string_view node_name,
        const std::string_view dep_prefix,
        const std::vector<Edge> &edges) {
    const std::vector<std::string> parents = get_parent_nodes(node_name, edges);

    if (parents.empty()) {
        result << dep_prefix << "\\- depends on: none\n";
    } else {
        result << dep_prefix << "\\- depends on: ";
        for (std::size_t j = 0; j < parents.size(); ++j) {
            if (j > 0) {
                result << ", ";
            }
            result << truncate_name(parents[j]);
        }
        result << "\n";
    }
}

} // anonymous namespace

void TaskVisualizer::add_node(
        const std::string &name, const TaskCategory category, const std::string &tooltip) {
    NodeInfo node_info{};
    node_info.name = name;
    node_info.category = category;
    node_info.tooltip = tooltip;

    nodes_[name] = node_info;
}

void TaskVisualizer::add_edge(
        const std::string &from, const std::string &to, const std::string &label) {
    // Check if edge already exists to avoid duplicates
    const auto edge_exists =
            std::any_of(edges_.begin(), edges_.end(), [&from, &to](const Edge &edge) {
                return edge.from == from && edge.to == to;
            });

    if (!edge_exists) {
        Edge edge{};
        edge.from = from;
        edge.to = to;
        edge.label = label;
        edges_.push_back(edge);
    }
}

void TaskVisualizer::clear() noexcept {
    nodes_.clear();
    edges_.clear();
}

std::string TaskVisualizer::to_string() const {
    if (nodes_.empty()) {
        return "Empty task graph\n";
    }

    std::ostringstream result;

    // Add title in formatted style
    if (!graph_title_.empty()) {
        result << "=== " << graph_title_ << " ===\n\n";
    }

    // Calculate generations for each node
    auto generations = calculate_node_generations();

    // Group nodes by generation
    std::map<int, std::vector<std::string>> nodes_by_generation;
    for (const auto &[name, generation] : generations) {
        nodes_by_generation[generation].push_back(name);
    }

    // Sort nodes within each generation for consistency
    for (auto &[gen, node_list] : nodes_by_generation) {
        std::sort(node_list.begin(), node_list.end());
    }

    // Render hybrid generation + tree style
    render_hybrid_tree(result, nodes_by_generation);

    return result.str();
}

std::unordered_map<std::string, int> TaskVisualizer::calculate_node_generations() const {
    std::unordered_map<std::string, int> generations;

    // Initialize all nodes with generation -1 (unvisited)
    for (const auto &[name, node_info] : nodes_) {
        generations[name] = -1;
    }

    // Find root nodes and set them to generation 0
    std::vector<std::string> current_level;
    for (const auto &[name, node_info] : nodes_) {
        const bool has_incoming =
                std::any_of(edges_.begin(), edges_.end(), [&name](const Edge &edge) {
                    return edge.to == name;
                });
        if (!has_incoming) {
            generations[name] = 0;
            current_level.push_back(name);
        }
    }

    // Propagate generations level by level
    int current_generation = 0;
    while (!current_level.empty()) {
        std::vector<std::string> next_level;

        for (const auto &current_node : current_level) {
            // Find children of current node
            for (const auto &edge : edges_) {
                if (edge.from == current_node) {
                    const std::string &child = edge.to;
                    // Only assign generation if not already assigned or if this would be
                    // a lower generation
                    if (generations[child] == -1 || generations[child] > current_generation + 1) {
                        generations[child] = current_generation + 1;
                        next_level.push_back(child);
                    }
                }
            }
        }

        current_level = std::move(next_level);
        current_generation++;
    }

    return generations;
}

void TaskVisualizer::render_hybrid_tree(
        std::ostringstream &result,
        const std::map<int, std::vector<std::string>> &nodes_by_generation) const {
    if (nodes_by_generation.empty()) {
        return;
    }

    for (const auto &[generation, node_list] : nodes_by_generation) {
        if (node_list.empty()) {
            continue;
        }

        // Render level header
        result << "Level " << generation << ":\n";

        // Render all nodes at this level
        for (std::size_t i = 0; i < node_list.size(); ++i) {
            const std::string &node_name = node_list[i];
            const bool is_last_node = (i == node_list.size() - 1);

            // Choose connector based on position (using ASCII characters)
            const std::string connector = is_last_node ? "\\-- " : "|-- ";

            // Render node with status indicators
            render_single_node(result, node_name, connector, nodes_);

            // Add dependency information with proper indentation
            const std::string dep_prefix = is_last_node ? "        " : "    |   ";
            render_node_dependencies(result, node_name, dep_prefix, edges_);
        }

        // Add spacing between levels (except for the last level)
        if (generation < static_cast<int>(nodes_by_generation.rbegin()->first)) {
            result << "\n";
        }
    }
}

void TaskVisualizer::render_node_tree(
        std::ostringstream &result,
        const std::string &node_name,
        const std::string &prefix,
        bool is_last,
        std::unordered_set<std::string> &visited) const {
    // Check if already visited to avoid infinite recursion
    if (visited.find(node_name) != visited.end()) {
        return;
    }
    visited.insert(node_name);

    // Get node info
    auto node_it = nodes_.find(node_name);
    if (node_it == nodes_.end()) {
        return;
    }
    const NodeInfo &node_info = node_it->second;

    // Render current node
    const std::string connector = is_last ? "└── " : "├── ";
    result << prefix << connector << node_name;

    // Add category info
    result << " [" << get_category_name(node_info.category) << "]";

    // Add status indicators
    if (node_info.has_failed) {
        result << " ✗";
    } else if (node_info.is_completed) {
        result << " ✓";
    }

    result << "\n";

    // Find children (nodes that depend on this one)
    std::vector<std::string> children;
    for (const auto &edge : edges_) {
        if (edge.from == node_name) {
            children.push_back(edge.to);
        }
    }

    // Sort children for consistent output
    std::sort(children.begin(), children.end());

    // Render children recursively
    const std::string child_prefix = prefix + (is_last ? "    " : "│   ");
    for (std::size_t i = 0; i < children.size(); ++i) {
        const bool is_last_child = (i == children.size() - 1);
        render_node_tree(result, children[i], child_prefix, is_last_child, visited);
    }
}

} // namespace framework::task
