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
 * @file task_visualizer_tests.cpp
 * @brief Unit tests for TaskVisualizer class
 */

#include <string>
#include <utility>

#include <gtest/gtest.h>

#include "task/task_category.hpp"
#include "task/task_visualizer.hpp"

namespace {
namespace ft = framework::task;

/**
 * Basic functionality tests for TaskVisualizer
 */
TEST(TaskVisualizer, DefaultConstructor) {
    const ft::TaskVisualizer visualizer{};

    EXPECT_EQ(visualizer.get_node_count(), 0U);
    EXPECT_EQ(visualizer.get_edge_count(), 0U);
}

TEST(TaskVisualizer, AddNodes) {
    ft::TaskVisualizer visualizer{};

    // Add nodes with different categories
    visualizer.add_node("task1", ft::TaskCategory{ft::BuiltinTaskCategory::Default});
    EXPECT_EQ(visualizer.get_node_count(), 1U);

    visualizer.add_node("task2", ft::TaskCategory{ft::BuiltinTaskCategory::HighPriority});
    EXPECT_EQ(visualizer.get_node_count(), 2U);

    visualizer.add_node(
            "task3", ft::TaskCategory{ft::BuiltinTaskCategory::Compute}, "Computation task");
    EXPECT_EQ(visualizer.get_node_count(), 3U);

    // Adding same task again should not increase count
    visualizer.add_node("task1", ft::TaskCategory{ft::BuiltinTaskCategory::IO});
    EXPECT_EQ(visualizer.get_node_count(), 3U);
}

TEST(TaskVisualizer, AddEdges) {
    ft::TaskVisualizer visualizer{};

    // Add some nodes first
    visualizer.add_node("task1", ft::TaskCategory{ft::BuiltinTaskCategory::Default});
    visualizer.add_node("task2", ft::TaskCategory{ft::BuiltinTaskCategory::Compute});
    visualizer.add_node("task3", ft::TaskCategory{ft::BuiltinTaskCategory::IO});

    // Add edges
    visualizer.add_edge("task1", "task2");
    EXPECT_EQ(visualizer.get_edge_count(), 1U);

    visualizer.add_edge("task2", "task3", "depends on");
    EXPECT_EQ(visualizer.get_edge_count(), 2U);

    visualizer.add_edge("task1", "task3");
    EXPECT_EQ(visualizer.get_edge_count(), 3U);

    // Adding duplicate edge should not increase count
    visualizer.add_edge("task1", "task2");
    EXPECT_EQ(visualizer.get_edge_count(), 3U);
}

TEST(TaskVisualizer, Clear) {
    ft::TaskVisualizer visualizer{};

    // Add some nodes and edges
    visualizer.add_node("task1", ft::TaskCategory{ft::BuiltinTaskCategory::Default});
    visualizer.add_node("task2", ft::TaskCategory{ft::BuiltinTaskCategory::Compute});
    visualizer.add_edge("task1", "task2");

    EXPECT_EQ(visualizer.get_node_count(), 2U);
    EXPECT_EQ(visualizer.get_edge_count(), 1U);

    // Clear everything
    visualizer.clear();

    EXPECT_EQ(visualizer.get_node_count(), 0U);
    EXPECT_EQ(visualizer.get_edge_count(), 0U);
}

TEST(TaskVisualizer, SetTitle) {
    ft::TaskVisualizer visualizer{};

    // Test setting title - should not throw
    EXPECT_NO_THROW(visualizer.set_title("My Task Graph"));
    EXPECT_NO_THROW(visualizer.set_title(""));
}

TEST(TaskVisualizer, MoveSemantics) {
    ft::TaskVisualizer visualizer1{};
    visualizer1.add_node("task1", ft::TaskCategory{ft::BuiltinTaskCategory::Default});

    // Move constructor
    ft::TaskVisualizer visualizer2{std::move(visualizer1)};
    EXPECT_EQ(visualizer2.get_node_count(), 1U);

    // Move assignment
    ft::TaskVisualizer visualizer3{};
    visualizer3 = std::move(visualizer2);
    EXPECT_EQ(visualizer3.get_node_count(), 1U);
}

/**
 * String visualization tests
 */
TEST(TaskVisualizer, GenerateVisualizationEmpty) {
    const ft::TaskVisualizer visualizer{};

    const std::string visualization = visualizer.to_string();
    EXPECT_EQ(visualization, "Empty task graph\n");
}

TEST(TaskVisualizer, GenerateVisualizationSingleNode) {
    ft::TaskVisualizer visualizer{};

    visualizer.add_node("single_task", ft::TaskCategory{ft::BuiltinTaskCategory::Default});

    const std::string visualization = visualizer.to_string();
    EXPECT_FALSE(visualization.empty());
    EXPECT_TRUE(visualization.find("single_task") != std::string::npos);
    EXPECT_TRUE(visualization.find("[Default]") != std::string::npos);
}

TEST(TaskVisualizer, GenerateVisualizationWithTitle) {
    ft::TaskVisualizer visualizer{};

    visualizer.set_title("Test Graph");
    visualizer.add_node("task1", ft::TaskCategory{ft::BuiltinTaskCategory::Default});

    const std::string visualization = visualizer.to_string();

    EXPECT_TRUE(visualization.find("=== Test Graph ===") != std::string::npos);
}

TEST(TaskVisualizer, GenerateVisualizationLinearChain) {
    ft::TaskVisualizer visualizer{};

    visualizer.add_node("start", ft::TaskCategory{ft::BuiltinTaskCategory::Default});
    visualizer.add_node("middle", ft::TaskCategory{ft::BuiltinTaskCategory::Compute});
    visualizer.add_node("end", ft::TaskCategory{ft::BuiltinTaskCategory::IO});

    visualizer.add_edge("start", "middle");
    visualizer.add_edge("middle", "end");

    const std::string visualization = visualizer.to_string();

    EXPECT_FALSE(visualization.empty());

    // Check all nodes are present
    EXPECT_TRUE(visualization.find("start") != std::string::npos);
    EXPECT_TRUE(visualization.find("middle") != std::string::npos);
    EXPECT_TRUE(visualization.find("end") != std::string::npos);

    // Check categories are shown
    EXPECT_TRUE(visualization.find("[Default]") != std::string::npos);
    EXPECT_TRUE(visualization.find("[Compute]") != std::string::npos);
    EXPECT_TRUE(visualization.find("[IO]") != std::string::npos);

    // Check tree structure (should have level headers and dependency info)
    EXPECT_TRUE(visualization.find("Level 0:") != std::string::npos);
    EXPECT_TRUE(visualization.find("Level 1:") != std::string::npos);
    EXPECT_TRUE(visualization.find("Level 2:") != std::string::npos);
    EXPECT_TRUE(visualization.find("depends on:") != std::string::npos);
}

TEST(TaskVisualizer, GenerateVisualizationBranchingGraph) {
    ft::TaskVisualizer visualizer{};

    visualizer.set_title("Branching Task Graph");

    // Create a branching structure
    visualizer.add_node("root", ft::TaskCategory{ft::BuiltinTaskCategory::Default});
    visualizer.add_node("branch1", ft::TaskCategory{ft::BuiltinTaskCategory::Compute});
    visualizer.add_node("branch2", ft::TaskCategory{ft::BuiltinTaskCategory::IO});
    visualizer.add_node("leaf1", ft::TaskCategory{ft::BuiltinTaskCategory::HighPriority});
    visualizer.add_node("leaf2", ft::TaskCategory{ft::BuiltinTaskCategory::Network});

    visualizer.add_edge("root", "branch1");
    visualizer.add_edge("root", "branch2");
    visualizer.add_edge("branch1", "leaf1");
    visualizer.add_edge("branch2", "leaf2");

    const std::string visualization = visualizer.to_string();

    EXPECT_FALSE(visualization.empty());

    // Check title
    EXPECT_TRUE(visualization.find("=== Branching Task Graph ===") != std::string::npos);

    // Check all nodes are present
    EXPECT_TRUE(visualization.find("root") != std::string::npos);
    EXPECT_TRUE(visualization.find("branch1") != std::string::npos);
    EXPECT_TRUE(visualization.find("branch2") != std::string::npos);
    EXPECT_TRUE(visualization.find("leaf1") != std::string::npos);
    EXPECT_TRUE(visualization.find("leaf2") != std::string::npos);

    // Check tree structure (should have level headers and connecting lines)
    EXPECT_TRUE(visualization.find("Level 0:") != std::string::npos);
    EXPECT_TRUE(visualization.find("Level 1:") != std::string::npos);
    EXPECT_TRUE(visualization.find("Level 2:") != std::string::npos);
    EXPECT_TRUE(visualization.find("depends on:") != std::string::npos);
    EXPECT_TRUE(visualization.find('|') != std::string::npos);
}

TEST(TaskVisualizer, GenerateVisualizationNameTruncation) {
    ft::TaskVisualizer visualizer{};

    // Add node with very long name
    visualizer.add_node(
            "this_is_a_very_long_task_name_that_should_be_truncated",
            ft::TaskCategory{ft::BuiltinTaskCategory::Default});
    visualizer.add_node("short", ft::TaskCategory{ft::BuiltinTaskCategory::Compute});

    visualizer.add_edge("this_is_a_very_long_task_name_that_should_be_truncated", "short");

    const std::string visualization = visualizer.to_string();

    EXPECT_FALSE(visualization.empty());

    // Long name should be truncated with ellipsis
    EXPECT_TRUE(visualization.find("...") != std::string::npos);

    // Short name should appear unchanged
    EXPECT_TRUE(visualization.find("short [Compute]") != std::string::npos);

    // Should not contain the full long name
    EXPECT_FALSE(
            visualization.find("this_is_a_very_long_task_name_that_should_be_truncated") !=
            std::string::npos);
}

TEST(TaskVisualizer, GenerateVisualizationComplexExample) {
    ft::TaskVisualizer visualizer{};

    visualizer.set_title("Complex Pipeline Example");

    // Create a complex pipeline structure
    visualizer.add_node("init", ft::TaskCategory{ft::BuiltinTaskCategory::Default});
    visualizer.add_node("load_config", ft::TaskCategory{ft::BuiltinTaskCategory::IO});
    visualizer.add_node("load_data", ft::TaskCategory{ft::BuiltinTaskCategory::IO});
    visualizer.add_node("preprocess", ft::TaskCategory{ft::BuiltinTaskCategory::Compute});
    visualizer.add_node("parallel_task_a", ft::TaskCategory{ft::BuiltinTaskCategory::Compute});
    visualizer.add_node("parallel_task_b", ft::TaskCategory{ft::BuiltinTaskCategory::Compute});
    visualizer.add_node("merge_results", ft::TaskCategory{ft::BuiltinTaskCategory::Compute});
    visualizer.add_node("validate", ft::TaskCategory{ft::BuiltinTaskCategory::Default});
    visualizer.add_node("save_output", ft::TaskCategory{ft::BuiltinTaskCategory::IO});

    // Build dependency chain
    visualizer.add_edge("init", "load_config");
    visualizer.add_edge("init", "load_data");
    visualizer.add_edge("load_config", "preprocess");
    visualizer.add_edge("load_data", "preprocess");
    visualizer.add_edge("preprocess", "parallel_task_a");
    visualizer.add_edge("preprocess", "parallel_task_b");
    visualizer.add_edge("parallel_task_a", "merge_results");
    visualizer.add_edge("parallel_task_b", "merge_results");
    visualizer.add_edge("merge_results", "validate");
    visualizer.add_edge("validate", "save_output");

    const std::string visualization = visualizer.to_string();

    EXPECT_FALSE(visualization.empty());
    EXPECT_TRUE(visualization.find("=== Complex Pipeline Example ===") != std::string::npos);

    // Check that nodes are present
    EXPECT_TRUE(visualization.find("init") != std::string::npos);
    EXPECT_TRUE(visualization.find("merge_results") != std::string::npos);
    EXPECT_TRUE(visualization.find("save_output") != std::string::npos);
}

TEST(TaskVisualizer, GenerateVisualizationDiamondPattern) {
    ft::TaskVisualizer visualizer{};

    visualizer.set_title("Diamond Dependency Pattern");

    /* Create diamond pattern:
           A
         /   \
        B     C
       / \   / \
      D   E F   G
       \ | | /
         H
    */
    visualizer.add_node("A", ft::TaskCategory{ft::BuiltinTaskCategory::Default});
    visualizer.add_node("B", ft::TaskCategory{ft::BuiltinTaskCategory::Compute});
    visualizer.add_node("C", ft::TaskCategory{ft::BuiltinTaskCategory::IO});
    visualizer.add_node("D", ft::TaskCategory{ft::BuiltinTaskCategory::Network});
    visualizer.add_node("E", ft::TaskCategory{ft::BuiltinTaskCategory::HighPriority});
    visualizer.add_node("F", ft::TaskCategory{ft::BuiltinTaskCategory::LowPriority});
    visualizer.add_node("G", ft::TaskCategory{ft::BuiltinTaskCategory::Message});
    visualizer.add_node("H", ft::TaskCategory{ft::BuiltinTaskCategory::Default});

    // Build diamond dependencies
    visualizer.add_edge("A", "B");
    visualizer.add_edge("A", "C");
    visualizer.add_edge("B", "D");
    visualizer.add_edge("B", "E");
    visualizer.add_edge("C", "E");
    visualizer.add_edge("C", "F");
    visualizer.add_edge("C", "G");
    visualizer.add_edge("D", "H");
    visualizer.add_edge("E", "H");
    visualizer.add_edge("F", "H");
    visualizer.add_edge("G", "H");

    const std::string visualization = visualizer.to_string();

    EXPECT_FALSE(visualization.empty());
    EXPECT_TRUE(visualization.find("=== Diamond Dependency Pattern ===") != std::string::npos);

    // Check all nodes are present
    EXPECT_TRUE(visualization.find("A [Default]") != std::string::npos);
    EXPECT_TRUE(visualization.find("B [Compute]") != std::string::npos);
    EXPECT_TRUE(visualization.find("C [IO]") != std::string::npos);
    EXPECT_TRUE(visualization.find("H [Default]") != std::string::npos);

    // Check level structure is present
    EXPECT_TRUE(visualization.find("Level 0:") != std::string::npos);
    EXPECT_TRUE(visualization.find("Level 1:") != std::string::npos);
    EXPECT_TRUE(visualization.find("Level 2:") != std::string::npos);
    EXPECT_TRUE(visualization.find("Level 3:") != std::string::npos);

    // Check dependency information is shown
    EXPECT_TRUE(visualization.find("depends on: none") != std::string::npos); // A has no deps
    EXPECT_TRUE(visualization.find("depends on: A") != std::string::npos);    // B and C depend on A
    EXPECT_TRUE(visualization.find("depends on:") != std::string::npos); // Dependencies are shown

    // Check tree structure connectors
    EXPECT_TRUE(visualization.find("|--") != std::string::npos);
    EXPECT_TRUE(visualization.find("\\--") != std::string::npos);
    EXPECT_TRUE(visualization.find('|') != std::string::npos);
}

} // namespace
