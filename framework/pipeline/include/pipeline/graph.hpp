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

#ifndef FRAMEWORK_CORE_GRAPH_HPP
#define FRAMEWORK_CORE_GRAPH_HPP

#include <span>

#include <driver_types.h>

#include <cuda.h>

#include "pipeline/igraph.hpp"

namespace framework::pipeline {

/**
 * @class Graph
 * @brief Concrete implementation of IGraph for CUDA graph management
 *
 * This class provides a complete implementation of the IGraph interface,
 * managing the lifecycle of CUDA graphs including creation, node addition,
 * instantiation, upload, and launch operations.
 */
class Graph final : public IGraph {
public:
    /**
     * Default constructor.
     */
    Graph() = default;

    /**
     * Destructor - cleans up CUDA graph resources.
     */
    ~Graph() override;

    // Non-copyable, non-movable (owns CUDA resources)
    Graph(const Graph &) = delete;
    Graph &operator=(const Graph &) = delete;
    Graph(Graph &&) = delete;
    Graph &operator=(Graph &&) = delete;

    /**
     * @brief Creates a CUDA graph
     */
    void create() override;

    /**
     * @brief Checks if the graph has been created
     * @return true if created, false otherwise
     */
    [[nodiscard]] bool is_created() const override { return graph_ != nullptr; }

    /**
     * @brief Returns the graph handle
     * @return The CUDA graph handle
     */
    [[nodiscard]] CUgraph handle() const override { return graph_; }

    /**
     * @brief Returns the root node of the graph
     * @return The root node of the graph
     */
    [[nodiscard]] CUgraphNode root_node() const override { return root_node_; }

    /**
     * @brief Adds a kernel node to the graph
     * @param[in] deps Dependency nodes that must complete before this node
     * executes
     * @param[in] params Kernel launch parameters for the node
     * @return The created graph node handle
     */
    [[nodiscard]] CUgraphNode add_kernel_node(
            std::span<const CUgraphNode> deps, const CUDA_KERNEL_NODE_PARAMS &params) override;

    /**
     * @brief Adds a child graph node to the graph
     * @param[in] deps Dependency nodes that must complete before this node
     * executes
     * @param[in] child_graph The CUDA graph to add as a child
     * @return The created graph node handle
     */
    [[nodiscard]] CUgraphNode
    add_child_graph_node(std::span<const CUgraphNode> deps, CUgraph child_graph) override;

    /**
     * @brief Returns the graph execution handle
     * @return The graph execution handle
     */
    [[nodiscard]] CUgraphExec exec_handle() const override { return exec_; }

    /**
     * @brief Instantiates the graph for execution
     * @param[in] flags Instantiation flags passed to cuGraphInstantiate
     */
    void instantiate(unsigned int flags) override;

    /**
     * @brief Uploads the graph to the device
     * @param[in] stream CUDA stream for upload passed to cuGraphUpload
     */
    void upload(cudaStream_t stream) override;

    /**
     * @brief Launches the graph on the specified stream
     * @param[in] stream CUDA stream for graph execution passed to cuGraphLaunch
     */
    void launch(cudaStream_t stream) override;

private:
    CUgraph graph_{nullptr};         //!< CUDA graph handle
    CUgraphExec exec_{nullptr};      //!< CUDA graph execution handle
    CUgraphNode root_node_{nullptr}; //!< Root node of the graph
    bool is_instantiated_{false};    //!< Whether graph has been instantiated
};

} // namespace framework::pipeline

#endif // FRAMEWORK_CORE_GRAPH_HPP
