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

#ifndef FRAMEWORK_CORE_IGRAPH_HPP
#define FRAMEWORK_CORE_IGRAPH_HPP

#include <span>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

namespace framework::pipeline {

class IGraphNodeProvider;

/**
 * @brief Abstract base class for all graph types
 * Focus on the essential operations needed for graph building
 */
class IGraph {
public:
    /**
     * Default constructor.
     */
    IGraph() = default;

    /**
     * @brief Virtual Destructor
     */
    virtual ~IGraph() = default;

    IGraph(const IGraph &) = delete;
    IGraph(IGraph &&) = delete;

    IGraph &operator=(const IGraph &) = delete;
    IGraph &operator=(IGraph &&) = delete;

    /**
     * @brief Creates a graph
     */
    virtual void create() = 0;

    /**
     * @brief Checks if the graph has been created
     * @return true if created, false otherwise
     */
    [[nodiscard]] virtual bool is_created() const = 0;

    /**
     * @brief Returns the graph handle
     * @return The graph handle
     */
    [[nodiscard]] virtual CUgraph handle() const = 0;

    /**
     * @brief Returns the root node of the graph
     * @return The root node of the graph
     */
    [[nodiscard]] virtual CUgraphNode root_node() const = 0;

    /**
     * @brief Adds a kernel node to the graph
     *
     * Creates a kernel node in the CUDA graph with the specified dependencies
     * and parameters. Returns the created node handle which should be stored by
     * the caller for later parameter updates.
     *
     * @param[in] deps Dependency nodes that must complete before this node
     * executes
     * @param[in] params Kernel launch parameters for the node
     * @return The created graph node handle
     *
     * @throws std::runtime_error if CUDA operation fails
     */
    [[nodiscard]] virtual CUgraphNode
    add_kernel_node(std::span<const CUgraphNode> deps, const CUDA_KERNEL_NODE_PARAMS &params) = 0;

    /**
     * @brief Adds a child graph node to the graph
     *
     * Integrates a pre-captured CUDA graph as a child node within this graph.
     * This is commonly used for TensorRT engines or other stream-captured
     * execution units. Returns the created node handle which should be stored by
     * the caller for later parameter updates.
     *
     * @param[in] deps Dependency nodes that must complete before this node
     * executes
     * @param[in] child_graph The CUDA graph to add as a child (must be a valid,
     * captured graph)
     * @return The created graph node handle
     *
     * @throws std::runtime_error if CUDA operation fails
     */
    [[nodiscard]] virtual CUgraphNode
    add_child_graph_node(std::span<const CUgraphNode> deps, CUgraph child_graph) = 0;

    /**
     * @brief Returns the graph execution handle
     * @return The graph execution handle
     */
    [[nodiscard]] virtual CUgraphExec exec_handle() const = 0;

    /**
     * @brief Instantiates the graph
     * @param[in] flags The flags passed to cuGraphInstantiate
     */
    virtual void instantiate(unsigned int flags) = 0;

    /**
     * @brief Uploads the graph to the device
     * @param[in] stream The stream passed to cuGraphUpload
     */
    virtual void upload(cudaStream_t stream) = 0;

    /**
     * @brief Launches the graph
     * @param[in] stream The stream passed to cuGraphLaunch
     */
    virtual void launch(cudaStream_t stream) = 0;
};
} // namespace framework::pipeline

#endif // FRAMEWORK_CORE_IGRAPH_HPP
