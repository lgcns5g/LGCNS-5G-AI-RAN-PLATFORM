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

#ifndef FRAMEWORK_CORE_GRAPH_MANAGER_HPP
#define FRAMEWORK_CORE_GRAPH_MANAGER_HPP

#include <memory>
#include <span>

#include <driver_types.h>

#include <cuda.h>

#include "pipeline/graph.hpp"
#include "pipeline/igraph.hpp"
#include "pipeline/igraph_manager.hpp"

namespace framework::pipeline {

/**
 * @class GraphManager
 * @brief Concrete implementation of IGraphManager
 *
 * This class manages the lifecycle of a CUDA graph, providing a high-level
 * interface for graph operations. It owns a single Graph instance for standard
 * pipeline execution.
 *
 * @note Future extensions could support multiple graphs for conditional
 * execution, DGL (Dynamic Graph Launch), or segmented execution modes.
 */
class GraphManager final : public IGraphManager {
public:
    /**
     * Default constructor.
     * Creates the internal graph instance.
     */
    GraphManager();

    /**
     * Destructor.
     */
    ~GraphManager() override = default;

    // Non-copyable, non-movable
    GraphManager(const GraphManager &) = delete;
    GraphManager &operator=(const GraphManager &) = delete;
    GraphManager(GraphManager &&) = delete;
    GraphManager &operator=(GraphManager &&) = delete;

    /**
     * @brief Instantiates the graph for execution
     */
    void instantiate_graph() const override;

    /**
     * @brief Uploads the graph to the device
     * @param[in] stream CUDA stream for upload operation
     */
    void upload_graph(cudaStream_t stream) const override;

    /**
     * @brief Launches the graph on the specified stream
     * @param[in] stream CUDA stream for graph execution
     */
    void launch_graph(cudaStream_t stream) const override;

    /**
     * @brief Get the graph execution handle
     *
     * Returns the CUgraphExec handle needed for dynamic parameter updates
     * via cuGraphExecKernelNodeSetParams.
     *
     * @return The graph execution handle
     * @throws std::runtime_error if graph is not instantiated
     */
    [[nodiscard]] CUgraphExec get_exec() const override;

    /**
     * @brief Adds kernel node(s) to the graph via a graph node provider
     * @param[in] graph_node_provider Provider that will add nodes to the graph
     * @param[in] deps Dependency nodes that must complete before these nodes
     * execute
     * @return Span of created graph node handles (can contain single or multiple nodes)
     */
    [[nodiscard]] std::span<const CUgraphNode> add_kernel_node(
            gsl_lite::not_null<IGraphNodeProvider *> graph_node_provider,
            std::span<const CUgraphNode> deps) const override;

private:
    std::unique_ptr<IGraph> main_graph_; //!< Main graph for standard execution

    // Future extensions:
    // std::vector<std::unique_ptr<IGraph>> conditional_graphs_;  // For
    // conditional mode std::vector<std::unique_ptr<IGraph>> dgl_graphs_;  // For
    // DGL mode std::array<std::unique_ptr<IGraph>, NUM_SEGMENTS>
    // segment_graphs_;  // For device graph mode
};

} // namespace framework::pipeline

#endif // FRAMEWORK_CORE_GRAPH_MANAGER_HPP
