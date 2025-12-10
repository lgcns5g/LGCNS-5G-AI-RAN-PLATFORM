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

#ifndef FRAMEWORK_CORE_IGRAPH_NODE_PROVIDER_HPP
#define FRAMEWORK_CORE_IGRAPH_NODE_PROVIDER_HPP

#include <span>
#include <vector>

#include <gsl-lite/gsl-lite.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

namespace framework::pipeline {

class IGraph;
struct DynamicParams; // Forward declaration

/**
 * @class IGraphNodeProvider
 * @brief Interface for providing a graph of nodes
 *
 * This interface provides methods for providing a graph of nodes, which are
 * connected by data dependencies. It also supports dynamic updates to graph
 * node parameters for scenarios requiring cuGraphExecKernelNodeSetParams.
 */
class IGraphNodeProvider {
public:
    /**
     * Default constructor.
     */
    IGraphNodeProvider() = default;

    /**
     * Virtual destructor.
     */
    virtual ~IGraphNodeProvider() = default;

    /**
     * Move constructor.
     */
    IGraphNodeProvider(IGraphNodeProvider &&) = default;

    /**
     * Move assignment operator.
     * @return Reference to this object
     */
    IGraphNodeProvider &operator=(IGraphNodeProvider &&) = default;

    IGraphNodeProvider(const IGraphNodeProvider &) = delete;
    IGraphNodeProvider &operator=(const IGraphNodeProvider &) = delete;

    /**
     * @brief Add node(s) to the graph
     * @param[in] graph The graph to add the node(s) to
     * @param[in] deps The dependencies of the node(s)
     * @return Span of created graph node handles (can contain single or multiple nodes)
     * @throws std::runtime_error if CUDA graph node creation fails
     */
    [[nodiscard]] virtual std::span<const CUgraphNode> add_node_to_graph(
            gsl_lite::not_null<IGraph *> graph, const std::span<const CUgraphNode> deps) = 0;

    /**
     * @brief Update graph node parameters for dynamic iteration changes
     *
     * This method enables dynamic updates to kernel launch parameters using
     * cuGraphExecKernelNodeSetParams. Modules can extract their specific
     * parameters from params.module_specific_params and update their graph
     * nodes accordingly (e.g., changing grid dimensions, shared memory size).
     *
     * @param exec The executable graph to update
     * @param params Dynamic parameters containing module-specific parameters
     * @throws std::runtime_error if cuGraphExecKernelNodeSetParams fails
     */
    virtual void update_graph_node_params(CUgraphExec exec, const DynamicParams &params) = 0;
};

} // namespace framework::pipeline

#endif // FRAMEWORK_CORE_IGRAPH_NODE_PROVIDER_HPP
