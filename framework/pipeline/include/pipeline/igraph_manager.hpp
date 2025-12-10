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

#ifndef FRAMEWORK_CORE_IGRAPH_MANAGER_HPP
#define FRAMEWORK_CORE_IGRAPH_MANAGER_HPP

#include <span>
#include <vector>

#include <gsl-lite/gsl-lite.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

namespace framework::pipeline {

class IGraphNodeProvider;

/**
 * @class IGraphManager
 * @brief Interface for managing CUDA graph lifecycle.
 *
 * This interface abstracts the creation, instantiation, uploading, and
 * launching of CUDA graphs. It provides a high-level API for pipeline
 * implementations to build and execute computational graphs efficiently.
 */
class IGraphManager {
public:
    /**
     * Default constructor.
     */
    IGraphManager() = default;

    /**
     * Virtual destructor.
     */
    virtual ~IGraphManager() = default;

    IGraphManager(const IGraphManager &) = delete;
    IGraphManager(IGraphManager &&) = delete;

    IGraphManager &operator=(const IGraphManager &) = delete;
    IGraphManager &operator=(IGraphManager &&) = delete;

    /**
     * Instantiate the graph for execution.
     *
     * Converts the graph definition into an executable form. Must be called
     * after all nodes have been added and before launching.
     *
     * @throws std::runtime_error if instantiation fails
     */
    virtual void instantiate_graph() const = 0;

    /**
     * Upload the graph to the device.
     *
     * Prepares the graph for efficient execution on the GPU. Should be called
     * after instantiation and before the first launch.
     *
     * @param[in] stream CUDA stream for upload operation
     * @throws std::runtime_error if upload fails
     */
    virtual void upload_graph(cudaStream_t stream) const = 0;

    /**
     * Launch the graph on the specified stream.
     *
     * Executes the pre-built graph on the given CUDA stream. The graph must
     * have been created, instantiated, and uploaded before calling this method.
     *
     * @param[in] stream CUDA stream for graph execution
     * @throws std::runtime_error if launch fails
     */
    virtual void launch_graph(cudaStream_t stream) const = 0;

    /**
     * Get the executable graph handle.
     *
     * Returns the CUDA graph execution handle, which can be used for dynamic
     * parameter updates via cuGraphExecKernelNodeSetParams.
     *
     * @return The graph execution handle
     * @throws std::runtime_error if graph is not instantiated
     */
    [[nodiscard]] virtual CUgraphExec get_exec() const = 0;

    /**
     * Add kernel node(s) to the graph via a graph node provider.
     *
     * Delegates to the provided graph node provider to add its kernel node(s)
     * to the managed graph. The provider specifies dependencies and returns
     * the created node handles.
     *
     * @param[in] graph_node_provider Provider that will add nodes to the graph
     * @param[in] deps Dependency nodes that must complete before these nodes
     * execute
     * @return Span of created graph node handles (can contain single or multiple nodes)
     * @throws std::runtime_error if node addition fails
     */
    [[nodiscard]] virtual std::span<const CUgraphNode> add_kernel_node(
            gsl_lite::not_null<IGraphNodeProvider *> graph_node_provider,
            std::span<const CUgraphNode> deps) const = 0;
};

} // namespace framework::pipeline

#endif // FRAMEWORK_CORE_IGRAPH_MANAGER_HPP
