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
#include <span>
#include <stdexcept>
#include <string>

#include <driver_types.h>
#include <quill/LogMacros.h>

#include <cuda.h>

#include "log/rt_log_macros.hpp"
#include "pipeline/graph.hpp"
#include "utils/core_log.hpp"
#include "utils/error_macros.hpp"

namespace framework::pipeline {

Graph::~Graph() {
    if (exec_ != nullptr) {
        RT_LOGC_DEBUG(
                utils::Core::CoreGraph,
                "Destroying CUDA graph executable: {}",
                static_cast<void *>(exec_));
        FRAMEWORK_CUDA_DRIVER_CHECK_NO_THROW(cuGraphExecDestroy(exec_));
        exec_ = nullptr;
    }

    if (graph_ != nullptr) {
        RT_LOGC_DEBUG(
                utils::Core::CoreGraph, "Destroying CUDA graph: {}", static_cast<void *>(graph_));
        FRAMEWORK_CUDA_DRIVER_CHECK_NO_THROW(cuGraphDestroy(graph_));
        graph_ = nullptr;
    }
}

void Graph::create() {
    RT_LOGC_DEBUG(utils::Core::CoreGraph, "Creating CUDA graph");

    // FLAGS is passed to cuGraphCreate to specify the graph creation
    constexpr unsigned int DEFAULT_GRAPH_FLAGS = 0;
    const CUresult result = cuGraphCreate(&graph_, DEFAULT_GRAPH_FLAGS);

    if (result != CUDA_SUCCESS) {
        const std::string error_msg =
                std::format("Failed to create CUDA graph: {}", static_cast<int>(result));
        RT_LOGC_ERROR(utils::Core::CoreGraph, "{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    RT_LOGC_INFO(utils::Core::CoreGraph, "CUDA graph created: {}", static_cast<void *>(graph_));
}

CUgraphNode Graph::add_kernel_node(
        const std::span<const CUgraphNode> deps, const CUDA_KERNEL_NODE_PARAMS &params) {

    RT_LOGC_DEBUG(
            utils::Core::CoreGraph,
            "Adding kernel node to graph (depends on {} nodes)",
            deps.size());

    CUgraphNode node{};
    const CUresult result = cuGraphAddKernelNode(&node, graph_, deps.data(), deps.size(), &params);

    if (result != CUDA_SUCCESS) {
        const std::string error_msg =
                std::format("Failed to add kernel node to graph: {}", static_cast<int>(result));
        RT_LOGC_ERROR(utils::Core::CoreGraph, "{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    RT_LOGC_DEBUG(utils::Core::CoreGraph, "Kernel node added: {}", static_cast<void *>(node));
    return node;
}

CUgraphNode
Graph::add_child_graph_node(const std::span<const CUgraphNode> deps, CUgraph child_graph) {

    RT_LOGC_DEBUG(
            utils::Core::CoreGraph,
            "Adding child graph node to graph (depends on {} nodes)",
            deps.size());

    CUgraphNode node{};
    const CUresult result =
            cuGraphAddChildGraphNode(&node, graph_, deps.data(), deps.size(), child_graph);

    if (result != CUDA_SUCCESS) {
        const std::string error_msg = std::format(
                "Failed to add child graph node to graph: {}", static_cast<int>(result));
        RT_LOGC_ERROR(utils::Core::CoreGraph, "{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    RT_LOGC_DEBUG(utils::Core::CoreGraph, "Child graph node added: {}", static_cast<void *>(node));
    return node;
}

void Graph::instantiate(const unsigned int flags) {
    RT_LOGC_DEBUG(utils::Core::CoreGraph, "Instantiating CUDA graph with flags={}", flags);

    const CUresult result = cuGraphInstantiate(&exec_, graph_, flags);

    if (result != CUDA_SUCCESS) {
        const std::string error_msg =
                std::format("Failed to instantiate CUDA graph: {}", static_cast<int>(result));
        RT_LOGC_ERROR(utils::Core::CoreGraph, "{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    is_instantiated_ = true;
    RT_LOGC_INFO(
            utils::Core::CoreGraph, "CUDA graph instantiated: exec={}", static_cast<void *>(exec_));
}

void Graph::upload(cudaStream_t stream) {
    RT_LOGC_DEBUG(
            utils::Core::CoreGraph,
            "Uploading CUDA graph to stream {}",
            static_cast<void *>(stream));

    if (!is_instantiated_) {
        const std::string error_msg = "Cannot upload graph: graph has not been instantiated";
        RT_LOGC_ERROR(utils::Core::CoreGraph, "{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    const CUresult result = cuGraphUpload(exec_, stream);

    if (result != CUDA_SUCCESS) {
        const std::string error_msg =
                std::format("Failed to upload CUDA graph: {}", static_cast<int>(result));
        RT_LOGC_ERROR(utils::Core::CoreGraph, "{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    RT_LOGC_INFO(utils::Core::CoreGraph, "CUDA graph uploaded");
}

void Graph::launch(cudaStream_t stream) {
    RT_LOGC_DEBUG(
            utils::Core::CoreGraph,
            "Launching CUDA graph on stream {}",
            static_cast<void *>(stream));

    if (!is_instantiated_) {
        const std::string error_msg = "Cannot launch graph: graph has not been instantiated";
        RT_LOGC_ERROR(utils::Core::CoreGraph, "{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    const CUresult result = cuGraphLaunch(exec_, stream);

    if (result != CUDA_SUCCESS) {
        const std::string error_msg =
                std::format("Failed to launch CUDA graph: {}", static_cast<int>(result));
        RT_LOGC_ERROR(utils::Core::CoreGraph, "{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    RT_LOGC_DEBUG(utils::Core::CoreGraph, "CUDA graph launched");
}

} // namespace framework::pipeline
