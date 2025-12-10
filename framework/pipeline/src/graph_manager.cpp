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

#include <memory>
#include <span>
#include <stdexcept>
#include <string>

#include <driver_types.h>
#include <quill/LogMacros.h>

#include <gsl-lite/gsl-lite.hpp>

#include <cuda.h>

#include "log/rt_log_macros.hpp"
#include "pipeline/graph.hpp"
#include "pipeline/graph_manager.hpp"
#include "pipeline/igraph.hpp"
#include "pipeline/igraph_node_provider.hpp"
#include "utils/core_log.hpp"

namespace framework::pipeline {

GraphManager::GraphManager() : main_graph_(std::make_unique<Graph>()) {
    RT_LOGC_DEBUG(
            utils::Core::CoreGraphManager, "GraphManager created - creating underlying CUDA graph");
    main_graph_->create();
    RT_LOGC_INFO(utils::Core::CoreGraphManager, "GraphManager initialized with empty CUDA graph");
}

void GraphManager::instantiate_graph() const {
    RT_LOGC_DEBUG(utils::Core::CoreGraphManager, "Instantiating graph via GraphManager");

    if (!main_graph_) {
        const std::string error_msg = "GraphManager::instantiate_graph() called with null graph";
        RT_LOGC_ERROR(utils::Core::CoreGraphManager, "{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    // FLAGS is passed to cuGraphInstantiate to specify the graph instantiation
    constexpr unsigned int DEFAULT_GRAPH_INSTANTIATION_FLAGS = 0;
    main_graph_->instantiate(DEFAULT_GRAPH_INSTANTIATION_FLAGS);
    RT_LOGC_INFO(utils::Core::CoreGraphManager, "Graph instantiated successfully via GraphManager");
}

void GraphManager::upload_graph(cudaStream_t stream) const {
    RT_LOGC_DEBUG(utils::Core::CoreGraphManager, "Uploading graph via GraphManager");

    if (!main_graph_) {
        const std::string error_msg = "GraphManager::upload_graph() called with null graph";
        RT_LOGC_ERROR(utils::Core::CoreGraphManager, "{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    main_graph_->upload(stream);

    RT_LOGC_INFO(utils::Core::CoreGraphManager, "Graph uploaded successfully via GraphManager");
}

void GraphManager::launch_graph(cudaStream_t stream) const {
    RT_LOGC_DEBUG(
            utils::Core::CoreCudaRuntime,
            "Launching graph via GraphManager on stream {}",
            static_cast<void *>(stream));

    if (!main_graph_) {
        const std::string error_msg = "GraphManager::launch_graph() called with null graph";
        RT_LOGC_ERROR(utils::Core::CoreGraphManager, "{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    main_graph_->launch(stream);
}

CUgraphExec GraphManager::get_exec() const {
    if (!main_graph_) {
        const std::string error_msg = "GraphManager::get_exec() called with null graph";
        RT_LOGC_ERROR(utils::Core::CoreGraphManager, "{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    return main_graph_->exec_handle();
}

std::span<const CUgraphNode> GraphManager::add_kernel_node(
        const gsl_lite::not_null<IGraphNodeProvider *> graph_node_provider,
        const std::span<const CUgraphNode> deps) const {

    RT_LOGC_DEBUG(
            utils::Core::CoreCudaRuntime,
            "Adding kernel node(s) via GraphManager (depends on {} nodes)",
            deps.size());

    if (!main_graph_) {
        const std::string error_msg = "GraphManager::add_kernel_node() called with null graph";
        RT_LOGC_ERROR(utils::Core::CoreGraphManager, "{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    // Delegate to the graph node provider and return the node handles
    return graph_node_provider->add_node_to_graph(
            gsl_lite::not_null<IGraph *>(main_graph_.get()), deps);
}

} // namespace framework::pipeline
