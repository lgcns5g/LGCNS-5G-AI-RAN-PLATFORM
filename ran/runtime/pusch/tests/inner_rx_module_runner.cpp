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
#include <vector>

#include <driver_types.h>

#include <gsl-lite/gsl-lite.hpp>

#include <cuda.h>

#include "aerial_tv/aerial_tv_utils.hpp"
#include "aerial_tv/cuphy_pusch_tv.hpp"
#include "inner_rx_module_runner.hpp"
#include "memory/unique_ptr_utils.hpp"
#include "pipeline/graph_manager.hpp"
#include "pipeline/igraph_node_provider.hpp"
#include "pipeline/types.hpp"
#include "pusch/inner_rx_module.hpp"
#include "pusch_test_utils.hpp"
#include "ran_common.hpp"
#include "utils/cuda_stream.hpp"

namespace ran::pusch {

namespace {
namespace pipeline = framework::pipeline;
namespace memory = framework::memory;
namespace utils = framework::utils;
} // namespace

InnerRxModuleRunner::InnerRxModuleRunner(
        const std::string &test_vector_path,
        const framework::pipeline::ExecutionMode execution_mode)
        : execution_mode_{execution_mode},
          test_vector_{std::make_unique<ran::aerial_tv::CuphyPuschTestVector>(
                  test_vector_path.c_str())} {

    // Load test vector
    phy_params_ = ran::aerial_tv::to_phy_params(*test_vector_);

    // Create InnerRxModule instance
    const InnerRxModule::StaticParams params{
            .phy_params = phy_params_, .execution_mode = execution_mode_};

    inner_rx_module_ = std::make_unique<InnerRxModule>("inner_rx_module", params);

    // Set connection copy mode based on execution mode
    if (execution_mode_ == pipeline::ExecutionMode::Stream) {
        inner_rx_module_->set_connection_copy_mode("xtf", pipeline::ConnectionCopyMode::ZeroCopy);
    } else {
        inner_rx_module_->set_connection_copy_mode("xtf", pipeline::ConnectionCopyMode::Copy);
    }

    // Allocate memory for outputs and setup the module with memory slice
    pipeline::ModuleMemorySlice memory_slice{};
    const auto memory_requirements = inner_rx_module_->get_requirements();
    const std::size_t output_size = memory_requirements.device_tensor_bytes;
    output_device_ptr_ = memory::make_unique_device<std::byte>(output_size);
    memory_slice.device_tensor_ptr = output_device_ptr_.get();
    inner_rx_module_->setup_memory(memory_slice);

    // Prepare input tensors
    // Note: Using default stream for setup phase - data must be ready before configure
    std::vector<pipeline::PortInfo> inputs;
    inputs.reserve(1);
    input_device_ptrs_ =
            prepare_pusch_inputs(inputs, phy_params_, *test_vector_, cudaStreamDefault);

    // Set the inputs
    inner_rx_module_->set_inputs(inputs);

    // For graph mode, create the graph manager
    if (execution_mode_ == pipeline::ExecutionMode::Graph) {
        graph_manager_ = std::make_unique<pipeline::GraphManager>();
    }
}

void InnerRxModuleRunner::configure(const utils::CudaStream &stream) {
    // Configure I/O
    const pipeline::DynamicParams dynamic_params{};
    inner_rx_module_->configure_io(dynamic_params, stream.get());

    // Synchronize after configure
    if (!stream.synchronize()) {
        throw std::runtime_error("Stream synchronization failed during configure");
    }

    // Warmup must be called before building graph
    inner_rx_module_->warmup(stream.get());

    // For graph mode, build the graph
    if (execution_mode_ == pipeline::ExecutionMode::Graph) {
        auto *graph_node_provider = inner_rx_module_->as_graph_node_provider();

        // Add module node(s) to graph with no dependencies
        const std::vector<CUgraphNode> no_deps{};
        const auto nodes = graph_manager_->add_kernel_node(
                gsl_lite::not_null<pipeline::IGraphNodeProvider *>(graph_node_provider), no_deps);
        if (nodes.empty()) {
            throw std::runtime_error("Failed to add kernel nodes to graph");
        }

        // Instantiate and upload graph
        graph_manager_->instantiate_graph();
        graph_manager_->upload_graph(stream.get());

        // Update graph node parameters
        auto *const exec = graph_manager_->get_exec();
        graph_node_provider->update_graph_node_params(exec, dynamic_params);
    }
}

void InnerRxModuleRunner::warmup(const utils::CudaStream &stream) {
    inner_rx_module_->warmup(stream.get());
}

void InnerRxModuleRunner::execute_once(const utils::CudaStream &stream) {
    if (execution_mode_ == pipeline::ExecutionMode::Stream) {
        inner_rx_module_->execute(stream.get());
    } else {
        graph_manager_->launch_graph(stream.get());
    }
}

std::vector<pipeline::PortInfo> InnerRxModuleRunner::get_outputs() const {
    return inner_rx_module_->get_outputs();
}

} // namespace ran::pusch
