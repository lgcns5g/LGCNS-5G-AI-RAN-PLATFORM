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

#ifndef RAN_INNER_RX_MODULE_RUNNER_HPP
#define RAN_INNER_RX_MODULE_RUNNER_HPP

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <driver_types.h>

#include <cuda_runtime_api.h>

#include "aerial_tv/cuphy_pusch_tv.hpp"
#include "log/components.hpp"
#include "memory/unique_ptr_utils.hpp"
#include "pipeline/graph_manager.hpp"
#include "pipeline/types.hpp"
#include "pusch/inner_rx_module.hpp"
#include "ran_common.hpp"
#include "utils/core_log.hpp"

// Forward declarations
namespace framework::utils {
class CudaStream;
}

namespace ran::pusch {

/**
 * InnerRx module execution helper for benchmarks and tests
 *
 * Encapsulates inner_rx module setup, configuration, and execution.
 * Follows RAII - module is fully initialized upon construction.
 */
class InnerRxModuleRunner {
public:
    /**
     * Construct and initialize inner_rx module with test vector
     *
     * @param[in] test_vector_path Full path to H5 test vector file
     * @param[in] execution_mode Execution mode (Stream or Graph)
     */
    InnerRxModuleRunner(
            const std::string &test_vector_path, framework::pipeline::ExecutionMode execution_mode);

    ~InnerRxModuleRunner() = default;

    InnerRxModuleRunner(const InnerRxModuleRunner &) = delete;
    InnerRxModuleRunner &operator=(const InnerRxModuleRunner &) = delete;
    /** Move constructor */
    InnerRxModuleRunner(InnerRxModuleRunner &&) noexcept = default;
    /**
     * Move assignment operator
     *
     * @return Reference to this object
     */
    InnerRxModuleRunner &operator=(InnerRxModuleRunner &&) noexcept = default;

    /**
     * Configure I/O and dynamic parameters
     *
     * Call once before execute_once() or after parameter changes.
     * Note: This method includes an initial warmup call.
     *
     * @param[in] stream CUDA stream for execution
     */
    void configure(const framework::utils::CudaStream &stream);

    /**
     * Warmup module execution paths
     *
     * Optional: Additional warmup can be performed by calling this method
     * or by executing the module multiple times before benchmarking.
     *
     * @param[in] stream CUDA stream for warmup
     */
    void warmup(const framework::utils::CudaStream &stream);

    /**
     * Execute one iteration of the module
     *
     * Only performs execute() or launch_graph() call.
     *
     * @param[in] stream CUDA stream for execution
     */
    void execute_once(const framework::utils::CudaStream &stream);

    /**
     * Get module execution mode
     *
     * @return Execution mode (Stream or Graph)
     */
    [[nodiscard]] framework::pipeline::ExecutionMode get_execution_mode() const {
        return execution_mode_;
    }

    /**
     * Get module outputs after execution
     *
     * @return Vector of output port information
     */
    [[nodiscard]] std::vector<framework::pipeline::PortInfo> get_outputs() const;

    /**
     * Get PhyParams
     *
     * @return Physical layer parameters
     */
    [[nodiscard]] const ran::common::PhyParams &get_phy_params() const { return phy_params_; }

    /**
     * Get test vector
     *
     * @return Test vector used for module configuration
     */
    [[nodiscard]] const ran::aerial_tv::CuphyPuschTestVector &get_test_vector() const {
        return *test_vector_;
    }

private:
    framework::pipeline::ExecutionMode execution_mode_;
    ran::common::PhyParams phy_params_{};
    std::unique_ptr<InnerRxModule> inner_rx_module_;
    std::unique_ptr<framework::pipeline::GraphManager> graph_manager_;
    std::unique_ptr<ran::aerial_tv::CuphyPuschTestVector> test_vector_;
    std::vector<framework::memory::UniqueDevicePtr<std::byte>> input_device_ptrs_;
    framework::memory::UniqueDevicePtr<std::byte> output_device_ptr_;
};

} // namespace ran::pusch

#endif // RAN_INNER_RX_MODULE_RUNNER_HPP
