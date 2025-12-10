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

#ifndef RAN_PUSCH_PIPELINE_RUNNER_HPP
#define RAN_PUSCH_PIPELINE_RUNNER_HPP

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <driver_types.h>

#include "aerial_tv/cuphy_pusch_tv.hpp"
#include "memory/unique_ptr_utils.hpp"
#include "pipeline/types.hpp"
#include "pusch/pusch_module_factories.hpp"
#include "pusch/pusch_pipeline.hpp"
#include "ran_common.hpp"

// Forward declarations
namespace framework::utils {
class CudaStream;
}

namespace ran::pusch {

/**
 * Pipeline execution helper for benchmarks and tests
 *
 * Encapsulates pipeline setup, configuration, and execution for benchmarking.
 * Follows RAII - pipeline is fully initialized upon construction.
 */
class PuschPipelineRunner {
public:
    /**
     * Construct and initialize pipeline with test vector
     *
     * @param[in] test_vector_path Full path to H5 test vector file
     * @param[in] execution_mode Execution mode (Stream or Graph)
     */
    PuschPipelineRunner(
            const std::string &test_vector_path, framework::pipeline::ExecutionMode execution_mode);

    ~PuschPipelineRunner() = default;

    PuschPipelineRunner(const PuschPipelineRunner &) = delete;
    PuschPipelineRunner &operator=(const PuschPipelineRunner &) = delete;
    /** Move constructor */
    PuschPipelineRunner(PuschPipelineRunner &&) noexcept = default;
    /**
     * Move assignment operator
     *
     * @return Reference to this object
     */
    PuschPipelineRunner &operator=(PuschPipelineRunner &&) noexcept = default;

    /**
     * Configure I/O and dynamic parameters
     *
     * Call once before execute_once() or after parameter changes.
     *
     * @param[in] stream CUDA stream for execution
     */
    void configure(const framework::utils::CudaStream &stream);

    /**
     * Warmup pipeline execution paths
     *
     * @param[in] stream CUDA stream for warmup
     */
    void warmup(cudaStream_t stream);

    /**
     * Execute one iteration of the pipeline
     *
     * Only performs execute_stream() or execute_graph() call.
     *
     * @param[in] stream CUDA stream for execution
     */
    void execute_once(const framework::utils::CudaStream &stream);

    /**
     * Get pipeline execution mode
     *
     * @return Execution mode (Stream or Graph)
     */
    [[nodiscard]] framework::pipeline::ExecutionMode get_execution_mode() const;

    /**
     * Get number of external outputs
     *
     * @return Count of external output ports
     */
    [[nodiscard]] std::size_t get_num_external_outputs() const;

    /**
     * Get external outputs after execution
     *
     * @return Vector of external output port information
     */
    [[nodiscard]] const std::vector<framework::pipeline::PortInfo> &get_external_outputs() const {
        return external_outputs_;
    }

    /**
     * Get PhyParams
     *
     * @return Physical layer parameters
     */
    [[nodiscard]] const ran::common::PhyParams &get_phy_params() const { return phy_params_; }

    /**
     * Get test vector
     *
     * @return Test vector used for pipeline configuration
     */
    [[nodiscard]] const ran::aerial_tv::CuphyPuschTestVector &get_test_vector() const {
        return *test_vector_;
    }

private:
    ran::common::PhyParams phy_params_{};
    std::unique_ptr<PuschModuleFactory> module_factory_;
    std::unique_ptr<PuschPipeline> pipeline_;
    std::unique_ptr<ran::aerial_tv::CuphyPuschTestVector> test_vector_;
    std::vector<framework::memory::UniqueDevicePtr<std::byte>> input_device_ptrs_;
    std::vector<framework::pipeline::PortInfo> external_inputs_;
    std::vector<framework::pipeline::PortInfo> external_outputs_;
};

} // namespace ran::pusch

#endif // RAN_PUSCH_PIPELINE_RUNNER_HPP
