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

#ifndef RAN_PUSCH_TEST_UTILS_HPP
#define RAN_PUSCH_TEST_UTILS_HPP

#include <cstddef>
#include <string>
#include <vector>

#include <driver_types.h>

#include "aerial_tv/cuphy_pusch_tv.hpp"
#include "memory/unique_ptr_utils.hpp"
#include "pipeline/types.hpp"
#include "ran_common.hpp"
#include "tensor/tensor_info.hpp"

namespace ran::pusch {

/**
 * Prepare PUSCH inputs with managed memory using stream-aware copy
 *
 * @param[out] inputs Input ports
 * @param[in] phy_params Physical layer parameters
 * @param[in] test_vector Test vector
 * @param[in] stream CUDA stream for async memory copy
 * @return Managed device pointers (memory freed automatically via RAII)
 */
[[nodiscard]] std::vector<framework::memory::UniqueDevicePtr<std::byte>> prepare_pusch_inputs(
        std::vector<framework::pipeline::PortInfo> &inputs,
        const ran::common::PhyParams &phy_params,
        const ran::aerial_tv::CuphyPuschTestVector &test_vector,
        cudaStream_t stream);

/**
 * Copy device tensor data to host vector using stream-aware async copy
 *
 * Computes number of elements from tensor_info, creates a std::vector<T>
 * with that capacity, and copies data from device_ptr to the vector using
 * cudaMemcpyAsync on the specified stream for proper synchronization.
 *
 * @param[in] tensor_info Tensor information containing dimensions
 * @param[in] device_ptr Device pointer to tensor data
 * @param[in] stream CUDA stream to use for async memory copy (ensures proper ordering)
 * @return Vector containing the copied data
 */
template <typename T>
[[nodiscard]] std::vector<T> tensor_to_host_vector(
        const framework::tensor::TensorInfo &tensor_info,
        const void *device_ptr,
        cudaStream_t stream);

/**
 * Create PUSCH pipeline specification for testing/benchmarking
 *
 * @param[in] instance_id Instance identifier for the pipeline
 * @param[in] phy_params Physical layer parameters
 * @param[in] execution_mode Execution mode (Stream or Graph)
 * @return Pipeline specification with all module configurations
 */
[[nodiscard]] framework::pipeline::PipelineSpec create_pusch_pipeline_spec(
        const std::string &instance_id,
        const ran::common::PhyParams &phy_params,
        framework::pipeline::ExecutionMode execution_mode =
                framework::pipeline::ExecutionMode::Stream);

/**
 * Benchmark timing statistics
 */
struct BenchmarkStatistics {
    double min{};        //!< Minimum value
    double max{};        //!< Maximum value
    double mean{};       //!< Arithmetic mean
    double median{};     //!< Median value (50th percentile)
    double stddev{};     //!< Standard deviation
    double p95{};        //!< 95th percentile
    std::size_t count{}; //!< Number of samples
};

/**
 * Compute statistics from benchmark timing samples
 *
 * @param[in] times Vector of timing measurements
 * @return Computed statistics
 */
[[nodiscard]] BenchmarkStatistics compute_benchmark_statistics(const std::vector<double> &times);

} // namespace ran::pusch

#endif // RAN_PUSCH_TEST_UTILS_HPP
