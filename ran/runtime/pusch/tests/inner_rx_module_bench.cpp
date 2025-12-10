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

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <format>
#include <string>
#include <string_view>
#include <vector>

#include <benchmark/benchmark.h>

#include "aerial_tv/aerial_tv_utils.hpp"
#include "inner_rx_module_runner.hpp"
#include "log/components.hpp"
#include "log/rt_log.hpp"
#include "log/rt_log_macros.hpp"
#include "pipeline/types.hpp"
#include "pusch_test_utils.hpp"
#include "task/nvtx.hpp"
#include "utils/cuda_stream.hpp"

namespace {

using ran::pusch::InnerRxModuleRunner;
namespace pipeline = framework::pipeline;
using framework::utils::CudaStream;

constexpr std::string_view TEST_VECTOR_DIR = TEST_VECTOR_PATH;

/// Get full path to test vector file
std::string get_test_vector_path(const std::string_view filename) {
    return std::format("{}/{}", TEST_VECTOR_DIR, filename);
}

/// Register statistics as benchmark user counters
void register_statistics(benchmark::State &state, const ran::pusch::BenchmarkStatistics &stats) {
    state.counters["min_us"] = benchmark::Counter(stats.min);
    state.counters["mean_us"] = benchmark::Counter(stats.mean);
    state.counters["median_us"] = benchmark::Counter(stats.median);
    state.counters["p95_us"] = benchmark::Counter(stats.p95);
    state.counters["max_us"] = benchmark::Counter(stats.max);
    state.counters["stddev_us"] = benchmark::Counter(stats.stddev);
}

/// Common benchmark implementation for InnerRx module
void bm_inner_rx_module_impl(
        benchmark::State &state,
        const std::string_view test_vector_filename,
        const pipeline::ExecutionMode execution_mode) {

    // Print test vector information before reducing log level
    RT_LOG_INFO("InnerRx Module Benchmark - Test Vector: {}", test_vector_filename);

    // Configure logger to Error level for benchmarking
    framework::log::Logger::set_level(framework::log::LogLevel::Error);

    // Get test vector path
    const auto test_vector_path = get_test_vector_path(test_vector_filename);
    if (!std::filesystem::exists(test_vector_path)) {
        state.SkipWithError(std::format("Test vector not found: {}", test_vector_path));
        return;
    }

    // Create runner and stream
    InnerRxModuleRunner runner(test_vector_path, execution_mode);
    const CudaStream stream;

    // Configure once (outside the benchmark loop - includes initial warmup)
    {
        NVTX_RANGE("InnerRxModule::Configure");
        runner.configure(stream);
    }

    // Additional warmup iterations
    {
        NVTX_RANGE("InnerRxModule::Warmup");
        constexpr std::size_t NUM_WARMUP = 10;
        for (std::size_t i = 0; i < NUM_WARMUP; ++i) {
            runner.execute_once(stream);
            if (!stream.synchronize()) {
                RT_LOG_ERROR("Stream synchronization failed during warmup");
                state.SkipWithError("CUDA stream synchronization failed during warmup");
                return;
            }
        }
    }

    // Benchmark loop
    std::vector<double> iteration_times;
    iteration_times.reserve(static_cast<std::size_t>(state.max_iterations));

    for ([[maybe_unused]] auto _ : state) {
        const auto start = std::chrono::high_resolution_clock::now();

        {
            NVTX_RANGE("InnerRxModule::ExecuteAndSynchronize");
            runner.execute_once(stream);
            if (!stream.synchronize()) {
                RT_LOG_ERROR("Stream synchronization failed during benchmark iteration");
                state.SkipWithError("CUDA stream synchronization failed");
                return;
            }
        }

        const auto end = std::chrono::high_resolution_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        iteration_times.push_back(static_cast<double>(duration.count()));
    }

    // Compute statistics and register as user counters
    register_statistics(state, ran::pusch::compute_benchmark_statistics(iteration_times));
}

void bm_inner_rx_module_stream(benchmark::State &state) {
    bm_inner_rx_module_impl(
            state, ran::aerial_tv::TEST_HDF5_FILES[1], pipeline::ExecutionMode::Stream);
}

void bm_inner_rx_module_graph(benchmark::State &state) {
    bm_inner_rx_module_impl(
            state, ran::aerial_tv::TEST_HDF5_FILES[1], pipeline::ExecutionMode::Graph);
}

} // namespace

BENCHMARK(bm_inner_rx_module_stream)->Unit(benchmark::kMicrosecond)->MinWarmUpTime(1);
BENCHMARK(bm_inner_rx_module_graph)->Unit(benchmark::kMicrosecond)->MinWarmUpTime(1);
