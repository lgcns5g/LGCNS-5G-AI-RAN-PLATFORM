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

/**
 * @file pusch_pipeline_bench.cpp
 * @brief Google Benchmark for PUSCH Pipeline performance
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
#include <driver_types.h>

#include "aerial_tv/aerial_tv_utils.hpp"
#include "log/components.hpp"
#include "log/rt_log.hpp"
#include "log/rt_log_macros.hpp"
#include "pipeline/types.hpp"
#include "pusch_pipeline_runner.hpp"
#include "pusch_test_utils.hpp"
#include "task/nvtx.hpp"
#include "utils/cuda_stream.hpp"

namespace {

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

using ran::pusch::PuschPipelineRunner;
namespace pipeline = framework::pipeline;
namespace utils = framework::utils;

/// Get test vector full path
std::string get_test_vector_path(const std::string_view filename) {
    const std::filesystem::path test_vector_dir{TEST_VECTOR_PATH};
    return (test_vector_dir / filename).string();
}

/// Register timing statistics as benchmark counters
void register_statistics(benchmark::State &state, const std::vector<double> &times) {
    const auto stats = ran::pusch::compute_benchmark_statistics(times);
    if (stats.count == 0) {
        return;
    }

    state.counters["min_us"] = benchmark::Counter(stats.min);
    state.counters["mean_us"] = benchmark::Counter(stats.mean);
    state.counters["median_us"] = benchmark::Counter(stats.median);
    state.counters["p95_us"] = benchmark::Counter(stats.p95);
    state.counters["max_us"] = benchmark::Counter(stats.max);
    state.counters["stddev_us"] = benchmark::Counter(stats.stddev);
}

/// Common benchmark implementation for PUSCH pipeline
void bm_pusch_pipeline_impl(
        benchmark::State &state,
        const std::string_view test_vector_filename,
        const pipeline::ExecutionMode execution_mode) {

    // Print test vector information before reducing log level
    RT_LOG_INFO("PUSCH Pipeline Benchmark - Test Vector: {}", test_vector_filename);

    framework::log::Logger::set_level(framework::log::LogLevel::Error);

    // Get test vector path
    const auto test_vector_path = get_test_vector_path(test_vector_filename);
    if (!std::filesystem::exists(test_vector_path)) {
        state.SkipWithError(std::format("Test vector not found: {}", test_vector_path));
        return;
    }

    // Create CUDA stream using RAII wrapper
    const utils::CudaStream stream;

    // Create pipeline runner
    PuschPipelineRunner runner{test_vector_path, execution_mode};

    // Configure I/O (done once outside benchmark loop)
    {
        NVTX_RANGE("PuschPipeline::Configure");
        runner.configure(stream);
    }

    // Warmup
    {
        NVTX_RANGE("PuschPipeline::Warmup");
        constexpr std::size_t NUM_WARMUP = 10;
        runner.warmup(stream.get());
        for (std::size_t i = 0; i < NUM_WARMUP; ++i) {
            runner.execute_once(stream);
            if (!stream.synchronize()) {
                RT_LOG_ERROR("Stream synchronization failed during warmup");
                state.SkipWithError("CUDA stream synchronization failed during warmup");
                return;
            }
        }
    }

    // Benchmark loop - measure pure execution time
    std::vector<double> iteration_times;
    iteration_times.reserve(1000);

    for ([[maybe_unused]] auto _ : state) {
        const auto start = std::chrono::high_resolution_clock::now();

        {
            NVTX_RANGE("PuschPipeline::ExecuteAndSynchronize");
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

    // Register statistics
    register_statistics(state, iteration_times);
}

/// Benchmark PUSCH pipeline in Stream mode
void bm_pusch_pipeline_stream(benchmark::State &state) {
    bm_pusch_pipeline_impl(
            state, ran::aerial_tv::TEST_HDF5_FILES[1], pipeline::ExecutionMode::Stream);
}

/// Benchmark PUSCH pipeline in Graph mode
void bm_pusch_pipeline_graph(benchmark::State &state) {
    bm_pusch_pipeline_impl(
            state, ran::aerial_tv::TEST_HDF5_FILES[1], pipeline::ExecutionMode::Graph);
}

// Register benchmarks
BENCHMARK(bm_pusch_pipeline_stream)->Unit(benchmark::kMicrosecond)->MinWarmUpTime(1);

BENCHMARK(bm_pusch_pipeline_graph)->Unit(benchmark::kMicrosecond)->MinWarmUpTime(1);

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

} // namespace

/// Google Benchmark main function
BENCHMARK_MAIN();
