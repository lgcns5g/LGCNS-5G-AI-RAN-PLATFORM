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
 * @file cholesky_factor_inv_plugin_bench.cpp
 * @brief Google Benchmark for TensorRT Cholesky Factor Inverse plugin performance
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <format>
#include <optional>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <benchmark/benchmark.h>
#include <driver_types.h>

#include "trt_test_utils.hpp"

namespace {

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

using namespace ran::trt_utils;

/// Register statistics as benchmark user counters
void register_statistics(benchmark::State &state, const Statistics &stats) {
    state.counters["min_us"] = benchmark::Counter(stats.min);
    state.counters["median_us"] = benchmark::Counter(stats.median);
    state.counters["p95_us"] = benchmark::Counter(stats.p95);
    state.counters["max_us"] = benchmark::Counter(stats.max);
    state.counters["stddev_us"] = benchmark::Counter(stats.stddev);
}

/// Benchmark the Cholesky Factor Inverse plugin with specified execution mode
void bm_cholesky_factor_inv_plugin_impl(benchmark::State &state, const ExecutionMode mode) {
    // Get parameters from benchmark args
    const auto matrix_size = static_cast<std::int32_t>(state.range(0));
    const auto batch_size = static_cast<std::int32_t>(state.range(1));

    StdioLogger logger(nvinfer1::ILogger::Severity::kERROR);

    // Load custom plugins
    if (!init_ran_plugins(&logger)) {
        state.SkipWithError("Failed to load custom plugins");
        return;
    }

    // Load engine
    const TrtEngine engine("cholesky_test.trtengine", logger);

    // Get execution context
    auto *context = engine.get_context();
    auto *cuda_engine = engine.get_engine();

    // Create CUDA stream
    const CudaStream stream;

    // Create tensors
    CudaTensor<float> input({.nbDims = 3, .d = {batch_size, matrix_size, matrix_size}}, "Input");
    CudaTensor<float> output({.nbDims = 3, .d = {batch_size, matrix_size, matrix_size}}, "Output");

    // Initialize input data
    input.copy_to_device(stream.get());
    output.copy_to_device(stream.get());

    // Bind tensors
    TensorBinder binder;
    if (!binder.bind("arg0", input, "input buffer")
                 .bind("result0", output, "output buffer")
                 .apply(context, cuda_engine, logger)) {
        state.SkipWithError("Failed to bind tensors");
        return;
    }

    stream.synchronize();

    // Prepare executor with warmup
    TrtExecutor executor(mode);
    try {
        executor.prepare(context, stream.get(), 1 /* warmup launches */);
    } catch (const std::exception &e) {
        state.SkipWithError(std::format("Failed to prepare: {}", e.what()));
        return;
    }

    // Vector to store iteration times (in microseconds)
    std::vector<double> iteration_times{};
    iteration_times.reserve(10000);

    // Benchmark loop
    for ([[maybe_unused]] auto _ : state) {
        const auto start = std::chrono::high_resolution_clock::now();

        executor.execute(stream.get());

        stream.synchronize();

        const auto end = std::chrono::high_resolution_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        iteration_times.push_back(static_cast<double>(duration.count()));
    }

    // Compute statistics and register as user counters
    register_statistics(state, compute_statistics(iteration_times));
}

void bm_cholesky_factor_inv_plugin_stream(benchmark::State &state) {
    bm_cholesky_factor_inv_plugin_impl(state, ExecutionMode::Stream);
}

void bm_cholesky_factor_inv_plugin_graph(benchmark::State &state) {
    bm_cholesky_factor_inv_plugin_impl(state, ExecutionMode::Graph);
}

BENCHMARK(bm_cholesky_factor_inv_plugin_stream)
        ->Args({2, 1}) // MATRIX_SIZE=2, BATCH_SIZE=1
        ->ArgNames({"matrix_size", "batch"})
        ->Unit(benchmark::kMicrosecond)
        ->MinWarmUpTime(2.0);

BENCHMARK(bm_cholesky_factor_inv_plugin_graph)
        ->Args({2, 1}) // MATRIX_SIZE=2, BATCH_SIZE=1
        ->ArgNames({"matrix_size", "batch"})
        ->Unit(benchmark::kMicrosecond)
        ->MinWarmUpTime(2.0);

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

} // namespace

/// Google Benchmark main function
BENCHMARK_MAIN();
