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
 * @file fft_plugin_bench.cpp
 * @brief Google Benchmark for TensorRT FFT plugin performance
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

void bm_fft_plugin_impl(benchmark::State &state, const ExecutionMode mode) {
    // Get FFT parameters from benchmark args
    const auto fft_size = static_cast<std::int32_t>(state.range(0));
    const auto batch_size = static_cast<std::int32_t>(state.range(1));

    StdioLogger logger(nvinfer1::ILogger::Severity::kERROR);

    // Load custom plugins
    if (!init_ran_plugins(&logger)) {
        state.SkipWithError("Failed to load custom plugins");
        return;
    }

    // Load engine
    const TrtEngine engine("fft_test.trtengine", logger);

    // Get execution context
    auto *context = engine.get_context();
    auto *cuda_engine = engine.get_engine();

    // Create CUDA stream
    const CudaStream stream;

    // Create tensors for FFT input and output
    CudaTensor<float> input_real({.nbDims = 2, .d = {batch_size, fft_size}}, "Input Real");
    CudaTensor<float> input_imag({.nbDims = 2, .d = {batch_size, fft_size}}, "Input Imag");
    CudaTensor<float> output_real({.nbDims = 2, .d = {batch_size, fft_size}}, "Output Real");
    CudaTensor<float> output_imag({.nbDims = 2, .d = {batch_size, fft_size}}, "Output Imag");

    // Initialize input data
    input_real.copy_to_device(stream.get());
    input_imag.copy_to_device(stream.get());
    output_real.copy_to_device(stream.get());
    output_imag.copy_to_device(stream.get());

    // Bind tensors
    TensorBinder binder;
    if (!binder.bind("arg0", input_real, "input_real buffer")
                 .bind("arg1", input_imag, "input_imag buffer")
                 .bind("result0", output_real, "output_real buffer")
                 .bind("result1", output_imag, "output_imag buffer")
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

void bm_fft_plugin_stream(benchmark::State &state) {
    bm_fft_plugin_impl(state, ExecutionMode::Stream);
}

void bm_fft_plugin_graph(benchmark::State &state) {
    bm_fft_plugin_impl(state, ExecutionMode::Graph);
}

BENCHMARK(bm_fft_plugin_stream)
        ->Args({2048, 4})
        ->ArgNames({"fft_size", "batch"})
        ->Unit(benchmark::kMicrosecond)
        ->MinWarmUpTime(2.0);

BENCHMARK(bm_fft_plugin_graph)
        ->Args({2048, 4})
        ->ArgNames({"fft_size", "batch"})
        ->Unit(benchmark::kMicrosecond)
        ->MinWarmUpTime(2.0);

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

} // namespace

/// Google Benchmark main function
BENCHMARK_MAIN();
