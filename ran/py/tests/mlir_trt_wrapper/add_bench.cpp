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
 * @file add_bench.cpp
 * @brief Google Benchmark for MLIR-TensorRT add operation performance
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <benchmark/benchmark.h>
#include <driver_types.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

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

/// Template benchmark for add operation with different data types
template <typename T>
void bm_add_operation(benchmark::State &state, const std::string &engine_path) {
    static constexpr int32_t SIZE = 2;

    StdioLogger logger(nvinfer1::ILogger::Severity::kERROR);

    // Load engine
    const TrtEngine engine(engine_path, logger);

    // Get execution context
    auto *context = engine.get_context();
    auto *cuda_engine = engine.get_engine();

    // Create CUDA stream
    const CudaStream stream;

    // Create tensors
    CudaTensor<T> input1({.nbDims = 1, .d = {SIZE}}, "Input1");
    CudaTensor<T> input2({.nbDims = 1, .d = {SIZE}}, "Input2");
    CudaTensor<T> output({.nbDims = 1, .d = {SIZE}}, "Output");

    // Initialize input data: [12.34, 56.78] + [23.45, 67.89] = [35.79, 124.67]
    input1.host() = {from_float<T>(12.34F), from_float<T>(56.78F)};
    input2.host() = {from_float<T>(23.45F), from_float<T>(67.89F)};

    input1.copy_to_device(stream.get());
    input2.copy_to_device(stream.get());

    // Bind tensors
    TensorBinder binder;
    if (!binder.bind("arg0", input1, "first input")
                 .bind("arg1", input2, "second input")
                 .bind("result0", output, "output")
                 .apply(context, cuda_engine, logger)) {
        state.SkipWithError("Failed to bind tensors");
        return;
    }

    stream.synchronize();

    // Vector to store iteration times (in microseconds)
    std::vector<double> iteration_times{};
    iteration_times.reserve(10000);

    // Benchmark loop
    for (auto _ : state) {
        const auto start = std::chrono::high_resolution_clock::now();

        // Run the engine
        context->enqueueV3(stream.get());

        // Synchronize stream
        stream.synchronize();

        const auto end = std::chrono::high_resolution_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        iteration_times.push_back(static_cast<double>(duration.count()));
    }

    // Compute statistics and register as user counters
    register_statistics(state, compute_statistics(iteration_times));
    state.SetItemsProcessed(state.iterations());
}

/// Benchmark wrappers for each data type
void bm_add_float32(benchmark::State &state) {
    bm_add_operation<float>(state, "add_func_float32.trtengine");
}

void bm_add_float16(benchmark::State &state) {
    bm_add_operation<__half>(state, "add_func_float16.trtengine");
}

void bm_add_b_float16(benchmark::State &state) {
    bm_add_operation<__nv_bfloat16>(state, "add_func_bfloat16.trtengine");
}

void bm_add_int32(benchmark::State &state) {
    bm_add_operation<std::int32_t>(state, "add_func_int32.trtengine");
}

// Register benchmarks
BENCHMARK(bm_add_float32)->Unit(benchmark::kMicrosecond)->MinWarmUpTime(2.0);

BENCHMARK(bm_add_float16)->Unit(benchmark::kMicrosecond)->MinWarmUpTime(2.0);

BENCHMARK(bm_add_b_float16)->Unit(benchmark::kMicrosecond)->MinWarmUpTime(2.0);

BENCHMARK(bm_add_int32)->Unit(benchmark::kMicrosecond)->MinWarmUpTime(2.0);

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

} // namespace

/// Google Benchmark main function
BENCHMARK_MAIN();
