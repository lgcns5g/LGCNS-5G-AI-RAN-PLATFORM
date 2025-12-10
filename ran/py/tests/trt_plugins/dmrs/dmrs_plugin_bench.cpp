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
 * @file dmrs_plugin_bench.cpp
 * @brief Google Benchmark for TensorRT DMRS plugin performance
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

/// Benchmark the DMRS plugin with specified execution mode
void bm_dmrs_plugin_impl(benchmark::State &state, const ExecutionMode mode) {
    // Get DMRS parameters from benchmark args
    const auto n_f = static_cast<std::int32_t>(state.range(0));
    const auto n_dmrs_symbols = static_cast<std::int32_t>(state.range(1));

    // Fixed parameters
    static constexpr std::int32_t SLOT_NUMBER = 0;
    static constexpr std::int32_t N_DMRS_ID = 0;
    static constexpr std::int32_t N_SYMBOLS_PER_SLOT = 14;
    static constexpr std::int32_t N_REAL_IMAG = 2; // [real, imag]

    StdioLogger logger(nvinfer1::ILogger::Severity::kERROR);

    // Load custom plugins
    if (!init_ran_plugins(&logger)) {
        state.SkipWithError("Failed to load custom plugins");
        return;
    }

    // Load engine
    const TrtEngine engine("dmrs_test.trtengine", logger);

    // Get execution context
    auto *context = engine.get_context();
    auto *cuda_engine = engine.get_engine();

    // Create CUDA stream
    const CudaStream stream;

    // Create tensors
    CudaTensor<std::int32_t> slot({.nbDims = 0}, "Slot");
    CudaTensor<std::int32_t> dmrs_id({.nbDims = 0}, "DMRS ID");
    CudaTensor<std::int32_t> scr_seq(
            {.nbDims = 3, .d = {N_SYMBOLS_PER_SLOT, n_dmrs_symbols, n_f}}, "Scrambling Sequence");
    CudaTensor<float> rdmrs(
            {.nbDims = 4, .d = {N_REAL_IMAG, N_SYMBOLS_PER_SLOT, n_dmrs_symbols, n_f / 2}}, "DMRS");

    // Initialize input data
    slot[0] = SLOT_NUMBER;
    dmrs_id[0] = N_DMRS_ID;
    slot.copy_to_device(stream.get());
    dmrs_id.copy_to_device(stream.get());
    scr_seq.copy_to_device(stream.get());
    rdmrs.copy_to_device(stream.get());

    // Bind tensors
    TensorBinder binder;
    if (!binder.bind("arg0", slot, "slot buffer")
                 .bind("arg1", dmrs_id, "dmrs_id buffer")
                 .bind("result0", rdmrs, "r_dmrs buffer")
                 .bind("result1", scr_seq, "scr_seq buffer")
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

void bm_dmrs_plugin_stream(benchmark::State &state) {
    bm_dmrs_plugin_impl(state, ExecutionMode::Stream);
}

void bm_dmrs_plugin_graph(benchmark::State &state) {
    bm_dmrs_plugin_impl(state, ExecutionMode::Graph);
}

BENCHMARK(bm_dmrs_plugin_stream)
        ->Args({3276, 2}) // N_F=3276, N_DMRS_SYMBOLS=2
        ->ArgNames({"n_f", "n_dmrs_symbols"})
        ->Unit(benchmark::kMicrosecond)
        ->MinWarmUpTime(2);

BENCHMARK(bm_dmrs_plugin_graph)
        ->Args({3276, 2}) // N_F=3276, N_DMRS_SYMBOLS=2
        ->ArgNames({"n_f", "n_dmrs_symbols"})
        ->Unit(benchmark::kMicrosecond)
        ->MinWarmUpTime(2);

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

} // namespace

/// Google Benchmark main function
BENCHMARK_MAIN();
