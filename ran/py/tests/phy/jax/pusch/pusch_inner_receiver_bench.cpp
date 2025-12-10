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
 * @file pusch_inner_receiver_bench.cpp
 * @brief Google Benchmark for PUSCH Inner Receiver TensorRT pipeline performance
 */

#include <algorithm>
#include <chrono>
#include <exception>
#include <format>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <NvInfer.h>
#include <benchmark/benchmark.h>
#include <driver_types.h>

#include <cuda_fp16.h>

#include "pusch_inner_receiver_test_utils.hpp"
#include "trt_test_utils.hpp"

namespace {

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

using namespace ran::trt_utils;
using ran::pusch_test_utils::PuschTestParams;

/// Register statistics as benchmark user counters
void register_statistics(benchmark::State &state, const Statistics &stats) {
    state.counters["min_us"] = benchmark::Counter(stats.min);
    state.counters["median_us"] = benchmark::Counter(stats.median);
    state.counters["p95_us"] = benchmark::Counter(stats.p95);
    state.counters["max_us"] = benchmark::Counter(stats.max);
    state.counters["stddev_us"] = benchmark::Counter(stats.stddev);
}

/// Common benchmark implementation for PUSCH Inner Receiver TensorRT pipeline
void bm_pusch_inner_receiver_impl(
        benchmark::State &state, const std::string_view filter_type, const ExecutionMode mode) {
    // Test parameters (matching the generated engine)
    const PuschTestParams params{};

    StdioLogger logger(nvinfer1::ILogger::Severity::kERROR);

    // Load custom plugins
    if (!init_ran_plugins(&logger)) {
        state.SkipWithError("Failed to load custom plugins");
        return;
    }

    // Load engine
    const std::string engine_name = std::format("pusch_inner_receiver_{}.trtengine", filter_type);
    const TrtEngine engine(engine_name, logger);

    // Get execution context
    auto *context = engine.get_context();
    auto *cuda_engine = engine.get_engine();

    // Create CUDA stream
    const CudaStream stream;

    // Create tensors
    // Note: XTF input and LLR output are float16 (__half) as required by the TensorRT engine
    CudaTensor<__half> xtf(
            {.nbDims = 4, .d = {params.num_rxant, params.n_sym, params.n_sc, params.n_real_imag}},
            "XTF Input");
    CudaTensor<__half> llr(
            {.nbDims = 4, .d = {params.n_datasym, params.n_sc, params.n_layer, params.qam_bits}},
            "LLR Output");
    CudaTensor<float> post_eq_noise_var_db({.nbDims = 1, .d = {1}}, "Post-EQ Noise Variance");
    CudaTensor<float> post_eq_sinr_db({.nbDims = 1, .d = {1}}, "Post-EQ SINR");

    // Initialize input data - load from binary file (required)
    const std::string engine_base_dir = get_trt_engine_path(logger);
    const std::string xtf_input_path = ran::pusch_test_utils::get_test_vector_path(
            engine_base_dir, filter_type, "xtf_input.bin");
    if (!ran::pusch_test_utils::load_xtf_input(xtf, xtf_input_path)) {
        state.SkipWithError("Failed to load XTF input");
        return;
    }

    // Copy input data to device
    xtf.copy_to_device(stream.get());
    llr.copy_to_device(stream.get());
    post_eq_noise_var_db.copy_to_device(stream.get());
    post_eq_sinr_db.copy_to_device(stream.get());

    // Bind tensors
    TensorBinder binder;
    if (!binder.bind("arg0", xtf, "xtf input buffer")
                 .bind("result0", post_eq_noise_var_db, "noise variance output buffer")
                 .bind("result1", post_eq_sinr_db, "SINR output buffer")
                 .bind("result2", llr, "LLR output buffer")
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

void bm_pusch_inner_receiver_ai_tukey_filter_stream(benchmark::State &state) {
    // Skip if ai_tukey_filter engine not found (optional filter)
    if (!engine_exists("pusch_inner_receiver_ai_tukey_filter.trtengine")) {
        state.SkipWithMessage(
                "WARNING: ai_tukey_filter engine not found - skipping optional benchmark");
        return;
    }
    bm_pusch_inner_receiver_impl(state, "ai_tukey_filter", ExecutionMode::Stream);
}

void bm_pusch_inner_receiver_ai_tukey_filter_graph(benchmark::State &state) {
    // Skip if ai_tukey_filter engine not found (optional filter)
    if (!engine_exists("pusch_inner_receiver_ai_tukey_filter.trtengine")) {
        state.SkipWithMessage(
                "WARNING: ai_tukey_filter engine not found - skipping optional benchmark");
        return;
    }
    bm_pusch_inner_receiver_impl(state, "ai_tukey_filter", ExecutionMode::Graph);
}

void bm_pusch_inner_receiver_free_energy_filter_stream(benchmark::State &state) {
    bm_pusch_inner_receiver_impl(state, "free_energy_filter", ExecutionMode::Stream);
}

void bm_pusch_inner_receiver_free_energy_filter_graph(benchmark::State &state) {
    bm_pusch_inner_receiver_impl(state, "free_energy_filter", ExecutionMode::Graph);
}

void bm_pusch_inner_receiver_weighted_threshold_filter_stream(benchmark::State &state) {
    bm_pusch_inner_receiver_impl(state, "weighted_threshold_filter", ExecutionMode::Stream);
}

void bm_pusch_inner_receiver_weighted_threshold_filter_graph(benchmark::State &state) {
    bm_pusch_inner_receiver_impl(state, "weighted_threshold_filter", ExecutionMode::Graph);
}

BENCHMARK(bm_pusch_inner_receiver_ai_tukey_filter_stream)
        ->Unit(benchmark::kMicrosecond)
        ->MinWarmUpTime(2);

BENCHMARK(bm_pusch_inner_receiver_ai_tukey_filter_graph)
        ->Unit(benchmark::kMicrosecond)
        ->MinWarmUpTime(2);

BENCHMARK(bm_pusch_inner_receiver_free_energy_filter_stream)
        ->Unit(benchmark::kMicrosecond)
        ->MinWarmUpTime(2);

BENCHMARK(bm_pusch_inner_receiver_free_energy_filter_graph)
        ->Unit(benchmark::kMicrosecond)
        ->MinWarmUpTime(2);

BENCHMARK(bm_pusch_inner_receiver_weighted_threshold_filter_stream)
        ->Unit(benchmark::kMicrosecond)
        ->MinWarmUpTime(2);

BENCHMARK(bm_pusch_inner_receiver_weighted_threshold_filter_graph)
        ->Unit(benchmark::kMicrosecond)
        ->MinWarmUpTime(2);

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

} // namespace

/// Google Benchmark main function
BENCHMARK_MAIN();
