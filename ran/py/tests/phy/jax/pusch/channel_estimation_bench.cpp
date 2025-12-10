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
 * @file channel_estimation_bench.cpp
 * @brief Google Benchmark for Channel Estimation TensorRT pipeline performance
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <format>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <NvInfer.h>
#include <benchmark/benchmark.h>
#include <driver_types.h>

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

/// Common benchmark implementation for Channel Estimation TensorRT pipeline
void bm_channel_estimation_impl(
        benchmark::State &state, const std::string_view filter_type, const ExecutionMode mode) {
    // Test parameters matching channel_estimator outputs
    static constexpr std::int32_t N_REAL_IMAG = 2;
    static constexpr std::int32_t N_SYM = 14;
    static constexpr std::int32_t NUM_RXANT = 4;
    static constexpr std::int32_t N_SC = 3276;
    static constexpr std::int32_t N_PORT = 1;
    static constexpr std::int32_t N_PRB = 273;
    static constexpr std::int32_t N_DMRS_SC = 1638; // 273 * 6 for Type 1 DMRS
    static constexpr std::int32_t N_DMRS_SYMS = 1;
    static constexpr std::int32_t N_UE = 1;

    StdioLogger logger(nvinfer1::ILogger::Severity::kERROR);

    // Load custom plugins
    if (!init_ran_plugins(&logger)) {
        state.SkipWithError("Failed to load custom plugins");
        return;
    }

    // Get engine directory from environment
    const std::string engine_base_dir = get_trt_engine_path(logger);

    // Load engine (follows same pattern as pusch_inner_receiver_bench)
    const std::string engine_name = std::format("channel_estimator_{}.trtengine", filter_type);
    const TrtEngine engine(engine_name, logger);

    // Get execution context
    auto *context = engine.get_context();
    auto *cuda_engine = engine.get_engine();

    // Create CUDA stream
    const CudaStream stream;

    // Create input tensor (float16 as required by TensorRT engine)
    CudaTensor<__half> arg0({.nbDims = 4, .d = {N_REAL_IMAG, N_SYM, NUM_RXANT, N_SC}}, "Input");

    // Create output tensors (match engine output order from channel_estimator function)
    // result0: h_est - channel estimates per DMRS symbol (2, 1, 1, 4, 1638)
    CudaTensor<__half> result0(
            {.nbDims = 5, .d = {N_REAL_IMAG, N_PORT, N_DMRS_SYMS, NUM_RXANT, N_DMRS_SC}},
            "Output 0");

    // result1: h_interp - interpolated channel estimates (2, 1, 4, 3276)
    CudaTensor<__half> result1(
            {.nbDims = 4, .d = {N_REAL_IMAG, N_PORT, NUM_RXANT, N_SC}}, "Output 1");

    // result2: n_cov - noise covariance matrix (2, 4, 4, 273)
    CudaTensor<__half> result2(
            {.nbDims = 4, .d = {N_REAL_IMAG, NUM_RXANT, NUM_RXANT, N_PRB}}, "Output 2");

    // result3: noise_var_db - scalar metric (float16, not float32!)
    CudaTensor<__half> result3({.nbDims = 1, .d = {1}}, "Output 3");

    // result4: rsrp_db - small array metric (float16)
    CudaTensor<__half> result4({.nbDims = 2, .d = {N_UE, N_DMRS_SYMS}}, "Output 4");

    // result5: sinr_db - small array metric (float16)
    CudaTensor<__half> result5({.nbDims = 2, .d = {N_UE, N_DMRS_SYMS}}, "Output 5");

    // Initialize input data - try to load from binary file if available (in test_vectors
    // subdirectory)
    const std::string xtf_input_path = engine_base_dir + "/test_vectors/pusch_channel_estimation/" +
                                       std::string{filter_type} + "/xtf_input.bin";
    std::ifstream xtf_file(xtf_input_path, std::ios::binary);

    if (xtf_file.is_open()) {
        xtf_file.read(
                // cppcheck-suppress invalidPointerCast
                reinterpret_cast<char *>( // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
                        arg0.host().data()),
                static_cast<std::streamsize>(arg0.size() * sizeof(__half)));
        if (!xtf_file) {
            std::fill(arg0.host().begin(), arg0.host().end(), __half{1.0F});
        }
        xtf_file.close();
    } else {
        // Initialize with random-like data
        std::fill(arg0.host().begin(), arg0.host().end(), __half{0.1F});
    }

    // Copy input data to device
    arg0.copy_to_device(stream.get());

    // Copy output buffers to device (allocates device memory before benchmarking)
    result0.copy_to_device(stream.get());
    result1.copy_to_device(stream.get());
    result2.copy_to_device(stream.get());
    result3.copy_to_device(stream.get());
    result4.copy_to_device(stream.get());
    result5.copy_to_device(stream.get());

    // Bind tensors
    TensorBinder binder;
    if (!binder.bind("arg0", arg0, "input buffer")
                 .bind("result0", result0, "output 0")
                 .bind("result1", result1, "output 1")
                 .bind("result2", result2, "output 2")
                 .bind("result3", result3, "output 3")
                 .bind("result4", result4, "output 4")
                 .bind("result5", result5, "output 5")
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

void bm_channel_estimation_ai_tukey_stream(benchmark::State &state) {
    bm_channel_estimation_impl(state, "ai_tukey_filter", ExecutionMode::Stream);
}

void bm_channel_estimation_ai_tukey_graph(benchmark::State &state) {
    bm_channel_estimation_impl(state, "ai_tukey_filter", ExecutionMode::Graph);
}

void bm_channel_estimation_free_energy_stream(benchmark::State &state) {
    bm_channel_estimation_impl(state, "free_energy_filter", ExecutionMode::Stream);
}

void bm_channel_estimation_free_energy_graph(benchmark::State &state) {
    bm_channel_estimation_impl(state, "free_energy_filter", ExecutionMode::Graph);
}

void bm_channel_estimation_weighted_threshold_stream(benchmark::State &state) {
    bm_channel_estimation_impl(state, "weighted_threshold_filter", ExecutionMode::Stream);
}

void bm_channel_estimation_weighted_threshold_graph(benchmark::State &state) {
    bm_channel_estimation_impl(state, "weighted_threshold_filter", ExecutionMode::Graph);
}

void bm_channel_estimation_identity_stream(benchmark::State &state) {
    bm_channel_estimation_impl(state, "identity_filter", ExecutionMode::Stream);
}

void bm_channel_estimation_identity_graph(benchmark::State &state) {
    bm_channel_estimation_impl(state, "identity_filter", ExecutionMode::Graph);
}

BENCHMARK(bm_channel_estimation_ai_tukey_stream)->Unit(benchmark::kMicrosecond)->MinWarmUpTime(2);

BENCHMARK(bm_channel_estimation_ai_tukey_graph)->Unit(benchmark::kMicrosecond)->MinWarmUpTime(2);

BENCHMARK(bm_channel_estimation_free_energy_stream)
        ->Unit(benchmark::kMicrosecond)
        ->MinWarmUpTime(2);

BENCHMARK(bm_channel_estimation_free_energy_graph)->Unit(benchmark::kMicrosecond)->MinWarmUpTime(2);

BENCHMARK(bm_channel_estimation_weighted_threshold_stream)
        ->Unit(benchmark::kMicrosecond)
        ->MinWarmUpTime(2);

BENCHMARK(bm_channel_estimation_weighted_threshold_graph)
        ->Unit(benchmark::kMicrosecond)
        ->MinWarmUpTime(2);

BENCHMARK(bm_channel_estimation_identity_stream)->Unit(benchmark::kMicrosecond)->MinWarmUpTime(2);

BENCHMARK(bm_channel_estimation_identity_graph)->Unit(benchmark::kMicrosecond)->MinWarmUpTime(2);

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

} // namespace

/// Google Benchmark main function
BENCHMARK_MAIN();
