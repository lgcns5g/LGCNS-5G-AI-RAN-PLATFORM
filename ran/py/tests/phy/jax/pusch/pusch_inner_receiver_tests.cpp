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
 * @file pusch_inner_receiver_tests.cpp
 * @brief C++ Google Test suite for PUSCH Inner Receiver TensorRT pipeline
 *
 * This test suite validates the PUSCH Inner Receiver TensorRT pipeline
 * that includes channel estimation, equalization, and soft demapping.
 */

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <NvInfer.h>
#include <driver_types.h>

#include <gtest/gtest.h>

#include <cuda_fp16.h>

#include "pusch_inner_receiver_test_utils.hpp"
#include "trt_test_utils.hpp"

namespace {

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

using namespace ran::trt_utils;
using ran::pusch_test_utils::PuschTestParams;

/// Print debug information about LLR values and test results
void print_debug_info(
        const CudaTensor<__half> &llr,
        const CudaTensor<float> &post_eq_noise_var_db,
        const CudaTensor<float> &post_eq_sinr_db,
        const PuschTestParams &params) {
    // Debug: Check raw bytes of first few LLR values
    std::cout << "\nDebug: First 10 LLR values (raw __half):\n  ";
    for (std::size_t i = 0; i < std::min(static_cast<std::size_t>(10), llr.size()); ++i) {
        const __half raw_val = llr.host()[i];
        const auto raw_bits = std::bit_cast<uint16_t>(raw_val);
        std::cout << std::format("0x{:04x} ", raw_bits);
    }
    std::cout << '\n';

    // Print results
    std::cout << std::format(
            "\nPUSCH Inner Receiver Test Results:\n"
            "  N_SC: {}, N_SYM: {}, QAM_BITS: {}, NUM_RXANT: {}\n",
            params.n_sc,
            params.n_sym,
            params.qam_bits,
            params.num_rxant);

    // Print post-EQ metrics
    std::cout << std::format(
            "  Post-EQ Noise Variance: {:.3f} dB\n"
            "  Post-EQ SINR: {:.3f} dB\n",
            post_eq_noise_var_db[0],
            post_eq_sinr_db[0]);

    // Print sample LLR values (first 10)
    std::cout << "\nSample LLR values (first 10 elements):\n  ";
    for (std::size_t i = 0; i < std::min(static_cast<std::size_t>(10), llr.size()); ++i) {
        const auto llr_val = static_cast<float>(llr.host()[i]);
        std::cout << std::format("{:.3f} ", llr_val);
    }
    std::cout << '\n';
}

/// Validate outputs against reference data
void validate_outputs(
        const CudaTensor<__half> &llr,
        const CudaTensor<float> &post_eq_noise_var_db,
        const CudaTensor<float> &post_eq_sinr_db,
        const std::string &engine_base_dir,
        std::string_view filter_type) {
    // Basic validation: Check that outputs are computed (non-zero or reasonable values)
    // Note: If input is all zeros, outputs might be zero, so we check for finite values
    const auto finiteness = ran::pusch_test_utils::check_llr_finiteness(llr);

    if (!finiteness.has_finite_values) {
        std::cout << std::format(
                "LLR validation failed: NaN count: {}, Inf count: {}\n",
                finiteness.nan_count,
                finiteness.inf_count);
    }

    EXPECT_TRUE(finiteness.has_finite_values) << "LLR output should contain finite values";
    EXPECT_TRUE(std::isfinite(post_eq_noise_var_db[0]))
            << "Post-EQ noise variance should be finite";
    EXPECT_TRUE(std::isfinite(post_eq_sinr_db[0])) << "Post-EQ SINR should be finite";

    // Validate post-EQ noise variance against reference
    const std::string noise_ref_path = ran::pusch_test_utils::get_test_vector_path(
            engine_base_dir, filter_type, "post_eq_noise_var_db_output.bin");
    const auto noise_ref_opt = ran::pusch_test_utils::load_scalar_reference(noise_ref_path);
    ASSERT_TRUE(noise_ref_opt.has_value())
            << "Failed to load noise variance reference from " << noise_ref_path;

    static constexpr float NOISE_TOLERANCE_DB = 2.0F;
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    EXPECT_NEAR(post_eq_noise_var_db[0], *noise_ref_opt, NOISE_TOLERANCE_DB)
            << "Post-EQ noise variance should match reference within tolerance";

    // Validate post-EQ SINR against reference
    const std::string sinr_ref_path = ran::pusch_test_utils::get_test_vector_path(
            engine_base_dir, filter_type, "post_eq_sinr_db_output.bin");
    const auto sinr_ref_opt = ran::pusch_test_utils::load_scalar_reference(sinr_ref_path);
    ASSERT_TRUE(sinr_ref_opt.has_value()) << "Failed to load SINR reference from " << sinr_ref_path;

    static constexpr float SINR_TOLERANCE_DB = 2.0F;
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    EXPECT_NEAR(post_eq_sinr_db[0], *sinr_ref_opt, SINR_TOLERANCE_DB)
            << "Post-EQ SINR should match reference within tolerance";

    // Validate against reference outputs
    // Note: Reference data is stored as float32, but our engine outputs float16
    const std::string llr_ref_path = ran::pusch_test_utils::get_test_vector_path(
            engine_base_dir, filter_type, "llr_output.bin");
    const auto llr_ref_opt = ran::pusch_test_utils::load_llr_reference(llr_ref_path, llr.size());
    ASSERT_TRUE(llr_ref_opt.has_value()) << "Failed to load LLR reference from " << llr_ref_path;

    static constexpr float REL_TOLERANCE = 1e-1F; // 10% relative error
    static constexpr float ABS_TOLERANCE = 1e-2F; // For small values near zero

    const auto result = ran::pusch_test_utils::validate_llr_output(

            llr,
            *llr_ref_opt, // NOLINT(bugprone-unchecked-optional-access)
            REL_TOLERANCE,
            ABS_TOLERANCE);

    std::cout << std::format(
            "\nValidation against reference:\n"
            "  Max relative error: {:.4f}%\n"
            "  Mismatches: {} / {} ({:.2f}%)\n",
            result.max_rel_error * 100.0F,
            result.mismatch_count,
            result.total_count,
            (100.0F * static_cast<float>(result.mismatch_count)) /
                    static_cast<float>(result.total_count));

    EXPECT_LT(result.max_rel_error, REL_TOLERANCE)
            << "LLR output should match reference within tolerance";
}

/// Test the PUSCH Inner Receiver TensorRT pipeline with specified execution mode
void test_pusch_inner_receiver_engine(
        const std::string_view filter_type, const ExecutionMode mode) {
    StdioLogger logger(nvinfer1::ILogger::Severity::kVERBOSE);

    // Load custom plugins
    ASSERT_TRUE(init_ran_plugins(&logger)) << "Failed to load custom plugins";

    // Load engine
    const std::string engine_name = std::format("pusch_inner_receiver_{}.trtengine", filter_type);
    const TrtEngine engine(engine_name, logger);

    // Print engine information
    engine.print_engine_info(logger);

    // Get execution context
    auto *context = engine.get_context();
    auto *cuda_engine = engine.get_engine();

    // Test parameters (matching the generated engine)
    const PuschTestParams params{};

    // Create CUDA stream for async operations
    const CudaStream stream;

    // Create tensors (input: xtf, outputs: llr, post_eq_noise_var_db, post_eq_sinr_db)
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
    ASSERT_TRUE(ran::pusch_test_utils::load_xtf_input(xtf, xtf_input_path))
            << "Failed to load XTF input from " << xtf_input_path;
    std::cout << "Loaded XTF input from " << xtf_input_path << '\n';

    // Initialize output buffers
    std::fill(llr.host().begin(), llr.host().end(), __half{0.0F});
    std::fill(post_eq_noise_var_db.host().begin(), post_eq_noise_var_db.host().end(), 0.0F);
    std::fill(post_eq_sinr_db.host().begin(), post_eq_sinr_db.host().end(), 0.0F);

    // Copy input data to GPU asynchronously
    xtf.copy_to_device(stream.get());
    llr.copy_to_device(stream.get());
    post_eq_noise_var_db.copy_to_device(stream.get());
    post_eq_sinr_db.copy_to_device(stream.get());

    // Bind tensors with shape and type validation
    TensorBinder binder;
    ASSERT_TRUE(binder.bind("arg0", xtf, "xtf input buffer")
                        .bind("result0", post_eq_noise_var_db, "noise variance output buffer")
                        .bind("result1", post_eq_sinr_db, "SINR output buffer")
                        .bind("result2", llr, "LLR output buffer")
                        .apply(context, cuda_engine, logger))
            << "Failed to bind tensors";

    // Execute
    TrtExecutor executor(mode);
    executor.prepare(context, stream.get());
    executor.execute(stream.get());
    stream.synchronize();

    // Copy outputs back to host asynchronously
    llr.copy_from_device(stream.get());
    post_eq_noise_var_db.copy_from_device(stream.get());
    post_eq_sinr_db.copy_from_device(stream.get());

    // Final synchronization to ensure all copies complete
    stream.synchronize();

    // Print debug information
    std::cout << std::format("\n{} Mode Execution\n", execution_mode_to_string(mode));
    print_debug_info(llr, post_eq_noise_var_db, post_eq_sinr_db, params);

    // Validate outputs against reference data
    validate_outputs(llr, post_eq_noise_var_db, post_eq_sinr_db, engine_base_dir, filter_type);
}

// Test: Verifies PUSCH Inner Receiver TensorRT pipeline with AI Tukey Filter in stream mode
// Note: This test is skipped if the ai_tukey_filter engine is not generated (optional filter)
TEST(PuschInnerReceiverPluginTest, ai_tukey_filter_stream_mode) {
    ASSERT_TRUE(set_cuda_device(0));
    if (!engine_exists("pusch_inner_receiver_ai_tukey_filter.trtengine")) {
        GTEST_SKIP() << "ai_tukey_filter engine not found - skipping optional test";
    }
    EXPECT_NO_THROW(test_pusch_inner_receiver_engine("ai_tukey_filter", ExecutionMode::Stream));
}

// Test: Verifies PUSCH Inner Receiver TensorRT pipeline with AI Tukey Filter in graph mode
// Note: This test is skipped if the ai_tukey_filter engine is not generated (optional filter)
TEST(PuschInnerReceiverPluginTest, ai_tukey_filter_graph_mode) {
    ASSERT_TRUE(set_cuda_device(0));
    if (!engine_exists("pusch_inner_receiver_ai_tukey_filter.trtengine")) {
        GTEST_SKIP() << "ai_tukey_filter engine not found - skipping optional test";
    }
    EXPECT_NO_THROW(test_pusch_inner_receiver_engine("ai_tukey_filter", ExecutionMode::Graph));
}

// Test: Verifies PUSCH Inner Receiver TensorRT pipeline with Free Energy Filter in stream mode
TEST(PuschInnerReceiverPluginTest, free_energy_filter_stream_mode) {
    ASSERT_TRUE(set_cuda_device(0));
    EXPECT_NO_THROW(test_pusch_inner_receiver_engine("free_energy_filter", ExecutionMode::Stream));
}

// Test: Verifies PUSCH Inner Receiver TensorRT pipeline with Free Energy Filter in graph mode
TEST(PuschInnerReceiverPluginTest, free_energy_filter_graph_mode) {
    ASSERT_TRUE(set_cuda_device(0));
    EXPECT_NO_THROW(test_pusch_inner_receiver_engine("free_energy_filter", ExecutionMode::Graph));
}

// Test: Verifies PUSCH Inner Rx TensorRT pipeline with Weighted Threshold Filter in stream mode
TEST(PuschInnerReceiverPluginTest, weighted_threshold_filter_stream_mode) {
    ASSERT_TRUE(set_cuda_device(0));
    EXPECT_NO_THROW(
            test_pusch_inner_receiver_engine("weighted_threshold_filter", ExecutionMode::Stream));
}

// Test: Verifies PUSCH Inner Rx TensorRT pipeline with Weighted Threshold Filter in graph mode
TEST(PuschInnerReceiverPluginTest, weighted_threshold_filter_graph_mode) {
    ASSERT_TRUE(set_cuda_device(0));
    EXPECT_NO_THROW(
            test_pusch_inner_receiver_engine("weighted_threshold_filter", ExecutionMode::Graph));
}

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

} // namespace
