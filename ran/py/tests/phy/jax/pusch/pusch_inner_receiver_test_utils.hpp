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
 * @file pusch_inner_receiver_test_utils.hpp
 * @brief Common utilities for PUSCH Inner Receiver tests and benchmarks
 */

#ifndef RAN_PY_TESTS_PHY_JAX_PUSCH_PUSCH_INNER_RECEIVER_PUSCH_INNER_RECEIVER_TEST_UTILS_HPP
#define RAN_PY_TESTS_PHY_JAX_PUSCH_PUSCH_INNER_RECEIVER_PUSCH_INNER_RECEIVER_TEST_UTILS_HPP

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <cuda_fp16.h>

#include "trt_test_utils.hpp"

namespace ran::pusch_test_utils {

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

/// Test configuration parameters for PUSCH Inner Receiver
struct PuschTestParams final {
    std::int32_t n_sc{3276};     //!< Number of subcarriers
    std::int32_t n_sym{14};      //!< Number of OFDM symbols
    std::int32_t qam_bits{8};    //!< QAM modulation bits (256-QAM)
    std::int32_t num_rxant{4};   //!< Number of receive antennas
    std::int32_t n_datasym{13};  //!< Number of data symbols
    std::int32_t n_layer{1};     //!< Number of layers
    std::int32_t n_real_imag{2}; //!< Real and imaginary components
};

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

/// Validation result for LLR output comparison
struct ValidationResult final {
    float max_rel_error{};        //!< Maximum relative error
    std::size_t mismatch_count{}; //!< Number of mismatched values
    std::size_t total_count{};    //!< Total number of values compared
};

/// Finiteness check result for LLR output
struct FinitenessResult final {
    bool has_finite_values{};  //!< True if any finite values found
    std::size_t nan_count{};   //!< Number of NaN values found
    std::size_t inf_count{};   //!< Number of Inf values found
    std::size_t total_count{}; //!< Total number of values checked
};

/**
 * Build path to test vector file
 *
 * @param[in] engine_base_dir Base directory for TensorRT engines
 * @param[in] filter_type Filter type (e.g., "ai_tukey_filter")
 * @param[in] filename File name (e.g., "xtf_input.bin")
 * @return Full path to test vector file
 */
[[nodiscard]] std::string get_test_vector_path(
        const std::string &engine_base_dir,
        std::string_view filter_type,
        std::string_view filename);

/**
 * Load XTF input from binary file
 *
 * @param[in,out] xtf XTF tensor to populate
 * @param[in] file_path Path to binary input file
 * @return true if file was loaded successfully, false otherwise
 */
[[nodiscard]] bool
load_xtf_input(ran::trt_utils::CudaTensor<__half> &xtf, const std::string &file_path);

/**
 * Load LLR reference data from binary file
 *
 * @param[in] file_path Path to binary reference file
 * @param[in] expected_size Expected number of float values
 * @return Vector of reference values if successful, std::nullopt otherwise
 */
[[nodiscard]] std::optional<std::vector<float>>
load_llr_reference(const std::string &file_path, std::size_t expected_size);

/**
 * Load scalar reference value from binary file
 *
 * @param[in] file_path Path to binary reference file
 * @return Scalar reference value if successful, std::nullopt otherwise
 */
[[nodiscard]] std::optional<float> load_scalar_reference(const std::string &file_path);

/**
 * Validate LLR output against reference
 *
 * @param[in] llr LLR output tensor (float16)
 * @param[in] llr_ref Reference LLR values (float32)
 * @param[in] rel_tolerance Relative error tolerance
 * @param[in] abs_tolerance Absolute error tolerance for small values
 * @return Validation statistics
 */
[[nodiscard]] ValidationResult validate_llr_output(
        const ran::trt_utils::CudaTensor<__half> &llr,
        const std::vector<float> &llr_ref,
        float rel_tolerance,
        float abs_tolerance);

/**
 * Check LLR output for NaN and Inf values
 *
 * @param[in] llr LLR output tensor (float16)
 * @return Finiteness check statistics
 */
[[nodiscard]] FinitenessResult check_llr_finiteness(const ran::trt_utils::CudaTensor<__half> &llr);

} // namespace ran::pusch_test_utils
#endif // RAN_PY_TESTS_PHY_JAX_PUSCH_PUSCH_INNER_RECEIVER_PUSCH_INNER_RECEIVER_TEST_UTILS_HPP
