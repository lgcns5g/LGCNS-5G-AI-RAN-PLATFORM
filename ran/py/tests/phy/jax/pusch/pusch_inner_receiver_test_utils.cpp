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
 * @file pusch_inner_receiver_test_utils.cpp
 * @brief Implementation of common utilities for PUSCH Inner Receiver tests
 */

#include <algorithm>
#include <cmath>
#include <format>
#include <fstream>
#include <iostream>

#include "pusch_inner_receiver_test_utils.hpp"

namespace ran::pusch_test_utils {

std::string get_test_vector_path(
        const std::string &engine_base_dir,
        const std::string_view filter_type,
        const std::string_view filename) {
    return std::format(
            "{}/test_vectors/pusch_inner_receiver/{}/{}", engine_base_dir, filter_type, filename);
}

bool load_xtf_input(ran::trt_utils::CudaTensor<__half> &xtf, const std::string &file_path) {
    std::ifstream file(file_path, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Error: Failed to open " << file_path << '\n';
        return false;
    }

    file.read(
            // cppcheck-suppress invalidPointerCast
            reinterpret_cast<char *>( // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
                    xtf.host().data()),
            static_cast<std::streamsize>(xtf.size() * sizeof(__half)));

    if (!file) {
        std::cerr << "Error: Failed to read " << file_path << '\n';
        file.close();
        return false;
    }

    file.close();
    return true;
}

std::optional<std::vector<float>>
load_llr_reference(const std::string &file_path, const std::size_t expected_size) {
    std::ifstream file(file_path, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Error: Failed to open LLR reference file: " << file_path << '\n';
        return std::nullopt;
    }

    // Get file size
    file.seekg(0, std::ios::end);
    const auto file_size = static_cast<std::streamsize>(file.tellg());
    file.seekg(0, std::ios::beg);

    // LLR reference is stored as float16 (__half) in the file
    const auto expected_bytes_fp16 = static_cast<std::streamsize>(expected_size * sizeof(__half));
    if (file_size < expected_bytes_fp16) {
        std::cerr << std::format(
                "Error: LLR reference file size mismatch.\n"
                "  File: {}\n"
                "  Expected size: {} bytes ({} float16 values)\n"
                "  Actual size: {} bytes\n",
                file_path,
                expected_bytes_fp16,
                expected_size,
                file_size);
        return std::nullopt;
    }

    // Read float16 data
    std::vector<__half> llr_ref_fp16(expected_size);
    file.read(
            // cppcheck-suppress invalidPointerCast
            reinterpret_cast<char *>( // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
                    llr_ref_fp16.data()),
            expected_bytes_fp16);

    if (!file) {
        std::cerr << "Error: Failed to read LLR reference from " << file_path << '\n';
        return std::nullopt;
    }

    // Convert float16 to float32
    std::vector<float> llr_ref(expected_size);
    for (std::size_t i = 0; i < expected_size; ++i) {
        llr_ref[i] = static_cast<float>(llr_ref_fp16[i]);
    }

    std::cout << std::format(
            "Loaded LLR reference from {} ({} values)\n", file_path, expected_size);
    return llr_ref;
}

std::optional<float> load_scalar_reference(const std::string &file_path) {
    std::ifstream file(file_path, std::ios::binary);

    if (!file.is_open()) {
        return std::nullopt;
    }

    float value{};
    file.read(
            // cppcheck-suppress invalidPointerCast
            reinterpret_cast<char *>( // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
                    &value),
            static_cast<std::streamsize>(sizeof(float)));

    if (!file) {
        return std::nullopt;
    }

    return value;
}

ValidationResult validate_llr_output(
        const ran::trt_utils::CudaTensor<__half> &llr,
        const std::vector<float> &llr_ref,
        const float rel_tolerance,
        const float abs_tolerance) {
    ValidationResult result{};
    result.total_count = llr.size();

    for (std::size_t i = 0; i < llr.size(); ++i) {
        const auto llr_float = static_cast<float>(llr[i]); // Convert fp16 to fp32 for comparison
        const float abs_ref = std::abs(llr_ref[i]);
        const float abs_diff = std::abs(llr_float - llr_ref[i]);

        if (abs_ref > abs_tolerance) {
            const float rel_error = abs_diff / abs_ref;
            result.max_rel_error = std::max(result.max_rel_error, rel_error);

            if (rel_error > rel_tolerance) {
                ++result.mismatch_count;
            }
        } else if (abs_diff > abs_tolerance) {
            ++result.mismatch_count;
        }
    }

    return result;
}

FinitenessResult check_llr_finiteness(const ran::trt_utils::CudaTensor<__half> &llr) {
    FinitenessResult result{};
    result.total_count = llr.size();

    for (std::size_t i = 0; i < llr.size(); ++i) {
        const auto llr_val = static_cast<float>(llr[i]);
        if (std::isfinite(llr_val)) {
            result.has_finite_values = true;
        } else if (std::isnan(llr_val)) {
            ++result.nan_count;
        } else {
            ++result.inf_count;
        }
    }

    return result;
}

} // namespace ran::pusch_test_utils
