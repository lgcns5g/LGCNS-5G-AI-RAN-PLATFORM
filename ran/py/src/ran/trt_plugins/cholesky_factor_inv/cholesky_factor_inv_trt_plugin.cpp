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

#include <exception>
#include <format>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <string_view>

#include <NvInfer.h>

#include "cholesky_factor_inv_trt_plugin.hpp"

namespace ran::trt_plugin {

namespace {

/**
 * Populates field collection with Cholesky plugin fields
 *
 * @param[in,out] fields Vector to populate with field definitions
 * @param[in,out] collection Field collection to update
 * @param[in] matrix_size Pointer to matrix_size value (nullptr for creator)
 * @param[in] is_complex Pointer to is_complex value (nullptr for creator)
 */
void populate_cholesky_factor_inv_fields(
        std::vector<nvinfer1::PluginField> &fields,
        nvinfer1::PluginFieldCollection &collection,
        const std::int32_t *matrix_size = nullptr,
        const std::int32_t *is_complex = nullptr) {

    fields.clear();
    fields.reserve(2);

    fields.emplace_back("matrix_size", matrix_size, nvinfer1::PluginFieldType::kINT32, 1);

    fields.emplace_back("is_complex", is_complex, nvinfer1::PluginFieldType::kINT32, 1);

    collection.nbFields = static_cast<std::int32_t>(fields.size());
    collection.fields = fields.data();
}

} // anonymous namespace

// Constructor implementation
CholeskyFactorInvPlugin::CholeskyFactorInvPlugin(
        const std::string_view name,
        const std::string_view name_space,
        const std::int32_t matrix_size,
        const bool is_complex)
        : TrtPluginBase(name, name_space), m_matrix_size_(matrix_size), m_is_complex_(is_complex) {}

nvinfer1::IPluginV3 *CholeskyFactorInvPlugin::clone() noexcept {
    try {
        // Create new plugin instance with identical configuration
        auto plugin = std::make_unique<CholeskyFactorInvPlugin>(
                m_plugin_name_, m_namespace_, m_matrix_size_, m_is_complex_);
        return plugin.release();
    } catch (const std::exception &e) {
        std::cerr << std::format("Error cloning CholeskyFactorInvPlugin: {}\n", e.what());
        return nullptr;
    }
}

// IPluginV3OneBuild interface implementation
std::int32_t CholeskyFactorInvPlugin::getOutputShapes(
        nvinfer1::DimsExprs const *inputs,
        const std::int32_t nb_inputs,
        [[maybe_unused]] nvinfer1::DimsExprs const *shape_inputs,
        [[maybe_unused]] const std::int32_t nb_shape_inputs,
        nvinfer1::DimsExprs *outputs,
        const std::int32_t nb_outputs,
        [[maybe_unused]] nvinfer1::IExprBuilder &expr_builder) noexcept {

    // Validate inputs and outputs counts before creating spans
    const std::int32_t expected_count = m_is_complex_ ? 2 : 1;
    if (nb_inputs < expected_count) {
        std::cerr << std::format(
                "CholeskyFactorInvPlugin::getOutputShapes error: Expected {} input(s) for "
                "is_complex={}, got {}\n",
                expected_count,
                m_is_complex_,
                nb_inputs);
        return -1;
    }
    if (nb_outputs < expected_count) {
        std::cerr << std::format(
                "CholeskyFactorInvPlugin::getOutputShapes error: Expected {} output(s) for "
                "is_complex={}, got {}\n",
                expected_count,
                m_is_complex_,
                nb_outputs);
        return -1;
    }

    // Use span to avoid pointer arithmetic
    const std::span<const nvinfer1::DimsExprs> inputs_span(inputs, m_is_complex_ ? 2 : 1);
    const std::span<nvinfer1::DimsExprs> outputs_span(outputs, m_is_complex_ ? 2 : 1);

    // Output shape(s) match input shape(s)
    // Input can be:
    //   - 2D: [n_ant, n_ant] (single matrix)
    //   - 3D: [n_prb, n_ant, n_ant] (batched)
    //   - 4D: [batch_size, n_prb, n_ant, n_ant] (batched with outer batch dim)
    // Output has same shape as input
    outputs_span[0].nbDims = inputs_span[0].nbDims;
    const auto dim_count = inputs_span[0].nbDims;
    for (int i = 0; i < dim_count; ++i) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
        outputs_span[0].d[i] = inputs_span[0].d[i];
    }

    // For complex data, second output (imaginary) has same shape as first
    if (m_is_complex_ && nb_outputs >= 2) {
        outputs_span[1].nbDims = inputs_span[0].nbDims;
        for (int i = 0; i < dim_count; ++i) {
            // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
            outputs_span[1].d[i] = inputs_span[0].d[i]; // cppcheck-suppress unreadVariable
        }
    }
    return 0;
}

bool CholeskyFactorInvPlugin::supportsFormatCombination(
        const std::int32_t pos,
        nvinfer1::DynamicPluginTensorDesc const *in_out,
        [[maybe_unused]] const std::int32_t nb_inputs,
        [[maybe_unused]] const std::int32_t nb_outputs) noexcept {

    // Support only FLOAT data type with linear format - use span to avoid pointer arithmetic
    const std::span<const nvinfer1::DynamicPluginTensorDesc> in_out_span(in_out, pos + 1);
    return (in_out_span[pos].desc.type == nvinfer1::DataType::kFLOAT &&
            in_out_span[pos].desc.format == nvinfer1::PluginFormat::kLINEAR);
}

std::int32_t CholeskyFactorInvPlugin::getNbOutputs() const noexcept {
    return m_is_complex_ ? 2 : 1; // Complex: 2 outputs (real, imag), Real: 1 output
}

std::int32_t CholeskyFactorInvPlugin::getOutputDataTypes(
        nvinfer1::DataType *output_types,
        const std::int32_t nb_outputs,
        nvinfer1::DataType const *input_types,
        const std::int32_t nb_inputs) const noexcept {
    // Validate inputs and outputs counts before creating spans
    const std::int32_t expected_count = m_is_complex_ ? 2 : 1;
    if (nb_inputs < expected_count) {
        std::cerr << std::format(
                "CholeskyFactorInvPlugin::getOutputDataTypes error: Expected {} input(s) for "
                "is_complex={}, got {}\n",
                expected_count,
                m_is_complex_,
                nb_inputs);
        return -1;
    }
    if (nb_outputs < expected_count) {
        std::cerr << std::format(
                "CholeskyFactorInvPlugin::getOutputDataTypes error: Expected {} output(s) for "
                "is_complex={}, got {}\n",
                expected_count,
                m_is_complex_,
                nb_outputs);
        return -1;
    }

    // Output data type matches input data type (float) - use span to avoid pointer arithmetic
    const std::span<nvinfer1::DataType> output_types_span(output_types, m_is_complex_ ? 2 : 1);
    const std::span<const nvinfer1::DataType> input_types_span(input_types, m_is_complex_ ? 2 : 1);
    output_types_span[0] = input_types_span[0];
    if (m_is_complex_) {
        output_types_span[1] = input_types_span[1]; // Imaginary output matches imaginary input
    }
    return 0;
}

// IPluginV3OneRuntime interface implementation
std::int32_t CholeskyFactorInvPlugin::enqueue(
        nvinfer1::PluginTensorDesc const *input_desc,
        [[maybe_unused]] nvinfer1::PluginTensorDesc const *output_desc,
        void const *const *inputs,
        void *const *outputs,
        void *workspace,
        cudaStream_t stream) noexcept {
    try {
        // Use span to avoid pointer arithmetic
        const std::span<const nvinfer1::PluginTensorDesc> input_desc_span(
                input_desc, m_is_complex_ ? 2 : 1);
        const std::span<void const *const> inputs_span(inputs, m_is_complex_ ? 2 : 1);
        const std::span<void *const> outputs_span(outputs, m_is_complex_ ? 2 : 1);

        // Extract input dimensions
        const auto &input_dims = input_desc_span[0].dims;
        std::int32_t total_matrices = 1;

        // Calculate total number of matrices to process based on input shape
        if (input_dims.nbDims == 4) {
            // 4D input: [batch_size, n_prb, n_ant, n_ant]
            const auto batch_size = static_cast<std::int32_t>(input_dims.d[0]);
            const auto n_prb = static_cast<std::int32_t>(input_dims.d[1]);
            total_matrices = batch_size * n_prb;
        } else if (input_dims.nbDims == 3) {
            // 3D input: [n_prb, n_ant, n_ant]
            total_matrices = static_cast<std::int32_t>(input_dims.d[0]);
        } else if (input_dims.nbDims == 2) {
            // 2D input: [n_ant, n_ant] (single matrix)
            total_matrices = 1;
        }

        // Extract input and output pointers based on data type
        const auto *input_real = static_cast<const float *>(inputs_span[0]);
        const auto *input_imag =
                m_is_complex_ ? static_cast<const float *>(inputs_span[1]) : nullptr;
        auto *output_real = static_cast<float *>(outputs_span[0]);
        auto *output_imag = m_is_complex_ ? static_cast<float *>(outputs_span[1]) : nullptr;

        // Pre-validate inputs to avoid kernel launch overhead for invalid data
        if (total_matrices <= 0 || m_matrix_size_ <= 0) {
            std::cerr << std::format(
                    "CholeskyFactorInvPlugin::enqueue error: Invalid parameters "
                    "(total_matrices={}, matrix_size={})\n",
                    total_matrices,
                    m_matrix_size_);
            return -1;
        }

        // Ensure all required pointers are valid before kernel launch
        if (input_real == nullptr || output_real == nullptr ||
            (m_is_complex_ && (input_imag == nullptr || output_imag == nullptr))) {
            std::cerr << std::format(
                    "CholeskyFactorInvPlugin::enqueue error: Null pointer detected "
                    "(input_real={}, output_real={}, input_imag={}, output_imag={}, "
                    "is_complex={})\n",
                    static_cast<const void *>(input_real),
                    static_cast<void *>(output_real),
                    static_cast<const void *>(input_imag),
                    static_cast<void *>(output_imag),
                    m_is_complex_);
            return -1;
        }

        // Launch the Cholesky factorization and inversion kernel
        launch_cholesky_factor_inv_kernel(
                input_real,
                input_imag,
                m_matrix_size_,
                total_matrices,
                output_real,
                output_imag,
                workspace,
                stream,
                m_is_complex_);

        return 0;
    } catch (const std::exception &e) {
        std::cerr << std::format("Error in CholeskyFactorInvPlugin::enqueue: {}\n", e.what());
        return -1;
    }
}

nvinfer1::PluginFieldCollection const *CholeskyFactorInvPlugin::getFieldsToSerialize() noexcept {
    // Store m_is_complex_ as std::int32_t for serialization
    static std::int32_t is_complex_int{};
    is_complex_int = static_cast<std::int32_t>(m_is_complex_);

    populate_cholesky_factor_inv_fields(
            m_serialization_fields_, m_serialization_collection_, &m_matrix_size_, &is_complex_int);

    return &m_serialization_collection_;
}

// Plugin Creator implementation
CholeskyFactorInvPluginCreator::CholeskyFactorInvPluginCreator(const std::string_view name_space)
        : TrtPluginCreatorBase(name_space) {
    populate_cholesky_factor_inv_fields(m_plugin_fields_, m_field_collection_);
}

nvinfer1::IPluginV3 *CholeskyFactorInvPluginCreator::createPlugin(
        nvinfer1::AsciiChar const *name,
        nvinfer1::PluginFieldCollection const *fc,
        [[maybe_unused]] nvinfer1::TensorRTPhase phase) noexcept {
    try {
        // Extract plugin configuration from field collection
        const auto matrix_size = get_plugin_field<std::int32_t>(
                fc,
                "matrix_size",
                nvinfer1::PluginFieldType::kINT32,
                CholeskyFactorInvPlugin::DEFAULT_MATRIX_SIZE);

        const auto is_complex_int = get_plugin_field<std::int32_t>(
                fc, "is_complex", nvinfer1::PluginFieldType::kINT32, 0);
        const bool is_complex = static_cast<bool>(is_complex_int);

        auto plugin = std::make_unique<CholeskyFactorInvPlugin>(
                name, m_namespace_, matrix_size, is_complex);
        return plugin.release();
    } catch (const std::exception &e) {
        std::cerr << std::format("Error creating CholeskyFactorInvPlugin: {}\n", e.what());
        return nullptr;
    }
}

} // namespace ran::trt_plugin
