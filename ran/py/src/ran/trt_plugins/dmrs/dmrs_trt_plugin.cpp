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

#include <cuda_runtime_api.h>

#include "dmrs_trt_plugin.hpp"

namespace ran::trt_plugin {

// Default DMRS sequence length and OFDM symbols
static constexpr int32_t DEFAULT_SEQUENCE_LENGTH = 42;
static constexpr int32_t DEFAULT_N_T = 14;

namespace {

/**
 * Populates field collection with DMRS plugin fields
 *
 * @param[in,out] fields Vector to populate with field definitions
 * @param[in,out] collection Field collection to update
 * @param[in] sequence_length Pointer to sequence_length value (nullptr for creator)
 * @param[in] n_t Pointer to n_t value (nullptr for creator)
 */
void populate_dmrs_fields(
        std::vector<nvinfer1::PluginField> &fields,
        nvinfer1::PluginFieldCollection &collection,
        const std::int32_t *sequence_length = nullptr,
        const std::int32_t *n_t = nullptr) {

    fields.clear();
    fields.reserve(2);

    fields.emplace_back("sequence_length", sequence_length, nvinfer1::PluginFieldType::kINT32, 1);
    fields.emplace_back("n_t", n_t, nvinfer1::PluginFieldType::kINT32, 1);

    collection.nbFields = static_cast<std::int32_t>(fields.size());
    collection.fields = fields.data();
}

} // anonymous namespace

DMRSTrtPlugin::DMRSTrtPlugin(
        const std::string_view name,
        const std::string_view name_space,
        const std::int32_t sequence_length,
        const std::int32_t n_t)
        : TrtPluginBase(name, name_space), m_sequence_length_(sequence_length), m_n_t_(n_t) {}

nvinfer1::IPluginV3 *DMRSTrtPlugin::clone() noexcept {
    try {
        // Create new plugin instance with identical configuration
        auto plugin = std::make_unique<DMRSTrtPlugin>(
                m_plugin_name_, m_namespace_, m_sequence_length_, m_n_t_);
        return plugin.release();
    } catch (const std::exception &e) {
        std::cerr << std::format("Error cloning DMRSTrtPlugin: {}\n", e.what());
        return nullptr;
    }
}

// IPluginV3OneBuild interface implementation
int32_t DMRSTrtPlugin::getOutputShapes(
        nvinfer1::DimsExprs const *inputs,
        int32_t nb_inputs,
        [[maybe_unused]] nvinfer1::DimsExprs const *shape_inputs,
        [[maybe_unused]] int32_t nb_shape_inputs,
        nvinfer1::DimsExprs *outputs,
        int32_t nb_outputs,
        nvinfer1::IExprBuilder &expr_builder) noexcept {

    // Validate inputs and outputs counts before creating spans
    if (nb_inputs < 1) {
        std::cerr << std::format(
                "DMRSTrtPlugin::getOutputShapes error: Invalid input count (nb_inputs={})\n",
                nb_inputs);
        return -1;
    }
    if (nb_outputs < 2) {
        std::cerr << std::format(
                "DMRSTrtPlugin::getOutputShapes error: Invalid output count (nb_outputs={}), "
                "expected 2\n",
                nb_outputs);
        return -1;
    }

    // Use span to avoid pointer arithmetic
    const std::span<const nvinfer1::DimsExprs> inputs_span(inputs, 1);
    const std::span<nvinfer1::DimsExprs> outputs_span(outputs, 2);

    // Input shape is [2] for [slot_number, n_dmrs_id]
    // Output 0: Complex DMRS shape (2, n_t, 2, sequence_length/2)
    // - Dimension 0: 2 for [real, imag]
    // - Dimension 1: n_t OFDM symbols
    // - Dimension 2: 2 ports (n_scid=0,1)
    // - Dimension 3: sequence_length/2 complex values
    // Output 1: Binary gold sequence shape (n_t, 2, sequence_length)
    // - Dimension 0: n_t OFDM symbols
    // - Dimension 1: 2 ports (n_scid=0,1)
    // - Dimension 2: sequence_length subcarriers
    if (inputs_span[0].nbDims != 1) {
        std::cerr << std::format(
                "DMRSTrtPlugin::getOutputShapes error: Expected input nbDims=1, got {}\n",
                inputs_span[0].nbDims);
        return -1;
    }

    // Set output 0 shape to [2, n_t, 2, sequence_length/2] (complex DMRS)
    outputs_span[0].nbDims = 4;
    outputs_span[0].d[0] = expr_builder.constant(2); // real/imag
    outputs_span[0].d[1] = expr_builder.constant(m_n_t_);
    outputs_span[0].d[2] = expr_builder.constant(2); // n_scid
    outputs_span[0].d[3] = expr_builder.constant(m_sequence_length_ / 2);

    // Set output 1 shape to [n_t, 2, sequence_length] (binary sequence)
    outputs_span[1].nbDims = 3;
    outputs_span[1].d[0] = expr_builder.constant(m_n_t_);
    outputs_span[1].d[1] = expr_builder.constant(2);
    // cppcheck-suppress unreadVariable
    outputs_span[1].d[2] = expr_builder.constant(m_sequence_length_);

    return 0;
}

bool DMRSTrtPlugin::supportsFormatCombination(
        const int32_t pos,
        nvinfer1::DynamicPluginTensorDesc const *in_out,
        [[maybe_unused]] const int32_t nb_inputs,
        [[maybe_unused]] const int32_t nb_outputs) noexcept {

    // Use span to avoid pointer arithmetic
    const std::span<const nvinfer1::DynamicPluginTensorDesc> in_out_span(in_out, pos + 1);

    // pos 0: input (INT32, linear)
    // pos 1: output 0 - complex DMRS (FLOAT32, linear)
    // pos 2: output 1 - binary sequence (INT32, linear)
    if (pos == 0 || pos == 2) {
        // Input and output 1 are INT32
        return (in_out_span[pos].desc.type == nvinfer1::DataType::kINT32 &&
                in_out_span[pos].desc.format == nvinfer1::PluginFormat::kLINEAR);
    } else if (pos == 1) {
        // Output 0 is FLOAT32
        return (in_out_span[pos].desc.type == nvinfer1::DataType::kFLOAT &&
                in_out_span[pos].desc.format == nvinfer1::PluginFormat::kLINEAR);
    }
    return false;
}

int32_t DMRSTrtPlugin::getNbOutputs() const noexcept { return 2; }

int32_t DMRSTrtPlugin::getOutputDataTypes(
        nvinfer1::DataType *output_types,
        const int32_t nb_outputs,
        [[maybe_unused]] nvinfer1::DataType const *input_types,
        const int32_t nb_inputs) const noexcept {
    // Validate inputs and outputs counts before creating spans
    if (nb_inputs < 1) {
        std::cerr << std::format(
                "DMRSTrtPlugin::getOutputDataTypes error: Invalid input count (nb_inputs={})\n",
                nb_inputs);
        return -1;
    }
    if (nb_outputs < 2) {
        std::cerr << std::format(
                "DMRSTrtPlugin::getOutputDataTypes error: Invalid output count (nb_outputs={}), "
                "expected 2\n",
                nb_outputs);
        return -1;
    }

    // Set data types for two outputs - use span to avoid pointer arithmetic
    const std::span<nvinfer1::DataType> output_types_span(output_types, 2);
    output_types_span[0] = nvinfer1::DataType::kFLOAT; // Complex DMRS (real/imag)
    output_types_span[1] = nvinfer1::DataType::kINT32; // Binary gold sequence
    return 0;
}

// IPluginV3OneRuntime interface implementation
int32_t DMRSTrtPlugin::enqueue(
        nvinfer1::PluginTensorDesc const *input_desc,
        [[maybe_unused]] nvinfer1::PluginTensorDesc const *output_desc,
        void const *const *inputs,
        void *const *outputs,
        [[maybe_unused]] void *workspace,
        cudaStream_t stream) noexcept {
    // Validate input pointers first
    if (input_desc == nullptr) {
        std::cerr << "ERROR: input_desc is nullptr\n";
        return -1;
    }
    if (inputs == nullptr) {
        std::cerr << "ERROR: inputs is nullptr\n";
        return -1;
    }
    if (outputs == nullptr) {
        std::cerr << "ERROR: outputs is nullptr\n";
        return -1;
    }

    // Use span to avoid pointer arithmetic
    const std::span<void const *const> inputs_span(inputs, 1);
    const std::span<void *const> outputs_span(outputs, 2); // 2 outputs now
    const std::span<const nvinfer1::PluginTensorDesc> input_desc_span(input_desc, 1);

    // Extract input and output pointers
    // Input is concatenated [slot_number, n_dmrs_id] (2 scalar values)
    const auto *concatenated_input = static_cast<const int32_t *>(inputs_span[0]);
    auto *r_dmrs_ri_sym_cdm_sc = static_cast<float *>(outputs_span[0]); // Complex DMRS (float32)
    auto *scr_seq_sym_ri_sc = static_cast<int32_t *>(outputs_span[1]);  // Binary sequence (int32)

    // Validate pointers before accessing
    if (concatenated_input == nullptr) {
        std::cerr << "DMRSTrtPlugin::enqueue error: Null input pointer\n";
        return -1;
    }
    if (r_dmrs_ri_sym_cdm_sc == nullptr) {
        std::cerr << "DMRSTrtPlugin::enqueue error: Null r_dmrs_ri_sym_cdm_sc pointer\n";
        return -1;
    }
    if (scr_seq_sym_ri_sc == nullptr) {
        std::cerr << "DMRSTrtPlugin::enqueue error: Null scr_seq_sym_ri_sc pointer\n";
        return -1;
    }

    // Use configured sequence length and n_t (compile-time constants)
    const int32_t sequence_length = m_sequence_length_;
    const int32_t n_t = m_n_t_;

    // Verify input dimensions - should be [2] for [slot_number, n_dmrs_id]
    static constexpr int32_t PARAMS_COUNT = 2;

    // Check nbDims first before accessing d[0]
    if (input_desc_span[0].dims.nbDims != 1) {
        std::cerr << std::format(
                "DMRSTrtPlugin::enqueue error: Expected 1D tensor, got {}D tensor\n",
                input_desc_span[0].dims.nbDims);
        return -1;
    }

    // Now safe to access d[0] since we know nbDims == 1
    if (input_desc_span[0].dims.d[0] != PARAMS_COUNT) {
        std::cerr << std::format(
                "DMRSTrtPlugin::enqueue error: Expected input shape [2], got [{}]\n",
                input_desc_span[0].dims.d[0]);
        return -1;
    }

    // NOTE: concatenated_input is GPU memory - we CANNOT dereference it from CPU!
    // We must pass it to the kernel and let the kernel read the values on the GPU

    try {
        // Clear any stale CUDA errors from previous operations
        const cudaError_t stale_error = cudaGetLastError();
        if (stale_error != cudaSuccess) {
            std::cerr << std::format(
                    "DMRSTrtPlugin::enqueue warning: Cleared stale CUDA error: {}\n",
                    cudaGetErrorString(stale_error));
        }

        // Pre-validate inputs
        if (sequence_length <= 0 || n_t <= 0) {
            std::cerr << std::format(
                    "DMRSTrtPlugin::enqueue error: Invalid parameters (sequence_length={}, "
                    "n_t={})\n",
                    sequence_length,
                    n_t);
            return -1;
        }

        // Launch CUDA kernel for DMRS sequence generation
        // Output 0: Complex DMRS shape (2, n_t, 2, sequence_length/2)
        // Output 1: Binary sequence shape (n_t, 2, sequence_length)
        // The kernel internally loops over n_t symbols and n_scid=[0,1] ports
        // Pass GPU pointer - kernel will read slot_number and n_dmrs_id from GPU memory
        launch_dmrs_kernel(
                concatenated_input, // GPU pointer to [slot_number, n_dmrs_id]
                sequence_length,
                n_t,
                r_dmrs_ri_sym_cdm_sc, // Complex DMRS output (float32)
                scr_seq_sym_ri_sc,    // Binary gold sequence output (int32)
                stream);

        return 0; // Success
    } catch (const std::exception &e) {
        std::cerr << std::format("DMRSTrtPlugin::enqueue error: {}\n", e.what());
        return -1; // Failure
    }
}

nvinfer1::PluginFieldCollection const *DMRSTrtPlugin::getFieldsToSerialize() noexcept {
    populate_dmrs_fields(
            m_serialization_fields_, m_serialization_collection_, &m_sequence_length_, &m_n_t_);
    return &m_serialization_collection_;
}

// Plugin Creator Implementation
DMRSTrtPluginCreator::DMRSTrtPluginCreator(const std::string_view name_space)
        : TrtPluginCreatorBase(name_space) {
    populate_dmrs_fields(m_plugin_fields_, m_field_collection_);
}

nvinfer1::IPluginV3 *DMRSTrtPluginCreator::createPlugin(
        nvinfer1::AsciiChar const *name,
        nvinfer1::PluginFieldCollection const *fc,
        [[maybe_unused]] nvinfer1::TensorRTPhase phase) noexcept {

    try {
        // Extract sequence length and n_t from field collection with defaults
        const auto sequence_length = get_plugin_field<std::int32_t>(
                fc, "sequence_length", nvinfer1::PluginFieldType::kINT32, DEFAULT_SEQUENCE_LENGTH);
        const auto n_t = get_plugin_field<std::int32_t>(
                fc, "n_t", nvinfer1::PluginFieldType::kINT32, DEFAULT_N_T);

        // Create plugin instance with configured parameters
        auto plugin = std::make_unique<DMRSTrtPlugin>(name, m_namespace_, sequence_length, n_t);
        return plugin.release();
    } catch (const std::exception &e) {
        std::cerr << std::format("Error creating DMRSTrtPlugin: {}\n", e.what());
        return nullptr;
    }
}

} // namespace ran::trt_plugin
