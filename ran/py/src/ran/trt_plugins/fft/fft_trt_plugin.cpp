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
#include <string_view>

#include <NvInfer.h>

#include "fft_trt_plugin.hpp"

namespace ran::trt_plugin {

namespace {

/**
 * Populates field collection with FFT plugin fields
 *
 * @param[in,out] fields Vector to populate with field definitions
 * @param[in,out] collection Field collection to update
 * @param[in] fft_size Pointer to fft_size value (nullptr for creator)
 * @param[in] direction Pointer to direction value (nullptr for creator)
 */
void populate_fft_fields(
        std::vector<nvinfer1::PluginField> &fields,
        nvinfer1::PluginFieldCollection &collection,
        const std::int32_t *fft_size = nullptr,
        const std::int32_t *direction = nullptr) {

    fields.clear();
    fields.reserve(2);

    fields.emplace_back("fft_size", fft_size, nvinfer1::PluginFieldType::kINT32, 1);

    fields.emplace_back("direction", direction, nvinfer1::PluginFieldType::kINT32, 1);

    collection.nbFields = static_cast<std::int32_t>(fields.size());
    collection.fields = fields.data();
}

} // anonymous namespace

// Constructor implementation
FftTrtPlugin::FftTrtPlugin(
        const std::string_view name,
        const std::string_view name_space,
        const FftTrtPluginParams &params)
        : TrtPluginBase(name, name_space), m_fft_size_(params.fft_size),
          m_precision_(params.precision), m_fft_type_(params.fft_type),
          m_ffts_per_block_(params.ffts_per_block),
          m_elements_per_thread_(params.elements_per_thread) {
    // Convert string direction to integer
    if (params.direction == "inverse") {
        m_direction_ = 1;
    } else {
        m_direction_ = 0; // Default to forward for any other value
    }
}

nvinfer1::IPluginV3 *FftTrtPlugin::clone() noexcept {
    try {
        // Create new plugin instance with identical configuration
        const std::string direction = (m_direction_ == 0) ? "forward" : "inverse";
        auto plugin = std::make_unique<FftTrtPlugin>(
                m_plugin_name_,
                m_namespace_,
                FftTrtPluginParams{
                        .fft_size = m_fft_size_,
                        .precision = m_precision_,
                        .fft_type = m_fft_type_,
                        .direction = direction,
                        .ffts_per_block = m_ffts_per_block_,
                        .elements_per_thread = m_elements_per_thread_});
        return plugin.release();
    } catch (const std::exception &e) {
        std::cerr << std::format("Error cloning FftTrtPlugin: {}\n", e.what());
        return nullptr;
    }
}

// IPluginV3OneBuild interface implementation
int32_t FftTrtPlugin::getOutputShapes(
        nvinfer1::DimsExprs const *inputs,
        int32_t nb_inputs,
        [[maybe_unused]] nvinfer1::DimsExprs const *shape_inputs,
        [[maybe_unused]] int32_t nb_shape_inputs,
        nvinfer1::DimsExprs *outputs,
        int32_t nb_outputs,
        [[maybe_unused]] nvinfer1::IExprBuilder &expr_builder) noexcept {

    // Validate inputs and outputs counts before creating spans
    static constexpr int32_t EXPECTED_INPUTS = 2;
    static constexpr int32_t EXPECTED_OUTPUTS = 2;

    if (nb_inputs != EXPECTED_INPUTS) {
        std::cerr << std::format(
                "FftTrtPlugin::getOutputShapes error: Expected {} inputs, got {}\n",
                EXPECTED_INPUTS,
                nb_inputs);
        return -1;
    }
    if (nb_outputs != EXPECTED_OUTPUTS) {
        std::cerr << std::format(
                "FftTrtPlugin::getOutputShapes error: Expected {} outputs, got {}\n",
                EXPECTED_OUTPUTS,
                nb_outputs);
        return -1;
    }

    // Use span to avoid pointer arithmetic
    const std::span<const nvinfer1::DimsExprs> inputs_span(inputs, 2);
    const std::span<nvinfer1::DimsExprs> outputs_span(outputs, 2);

    // Output shape matches input shape: [batch_size, fft_size]
    // Input is complex data, output is also complex data
    // Both inputs should have the same shape, so we use inputs[0] as reference
    static constexpr int32_t TWO_DIMS = 2;
    static constexpr int32_t ONE_DIM = 1;
    if (inputs_span[0].nbDims == TWO_DIMS) {
        // Batched input - [batch_size, fft_size]
        outputs_span[0].nbDims = TWO_DIMS;
        outputs_span[0].d[0] = inputs_span[0].d[0]; // batch_size
        outputs_span[0].d[1] = inputs_span[0].d[1]; // fft_size
        outputs_span[1].nbDims = TWO_DIMS;
        outputs_span[1].d[0] = inputs_span[0].d[0]; // batch_size
        // cppcheck-suppress unreadVariable ; false positive - span modifies underlying array
        outputs_span[1].d[1] = inputs_span[0].d[1]; // fft_size
    } else if (inputs_span[0].nbDims == ONE_DIM) {
        // Single input - [fft_size]
        outputs_span[0].nbDims = ONE_DIM;
        outputs_span[0].d[0] = inputs_span[0].d[0]; // fft_size
        outputs_span[1].nbDims = ONE_DIM;
        // cppcheck-suppress unreadVariable ; false positive - span modifies underlying array
        outputs_span[1].d[0] = inputs_span[0].d[0]; // fft_size
    } else {
        // Handle scalar input case (nbDims == 0)
        outputs_span[0].nbDims = 0;
        // cppcheck-suppress unreadVariable ; false positive - span modifies underlying array
        outputs_span[1].nbDims = 0;
    }
    return 0;
}

bool FftTrtPlugin::supportsFormatCombination(
        const int32_t pos,
        nvinfer1::DynamicPluginTensorDesc const *in_out,
        [[maybe_unused]] const int32_t nb_inputs,
        [[maybe_unused]] const int32_t nb_outputs) noexcept {

    // Support only FLOAT data type with linear format for complex data - use span to avoid pointer
    // arithmetic
    const std::span<const nvinfer1::DynamicPluginTensorDesc> in_out_span(in_out, pos + 1);
    return (in_out_span[pos].desc.type == nvinfer1::DataType::kFLOAT &&
            in_out_span[pos].desc.format == nvinfer1::PluginFormat::kLINEAR);
}

int32_t FftTrtPlugin::getNbOutputs() const noexcept { return 2; }

int32_t FftTrtPlugin::getOutputDataTypes(
        nvinfer1::DataType *output_types,
        [[maybe_unused]] const int32_t nb_outputs,
        nvinfer1::DataType const *input_types,
        [[maybe_unused]] const int32_t nb_inputs) const noexcept {
    // Output data type matches input data type (complex float) - use span to avoid pointer
    // arithmetic
    const std::span<nvinfer1::DataType> output_types_span(output_types, 2);
    const std::span<const nvinfer1::DataType> input_types_span(input_types, 2);
    output_types_span[0] = input_types_span[0];
    output_types_span[1] = input_types_span[1];
    return 0;
}

// IPluginV3OneRuntime interface implementation
int32_t FftTrtPlugin::enqueue(
        nvinfer1::PluginTensorDesc const *input_desc,
        [[maybe_unused]] nvinfer1::PluginTensorDesc const *output_desc,
        void const *const *inputs,
        void *const *outputs,
        [[maybe_unused]] void *workspace,
        cudaStream_t stream) noexcept {
    // Use span to avoid pointer arithmetic
    const std::span<void const *const> inputs_span(inputs, 2);
    const std::span<void *const> outputs_span(outputs, 2);
    const std::span<const nvinfer1::PluginTensorDesc> input_desc_span(input_desc, 1);

    // Extract input and output pointers for separate real and imaginary parts
    const auto *input_real = static_cast<const float *>(inputs_span[0]);
    const auto *input_imag = static_cast<const float *>(inputs_span[1]);
    auto *output_real = static_cast<float *>(outputs_span[0]);
    auto *output_imag = static_cast<float *>(outputs_span[1]);

    // Use configured FFT size
    const int32_t fft_size = m_fft_size_;

    // Get batch size from input tensor
    static constexpr int32_t TWO_DIMS = 2;
    static constexpr int32_t ONE_DIM = 1;
    int32_t batch_size = 1; // Default to 1 for single input
    if (input_desc_span[0].dims.nbDims == TWO_DIMS) {
        // Batched input - [batch_size, fft_size]
        batch_size = static_cast<int32_t>(input_desc_span[0].dims.d[0]);
    } else if (input_desc_span[0].dims.nbDims == ONE_DIM) {
        // Single input - [fft_size]
        batch_size = 1;
    }

    try {
        // Pre-validate inputs to avoid kernel launch overhead for invalid data
        if (batch_size <= 0 || fft_size <= 0) {
            std::cerr << std::format(
                    "FftTrtPlugin::enqueue error: Invalid parameters (batch_size={}, "
                    "fft_size={})\n",
                    batch_size,
                    fft_size);
            return -1;
        }

        // Ensure all input pointers are valid before kernel launch
        if (input_real == nullptr || input_imag == nullptr || output_real == nullptr ||
            output_imag == nullptr) {
            std::cerr << std::format(
                    "FftTrtPlugin::enqueue error: Null pointer detected (input_real={}, "
                    "input_imag={}, output_real={}, output_imag={})\n",
                    static_cast<const void *>(input_real),
                    static_cast<const void *>(input_imag),
                    static_cast<void *>(output_real),
                    static_cast<void *>(output_imag));
            return -1;
        }

        // Launch cuFFT kernel with direction parameter
        static constexpr int32_t DEFAULT_PRECISION = 0;
        static constexpr int32_t DEFAULT_FFT_TYPE = 0;
        static constexpr int32_t DEFAULT_FFTS_PER_BLOCK = 1;
        launch_fft_kernel(
                input_real,
                input_imag,
                fft_size,
                batch_size,
                output_real,
                output_imag,
                workspace,
                stream,
                DEFAULT_PRECISION,
                DEFAULT_FFT_TYPE,
                m_direction_,
                DEFAULT_FFTS_PER_BLOCK,
                FftTrtPluginParams::DEFAULT_ELEMENTS_PER_THREAD);

        return 0; // Success
    } catch (const std::exception &e) {
        std::cerr << std::format("FftTrtPlugin::enqueue error: {}\n", e.what());
        return -1; // Failure
    }
}

nvinfer1::PluginFieldCollection const *FftTrtPlugin::getFieldsToSerialize() noexcept {
    populate_fft_fields(
            m_serialization_fields_, m_serialization_collection_, &m_fft_size_, &m_direction_);
    return &m_serialization_collection_;
}

// Plugin Creator Implementation
FftTrtPluginCreator::FftTrtPluginCreator(const std::string_view name_space)
        : TrtPluginCreatorBase(name_space) {
    populate_fft_fields(m_plugin_fields_, m_field_collection_);
}

nvinfer1::IPluginV3 *FftTrtPluginCreator::createPlugin(
        nvinfer1::AsciiChar const *name,
        nvinfer1::PluginFieldCollection const *fc,
        [[maybe_unused]] nvinfer1::TensorRTPhase phase) noexcept {

    try {
        // Extract FFT configuration from field collection with defaults
        const auto fft_size = get_plugin_field<std::int32_t>(
                fc,
                "fft_size",
                nvinfer1::PluginFieldType::kINT32,
                FftTrtPluginParams::DEFAULT_FFT_SIZE);

        const auto direction_int = get_plugin_field<std::int32_t>(
                fc, "direction", nvinfer1::PluginFieldType::kINT32, 0);

        // Create plugin with designated initializers - only override what's specified
        auto plugin = std::make_unique<FftTrtPlugin>(
                name,
                m_namespace_,
                FftTrtPluginParams{
                        .fft_size = fft_size,
                        .direction = (direction_int == 0) ? "forward" : "inverse"
                        // Other fields use defaults from struct
                });
        return plugin.release();
    } catch (const std::exception &e) {
        std::cerr << std::format("Error creating FftTrtPlugin: {}\n", e.what());
        return nullptr;
    }
}

} // namespace ran::trt_plugin
