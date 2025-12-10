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

#include <exception>  // for exception
#include <format>     // for format
#include <functional> // for multiplies
#include <iostream>   // for basic_ostream, operator<<, cerr, endl
#include <memory>     // for unique_ptr
#include <numeric>    // for accumulate
#include <span>       // for span
#include <string>

#include <NvInfer.h> // for IPluginRegistry

#include "sequential_sum_plugin.hpp"

namespace ran::trt_plugin {

namespace {

/**
 * Populates field collection with SequentialSum plugin fields
 *
 * This plugin has no configuration fields.
 *
 * @param[in,out] fields Vector to populate with field definitions
 * @param[in,out] collection Field collection to update
 */
void populate_sequential_sum_fields(
        std::vector<nvinfer1::PluginField> &fields, nvinfer1::PluginFieldCollection &collection) {

    fields.clear();
    // No fields for this simple plugin

    collection.nbFields = 0;
    collection.fields = nullptr;
}

} // anonymous namespace

// Plugin Implementation
SequentialSumPlugin::SequentialSumPlugin(
        const std::string_view name, const std::string_view name_space)
        : TrtPluginBase(name, name_space) {}

// IPluginV3 methods
nvinfer1::IPluginV3 *SequentialSumPlugin::clone() noexcept {
    try {
        auto plugin = std::make_unique<SequentialSumPlugin>(m_plugin_name_, m_namespace_);
        return plugin.release();
    } catch (const std::exception &e) {
        std::cerr << std::format("Error cloning SequentialSumPlugin: {}\n", e.what());
        return nullptr;
    }
}

// IPluginV3OneBuild methods
int32_t SequentialSumPlugin::getOutputShapes(
        nvinfer1::DimsExprs const *inputs,
        const int32_t nb_inputs,
        [[maybe_unused]] nvinfer1::DimsExprs const *shape_inputs,
        [[maybe_unused]] const int32_t nb_shape_inputs,
        nvinfer1::DimsExprs *outputs,
        const int32_t nb_outputs,
        [[maybe_unused]] nvinfer1::IExprBuilder &expr_builder) noexcept {

    // Validate inputs and outputs counts before creating spans
    if (nb_inputs < 1) {
        std::cerr << std::format(
                "SequentialSumPlugin::getOutputShapes error: Invalid input count (nb_inputs={})\n",
                nb_inputs);
        return -1;
    }
    if (nb_outputs < 1) {
        std::cerr << std::format(
                "SequentialSumPlugin::getOutputShapes error: Invalid output count "
                "(nb_outputs={})\n",
                nb_outputs);
        return -1;
    }

    // Output has same dimensions as input - use span to avoid pointer arithmetic
    const std::span<const nvinfer1::DimsExprs> inputs_span(inputs, 1);
    const std::span<nvinfer1::DimsExprs> outputs_span(outputs, 1);
    outputs_span[0] = inputs_span[0];
    return 0;
}

bool SequentialSumPlugin::supportsFormatCombination(
        const int32_t pos,
        nvinfer1::DynamicPluginTensorDesc const *in_out,
        [[maybe_unused]] const int32_t nb_inputs,
        [[maybe_unused]] const int32_t nb_outputs) noexcept {

    // Support only FP32 and linear format - use span to avoid pointer arithmetic
    const std::span<const nvinfer1::DynamicPluginTensorDesc> in_out_span(in_out, pos + 1);
    return (in_out_span[pos].desc.type == nvinfer1::DataType::kFLOAT &&
            in_out_span[pos].desc.format == nvinfer1::PluginFormat::kLINEAR);
}

int32_t SequentialSumPlugin::getNbOutputs() const noexcept { return 1; }

int32_t SequentialSumPlugin::getOutputDataTypes(
        nvinfer1::DataType *output_types,
        [[maybe_unused]] const int32_t nb_outputs,
        nvinfer1::DataType const *input_types,
        [[maybe_unused]] const int32_t nb_inputs) const noexcept {
    // Output type is same as input type - use span to avoid pointer arithmetic
    const std::span<nvinfer1::DataType> output_types_span(output_types, 1);
    const std::span<const nvinfer1::DataType> input_types_span(input_types, 1);
    output_types_span[0] = input_types_span[0];
    return 0;
}

// IPluginV3OneRuntime methods
int32_t SequentialSumPlugin::enqueue(
        nvinfer1::PluginTensorDesc const *input_desc,
        [[maybe_unused]] nvinfer1::PluginTensorDesc const *output_desc,
        void const *const *inputs,
        void *const *outputs,
        [[maybe_unused]] void *workspace,
        cudaStream_t stream) noexcept {

    // Use span to avoid pointer arithmetic
    const std::span<void const *const> inputs_span(inputs, 1);
    const std::span<void *const> outputs_span(outputs, 1);
    const std::span<const nvinfer1::PluginTensorDesc> input_desc_span(input_desc, 1);

    const auto *input = static_cast<const float *>(inputs_span[0]);
    auto *output = static_cast<float *>(outputs_span[0]);

    // Calculate total number of elements
    const std::span<const int64_t> dims_span(
            &input_desc_span[0].dims.d[0], input_desc_span[0].dims.nbDims);
    const int64_t size =
            std::accumulate(dims_span.begin(), dims_span.end(), int64_t{1}, std::multiplies<>());

    // Pre-validate inputs to avoid kernel launch overhead for invalid data
    if (size <= 0) {
        std::cerr << std::format(
                "SequentialSumPlugin::enqueue error: Invalid size (size={})\n", size);
        return -1;
    }

    // Ensure all input pointers are valid before kernel launch
    if (input == nullptr || output == nullptr) {
        std::cerr << std::format(
                "SequentialSumPlugin::enqueue error: Null pointer detected (input={}, "
                "output={})\n",
                static_cast<const void *>(input),
                static_cast<void *>(output));
        return -1;
    }

    try {
        launch_sequential_sum_kernel(input, output, size, stream);
        return 0; // Success
    } catch (const std::exception &e) {
        std::cerr << std::format("SequentialSumPlugin::enqueue error: {}\n", e.what());
        return -1; // Failure
    }
}

nvinfer1::PluginFieldCollection const *SequentialSumPlugin::getFieldsToSerialize() noexcept {
    populate_sequential_sum_fields(m_serialization_fields_, m_serialization_collection_);
    return &m_serialization_collection_;
}

// Plugin Creator Implementation
SequentialSumPluginCreator::SequentialSumPluginCreator(const std::string_view name_space)
        : TrtPluginCreatorBase(name_space) {
    populate_sequential_sum_fields(m_plugin_fields_, m_field_collection_);
}

nvinfer1::IPluginV3 *SequentialSumPluginCreator::createPlugin(
        nvinfer1::AsciiChar const *name,
        [[maybe_unused]] nvinfer1::PluginFieldCollection const *fc,
        [[maybe_unused]] nvinfer1::TensorRTPhase phase) noexcept {

    try {
        auto plugin = std::make_unique<SequentialSumPlugin>(name, m_namespace_);
        return plugin.release();
    } catch (const std::exception &e) {
        std::cerr << std::format("Error creating SequentialSumPlugin: {}\n", e.what());
        return nullptr;
    }
}

} // namespace ran::trt_plugin
