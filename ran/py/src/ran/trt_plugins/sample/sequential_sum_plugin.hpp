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

#ifndef RAN_SAMPLE_SEQUENTIAL_SUM_PLUGIN_HPP
#define RAN_SAMPLE_SEQUENTIAL_SUM_PLUGIN_HPP

#include <cstdint>
#include <string_view>
#include <vector>

#include <NvInfer.h>      // for IPluginV3
#include <driver_types.h> // for cudaStream_t

#include "trt_plugin_base.hpp"
#include "trt_plugin_creator_base.hpp"

namespace ran::trt_plugin {

/**
 * Sequential Sum Plugin - deliberately non-parallelizable operation
 *
 * This plugin computes a sequential sum where each element depends on the previous:
 * output[i] = input[i] + output[i-1]
 *
 * This is intentionally designed to be difficult to parallelize to demonstrate
 * a custom kernel that doesn't scale well with parallelization.
 */
class __attribute__((visibility("default"))) SequentialSumPlugin final
        : public TrtPluginBase<SequentialSumPlugin> {
public:
    static constexpr const char *PLUGIN_TYPE = "SequentialSum"; //!< Plugin type identifier
    static constexpr const char *PLUGIN_VERSION = "1";          //!< Plugin version string

    /**
     * Constructor for plugin creation
     *
     * @param[in] name Plugin instance name
     * @param[in] name_space Plugin namespace (defaults to empty)
     */
    explicit SequentialSumPlugin(std::string_view name, std::string_view name_space = "");

    /**
     * Destructor
     */
    ~SequentialSumPlugin() override = default;

    // Delete copy and move operations (use clone() for copying)
    SequentialSumPlugin(const SequentialSumPlugin &) = delete;
    SequentialSumPlugin &operator=(const SequentialSumPlugin &) = delete;
    SequentialSumPlugin(SequentialSumPlugin &&) = delete;
    SequentialSumPlugin &operator=(SequentialSumPlugin &&) = delete;

    /**
     * Creates a copy of the plugin
     *
     * @return New plugin instance
     */
    [[nodiscard]] nvinfer1::IPluginV3 *clone() noexcept override;

    /**
     * Determines output data types based on input types
     *
     * @param[out] output_types Array to store output data types
     * @param[in] nb_outputs Number of outputs
     * @param[in] input_types Array of input data types
     * @param[in] nb_inputs Number of inputs
     * @return 0 on success
     */
    std::int32_t getOutputDataTypes(
            nvinfer1::DataType *output_types,
            std::int32_t nb_outputs,
            nvinfer1::DataType const *input_types,
            std::int32_t nb_inputs) const noexcept override;

    /**
     * Computes output shapes based on input shapes
     *
     * @param[in] inputs Array of input tensor shapes
     * @param[in] nb_inputs Number of inputs
     * @param[in] shape_inputs Array of shape input tensors
     * @param[in] nb_shape_inputs Number of shape inputs
     * @param[out] outputs Array to store output shapes
     * @param[in] nb_outputs Number of outputs
     * @param[in,out] expr_builder Expression builder for shape calculations
     * @return 0 on success
     */
    std::int32_t getOutputShapes(
            nvinfer1::DimsExprs const *inputs,
            std::int32_t nb_inputs,
            nvinfer1::DimsExprs const *shape_inputs,
            std::int32_t nb_shape_inputs,
            nvinfer1::DimsExprs *outputs,
            std::int32_t nb_outputs,
            nvinfer1::IExprBuilder &expr_builder) noexcept override;

    /**
     * Checks if format combination is supported
     *
     * @param[in] pos Position in input/output array to check
     * @param[in] in_out Array of input and output tensor descriptors
     * @param[in] nb_inputs Number of inputs
     * @param[in] nb_outputs Number of outputs
     * @return true if format combination is supported
     */
    [[nodiscard]] bool supportsFormatCombination(
            std::int32_t pos,
            nvinfer1::DynamicPluginTensorDesc const *in_out,
            std::int32_t nb_inputs,
            std::int32_t nb_outputs) noexcept override;

    /**
     * Returns the number of outputs
     *
     * @return Number of output tensors
     */
    [[nodiscard]] std::int32_t getNbOutputs() const noexcept override;

    /**
     * Executes the plugin operation
     *
     * @param[in] input_desc Array of input tensor descriptors
     * @param[in] output_desc Array of output tensor descriptors
     * @param[in] inputs Array of input device buffers
     * @param[out] outputs Array of output device buffers
     * @param[in,out] workspace Workspace memory pointer
     * @param[in] stream CUDA stream for kernel execution
     * @return 0 on success
     */
    std::int32_t
    enqueue(nvinfer1::PluginTensorDesc const *input_desc,
            nvinfer1::PluginTensorDesc const *output_desc,
            void const *const *inputs,
            void *const *outputs,
            void *workspace,
            cudaStream_t stream) noexcept override;

    /**
     * Returns fields to be serialized
     *
     * @return Pointer to field collection for serialization
     */
    [[nodiscard]] nvinfer1::PluginFieldCollection const *getFieldsToSerialize() noexcept override;

private:
    // Serialization support - must be mutable for getFieldsToSerialize()
    mutable std::vector<nvinfer1::PluginField>
            m_serialization_fields_; //!< Fields for serialization
    mutable nvinfer1::PluginFieldCollection
            m_serialization_collection_; //!< Field collection for serialization
};

/**
 * Plugin Creator for SequentialSumPlugin
 */
class __attribute__((visibility("default"))) SequentialSumPluginCreator final
        : public TrtPluginCreatorBase<SequentialSumPlugin> {
public:
    /**
     * @brief Constructor with required namespace
     *
     * @param[in] name_space Plugin namespace
     */
    explicit SequentialSumPluginCreator(std::string_view name_space);
    ~SequentialSumPluginCreator() override = default;

    // Delete copy and move operations
    SequentialSumPluginCreator(const SequentialSumPluginCreator &) = delete;
    SequentialSumPluginCreator &operator=(const SequentialSumPluginCreator &) = delete;
    SequentialSumPluginCreator(SequentialSumPluginCreator &&) = delete;
    SequentialSumPluginCreator &operator=(SequentialSumPluginCreator &&) = delete;

    /**
     * Creates a new plugin instance
     *
     * @param[in] name Plugin instance name
     * @param[in] fc Field collection containing plugin parameters
     * @param[in] phase TensorRT phase (build or runtime)
     * @return New plugin instance or nullptr on failure
     */
    [[nodiscard]] nvinfer1::IPluginV3 *createPlugin(
            nvinfer1::AsciiChar const *name,
            nvinfer1::PluginFieldCollection const *fc,
            nvinfer1::TensorRTPhase phase) noexcept override;
};

/**
 * Launches CUDA kernel for sequential sum computation
 *
 * @param[in] input Input array on device
 * @param[out] output Output array on device
 * @param[in] size Number of elements in arrays
 * @param[in] stream CUDA stream for kernel execution
 */
void launch_sequential_sum_kernel(
        const float *input, float *output, int64_t size, cudaStream_t stream);

} // namespace ran::trt_plugin

#endif // RAN_SAMPLE_SEQUENTIAL_SUM_PLUGIN_HPP
