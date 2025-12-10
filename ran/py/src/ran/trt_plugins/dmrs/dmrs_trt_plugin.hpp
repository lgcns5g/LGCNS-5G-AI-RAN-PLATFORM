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

#ifndef RAN_DMRS_TRT_PLUGIN_HPP
#define RAN_DMRS_TRT_PLUGIN_HPP

#include <cstdint>
#include <string_view>
#include <vector> // for vector

#include <NvInfer.h>      // for IPluginV3
#include <driver_types.h> // for cudaStream_t

#include "trt_plugin_base.hpp"
#include "trt_plugin_creator_base.hpp"

namespace ran::trt_plugin {

/**
 * @brief TensorRT plugin for DMRS generation
 *
 * This plugin implements the 3GPP DMRS generation algorithm using CUDA
 * kernels for TensorRT inference. It takes slot number and DMRS ID as
 * inputs and produces two outputs: complex DMRS values and binary gold
 * sequences for all OFDM symbols and ports.
 *
 * Input: [2] containing [slot_number, n_dmrs_id]
 * Output 0: (2, n_t, 2, sequence_length/2) - Complex DMRS values
 * Output 1: (n_t, 2, sequence_length) - Binary gold sequence
 */
class __attribute__((visibility("default"))) DMRSTrtPlugin final
        : public TrtPluginBase<DMRSTrtPlugin> {
public:
    // Plugin constants required by base class
    static constexpr const char *PLUGIN_TYPE = "DmrsTrt";       //!< Plugin type identifier
    static constexpr const char *PLUGIN_VERSION = "1";          //!< Plugin version string
    static constexpr std::int32_t DEFAULT_SEQUENCE_LENGTH = 42; //!< Default DMRS sequence length
    static constexpr std::int32_t DEFAULT_N_T = 14;             //!< Default OFDM symbols per slot

    /**
     * @brief Constructor with explicit sequence length and n_t
     *
     * Creates a plugin instance with a specific sequence length and number of
     * OFDM symbols. This constructor is primarily used by the clone() method to
     * correctly initialize new instances.
     *
     * @param[in] name Plugin name identifier
     * @param[in] name_space Plugin namespace (defaults to empty)
     * @param[in] sequence_length Length of the DMRS sequence to generate
     * @param[in] n_t Number of OFDM symbols per slot
     */
    explicit DMRSTrtPlugin(
            std::string_view name,
            std::string_view name_space = "",
            std::int32_t sequence_length = DEFAULT_SEQUENCE_LENGTH,
            std::int32_t n_t = DEFAULT_N_T);

    /**
     * @brief Virtual destructor
     */
    ~DMRSTrtPlugin() override = default;

    // Delete copy and move operations (use clone() for copying)
    DMRSTrtPlugin(const DMRSTrtPlugin &) = delete;
    DMRSTrtPlugin &operator=(const DMRSTrtPlugin &) = delete;
    DMRSTrtPlugin(DMRSTrtPlugin &&) = delete;
    DMRSTrtPlugin &operator=(DMRSTrtPlugin &&) = delete;

    // IPluginV3 methods
    /**
     * @brief Creates a deep copy of the plugin instance
     *
     * Returns a new plugin instance with identical configuration and state.
     * This is used by TensorRT for plugin cloning and resource management.
     *
     * @return Pointer to a new plugin instance
     * @note The returned plugin is owned by the caller and must be deleted
     */
    [[nodiscard]] nvinfer1::IPluginV3 *clone() noexcept override;

    // IPluginV3OneBuild methods
    /**
     * @brief Returns the number of output tensors
     *
     * This plugin produces two outputs: complex DMRS values and binary gold sequence.
     *
     * @return Number of output tensors produced by the plugin (always 2)
     */
    [[nodiscard]] std::int32_t getNbOutputs() const noexcept override;

    /**
     * @brief Determines the output data types based on input types
     *
     * Sets the output tensor data types. This plugin produces two outputs:
     * Output 0 is FLOAT32 (complex DMRS values), Output 1 is INT32 (binary sequence).
     *
     * @param[out] output_types Array to store output data types
     * @param[in] nb_outputs Number of output tensors
     * @param[in] input_types Array of input data types
     * @param[in] nb_inputs Number of input tensors
     * @return 0 on success, non-zero on failure
     */
    std::int32_t getOutputDataTypes(
            nvinfer1::DataType *output_types,
            std::int32_t nb_outputs,
            nvinfer1::DataType const *input_types,
            std::int32_t nb_inputs) const noexcept override;

    /**
     * @brief Computes output tensor shapes based on input shapes
     *
     * Determines the output tensor dimensions using the expression builder
     * for dynamic shape support. This plugin produces two output tensors
     * from scalar input parameters.
     *
     * @param[in] inputs Array of input tensor shapes
     * @param[in] nb_inputs Number of input tensors
     * @param[in] shape_inputs Array of shape input tensors (unused)
     * @param[in] nb_shape_inputs Number of shape input tensors
     * @param[out] outputs Array to store output tensor shapes
     * @param[in] nb_outputs Number of output tensors
     * @param[in] expr_builder Expression builder for dynamic shape computation
     * @return 0 on success, non-zero on failure
     * @note Input shape: [2] - contains [slot_number, n_dmrs_id] parameters
     * @note Output 0 shape: (2, n_t, 2, sequence_length/2) - Complex DMRS values [real/imag,
     * symbols, ports, subcarriers]
     * @note Output 1 shape: (n_t, 2, sequence_length) - Binary gold sequence [symbols, ports,
     * subcarriers]
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
     * @brief Checks if the plugin supports the given format combination
     *
     * Validates that the input/output format combination is supported
     * by the plugin implementation.
     *
     * @param[in] pos Position in the input/output array to check
     * @param[in] in_out Array of input/output tensor descriptions
     * @param[in] nb_inputs Number of input tensors
     * @param[in] nb_outputs Number of output tensors
     * @return true if the format combination is supported, false otherwise
     */
    bool supportsFormatCombination(
            std::int32_t pos,
            nvinfer1::DynamicPluginTensorDesc const *in_out,
            std::int32_t nb_inputs,
            std::int32_t nb_outputs) noexcept override;

    // IPluginV3OneRuntime methods

    /**
     * @brief Executes the DMRS sequence generation kernel
     *
     * This is the main execution method called by TensorRT during inference.
     * It launches a CUDA kernel that generates pseudo-random DMRS sequences
     * for all n_t OFDM symbols and both n_scid ports (0, 1) based on the
     * input slot number and DMRS ID parameters.
     *
     * The kernel implements the 3GPP DMRS sequence algorithm:
     * 1. Generates two M-sequences (x1 and x2) using linear feedback shift
     * registers based on the computed c_init value
     * 2. Combines them using modulo-2 addition with a 1600-bit offset
     * 3. Outputs complex DMRS values and binary gold sequences for all
     * symbols and ports
     *
     * @param[in] input_desc Array of input tensor descriptions
     * @param[in] output_desc Array of output tensor descriptions
     * @param[in] inputs Array of input data pointers (GPU memory)
     * @param[out] outputs Array of output data pointers (GPU memory)
     * @param[in] workspace Workspace memory (unused)
     * @param[in] stream CUDA stream for asynchronous execution
     * @return 0 on success, -1 on failure
     * @note Input shape: [2] - contains [slot_number, n_dmrs_id] parameters
     * @note Output 0 shape: (2, n_t, 2, sequence_length/2) - Complex DMRS values
     * @note Output 1 shape: (n_t, 2, sequence_length) - Binary gold sequence
     * @note All data pointers must be valid GPU memory addresses
     * @see launch_dmrs_kernel for the underlying CUDA implementation
     */
    std::int32_t
    enqueue(nvinfer1::PluginTensorDesc const *input_desc,
            nvinfer1::PluginTensorDesc const *output_desc,
            void const *const *inputs,
            void *const *outputs,
            void *workspace,
            cudaStream_t stream) noexcept override;

    /**
     * @brief Returns the fields that should be serialized
     *
     * Specifies which plugin parameters should be saved when serializing
     * the model. This plugin serializes:
     * - sequence_length: Length of DMRS sequences to generate
     * - n_t: Number of OFDM symbols per slot
     *
     * @return Pointer to field collection containing sequence_length and n_t
     */
    [[nodiscard]] nvinfer1::PluginFieldCollection const *getFieldsToSerialize() noexcept override;

private:
    std::int32_t m_sequence_length_{
            DEFAULT_SEQUENCE_LENGTH}; //!< Length of DMRS sequences to generate
    std::int32_t m_n_t_{DEFAULT_N_T}; //!< Number of OFDM symbols per slot
    mutable std::vector<nvinfer1::PluginField>
            m_serialization_fields_; //!< Fields for serialization
    mutable nvinfer1::PluginFieldCollection
            m_serialization_collection_; //!< Field collection for serialization
};

/**
 * @brief Plugin Creator for DMRSTrtPlugin
 *
 * This class handles the creation and configuration of DMRSTrtPlugin
 * instances. It implements the TensorRT plugin creator interface and manages
 * plugin field collection for parameter configuration.
 *
 * The creator extracts sequence length parameters from the field collection
 * and ensures proper plugin initialization with the correct configuration.
 *
 * @see DMRSTrtPlugin for the main plugin implementation
 * @see TrtPluginCreatorBase for the base creator interface
 */
class __attribute__((visibility("default"))) DMRSTrtPluginCreator final
        : public TrtPluginCreatorBase<DMRSTrtPlugin> {
public:
    /**
     * @brief Constructor with required namespace
     *
     * Initializes the creator with the specified namespace
     * and prepares the plugin field collection.
     *
     * @param[in] name_space Plugin namespace
     */
    explicit DMRSTrtPluginCreator(std::string_view name_space);

    /**
     * @brief Virtual destructor
     */
    ~DMRSTrtPluginCreator() override = default;

    // Delete copy and move operations
    DMRSTrtPluginCreator(const DMRSTrtPluginCreator &) = delete;
    DMRSTrtPluginCreator &operator=(const DMRSTrtPluginCreator &) = delete;
    DMRSTrtPluginCreator(DMRSTrtPluginCreator &&) = delete;
    DMRSTrtPluginCreator &operator=(DMRSTrtPluginCreator &&) = delete;

    /**
     * @brief Creates a new plugin instance
     *
     * Instantiates a DMRSTrtPlugin with the specified name
     * and configuration from the field collection.
     *
     * @param[in] name Name for the new plugin instance
     * @param[in] fc Field collection containing configuration parameters
     * @param[in] phase TensorRT phase (build or runtime)
     * @return Pointer to the created plugin instance
     * @note The returned plugin is owned by the caller and must be deleted
     */
    [[nodiscard]] nvinfer1::IPluginV3 *createPlugin(
            nvinfer1::AsciiChar const *name,
            nvinfer1::PluginFieldCollection const *fc,
            nvinfer1::TensorRTPhase phase) noexcept override;
};

/**
 * Launches DMRS sequence generation kernel
 *
 * This function launches the CUDA kernel that generates DMRS sequences
 * for all n_t symbols and both n_scid ports (0, 1) using scalar inputs.
 * The kernel internally loops over n_t symbols and n_scid ports to produce
 * two output tensors: complex DMRS values and binary gold sequence.
 *
 * @param[in] input_params GPU pointer to [slot_number, n_dmrs_id] array
 * @param[in] sequence_length Sequence length per port (compile-time constant)
 * @param[in] n_t Number of OFDM symbols per slot (compile-time constant)
 * @param[out] r_dmrs_ri_sym_cdm_sc Complex DMRS output (GPU memory, 2 x n_t x 2 x
 * sequence_length/2)
 * @param[out] scr_seq_sym_ri_sc Binary gold sequence output (GPU memory, n_t x 2 x sequence_length)
 * @param[in] stream CUDA stream for asynchronous execution
 *
 * @note Complex output shape: (2, n_t, 2, sequence_length/2) where dim0: [0]=real, [1]=imag
 * @note Binary output shape: (n_t, 2, sequence_length)
 * @note input_params is GPU memory containing [slot_number, n_dmrs_id]
 * @see dmrs_kernel for the actual CUDA kernel implementation
 */
void launch_dmrs_kernel(
        const std::int32_t *input_params,
        std::int32_t sequence_length,
        std::int32_t n_t,
        float *r_dmrs_ri_sym_cdm_sc,
        std::int32_t *scr_seq_sym_ri_sc,
        cudaStream_t stream);

} // namespace ran::trt_plugin

#endif // RAN_DMRS_TRT_PLUGIN_HPP
