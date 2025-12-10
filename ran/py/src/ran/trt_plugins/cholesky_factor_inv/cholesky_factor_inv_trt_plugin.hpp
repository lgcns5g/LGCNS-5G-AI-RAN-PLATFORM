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

#ifndef RAN_CHOLESKY_FACTOR_INV_TRT_PLUGIN_HPP
#define RAN_CHOLESKY_FACTOR_INV_TRT_PLUGIN_HPP

#include <cstdint> // for std::int32_t
#include <string_view>
#include <vector> // for vector

#include <NvInfer.h>      // for IPluginV3
#include <driver_types.h> // for cudaStream_t

#include "trt_plugin_base.hpp"
#include "trt_plugin_creator_base.hpp"

namespace ran::trt_plugin {

/**
 * @brief TensorRT plugin for Cholesky decomposition and matrix inversion
 *
 * This plugin implements Cholesky decomposition followed by matrix inversion
 * using NVIDIA's cuSOLVERDx library for high-performance execution during
 * TensorRT inference.
 *
 * The plugin computes: A^{-1} where A is a positive definite matrix
 * Method: Cholesky decomposition A = L*L^H, then solve L*L^H*X = I for X
 */
class __attribute__((visibility("default"))) CholeskyFactorInvPlugin final
        : public TrtPluginBase<CholeskyFactorInvPlugin> {
public:
    // Plugin constants required by base class
    static constexpr const char *PLUGIN_TYPE = "CholeskyFactorInv"; //!< Plugin type identifier
    static constexpr const char *PLUGIN_VERSION = "1";              //!< Plugin version string
    static constexpr std::int32_t DEFAULT_MATRIX_SIZE = 2;          //!< Default matrix size
    /**
     * @brief Constructor with explicit matrix size and complex flag
     *
     * Creates a plugin instance with a specific matrix size and data type.
     * This constructor is primarily used by the clone() method to correctly
     * initialize new instances.
     *
     * @param[in] name Plugin name identifier
     * @param[in] name_space Plugin namespace (defaults to empty)
     * @param[in] matrix_size Size of the square matrix (N for NxN matrix)
     * @param[in] is_complex Whether data is complex (true) or real (false)
     */
    explicit CholeskyFactorInvPlugin(
            std::string_view name,
            std::string_view name_space = "",
            std::int32_t matrix_size = DEFAULT_MATRIX_SIZE,
            bool is_complex = false);

    /**
     * @brief Virtual destructor
     */
    ~CholeskyFactorInvPlugin() override = default;

    // Delete copy and move operations (use clone() for copying)
    CholeskyFactorInvPlugin(const CholeskyFactorInvPlugin &) = delete;
    CholeskyFactorInvPlugin &operator=(const CholeskyFactorInvPlugin &) = delete;
    CholeskyFactorInvPlugin(CholeskyFactorInvPlugin &&) = delete;
    CholeskyFactorInvPlugin &operator=(CholeskyFactorInvPlugin &&) = delete;

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
     * @return Number of output tensors produced by the plugin
     */
    [[nodiscard]] std::int32_t getNbOutputs() const noexcept override;

    /**
     * @brief Determines the output data types based on input types
     *
     * Sets the output tensor data types. For this plugin, outputs use
     * float type for inverted matrices.
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
     * @brief Computes output tensor shapes based on input shapes (batched)
     *
     * Determines the output tensor dimensions using the expression builder
     * for dynamic shape support. For batched processing, the output shape
     * is [batch_size, n_prb, n_ant, n_ant] where batch_size and n_prb come
     * from the input.
     *
     * @param[in] inputs Array of input tensor shapes
     * @param[in] nb_inputs Number of input tensors
     * @param[in] shape_inputs Array of shape input tensors (unused)
     * @param[in] nb_shape_inputs Number of shape input tensors
     * @param[out] outputs Array to store output tensor shapes
     * @param[in] nb_outputs Number of output tensors
     * @param[in] expr_builder Expression builder for dynamic shape computation
     * @return 0 on success, non-zero on failure
     * @note Input shape: [batch_size, n_prb, n_ant, n_ant] - covariance matrices
     * @note Output shape: [batch_size, n_prb, n_ant, n_ant] - inverted matrices
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
     * @brief Executes the cuSOLVERDx Cholesky inversion kernel (batched)
     *
     * This is the main execution method called by TensorRT during inference.
     * It launches a CUDA kernel that performs Cholesky decomposition followed
     * by matrix inversion using cuSOLVERDx for multiple matrices in parallel.
     *
     * The kernel uses cuSOLVERDx's POTRF and TRSM implementations:
     * 1. Takes covariance matrices of shape [batch_size, n_prb, n_ant, n_ant]
     * 2. Performs Cholesky decomposition: A = L*L^H
     * 3. Solves L*L^H*X = I to compute X = A^{-1}
     * 4. Outputs inverted matrices in the same shape
     *
     * @param[in] input_desc Array of input tensor descriptions
     * @param[in] output_desc Array of output tensor descriptions
     * @param[in] inputs Array of input data pointers (GPU memory)
     * @param[out] outputs Array of output data pointers (GPU memory)
     * @param[in] workspace Workspace memory for cuSOLVERDx computation
     * @param[in] stream CUDA stream for asynchronous execution
     * @return 0 on success, -1 on failure
     * @note Input shape: [batch_size, n_prb, n_ant, n_ant] - covariance matrices
     * @note Output shape: [batch_size, n_prb, n_ant, n_ant] - inverted matrices
     * @note All data pointers must be valid GPU memory addresses
     * @see launch_cholesky_factor_inv_kernel for the underlying CUDA
     * implementation
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
     * Specifies which plugin parameters should be saved when
     * serializing the model. This plugin has no serializable fields.
     *
     * @return Pointer to an empty field collection
     */
    [[nodiscard]] nvinfer1::PluginFieldCollection const *getFieldsToSerialize() noexcept override;

private:
    std::int32_t m_matrix_size_{DEFAULT_MATRIX_SIZE}; //!< Size of the square matrix (N for NxN)
    bool m_is_complex_{}; //!< Whether data is complex (true) or real (false)

    // Serialization support - must be mutable for getFieldsToSerialize()
    mutable std::vector<nvinfer1::PluginField>
            m_serialization_fields_; //!< Fields for serialization
    mutable nvinfer1::PluginFieldCollection
            m_serialization_collection_; //!< Field collection for serialization
};

/**
 * @brief Plugin Creator for CholeskyFactorInvPlugin
 *
 * This class handles the creation and configuration of
 * CholeskyFactorInvPlugin instances. It implements the TensorRT plugin
 * creator interface and manages plugin field collection for parameter
 * configuration.
 *
 * The creator extracts matrix size parameters from the field collection
 * and ensures proper plugin initialization with the correct configuration.
 *
 * @see CholeskyFactorInvPlugin for the main plugin implementation
 * @see IPluginCreatorV3One for the base creator interface
 */
class __attribute__((visibility("default"))) CholeskyFactorInvPluginCreator final
        : public TrtPluginCreatorBase<CholeskyFactorInvPlugin> {
public:
    /**
     * @brief Constructor with required namespace
     *
     * Initializes the creator with the specified namespace
     * and prepares the plugin field collection.
     *
     * @param[in] name_space Plugin namespace
     */
    explicit CholeskyFactorInvPluginCreator(std::string_view name_space);

    /**
     * @brief Virtual destructor
     */
    ~CholeskyFactorInvPluginCreator() override = default;

    // Delete copy and move operations
    CholeskyFactorInvPluginCreator(const CholeskyFactorInvPluginCreator &) = delete;
    CholeskyFactorInvPluginCreator &operator=(const CholeskyFactorInvPluginCreator &) = delete;
    CholeskyFactorInvPluginCreator(CholeskyFactorInvPluginCreator &&) = delete;
    CholeskyFactorInvPluginCreator &operator=(CholeskyFactorInvPluginCreator &&) = delete;

    /**
     * @brief Creates a new plugin instance
     *
     * Instantiates a CholeskyFactorInvPlugin with the specified name
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

private:
    std::int32_t m_stored_matrix_size_{
            CholeskyFactorInvPlugin::DEFAULT_MATRIX_SIZE}; //!< Stored matrix size from field
                                                           //!< collection
    bool m_stored_is_complex_{}; //!< Stored is_complex flag from field collection
};

/**
 * @brief CUDA kernel launcher for Cholesky factorization and inversion (batched)
 *
 * This function launches the CUDA kernel that performs Cholesky decomposition
 * followed by matrix inversion using cuSOLVERDx library for multiple matrices
 * in parallel.
 *
 * Supports both real and complex data types:
 * - For REAL data (is_complex=false):
 *   * input_real = the real input data
 *   * input_imag = nullptr (unused)
 *   * output_real = the real output data
 *   * output_imag = nullptr (unused)
 *
 * - For COMPLEX data (is_complex=true):
 *   * input_real = real part of complex input
 *   * input_imag = imaginary part of complex input
 *   * output_real = real part of complex output
 *   * output_imag = imaginary part of complex output
 *
 * TensorRT doesn't support complex types, so complex data is split into
 * separate real and imaginary arrays at the interface level.
 *
 * @param[in] input_real Real data (if is_complex=false) or real part (if is_complex=true)
 * @param[in] input_imag Imaginary part (if is_complex=true) or nullptr (if is_complex=false)
 * @param[in] matrix_size Size of each square matrix (n_ant)
 * @param[in] batch_size Total number of matrices (batch_size * n_prb)
 * @param[out] output_real Real output (if is_complex=false) or real part (if is_complex=true)
 * @param[out] output_imag Imaginary part (if is_complex=true) or nullptr (if is_complex=false)
 * @param[in] workspace Workspace memory for cuSOLVERDx computation
 * @param[in] stream CUDA stream for asynchronous execution
 * @param[in] is_complex false for real data, true for complex data
 *
 * @see cholesky_factor_inv_kernel for the actual CUDA kernel implementation
 */
void launch_cholesky_factor_inv_kernel(
        const float *input_real,
        const float *input_imag,
        std::int32_t matrix_size,
        std::int32_t batch_size,
        float *output_real,
        float *output_imag,
        void *workspace,
        cudaStream_t stream,
        bool is_complex = false);

} // namespace ran::trt_plugin

#endif // RAN_CHOLESKY_FACTOR_INV_TRT_PLUGIN_HPP
