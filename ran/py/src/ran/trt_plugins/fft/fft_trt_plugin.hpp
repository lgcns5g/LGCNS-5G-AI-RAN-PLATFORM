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

#ifndef RAN_FFT_TRT_PLUGIN_HPP
#define RAN_FFT_TRT_PLUGIN_HPP

#include <cstdint>
#include <string> // for string
#include <string_view>
#include <vector> // for vector

#include <NvInfer.h>      // for IPluginV3
#include <driver_types.h> // for cudaStream_t

#include "trt_plugin_base.hpp"
#include "trt_plugin_creator_base.hpp"

namespace ran::trt_plugin {

/**
 * @brief FFT plugin configuration parameters
 *
 * Configuration struct for FftTrtPlugin initialization.
 * Use designated initializers to specify non-default values.
 */
struct FftTrtPluginParams {
    static constexpr std::int32_t DEFAULT_FFT_SIZE = 128;          //!< Default FFT size
    static constexpr std::int32_t DEFAULT_FFTS_PER_BLOCK = 1;      //!< Default FFTs per block
    static constexpr std::int32_t DEFAULT_ELEMENTS_PER_THREAD = 8; //!< Default elements per thread

    std::int32_t fft_size{DEFAULT_FFT_SIZE};             //!< Size of the FFT to compute
    std::string precision{"float"};                      //!< FFT precision (float, double)
    std::string fft_type{"c2c"};                         //!< FFT type (c2c, r2c, c2r)
    std::string direction{"forward"};                    //!< FFT direction (forward, inverse)
    std::int32_t ffts_per_block{DEFAULT_FFTS_PER_BLOCK}; //!< Number of FFTs per CUDA block
    std::int32_t elements_per_thread{
            DEFAULT_ELEMENTS_PER_THREAD}; //!< Number of elements per thread
};

/**
 * @brief TensorRT plugin for FFT computation (batched)
 *
 * This plugin implements FFT computation using NVIDIA's cuFFTDx library
 * for high-performance execution during TensorRT inference.
 */
class __attribute__((visibility("default"))) FftTrtPlugin final
        : public TrtPluginBase<FftTrtPlugin> {
public:
    // Plugin constants required by base class
    static constexpr const char *PLUGIN_TYPE = "FftTrt"; //!< Plugin type identifier
    static constexpr const char *PLUGIN_VERSION = "1";   //!< Plugin version string
    /**
     * @brief Constructor with FFT configuration parameters
     *
     * Creates a plugin instance with specified FFT configuration.
     * Use designated initializers to override default values.
     *
     * @param[in] name Plugin name identifier
     * @param[in] name_space Plugin namespace (defaults to empty)
     * @param[in] params FFT configuration parameters
     *
     * @code
     * // All defaults
     * auto plugin = new FftTrtPlugin("fft1");
     *
     * // Override FFT size only
     * auto plugin = new FftTrtPlugin("fft2", "", FftTrtPluginParams{.fft_size = 256});
     *
     * // Override size and direction
     * auto plugin = new FftTrtPlugin("fft3", "", FftTrtPluginParams{
     *     .fft_size = 512,
     *     .direction = "inverse"
     * });
     * @endcode
     */
    explicit FftTrtPlugin(
            std::string_view name,
            std::string_view name_space = "",
            const FftTrtPluginParams &params = {});

    /**
     * @brief Virtual destructor
     */
    ~FftTrtPlugin() override = default;

    // Delete copy and move operations (use clone() for copying)
    FftTrtPlugin(const FftTrtPlugin &) = delete;
    FftTrtPlugin &operator=(const FftTrtPlugin &) = delete;
    FftTrtPlugin(FftTrtPlugin &&) = delete;
    FftTrtPlugin &operator=(FftTrtPlugin &&) = delete;

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
     * complex float type for FFT results.
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
     * is [batch_size, fft_size] where batch_size comes from the input.
     *
     * @param[in] inputs Array of input tensor shapes
     * @param[in] nb_inputs Number of input tensors
     * @param[in] shape_inputs Array of shape input tensors (unused)
     * @param[in] nb_shape_inputs Number of shape input tensors
     * @param[out] outputs Array to store output tensor shapes
     * @param[in] nb_outputs Number of output tensors
     * @param[in] expr_builder Expression builder for dynamic shape computation
     * @return 0 on success, non-zero on failure
     * @note Input shape: [batch_size, fft_size] - complex input data
     * @note Output shape: [batch_size, fft_size] - complex FFT results
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
     * @brief Executes the MatX FFT computation kernel (batched)
     *
     * This is the main execution method called by TensorRT during inference.
     * It launches a CUDA kernel that performs FFT computation using MatX
     * for multiple input signals in parallel.
     *
     * The kernel uses MatX's FFT implementation:
     * 1. Takes complex input data of shape [batch_size, fft_size]
     * 2. Performs FFT computation using MatX's optimized FFT routines
     * 3. Outputs complex FFT results in the same shape
     *
     * @param[in] input_desc Array of input tensor descriptions
     * @param[in] output_desc Array of output tensor descriptions
     * @param[in] inputs Array of input data pointers (GPU memory)
     * @param[out] outputs Array of output data pointers (GPU memory)
     * @param[in] workspace Workspace memory for FFT computation
     * @param[in] stream CUDA stream for asynchronous execution
     * @return 0 on success, -1 on failure
     * @note Input shape: [batch_size, fft_size] - complex input data
     * @note Output shape: [batch_size, fft_size] - complex FFT results
     * @note All data pointers must be valid GPU memory addresses
     * @see launch_cufft_fft_kernel for the underlying CUDA implementation
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
     * serializing the model. Serializes FFT size and direction.
     *
     * @return Pointer to field collection containing m_fft_size_ and m_direction_
     */
    [[nodiscard]] nvinfer1::PluginFieldCollection const *getFieldsToSerialize() noexcept override;

private:
    std::int32_t m_fft_size_{FftTrtPluginParams::DEFAULT_FFT_SIZE}; //!< Size of the FFT to compute
    std::int32_t m_direction_{}; //!< FFT direction (0=forward, 1=inverse)

    // Serialization support - must be mutable for getFieldsToSerialize()
    mutable std::vector<nvinfer1::PluginField>
            m_serialization_fields_; //!< Fields for serialization
    mutable nvinfer1::PluginFieldCollection
            m_serialization_collection_; //!< Field collection for serialization

    // cuFFTDx configuration parameters
    std::string m_precision_; //!< FFT precision (float, double)
    std::string m_fft_type_;  //!< FFT type (c2c, r2c, c2r)
    std::int32_t m_ffts_per_block_{
            FftTrtPluginParams::DEFAULT_FFTS_PER_BLOCK}; //!< Number of FFTs per CUDA block
    std::int32_t m_elements_per_thread_{
            FftTrtPluginParams::DEFAULT_ELEMENTS_PER_THREAD}; //!< Number of elements per thread
};

/**
 * @brief Plugin Creator for FftTrtPlugin
 *
 * This class handles the creation and configuration of FftTrtPlugin
 * instances. It implements the TensorRT plugin creator interface and manages
 * plugin field collection for parameter configuration.
 *
 * The creator extracts FFT size parameters from the field collection
 * and ensures proper plugin initialization with the correct configuration.
 *
 * @see FftTrtPlugin for the main plugin implementation
 * @see IPluginCreatorV3One for the base creator interface
 */
class __attribute__((visibility("default"))) FftTrtPluginCreator final
        : public TrtPluginCreatorBase<FftTrtPlugin> {
public:
    /**
     * @brief Constructor with required namespace
     *
     * Initializes the creator with the specified namespace
     * and prepares the plugin field collection.
     *
     * @param[in] name_space Plugin namespace
     */
    explicit FftTrtPluginCreator(std::string_view name_space);

    /**
     * @brief Virtual destructor
     */
    ~FftTrtPluginCreator() override = default;

    // Delete copy and move operations
    FftTrtPluginCreator(const FftTrtPluginCreator &) = delete;
    FftTrtPluginCreator &operator=(const FftTrtPluginCreator &) = delete;
    FftTrtPluginCreator(FftTrtPluginCreator &&) = delete;
    FftTrtPluginCreator &operator=(FftTrtPluginCreator &&) = delete;

    /**
     * @brief Creates a new plugin instance
     *
     * Instantiates a FftTrtPlugin with the specified name
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
    std::int32_t m_stored_fft_size_{
            FftTrtPluginParams::DEFAULT_FFT_SIZE}; //!< Stored FFT size from field collection
    std::int32_t m_stored_direction_{}; //!< Stored FFT direction from field collection (0=forward,
                                        //!< 1=inverse)
};

/**
 * @brief CUDA kernel launcher for FFT computation (batched)
 *
 * This function launches the CUDA kernel that performs FFT computation
 * using cuFFTDx library for multiple input signals in parallel.
 *
 * @param[in] input_real Pointer to input real component array (GPU memory, batch_size * fft_size
 * elements)
 * @param[in] input_imag Pointer to input imaginary component array (GPU memory, batch_size *
 * fft_size elements)
 * @param[in] fft_size Size of the FFT to compute
 * @param[in] batch_size Number of input signals to process in parallel
 * @param[out] output_real Pointer to output real component buffer (GPU memory, batch_size *
 * fft_size)
 * @param[out] output_imag Pointer to output imaginary component buffer (GPU memory, batch_size *
 * fft_size)
 * @param[in] workspace Workspace memory for FFT computation
 * @param[in] stream CUDA stream for asynchronous execution
 * @param[in] precision Precision mode (0=float, 1=double)
 * @param[in] fft_type FFT type (0=C2C, 1=R2C, 2=C2R)
 * @param[in] direction FFT direction (0=forward, 1=inverse)
 * @param[in] ffts_per_block Number of FFTs per block
 * @param[in] elements_per_thread Number of elements processed per thread
 *
 * @see cufft_kernel for the actual CUDA kernel implementation
 */
void launch_fft_kernel(
        const float *input_real,
        const float *input_imag,
        std::int32_t fft_size,
        std::int32_t batch_size,
        float *output_real,
        float *output_imag,
        void *workspace,
        cudaStream_t stream,
        std::int32_t precision = 0,
        std::int32_t fft_type = 0,
        std::int32_t direction = 0,
        std::int32_t ffts_per_block = 1,
        std::int32_t elements_per_thread = FftTrtPluginParams::DEFAULT_ELEMENTS_PER_THREAD);

} // namespace ran::trt_plugin

#endif // RAN_FFT_TRT_PLUGIN_HPP
