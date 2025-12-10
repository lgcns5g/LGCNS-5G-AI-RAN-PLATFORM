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

#ifndef FRAMEWORK_TRT_ENGINE_HPP
#define FRAMEWORK_TRT_ENGINE_HPP

#include <filesystem>
#include <memory>
#include <span>

#include <NvInfer.h>

#include "tensorrt/trt_engine_interface.hpp"
#include "tensorrt/trt_engine_logger.hpp"

namespace framework::tensorrt {

/**
 * @brief Concrete implementation of ITrtEngine using NVIDIA TensorRT
 *
 * This class provides the actual TensorRT implementation for engine
 * operations, wrapping the TensorRT IRuntime, ICudaEngine, and
 * IExecutionContext components.
 */
class TrtEngine final : public ITrtEngine {
public:
    /**
     * @brief Initialize TensorRT engine from serialized engine data
     * @param[in] engine_data Span containing serialized engine binary data
     * @param[in] logger TensorRT logger instance for engine messages
     * @throws std::runtime_error on initialization failure
     */
    TrtEngine(std::span<const std::byte> engine_data, nvinfer1::ILogger &logger);

    /**
     * @brief Initialize TensorRT engine from engine file
     * @param[in] engine_file_path Path to the serialized engine file
     * @param[in] logger TensorRT logger instance for engine messages
     * @throws std::runtime_error on file read or initialization failure
     */
    TrtEngine(const std::filesystem::path &engine_file_path, nvinfer1::ILogger &logger);

    ~TrtEngine() final = default;
    TrtEngine(const TrtEngine &engine) = delete;
    TrtEngine &operator=(const TrtEngine &engine) = delete;
    TrtEngine(TrtEngine &&engine) = delete;
    TrtEngine &operator=(TrtEngine &&engine) = delete;

    /**
     * @brief Set the shape of an input tensor
     * @param[in] tensor_name Name of the input tensor
     * @param[in] dims Dimensions to set for the tensor
     * @return utils::NvErrc::Success on success, error code on failure
     */
    [[nodiscard]]
    utils::NvErrc
    set_input_shape(const std::string_view tensor_name, const nvinfer1::Dims &dims) final;

    /**
     * @brief Set the memory address for a tensor
     * @param[in] tensor_name Name of the tensor
     * @param[in] address Memory address to associate with the tensor
     * @return utils::NvErrc::Success on success, error code on failure
     */
    [[nodiscard]]
    utils::NvErrc set_tensor_address(const std::string_view tensor_name, void *address) final;

    /**
     * @brief Execute inference asynchronously
     * @param[in] cu_stream CUDA stream for asynchronous execution
     * @return utils::NvErrc::Success on success, error code on failure
     */
    [[nodiscard]]
    utils::NvErrc enqueue_inference(cudaStream_t cu_stream) final;

    /**
     * @brief Check if all input dimensions have been specified
     * @return true if all input dimensions are specified, false otherwise
     */
    [[nodiscard]]
    bool all_input_dimensions_specified() const final;

private:
    /**
     * @brief Private initialization method called from constructor
     * @param[in] engine_data Span containing serialized engine binary data
     * @param[in] logger TensorRT logger instance for runtime messages
     * @return utils::NvErrc::Success on success, error code on failure
     */
    [[nodiscard]]
    utils::NvErrc initialize(std::span<const std::byte> engine_data, nvinfer1::ILogger &logger);

    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
};

} // namespace framework::tensorrt

#endif // FRAMEWORK_TRT_ENGINE_HPP
