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

#ifndef FRAMEWORK_ITRT_ENGINE_HPP
#define FRAMEWORK_ITRT_ENGINE_HPP

#include <cstdint>
#include <span>
#include <vector>

#include <NvInfer.h>
#include <driver_types.h>

#include "utils/errors.hpp"

namespace framework::tensorrt {

/**
 * @brief Abstract interface for TensorRT engine operations
 *
 * This interface abstracts the TensorRT components (IRuntime, ICudaEngine,
 * IExecutionContext) into a unified API for engine initialization,
 * configuration, and execution.
 */
class ITrtEngine {
public:
    ITrtEngine() = default;
    virtual ~ITrtEngine() = default;

    /**
     * Copy constructor
     * @param[in] engine Source object to copy from
     */
    ITrtEngine(const ITrtEngine &engine) = default;

    /**
     * Copy assignment operator
     * @param[in] engine Source object to copy from
     * @return Reference to this object
     */
    ITrtEngine &operator=(const ITrtEngine &engine) = default;

    /**
     * Move constructor
     * @param[in] engine Source object to move from
     */
    ITrtEngine(ITrtEngine &&engine) = default;

    /**
     * Move assignment operator
     * @param[in] engine Source object to move from
     * @return Reference to this object
     */
    ITrtEngine &operator=(ITrtEngine &&engine) = default;

    /**
     * @brief Set the shape of an input tensor
     * @param[in] tensor_name Name of the input tensor
     * @param[in] dims Dimensions to set for the tensor
     * @return utils::NvErrc::Success on success, error code on failure
     */
    [[nodiscard]]
    virtual utils::NvErrc
    set_input_shape(const std::string_view tensor_name, const nvinfer1::Dims &dims) = 0;

    /**
     * @brief Set the memory address for a tensor
     * @param[in] tensor_name Name of the tensor
     * @param[in] address Memory address to associate with the tensor
     * @return utils::NvErrc::Success on success, error code on failure
     */
    [[nodiscard]]
    virtual utils::NvErrc set_tensor_address(const std::string_view tensor_name, void *address) = 0;

    /**
     * @brief Execute inference asynchronously
     * @param[in] cu_stream CUDA stream for asynchronous execution
     * @return utils::NvErrc::Success on success, error code on failure
     */
    [[nodiscard]]
    virtual utils::NvErrc enqueue_inference(cudaStream_t cu_stream) = 0;

    /**
     * @brief Check if all input dimensions have been specified
     * @return true if all input dimensions are specified, false otherwise
     */
    [[nodiscard]]
    virtual bool all_input_dimensions_specified() const = 0;
};

/**
 * @brief Null object implementation of ITrtEngine
 *
 * Provides a default implementation that returns appropriate error codes
 * for all TensorRT engine operations.
 */
class NullTrtEngine final : public ITrtEngine {
public:
    /**
     * @brief Set the shape of an input tensor
     * @param[in] tensor_name Name of the input tensor
     * @param[in] dims Dimensions to set for the tensor
     * @return utils::NvErrc::NotSupported
     */
    [[nodiscard]]
    utils::NvErrc set_input_shape(
            [[maybe_unused]] const std::string_view tensor_name,
            [[maybe_unused]] const nvinfer1::Dims &dims) final {
        return utils::NvErrc::NotSupported;
    }

    /**
     * @brief Set the memory address for a tensor
     * @param[in] tensor_name Name of the tensor
     * @param[in] address Memory address to associate with the tensor
     * @return utils::NvErrc::NotSupported
     */
    [[nodiscard]]
    utils::NvErrc set_tensor_address(
            [[maybe_unused]] const std::string_view tensor_name,
            [[maybe_unused]] void *address) final {
        return utils::NvErrc::NotSupported;
    }

    /**
     * @brief Execute inference asynchronously
     * @param[in] cu_stream CUDA stream for asynchronous execution
     * @return utils::NvErrc::NotSupported
     */
    [[nodiscard]]
    utils::NvErrc enqueue_inference([[maybe_unused]] cudaStream_t cu_stream) final {
        return utils::NvErrc::NotSupported;
    }

    /**
     * @brief Check if all input dimensions have been specified
     * @return false (null implementation)
     */
    [[nodiscard]]
    bool all_input_dimensions_specified() const final {
        return false;
    }
};

} // namespace framework::tensorrt

#endif // FRAMEWORK_ITRT_ENGINE_HPP
