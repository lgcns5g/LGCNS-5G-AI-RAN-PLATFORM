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

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <NvInfer.h>
#include <driver_types.h>
#include <quill/LogMacros.h>

#include <cuda_runtime_api.h>

#include "log/rt_log_macros.hpp"
#include "tensorrt/mlir_trt_engine.hpp"
#include "tensorrt/trt_engine_interface.hpp"
#include "tensorrt/trt_engine_interfaces.hpp"
#include "tensorrt/trt_engine_params.hpp"
#include "tensorrt/trt_null_pre_post_enqueue.hpp"
#include "utils/core_log.hpp"
#include "utils/error_macros.hpp"
#include "utils/errors.hpp"

namespace framework::tensorrt {

namespace {

/**
 * @brief Compute row-major strides from dimensions
 *
 * @details Computes strides for row-major (C-style) layout where the last
 * dimension has stride 1 and each preceding stride is the product of all
 * following dimensions. For shape [16, 3, 224, 224], the strides are
 * [150528, 50176, 224, 1].
 *
 * @param[in] dims Tensor dimensions
 * @param[in] rank Number of dimensions
 * @return Row-major strides
 */
std::array<std::size_t, MLIRTensorParams::MAX_TENSOR_RANK> compute_row_major_strides(
        const std::array<std::size_t, MLIRTensorParams::MAX_TENSOR_RANK> &dims, std::size_t rank) {
    std::array<std::size_t, MLIRTensorParams::MAX_TENSOR_RANK> strides{};

    // Rank is already validated in MLIRTrtEngine constructor
    // This assert provides debug-time safety check only
    assert(rank <= MLIRTensorParams::MAX_TENSOR_RANK);

    // Scalars have no strides
    if (rank == 0) {
        return strides; // Already zero-initialized
    }

    const std::span<std::size_t> strides_view(strides.data(), rank);
    const std::span<const std::size_t> dims_view(dims.data(), rank);

    // Last stride is always 1 for row-major
    strides_view[rank - 1] = 1;

    // Compute strides from right to left (suffix product)
    for (std::size_t i = rank - 1; i > 0; --i) {
        strides_view[i - 1] = strides_view[i] * dims_view[i];
    }

    return strides;
}

/**
 * @brief Validate tensor parameters and auto-compute strides if needed
 *
 * @param[in,out] tensor Tensor parameters to validate and update
 * @param[in] tensor_type Description of tensor type for error messages (e.g.,
 * "Input" or "Output")
 * @throws std::invalid_argument If tensor name is empty or rank is invalid
 */
void validate_and_compute_strides(MLIRTensorParams &tensor, const char *tensor_type) {
    if (tensor.name.empty()) {
        FRAMEWORK_NV_THROW(
                std::invalid_argument, std::string(tensor_type) + " tensor name cannot be empty");
    }

    if (tensor.rank > MLIRTensorParams::MAX_TENSOR_RANK) {
        FRAMEWORK_NV_THROW(
                std::invalid_argument,
                std::string(tensor_type) + " tensor rank must be between 0 (scalar) and " +
                        std::to_string(MLIRTensorParams::MAX_TENSOR_RANK));
    }

    // Scalars have no strides, skip auto-computation
    if (tensor.rank == 0) {
        return;
    }

    // Auto-compute strides if not provided (last stride == 0)
    const std::span<const std::size_t> strides_span(tensor.strides.data(), tensor.rank);
    if (strides_span[tensor.rank - 1] == 0) {
        tensor.strides = compute_row_major_strides(tensor.dims, tensor.rank);
        RT_LOGC_DEBUG(
                utils::Core::CoreNvApi,
                "Auto-computed row-major strides for {} tensor '{}'",
                tensor_type,
                tensor.name);
    }
}

} // anonymous namespace

MLIRTrtEngine::MLIRTrtEngine(
        std::vector<MLIRTensorParams> input_tensor_prms,
        std::vector<MLIRTensorParams> output_tensor_prms,
        std::unique_ptr<ITrtEngine> tensorrt_runtime,
        std::unique_ptr<IPrePostTrtEngEnqueue> pre_post_trt_eng_enqueue)
        : input_tensor_prms_(std::move(input_tensor_prms)),
          output_tensor_prms_(std::move(output_tensor_prms)),
          tensorrt_runtime_(std::move(tensorrt_runtime)) {

    // Validate required runtime
    if (!tensorrt_runtime_) {
        FRAMEWORK_NV_THROW(std::invalid_argument, "TensorRT runtime cannot be null");
    }

    // Setup optional components with null objects if not provided
    if (!pre_post_trt_eng_enqueue) {
        pre_post_trt_eng_enqueue_ = std::make_unique<NullPrePostTrtEngEnqueue>();
    } else {
        pre_post_trt_eng_enqueue_ = std::move(pre_post_trt_eng_enqueue);
    }

    // Validate tensor parameters
    if (input_tensor_prms_.empty()) {
        FRAMEWORK_NV_THROW(std::invalid_argument, "Input tensor parameters cannot be empty");
    }

    if (output_tensor_prms_.empty()) {
        FRAMEWORK_NV_THROW(std::invalid_argument, "Output tensor parameters cannot be empty");
    }

    // Validate and auto-compute strides for input tensors
    for (auto &tensor : input_tensor_prms_) {
        validate_and_compute_strides(tensor, "Input");
    }

    // Validate and auto-compute strides for output tensors
    for (auto &tensor : output_tensor_prms_) {
        validate_and_compute_strides(tensor, "Output");
    }

    RT_LOGC_INFO(
            utils::Core::CoreNvApi,
            "MLIRTrtEngine initialized with {} inputs, {} outputs",
            input_tensor_prms_.size(),
            output_tensor_prms_.size());
}

utils::NvErrc MLIRTrtEngine::warmup(cudaStream_t cu_stream) {
    RT_LOGC_DEBUG(utils::Core::CoreNvApi, "Starting MLIRTrtEngine warmup");

    // Validate that setup was called first (buffers should be populated)
    if (input_buffers_.empty() || output_buffers_.empty() ||
        input_buffers_.size() != input_tensor_prms_.size() ||
        output_buffers_.size() != output_tensor_prms_.size()) {
        RT_LOGC_ERROR(utils::Core::CoreNvApi, "warmup() requires setup() to be called first");
        return utils::NvErrc::InternalError;
    }

    // Configure input tensors
    for (std::size_t i = 0; i < input_buffers_.size(); ++i) {
        if (const auto result = configure_input_tensor(i); result != utils::NvErrc::Success) {
            return result;
        }
    }

    // Configure output tensors
    for (std::size_t i = 0; i < output_buffers_.size(); ++i) {
        if (const auto result = configure_output_tensor(i); result != utils::NvErrc::Success) {
            return result;
        }
    }

    // Execute warmup
    if (const auto ret = pre_post_trt_eng_enqueue_->pre_enqueue(cu_stream);
        ret != utils::NvErrc::Success) {
        RT_LOGC_ERROR(utils::Core::CoreNvApi, "Failed to pre-enqueue during warmup");
        return ret;
    }

    if (const auto result = tensorrt_runtime_->enqueue_inference(cu_stream);
        result != utils::NvErrc::Success) {
        RT_LOGC_ERROR(utils::Core::CoreNvApi, "Failed to enqueue inference during warmup");
        return result;
    }

    if (const auto ret = pre_post_trt_eng_enqueue_->post_enqueue(cu_stream);
        ret != utils::NvErrc::Success) {
        RT_LOGC_ERROR(utils::Core::CoreNvApi, "Failed to post-enqueue during warmup");
        return ret;
    }

    // Synchronize to ensure warmup is complete
    FRAMEWORK_CUDA_RUNTIME_CHECK_THROW(cudaStreamSynchronize(cu_stream));

    RT_LOGC_DEBUG(utils::Core::CoreNvApi, "MLIRTrtEngine warmup completed successfully");
    return utils::NvErrc::Success;
}

utils::NvErrc MLIRTrtEngine::setup(
        const std::vector<void *> &input_buffers, const std::vector<void *> &output_buffers) {
    RT_LOGC_DEBUG(
            utils::Core::CoreNvApi,
            "Setting up MLIRTrtEngine with {} inputs, {} outputs",
            input_buffers.size(),
            output_buffers.size());

    // Validate buffer counts match tensor parameter counts
    if (input_buffers.size() != input_tensor_prms_.size()) {
        RT_LOGC_ERROR(
                utils::Core::CoreNvApi,
                "Input buffer count ({}) does not match tensor parameter count ({})",
                input_buffers.size(),
                input_tensor_prms_.size());
        return utils::NvErrc::InvalidArgument;
    }

    if (output_buffers.size() != output_tensor_prms_.size()) {
        RT_LOGC_ERROR(
                utils::Core::CoreNvApi,
                "Output buffer count ({}) does not match tensor parameter count ({})",
                output_buffers.size(),
                output_tensor_prms_.size());
        return utils::NvErrc::InvalidArgument;
    }

    // Cache buffer pointers
    input_buffers_ = input_buffers;
    output_buffers_ = output_buffers;

    RT_LOGC_DEBUG(utils::Core::CoreNvApi, "MLIRTrtEngine setup completed successfully");
    return utils::NvErrc::Success;
}

utils::NvErrc MLIRTrtEngine::run(cudaStream_t cu_stream) const {
    RT_LOGC_DEBUG(utils::Core::CoreNvApi, "Running MLIRTrtEngine");

    // Validate setup was called (buffers should be populated)
    if (input_buffers_.empty() || output_buffers_.empty() ||
        input_buffers_.size() != input_tensor_prms_.size() ||
        output_buffers_.size() != output_tensor_prms_.size()) {
        RT_LOGC_ERROR(utils::Core::CoreNvApi, "Engine not setup - call setup() first");
        return utils::NvErrc::InternalError;
    }

    // Configure input tensors
    for (std::size_t i = 0; i < input_buffers_.size(); ++i) {
        if (const auto result = configure_input_tensor(i); result != utils::NvErrc::Success) {
            return result;
        }
    }

    // Configure output tensors
    for (std::size_t i = 0; i < output_buffers_.size(); ++i) {
        if (const auto result = configure_output_tensor(i); result != utils::NvErrc::Success) {
            return result;
        }
    }

    // Verify all input dimensions are specified
    if (!tensorrt_runtime_->all_input_dimensions_specified()) {
        RT_LOGC_ERROR(utils::Core::CoreNvApi, "Not all input dimensions specified");
        return utils::NvErrc::InternalError;
    }

    // In Run, we do not need pre/post enqueue operations since this is expensive
    // and we do not need to capture the graph in this case.

    if (const auto result = tensorrt_runtime_->enqueue_inference(cu_stream);
        result != utils::NvErrc::Success) {
        RT_LOGC_ERROR(utils::Core::CoreNvApi, "Failed to enqueue");
        return result;
    }

    RT_LOGC_DEBUG(utils::Core::CoreNvApi, "MLIRTrtEngine completed successfully");
    return utils::NvErrc::Success;
}

utils::NvErrc MLIRTrtEngine::configure_input_tensor(const std::size_t tensor_index) const {
    const auto &tensor_params = input_tensor_prms_[tensor_index];
    const std::string &tensor_name = tensor_params.name;

    // Set tensor address using direct data buffer pointer
    if (const auto result =
                tensorrt_runtime_->set_tensor_address(tensor_name, input_buffers_[tensor_index]);
        result != utils::NvErrc::Success) {
        RT_LOGC_ERROR(utils::Core::CoreNvApi, "Failed to set tensor address for '{}'", tensor_name);
        return result;
    }

    // Set input shape from tensor parameters
    nvinfer1::Dims dims;
    dims.nbDims = static_cast<int32_t>(tensor_params.rank);

    // Ensure we don't exceed TensorRT's maximum dimensions
    if (tensor_params.rank > nvinfer1::Dims::MAX_DIMS) {
        RT_LOGC_ERROR(
                utils::Core::CoreNvApi,
                "Tensor rank {} exceeds TensorRT maximum dimensions {}",
                tensor_params.rank,
                nvinfer1::Dims::MAX_DIMS);
        return utils::NvErrc::InvalidArgument;
    }

    // Copy dims from tensor parameters
    std::transform(
            tensor_params.dims.begin(),
            std::next(tensor_params.dims.begin(), static_cast<std::ptrdiff_t>(tensor_params.rank)),
            std::begin(dims.d),
            [](const std::size_t dim) { return static_cast<int32_t>(dim); });

    if (const auto result = tensorrt_runtime_->set_input_shape(tensor_name, dims);
        result != utils::NvErrc::Success) {
        RT_LOGC_ERROR(utils::Core::CoreNvApi, "Failed to set input shape for '{}'", tensor_name);
        return result;
    }

    RT_LOGC_DEBUG(
            utils::Core::CoreNvApi,
            "Configured input tensor '{}' with rank {} dimensions",
            tensor_name,
            tensor_params.rank);
    return utils::NvErrc::Success;
}

utils::NvErrc MLIRTrtEngine::configure_output_tensor(const std::size_t tensor_index) const {
    const auto &tensor_params = output_tensor_prms_[tensor_index];
    const std::string &tensor_name = tensor_params.name;

    // Set tensor address using direct data buffer pointer
    if (const auto result =
                tensorrt_runtime_->set_tensor_address(tensor_name, output_buffers_[tensor_index]);
        result != utils::NvErrc::Success) {
        RT_LOGC_ERROR(utils::Core::CoreNvApi, "Failed to set tensor address for '{}'", tensor_name);
        return result;
    }

    RT_LOGC_DEBUG(utils::Core::CoreNvApi, "Configured output tensor '{}'", tensor_name);
    return utils::NvErrc::Success;
}

} // namespace framework::tensorrt
