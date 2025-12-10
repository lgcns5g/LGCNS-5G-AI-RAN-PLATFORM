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

#include <cstddef>    // for byte
#include <cstring>    // for memcpy
#include <filesystem> // for path
#include <fstream>    // for ifstream
#include <memory>     // for unique_ptr
#include <span>       // for span
#include <stdexcept>  // for runtime_error
#include <string>
#include <string_view>
#include <vector> // for vector

#include <NvInfer.h>
#include <driver_types.h>    // for CUstream_st, cudaStream_t
#include <quill/LogMacros.h> // for QUILL_LOG_ERROR

#include <cuda_runtime_api.h>

#include "log/rt_log_macros.hpp"   // for RT_LOGC_ERROR
#include "tensorrt/trt_engine.hpp" // for TrtEngine
#include "utils/core_log.hpp"      // for Core
#include "utils/errors.hpp"        // for utils::NvErrc

namespace framework::tensorrt {

TrtEngine::TrtEngine(std::span<const std::byte> engine_data, nvinfer1::ILogger &logger) {
    if (const auto result = initialize(engine_data, logger); result != utils::NvErrc::Success) {
        throw std::runtime_error("Failed to initialize TensorRT engine");
    }
}

TrtEngine::TrtEngine(const std::filesystem::path &engine_file_path, nvinfer1::ILogger &logger) {
    // Open file in binary mode and seek to end to get file size
    std::ifstream file(engine_file_path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Failed to open engine file: " + engine_file_path.string());
    }

    const std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Read file contents into temporary char buffer
    std::vector<char> temp_buffer(static_cast<std::size_t>(size));
    if (!file.read(temp_buffer.data(), size)) {
        throw std::runtime_error("Failed to read engine file: " + engine_file_path.string());
    }

    // Convert char buffer to byte buffer
    std::vector<std::byte> buffer(static_cast<std::size_t>(size));
    std::memcpy(buffer.data(), temp_buffer.data(), static_cast<std::size_t>(size));

    // Initialize TensorRT engine with loaded engine data
    if (const auto result = initialize(buffer, logger); result != utils::NvErrc::Success) {
        throw std::runtime_error(
                "Failed to initialize TensorRT engine from file: " + engine_file_path.string());
    }
}

utils::NvErrc
TrtEngine::initialize(std::span<const std::byte> engine_data, nvinfer1::ILogger &logger) {
    if (engine_data.empty()) {
        return utils::NvErrc::InvalidArgument;
    }

    // Check CUDA availability before attempting to create TensorRT runtime
    int device_count = 0;
    const cudaError_t cuda_error = cudaGetDeviceCount(&device_count);
    if (cuda_error != cudaSuccess || device_count == 0) {
        RT_LOGC_ERROR(
                utils::Core::CoreNvApi,
                "CUDA not available: {} (device count: {})",
                cudaGetErrorString(cuda_error),
                device_count);
        return utils::NvErrc::InternalError;
    }

    runtime_ = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    if (!runtime_) {
        RT_LOGC_ERROR(utils::Core::CoreNvApi, "Failed to create TensorRT runtime");
        return utils::NvErrc::InternalError;
    }

    engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(
            runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    if (!engine_) {
        RT_LOGC_ERROR(utils::Core::CoreNvApi, "Failed to create TensorRT engine");
        return utils::NvErrc::InternalError;
    }

    context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
    if (!context_) {
        RT_LOGC_ERROR(utils::Core::CoreNvApi, "Failed to create TensorRT execution context");
        return utils::NvErrc::InternalError;
    }

    return utils::NvErrc::Success;
}

utils::NvErrc
TrtEngine::set_input_shape(const std::string_view tensor_name, const nvinfer1::Dims &dims) {
    if (tensor_name.empty()) {
        return utils::NvErrc::InvalidArgument;
    }

    if (!context_->setInputShape(std::string(tensor_name).c_str(), dims)) {
        return utils::NvErrc::InternalError;
    }

    return utils::NvErrc::Success;
}

utils::NvErrc TrtEngine::set_tensor_address(const std::string_view tensor_name, void *address) {
    if (tensor_name.empty() || address == nullptr) {
        return utils::NvErrc::InvalidArgument;
    }

    if (!context_->setTensorAddress(std::string(tensor_name).c_str(), address)) {
        return utils::NvErrc::InternalError;
    }

    return utils::NvErrc::Success;
}

utils::NvErrc TrtEngine::enqueue_inference(cudaStream_t cu_stream) {
    if (!context_->enqueueV3(cu_stream)) {
        return utils::NvErrc::InternalError;
    }

    return utils::NvErrc::Success;
}

bool TrtEngine::all_input_dimensions_specified() const {
    return context_->allInputDimensionsSpecified();
}

} // namespace framework::tensorrt
