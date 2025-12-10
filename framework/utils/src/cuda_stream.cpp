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

#include <format>    // for std::format
#include <stdexcept> // for std::runtime_error
#include <string>
#include <tuple>   // for std::ignore
#include <utility> // for std::exchange

#include <driver_types.h>
#include <quill/LogMacros.h>

#include <cuda_runtime.h>

#include "log/rt_log_macros.hpp" // for RT_LOGC_ERROR
#include "utils/core_log.hpp"    // for Core::CoreCudaRuntime
#include "utils/cuda_stream.hpp"

namespace framework::utils {

CudaStream::CudaStream() {
    const cudaError_t result = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
    if (result != cudaSuccess) {
        const std::string error_msg =
                std::format("Failed to create CUDA stream: {}", cudaGetErrorString(result));
        RT_LOGC_ERROR(Core::CoreCudaRuntime, "{}", error_msg);
        throw std::runtime_error(error_msg);
    }
}

CudaStream::CudaStream(CudaStream &&other) noexcept
        : stream_(std::exchange(other.stream_, nullptr)) {}

CudaStream &CudaStream::operator=(CudaStream &&other) noexcept {
    if (this == &other) {
        return *this;
    }

    // Destroy current stream if we have one
    if (stream_ != nullptr) {
        std::ignore = synchronize();
        if (const cudaError_t result = cudaStreamDestroy(stream_); result != cudaSuccess) {
            RT_LOGC_ERROR(
                    Core::CoreCudaRuntime,
                    "Failed to destroy CUDA stream during move: {}",
                    cudaGetErrorString(result));
        }
    }

    // Transfer ownership using std::exchange
    stream_ = std::exchange(other.stream_, nullptr);
    return *this;
}

CudaStream::~CudaStream() {
    if (stream_ == nullptr) {
        return;
    }

    // Synchronize stream before destroying to ensure all operations complete
    std::ignore = synchronize();

    // Destroy the stream
    if (const cudaError_t destroy_result = cudaStreamDestroy(stream_);
        destroy_result != cudaSuccess) {
        RT_LOGC_ERROR(
                Core::CoreCudaRuntime,
                "Failed to destroy CUDA stream: {}",
                cudaGetErrorString(destroy_result));
    }
}

bool CudaStream::synchronize() const noexcept {
    const cudaError_t result = cudaStreamSynchronize(stream_);
    if (result != cudaSuccess) {
        RT_LOGC_ERROR(
                Core::CoreCudaRuntime,
                "Failed to synchronize CUDA stream: {}",
                cudaGetErrorString(result));
        return false;
    }
    return true;
}

} // namespace framework::utils
