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

#include <cstddef>   // for size_t
#include <limits>    // for numeric_limits
#include <stdexcept> // for invalid_argument

#include <driver_types.h>    // for dim3
#include <quill/LogMacros.h> // for QUILL_LOG_ERROR

#include <cuda.h>         // for CUDA_KERNEL_NODE_PARAMS
#include <cuda_runtime.h> // for cudaGetFuncBySymbol

#include "pipeline/kernel_node_helper.hpp" // for KernelNodeHelper
#include "utils/error_macros.hpp"          // for FRAMEWORK_NV_THROW_IF, AERIA...

namespace framework::pipeline {

void KernelNodeHelper::setup_kernel_function(const void *kernel_func) {
    FRAMEWORK_NV_THROW_IF(
            kernel_func == nullptr, std::invalid_argument, "kernel_func must not be nullptr");
    FRAMEWORK_CUDA_RUNTIME_CHECK_THROW(
            cudaGetFuncBySymbol(&kernel_node_params_driver_.func, kernel_func));
}

void KernelNodeHelper::setup_kernel_dimensions(
        const dim3 grid_dim, const dim3 block_dim, const std::size_t shared_mem_bytes) {
    auto &params = kernel_node_params_driver_;

    params.gridDimX = grid_dim.x;
    params.gridDimY = grid_dim.y;
    params.gridDimZ = grid_dim.z;

    params.blockDimX = block_dim.x;
    params.blockDimY = block_dim.y;
    params.blockDimZ = block_dim.z;

    FRAMEWORK_NV_THROW_IF(
            shared_mem_bytes > std::numeric_limits<unsigned int>::max(),
            std::invalid_argument,
            "Shared memory size exceeds maximum supported value");
    params.sharedMemBytes = static_cast<unsigned int>(shared_mem_bytes);
    params.extra = nullptr; // Explicitly set extra to nullptr for CUDA graphs
}

const CUDA_KERNEL_NODE_PARAMS &KernelNodeHelper::get_kernel_params() const {
    return kernel_node_params_driver_;
}

void KernelNodeHelper::clear_kernel_params() { kernel_node_params_driver_.kernelParams = nullptr; }

void KernelNodeHelper::set_kernel_params_ptr(void **kernel_params_ptr) {
    kernel_node_params_driver_.kernelParams = kernel_params_ptr;
}

} // namespace framework::pipeline
