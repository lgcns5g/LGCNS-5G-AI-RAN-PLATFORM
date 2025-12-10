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

#include <cstddef> // for size_t
#include <vector>  // for vector

#include <driver_types.h> // for dim3, CUstream_st

#include <cuda.h> // for cuLaunchKernel, CUr...

#include "pipeline/dynamic_kernel_launch_config.hpp" // for DynamicKernelLaunch...
#include "pipeline/kernel_node_helper.hpp"           // for KernelNodeHelper

namespace framework::pipeline {

void DynamicKernelLaunchConfig::setup_kernel_function(const void *kernel_func) {
    kernel_node_helper_.setup_kernel_function(kernel_func);
}

void DynamicKernelLaunchConfig::setup_kernel_dimensions(
        const dim3 grid_dim, const dim3 block_dim, const std::size_t shared_mem_bytes) {
    kernel_node_helper_.setup_kernel_dimensions(grid_dim, block_dim, shared_mem_bytes);
}

CUresult DynamicKernelLaunchConfig::launch(cudaStream_t stream) const {
    const auto &params = kernel_node_helper_.get_kernel_params();
    return cuLaunchKernel(
            params.func,
            params.gridDimX,
            params.gridDimY,
            params.gridDimZ,
            params.blockDimX,
            params.blockDimY,
            params.blockDimZ,
            params.sharedMemBytes,
            stream,
            params.kernelParams,
            params.extra);
}

void DynamicKernelLaunchConfig::clear_arguments() {
    kernel_args_.clear();
    kernel_params_ptr_ = nullptr;
    kernel_node_helper_.clear_kernel_params();
}

void DynamicKernelLaunchConfig::add_argument(void *arg) { kernel_args_.push_back(arg); }

void DynamicKernelLaunchConfig::finalize_arguments() {
    if (!kernel_args_.empty()) {
        kernel_params_ptr_ = kernel_args_.data();
        kernel_node_helper_.set_kernel_params_ptr(kernel_params_ptr_);
    } else {
        kernel_node_helper_.clear_kernel_params();
    }
}

} // namespace framework::pipeline
