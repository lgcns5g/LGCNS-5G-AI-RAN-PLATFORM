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

#include <stdexcept>

#include <driver_types.h>
#include <quill/LogMacros.h>

#include <cuda_runtime_api.h>

#include "pipeline/kernel_descriptor_accessor.hpp"
#include "pipeline/types.hpp"
#include "utils/error_macros.hpp"

namespace framework::pipeline {

KernelDescriptorAccessor::KernelDescriptorAccessor(const ModuleMemorySlice &memory_slice)
        : memory_slice_(memory_slice) {}

void KernelDescriptorAccessor::copy_static_descriptors_to_device(cudaStream_t stream) const {
    if (memory_slice_.static_kernel_descriptor_bytes > 0) {
        if (memory_slice_.static_kernel_descriptor_gpu_ptr == nullptr ||
            memory_slice_.static_kernel_descriptor_cpu_ptr == nullptr) {
            FRAMEWORK_NV_THROW(
                    std::runtime_error, "Null pointer in static descriptor memory slice");
        }
        FRAMEWORK_CUDA_RUNTIME_CHECK_THROW(cudaMemcpyAsync(
                memory_slice_.static_kernel_descriptor_gpu_ptr,
                memory_slice_.static_kernel_descriptor_cpu_ptr,
                memory_slice_.static_kernel_descriptor_bytes,
                cudaMemcpyHostToDevice,
                stream));
    }
}

void KernelDescriptorAccessor::copy_dynamic_descriptors_to_device(cudaStream_t stream) const {
    if (memory_slice_.dynamic_kernel_descriptor_bytes > 0) {
        if (memory_slice_.dynamic_kernel_descriptor_gpu_ptr == nullptr ||
            memory_slice_.dynamic_kernel_descriptor_cpu_ptr == nullptr) {
            FRAMEWORK_NV_THROW(
                    std::runtime_error, "Null pointer in dynamic descriptor memory slice");
        }
        FRAMEWORK_CUDA_RUNTIME_CHECK_THROW(cudaMemcpyAsync(
                memory_slice_.dynamic_kernel_descriptor_gpu_ptr,
                memory_slice_.dynamic_kernel_descriptor_cpu_ptr,
                memory_slice_.dynamic_kernel_descriptor_bytes,
                cudaMemcpyHostToDevice,
                stream));
    }
}

} // namespace framework::pipeline
