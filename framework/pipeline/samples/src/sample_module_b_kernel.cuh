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

#ifndef FRAMEWORK_PIPELINES_SAMPLES_MODULE_B_KERNEL_CUH
#define FRAMEWORK_PIPELINES_SAMPLES_MODULE_B_KERNEL_CUH

#include <cstddef>

namespace framework::pipelines::samples {

/**
 * Static kernel parameters for SampleModuleB
 *
 * Contains parameters that don't change between executions.
 */
struct SampleModuleBStaticKernelParams final {
    float *output{nullptr}; //!< Output tensor pointer
    std::size_t size{0};    //!< Number of elements to process
};

/**
 * Dynamic kernel parameters for SampleModuleB
 *
 * Contains parameters that change per tick (input pointer from upstream).
 */
struct SampleModuleBDynamicKernelParams final {
    const float *input{nullptr}; //!< Input tensor pointer
};

/**
 * CUDA kernel for element-wise ReLU activation
 *
 * Applies ReLU(x) = max(0, x) to each element using kernel descriptor pattern.
 *
 * @param[in] static_params Static kernel parameters (output ptr, size)
 * @param[in] dynamic_params Dynamic kernel parameters (input ptr)
 */
__global__ void sample_module_b_kernel(
        const SampleModuleBStaticKernelParams *static_params,
        const SampleModuleBDynamicKernelParams *dynamic_params);

} // namespace framework::pipelines::samples

#endif // FRAMEWORK_PIPELINES_SAMPLES_MODULE_B_KERNEL_CUH
