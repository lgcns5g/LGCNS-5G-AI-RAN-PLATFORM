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

#include <cstdio>

#include "sample_module_b_kernel.cuh"

namespace framework::pipelines::samples {

__global__ void sample_module_b_kernel(
        const SampleModuleBStaticKernelParams *static_params,
        const SampleModuleBDynamicKernelParams *dynamic_params) {
    const std::size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (idx < static_params->size) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        static_params->output[idx] =
                // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
                fmaxf(0.0F, dynamic_params->input[idx]);
    }
}

} // namespace framework::pipelines::samples
