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

#include <cuda_runtime.h>

/**
 * @brief CUDA error checking macro
 *
 * Checks CUDA function calls and throws std::runtime_error if an error occurs.
 */
#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t error = call;                                                                  \
        if (error != cudaSuccess) {                                                                \
            throw std::runtime_error(cudaGetErrorString(error));                                   \
        }                                                                                          \
    } while (0)

namespace ran::trt_plugin {

/**
 * Sequential sum kernel - intentionally non-parallelizable
 *
 * This kernel computes: output[i] = input[i] + output[i-1]
 * Each element depends on the previous, making it difficult to parallelize.
 *
 * We use a single thread to perform the sequential computation to demonstrate
 * a workload that doesn't benefit from GPU parallelization.
 */
__global__ void sequential_sum_kernel(const float *input, float *output, int64_t size) {
    // Only thread 0 in block 0 does the work
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        if (size > 0) {
            output[0] = input[0];

            // Sequential computation - each element depends on previous
            for (int64_t i = 1; i < size; ++i) {
                output[i] = input[i] + output[i - 1];
            }
        }
    }
}

/**
 * @brief Launches the sequential sum kernel
 *
 * Computes cumulative sum where output[i] = input[i] + output[i-1].
 * This kernel intentionally uses a single thread to demonstrate
 * a non-parallelizable workload.
 *
 * @param[in] input Pointer to input data array (GPU memory)
 * @param[out] output Pointer to output data array (GPU memory)
 * @param[in] size Number of elements to process
 * @param[in] stream CUDA stream for asynchronous execution
 *
 * @throws std::invalid_argument if size is not positive
 * @throws std::runtime_error if CUDA kernel launch fails
 *
 * @see sequential_sum_kernel for the kernel implementation
 */
void launch_sequential_sum_kernel(
        const float *input, float *output, int64_t size, cudaStream_t stream) {

    if (size <= 0) {
        throw std::invalid_argument("Sequential sum size must be positive");
    }

    // Use single block, single thread for truly sequential operation
    // This demonstrates a workload that cannot be effectively parallelized
    dim3 grid(1);
    dim3 block(1);

    sequential_sum_kernel<<<grid, block, 0, stream>>>(input, output, size);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ran::trt_plugin
