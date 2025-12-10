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

#ifndef FRAMEWORK_CORE_KERNEL_NODE_HELPER_HPP
#define FRAMEWORK_CORE_KERNEL_NODE_HELPER_HPP

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>

namespace framework::pipeline {

/**
 * Helper class for managing CUDA kernel node parameters
 *
 * This class encapsulates common CUDA kernel launch configuration logic
 * that can be reused across different kernel launch configurations via
 * composition instead of inheritance.
 */
class KernelNodeHelper final {
public:
    /**
     * Setup kernel function pointer
     *
     * @param[in] kernel_func Pointer to the kernel function
     * @throws std::runtime_error if cudaGetFuncBySymbol fails
     */
    void setup_kernel_function(const void *kernel_func);

    /**
     * Setup kernel dimensions and shared memory
     *
     * @param[in] grid_dim Grid dimensions
     * @param[in] block_dim Block dimensions
     * @param[in] shared_mem_bytes Shared memory size in bytes (default: 0)
     * @throws std::invalid_argument if shared memory size exceeds maximum
     */
    void setup_kernel_dimensions(
            const dim3 grid_dim, const dim3 block_dim, const std::size_t shared_mem_bytes = 0);

    /**
     * Get const reference to kernel node parameters
     *
     * @return Const reference to CUDA_KERNEL_NODE_PARAMS for kernel launch
     */
    [[nodiscard]] const CUDA_KERNEL_NODE_PARAMS &get_kernel_params() const;

    /**
     * Clear kernel node parameters
     */
    void clear_kernel_params();

    /**
     * Set kernel node parameters pointer
     *
     * @note Caller must ensure the pointer is valid for the duration of the
     * kernel launch.
     *
     * @param[in] kernel_params_ptr Pointer to kernel parameters
     */
    void set_kernel_params_ptr(void **kernel_params_ptr);

private:
    /**
     * CUDA kernel node parameters for driver API launch
     */
    CUDA_KERNEL_NODE_PARAMS kernel_node_params_driver_{};
};

} // namespace framework::pipeline

#endif // FRAMEWORK_CORE_KERNEL_NODE_HELPER_HPP
