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

#ifndef FRAMEWORK_CORE_DYNAMIC_KERNEL_LAUNCH_CONFIG_HPP
#define FRAMEWORK_CORE_DYNAMIC_KERNEL_LAUNCH_CONFIG_HPP

#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "pipeline/ikernel_launch_config.hpp"
#include "pipeline/kernel_node_helper.hpp"

namespace framework::pipeline {

/**
 * Dynamic kernel launch configuration supporting arbitrary number of arguments
 *
 * This class extends the base kernel launch configuration to support kernels
 * with variable numbers of arguments. Use this when the number of kernel
 * arguments is not known at compile time or varies significantly. For known
 * fixed sizes, prefer KernelLaunchConfig<N>.
 */
class DynamicKernelLaunchConfig : public IKernelLaunchConfig {
public:
    /**
     * Setup kernel function pointer
     *
     * @param[in] kernel_func Pointer to the kernel function
     * @throws std::runtime_error if cudaGetFuncBySymbol fails
     */
    void setup_kernel_function(const void *kernel_func) override;

    /**
     * Setup kernel dimensions and shared memory
     *
     * @param[in] grid_dim Grid dimensions
     * @param[in] block_dim Block dimensions
     * @param[in] shared_mem_bytes Shared memory size in bytes (default: 0)
     * @throws std::invalid_argument if shared memory size exceeds maximum
     */
    void setup_kernel_dimensions(
            const dim3 grid_dim,
            const dim3 block_dim,
            const std::size_t shared_mem_bytes = 0) override;

    /**
     * Launch kernel using the configured parameters
     *
     * @param[in] stream CUDA stream for kernel execution
     * @return CUresult indicating success (CUDA_SUCCESS) or failure
     */
    CUresult launch(cudaStream_t stream) const override;

private:
    /**
     * Clear all kernel arguments
     */
    void clear_arguments() override;

    /**
     * Add a single kernel argument
     *
     * @param[in] arg Pointer to the argument
     */
    void add_argument(void *arg) override;

    /**
     * Finalize argument setup
     */
    void finalize_arguments() override;
    /**
     * Vector of kernel argument pointers
     */
    std::vector<void *> kernel_args_;

    /**
     * Pointer to kernel arguments for CUDA driver API
     */
    void **kernel_params_ptr_{nullptr};

    /**
     * Helper for managing CUDA kernel node parameters
     */
    KernelNodeHelper kernel_node_helper_{};
};

} // namespace framework::pipeline

#endif // FRAMEWORK_CORE_DYNAMIC_KERNEL_LAUNCH_CONFIG_HPP
