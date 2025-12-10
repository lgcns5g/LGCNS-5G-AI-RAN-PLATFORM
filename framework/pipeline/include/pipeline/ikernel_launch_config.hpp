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

#ifndef FRAMEWORK_CORE_IKERNEL_LAUNCH_CONFIG_HPP
#define FRAMEWORK_CORE_IKERNEL_LAUNCH_CONFIG_HPP

#include <cuda.h>
#include <cuda_runtime.h>

namespace framework::pipeline {

/**
 * Interface for kernel launch configuration
 *
 * This interface provides a polymorphic approach to kernel launch
 * configuration, supporting variable numbers of kernel arguments and different
 * configuration types while maintaining backward compatibility and clean
 * architecture.
 */
class IKernelLaunchConfig {
public:
    /**
     * Default constructor
     */
    IKernelLaunchConfig() = default;

    /**
     * Virtual destructor
     */
    virtual ~IKernelLaunchConfig() = default;

    /**
     * Copy constructor
     */
    IKernelLaunchConfig(const IKernelLaunchConfig &) = default;

    /**
     * Move constructor
     */
    IKernelLaunchConfig(IKernelLaunchConfig &&) = default;

    /**
     * Copy assignment operator
     * @return Reference to this object
     */
    IKernelLaunchConfig &operator=(const IKernelLaunchConfig &) = default;

    /**
     * Move assignment operator
     * @return Reference to this object
     */
    IKernelLaunchConfig &operator=(IKernelLaunchConfig &&) = default;

    /**
     * Launch kernel using the configured parameters
     *
     * @param[in] stream CUDA stream for kernel execution
     * @return CUresult indicating success (CUDA_SUCCESS) or failure
     */
    [[nodiscard]]
    virtual CUresult launch(cudaStream_t stream) const = 0;

    /**
     * Setup kernel function pointer
     *
     * @param[in] kernel_func Pointer to the kernel function
     */
    virtual void setup_kernel_function(const void *kernel_func) = 0;

    /**
     * Setup kernel dimensions and shared memory
     *
     * @param[in] grid_dim Grid dimensions
     * @param[in] block_dim Block dimensions
     * @param[in] shared_mem_bytes Shared memory size in bytes (default: 0)
     */
    virtual void setup_kernel_dimensions(
            const dim3 grid_dim, const dim3 block_dim, const std::size_t shared_mem_bytes = 0) = 0;

    /**
     * Setup kernel arguments using variadic template
     *
     * This method provides a compile-time interface for setting up kernel
     * arguments. It clears existing arguments, adds each argument, and finalizes
     * the configuration.
     *
     * @param[in] args Variadic arguments to pass to the kernel (must be pointers)
     */
    template <typename... Args>
        requires(std::is_pointer_v<std::decay_t<Args>> && ...)
    void setup_kernel_arguments(Args &&...args) {
        clear_arguments();
        // NOLINTNEXTLINE(bugprone-multi-level-implicit-pointer-conversion)
        (add_argument(std::forward<Args>(args)), ...);
        finalize_arguments();
    }

private:
    /**
     * Clear all kernel arguments
     */
    virtual void clear_arguments() = 0;

    /**
     * Add a single kernel argument
     *
     * @param[in] arg Pointer to the argument
     */
    virtual void add_argument(void *arg) = 0;

    /**
     * Finalize argument setup
     */
    virtual void finalize_arguments() = 0;
};

} // namespace framework::pipeline

#endif // FRAMEWORK_CORE_IKERNEL_LAUNCH_CONFIG_HPP
