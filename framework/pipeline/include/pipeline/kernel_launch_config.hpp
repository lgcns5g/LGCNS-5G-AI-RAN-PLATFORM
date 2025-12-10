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

#ifndef FRAMEWORK_CORE_KERNEL_LAUNCH_CONFIG_HPP
#define FRAMEWORK_CORE_KERNEL_LAUNCH_CONFIG_HPP

#include <array>
#include <cstddef>
#include <format>
#include <stdexcept>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>

#include "pipeline/ikernel_launch_config.hpp"
#include "pipeline/kernel_node_helper.hpp"

namespace framework::pipeline {

/**
 * Template-based kernel launch configuration
 *
 * This class is designed for CUDA Graph compatibility and efficient
 * kernel launching using the CUDA Driver API. It stores pre-configured
 * kernel launch parameters that can be reused across multiple kernel
 * invocations.
 *
 * Usage pattern:
 * 1. During setup(): Configure the CUDA_KERNEL_NODE_PARAMS once
 * 2. During execute(): Launch kernel with pre-configured parameters
 *
 * This approach enables:
 * - CUDA Graph capture and replay
 * - Reduced overhead (configuration done once, not per execution)
 * - Consistent error handling with CUresult
 * - Clean separation between setup and execution phases
 *
 * @tparam NUM_PARAMS Number of kernel parameters (must be > 0)
 */
template <std::size_t NUM_PARAMS> class KernelLaunchConfig : public IKernelLaunchConfig {
public:
    static_assert(NUM_PARAMS > 0, "Must have at least one kernel parameter");

    /**
     * Setup kernel function pointer
     *
     * @param[in] kernel_func Pointer to the kernel function
     * @throws std::runtime_error if cudaGetFuncBySymbol fails
     */
    void setup_kernel_function(const void *kernel_func) override {
        kernel_node_helper_.setup_kernel_function(kernel_func);
    }

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
            const std::size_t shared_mem_bytes = 0) override {
        kernel_node_helper_.setup_kernel_dimensions(grid_dim, block_dim, shared_mem_bytes);
    }

    /**
     * Launch kernel using the configured parameters
     *
     * @param[in] stream CUDA stream for kernel execution
     * @return CUresult indicating success (CUDA_SUCCESS) or failure
     */
    [[nodiscard]]
    CUresult launch(cudaStream_t stream) const override;

    /**
     * Get const reference to kernel node parameters for graph node creation
     *
     * @return Const reference to CUDA_KERNEL_NODE_PARAMS
     */
    [[nodiscard]]
    const CUDA_KERNEL_NODE_PARAMS &get_kernel_params() const {
        return kernel_node_helper_.get_kernel_params();
    }

private:
    /**
     * Clear all kernel arguments
     */
    void clear_arguments() override {
        kernel_args_.fill(nullptr);
        current_arg_index_ = 0;
    }

    /**
     * Add a single kernel argument
     *
     * @param[in] arg Pointer to the argument
     */
    void add_argument(void *arg) override {
        if (current_arg_index_ >= NUM_PARAMS) {
            throw std::runtime_error(std::format(
                    "Too many arguments for KernelLaunchConfig<{}> (max {})",
                    NUM_PARAMS,
                    NUM_PARAMS));
        }
        kernel_args_.at(current_arg_index_++) = arg;
    }

    /**
     * Finalize argument setup
     */
    void finalize_arguments() override {
        kernel_node_helper_.set_kernel_params_ptr(kernel_args_.data());
    }

    /**
     * Array of kernel argument pointers
     *
     * These pointers point to device memory addresses where the
     * parameter structures reside.
     */
    std::array<void *, NUM_PARAMS> kernel_args_{};

    /**
     * Current argument index for adding arguments
     */
    std::size_t current_arg_index_{};

    /**
     * Helper for managing CUDA kernel node parameters
     */
    KernelNodeHelper kernel_node_helper_{};
};

/**
 * Type alias for dual kernel configuration (2 parameters)
 */
using DualKernelLaunchConfig = KernelLaunchConfig<2>;

// Template function definitions
template <std::size_t NUM_PARAMS>
CUresult KernelLaunchConfig<NUM_PARAMS>::launch(cudaStream_t stream) const {
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

/**
 * Helper to setup kernel function in launch configuration
 *
 * @param config Kernel launch configuration to update
 * @param kernel_func Pointer to the kernel function
 * @throws std::runtime_error if cudaGetFuncBySymbol fails
 */
template <std::size_t NUM_PARAMS>
inline void setup_kernel_function(KernelLaunchConfig<NUM_PARAMS> &config, const void *kernel_func) {
    config.setup_kernel_function(kernel_func);
}

/**
 * Helper to setup kernel dimensions in launch configuration
 *
 * @param config Kernel launch configuration to update
 * @param grid_dim Grid dimensions
 * @param block_dim Block dimensions
 * @param shared_mem_bytes Shared memory size in bytes (default: 0)
 */
template <std::size_t NUM_PARAMS>
inline void setup_kernel_dimensions(
        KernelLaunchConfig<NUM_PARAMS> &config,
        const dim3 grid_dim,
        const dim3 block_dim,
        const std::size_t shared_mem_bytes = 0) {
    config.setup_kernel_dimensions(grid_dim, block_dim, shared_mem_bytes);
}

/**
 * Helper to setup kernel arguments in launch configuration
 *
 * @param config Kernel launch configuration to update
 * @param args Variable number of device pointer arguments
 */
template <std::size_t NUM_PARAMS, typename... Args>
void setup_kernel_arguments(KernelLaunchConfig<NUM_PARAMS> &config, Args &...args) {
    static_assert(
            sizeof...(args) == NUM_PARAMS, "Number of arguments must match template parameter");
    config.setup_kernel_arguments(std::addressof(args)...);
}

/**
 * @brief Launch a kernel using the pre-configured parameters
 *
 * This function wraps cuLaunchKernel with the parameters from
 * CUDA_KERNEL_NODE_PARAMS. It enables consistent kernel launching across all
 * modules using the cuBB pattern.
 *
 * @param kernel_node_params Pre-configured kernel launch parameters
 * @param stream CUDA stream for kernel execution
 * @return CUresult indicating success (CUDA_SUCCESS) or failure
 */
[[nodiscard]] inline CUresult
launch_kernel(const CUDA_KERNEL_NODE_PARAMS &kernel_node_params, cudaStream_t stream) {
    return cuLaunchKernel(
            kernel_node_params.func,
            kernel_node_params.gridDimX,
            kernel_node_params.gridDimY,
            kernel_node_params.gridDimZ,
            kernel_node_params.blockDimX,
            kernel_node_params.blockDimY,
            kernel_node_params.blockDimZ,
            kernel_node_params.sharedMemBytes,
            stream,
            kernel_node_params.kernelParams,
            kernel_node_params.extra);
}

} // namespace framework::pipeline

#endif // FRAMEWORK_CORE_KERNEL_LAUNCH_CONFIG_HPP
