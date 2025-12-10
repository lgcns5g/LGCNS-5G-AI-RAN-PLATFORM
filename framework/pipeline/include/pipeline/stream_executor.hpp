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

#ifndef FRAMEWORK_CORE_STREAM_EXECUTOR_HPP
#define FRAMEWORK_CORE_STREAM_EXECUTOR_HPP

#include <gsl-lite/gsl-lite.hpp>

#include "pipeline/ikernel_launch_config.hpp"
#include "pipeline/istream_executor.hpp"

namespace framework::pipeline {

/**
 * Concrete implementation of IStreamExecutor that executes CUDA kernels on a
 * stream.
 *
 * This class takes a kernel launch configuration and executes it on the
 * provided CUDA stream. It handles the actual kernel launch using the CUDA
 * Driver API.
 */
class StreamExecutor final : public IStreamExecutor {
public:
    /**
     * Constructor
     *
     * @param[in] kernel_launch_config Pointer to the kernel launch configuration
     *                                 This must remain valid for the lifetime of
     *                                 the executor
     */
    explicit StreamExecutor(const IKernelLaunchConfig *kernel_launch_config);

    /**
     * Execute the kernel on a CUDA stream.
     *
     * Uses kernel parameters previously configured via the kernel_launch_config.
     * Dynamic parameters should be set via setup_tick() on the owning module
     * before calling execute().
     *
     * @param[in] stream The CUDA stream to execute on
     * @throws std::runtime_error if kernel launch fails
     */
    void execute(cudaStream_t stream) override;

private:
    const IKernelLaunchConfig
            *kernel_launch_config_{}; //!< Non-owning pointer to kernel configuration
};

} // namespace framework::pipeline

#endif // FRAMEWORK_CORE_STREAM_EXECUTOR_HPP
