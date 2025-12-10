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

#ifndef FRAMEWORK_CORE_ISTREAM_EXECUTOR_HPP
#define FRAMEWORK_CORE_ISTREAM_EXECUTOR_HPP

#include <cuda_runtime.h>

#include "pipeline/types.hpp"

namespace framework::pipeline {

/**
 * Interface for executing operations directly on a CUDA stream.
 *
 * This interface provides a mechanism for executing operations on a CUDA
 * stream, typically when CUDA graph execution is not available or suitable.
 * This interface is designed to be used via composition rather than
 * inheritance, allowing modules to delegate execution to concrete executor
 * implementations.
 */
class IStreamExecutor {
public:
    /**
     * Default constructor.
     */
    IStreamExecutor() = default;

    /**
     * Virtual destructor.
     */
    virtual ~IStreamExecutor() = default;

    /**
     * Copy constructor.
     */
    IStreamExecutor(const IStreamExecutor &) = default;

    /**
     * Move constructor.
     */
    IStreamExecutor(IStreamExecutor &&) = default;

    /**
     * Copy assignment operator.
     * @return Reference to this object
     */
    IStreamExecutor &operator=(const IStreamExecutor &) = default;

    /**
     * Move assignment operator.
     * @return Reference to this object
     */
    IStreamExecutor &operator=(IStreamExecutor &&) = default;

    /**
     * Execute operations on a CUDA stream.
     *
     * This method launches the module's GPU operations using parameters
     * previously set by setup_tick(). The separation between setup_tick()
     * and execute() allows the same pattern to work for both stream and
     * graph execution modes.
     *
     * @param[in] stream The CUDA stream to execute on
     * @note setup_tick() must be called before execute() to prepare parameters
     */
    virtual void execute(cudaStream_t stream) = 0;
};

} // namespace framework::pipeline

#endif // FRAMEWORK_CORE_ISTREAM_EXECUTOR_HPP
