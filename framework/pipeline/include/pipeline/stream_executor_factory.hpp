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

#ifndef FRAMEWORK_CORE_STREAM_EXECUTOR_FACTORY_HPP
#define FRAMEWORK_CORE_STREAM_EXECUTOR_FACTORY_HPP

#include <memory>

#include <wise_enum.h>

#include "pipeline/istream_executor.hpp"

namespace framework::pipeline {

// Forward declarations
class IKernelLaunchConfig;
class DynamicKernelLaunchConfig;
template <std::size_t NUM_PARAMS> class KernelLaunchConfig;

/**
 * Factory for creating IStreamExecutor instances.
 *
 * This factory can create either real StreamExecutor instances that execute
 * CUDA kernels, or NullStreamExecutor instances for testing or no-op scenarios.
 */
class StreamExecutorFactory final {
public:
    /**
     * Executor type enumeration
     */
    enum class ExecutorType {
        Real, //!< Create real StreamExecutor instances
        Null  //!< Create NullStreamExecutor instances
    };

    /**
     * Constructor
     *
     * @param[in] type The type of executors this factory should create
     */
    explicit StreamExecutorFactory(ExecutorType type);

    /**
     * Create a stream executor instance.
     *
     * @param[in] kernel_config Pointer to kernel launch configuration
     *                          Required for Real executors
     *                          Can be nullptr for Null executors.
     *                          kernel_config is ignored in case of Null
     *                          executors.
     * @return A unique pointer to the created IStreamExecutor instance
     */
    [[nodiscard]] std::unique_ptr<IStreamExecutor>
    create_stream_executor(const IKernelLaunchConfig *kernel_config) const;

    /**
     * Create a stream executor instance with KernelLaunchConfig<NUM_PARAMS>.
     *
     * @tparam NUM_PARAMS Number of kernel parameters
     * @param[in] kernel_config Pointer to kernel launch configuration
     * @return A unique pointer to the created IStreamExecutor instance
     */
    template <std::size_t NUM_PARAMS>
    [[nodiscard]] std::unique_ptr<IStreamExecutor>
    create_stream_executor(const KernelLaunchConfig<NUM_PARAMS> *kernel_config) const {
        return create_stream_executor(static_cast<const IKernelLaunchConfig *>(kernel_config));
    }

    /**
     * Create a stream executor instance with DynamicKernelLaunchConfig.
     *
     * @param[in] kernel_config Pointer to dynamic kernel launch configuration
     * @return A unique pointer to the created IStreamExecutor instance
     */
    [[nodiscard]] std::unique_ptr<IStreamExecutor>
    create_stream_executor(const DynamicKernelLaunchConfig *kernel_config) const;

private:
    ExecutorType type_{}; //!< The type of executors to create
};

} // namespace framework::pipeline

// WISE_ENUM_ADAPT must be called at global namespace scope for ADL to work
WISE_ENUM_ADAPT(framework::pipeline::StreamExecutorFactory::ExecutorType, Real, Null)

#endif // FRAMEWORK_CORE_STREAM_EXECUTOR_FACTORY_HPP
