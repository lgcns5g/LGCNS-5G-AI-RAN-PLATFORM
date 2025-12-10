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

#include <memory>    // for make_unique, unique...
#include <stdexcept> // for invalid_argument

#include <quill/LogMacros.h> // for QUILL_LOG_ERROR

#include "pipeline/dynamic_kernel_launch_config.hpp" // for DynamicKernelLaunch...
#include "pipeline/ikernel_launch_config.hpp"        // for IKernelLaunchConfig
#include "pipeline/istream_executor.hpp"             // for IStreamExecutor
#include "pipeline/null_stream_executor.hpp"         // for NullStreamExecutor
#include "pipeline/stream_executor.hpp"              // for StreamExecutor
#include "pipeline/stream_executor_factory.hpp"      // for StreamExecutorFactory
#include "utils/error_macros.hpp"                    // for FRAMEWORK_NV_THROW_IF

namespace framework::pipeline {

StreamExecutorFactory::StreamExecutorFactory(ExecutorType type) : type_(type) {}

std::unique_ptr<IStreamExecutor>
StreamExecutorFactory::create_stream_executor(const IKernelLaunchConfig *kernel_config) const {

    switch (type_) {
    case ExecutorType::Real:
        FRAMEWORK_NV_THROW_IF(
                kernel_config == nullptr,
                std::invalid_argument,
                "Kernel launch configuration is required for Real executors");
        return std::make_unique<StreamExecutor>(kernel_config);

    case ExecutorType::Null:
        // NullStreamExecutor doesn't need kernel config
        return std::make_unique<NullStreamExecutor>();

    default:
        FRAMEWORK_NV_THROW(std::invalid_argument, "Unknown executor type");
    }
}

std::unique_ptr<IStreamExecutor> StreamExecutorFactory::create_stream_executor(
        const DynamicKernelLaunchConfig *kernel_config) const {
    return create_stream_executor(static_cast<const IKernelLaunchConfig *>(kernel_config));
}

} // namespace framework::pipeline
