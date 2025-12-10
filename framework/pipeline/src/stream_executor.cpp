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

#include <stdexcept> // for invalid_argument

#include <driver_types.h>    // for CUstream_st, cudaStream_t
#include <quill/LogMacros.h> // for QUILL_LOG_ERROR

#include "pipeline/ikernel_launch_config.hpp" // for IKernelLaunchConfig
#include "pipeline/stream_executor.hpp"       // for StreamExecutor
#include "utils/error_macros.hpp"             // for FRAMEWORK_CUDA_DRIVER_CHE...

namespace framework::pipeline {

StreamExecutor::StreamExecutor(const IKernelLaunchConfig *kernel_launch_config)
        : kernel_launch_config_(kernel_launch_config) {
    FRAMEWORK_NV_THROW_IF(
            kernel_launch_config_ == nullptr,
            std::invalid_argument,
            "Kernel launch configuration is required for StreamExecutor");
}

void StreamExecutor::execute(cudaStream_t stream) {
    // The kernel launch configuration should already be fully set up
    // by the module during its configure_io() phase
    FRAMEWORK_CUDA_DRIVER_CHECK_THROW(kernel_launch_config_->launch(stream));
}

} // namespace framework::pipeline
