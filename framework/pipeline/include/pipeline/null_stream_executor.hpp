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

#ifndef FRAMEWORK_CORE_NULL_STREAM_EXECUTOR_HPP
#define FRAMEWORK_CORE_NULL_STREAM_EXECUTOR_HPP

#include <cuda_runtime.h>

#include "pipeline/istream_executor.hpp"

namespace framework::pipeline {

/**
 * Null implementation of IStreamExecutor for testing and placeholder scenarios.
 *
 * This class provides a no-op implementation of the IStreamExecutor interface.
 * It can be used in unit tests or as a placeholder when stream execution is not
 * needed.
 */
class NullStreamExecutor final : public IStreamExecutor {
public:
    /**
     * Execute operation - does nothing.
     *
     * @param[in] stream The CUDA stream to execute on (ignored)
     */
    void execute([[maybe_unused]] cudaStream_t stream) override {
        // Intentionally empty - null object pattern
        // Could add logging here if needed for debugging
    }
};

} // namespace framework::pipeline

#endif // FRAMEWORK_CORE_NULL_STREAM_EXECUTOR_HPP
