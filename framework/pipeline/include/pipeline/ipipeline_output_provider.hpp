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

#ifndef FRAMEWORK_PIPELINE_IPIPELINE_OUTPUT_PROVIDER_HPP
#define FRAMEWORK_PIPELINE_IPIPELINE_OUTPUT_PROVIDER_HPP

#include <span>

#include "pipeline/types.hpp"

namespace framework::pipeline {

/**
 * @class IPipelineOutputProvider
 * @brief Interface for accessing pipeline output buffer addresses
 *
 * This interface provides access to stable output buffer addresses from
 * pipelines (e.g., Order Kernel) for zero-copy integration with downstream
 * consumers. Addresses remain valid after pipeline warmup for the pipeline's
 * lifetime.
 *
 * Thread-safety: Implementations must ensure thread-safe access if called
 * from multiple threads.
 */
class IPipelineOutputProvider {
public:
    /**
     * Default constructor.
     */
    IPipelineOutputProvider() = default;

    /**
     * Virtual destructor for proper cleanup of derived classes.
     */
    virtual ~IPipelineOutputProvider() = default;

    /**
     * Move constructor.
     */
    IPipelineOutputProvider(IPipelineOutputProvider &&) = default;

    /**
     * Move assignment operator.
     *
     * @return Reference to this object
     */
    IPipelineOutputProvider &operator=(IPipelineOutputProvider &&) = default;

    /**
     * Deleted copy constructor (non-copyable).
     */
    IPipelineOutputProvider(const IPipelineOutputProvider &) = delete;

    /**
     * Deleted copy assignment operator (non-copyable).
     *
     * @return Reference to this object
     */
    IPipelineOutputProvider &operator=(const IPipelineOutputProvider &) = delete;

    /**
     * Get Order Kernel pipeline output addresses
     *
     * Provides access to stable output buffer addresses captured after
     * Order Kernel warmup. These addresses can be used for zero-copy data
     * passing to downstream pipelines (e.g., PUSCH pipeline).
     *
     * @return Span of PortInfo describing Order Kernel outputs. Empty span
     *         indicates outputs are not available (e.g., pipeline not initialized).
     * @note Addresses are stable after warmup and remain valid for pipeline lifetime
     * @note Thread-safe if implementation provides thread-safety guarantees
     */
    [[nodiscard]] virtual std::span<const PortInfo> get_order_kernel_outputs() const noexcept = 0;
};

} // namespace framework::pipeline

#endif // FRAMEWORK_PIPELINE_IPIPELINE_OUTPUT_PROVIDER_HPP
