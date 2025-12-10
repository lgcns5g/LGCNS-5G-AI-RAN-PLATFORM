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

#ifndef FRAMEWORK_CORE_TRT_NULL_PRE_POST_ENQUEUE_HPP
#define FRAMEWORK_CORE_TRT_NULL_PRE_POST_ENQUEUE_HPP

#include <driver_types.h>

#include "tensorrt/trt_engine_interfaces.hpp"
#include "utils/errors.hpp"

namespace framework::tensorrt {

/**
 * Null/No-op implementation of IPrePostTrtEngEnqueue
 *
 * This class provides a null object pattern implementation for scenarios
 * where CUDA graph capture is not needed during TensorRT engine warmup.
 *
 * Use Cases:
 * - Pure stream-mode pipelines with no graph execution requirements
 * - Unit tests that only exercise stream-based execution paths
 * - Performance-critical scenarios where graph capture overhead must be avoided
 *
 * Design Tradeoff:
 * - Eliminates graph capture overhead during warmup (~milliseconds one-time
 * cost)
 * - Cannot support graph-based execution mode (execute_graph will fail)
 * - Reduces memory footprint (no captured graph stored)
 *
 * Example Usage:
 * @code
 * // For stream-only execution
 * auto null_capturer = std::make_unique<NullPrePostTrtEngEnqueue>();
 * auto trt_engine = std::make_unique<MLIRTrtEngine>(
 *     inputs, outputs,
 *     std::move(tensorrt_runtime),
 *     std::move(null_capturer)  // No graph capture
 * );
 *
 * // Warmup loads engine and runs once, but doesn't capture graph
 * trt_engine->warmup(stream);
 *
 * // Stream execution works normally
 * trt_engine->run(stream);  // OK
 *
 * // Graph execution would fail (no captured graph available)
 * // graph_capturer->get_graph();  // Would throw/fail
 * @endcode
 *
 * @see CaptureStreamPrePostTrtEngEnqueue for graph-mode capture
 * @see IPrePostTrtEngEnqueue for interface documentation
 */
class NullPrePostTrtEngEnqueue final : public IPrePostTrtEngEnqueue {
public:
    /**
     * Default constructor
     */
    NullPrePostTrtEngEnqueue() = default;

    /**
     * Destructor
     */
    ~NullPrePostTrtEngEnqueue() override = default;

    // Non-copyable, non-movable
    NullPrePostTrtEngEnqueue(const NullPrePostTrtEngEnqueue &) = delete;
    NullPrePostTrtEngEnqueue &operator=(const NullPrePostTrtEngEnqueue &) = delete;
    NullPrePostTrtEngEnqueue(NullPrePostTrtEngEnqueue &&) = delete;
    NullPrePostTrtEngEnqueue &operator=(NullPrePostTrtEngEnqueue &&) = delete;

    /**
     * Pre-enqueue hook (no-op for null implementation)
     *
     * This method does nothing and immediately returns success.
     * No graph capture or stream operations are performed.
     *
     * @param[in] cu_stream CUDA stream (unused)
     * @return utils::NvErrc::Success Always succeeds
     */
    [[nodiscard]] utils::NvErrc pre_enqueue([[maybe_unused]] cudaStream_t cu_stream) override {
        // No-op - no graph capture needed for stream-only execution
        return utils::NvErrc::Success;
    }

    /**
     * Post-enqueue hook (no-op for null implementation)
     *
     * This method does nothing and immediately returns success.
     * No graph capture or stream operations are performed.
     *
     * @param[in] cu_stream CUDA stream (unused)
     * @return utils::NvErrc::Success Always succeeds
     */
    [[nodiscard]] utils::NvErrc post_enqueue([[maybe_unused]] cudaStream_t cu_stream) override {
        // No-op - no graph capture needed for stream-only execution
        return utils::NvErrc::Success;
    }
};

} // namespace framework::tensorrt

#endif // FRAMEWORK_CORE_TRT_NULL_PRE_POST_ENQUEUE_HPP
