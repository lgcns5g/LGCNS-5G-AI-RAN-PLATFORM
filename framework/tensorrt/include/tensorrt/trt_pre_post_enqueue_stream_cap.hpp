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

#ifndef FRAMEWORK_TRT_PRE_POST_ENQUEUE_STREAM_CAP_HPP
#define FRAMEWORK_TRT_PRE_POST_ENQUEUE_STREAM_CAP_HPP

#include <driver_types.h>

#include <cuda.h>

#include "tensorrt/trt_engine_interfaces.hpp"
#include "utils/errors.hpp"

namespace framework::tensorrt {

/**
 * @brief Specific implementation of interface IPrePostTrtEngEnqueue
 *
 * Implements stream capture for interface IPrePostTrtEngEnqueue
 */
class CaptureStreamPrePostTrtEngEnqueue final : public IPrePostTrtEngEnqueue {
public:
    /**
     * Default constructor
     */
    CaptureStreamPrePostTrtEngEnqueue() = default;

    /**
     * Destructor, Destroys the captured CUDA graph
     */
    ~CaptureStreamPrePostTrtEngEnqueue() final;

    // Non-copyable, non-movable
    CaptureStreamPrePostTrtEngEnqueue(const CaptureStreamPrePostTrtEngEnqueue &) = delete;
    CaptureStreamPrePostTrtEngEnqueue &
    operator=(const CaptureStreamPrePostTrtEngEnqueue &) = delete;
    CaptureStreamPrePostTrtEngEnqueue(CaptureStreamPrePostTrtEngEnqueue &&) = delete;
    CaptureStreamPrePostTrtEngEnqueue &operator=(CaptureStreamPrePostTrtEngEnqueue &&) = delete;

    /**
     * Start Capture/End capture of stream
     * @param cu_stream stream to use
     * @return utils::NvErrc for SUCCESS or any failure
     */
    [[nodiscard]]
    utils::NvErrc pre_enqueue(cudaStream_t cu_stream) final;
    [[nodiscard]]
    utils::NvErrc post_enqueue(cudaStream_t cu_stream) final;

    /**
     * Get the captured CUDA graph
     * @return Pointer to the captured CUDA graph, or nullptr if no graph is
     * captured
     */
    [[nodiscard]] CUgraph get_graph() const { return graph_; }

private:
    /**
     * CUDA graph created when we call post_enqueue on the stream.
     * This CUDA graph will be used in a top level graph and we own it and destroy
     * it in the destructor*
     */
    CUgraph graph_{};
};

} // namespace framework::tensorrt

#endif // FRAMEWORK_TRT_PRE_POST_ENQUEUE_STREAM_CAP_HPP
