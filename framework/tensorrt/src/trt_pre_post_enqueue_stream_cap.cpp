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

#include <driver_types.h>
#include <quill/LogMacros.h>

#include <cuda_runtime_api.h>

#include "tensorrt/trt_pre_post_enqueue_stream_cap.hpp"
#include "utils/error_macros.hpp"
#include "utils/errors.hpp"

namespace framework::tensorrt {

CaptureStreamPrePostTrtEngEnqueue::~CaptureStreamPrePostTrtEngEnqueue() {
    cudaGraphDestroy(graph_);
}

utils::NvErrc CaptureStreamPrePostTrtEngEnqueue::pre_enqueue(cudaStream_t cu_stream) {
    FRAMEWORK_CUDA_RUNTIME_CHECK_THROW(
            cudaStreamBeginCapture(cu_stream, cudaStreamCaptureModeGlobal));
    return utils::NvErrc::Success;
}

utils::NvErrc CaptureStreamPrePostTrtEngEnqueue::post_enqueue(cudaStream_t cu_stream) {
    FRAMEWORK_CUDA_RUNTIME_CHECK_THROW(cudaStreamEndCapture(cu_stream, &graph_));
    return utils::NvErrc::Success;
}

} // namespace framework::tensorrt
