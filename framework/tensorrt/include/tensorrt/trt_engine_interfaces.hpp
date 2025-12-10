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

#ifndef FRAMEWORK_TRT_ENGINE_INTERFACES_HPP
#define FRAMEWORK_TRT_ENGINE_INTERFACES_HPP

#include <any>
#include <vector>

#include <gsl-lite/gsl-lite.hpp>

#include "tensorrt/trt_engine_params.hpp"
#include "utils/errors.hpp"

namespace framework::tensorrt {

/**
 * @brief Define an interface for Pre/Post Trt Engine EnqueueV3
 *
 * For pre and post enqueue_v3() in TrtEngine
 */
class IPrePostTrtEngEnqueue {
public:
    IPrePostTrtEngEnqueue() = default;
    virtual ~IPrePostTrtEngEnqueue() = default;

    /**
     * Copy constructor
     * @param[in] pre_post_trt_eng_enqueue Source object to copy from
     */
    IPrePostTrtEngEnqueue(const IPrePostTrtEngEnqueue &pre_post_trt_eng_enqueue) = default;

    /**
     * Copy assignment operator
     * @param[in] pre_post_trt_eng_enqueue Source object to copy from
     * @return Reference to this object
     */
    IPrePostTrtEngEnqueue &
    operator=(const IPrePostTrtEngEnqueue &pre_post_trt_eng_enqueue) = default;

    /**
     * Move constructor
     * @param[in] pre_post_trt_eng_enqueue Source object to move from
     */
    IPrePostTrtEngEnqueue(IPrePostTrtEngEnqueue &&pre_post_trt_eng_enqueue) = default;

    /**
     * Move assignment operator
     * @param[in] pre_post_trt_eng_enqueue Source object to move from
     * @return Reference to this object
     */
    IPrePostTrtEngEnqueue &operator=(IPrePostTrtEngEnqueue &&pre_post_trt_eng_enqueue) = default;

    // API

    /**
     * @brief Pre Enqueue activity before calling enqueue_v3()
     * @param cu_stream stream to use
     * @return utils::NvErrc SUCCESS or error
     */
    [[nodiscard]]
    virtual utils::NvErrc pre_enqueue(cudaStream_t cu_stream) = 0;

    /**
     * @brief Post Enqueue activity after calling enqueue_v3()
     * @param cu_stream stream to use
     * @return utils::NvErrc SUCCESS or error
     */
    [[nodiscard]]
    virtual utils::NvErrc post_enqueue(cudaStream_t cu_stream) = 0;
};

} // namespace framework::tensorrt

#endif // FRAMEWORK_TRT_ENGINE_INTERFACES_HPP
