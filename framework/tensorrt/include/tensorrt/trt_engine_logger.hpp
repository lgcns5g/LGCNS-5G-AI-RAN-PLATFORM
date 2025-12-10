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

#ifndef FRAMEWORK_TRT_ENGINE_LOGGER_HPP
#define FRAMEWORK_TRT_ENGINE_LOGGER_HPP

#include <NvInfer.h>
#include <NvInferRuntime.h>

namespace framework::tensorrt {

/**
 * @brief Logger implementation for TensorRT engine
 *
 * @details Concrete implementation of nvinfer1::ILogger interface required by
 * TensorRT. Handles logging of TensorRT runtime messages with configurable
 * severity levels.
 */
class TrtLogger final : public nvinfer1::ILogger {
    void log(Severity severity, const char *msg) noexcept final;
};

} // namespace framework::tensorrt

#endif // FRAMEWORK_TRT_ENGINE_LOGGER_HPP
