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

#ifndef RAN_PUSCH_TRT_UTILS_HPP
#define RAN_PUSCH_TRT_UTILS_HPP

#include <string>

namespace ran::pusch {

/**
 * Initialize RAN TensorRT plugins
 *
 * Uses an internal static TensorRT logger that forwards all messages to RT_LOG.
 * Thread-safe and only initializes once.
 *
 * @return true if plugins initialized successfully
 */
[[nodiscard]] bool init_ran_trt_plugins() noexcept;

/**
 * Get TRT engine file path from environment variable
 *
 * Reads the full path to the TRT engine file from the RAN_TRT_ENGINE_PATH
 * environment variable and verifies the file exists.
 *
 * @return Full path to TRT engine file
 * @throws std::runtime_error if RAN_TRT_ENGINE_PATH not set or file doesn't exist
 */
[[nodiscard]] std::string get_trt_engine_path();

} // namespace ran::pusch

#endif // RAN_PUSCH_TRT_UTILS_HPP
