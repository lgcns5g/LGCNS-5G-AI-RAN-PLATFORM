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

#include <quill/LogMacros.h>

#include "log/rt_log_macros.hpp"
#include "tensorrt/trt_engine_logger.hpp"
#include "utils/core_log.hpp"

namespace framework::tensorrt {

void TrtLogger::log(Severity severity, const char *msg) noexcept {
    // Only log warnings or more important.
    if (severity <= Severity::kWARNING) {
        RT_LOGC_WARN(utils::Core::CoreNvApi, "{}", msg);
    }
}

} // namespace framework::tensorrt
