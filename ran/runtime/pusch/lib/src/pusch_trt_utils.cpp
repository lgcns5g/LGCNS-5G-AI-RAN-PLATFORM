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

#include <cstdlib>
#include <filesystem>
#include <format>
#include <mutex>
#include <stdexcept>
#include <string>

#include <NvInfer.h>

#include "log/rt_log_macros.hpp"
#include "pusch/pusch_trt_utils.hpp"

// RAN TensorRT plugin initialization function
// registered in the plugin library ran_trt_plugin
extern "C" {
bool init_ran_plugins(void *logger, const char *lib_namespace = "");
}

namespace ran::pusch {

namespace {

/// TensorRT logger that forwards all messages to RT_LOG
class TrtLogger : public nvinfer1::ILogger {
public:
    void log(const Severity severity, const char *msg) noexcept override {
        switch (severity) {
        case Severity::kINTERNAL_ERROR:
            RT_LOG_ERROR("TensorRT Internal Error: {}", msg);
            break;
        case Severity::kERROR:
            RT_LOG_ERROR("TensorRT Error: {}", msg);
            break;
        case Severity::kWARNING:
            RT_LOG_WARN("TensorRT Warning: {}", msg);
            break;
        case Severity::kINFO:
            RT_LOG_INFO("TensorRT Info: {}", msg);
            break;
        case Severity::kVERBOSE:
            RT_LOG_DEBUG("TensorRT Verbose: {}", msg);
            break;
        }
    }
};

} // namespace

bool init_ran_trt_plugins() noexcept {
    static TrtLogger logger{};
    static std::once_flag plugin_init_flag;
    static bool plugin_load_success{false};

    std::call_once(plugin_init_flag, []() { plugin_load_success = init_ran_plugins(&logger, ""); });
    return plugin_load_success;
}

std::string get_trt_engine_path() {
    // NOLINTNEXTLINE(concurrency-mt-unsafe) - Called during initialization
    const char *env_path = std::getenv("RAN_TRT_ENGINE_PATH");
    if (env_path == nullptr) {
        const std::string error_msg =
                "RAN_TRT_ENGINE_PATH environment variable is not set. "
                "This variable should contain the full path to the TensorRT engine file.";
        RT_LOG_ERROR("{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    const std::string engine_path{env_path};

    // Verify the engine file exists and is a regular file
    if (!std::filesystem::exists(engine_path)) {
        const std::string error_msg = std::format(
                "TensorRT engine file does not exist: {}. "
                "Ensure the Python tests have been run to generate the engine files.",
                engine_path);
        RT_LOG_ERROR("{}", error_msg);
        throw std::runtime_error(error_msg);
    }
    if (!std::filesystem::is_regular_file(engine_path)) {
        const std::string error_msg = std::format(
                "TensorRT engine path exists but is not a regular file: {}. "
                "Path must point to a .trtengine file, not a directory.",
                engine_path);
        RT_LOG_ERROR("{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    return engine_path;
}

} // namespace ran::pusch
