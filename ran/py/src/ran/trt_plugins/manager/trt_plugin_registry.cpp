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

#include <exception>
#include <format>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>

#include <NvInfer.h>

#include "ran/trt_plugins/cholesky_factor_inv/cholesky_factor_inv_trt_plugin.hpp"
#include "ran/trt_plugins/dmrs/dmrs_trt_plugin.hpp"
#include "ran/trt_plugins/fft/fft_trt_plugin.hpp"
#include "ran/trt_plugins/sample/sequential_sum_plugin.hpp"

// TensorRT plugin registration functions
extern "C" {

__attribute__((visibility("default"))) bool
init_ran_plugins([[maybe_unused]] void *logger, const char *lib_namespace = "") {

    // Use provided namespace or empty string
    const std::string_view namespace_sv = (lib_namespace != nullptr) ? lib_namespace : "";

    static ran::trt_plugin::SequentialSumPluginCreator sequential_sum_creator(namespace_sv);
    static ran::trt_plugin::DMRSTrtPluginCreator dmrs_trt_creator(namespace_sv);
    static ran::trt_plugin::FftTrtPluginCreator fft_creator(namespace_sv);
    static ran::trt_plugin::CholeskyFactorInvPluginCreator cholesky_factor_inv_creator(
            namespace_sv);

    // Try to register with the plugin registry if available
    try {
        auto *registry = getPluginRegistry();
        if (registry != nullptr) {
            registry->registerCreator(sequential_sum_creator, lib_namespace);
            registry->registerCreator(dmrs_trt_creator, lib_namespace);
            registry->registerCreator(fft_creator, lib_namespace);
            registry->registerCreator(cholesky_factor_inv_creator, lib_namespace);
        }
    } catch (const std::exception &e) {
        std::cerr << std::format("Error registering TensorRT plugins: {}\n", e.what());
        return false;
    } catch (...) {
        std::cerr << std::format("Unknown error registering TensorRT plugins\n");
        return false;
    }
    return true;
}

__attribute__((visibility("default"))) nvinfer1::IPluginCreatorInterface *
get_sequential_sum_trt_creator() {
    return std::make_unique<ran::trt_plugin::SequentialSumPluginCreator>("").release();
}

__attribute__((visibility("default"))) nvinfer1::IPluginV3 *get_sequential_sum_trt_plugin(
        const char *name,
        [[maybe_unused]] const nvinfer1::PluginFieldCollection *fc,
        [[maybe_unused]] nvinfer1::TensorRTPhase phase) {
    return std::make_unique<ran::trt_plugin::SequentialSumPlugin>(name).release();
}

__attribute__((visibility("default"))) nvinfer1::IPluginCreatorInterface *get_dmrs_trt_creator() {
    return std::make_unique<ran::trt_plugin::DMRSTrtPluginCreator>("").release();
}

__attribute__((visibility("default"))) nvinfer1::IPluginV3 *get_dmrs_trt_plugin(
        const char *name,
        [[maybe_unused]] const nvinfer1::PluginFieldCollection *fc,
        [[maybe_unused]] nvinfer1::TensorRTPhase phase) {
    return std::make_unique<ran::trt_plugin::DMRSTrtPlugin>(name).release();
}

__attribute__((visibility("default"))) nvinfer1::IPluginCreatorInterface *get_fft_trt_creator() {
    return std::make_unique<ran::trt_plugin::FftTrtPluginCreator>("").release();
}

__attribute__((visibility("default"))) nvinfer1::IPluginV3 *get_fft_trt_plugin(
        const char *name,
        [[maybe_unused]] const nvinfer1::PluginFieldCollection *fc,
        [[maybe_unused]] nvinfer1::TensorRTPhase phase) {
    return std::make_unique<ran::trt_plugin::FftTrtPlugin>(name).release();
}

__attribute__((visibility("default"))) nvinfer1::IPluginCreatorInterface *
get_cholesky_factor_inv_creator() {
    return std::make_unique<ran::trt_plugin::CholeskyFactorInvPluginCreator>("").release();
}

__attribute__((visibility("default"))) nvinfer1::IPluginV3 *get_cholesky_factor_inv_plugin(
        const char *name,
        [[maybe_unused]] const nvinfer1::PluginFieldCollection *fc,
        [[maybe_unused]] nvinfer1::TensorRTPhase phase) {
    return std::make_unique<ran::trt_plugin::CholeskyFactorInvPlugin>(name).release();
}

} // extern "C"
