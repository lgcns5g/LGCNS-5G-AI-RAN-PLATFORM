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

#include <memory>

#include <gtest/gtest.h>

#include "ldpc/ldpc_log.hpp"
#include "log/components.hpp"
#include "log/rt_log.hpp"
#include "log/rt_log_macros.hpp"
#include "pusch/pusch_log.hpp"
#include "pusch/pusch_trt_utils.hpp"

namespace {

/**
 * Global test setup - runs once before all tests
 *
 * Loads TensorRT plugins and sets up logging for all PUSCH tests.
 */
class PuschTestEnv : public ::testing::Environment {
public:
    void SetUp() override {
        // Set up logging for all tests
        framework::log::Logger::set_level(framework::log::LogLevel::Debug);
        framework::log::register_component<ran::pusch::PuschComponent>(
                framework::log::LogLevel::Info);
        framework::log::register_component<ran::ldpc::LdpcComponent>(
                framework::log::LogLevel::Info);

        RT_LOG_DEBUG("Initialized PUSCH logging environment.");

        // Load and register TRT plugins once for the entire test session
        const bool success = ran::pusch::init_ran_trt_plugins();
        if (!success) {
            RT_LOG_ERROR("Failed to initialize TensorRT plugins");
            FAIL() << "Failed to initialize TensorRT plugins";
        }
        RT_LOG_INFO("TensorRT plugin environment initialized successfully.");
    }

    void TearDown() override {}
};

// NOLINTNEXTLINE(cert-err58-cpp,cppcoreguidelines-owning-memory,readability-identifier-naming)
const auto *const g_env =
        ::testing::AddGlobalTestEnvironment(std::make_unique<PuschTestEnv>().release());

} // namespace
