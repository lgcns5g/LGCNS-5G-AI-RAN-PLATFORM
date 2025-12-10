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

/**
 * @file test_logging_setup.cpp
 * @brief Global logging initialization for framework tests
 */

#include <quill/LogMacros.h> // for QUILL_LOG_INFO

#include <gtest/gtest.h> // for AddGlobalTestEnvironment, Environment

#include "log/components.hpp"    // for LogLevel, register_component
#include "log/rt_log.hpp"        // for Logger, Logger::set_level
#include "log/rt_log_macros.hpp" // for RT_LOG_INFO, RT_LOGC_INFO
#include "utils/core_log.hpp"    // for Core

namespace {

/**
 * Global test environment for framework logging initialization
 *
 * Sets up logging once before all tests run and ensures proper cleanup
 * after all tests complete. Uses DEBUG level across all components.
 */
class FrameworkTestLoggingEnvironment final : public ::testing::Environment {
public:
    void SetUp() override {
        // Initialize logging with DEBUG level
        framework::log::Logger::set_level(framework::log::LogLevel::Debug);

        // Set DEBUG level for all core components
        framework::log::register_component<framework::utils::Core>(framework::log::LogLevel::Debug);

        RT_LOG_INFO("===== Framework Test Suite Started =====");
        RT_LOGC_INFO(framework::utils::Core::CoreNvApi, "Framework logging initialized for tests");
    }

    void TearDown() override {
        RT_LOG_INFO("===== Framework Test Suite Completed =====");
        framework::log::Logger::flush();
    }
};

// Register the global test environment - Google Test manages lifetime
// NOLINTNEXTLINE(cppcoreguidelines-owning-memory,cert-err58-cpp,readability-identifier-naming)
[[maybe_unused]] const auto *const framework_logging_env =
        // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
        ::testing::AddGlobalTestEnvironment(new FrameworkTestLoggingEnvironment);

} // namespace
