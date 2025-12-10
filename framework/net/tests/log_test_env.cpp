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

#include "log/components.hpp"
#include "log/rt_log.hpp"
#include "net/net_log.hpp"
#include "net_test_helpers.hpp"

namespace {

/**
 * Global test setup - runs once before all tests
 *
 * Sets up debug logging for all network tests.
 */
class LogTestEnv : public ::testing::Environment {
public:
    void SetUp() override {
        // Set up logging for all tests

        framework::log::Logger::set_level(framework::log::LogLevel::Debug);
        framework::log::register_component<framework::net::Net>(framework::log::LogLevel::Debug);

        // Enable sanitizer compatibility for processes with elevated capabilities
        framework::net::enable_sanitizer_compatibility();

        if (!framework::net::has_cuda_device()) {
            FAIL() << "CUDA device availability check failed";
        }
        if (!framework::net::has_mellanox_nic()) {
            FAIL() << "Mellanox NIC availability check failed";
        }
    }

    void TearDown() override {}
};

// NOLINTNEXTLINE(cert-err58-cpp,cppcoreguidelines-owning-memory,readability-identifier-naming)
const auto *const g_env =
        ::testing::AddGlobalTestEnvironment(std::make_unique<LogTestEnv>().release());

} // namespace
