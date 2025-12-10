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

#include <memory> // for allocator

#include <driver_types.h> // for cudaMemcpyKind

#include <gtest/gtest.h> // for Test, TestInfo (ptr only), EXPECT_EQ

#include "memory/memcpy_helper.hpp" // for MemcpyHelper

namespace {

// Test: Verifies device-to-device memory copy kind is correctly determined
TEST(MemcpyHelperTest, DeviceToDeviceKind) {
    constexpr auto EXPECTED_KIND = framework::memory::
            MemcpyHelper<framework::memory::DeviceAlloc, framework::memory::DeviceAlloc>::KIND;
    EXPECT_EQ(EXPECTED_KIND, cudaMemcpyDeviceToDevice);
}

// Test: Verifies device-to-host memory copy kind is correctly determined
TEST(MemcpyHelperTest, DeviceToHostKind) {
    constexpr auto EXPECTED_KIND = framework::memory::
            MemcpyHelper<framework::memory::PinnedAlloc, framework::memory::DeviceAlloc>::KIND;
    EXPECT_EQ(EXPECTED_KIND, cudaMemcpyDeviceToHost);
}

// Test: Verifies host-to-device memory copy kind is correctly determined
TEST(MemcpyHelperTest, HostToDeviceKind) {
    constexpr auto EXPECTED_KIND = framework::memory::
            MemcpyHelper<framework::memory::DeviceAlloc, framework::memory::PinnedAlloc>::KIND;
    EXPECT_EQ(EXPECTED_KIND, cudaMemcpyHostToDevice);
}

// Test: Verifies host-to-host memory copy kind is correctly determined
TEST(MemcpyHelperTest, HostToHostKind) {
    constexpr auto EXPECTED_KIND = framework::memory::
            MemcpyHelper<framework::memory::PinnedAlloc, framework::memory::PinnedAlloc>::KIND;
    EXPECT_EQ(EXPECTED_KIND, cudaMemcpyHostToHost);
}

} // namespace
