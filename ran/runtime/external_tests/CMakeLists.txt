# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

enable_testing()

include(${CMAKE_SOURCE_DIR}/cmake/Tests.cmake)

# Test for ru_emulator executable (prints usage)
add_test(NAME ran.runtime.ru_emulator_help COMMAND $<TARGET_FILE:ru_emulator> --help)
label_parallel(ran.runtime.ru_emulator_help)

# Test for test_mac executable (prints usage)
add_test(NAME ran.runtime.test_mac_help COMMAND $<TARGET_FILE:test_mac> --help)
label_parallel(ran.runtime.test_mac_help)
