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

function(check_libfuzzer_support var_name)
    set(LibFuzzerTestSource
        "
#include <cstdint>

extern \"C\" int LLVMFuzzerTestOneInput(const std::uint8_t *data, std::size_t size) {
  return 0;
}
    ")

    include(CheckCXXSourceCompiles)

    set(CMAKE_REQUIRED_FLAGS "-fsanitize=fuzzer")
    set(CMAKE_REQUIRED_LINK_OPTIONS "-fsanitize=fuzzer")
    check_cxx_source_compiles("${LibFuzzerTestSource}" ${var_name})

endfunction()
