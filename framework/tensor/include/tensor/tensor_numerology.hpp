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

#ifndef FRAMEWORK_TENSOR_NUMEROLOGY_HPP
#define FRAMEWORK_TENSOR_NUMEROLOGY_HPP

#include <cstdint>

#include <wise_enum.h>

namespace framework::tensor {

/**
 * Tensor dimension counts and limits
 */
enum class TensorDimension : std::uint8_t {
    Dim1 = 1,  //!< 1-D tensor dimension count
    Dim2 = 2,  //!< 2-D tensor dimension count
    Dim3 = 3,  //!< 3-D tensor dimension count
    Dim4 = 4,  //!< 4-D tensor dimension count
    Dim5 = 5,  //!< 5-D tensor dimension count
    Max = Dim5 //!< Maximum tensor dimensions supported
};

static_assert(
        static_cast<std::uint8_t>(TensorDimension::Dim5) ==
                static_cast<std::uint8_t>(TensorDimension::Max),
        "TensorDimension::Dim5 must track the highest dimension enum");

} // namespace framework::tensor

// WISE_ENUM_ADAPT must be called at global namespace scope for ADL to work
WISE_ENUM_ADAPT(framework::tensor::TensorDimension, Dim1, Dim2, Dim3, Dim4, Dim5, Max)

#endif // FRAMEWORK_TENSOR_NUMEROLOGY_HPP
