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

#include <cstddef>

#include "tensor/data_types.hpp"

namespace framework::tensor {

// clang-format off
std::size_t get_nv_type_storage_element_size(const NvDataType type) noexcept
{
    switch (type)
    {
    default:
    case TensorVoid:  return 0;                                           // uninitialized type
    case TensorBit:   return sizeof(data_type_traits<TensorBit>::Type);    // 1-bit value - special handling for sub-byte types
    case TensorR8I:   return sizeof(data_type_traits<TensorR8I>::Type);    // 8-bit signed integer real values
    case TensorC8I:   return sizeof(data_type_traits<TensorC8I>::Type);    // 8-bit signed integer complex values
    case TensorR8U:   return sizeof(data_type_traits<TensorR8U>::Type);    // 8-bit unsigned integer real values
    case TensorC8U:   return sizeof(data_type_traits<TensorC8U>::Type);    // 8-bit unsigned integer complex values
    case TensorR16I:  return sizeof(data_type_traits<TensorR16I>::Type);   // 16-bit signed integer real values
    case TensorC16I:  return sizeof(data_type_traits<TensorC16I>::Type);   // 16-bit signed integer complex values
    case TensorR16U:  return sizeof(data_type_traits<TensorR16U>::Type);   // 16-bit unsigned integer real values
    case TensorC16U:  return sizeof(data_type_traits<TensorC16U>::Type);   // 16-bit unsigned integer complex values
    case TensorR32I:  return sizeof(data_type_traits<TensorR32I>::Type);   // 32-bit signed integer real values
    case TensorC32I:  return sizeof(data_type_traits<TensorC32I>::Type);   // 32-bit signed integer complex values
    case TensorR32U:  return sizeof(data_type_traits<TensorR32U>::Type);   // 32-bit unsigned integer real values
    case TensorC32U:  return sizeof(data_type_traits<TensorC32U>::Type);   // 32-bit unsigned integer complex values
    case TensorR16F:  return sizeof(data_type_traits<TensorR16F>::Type);   // half precision (16-bit) real values
    case TensorC16F:  return sizeof(data_type_traits<TensorC16F>::Type);   // half precision (16-bit) complex values
    case TensorR32F:  return sizeof(data_type_traits<TensorR32F>::Type);   // single precision (32-bit) real values
    case TensorC32F:  return sizeof(data_type_traits<TensorC32F>::Type);   // single precision (32-bit) complex values
    case TensorR64F:  return sizeof(data_type_traits<TensorR64F>::Type);   // double precision (64-bit) real values
    case TensorC64F:  return sizeof(data_type_traits<TensorC64F>::Type);   // double precision (64-bit) complex values
    }
}
// clang-format on

} // namespace framework::tensor
