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

#ifndef FRAMEWORK_VARIANT_TYPES_HPP
#define FRAMEWORK_VARIANT_TYPES_HPP

#include <variant>

#include <cuComplex.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace framework::utils {

/**
 * Variant type containing all possible types from cuphyVariant_t union
 *
 * This type-safe variant replaces the C-style union used in cuphyVariant_t,
 * providing compile-time type safety and modern C++ semantics.
 */
using VariantTypes = std::variant<
        unsigned int,   // CUPHY_BIT and CUPHY_R_32U (b1, r32u)
        signed char,    // CUPHY_R_8I (r8i)
        char2,          // CUPHY_C_8I (c8i)
        unsigned char,  // CUPHY_R_8U (r8u)
        uchar2,         // CUPHY_C_8U (c8u)
        short,          // CUPHY_R_16I (r16i)
        short2,         // CUPHY_C_16I (c16i)
        unsigned short, // CUPHY_R_16U (r16u)
        ushort2,        // CUPHY_C_16U (c16u)
        int,            // CUPHY_R_32I (r32i)
        int2,           // CUPHY_C_32I (c32i)
        uint2,          // CUPHY_C_32U (c32u)
        __half_raw,     // CUPHY_R_16F (r16f)
        __half2_raw,    // CUPHY_C_16F (c16f)
        float,          // CUPHY_R_32F (r32f)
        cuComplex,      // CUPHY_C_32F (c32f)
        double,         // CUPHY_R_64F (r64f)
        cuDoubleComplex // CUPHY_C_64F (c64f)
        >;

} // namespace framework::utils

#endif // FRAMEWORK_VARIANT_TYPES_HPP
