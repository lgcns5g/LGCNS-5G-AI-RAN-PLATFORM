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

#ifndef FRAMEWORK_TRT_ENGINE_PARAMS_HPP
#define FRAMEWORK_TRT_ENGINE_PARAMS_HPP

#include <array>
#include <cstddef>
#include <string>

#include <gsl-lite/gsl-lite.hpp>

#include "tensor/data_types.hpp"
#include "utils/error_macros.hpp"

namespace framework::tensorrt {

/**
 * @brief Tensor parameters for MLIR-TensorRT engines
 *
 * This structure provides tensor parameter representation for MLIR-TensorRT
 * engines where tensor shapes and strides are provided by the user during
 * initialization.
 *
 * @details The user must provide the rank and dimensions. Strides are optional:
 * - If strides are provided (last stride != 0), they are used as-is
 * - If strides are not provided (last stride == 0), row-major strides are
 *   automatically computed from dimensions
 *
 * @note Maximum rank is 8, minimum rank is 0 (scalar), matching MLIR-TensorRT implementation limits
 *
 * @see tensor::NvDataType for supported data type enumeration
 */
struct MLIRTensorParams final {
    static constexpr std::size_t MAX_TENSOR_RANK = 8; //!< Maximum supported tensor rank

    /**
     * Set the number of dimensions for this tensor (rank)
     * @param[in] n Number of dimensions (must be <= MAX_TENSOR_RANK)
     */
    constexpr void set_rank(const std::size_t n) noexcept(!utils::GSL_CONTRACT_THROWS) {
        gsl_Expects(n <= MAX_TENSOR_RANK);
        rank = n;
    }

    std::string name;               //!< Tensor name identifier
    tensor::NvDataType data_type{}; //!< Data type of tensor elements
    std::size_t rank{};             //!< Number of dimensions (0 for scalar, 1-8 for tensors)
    std::array<std::size_t, MAX_TENSOR_RANK>
            dims{}; //!< Tensor dimensions (first rank elements valid)
    std::array<std::size_t, MAX_TENSOR_RANK> strides{}; //!< Tensor strides (first rank elements
                                                        //!< valid, auto-computed if not provided)
};

} // namespace framework::tensorrt

#endif // FRAMEWORK_TRT_ENGINE_PARAMS_HPP
