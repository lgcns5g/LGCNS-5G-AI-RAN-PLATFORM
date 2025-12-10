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
 * @file gpunetio_kernels.hpp
 * @brief Header for GPUNetIO CUDA kernels
 */

#ifndef FRAMEWORK_NET_GPUNETIO_KERNELS_HPP
#define FRAMEWORK_NET_GPUNETIO_KERNELS_HPP

#include <cstdint>

#include <doca_error.h>
#include <doca_types.h>

#include <cuda_runtime.h>

#include "net/doca_types.hpp"

namespace framework::net {

/**
 * Launch kernel to send a single packet
 *
 * @param[in] stream CUDA stream for kernel execution
 * @param[in] txq TX queue structure containing GPU handles
 * @return 0 on success, < 0 otherwise
 */
[[nodiscard]] int launch_gpunetio_sender_kernel(cudaStream_t stream, const DocaTxQParams &txq);

/**
 * Launch kernel to receive a single packet and block until received
 *
 * @param[in] stream CUDA stream for kernel execution
 * @param[in] rxq RX queue structure containing GPU handles
 * @param[in] gpu_exit_condition GPU exit condition flag
 * @return 0 on success, < 0 otherwise
 */
[[nodiscard]] int launch_gpunetio_receiver_kernel(
        cudaStream_t stream, const DocaRxQParams &rxq, uint32_t *gpu_exit_condition);

} // namespace framework::net

#endif // FRAMEWORK_NET_GPUNETIO_KERNELS_HPP
