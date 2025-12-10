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
 * @file oran_order_kernels.hpp
 * @brief CUDA kernel declarations for ORAN order processing
 *
 * This header declares the CUDA kernels used for ORAN fronthaul packet ordering
 * and processing. The kernel uses descriptor-based interface for configuration.
 */

#ifndef RAN_FRONTHAUL_ORAN_ORDER_KERNELS_HPP
#define RAN_FRONTHAUL_ORAN_ORDER_KERNELS_HPP

#include <cuda_runtime.h>

#include "fronthaul/order_kernel_descriptors.hpp"

namespace ran::fronthaul {

/**
 * Unified ORAN order kernel with descriptor-based interface
 *
 * Template parameters used by OrderKernelModule:
 * - ok_tb_enable = false (no test bench)
 * - ul_rx_pkt_tracing_level = 0 (no packet tracing)
 * - srs_enable = 0 (PUSCH only)
 * - NUM_THREADS = 320
 * - NUM_CTAS_PER_SM = 1
 *
 * @param[in] static_desc Static kernel parameters (GDRCopy buffers, DOCA handles)
 * @param[in] dynamic_desc Dynamic kernel parameters (timing, frame/slot IDs)
 */
template <bool, uint8_t, uint8_t, int, int>
// NOLINTNEXTLINE(readability-identifier-naming) - CUDA kernel name matches existing convention
__global__ void order_kernel_doca_single_subSlot_pingpong(
        const OrderKernelStaticDescriptor *static_desc,
        const OrderKernelDynamicDescriptor *dynamic_desc);

} // namespace ran::fronthaul

#endif // RAN_FRONTHAUL_ORAN_ORDER_KERNELS_HPP
