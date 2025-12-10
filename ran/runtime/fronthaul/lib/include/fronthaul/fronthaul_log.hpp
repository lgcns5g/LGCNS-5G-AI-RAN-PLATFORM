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

// Logging components for Fronthaul (U-plane and C-plane)
// Enables component-based logging with RT_LOGC_* macros

#ifndef RAN_FRONTHAUL_LOG_HPP
#define RAN_FRONTHAUL_LOG_HPP

#include "log/components.hpp"

namespace ran::fronthaul {

/**
 * Declare logging components for fronthaul subsystem
 */
DECLARE_LOG_COMPONENT(
        FronthaulLog,
        FronthaulGeneral,
        FronthaulParser,
        FronthaulTiming,
        FronthaulNetwork,
        FapiFileReplay);

/**
 * Fronthaul Kernels Logging Components
 *
 * Defines component categories for fronthaul U-plane (order kernel).
 * Organized by functional area for filtering and organization.
 *
 * Order Kernel (U-plane) Components:
 * - OrderModule: OrderKernelModule lifecycle, port configuration, execution
 * - OrderPipeline: OrderKernelPipeline setup, routing, graph management
 * - OrderFactory: Module and pipeline factory operations
 * - OrderMemory: Memory allocation, GDRCopy buffer management, descriptors
 * - OrderKernel: CUDA kernel launch, parameters, device function calls
 * - OrderDoca: DOCA RX queue interaction, semaphore handling, packet processing
 *
 */
DECLARE_LOG_COMPONENT(
        FronthaulKernels,
        OrderModule,
        OrderPipeline,
        OrderFactory,
        OrderMemory,
        OrderKernel,
        OrderDoca);

/**
 * Fronthaul Application Logging Components
 *
 * Defines component categories for fronthaul sample application.
 * Used for application-level logging in samples/ directory.
 *
 * Components:
 * - App: CLI parsing, initialization, application lifecycle
 * - UPlane: U-Plane message processing
 * - CPlane: C-Plane message processing
 * - Config: YAML configuration parsing
 * - Stats: Statistics display and reporting
 *
 */
DECLARE_LOG_COMPONENT(FronthaulApp, App, UPlane, CPlane, Config, Stats);

} // namespace ran::fronthaul

#endif // RAN_FRONTHAUL_LOG_HPP
