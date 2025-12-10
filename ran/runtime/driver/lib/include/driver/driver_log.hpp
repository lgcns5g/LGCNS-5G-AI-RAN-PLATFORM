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

#ifndef RAN_DRIVER_LOG_HPP
#define RAN_DRIVER_LOG_HPP

#include "log/components.hpp"

namespace ran::driver {

/**
 * Driver logging components
 */
DECLARE_LOG_COMPONENT(DriverComponent, Driver, PuschPipelineContext);

/**
 * Driver event logging identifiers
 */
DECLARE_LOG_EVENT(
        DriverEvent,
        CreatePuschPipeline,
        SlotResponseReceived,
        ResetSlotStatus,
        LaunchPipelines,
        UlIndication,
        AllocatePuschInputMemory,
        GetPipelineResource,
        ReleasePipelineResource,
        GetHostBuffers,
        ReleaseHostBuffers,
        SaveHostBuffersIndex,
        SavePipelineIndex,
        GetSlotResources,
        ClearSlotResources,
        PrepareInputData);

} // namespace ran::driver

#endif // RAN_DRIVER_LOG_HPP
