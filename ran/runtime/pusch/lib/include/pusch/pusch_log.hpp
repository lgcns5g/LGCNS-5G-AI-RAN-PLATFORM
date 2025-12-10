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

#ifndef RAN_PUSCH_LOG_HPP
#define RAN_PUSCH_LOG_HPP

#include "log/components.hpp"

namespace ran::pusch {

/**
 * PUSCH logging components
 */
DECLARE_LOG_COMPONENT(
        PuschComponent, PuschPipeline, PuschModuleFactory, InnerRxModuleFactory, InnerRxModule);

/**
 * PUSCH pipeline event logging identifiers
 */
DECLARE_LOG_EVENT(
        PuschPipelineEvent,
        CreatePipeline,
        CreateModules,
        PipelineSetup,
        PipelineWarmup,
        PipelineConfigureIo,
        PipelineExecuteStream,
        PipelineBuildGraph,
        PipelineExecuteGraph,
        ModuleSetupMemory,
        ModuleSetInputs,
        ModuleWarmup,
        ModuleConfigureIo,
        ModuleGetOutputs,
        ModuleExecute,
        ModuleAddNodeToGraph);

/**
 * PUSCH error event logging identifiers
 */
DECLARE_LOG_EVENT(PuschErrorEvent, InvalidParam, InvalidState);

} // namespace ran::pusch

#endif // RAN_PUSCH_LOG_HPP
