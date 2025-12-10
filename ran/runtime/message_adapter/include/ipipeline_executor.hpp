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

#ifndef RAN_I_PIPELINE_EXECUTOR_HPP
#define RAN_I_PIPELINE_EXECUTOR_HPP

#include <cstddef>

namespace ran::message_adapter {

/**
 * Interface for executing PUSCH pipelines
 *
 * This interface provides access to pipeline execution functionality,
 * allowing PUSCH RX tasks to trigger pipeline processing for a given slot.
 *
 * Used by PUSCH task to launch pipeline execution after U-Plane processing
 * completes.
 *
 * Thread safety: The provider implementation must ensure thread-safe access
 * to the pipeline execution mechanism.
 */
class IPipelineExecutor {
public:
    /**
     * Virtual destructor for proper cleanup
     */
    virtual ~IPipelineExecutor() = default;

    /**
     * Launch PUSCH pipelines for the given slot
     *
     * Triggers execution of the PUSCH pipeline(s) for the specified slot.
     * This should be called by the PUSCH RX task after U-Plane processing
     * has prepared the I/Q data.
     *
     * Thread-safe: Implementation uses appropriate synchronization.
     *
     * @param[in] slot Slot number to process (0-19 for 30kHz SCS)
     */
    virtual void launch_pipelines(std::size_t slot) = 0;

    /**
     * Copy constructor (disabled for abstract base)
     */
    IPipelineExecutor(const IPipelineExecutor &) = delete;

    /**
     * Assignment operator (disabled for abstract base)
     */
    IPipelineExecutor &operator=(const IPipelineExecutor &) = delete;

    /**
     * Move constructor (disabled for abstract base)
     */
    IPipelineExecutor(IPipelineExecutor &&) = delete;

    /**
     * Move assignment operator (disabled for abstract base)
     */
    IPipelineExecutor &operator=(IPipelineExecutor &&) = delete;

protected:
    /**
     * Protected constructor for abstract base
     */
    IPipelineExecutor() = default;
};

} // namespace ran::message_adapter

#endif // RAN_I_PIPELINE_EXECUTOR_HPP
