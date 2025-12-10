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

#ifndef RAN_PIPELINE_INTERFACE_HPP
#define RAN_PIPELINE_INTERFACE_HPP

#include <functional>
#include <memory>
#include <string>

#include <nv_ipc.hpp>
namespace ran::message_adapter {
/**
 * PHY-MAC message descriptor
 *
 * Extends nv_ipc_msg_t to provide additional functionality for
 * PHY-MAC layer message handling with reset capability.
 */
struct PhyMacMsgDesc : public nv_ipc_msg_t {
    /**
     * Default constructor - initializes and resets all fields
     */
    PhyMacMsgDesc() : nv_ipc_msg_t() { reset(); }

    /**
     * Construct from existing Ipc message
     *
     * @param[in] msg Source Ipc message to copy
     */
    explicit PhyMacMsgDesc(nv_ipc_msg_t msg) : nv_ipc_msg_t(msg) {
        // Base class is initialized with the parameter
        // All members are already set through the base class copy constructor
    }

public:
    /**
     * Reset all message fields to default values
     */
    void reset() {
        cell_id = msg_id = msg_len = data_len = 0;
        data_pool = NV_IPC_MEMPOOL_CPU_MSG;
        msg_buf = data_buf = nullptr;
    }
};

/**
 * Abstract interface for pipeline components
 *
 * This interface defines the contract for pipeline components that can be
 * managed by the message adapter. Derived classes should implement the
 * specific pipeline functionality.
 */
class PipelineInterface {
public:
    /**
     * Public constructor for abstract base
     */
    PipelineInterface() = default;

    /**
     * Virtual destructor for proper cleanup
     */
    virtual ~PipelineInterface() = default;

    /**
     * Process incoming message
     *
     * @param[in,out] msg Input message to be processed
     * @param[in] ipc Pointer to Ipc interface
     */
    virtual void process_msg(nv_ipc_msg_t &msg, nv_ipc_t *ipc) = 0;

    /**
     * Copy constructor (disabled for abstract base)
     */
    PipelineInterface(const PipelineInterface &) = delete;

    /**
     * Assignment operator (disabled for abstract base)
     */
    PipelineInterface &operator=(const PipelineInterface &) = delete;

    /**
     * Move constructor (disabled for abstract base)

     */
    PipelineInterface(PipelineInterface &&) = delete;

    /**
     * Move assignment operator (disabled for abstract base)
     */
    PipelineInterface &operator=(PipelineInterface &&) = delete;
};

/**
 * Smart pointer type for pipeline components
 */
using PipelinePtr = std::unique_ptr<PipelineInterface>;

/**
 * Factory function type for creating pipeline components
 */
using PipelineFactory = std::function<PipelinePtr(const std::string &)>;

} // namespace ran::message_adapter

#endif // RAN_PIPELINE_INTERFACE_HPP
