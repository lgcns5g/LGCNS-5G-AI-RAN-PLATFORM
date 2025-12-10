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

#ifndef RAN_I_FAPI_MESSAGE_PROCESSOR_HPP
#define RAN_I_FAPI_MESSAGE_PROCESSOR_HPP

#include <nv_ipc.hpp>

namespace ran::message_adapter {

/**
 * Interface for processing FAPI messages
 *
 * This interface defines the contract for FAPI message processing.
 * Implementations handle incoming FAPI messages (CONFIG_REQUEST,
 * START_REQUEST, UL_TTI_REQUEST, SLOT_RESPONSE, etc.) from the
 * FAPI RX task.
 *
 * Thread safety: Implementations must be thread-safe if called from
 * multiple threads, though typically called from single FAPI RX task.
 */
class IFapiMessageProcessor {
public:
    /**
     * Virtual destructor for proper cleanup
     */
    virtual ~IFapiMessageProcessor() = default;

    /**
     * Process incoming FAPI message
     *
     * Handles the incoming message based on its type (msg_id) and updates
     * internal state accordingly. The processor owns the nvIPC endpoint
     * and manages message responses internally.
     *
     * @param[in,out] msg FAPI message to process
     */
    virtual void process_msg(nv_ipc_msg_t &msg) = 0;

    /**
     * Copy constructor (disabled for abstract base)
     */
    IFapiMessageProcessor(const IFapiMessageProcessor &) = delete;

    /**
     * Assignment operator (disabled for abstract base)
     */
    IFapiMessageProcessor &operator=(const IFapiMessageProcessor &) = delete;

    /**
     * Move constructor (disabled for abstract base)
     */
    IFapiMessageProcessor(IFapiMessageProcessor &&) = delete;

    /**
     * Move assignment operator (disabled for abstract base)
     */
    IFapiMessageProcessor &operator=(IFapiMessageProcessor &&) = delete;

protected:
    /**
     * Protected constructor for abstract base
     */
    IFapiMessageProcessor() = default;
};

} // namespace ran::message_adapter

#endif // RAN_I_FAPI_MESSAGE_PROCESSOR_HPP
