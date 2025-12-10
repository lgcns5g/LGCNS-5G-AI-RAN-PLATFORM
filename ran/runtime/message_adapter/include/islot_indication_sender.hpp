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

#ifndef RAN_I_SLOT_INDICATION_SENDER_HPP
#define RAN_I_SLOT_INDICATION_SENDER_HPP

namespace ran::message_adapter {

/**
 * Interface for sending slot indications
 *
 * This interface abstracts the slot indication mechanism, allowing the
 * TimedTrigger to send periodic slot indications without depending on
 * the concrete implementation. Implementations should send slot indications
 * to all active cells.
 *
 * Thread safety: Implementations must be thread-safe as this may be called
 * from a timer thread.
 */
class ISlotIndicationSender {
public:
    /**
     * Virtual destructor for proper cleanup
     */
    virtual ~ISlotIndicationSender() = default;

    /**
     * Send slot indications to all active cells
     *
     * Called periodically (typically every 500µs) to advance the slot counter
     * and send slot indication messages to all cells in running state.
     */
    virtual void send_slot_indications() = 0;

    /**
     * Copy constructor (disabled for abstract base)
     */
    ISlotIndicationSender(const ISlotIndicationSender &) = delete;

    /**
     * Assignment operator (disabled for abstract base)
     */
    ISlotIndicationSender &operator=(const ISlotIndicationSender &) = delete;

    /**
     * Move constructor (disabled for abstract base)
     */
    ISlotIndicationSender(ISlotIndicationSender &&) = delete;

    /**
     * Move assignment operator (disabled for abstract base)
     */
    ISlotIndicationSender &operator=(ISlotIndicationSender &&) = delete;

protected:
    /**
     * Protected constructor for abstract base
     */
    ISlotIndicationSender() = default;
};

} // namespace ran::message_adapter

#endif // RAN_I_SLOT_INDICATION_SENDER_HPP
