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

#ifndef RAN_I_FAPI_SLOT_INFO_PROVIDER_HPP
#define RAN_I_FAPI_SLOT_INFO_PROVIDER_HPP

#include <span>

#include "fapi/fapi_file_writer.hpp"
#include "fapi/fapi_state.hpp"

namespace ran::message_adapter {

/**
 * Interface for accessing slot information and captured FAPI messages
 *
 * This interface provides read-only access to:
 * - Current slot information (SFN and slot number)
 * - Accumulated FAPI messages (specifically UL-TTI-REQUEST messages) for a slot
 *
 * Used by C-plane and U-plane tasks to retrieve slot state and messages
 * after graph scheduling.
 *
 * Thread safety: The provider implementation must ensure thread-safe access
 * to the slot information and message collection. Typically, the graph scheduling
 * mechanism provides happens-before guarantees that make the data safe to read.
 */
class IFapiSlotInfoProvider {
public:
    /**
     * Virtual destructor for proper cleanup
     */
    virtual ~IFapiSlotInfoProvider() = default;

    /**
     * Get current slot information (SFN and slot number)
     *
     * Returns the current SFN and slot being processed by the Message Adapter.
     * Used by C-plane to calculate absolute slot counter.
     *
     * Thread-safe: Uses atomic operations internally.
     *
     * @return ran::fapi::SlotInfo with sfn in [0, 1023] and slot in [0, 19] for 30kHz SCS
     */
    [[nodiscard]] virtual ran::fapi::SlotInfo get_current_slot() const = 0;

    /**
     * Get accumulated UL-TTI messages for current slot
     *
     * Returns a non-owning view of the accumulated messages. The span
     * remains valid until the next slot's messages are accumulated.
     * Messages are in the order they were received.
     *
     * @param[in] slot Slot number (0-19 for 30kHz SCS) to get accumulated messages for
     * @return Span of captured FAPI messages (may be empty if no messages)
     */
    [[nodiscard]] virtual std::span<const ran::fapi::CapturedFapiMessage>
    get_accumulated_ul_tti_msgs(std::uint16_t slot) const = 0;

    /**
     * Get absolute slot number for given slot info
     *
     * Calculates absolute slot number accounting for SFN wrap-arounds.
     * The absolute slot is a monotonic counter that never wraps.
     *
     * @param[in] slot_info Slot information containing SFN and slot
     * @return Absolute slot number since initialization
     */
    [[nodiscard]] virtual std::uint64_t
    get_current_absolute_slot(ran::fapi::SlotInfo slot_info) const noexcept = 0;

    /**
     * Copy constructor (disabled for abstract base)
     */
    IFapiSlotInfoProvider(const IFapiSlotInfoProvider &) = delete;

    /**
     * Assignment operator (disabled for abstract base)
     */
    IFapiSlotInfoProvider &operator=(const IFapiSlotInfoProvider &) = delete;

    /**
     * Move constructor (disabled for abstract base)
     */
    IFapiSlotInfoProvider(IFapiSlotInfoProvider &&) = delete;

    /**
     * Move assignment operator (disabled for abstract base)
     */
    IFapiSlotInfoProvider &operator=(IFapiSlotInfoProvider &&) = delete;

protected:
    /**
     * Protected constructor for abstract base
     */
    IFapiSlotInfoProvider() = default;
};

} // namespace ran::message_adapter

#endif // RAN_I_FAPI_SLOT_INFO_PROVIDER_HPP
