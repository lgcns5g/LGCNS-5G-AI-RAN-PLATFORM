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

#ifndef RAN_FAPI_FAPI_FILE_REPLAY_HPP
#define RAN_FAPI_FAPI_FILE_REPLAY_HPP

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <scf_5g_fapi.h>

#include "fapi/fapi_export.hpp"

namespace ran::fapi {

/**
 * Slot timing information for FAPI replay
 *
 * Contains raw timing data needed for conversion to ORAN slot timing.
 * This is independent of ORAN to avoid circular dependencies.
 */
struct FapiSlotTiming final {
    std::uint64_t absolute_slot{};     //!< Absolute slot number since slot 0
    std::uint8_t slots_per_subframe{}; //!< Slots per subframe (numerology-dependent)
};

/**
 * FAPI file replay state
 *
 * Manages replaying captured FAPI UL_TTI_REQUEST messages with proper
 * timing updates. Tracks current slot position and updates SFN/slot
 * fields in FAPI buffers to match current replay position.
 */
class FAPI_EXPORT FapiFileReplay final {
public:
    /**
     * Load FAPI UL_TTI_REQUEST messages from capture file
     *
     * @param[in] fapi_file_path Path to .fapi capture file
     * @param[in] slots_per_subframe Slots per subframe (numerology-dependent)
     * @throws std::runtime_error if file cannot be loaded or contains no valid messages
     */
    explicit FapiFileReplay(const std::string &fapi_file_path, std::uint8_t slots_per_subframe);

    /**
     * Advance to next slot
     *
     * Increments the absolute slot counter and updates frame/subframe/slot_id state.
     * For each cell, checks if the current request matched the previous slot (before advancing).
     * If a match is found, that request was consumed, so advances that cell's request index
     * to the next request (with wraparound).
     *
     * This centralizes the file state update in one place. After calling this method,
     * get_request_for_current_slot() will return requests for the new slot.
     *
     * This should be called once per slot by the application, not per cell.
     *
     * @return Current absolute slot number after increment
     */
    [[nodiscard]] std::uint64_t advance_slot();

    /**
     * Result of request lookup containing pointer and body length
     */
    struct RequestWithSize final {
        scf_fapi_ul_tti_req_t
                *request{}; //!< Non-owning pointer to request structure (owned by FapiFileReplay)
        std::size_t body_len{}; //!< Body length (buffer size minus body header)
    };

    /**
     * Get request for current slot and cell
     *
     * Returns the FAPI request buffer for the given cell that matches the current
     * slot timing. Updates the SFN and slot fields in the buffer to match current
     * replay position. Returns std::nullopt if no matching request or cell not found.
     *
     * This method can be called multiple times for the same slot without side effects
     * on the request index state (which is advanced by advance_slot()).
     *
     * @note This method updates the returned FAPI buffer's timing fields (sfn, slot)
     * to match the current replay position, but does not advance internal request indices.
     *
     * @param[in] cell_id Cell identifier
     * @return Optional RequestWithSize containing pointer and body_len, or std::nullopt if no match
     */
    [[nodiscard]] std::optional<RequestWithSize>
    get_request_for_current_slot(std::uint16_t cell_id);

    /**
     * Get all cell IDs present in loaded data
     *
     * @return Const reference to vector of cell IDs
     */
    [[nodiscard]] const std::vector<std::uint16_t> &get_cell_ids() const noexcept;

    /**
     * Get number of requests for a cell
     *
     * @param[in] cell_id Cell identifier
     * @return Number of requests, or 0 if cell not found
     */
    [[nodiscard]] std::size_t get_request_count(std::uint16_t cell_id) const noexcept;

    /**
     * Get total request count across all cells
     *
     * @return Total number of requests
     */
    [[nodiscard]] std::size_t get_total_request_count() const noexcept;

    /**
     * Get number of cells
     *
     * @return Cell count
     */
    [[nodiscard]] std::size_t get_cell_count() const noexcept {
        return fapi_cell_request_buffers_.size();
    }

    /**
     * Get current absolute slot number
     *
     * @return Current absolute slot
     */
    [[nodiscard]] std::uint64_t get_current_absolute_slot() const noexcept {
        return current_absolute_slot_;
    }

    /**
     * Get current slot timing
     *
     * @return Current frame/subframe/slot timing
     */
    [[nodiscard]] FapiSlotTiming get_current_slot_timing() const noexcept;

private:
    /**
     * Per-cell replay state
     */
    struct CellState final {
        std::size_t current_request_index{}; //!< Current position in request buffer vector
    };

    void load_ul_tti_requests_from_file(const std::string &fapi_file_path);

    std::string fapi_file_path_;
    std::uint8_t slots_per_subframe_{};

    // FAPI request buffers organized by cell (unordered for O(1) lookup)
    std::unordered_map<std::uint16_t, std::vector<std::vector<std::uint8_t>>>
            fapi_cell_request_buffers_;
    std::unordered_map<std::uint16_t, CellState> cell_states_;

    // Cached cell IDs computed once during construction
    std::vector<std::uint16_t> cell_ids_;

    // Current slot state for replay
    std::uint64_t current_absolute_slot_{0};
};

} // namespace ran::fapi

#endif // RAN_FAPI_FAPI_FILE_REPLAY_HPP
