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

#include <cstddef>
#include <cstdint>
#include <exception>
#include <format>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <quill/LogMacros.h>
#include <scf_5g_fapi.h>

#include "fapi/fapi_buffer.hpp"
#include "fapi/fapi_file_reader.hpp"
#include "fapi/fapi_file_replay.hpp"
#include "fapi/fapi_file_writer.hpp"
#include "fapi/fapi_log.hpp"
#include "log/rt_log_macros.hpp"

namespace ran::fapi {

namespace {

constexpr std::uint16_t SUBFRAMES_PER_FRAME = 10;

/**
 * Calculate subframe and slot from slot-in-frame
 *
 * @param[in] slots_in_frame Slot offset within frame
 * @param[in] slots_per_subframe Slots per subframe
 * @return Pair of (subframe_id, slot_id)
 */
[[nodiscard]] auto calculate_subframe_slot(
        const std::uint64_t slots_in_frame, const std::uint8_t slots_per_subframe) noexcept {
    const auto subframe_id = static_cast<std::uint8_t>(slots_in_frame / slots_per_subframe);
    const auto slot_id = static_cast<std::uint8_t>(slots_in_frame % slots_per_subframe);
    return std::pair{subframe_id, slot_id};
}

} // namespace

FapiFileReplay::FapiFileReplay(
        // NOLINTNEXTLINE(modernize-pass-by-value)
        const std::string &fapi_file_path,
        const std::uint8_t slots_per_subframe)
        : fapi_file_path_(fapi_file_path), slots_per_subframe_(slots_per_subframe) {

    try {
        load_ul_tti_requests_from_file(fapi_file_path_);
    } catch (const std::exception &e) {
        const std::string error_msg =
                std::format("Failed to load FAPI capture file {}: {}", fapi_file_path_, e.what());
        RT_LOGC_ERROR(FapiComponent::FapiFileReplay, "{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    if (fapi_cell_request_buffers_.empty()) {
        const std::string error_msg = "No UL_TTI_REQUEST messages found in FAPI capture file";
        RT_LOGC_ERROR(FapiComponent::FapiFileReplay, "{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    // Initialize request indices and compute cell IDs for each cell
    cell_ids_.reserve(fapi_cell_request_buffers_.size());
    for (const auto &[cell_id, request_buffers] : fapi_cell_request_buffers_) {
        if (request_buffers.empty()) {
            const std::string error_msg = std::format("Cell {} has no request buffers", cell_id);
            RT_LOGC_ERROR(FapiComponent::FapiFileReplay, "{}", error_msg);
            throw std::logic_error(error_msg);
        }
        cell_states_[cell_id] = CellState{};
        cell_ids_.push_back(cell_id);
    }
}

void FapiFileReplay::load_ul_tti_requests_from_file(const std::string &fapi_file_path) {
    ran::fapi::FapiFileReader reader(fapi_file_path);

    while (auto msg = reader.read_next()) {
        // Filter for UL_TTI_REQUEST messages only
        if (msg->msg_id != SCF_FAPI_UL_TTI_REQUEST) {
            continue;
        }

        // Skip empty messages
        if (msg->msg_data.empty()) {
            continue;
        }

        const std::uint16_t cell_id = msg->cell_id;

        // Copy message data to buffer
        std::vector<std::uint8_t> msg_buffer(msg->msg_data.begin(), msg->msg_data.end());

        // Add to cell's request buffer vector
        fapi_cell_request_buffers_[cell_id].push_back(std::move(msg_buffer));
    }
}

std::uint64_t FapiFileReplay::advance_slot() {
    // Calculate timing for the previous slot (before advancing) to check if current requests
    // matched
    const std::uint64_t slots_per_frame =
            static_cast<std::uint64_t>(SUBFRAMES_PER_FRAME) * slots_per_subframe_;
    const std::uint64_t slots_in_frame = current_absolute_slot_ % slots_per_frame;
    const auto [previous_subframe_id, previous_slot_id] =
            calculate_subframe_slot(slots_in_frame, slots_per_subframe_);

    // Advance to next slot
    ++current_absolute_slot_;

    // For each cell, check if current request matched the previous slot
    // If it matched, that request was consumed, so advance to next request
    for (auto &[cell_id, cell_state] : cell_states_) {
        const auto &request_buffers = fapi_cell_request_buffers_.at(cell_id);
        const std::size_t request_buffer_size = request_buffers.size();
        if (request_buffer_size == 0) {
            RT_LOGC_WARN(FapiComponent::FapiFileReplay, "Cell {} has no request buffers", cell_id);
            continue;
        }

        const auto &buffer = request_buffers.at(cell_state.current_request_index);
        const auto *request = fapi::assume_cast<scf_fapi_ul_tti_req_t>(buffer.data());

        // Check if request matched the previous slot timing (subframe and slot only, ignore SFN)
        const auto [req_subframe, req_slot] =
                calculate_subframe_slot(request->slot, slots_per_subframe_);

        const bool slot_matched =
                (req_subframe == previous_subframe_id) && (req_slot == previous_slot_id);

        if (slot_matched) {
            // Request matched the previous slot - it was consumed, advance to next request (with
            // wraparound)
            cell_state.current_request_index =
                    (cell_state.current_request_index + 1) % request_buffer_size;
        }
        // If no match, DON'T advance the index - might be a DL-only slot or gap in data
        // We'll check this same request again in the new slot
    }

    return current_absolute_slot_;
}

std::optional<FapiFileReplay::RequestWithSize>
FapiFileReplay::get_request_for_current_slot(const std::uint16_t cell_id) {
    // Check if cell exists
    const auto cell_it = fapi_cell_request_buffers_.find(cell_id);
    if (cell_it == fapi_cell_request_buffers_.end()) {
        const std::string error_msg =
                std::format("Cell ID {} not found in FAPI replay data", cell_id);
        RT_LOGC_ERROR(FapiComponent::FapiFileReplay, "{}", error_msg);
        throw std::invalid_argument(error_msg);
    }

    auto &request_buffers = cell_it->second;
    const auto &current_index = cell_states_.at(cell_id).current_request_index;
    const std::size_t request_buffer_size = request_buffers.size();
    if (request_buffer_size == 0) {
        RT_LOGC_WARN(FapiComponent::FapiFileReplay, "Cell {} has no request buffers", cell_id);
        return std::nullopt;
    }

    // Get current request buffer (using .at() for bounds checking)
    auto &buffer = request_buffers.at(current_index);
    auto *request = fapi::assume_cast<scf_fapi_ul_tti_req_t>(buffer.data());

    // Calculate current slot timing from absolute slot
    const std::uint64_t slots_per_frame =
            static_cast<std::uint64_t>(SUBFRAMES_PER_FRAME) * slots_per_subframe_;
    const std::uint64_t slots_in_frame = current_absolute_slot_ % slots_per_frame;
    const auto [current_subframe_id, current_slot_id] =
            calculate_subframe_slot(slots_in_frame, slots_per_subframe_);

    // Check if request matches current slot timing (subframe and slot only, ignore SFN)
    const auto [req_subframe, req_slot] =
            calculate_subframe_slot(request->slot, slots_per_subframe_);

    const bool slot_matches =
            (req_subframe == current_subframe_id) && (req_slot == current_slot_id);

    if (!slot_matches) {
        // Current slot doesn't match - might be a DL-only slot or gap in data
        return std::nullopt;
    }

    // Match found. Update the FAPI buffer timing to match current replay position
    // This is critical: the SFN in the capture file may not match our replay position
    // Calculate frame_id with proper uint8_t wrapping
    static constexpr std::uint16_t FAPI_SFN_MAX = 1024;
    const std::uint64_t frames = current_absolute_slot_ / slots_per_frame;
    const auto frame_id = static_cast<std::uint8_t>((frames % FAPI_SFN_MAX));

    request->sfn = frame_id;
    request->slot =
            static_cast<std::uint16_t>(current_subframe_id * slots_per_subframe_ + current_slot_id);

    // Validate buffer size before calculating body_len
    if (buffer.size() < sizeof(scf_fapi_body_header_t)) {
        const std::string error_msg = std::format(
                "Invalid FAPI message buffer size {} is smaller than FAPI body header size {} for "
                "cell {}",
                buffer.size(),
                sizeof(scf_fapi_body_header_t),
                cell_id);
        RT_LOGC_ERROR(FapiComponent::FapiFileReplay, "{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    // Calculate body_len: total buffer size minus FAPI body header
    // This matches the pattern used in tests and fapi_state processing
    const std::size_t body_len = buffer.size() - sizeof(scf_fapi_body_header_t);

    return RequestWithSize{request, body_len};
}

FapiSlotTiming FapiFileReplay::get_current_slot_timing() const noexcept {
    return FapiSlotTiming{current_absolute_slot_, slots_per_subframe_};
}

const std::vector<std::uint16_t> &FapiFileReplay::get_cell_ids() const noexcept {
    return cell_ids_;
}

std::size_t FapiFileReplay::get_request_count(const std::uint16_t cell_id) const noexcept {
    const auto it = fapi_cell_request_buffers_.find(cell_id);
    if (it == fapi_cell_request_buffers_.end()) {
        return 0;
    }
    return it->second.size();
}

std::size_t FapiFileReplay::get_total_request_count() const noexcept {
    std::size_t total{0};
    for (const auto &[_, request_buffers] : fapi_cell_request_buffers_) {
        total += request_buffers.size();
    }
    return total;
}

} // namespace ran::fapi
