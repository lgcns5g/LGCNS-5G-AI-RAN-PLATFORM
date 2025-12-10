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

/**
 * @file task_factories.cpp
 * @brief Implementation of C-Plane and U-Plane task factory functions
 *
 * Logic copied from fronthaul_app_utils.cpp (lines 266-334).
 * Phase 1: C-plane task uses IFapiSlotInfoProvider for slot info and accumulated FAPI messages.
 */

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <span>
#include <vector>

#include <quill/LogMacros.h>
#include <scf_5g_fapi.h>

#include "fapi/fapi_buffer.hpp"
#include "fapi/fapi_file_writer.hpp"
#include "fapi/fapi_state.hpp"
#include "fronthaul/fronthaul.hpp"
#include "ifapi_slot_info_provider.hpp"
#include "ipipeline_executor.hpp"
#include "islot_indication_sender.hpp"
#include "log/rt_log_macros.hpp"
#include "oran/numerology.hpp"
#include "phy_ran_app/fapi_rx_handler.hpp"
#include "phy_ran_app/phy_ran_app_log.hpp"
#include "phy_ran_app/task_factories.hpp"

namespace ran::phy_ran_app {

std::function<void()> make_process_cplane_func(
        ran::fronthaul::Fronthaul &fronthaul,
        ran::message_adapter::IFapiSlotInfoProvider &slot_info_provider,
        std::atomic<ran::fapi::SlotInfo> &slot_info,
        const std::chrono::nanoseconds t0,
        const std::chrono::nanoseconds tai_offset) {
    // SAFETY: Lambda captures fronthaul, slot_info_provider, and slot_info by reference.
    // Caller must ensure these objects outlive the returned function.
    // processing_count is mutable capture to track C-Plane processing cycles for logging
    return [&fronthaul,
            &slot_info_provider,
            &slot_info,
            t0,
            tai_offset,
            processing_count = std::uint64_t{0}]() mutable {
        const auto current_sfn_slot = slot_info.load(std::memory_order_acquire);
        RT_LOGC_INFO(
                PhyRanApp::CPlane,
                "C-plane task started for sfn {} slot {}",
                current_sfn_slot.sfn,
                current_sfn_slot.slot);

        // Get absolute slot number accounting for SFN wrap-arounds
        const std::uint64_t absolute_slot =
                slot_info_provider.get_current_absolute_slot(current_sfn_slot);

        // Phase 1: Get current slot and accumulated messages from slot info provider
        const auto &accumulated_msgs =
                slot_info_provider.get_accumulated_ul_tti_msgs(current_sfn_slot.slot);

        // Log how many messages we're processing
        RT_LOGC_INFO(
                PhyRanApp::CPlane,
                "C-plane processing {} accumulated UL-TTI messages for sfn {} slot {} "
                "absolute_slot {}",
                accumulated_msgs.size(),
                current_sfn_slot.sfn,
                current_sfn_slot.slot,
                absolute_slot);

        // Phase 1: Loop through accumulated messages and send C-plane for each
        for (const auto &captured_msg : accumulated_msgs) {
            // Extract UL_TTI_REQUEST from captured message
            const auto *body_hdr =
                    fapi::assume_cast<const scf_fapi_body_header_t>(captured_msg.msg_data.data());
            const auto *ul_tti_request = fapi::assume_cast<const scf_fapi_ul_tti_req_t>(body_hdr);

            // Extract sfn/slot from the message
            const auto sfn = ul_tti_request->sfn;
            const auto slot = ul_tti_request->slot;

            // Send C-plane for this message
            fronthaul.send_ul_cplane(
                    *ul_tti_request,
                    body_hdr->length,
                    captured_msg.cell_id,
                    absolute_slot,
                    t0,
                    tai_offset);

            // Log Phase 1 marker for verification
            RT_LOGC_INFO(
                    PhyRanApp::CPlane,
                    "Sent C-plane: cell={} sfn={} slot={}",
                    captured_msg.cell_id,
                    sfn,
                    slot);
        }

        // Track C-Plane processing cycles (local counter, no atomic synchronization)
        ++processing_count;

        // Log periodic status every 100 C-Plane processing cycles
        static constexpr std::uint64_t LOG_INTERVAL = 100;
        if (processing_count % LOG_INTERVAL == 0) {
            RT_LOGC_DEBUG(
                    PhyRanApp::Stats,
                    "C-Plane processing status: {} cycles completed",
                    processing_count);
        }
    };
}

std::function<void()> make_process_uplane_func(
        ran::fronthaul::Fronthaul &fronthaul,
        ran::message_adapter::IFapiSlotInfoProvider &slot_info_provider,
        std::atomic<ran::fapi::SlotInfo> &slot_info) {
    // SAFETY: Lambda captures fronthaul, slot_info_provider, and slot_info by reference.
    // Caller must ensure objects outlive the returned function.
    return [&fronthaul, &slot_info_provider, &slot_info]() {
        const auto current_sfn_slot = slot_info.load(std::memory_order_acquire);
        RT_LOGC_INFO(
                PhyRanApp::UPlane,
                "U-plane task started for slot sfn={} slot={}",
                current_sfn_slot.sfn,
                current_sfn_slot.slot);

        // Phase 1: Check if current slot has UL data by checking accumulated messages
        // U-plane processing should only occur for slots that had C-plane sent
        const auto &accumulated_msgs =
                slot_info_provider.get_accumulated_ul_tti_msgs(current_sfn_slot.slot);

        if (accumulated_msgs.empty()) {
            // No UL data for this slot - skip U-plane processing
            // This prevents launching DOCA kernels for slots without FAPI data
            return;
        }

        // Get absolute slot number accounting for SFN wrap-arounds
        const std::uint64_t absolute_slot =
                slot_info_provider.get_current_absolute_slot(current_sfn_slot);

        // Convert to ORAN slot timing (frame, subframe, slot)
        const auto oran_slot_timing =
                fronthaul.config().numerology.calculate_slot_timing(absolute_slot);

        RT_LOGC_INFO(
                PhyRanApp::UPlane,
                "U-plane processing: sfn={} slot={} (absolute_slot={}, {} accumulated messages)",
                current_sfn_slot.sfn,
                current_sfn_slot.slot,
                absolute_slot,
                accumulated_msgs.size());

        try {
            fronthaul.process_uplane(oran_slot_timing);
        } catch (const std::exception &e) {
            RT_LOGC_ERROR(PhyRanApp::UPlane, "Failed to process U-Plane: {}", e.what());
        }
    };
}

std::function<void()> make_process_pusch_func(
        ran::message_adapter::IPipelineExecutor &pipeline_executor,
        std::atomic<ran::fapi::SlotInfo> &slot_info) {
    // SAFETY: Lambda captures pipeline_executor and slot_info by reference.
    // Caller must ensure objects outlive the returned function.
    return [&pipeline_executor, &slot_info]() {
        RT_LOGC_INFO(PhyRanApp::PuschRx, "PUSCH task started");

        const auto current_sfn_slot = slot_info.load(std::memory_order_acquire);
        // Phase 1: Get current slot from Sample5GPipeline
        RT_LOGC_INFO(
                PhyRanApp::PuschRx,
                "PUSCH task got current slot: sfn={} slot={}",
                current_sfn_slot.sfn,
                current_sfn_slot.slot);

        // Phase 1: Launch PUSCH pipeline execution for this slot
        // Driver will execute the pipeline and call the UL indication callback
        pipeline_executor.launch_pipelines(current_sfn_slot.slot);

        RT_LOGC_INFO(
                PhyRanApp::PuschRx,
                "PUSCH processing launched: sfn={} slot={}",
                current_sfn_slot.sfn,
                current_sfn_slot.slot);
    };
}

std::function<void()> make_slot_indication_func(
        ran::message_adapter::ISlotIndicationSender &slot_sender,
        const ran::phy_ran_app::FapiRxHandler &fapi_rx_handler,
        std::atomic_bool &running) {
    // SAFETY: Lambda captures slot_sender, fapi_rx_handler, and running by reference.
    // Caller must ensure these objects outlive the returned function.
    // slot_count is mutable capture to track number of indications sent
    return [&slot_sender, &fapi_rx_handler, &running, slot_count = std::size_t{0}]() mutable {
        const std::size_t num_cells = fapi_rx_handler.get_num_cells_running();

        // Check for shutdown condition: all cells stopped during runtime
        // Note: main() guarantees cells are running before trigger starts,
        // so num_cells == 0 means cells stopped after startup
        if (num_cells == 0) {
            if (running.load()) {
                RT_LOGC_INFO(PhyRanApp::SlotIndication, "All cells stopped, requesting shutdown");
                running.store(false);
            }
            return;
        }

        // Send slot indications to all running cells via Message Adapter
        slot_sender.send_slot_indications();

        // Increment local counter
        ++slot_count;

        // Log periodic status every 100 slot indications
        static constexpr std::size_t LOG_INTERVAL = 100;
        if (slot_count % LOG_INTERVAL == 0) {
            RT_LOGC_DEBUG(
                    PhyRanApp::Stats, "Slot indication status: {} indications sent", slot_count);
        }
    };
}

} // namespace ran::phy_ran_app
