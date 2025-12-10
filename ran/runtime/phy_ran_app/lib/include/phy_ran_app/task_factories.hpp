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
 * @file task_factories.hpp
 * @brief Factory functions for C-Plane and U-Plane processing tasks
 *
 * Adapted from fronthaul_app_utils.hpp to work with phy_ran_app architecture.
 * Tasks copied from fronthaul_app and modified to integrate with Message Adapter.
 */

#ifndef RAN_PHY_RAN_APP_TASK_FACTORIES_HPP
#define RAN_PHY_RAN_APP_TASK_FACTORIES_HPP

#include <atomic>
#include <chrono>
#include <functional>
#include <optional>

#include "fapi/fapi_state.hpp"

// Forward declarations
namespace ran::fronthaul {
class Fronthaul;
} // namespace ran::fronthaul

namespace ran::message_adapter {
class IFapiSlotInfoProvider;
class IPipelineExecutor;
class ISlotIndicationSender;
} // namespace ran::message_adapter

namespace ran::phy_ran_app {

// Forward declaration
class FapiRxHandler;

/**
 * @brief Create C-Plane processing task function
 *
 * Processes C-Plane messages and transmits to RU via DPDK.
 * Uses IFapiSlotInfoProvider to access slot info and accumulated FAPI messages.
 * Absolute slot is obtained from slot_info_provider (monotonic counter).
 *
 * @warning LIFETIME SAFETY: Captures fronthaul, slot_info_provider, and slot_info
 * by reference. Caller must ensure objects outlive returned function.
 *
 * @param[in] fronthaul Fronthaul instance reference
 * @param[in] slot_info_provider Slot info provider for slot state and accumulated FAPI messages
 * @param[in,out] slot_info Atomic slot info for current slot timing
 * @param[in] t0 Time for SFN 0, subframe 0, slot 0
 * @param[in] tai_offset TAI offset
 * @return Function to process C-Plane for a slot
 */
[[nodiscard]] std::function<void()> make_process_cplane_func(
        ran::fronthaul::Fronthaul &fronthaul,
        ran::message_adapter::IFapiSlotInfoProvider &slot_info_provider,
        std::atomic<ran::fapi::SlotInfo> &slot_info,
        std::chrono::nanoseconds t0,
        std::chrono::nanoseconds tai_offset);

/**
 * @brief Create U-Plane processing task function
 *
 * Processes U-Plane packets received via DOCA and orders them.
 * Uses IFapiSlotInfoProvider to check for UL data and get current slot timing.
 * Absolute slot is obtained from slot_info_provider (monotonic counter).
 *
 * @warning LIFETIME SAFETY: Captures fronthaul, slot_info_provider, and slot_info by reference.
 * Caller must ensure objects outlive returned function.
 *
 * @param[in] fronthaul Fronthaul instance reference
 * @param[in] slot_info_provider Slot info provider for slot state and accumulated FAPI messages
 * @param[in,out] slot_info Atomic slot info for current slot timing
 * @return Function to process U-Plane per slot
 */
[[nodiscard]] std::function<void()> make_process_uplane_func(
        ran::fronthaul::Fronthaul &fronthaul,
        ran::message_adapter::IFapiSlotInfoProvider &slot_info_provider,
        std::atomic<ran::fapi::SlotInfo> &slot_info);

/**
 * @brief Create PUSCH RX processing task function
 *
 * Executes PUSCH pipeline for the current slot via IPipelineExecutor interface.
 * Phase 1: Invokes pipeline execution after U-Plane processing.
 *
 * @warning LIFETIME SAFETY: Captures pipeline_executor and slot_info by reference.
 * Caller must ensure objects outlive returned function.
 *
 * @param[in] pipeline_executor Pipeline executor for launching PUSCH pipelines
 * @param[in,out] slot_info Atomic slot info for getting current slot
 * @return Function to process PUSCH RX per slot
 */
[[nodiscard]] std::function<void()> make_process_pusch_func(
        ran::message_adapter::IPipelineExecutor &pipeline_executor,
        std::atomic<ran::fapi::SlotInfo> &slot_info);

/**
 * @brief Create slot indication function for timed trigger
 *
 * Creates a function that sends SLOT.indication messages to all running cells
 * via the ISlotIndicationSender interface. Handles shutdown condition:
 * - All cells stopped during runtime
 *
 * @note main() guarantees cells are running before trigger starts, so this
 * function only needs to detect runtime shutdown (cells stopping after startup).
 *
 * @warning LIFETIME SAFETY: Captures slot_sender, fapi_rx_handler, and running
 * by reference. Caller must ensure objects outlive returned function.
 *
 * @param[in] slot_sender Interface for sending slot indications
 * @param[in] fapi_rx_handler Reference to FapiRxHandler (for cell count)
 * @param[in,out] running Atomic flag to signal shutdown
 * @return Function to be called periodically by TimedTrigger
 */
[[nodiscard]] std::function<void()> make_slot_indication_func(
        ran::message_adapter::ISlotIndicationSender &slot_sender,
        const ran::phy_ran_app::FapiRxHandler &fapi_rx_handler,
        std::atomic_bool &running);

} // namespace ran::phy_ran_app

#endif // RAN_PHY_RAN_APP_TASK_FACTORIES_HPP
