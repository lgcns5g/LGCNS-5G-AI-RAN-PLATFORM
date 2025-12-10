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

#ifndef RAN_FAPI_5G_CELL_HPP
#define RAN_FAPI_5G_CELL_HPP

#include <atomic>
#include <cstdint>

#include <wise_enum.h>

#include "ran_common.hpp"

namespace ran::fapi_5g {

/**
 * FAPI state machine states for each cell
 */
enum class FapiStateT : std::uint8_t {
    FapiStateIdle = 0,   //!< Cell not configured
    FapiStateConfigured, //!< Cell configured but not running
    FapiStateRunning,    //!< Cell running and processing slots
    FapiStateStopped     //!< Cell stopped after running
};

} // namespace ran::fapi_5g

// WISE_ENUM_ADAPT must be called at global namespace scope for ADL to work
WISE_ENUM_ADAPT(
        ran::fapi_5g::FapiStateT,
        FapiStateIdle,
        FapiStateConfigured,
        FapiStateRunning,
        FapiStateStopped)

namespace ran::fapi_5g {

/**
 * Cell
 *
 * Represents a single cell in the 5G network.
 * Manages cell-specific configuration and state.
 *
 * Thread-safety: The fapi_state member is atomic and can be safely
 * accessed from multiple threads. Other members should only be modified
 * during configuration before cells become active.
 */
struct Cell final {
    /**
     * Constructor
     *
     * @param[in] in_phy_cell_id Physical cell ID
     * @param[in] in_phy_params Physical layer parameters
     */
    Cell(const std::uint16_t in_phy_cell_id, const ran::common::PhyParams &in_phy_params)
            : phy_cell_id(in_phy_cell_id), ul_phy_params(in_phy_params),
              fapi_state(FapiStateT::FapiStateIdle) {}

    /**
     * Destructor
     */
    ~Cell() = default;

    // Non-copyable, non-movable (due to std::atomic member)
    Cell(const Cell &) = delete;
    Cell &operator=(const Cell &) = delete;
    Cell(Cell &&) = delete;
    Cell &operator=(Cell &&) = delete;

    std::uint16_t phy_cell_id{};          //!< Physical cell ID
    ran::common::PhyParams ul_phy_params; //!< Physical layer parameters
    std::atomic<FapiStateT> fapi_state;   //!< FAPI state (atomic for thread safety)
};

} // namespace ran::fapi_5g

#endif // RAN_FAPI_5G_CELL_HPP
