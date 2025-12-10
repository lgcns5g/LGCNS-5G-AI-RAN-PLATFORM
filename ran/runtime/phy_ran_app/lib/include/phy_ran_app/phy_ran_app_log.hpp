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
 * @file phy_ran_app_log.hpp
 * @brief Logging categories for PHY RAN App
 *
 * Defines logging component categories for structured logging throughout the application.
 */

#ifndef RAN_PHY_RAN_APP_LOG_HPP
#define RAN_PHY_RAN_APP_LOG_HPP

#include "log/rt_log_macros.hpp"

namespace ran::phy_ran_app {

/**
 * PHY RAN App logging component categories
 *
 * Usage:
 * @code
 * RT_LOGC_INFO(ran::phy_ran_app::PhyRanApp::App, "Application started");
 * RT_LOGC_DEBUG(ran::phy_ran_app::PhyRanApp::FapiRx, "Received message");
 * @endcode
 */
DECLARE_LOG_COMPONENT(
        PhyRanApp,
        App,            ///< Main application flow, initialization, lifecycle
        Config,         ///< Configuration parsing (YAML, CLI)
        FapiRx,         ///< FAPI message reception and accumulation
        CPlane,         ///< C-Plane processing (ORAN)
        UPlane,         ///< U-Plane processing (Order Kernel)
        MessageAdapter, ///< Message adapter (FAPI to PUSCH)
        PuschRx,        ///< PUSCH RX pipeline
        SlotIndication, ///< Slot indication trigger (500µs)
        Stats);         ///< Statistics and reporting

} // namespace ran::phy_ran_app

#endif // RAN_PHY_RAN_APP_LOG_HPP
