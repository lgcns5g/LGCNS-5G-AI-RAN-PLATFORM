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
 * @file uplane_network_config.hpp
 * @brief U-Plane network environment configuration utilities
 */

#ifndef RAN_FRONTHAUL_UPLANE_NETWORK_CONFIG_HPP
#define RAN_FRONTHAUL_UPLANE_NETWORK_CONFIG_HPP

#include "fronthaul/fronthaul_parser.hpp"
#include "fronthaul/uplane_config.hpp"
#include "net/env.hpp"

namespace ran::fronthaul {

/**
 * Populate network environment configuration with U-Plane DOCA RX queue settings
 *
 * Adds DOCA GPUNetIO RX queue configuration to an existing EnvConfig for receiving
 * O-RAN U-Plane packets with GPU-accelerated packet processing via the Order Kernel pipeline.
 *
 * The input EnvConfig must already contain:
 * - nic_pcie_addr (from C-plane configuration)
 * - gpu_device_id (from C-plane configuration)
 * - DPDK configuration (from C-plane configuration)
 *
 * This function adds:
 * - DOCA RX queue with O-RAN eCPRI EtherType filtering
 * - GPU semaphore setup for packet metadata communication
 * - MAC address filtering for RU identification
 *
 * @param[in,out] config Existing EnvConfig (with C-plane settings) to populate with U-plane DOCA RX
 * queue
 * @param[in] yaml_config Parsed YAML configuration containing RU MAC address
 * @param[in] uplane_config U-Plane configuration parameters (timing, DOCA settings)
 *
 * @throws std::invalid_argument if configuration parameters are invalid
 */
void populate_uplane_env_config(
        framework::net::EnvConfig &config,
        const FronthaulYamlConfig &yaml_config,
        const UPlaneConfig &uplane_config);

} // namespace ran::fronthaul

#endif // RAN_FRONTHAUL_UPLANE_NETWORK_CONFIG_HPP
