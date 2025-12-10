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

#include <format>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <quill/LogMacros.h>
#include <tl/expected.hpp>

#include "fronthaul/fronthaul_log.hpp"
#include "fronthaul/fronthaul_parser.hpp"
#include "fronthaul/order_kernel_descriptors.hpp" // for DEFAULT_DPDK_CORE_ID, ORAN_ORU_ETHER_TYPE
#include "fronthaul/uplane_config.hpp"
#include "fronthaul/uplane_network_config.hpp"
#include "log/rt_log_macros.hpp"
#include "net/doca_rxq.hpp"
#include "net/doca_types.hpp"
#include "net/dpdk_types.hpp"
#include "net/env.hpp"
#include "net/nic.hpp"

namespace ran::fronthaul {

void populate_uplane_env_config(
        framework::net::EnvConfig &config,
        const FronthaulYamlConfig &yaml_config,
        const UPlaneConfig &uplane_config) {

    // Config already has from C-plane (create_network_config):
    // - config.nic_config.nic_pcie_addr
    // - config.gpu_device_id
    // - config.dpdk_config (app_name, file_prefix, dpdk_core_id)
    // - DPDK TX queues
    //
    // We ONLY add DOCA RX queue configuration:

    // Validate configuration
    if (yaml_config.cells.empty()) {
        const std::string error_message =
                "No cells configured - at least one cell required for RU MAC address";
        RT_LOGC_ERROR(FronthaulLog::FronthaulNetwork, "{}", error_message);
        throw std::invalid_argument(error_message);
    }

    // DOCA RX queue configuration for O-RAN U-Plane packet reception
    framework::net::DocaRxQConfig rx_config{};

    // Use existing NIC address (already set by C-plane)
    rx_config.nic_pcie_addr = config.nic_config.nic_pcie_addr;

    const auto &cell = yaml_config.cells[0]; // currently only one cell is supported for U-plane

    // RU MAC address from YAML
    const auto ru_mac_result = framework::net::MacAddress::from_string(cell.mac_address);
    if (!ru_mac_result.has_value()) {
        RT_LOGC_ERROR(
                FronthaulLog::FronthaulNetwork,
                "Invalid RU MAC address: {}",
                ru_mac_result.error());
        throw std::invalid_argument(std::format(
                "populate_uplane_env_config: Invalid RU MAC address: {}", ru_mac_result.error()));
    }
    rx_config.sender_mac_addr = ru_mac_result.value();

    // DOCA packet configuration
    rx_config.max_pkt_num = uplane_config.num_packets;
    rx_config.max_pkt_size = uplane_config.max_packet_size;
    rx_config.ether_type = ORAN_ORU_ETHER_TYPE;

    // VLAN TCI from YAML
    rx_config.vlan_tci = cell.vlan_tci;

    // Configure GPU semaphores for order kernel packet count communication
    rx_config.sem_items = framework::net::DocaSemItems{
            .num_items = uplane_config.gpu_semaphore_items,
            .item_size = sizeof(DocaOrderSemInfo),
    };

    // Add DOCA RX queue to existing config
    config.nic_config.doca_rxq_configs.push_back(rx_config);
}

} // namespace ran::fronthaul
