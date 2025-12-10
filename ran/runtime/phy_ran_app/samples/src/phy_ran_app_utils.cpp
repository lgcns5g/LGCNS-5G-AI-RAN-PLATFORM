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
 * @file phy_ran_app_utils.cpp
 * @brief Implementation of utility functions for PHY RAN App
 */

#include <algorithm>
#include <filesystem>
#include <format>
#include <functional>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <quill/LogMacros.h>

#include <CLI/CLI.hpp>

#include "fapi/fapi_log.hpp"
#include "fapi/fapi_utils.hpp"
#include "fronthaul/fronthaul.hpp"
#include "fronthaul/fronthaul_log.hpp"
#include "fronthaul/fronthaul_parser.hpp"
#include "fronthaul/uplane_config.hpp"
#include "fronthaul/uplane_network_config.hpp"
#include "internal_use_only/config.hpp"
#include "log/components.hpp"
#include "log/rt_log.hpp"
#include "net/doca_rxq.hpp"
#include "net/doca_txq.hpp"
#include "net/dpdk_txq.hpp"
#include "net/dpdk_types.hpp"
#include "net/env.hpp"
#include "net/gpu.hpp"
#include "net/mempool.hpp"
#include "net/nic.hpp"
#include "oran/numerology.hpp"
#include "phy_ran_app/phy_ran_app_log.hpp"
#include "phy_ran_app_utils.hpp"
#include "task/task_utils.hpp"

namespace ran::phy_ran_app {

void setup_logging() {

    framework::task::enable_sanitizer_compatibility();

    // Set global log level
    framework::log::Logger::set_level(framework::log::LogLevel::Debug);

    // Register PHY RAN App components
    framework::log::register_component<PhyRanApp>(framework::log::LogLevel::Debug);

    // Register FAPI components
    framework::log::register_component<ran::fapi::FapiComponent>(framework::log::LogLevel::Info);

    // Register Fronthaul components (enable Debug for detailed Order Kernel tracing)
    framework::log::register_component<ran::fronthaul::FronthaulLog>(
            framework::log::LogLevel::Debug);
    framework::log::register_component<ran::fronthaul::FronthaulKernels>(
            framework::log::LogLevel::Debug);
    framework::log::register_component<ran::fronthaul::FronthaulApp>(
            framework::log::LogLevel::Debug);
}

tl::expected<AppArguments, std::string> parse_arguments(int argc, char **argv) {
    CLI::App app{"PHY RAN App - Integrates FAPI (MAC-PHY) and Fronthaul (RU) interfaces"};

    AppArguments args{};

    // Required arguments (based on fronthaul_app pattern)
    app.add_option("-n,--nic", args.nic_pcie_addr, "DU NIC PCIe address")->required();
    app.add_option("-c,--config", args.config_file_path, "Path to YAML configuration file")
            ->required()
            ->check(CLI::ExistingFile);

    // Optional arguments (based on fronthaul_app pattern + FAPI extensions)
    app.add_option("-s,--slots", args.num_slots, "Number of slots to run (omit for unlimited)");
    app.add_option(
            "--rx-core",
            args.rx_core,
            std::format("FAPI RX thread core (default: {})", DEFAULT_RX_CORE));
    app.add_option(
            "-t,--slot-indication-core",
            args.slot_indication_core,
            std::format(
                    "Slot indication trigger core (default: {})", DEFAULT_SLOT_INDICATION_CORE));
    app.add_option(
            "--cplane-core",
            args.cplane_core,
            std::format("C-Plane processing core (default: {})", DEFAULT_CPLANE_CORE));
    app.add_option(
            "--uplane-core",
            args.uplane_core,
            std::format("U-Plane processing core (default: {})", DEFAULT_UPLANE_CORE));
    app.add_option(
            "--pusch-core",
            args.pusch_core,
            std::format("PUSCH RX processing core (default: {})", DEFAULT_PUSCH_CORE));
    app.add_option(
            "--slot-ahead",
            args.slot_ahead,
            std::format("Slots to process ahead (default: {})", DEFAULT_SLOT_AHEAD));
    app.add_option(
            "-g,--gpu-device",
            args.gpu_device_id,
            "GPU device ID for U-Plane processing (default: 0)");
    app.add_option(
            "--expected-cells",
            args.expected_cells,
            std::format("Expected number of cells (default: {})", DEFAULT_EXPECTED_CELLS));
    app.add_flag(
            "--validate", args.validate, "Validate runtime statistics and fail test on errors");

    app.set_version_flag(
            "--version",
            std::string{framework::cmake::project_version},
            "Show version information");

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
        const int exit_code = app.exit(e); // Print help or error message
        if (exit_code == 0) {
            // Success codes (e.g., --help, --version) - return empty error string
            return tl::unexpected("");
        }
        // Actual error
        const std::string error_msg = std::format("Argument parsing failed: {}", e.what());
        return tl::unexpected(error_msg);
    }

    // Validate cell count against TensorRT engine maximum
    if (args.expected_cells > DEFAULT_EXPECTED_CELLS) {
        const std::string error_msg = std::format(
                "Cell count exceeds maximum supported by TensorRT engine. "
                "Got: {}, Maximum: {}",
                args.expected_cells,
                DEFAULT_EXPECTED_CELLS);
        return tl::unexpected(error_msg);
    }

    return args;
}

std::string create_nvipc_config() { return ran::fapi::create_default_nvipc_config("phy_ran_app"); }

/**
 * @brief Create network environment configuration for DPDK and DOCA
 *
 * Helper function adapted from fronthaul_app_utils.cpp::create_network_config().
 *
 * @param[in] nic_pcie_addr NIC PCIe address
 * @param[in] gpu_device_id GPU device ID
 * @param[in] dpdk_core DPDK lcore
 * @param[in] mtu_size MTU size for mempool
 * @return Network environment configuration
 */
static framework::net::EnvConfig create_network_config(
        const std::string &nic_pcie_addr,
        std::uint32_t gpu_device_id, // NOLINT(bugprone-easily-swappable-parameters)
        std::uint32_t dpdk_core,
        std::uint32_t mtu_size) {

    /// DPDK TX queue size - need room for 168 packets per slot
    static constexpr std::uint16_t DPDK_TXQ_SIZE = 256;
    /// Mempool number of mbufs
    static constexpr std::uint32_t MEMPOOL_NUM_MBUFS = 8192;

    framework::net::EnvConfig net_config{};

    // Set GPU device ID (needed for U-plane DOCA support)
    net_config.gpu_device_id = framework::net::GpuDeviceId{gpu_device_id};

    net_config.dpdk_config.app_name = "phy_ran_app";
    net_config.dpdk_config.file_prefix = "phy_ran_app_prefix";
    net_config.dpdk_config.dpdk_core_id = dpdk_core;
    net_config.nic_config.nic_pcie_addr = nic_pcie_addr;
    net_config.nic_config.enable_accurate_send_scheduling = true;

    // DPDK TX queue only
    framework::net::DpdkTxQConfig dpdk_tx_config{};
    dpdk_tx_config.txq_size = DPDK_TXQ_SIZE;
    net_config.nic_config.dpdk_txq_configs.push_back(dpdk_tx_config);

    // Single mempool - CPU mode, no host pinning
    framework::net::MempoolConfig mempool_config{};
    mempool_config.name = "phy_ran_app_mempool";
    mempool_config.num_mbufs = MEMPOOL_NUM_MBUFS;
    mempool_config.mtu_size = mtu_size;
    mempool_config.host_pinned = framework::net::HostPinned::No;
    net_config.nic_config.mempool_configs.push_back(mempool_config);

    return net_config;
}

tl::expected<ran::fronthaul::FronthaulConfig, std::string> create_fronthaul_config(
        const std::string &yaml_config_path,
        const std::string &nic_pcie_addr,
        std::uint32_t gpu_device_id, // NOLINT(bugprone-easily-swappable-parameters)
        std::uint32_t dpdk_core,
        std::uint32_t slot_ahead) {
    namespace rf = ran::fronthaul;

    // 1. Parse YAML configuration
    const auto yaml_config = rf::parse_fronthaul_config(yaml_config_path);
    if (!yaml_config.has_value()) {
        const auto error_msg =
                std::format("Failed to parse YAML config file: {}", yaml_config.error());
        RT_LOGC_ERROR(PhyRanApp::Config, "{}", error_msg);
        return tl::unexpected(error_msg);
    }

    RT_LOGC_DEBUG(PhyRanApp::Config, "Loaded config from {}: {}", yaml_config_path, *yaml_config);

    // 2. Convert cell MAC addresses from YAML config
    std::vector<framework::net::MacAddress> cell_macs;
    cell_macs.reserve(yaml_config->cells.size());
    for (const auto &cell : yaml_config->cells) {
        const auto mac_result = framework::net::MacAddress::from_string(cell.mac_address);
        if (!mac_result.has_value()) {
            const auto error_msg =
                    std::format("Invalid MAC address in config: {}", mac_result.error());
            RT_LOGC_ERROR(PhyRanApp::Config, "{}", error_msg);
            return tl::unexpected(error_msg);
        }
        cell_macs.push_back(mac_result.value());
    }

    // 3. Extract VLAN TCIs from YAML config
    std::vector<std::uint16_t> cell_vlans;
    cell_vlans.reserve(yaml_config->cells.size());
    std::ranges::transform(
            yaml_config->cells, std::back_inserter(cell_vlans), [](const auto &cell) {
                return cell.vlan_tci;
            });

    // Validate at least one cell is configured (required for U-plane RU MAC address)
    if (yaml_config->cells.empty()) {
        const auto error_msg =
                std::string{"No cells configured in YAML - fronthaul requires at least one cell"};
        RT_LOGC_ERROR(PhyRanApp::Config, "{}", error_msg);
        return tl::unexpected(error_msg);
    }

    // 4. Create base network config (C-plane) with MTU from YAML
    framework::net::EnvConfig net_config =
            create_network_config(nic_pcie_addr, gpu_device_id, dpdk_core, yaml_config->mtu_size);

    // 5. Add U-plane DOCA RX queue configuration to the same net_config
    rf::UPlaneConfig uplane_config{}; // Use default U-plane parameters
    // Extract eAxC IDs from first cell config
    if (!yaml_config->cells.empty() && !yaml_config->cells[0].eaxc_ul.empty()) {
        uplane_config.eaxc_ids = yaml_config->cells[0].eaxc_ul;
        RT_LOGC_DEBUG(
                PhyRanApp::Config, "Using eAxC IDs from YAML config: {}", uplane_config.eaxc_ids);
    }
    rf::populate_uplane_env_config(net_config, *yaml_config, uplane_config);

    // 6. Create and return fronthaul config with complete net_config
    static constexpr std::uint32_t SCS_KHZ = 30;
    return rf::FronthaulConfig{
            .net_config = net_config,
            .cell_dest_macs = cell_macs,
            .cell_vlan_tcis = cell_vlans,
            .numerology = ran::oran::from_scs_khz(SCS_KHZ),
            .slot_ahead = slot_ahead,
            .t1a_max_cp_ul_ns = yaml_config->timing.t1a_max_ns,
            .t1a_min_cp_ul_ns = yaml_config->timing.t1a_min_ns,
            .uplane_config = uplane_config,
    };
}

TimingResult calculate_timing_parameters(
        const framework::task::StartTimeParams &start_params,
        const std::uint32_t slot_ahead,
        const std::uint64_t slot_period_ns) {

    // Calculate TAI offset
    const auto tai_offset = framework::task::calculate_tai_offset();

    const auto next_boundary_ns =
            framework::task::calculate_start_time_for_next_period(start_params, tai_offset);

    // Validate slot_ahead won't cause underflow
    const auto slot_offset_ns = slot_ahead * slot_period_ns;
    if (slot_offset_ns > next_boundary_ns) {
        throw std::runtime_error(std::format(
                "Invalid slot_ahead={}: offset {}ns exceeds boundary {}ns",
                slot_ahead,
                slot_offset_ns,
                next_boundary_ns));
    }
    const auto start_time_ns = next_boundary_ns - slot_offset_ns;

    // t0 is the time for SFN 0, subframe 0, slot 0
    const auto t0 = std::chrono::nanoseconds{next_boundary_ns};

    // Log timing information
    const auto delta_ns = static_cast<std::int64_t>(start_time_ns) -
                          static_cast<std::int64_t>(start_params.current_time_ns);
    const double delta_s = static_cast<double>(delta_ns) / 1e9;
    RT_LOGC_DEBUG(
            PhyRanApp::App,
            "Timing calculation: start_time in {:0.3f}s, {} slots ahead of SFN boundary "
            "(current={}ns, boundary={}ns, start={}ns)",
            delta_s,
            slot_ahead,
            start_params.current_time_ns,
            next_boundary_ns,
            start_time_ns);

    return TimingResult{
            .start_time_ns = start_time_ns,
            .t0 = t0,
            .tai_offset = tai_offset,
    };
}

bool print_and_validate_stats(
        const ran::message_adapter::PhyStats &stats, const PrintAndValidate fail_on_errors) {
    const auto &cell_stats = stats.get_stats();
    const std::uint64_t total_failures =
            std::accumulate(cell_stats.begin(), cell_stats.end(), std::uint64_t{0});

    RT_LOGC_INFO(
            PhyRanApp::App,
            "\n=== PHY Runtime Statistics ===\nTotal CRC failures: {}",
            total_failures);

    // Print per-cell breakdown
    for (std::uint32_t cell_id = 0; cell_id < cell_stats.size(); ++cell_id) {
        const auto failures = cell_stats[cell_id];
        if (failures > 0) {
            RT_LOGC_WARN(PhyRanApp::App, "Cell {}: {} CRC failures", cell_id, failures);
        } else {
            RT_LOGC_INFO(PhyRanApp::App, "Cell {}: {} CRC failures", cell_id, failures);
        }
    }

    // Validate if requested
    if (fail_on_errors.get() && total_failures > 0) {
        RT_LOGC_ERROR(PhyRanApp::App, "Validation FAILED: CRC errors detected");
        return false;
    }

    if (fail_on_errors.get()) {
        RT_LOGC_INFO(PhyRanApp::App, "Validation PASSED: No CRC errors");
    }

    return true;
}

} // namespace ran::phy_ran_app
