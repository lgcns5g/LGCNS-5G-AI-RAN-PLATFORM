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
 * @file fronthaul_app_utils.cpp
 * @brief Implementation of utility functions for fronthaul application
 */

#include <algorithm>
#include <exception>
#include <filesystem>
#include <format>
#include <optional>
#include <stdexcept>
#include <tuple>

#include <quill/LogMacros.h>
#include <tl/expected.hpp>

#include <CLI/CLI.hpp>

#include "fapi/fapi_file_replay.hpp"
#include "fronthaul/fronthaul.hpp"
#include "fronthaul/fronthaul_log.hpp"
#include "fronthaul/fronthaul_parser.hpp"
#include "fronthaul/uplane_config.hpp"
#include "fronthaul/uplane_network_config.hpp"
#include "fronthaul_app_utils.hpp"
#include "internal_use_only/config.hpp"
#include "log/components.hpp"
#include "log/rt_log.hpp"
#include "log/rt_log_macros.hpp"
#include "net/doca_rxq.hpp"
#include "net/doca_txq.hpp"
#include "net/dpdk_txq.hpp"
#include "net/env.hpp"
#include "net/gpu.hpp"
#include "net/mempool.hpp"
#include "net/net_log.hpp"
#include "net/nic.hpp"
#include "oran/numerology.hpp"
#include "oran/oran_log.hpp"
#include "task/task_log.hpp"
#include "task/task_utils.hpp"

namespace ran::fronthaul::samples {

namespace rf = ran::fronthaul;

void setup_logging() {
    framework::task::enable_sanitizer_compatibility();
    framework::log::Logger::set_level(framework::log::LogLevel::Debug);
    framework::log::register_component<framework::net::Net>(framework::log::LogLevel::Debug);
    framework::log::register_component<framework::task::TaskLog>(framework::log::LogLevel::Debug);
    framework::log::register_component<ran::oran::Oran>(framework::log::LogLevel::Debug);
    framework::log::register_component<rf::FronthaulLog>(framework::log::LogLevel::Debug);
    framework::log::register_component<rf::FronthaulKernels>(framework::log::LogLevel::Debug);
    framework::log::register_component<rf::FronthaulApp>(framework::log::LogLevel::Debug);
}

std::string get_fapi_file_path(const std::string &cli_path, const char *argv0) {
    // Use CLI argument if provided
    if (!cli_path.empty()) {
        if (std::filesystem::exists(cli_path)) {
            return cli_path;
        }
        throw std::runtime_error(
                std::format("FAPI capture file specified does not exist: {}", cli_path));
    }

    // Fall back to searching in executable directory
    const std::filesystem::path exe_path{argv0};
    const std::filesystem::path exe_dir = exe_path.empty() ? "." : exe_path.parent_path();

    // Look for any .fapi file in the directory
    for (const auto &entry : std::filesystem::directory_iterator(exe_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".fapi") {
            return entry.path().string();
        }
    }

    throw std::runtime_error(std::format(
            "No FAPI capture file found in {}. Use --fapi-file to specify path", exe_dir.string()));
}

tl::expected<AppArguments, std::string> parse_arguments(const int argc, const char **argv) {

    AppArguments args{};

    CLI::App app{std::format(
            "Fronthaul Application - {} version {}",
            framework::cmake::project_name,
            framework::cmake::project_version)};

    app.add_option("-n,--nic", args.nic_pcie_addr, "DU NIC PCIe address")->required();
    app.add_option("-c,--config", args.config_file_path, "Path to ru_emulator_config.yaml")
            ->required();
    app.add_option(
            "-f,--fapi-file",
            args.fapi_file_path,
            "FAPI capture file (auto-detects .fapi in exe dir if not specified)");
    app.add_option("-s,--slots", args.num_slots, "Number of slots to run (omit for unlimited)");
    app.add_option(
            "-w,--worker-core",
            args.worker_core,
            std::format("Worker CPU core (default: {})", DEFAULT_WORKER_CORE));
    app.add_option(
            "-t,--trigger-core",
            args.trigger_core,
            std::format("Trigger CPU core (default: {})", DEFAULT_TRIGGER_CORE));
    app.add_option(
            "--slot-ahead",
            args.slot_ahead,
            std::format("Slots to process ahead (default: {})", DEFAULT_SLOT_AHEAD));
    app.add_option(
            "-g,--gpu-device",
            args.gpu_device_id,
            "GPU device ID for U-Plane processing (default: 0)");

    app.add_flag(
            "--validate",
            args.validate_uplane_prbs,
            "Validate U-Plane PRB counts against expected");

    app.set_version_flag(
            "--version",
            std::string{framework::cmake::project_version},
            "Show version information");

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
        const int exit_code = app.exit(e); // Print help or error message
        if (exit_code == 0) {
            // Success codes (--help or --version) - return empty error string
            return tl::unexpected("");
        }
        // Actual error
        const std::string error_msg = std::format("Argument parsing failed: {}", e.what());
        return tl::unexpected(error_msg);
    }

    // Validate that the specified NIC exists in the list of discovered NICs
    const auto available_nics = framework::net::discover_mellanox_nics();
    if (available_nics.empty()) {
        return tl::unexpected("No Mellanox NICs available. At least one Mellanox NIC is required.");
    }

    if (std::find(available_nics.cbegin(), available_nics.cend(), args.nic_pcie_addr) ==
        available_nics.cend()) {
        std::string nic_list;
        for (const auto &nic : available_nics) {
            if (!nic_list.empty()) {
                nic_list += ", ";
            }
            nic_list += nic;
        }
        return tl::unexpected(std::format(
                "Invalid NIC PCIe address: {} not in device list: [{}].",
                args.nic_pcie_addr,
                nic_list));
    }

    return args;
}

std::vector<framework::net::MacAddress>
convert_mac_addresses(const std::vector<std::string> &mac_strings) {
    std::vector<framework::net::MacAddress> cell_macs{};
    cell_macs.reserve(mac_strings.size());

    for (const auto &mac_str : mac_strings) {
        const auto mac_result = framework::net::MacAddress::from_string(mac_str);
        if (!mac_result.has_value()) {
            throw std::runtime_error(mac_result.error());
        }
        cell_macs.push_back(mac_result.value());
    }

    return cell_macs;
}

/**
 * Create network configuration for DPDK
 *
 * @param[in] nic_pcie_addr NIC PCIe address
 * @param[in] gpu_device_id GPU device ID for U-Plane DOCA support
 * @param[in] dpdk_core DPDK CPU core
 * @param[in] mtu_size MTU size for network interface
 * @return Network environment configuration
 */
static framework::net::EnvConfig create_network_config(
        const std::string &nic_pcie_addr,
        const std::uint32_t gpu_device_id, // NOLINT(bugprone-easily-swappable-parameters)
        const std::uint32_t dpdk_core,
        const std::uint32_t mtu_size) {
    /// DPDK TX queue size - need room for 168 packets per slot
    static constexpr std::uint16_t DPDK_TXQ_SIZE = 256;
    /// Mempool number of mbufs
    static constexpr std::uint32_t MEMPOOL_NUM_MBUFS = 8192;

    framework::net::EnvConfig net_config{};

    // Set GPU device ID (needed for U-plane DOCA support)
    net_config.gpu_device_id = framework::net::GpuDeviceId{gpu_device_id};

    net_config.dpdk_config.app_name = "fronthaul_app";
    net_config.dpdk_config.file_prefix = "fronthaul_app_prefix";
    net_config.dpdk_config.dpdk_core_id = dpdk_core;
    net_config.nic_config.nic_pcie_addr = nic_pcie_addr;
    net_config.nic_config.enable_accurate_send_scheduling = true;

    // DPDK TX queue only
    framework::net::DpdkTxQConfig dpdk_tx_config{};
    dpdk_tx_config.txq_size = DPDK_TXQ_SIZE;
    net_config.nic_config.dpdk_txq_configs.push_back(dpdk_tx_config);

    // Single mempool - CPU mode, no host pinning
    framework::net::MempoolConfig mempool_config{};
    mempool_config.name = "fronthaul_mempool";
    mempool_config.num_mbufs = MEMPOOL_NUM_MBUFS;
    mempool_config.mtu_size = mtu_size;
    mempool_config.host_pinned = framework::net::HostPinned::No;
    net_config.nic_config.mempool_configs.push_back(mempool_config);

    return net_config;
}

TimingResult calculate_timing_parameters(
        const framework::task::StartTimeParams &start_params,
        const std::uint32_t slot_ahead,
        const std::uint64_t slot_period_ns) {

    // Calculate TAI offset
    const auto tai_offset = framework::task::calculate_tai_offset();

    const auto next_boundary_ns =
            framework::task::calculate_start_time_for_next_period(start_params, tai_offset);
    const auto start_time_ns = next_boundary_ns - (slot_ahead * slot_period_ns);

    // t0 is the time for SFN 0, subframe 0, slot 0
    const auto t0 = std::chrono::nanoseconds{next_boundary_ns};

    // Log timing information
    const auto delta_ns = static_cast<std::int64_t>(start_time_ns) -
                          static_cast<std::int64_t>(start_params.current_time_ns);
    RT_LOGC_INFO(
            rf::FronthaulApp::App,
            "Current time: {}ns, Next system frame boundary: {}ns, Start "
            "time: {}ns (+{} ns) is {} slots ahead of the boundary",
            start_params.current_time_ns,
            next_boundary_ns,
            start_time_ns,
            delta_ns,
            slot_ahead);

    return TimingResult{
            .start_time_ns = start_time_ns,
            .t0 = t0,
            .tai_offset = tai_offset,
    };
}

std::function<void()> make_process_cplane_func(
        ran::fronthaul::Fronthaul &fronthaul,
        ran::fapi::FapiFileReplay &fapi_replay,
        bool &is_first_slot,
        const std::chrono::nanoseconds t0,
        const std::chrono::nanoseconds tai_offset) {
    // SAFETY: Lambda captures fronthaul, fapi_replay, and is_first_slot by reference.
    // Caller must ensure these objects outlive the returned function.
    return [&fronthaul, &fapi_replay, &is_first_slot, t0, tai_offset]() {
        // Advance to next slot (skip on first call to avoid skipping slot 0)
        if (!is_first_slot) {
            std::ignore = fapi_replay.advance_slot();
        } else {
            is_first_slot = false;
        }

        // Get current slot
        const std::uint64_t absolute_slot = fapi_replay.get_current_absolute_slot();

        // Process each cell for current slot
        for (const auto cell_id : fapi_replay.get_cell_ids()) {
            // Get request for current slot
            // Returns std::nullopt if no match (e.g., DL-only slot)
            const auto request_opt = fapi_replay.get_request_for_current_slot(cell_id);

            if (!request_opt) {
                // No UL data for this cell/slot
                continue;
            }

            // example-begin send-cplane-1
            // Send C-plane for this cell
            // Request already has updated sfn/slot from FapiFileReplay
            const auto &req_info = request_opt.value();
            fronthaul.send_ul_cplane(
                    *req_info.request, req_info.body_len, cell_id, absolute_slot, t0, tai_offset);
            // example-end send-cplane-1
        }
    };
}

std::function<void()> make_process_uplane_func(
        ran::fronthaul::Fronthaul &fronthaul, ran::fapi::FapiFileReplay &fapi_replay) {

    // SAFETY: Lambda captures fronthaul and fapi_replay by reference for performance.
    // Caller must ensure these objects outlive the returned function.
    return [&fronthaul, &fapi_replay]() {
        // Check if current slot has UL data for any cell
        // U-plane processing should only occur for slots that had C-plane sent
        const auto cell_ids = fapi_replay.get_cell_ids();
        const bool has_ul_data =
                std::any_of(cell_ids.begin(), cell_ids.end(), [&fapi_replay](const auto cell_id) {
                    return fapi_replay.get_request_for_current_slot(cell_id);
                });

        if (!has_ul_data) {
            // No UL data for this slot - skip U-plane processing
            // This prevents launching kernels for slots without FAPI data
            return;
        }

        // example-begin process-uplane-1
        // Get current slot timing and convert to ORAN format
        const auto oran_slot_timing =
                ran::oran::fapi_to_oran_slot_timing(fapi_replay.get_current_slot_timing());

        try {
            fronthaul.process_uplane(oran_slot_timing);
        } catch (const std::exception &e) {
            RT_LOGC_ERROR(rf::FronthaulApp::UPlane, "Failed to process U-Plane: {}", e.what());
        }
        // example-end process-uplane-1
    };
}

void print_statistics(
        const ran::fronthaul::Fronthaul &fronthaul,
        framework::task::TaskScheduler &scheduler,
        framework::task::TimedTrigger &trigger,
        const std::optional<std::size_t> num_slots) {

    // example-begin get-stats-1
    RT_LOGC_INFO(rf::FronthaulApp::Stats, "\n=== Fronthaul Statistics ===");
    RT_LOGC_INFO(rf::FronthaulApp::Stats, "{}", fronthaul.get_stats());
    // example-end get-stats-1

    // Only print detailed stats for short runs (< 2000 slots)
    static constexpr std::size_t MAX_SLOTS_FOR_DETAILED_STATS = 2000;
    if (num_slots.has_value() && num_slots.value() < MAX_SLOTS_FOR_DETAILED_STATS) {
        RT_LOGC_INFO(rf::FronthaulApp::Stats, "\n=== Task Statistics ===");
        scheduler.print_monitor_stats();

        RT_LOGC_INFO(rf::FronthaulApp::Stats, "\n=== Trigger Statistics ===");
        trigger.print_summary();
    }
}

bool validate_kernel_results(const ran::fronthaul::Fronthaul &fronthaul) {
    try {
        const auto stats = fronthaul.read_kernel_statistics();

        RT_LOGC_INFO(rf::FronthaulApp::Stats, "\n=== U-Plane Kernel Statistics ===\n{}", stats);

        // Validate total PRB count matches expected
        const bool prbs_match = (stats.total_pusch_prbs == stats.total_expected_prbs);

        if (!prbs_match) {
            RT_LOGC_ERROR(
                    rf::FronthaulApp::Stats,
                    "Validation FAILED: PRB mismatch - expected {}, got {}",
                    stats.total_expected_prbs,
                    stats.total_pusch_prbs);
            return false;
        }

        RT_LOGC_INFO(rf::FronthaulApp::Stats, "Validation PASSED");
        return true;

    } catch (const std::exception &e) {
        RT_LOGC_ERROR(rf::FronthaulApp::Stats, "Validation failed: {}", e.what());
        return false;
    }
}

std::optional<ran::fronthaul::FronthaulConfig> create_fronthaul_config_from_yaml(
        const std::string &yaml_config_path,
        const std::string &nic_pcie_addr,
        const std::uint32_t gpu_device_id,
        const std::uint32_t dpdk_core,
        const std::uint32_t slot_ahead) {

    // 1. Parse YAML configuration
    const auto yaml_config = ran::fronthaul::parse_fronthaul_config(yaml_config_path);
    if (!yaml_config.has_value()) {
        RT_LOGC_ERROR(
                rf::FronthaulApp::Config,
                "Failed to parse YAML config file: {}",
                yaml_config.error());
        return std::nullopt;
    }

    RT_LOGC_DEBUG(
            rf::FronthaulApp::Config, "Loaded config from {}: {}", yaml_config_path, *yaml_config);

    // 2. Convert cell MAC addresses from YAML config
    std::vector<framework::net::MacAddress> cell_macs;
    cell_macs.reserve(yaml_config->cells.size());
    for (const auto &cell : yaml_config->cells) {
        const auto mac_result = framework::net::MacAddress::from_string(cell.mac_address);
        if (!mac_result.has_value()) {
            RT_LOGC_ERROR(
                    rf::FronthaulApp::Config,
                    "Invalid MAC address in config: {}",
                    mac_result.error());
            return std::nullopt;
        }
        cell_macs.push_back(mac_result.value());
    }

    // 3. Extract VLAN TCIs from YAML config
    std::vector<std::uint16_t> cell_vlans;
    cell_vlans.reserve(yaml_config->cells.size());
    for (const auto &cell : yaml_config->cells) {
        cell_vlans.push_back(cell.vlan_tci);
    }

    // Validate at least one cell is configured (required for U-plane RU MAC address)
    if (yaml_config->cells.empty()) {
        RT_LOGC_ERROR(
                rf::FronthaulApp::Config,
                "No cells configured in YAML - fronthaul requires at least one cell");
        return std::nullopt;
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
                rf::FronthaulApp::Config,
                "Using eAxC IDs from YAML config: {}",
                uplane_config.eaxc_ids);
    }
    rf::populate_uplane_env_config(net_config, *yaml_config, uplane_config);

    // example-begin create-config-1
    // 6. Create and return fronthaul config with complete net_config
    static constexpr std::uint32_t SCS_KHZ = 30;
    return ran::fronthaul::FronthaulConfig{
            .net_config = net_config,
            .cell_dest_macs = cell_macs,
            .cell_vlan_tcis = cell_vlans,
            .numerology = ran::oran::from_scs_khz(SCS_KHZ),
            .slot_ahead = slot_ahead,
            .t1a_max_cp_ul_ns = yaml_config->timing.t1a_max_ns,
            .t1a_min_cp_ul_ns = yaml_config->timing.t1a_min_ns,
            .uplane_config = uplane_config,
    };
    // example-end create-config-1
}

} // namespace ran::fronthaul::samples
