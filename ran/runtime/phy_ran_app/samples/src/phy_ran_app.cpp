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
 * @file phy_ran_app.cpp
 * @brief PHY RAN App - Integrates FAPI (MAC-PHY) and Fronthaul (RU) interfaces
 *
 * Implements FAPI message processing with Message Adapter, uplink processing graph,
 * and slot indication trigger.
 *
 * See ran/runtime/phy-ran-app/README.md for usage and design documents for architecture.
 *
 * Coordinates 3 processes:
 *   test_mac ←→ [NVIPC/FAPI] ←→ phy_ran_app ←→ [DPDK/Fronthaul] ←→ ru_emulator
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <format>
#include <functional>
#include <optional>
#include <ratio>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include <quill/LogMacros.h>
#include <tl/expected.hpp>

#include <gsl-lite/gsl-lite.hpp>

#include "fapi/fapi_state.hpp" // for SlotInfo
#include "fronthaul/fronthaul.hpp"
#include "log/rt_log_macros.hpp"
#include "oran/numerology.hpp"
#include "phy_ran_app/fapi_rx_handler.hpp"
#include "phy_ran_app/phy_ran_app_log.hpp"
#include "phy_ran_app/task_factories.hpp"
#include "phy_ran_app_utils.hpp"
#include "task/task_category.hpp"
#include "task/task_graph.hpp"
#include "task/task_scheduler.hpp"
#include "task/task_utils.hpp"
#include "task/task_worker.hpp"
#include "task/time.hpp"
#include "task/timed_trigger.hpp"

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

namespace {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::atomic_bool g_stop_requested{false};

void signal_handler(const int signum) {
    if (signum == SIGINT || signum == SIGTERM) {
        g_stop_requested = true;
    }
}

} // namespace

/**
 * PHY RAN application entry point
 *
 * @param[in] argc Argument count
 * @param[in] argv Argument values
 * @return Exit status code
 */
int main(int argc, char **argv) {
    namespace rpa = ran::phy_ran_app;
    using namespace std::chrono_literals;

    try {
        rpa::setup_logging();

        const auto args_result = rpa::parse_arguments(argc, argv);
        if (!args_result) {
            if (!args_result.error().empty()) {
                RT_LOGC_ERROR(rpa::PhyRanApp::App, "{}", args_result.error());
                return EXIT_FAILURE;
            }
            return EXIT_SUCCESS;
        }

        const auto &args = *args_result;

        RT_LOGC_INFO(rpa::PhyRanApp::App, "PHY RAN App starting");
        RT_LOGC_INFO(rpa::PhyRanApp::App, "Configuration: {}", args);

        std::atomic_bool running{true};

        // ==================== FAPI INTEGRATION ====================

        // Initialize FAPI RX Handler with Message Adapter (Sample5GPipeline)
        const std::string nvipc_yaml = rpa::create_nvipc_config();

        ran::phy_ran_app::FapiRxHandler::InitParams fapi_params{};
        fapi_params.nvipc_config_string = nvipc_yaml;
        fapi_params.max_cells = args.expected_cells;
        fapi_params.max_sfn = 1024;
        fapi_params.max_slot = 20;

        // ==================== FRONTHAUL INTEGRATION ====================

        // Initialize Fronthaul (DPDK for C-Plane TX, DOCA for U-Plane RX)
        RT_LOGC_INFO(rpa::PhyRanApp::App, "Initializing Fronthaul (DPDK/DOCA)...");

        static constexpr std::uint32_t DPDK_CORE = 0; // calls eal init and then sleeps forever
        static constexpr int RT_PRIORITY = 95;
        static constexpr std::uint32_t MONITOR_CORE = 0;

        // Task categories for worker assignment
        const framework::task::TaskCategory network_category{
                framework::task::BuiltinTaskCategory::Network};
        const framework::task::TaskCategory compute_category{
                framework::task::BuiltinTaskCategory::Compute};
        const framework::task::TaskCategory default_category{
                framework::task::BuiltinTaskCategory::Default};

        const auto fh_config_result = rpa::create_fronthaul_config(
                args.config_file_path,
                args.nic_pcie_addr,
                args.gpu_device_id,
                DPDK_CORE,
                args.slot_ahead);

        if (!fh_config_result.has_value()) {
            RT_LOGC_ERROR(
                    rpa::PhyRanApp::App,
                    "Failed to create Fronthaul configuration: {}",
                    fh_config_result.error());
            return EXIT_FAILURE;
        }

        const auto &fh_config = *fh_config_result;
        ran::fronthaul::Fronthaul fronthaul(fh_config);
        RT_LOGC_INFO(rpa::PhyRanApp::App, "Fronthaul initialized");

        // Calculate timing parameters for C-Plane transmission and trigger start
        // Uses GPS-based timing calculation (gps_alpha=0, gps_beta=0 for non-GPS deployments)
        const auto slot_period_ns = fh_config.numerology.slot_period_ns;
        const auto now_ns = framework::task::Time::now_ns().count();

        gsl_Expects(now_ns >= 0);
        const framework::task::StartTimeParams start_params{
                .current_time_ns = gsl_lite::narrow_cast<std::uint64_t>(now_ns),
                .period_ns = ran::oran::SFN_PERIOD_NS,
                .gps_alpha = 0, // No GPS frequency correction (not in YAML config)
                .gps_beta = 0,  // No GPS phase correction (not in YAML config)
        };

        const auto [start_time_ns, t0, tai_offset] =
                rpa::calculate_timing_parameters(start_params, args.slot_ahead, slot_period_ns);

        RT_LOGC_DEBUG(
                rpa::PhyRanApp::App,
                "Timing parameters: t0={}ns, tai_offset={}ns, slot_ahead={}",
                t0.count(),
                tai_offset.count(),
                args.slot_ahead);

        // ==================== UPLINK PROCESSING GRAPH ====================

        std::atomic<ran::fapi::SlotInfo> captured_slot_info{{0, 0}};
        // Create task scheduler with 3 pinned RT workers (one for each processing task)
        std::vector<framework::task::WorkerConfig> uplink_workers;
        static constexpr auto MAX_WORKERS = 4U;
        uplink_workers.reserve(MAX_WORKERS);
        uplink_workers.push_back(framework::task::WorkerConfig::create_pinned_rt(
                args.cplane_core, RT_PRIORITY, {network_category}));
        uplink_workers.push_back(framework::task::WorkerConfig::create_pinned_rt(
                args.uplane_core, RT_PRIORITY, {network_category}));
        uplink_workers.push_back(framework::task::WorkerConfig::create_pinned_rt(
                args.pusch_core, RT_PRIORITY, {compute_category}));

        auto uplink_scheduler = framework::task::TaskScheduler::create()
                                        .workers(framework::task::WorkersConfig{uplink_workers})
                                        .monitor_core(MONITOR_CORE)
                                        .build();

        framework::task::TaskGraph uplink_graph("uplink_processing");

        // Callback invoked when SLOT_RESPONSE received (triggers uplink graph scheduling)
        auto on_graph_schedule = [&uplink_scheduler, &uplink_graph, &captured_slot_info](
                                         ran::fapi::SlotInfo scheduled_slot) {
            RT_LOGC_INFO(
                    rpa::PhyRanApp::App,
                    "Graph schedule callback invoked for sfn {} slot {}, scheduling uplink graph",
                    scheduled_slot.sfn,
                    scheduled_slot.slot);
            captured_slot_info.store(scheduled_slot, std::memory_order_release);
            uplink_scheduler.schedule(uplink_graph);
            RT_LOGC_INFO(rpa::PhyRanApp::App, "Uplink graph scheduled");
        };

        RT_LOGC_INFO(rpa::PhyRanApp::App, "Initializing FAPI RX Handler with Message Adapter...");
        ran::phy_ran_app::FapiRxHandler fapi_rx_handler(
                fapi_params, std::move(on_graph_schedule), fronthaul);
        RT_LOGC_INFO(
                rpa::PhyRanApp::App,
                "FAPI RX Handler initialized (Message Adapter: Sample5GPipeline)");

        // Get slot info provider interface from FapiRxHandler
        auto *slot_info_provider = fapi_rx_handler.get_slot_info_provider();
        if (slot_info_provider == nullptr) {
            RT_LOGC_ERROR(rpa::PhyRanApp::App, "Failed to get slot info provider");
            running.store(false);
            return EXIT_FAILURE;
        }

        // Get pipeline executor interface from FapiRxHandler
        auto *pipeline_executor = fapi_rx_handler.get_pipeline_executor();
        if (pipeline_executor == nullptr) {
            RT_LOGC_ERROR(rpa::PhyRanApp::App, "Failed to get pipeline executor");
            running.store(false);
            return EXIT_FAILURE;
        }

        // Create task functions for uplink processing (C-plane → U-plane → PUSCH)
        auto process_cplane_func = rpa::make_process_cplane_func(
                fronthaul, *slot_info_provider, captured_slot_info, t0, tai_offset);

        auto process_uplane_func =
                rpa::make_process_uplane_func(fronthaul, *slot_info_provider, captured_slot_info);
        auto process_pusch_func =
                rpa::make_process_pusch_func(*pipeline_executor, captured_slot_info);

        // Build 3-task graph with serial dependencies and category assignments
        // Each task is assigned to a specific category that routes it to a dedicated RT worker
        auto cplane_task = uplink_graph.register_task("cplane_processing")
                                   .category(network_category)
                                   .function(process_cplane_func)
                                   .add();
        auto uplane_task = uplink_graph.register_task("uplane_processing")
                                   .category(network_category)
                                   .depends_on(cplane_task)
                                   .function(process_uplane_func)
                                   .add();
        uplink_graph.register_task("pusch_rx")
                .category(compute_category)
                .depends_on(uplane_task)
                .function(process_pusch_func)
                .add();
        uplink_graph.build();

        RT_LOGC_INFO(
                rpa::PhyRanApp::App, "3-task uplink graph created: C-Plane -> U-Plane -> PUSCH RX");

        // ==================== RX PROCESSING GRAPH ====================

        // Create separate scheduler for RX task with dedicated RT worker
        auto rx_scheduler =
                framework::task::TaskScheduler::create()
                        .workers(framework::task::WorkersConfig{
                                {framework::task::WorkerConfig::create_pinned_rt(
                                        args.rx_core, RT_PRIORITY, {default_category})}})
                        .monitor_core(MONITOR_CORE)
                        .build();

        // Create RX message processing task (runs continuously until running flag cleared)
        // FapiRxHandler polls nvIPC and forwards messages to Sample5GPipeline
        auto rx_graph = framework::task::TaskGraph::create("fapi_rx_graph")
                                .single_task("fapi_rx_task")
                                .category(default_category)
                                .function([&fapi_rx_handler, &running]() {
                                    fapi_rx_handler.receive_and_process_messages(running);
                                })
                                .build();

        // Schedule RX processing task
        rx_scheduler.schedule(rx_graph);
        RT_LOGC_INFO(rpa::PhyRanApp::App, "FAPI RX task scheduled on dedicated core");

        // ==================== SLOT INDICATION TIMED TRIGGER ====================

        // Get ISlotIndicationSender interface from FapiRxHandler
        auto *slot_sender = fapi_rx_handler.get_slot_indication_sender();
        if (slot_sender == nullptr) {
            RT_LOGC_ERROR(rpa::PhyRanApp::App, "Failed to get slot indication sender interface");
            running.store(false);
            return EXIT_FAILURE;
        }

        // Create slot indication function using factory
        auto slot_indication_func =
                rpa::make_slot_indication_func(*slot_sender, fapi_rx_handler, running);

        // Create timed trigger with 500us period (mimicking half-slot timing)
        constexpr auto SLOT_INDICATION_PERIOD = 500us;
        auto slot_indication_trigger_builder =
                framework::task::TimedTrigger::create(slot_indication_func, SLOT_INDICATION_PERIOD)
                        .pin_to_core(args.slot_indication_core)
                        .with_stats_core(MONITOR_CORE)
                        .with_rt_priority(RT_PRIORITY)
                        .enable_statistics();

        if (args.num_slots.has_value()) {
            // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
            slot_indication_trigger_builder =
                    slot_indication_trigger_builder.max_triggers(args.num_slots.value());
        }

        auto slot_indication_trigger = slot_indication_trigger_builder.build();

        // Setup signal handler for unlimited slots mode
        std::optional<std::reference_wrapper<std::atomic_bool>> stop_flag = std::nullopt;
        if (!args.num_slots.has_value()) {
            stop_flag = std::optional{std::ref(g_stop_requested)};
            std::signal(SIGINT, signal_handler);  // NOLINT(cert-err33-c)
            std::signal(SIGTERM, signal_handler); // NOLINT(cert-err33-c)
        }

        RT_LOGC_INFO(
                rpa::PhyRanApp::App,
                "Slot indication trigger created (period: {}µs)",
                std::chrono::duration_cast<std::chrono::microseconds>(SLOT_INDICATION_PERIOD)
                        .count());

        // ==================== WAIT FOR CELLS ====================

        // Wait for cells to be configured and started BEFORE starting slot indication trigger
        // Sample5GPipeline tracks cell state with atomics

        RT_LOGC_INFO(
                rpa::PhyRanApp::App,
                "Waiting for MAC to configure and start {} cells...",
                args.expected_cells);

        // Wait for expected cells to start
        // Poll Sample5GPipeline for cell count (lock-free atomics, safe to call frequently)
        while (running.load() && fapi_rx_handler.get_num_cells_running() < args.expected_cells) {
            std::this_thread::sleep_for(10ms);
        }

        const std::size_t num_cells_running = fapi_rx_handler.get_num_cells_running();

        if (num_cells_running == 0) {
            RT_LOGC_ERROR(rpa::PhyRanApp::App, "No cells started, exiting");
            running.store(false);
            return EXIT_FAILURE;
        }

        if (num_cells_running < args.expected_cells) {
            RT_LOGC_WARN(
                    rpa::PhyRanApp::App,
                    "Expected {} cells but only {} are running, continuing anyway",
                    args.expected_cells,
                    num_cells_running);
        }

        RT_LOGC_INFO(
                rpa::PhyRanApp::App,
                "Starting FAPI message processing: {} cells running",
                num_cells_running);

        RT_LOGC_INFO(
                rpa::PhyRanApp::App,
                "Message Adapter (Sample5GPipeline) is handling FAPI message processing");

        // ==================== START SLOT INDICATION TRIGGER ====================

        // Recalculate start_time_ns to ensure trigger starts at next SFN boundary
        // (original calculation was done before waiting for cells, may now be in the past)
        const auto now_ns_trigger = framework::task::Time::now_ns().count();
        gsl_Expects(now_ns_trigger >= 0);
        const framework::task::StartTimeParams start_params_trigger{
                .current_time_ns = gsl_lite::narrow_cast<std::uint64_t>(now_ns_trigger),
                .period_ns = ran::oran::SFN_PERIOD_NS,
                .gps_alpha = 0,
                .gps_beta = 0,
        };
        const auto [start_time_ns_trigger, t0_trigger, tai_offset_trigger] =
                rpa::calculate_timing_parameters(
                        start_params_trigger, args.slot_ahead, slot_period_ns);

        RT_LOGC_DEBUG(
                rpa::PhyRanApp::App,
                "Recalculated trigger timing: start_time={}ns (delta from original: {}ns)",
                start_time_ns_trigger,
                static_cast<std::int64_t>(start_time_ns_trigger) -
                        static_cast<std::int64_t>(start_time_ns));

        // Now that cells are running, start the slot indication trigger
        if (const auto start_result =
                    slot_indication_trigger.start(framework::task::Nanos{start_time_ns_trigger});
            start_result) {
            RT_LOGC_ERROR(
                    rpa::PhyRanApp::App,
                    "Failed to start slot indication trigger: {}",
                    start_result.message());
            running.store(false);
            return EXIT_FAILURE;
        }

        const auto delta_to_start_ns = static_cast<std::int64_t>(start_time_ns_trigger) -
                                       framework::task::Time::now_ns().count();
        const double wait_s = static_cast<double>(delta_to_start_ns) / 1e9;
        const std::string slot_info = args.num_slots.has_value()
                                              // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
                                              ? std::format("for {} slots", args.num_slots.value())
                                              : "(unlimited slots)";
        RT_LOGC_INFO(
                rpa::PhyRanApp::App,
                "Slot indication trigger started - will begin firing in {:0.3f} seconds at SFN 0 "
                "{}",
                wait_s,
                slot_info);

        // ==================== MAIN LOOP ====================

        RT_LOGC_INFO(rpa::PhyRanApp::App, "Waiting for slot indication trigger to complete...");
        slot_indication_trigger.wait_for_completion(stop_flag);

        RT_LOGC_INFO(rpa::PhyRanApp::App, "Shutting down...");

        // Join all workers to ensure clean shutdown
        running.store(false); // signal all tasks to exit, rx_scheduler.
        uplink_scheduler.join_workers();
        rx_scheduler.join_workers();

        // Print runtime statistics and validate if requested
        const auto &stats = fapi_rx_handler.get_stats();
        if (!rpa::print_and_validate_stats(stats, rpa::PrintAndValidate{args.validate})) {
            return EXIT_FAILURE;
        }

        return EXIT_SUCCESS;
    } catch (const std::exception &e) {
        RT_LOGC_ERROR(rpa::PhyRanApp::App, "Fatal error: {}", e.what());
        return EXIT_FAILURE;
    } catch (...) {
        RT_LOGC_ERROR(rpa::PhyRanApp::App, "Fatal error: unknown exception");
        return EXIT_FAILURE;
    }
    // UNREACHABLE
}

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
