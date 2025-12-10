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
 * @file fronthaul_app.cpp
 * @brief Fronthaul library sample application
 */

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <format>
#include <functional>
#include <iostream>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

#include <quill/LogMacros.h>
#include <tl/expected.hpp>

#include "fapi/fapi_file_replay.hpp"
#include "fronthaul/fronthaul.hpp"
#include "fronthaul/fronthaul_log.hpp"
#include "fronthaul/order_kernel_descriptors.hpp"
#include "fronthaul/uplane_config.hpp"
#include "fronthaul_app_utils.hpp"
#include "log/rt_log_macros.hpp"
#include "net/dpdk_types.hpp"
#include "oran/numerology.hpp"
#include "task/task_category.hpp"
#include "task/task_errors.hpp"
#include "task/task_graph.hpp"
#include "task/task_scheduler.hpp"
#include "task/task_utils.hpp"
#include "task/task_worker.hpp"
#include "task/time.hpp"
#include "task/timed_trigger.hpp"

namespace ft = framework::task;

namespace {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::atomic_bool g_stop_requested{false};

void signal_handler(int signum) {
    if (signum == SIGINT || signum == SIGTERM) {
        g_stop_requested = true;
    }
}

} // namespace

/**
 * Main application entry point
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Argument strings
 * @return EXIT_SUCCESS or EXIT_FAILURE
 */
int main(const int argc, const char **argv) {
    namespace rf = ran::fronthaul;

    try {
        ran::fronthaul::samples::setup_logging();

        const auto args = ran::fronthaul::samples::parse_arguments(argc, argv);
        if (!args.has_value()) {
            // Empty error string means --help or --version was shown (success)
            // Non-empty error string means parse error (failure)
            if (!args.error().empty()) {
                RT_LOGC_ERROR(rf::FronthaulApp::App, "{}", args.error());
                return EXIT_FAILURE;
            }
            return EXIT_SUCCESS;
        }

        RT_LOGC_INFO(rf::FronthaulApp::App, "Parsed arguments: {}", *args);

        // Create fronthaul configuration from YAML
        static constexpr std::uint32_t DPDK_CORE = 0;
        const auto fh_config_opt = ran::fronthaul::samples::create_fronthaul_config_from_yaml(
                args->config_file_path,
                args->nic_pcie_addr,
                args->gpu_device_id,
                DPDK_CORE,
                args->slot_ahead);

        if (!fh_config_opt.has_value()) {
            RT_LOGC_ERROR(
                    rf::FronthaulApp::Config,
                    "Failed to create fronthaul config from YAML: {}",
                    args->config_file_path);
            return EXIT_FAILURE;
        }

        // Get mutable copy to add U-Plane configuration
        auto fh_config = fh_config_opt.value();

        // Get FAPI capture file (from CLI or auto-detect in executable directory)
        const std::string fapi_file_path =
                // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
                ran::fronthaul::samples::get_fapi_file_path(args->fapi_file_path, argv[0]);
        RT_LOGC_DEBUG(rf::FronthaulApp::App, "Using FAPI capture file: {}", fapi_file_path);

        // Create FAPI replay
        ran::fapi::FapiFileReplay fapi_replay(
                fapi_file_path, static_cast<std::uint8_t>(fh_config.numerology.slots_per_subframe));

        RT_LOGC_INFO(
                rf::FronthaulApp::App,
                "Loaded {} requests from {} cells",
                fapi_replay.get_total_request_count(),
                fapi_replay.get_cell_count());

        // Validate cell count matches
        if (fh_config.cell_dest_macs.size() != fapi_replay.get_cell_count()) {
            RT_LOGC_ERROR(
                    rf::FronthaulApp::App,
                    "Cell count mismatch: config has {}, FAPI file has {}",
                    fh_config.cell_dest_macs.size(),
                    fapi_replay.get_cell_count());
            return EXIT_FAILURE;
        }

        // Validate we have at least one cell configured
        if (fh_config.cell_dest_macs.empty()) {
            RT_LOGC_ERROR(
                    rf::FronthaulApp::App,
                    "No cells configured - fronthaul requires at least one cell");
            return EXIT_FAILURE;
        }

        // Validate cell count against ORDER_KERNEL maximum
        static constexpr std::size_t MAX_CELLS_SUPPORTED = rf::ORDER_KERNEL_MAX_CELLS_PER_SLOT;
        if (fh_config.cell_dest_macs.size() > MAX_CELLS_SUPPORTED) {
            RT_LOGC_ERROR(
                    rf::FronthaulApp::App,
                    "Cell count exceeds maximum supported by OrderKernel. "
                    "Got: {}, Maximum: {} (ORDER_KERNEL_MAX_CELLS_PER_SLOT)",
                    fh_config.cell_dest_macs.size(),
                    MAX_CELLS_SUPPORTED);
            return EXIT_FAILURE;
        }

        // example-begin fronthaul-construction-1
        // U-Plane configuration is already set in fh_config from create_fronthaul_config_from_yaml
        RT_LOGC_INFO(
                rf::FronthaulApp::UPlane,
                "U-Plane configured: slot_duration={}ns, Ta4 window=[{}ns, {}ns], RU MAC={}",
                fh_config.uplane_config.slot_duration_ns,
                fh_config.uplane_config.ta4_min_ns,
                fh_config.uplane_config.ta4_max_ns,
                fh_config.cell_dest_macs[0].to_string());

        // Create fronthaul instance (now handles both C-Plane and U-Plane)
        rf::Fronthaul fronthaul(fh_config);
        // example-end fronthaul-construction-1

        const auto slot_period_ns = fh_config.numerology.slot_period_ns;
        RT_LOGC_INFO(
                rf::FronthaulApp::App,
                "Fronthaul ready: slot_period={}ns, {} cells",
                slot_period_ns,
                fh_config.cell_dest_macs.size());

        // Create task scheduler with pinned RT worker
        static constexpr int RT_PRIORITY = 95;
        static constexpr std::uint32_t MONITOR_CORE = 0;
        const std::vector<ft::TaskCategory> worker_categories = {
                ft::TaskCategory{ft::BuiltinTaskCategory::Network},
                ft::TaskCategory{ft::BuiltinTaskCategory::Default}};
        auto scheduler = ft::TaskScheduler::create()
                                 .workers(ft::WorkersConfig{{ft::WorkerConfig::create_pinned_rt(
                                         args->worker_core, RT_PRIORITY, worker_categories)}})
                                 .monitor_core(MONITOR_CORE)
                                 .build();

        // Calculate TAI offset and timing parameters
        const auto now_ns = ft::Time::now_ns().count();
        const ft::StartTimeParams start_params{
                .current_time_ns = static_cast<std::uint64_t>(now_ns),
                .period_ns = ran::oran::SFN_PERIOD_NS,
                .gps_alpha = fh_config.gps_alpha,
                .gps_beta = fh_config.gps_beta,
        };

        const auto [start_time_ns, t0, tai_offset] =
                ran::fronthaul::samples::calculate_timing_parameters(
                        start_params, args->slot_ahead, slot_period_ns);

        // Create task graph with sequential C-Plane and U-Plane processing
        // Task execution order: process_cplane (C-Plane TX) -> process_uplane (U-Plane RX)
        ft::TaskGraph graph("fronthaul_processing");

        bool is_first_slot = true; // keep track of first slot (avoid skipping slot 0)
        auto cplane_task = graph.register_task("process_cplane")
                                   .category(ft::BuiltinTaskCategory::Network)
                                   .function(ran::fronthaul::samples::make_process_cplane_func(
                                           fronthaul, fapi_replay, is_first_slot, t0, tai_offset))
                                   .add();

        graph.register_task("process_uplane")
                .category(ft::BuiltinTaskCategory::Network)
                .depends_on(cplane_task)
                .function(ran::fronthaul::samples::make_process_uplane_func(fronthaul, fapi_replay))
                .add();

        graph.build();

        // Create timed trigger
        auto trigger_builder = ft::TimedTrigger::create(
                                       [&scheduler, &graph]() { scheduler.schedule(graph); },
                                       std::chrono::nanoseconds{slot_period_ns})
                                       .pin_to_core(args->trigger_core)
                                       .with_stats_core(MONITOR_CORE)
                                       .with_rt_priority(RT_PRIORITY)
                                       .enable_statistics();

        if (args->num_slots.has_value()) {
            // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
            trigger_builder = trigger_builder.max_triggers(args->num_slots.value());
        }

        auto trigger = trigger_builder.build();

        // Setup signal handler for unlimited slots mode
        std::optional<std::reference_wrapper<std::atomic_bool>> stop_flag = std::nullopt;
        if (!args->num_slots.has_value()) {
            stop_flag = std::optional{std::ref(g_stop_requested)};
            std::signal(SIGINT, signal_handler);  // NOLINT(cert-err33-c)
            std::signal(SIGTERM, signal_handler); // NOLINT(cert-err33-c)
        }

        // Start trigger with calculated start time
        if (const auto result = trigger.start(ft::Nanos{start_time_ns}); result) {
            RT_LOGC_ERROR(
                    rf::FronthaulApp::App,
                    "Failed to start trigger: {}",
                    ft::get_error_name(result));
            return EXIT_FAILURE;
        }

        const auto delta_to_start_ns =
                static_cast<std::int64_t>(start_time_ns) - ft::Time::now_ns().count();
        const double wait_s = static_cast<double>(delta_to_start_ns) / 1e9;
        const std::string slot_info = args->num_slots.has_value()
                                              // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
                                              ? std::format("for {} slots", args->num_slots.value())
                                              : "(unlimited slots)";
        RT_LOGC_INFO(
                rf::FronthaulApp::App,
                "Waiting {:0.3f} seconds for SFN 0 before starting fronthaul app {}...",
                wait_s,
                slot_info);

        // Wait for completion of number of slots or termination signal
        trigger.wait_for_completion(stop_flag);

        // Cleanup
        scheduler.join_workers();

        // Print statistics (conditionally prints task stats based on slot count)
        ran::fronthaul::samples::print_statistics(fronthaul, scheduler, trigger, args->num_slots);

        // Perform validation if requested
        if (args->validate_uplane_prbs) {
            const bool validation_passed =
                    ran::fronthaul::samples::validate_kernel_results(fronthaul);
            return validation_passed ? EXIT_SUCCESS : EXIT_FAILURE;
        }

        return EXIT_SUCCESS;

    } catch (const std::exception &e) {
        std::cerr << std::format("Unhandled exception: {}\n", e.what());
        return EXIT_FAILURE;
    } catch (...) {
        std::cerr << "Unknown exception occurred\n";
        return EXIT_FAILURE;
    }
}
