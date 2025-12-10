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
 * @file fapi_sample.cpp
 * @brief FAPI state machine sample application
 *
 * Demonstrates the FAPI state machine by:
 * - Initializing NVIPC as primary (PHY side)
 * - Processing incoming CONFIG.request from MAC
 * - Processing START.request
 * - Running slot indication loop
 * - Processing STOP.request and cleanup
 */

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <format>
#include <iostream>
#include <mutex>
#include <optional>
#include <ratio>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include <quill/LogMacros.h>
#include <scf_5g_fapi.h>

#include "fapi/fapi_file_writer.hpp"
#include "fapi/fapi_log.hpp"
#include "fapi/fapi_state.hpp"
#include "fapi_sample_utils.hpp"
#include "log/rt_log_macros.hpp"
#include "task/task_graph.hpp"
#include "task/task_scheduler.hpp"
#include "task/timed_trigger.hpp"

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

/**
 * Main application entry point
 *
 * @param[in] argc Number of command line arguments
 * @param[in] argv Array of command line argument strings
 * @return EXIT_SUCCESS on successful completion, EXIT_FAILURE on error
 */
int main(int argc, const char **argv) {
    using namespace ran::fapi;
    using namespace framework::task;
    using namespace std::chrono_literals;

    try {
        fapi_sample::setup_logging();

        const auto parse_result = fapi_sample::parse_arguments(argc, argv);
        if (!parse_result.has_config()) {
            return parse_result.exit_code;
        }

        // Safe to access config here since has_config() returned true
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        const auto &config = parse_result.config.value();
        // Initialize FAPI state machine with NVIPC from YAML string
        const std::string yaml_content = fapi_sample::create_nvipc_config();

        FapiState::InitParams params{};
        params.nvipc_config_string = yaml_content;
        params.max_cells = config.expected_cells;
        params.max_sfn = 1024;
        params.max_slot = 20;

        RT_LOGC_DEBUG(ran::fapi::FapiComponent::FapiSample, "Initializing FAPI state machine...");
        FapiState fapi_state(params);
        RT_LOGC_DEBUG(ran::fapi::FapiComponent::FapiSample, "FAPI state machine initialized");

        // Setup FAPI message capture if requested
        std::optional<FapiFileWriter> file_writer;
        if (!config.capture_file_path.empty()) {
            file_writer.emplace(config.capture_file_path);

            fapi_state.set_on_message([&file_writer](const FapiMessageData &msg) {
                file_writer->capture_message(msg);
            });

            RT_LOGC_INFO(
                    ran::fapi::FapiComponent::FapiSample,
                    "FAPI message capture enabled: {}",
                    config.capture_file_path);
        }

        // Per-cell message counters and callbacks
        std::vector<fapi_sample::MessageCounters> cell_counters(params.max_cells);
        fapi_sample::setup_message_callbacks(fapi_state, cell_counters, params.max_cells);

        std::atomic<bool> running{true};
        std::mutex cells_mutex;
        std::condition_variable cells_cv;

        // Set up callback to validate start requests and notify when cells start running
        fapi_state.set_on_start_request(
                [&cells_cv](
                        const uint16_t cell_id,
                        [[maybe_unused]] const scf_fapi_body_header_t &body_hdr,
                        [[maybe_unused]] const uint32_t body_len) {
                    RT_LOGC_DEBUG(
                            ran::fapi::FapiComponent::FapiSample,
                            "Cell {} start request validated, notifying waiter",
                            cell_id);
                    cells_cv.notify_one();
                    return SCF_ERROR_CODE_MSG_OK;
                });

        // Create task scheduler with one worker for processing RX messages
        auto scheduler = TaskScheduler::create().workers(1).build();

        // Create and schedule task for processing incoming messages (runs continuously)
        auto rx_graph = TaskGraph::create("rx_message_graph")
                                .single_task("rx_message_task")
                                .function([&fapi_state, &running]() {
                                    fapi_sample::process_rx_messages(fapi_state, running);
                                })
                                .build();

        // Schedule RX message processing once - it will run continuously on worker
        scheduler.schedule(rx_graph);

        RT_LOGC_INFO(
                ran::fapi::FapiComponent::FapiSample,
                "Waiting for MAC to configure and start {} cells...",
                config.expected_cells);

        // Wait for expected number of cells to be configured and started
        {
            std::unique_lock<std::mutex> lock(cells_mutex);
            cells_cv.wait(lock, [&fapi_state, &config, &running]() {
                return fapi_state.get_num_cells_running() >= config.expected_cells ||
                       !running.load();
            });
        }

        // Check if we exited due to interrupt signal
        if (!running.load()) {
            const auto num_cells = fapi_state.get_num_cells_running();
            RT_LOGC_WARN(
                    ran::fapi::FapiComponent::FapiSample,
                    "Startup interrupted by user/signal ({} of {} cells started)",
                    num_cells,
                    config.expected_cells);
            return EXIT_FAILURE;
        }

        const std::size_t num_cells_running = fapi_state.get_num_cells_running();

        // Handle no cells case - hard fail
        if (num_cells_running == 0) {
            RT_LOGC_ERROR(ran::fapi::FapiComponent::FapiSample, "No cells started, exiting");
            running.store(false);
            return EXIT_FAILURE;
        }

        // Handle partial cell startup - warn but continue
        if (num_cells_running < config.expected_cells) {
            RT_LOGC_WARN(
                    ran::fapi::FapiComponent::FapiSample,
                    "Expected {} cells but only {} are running, continuing anyway",
                    config.expected_cells,
                    num_cells_running);
        }

        RT_LOGC_INFO(
                ran::fapi::FapiComponent::FapiSample,
                "Starting slot indication loop: {} cells running (expected {})",
                num_cells_running,
                config.expected_cells);

        // Create timed trigger for slot indications
        std::atomic<std::uint64_t> slots_sent{0};
        auto slot_trigger_function = fapi_sample::make_slot_trigger_func(
                fapi_state, running, slots_sent, config.test_slots);
        auto slot_trigger =
                TimedTrigger::create(slot_trigger_function, config.slot_interval).build();

        // Start slot indication trigger
        if (const auto start_result = slot_trigger.start(); start_result) {
            RT_LOGC_ERROR(
                    ran::fapi::FapiComponent::FapiSample,
                    "Failed to start slot trigger: {}",
                    start_result.message());
            running.store(false);
            return EXIT_FAILURE;
        }

        // Wait until running is set to false (by slot trigger or external signal)
        while (running.load()) {
            std::this_thread::sleep_for(100ms);
        }

        // Signal shutdown and stop trigger and workers
        running.store(false);
        slot_trigger.stop();
        scheduler.join_workers();

        // Print statistics
        slot_trigger.print_summary();
        scheduler.print_monitor_stats();

        RT_LOGC_INFO(
                ran::fapi::FapiComponent::FapiSample,
                "FAPI sample completed: {} slots sent, {} cells configured, {} cells running",
                slots_sent.load(),
                fapi_state.get_num_cells_configured(),
                fapi_state.get_num_cells_running());

        const auto actual_counts =
                fapi_sample::snapshot_counters(cell_counters, config.expected_cells);
        RT_LOGC_DEBUG(
                ran::fapi::FapiComponent::FapiSample, "Per-cell message counts: {}", actual_counts);

        // Perform validation if requested
        const bool validation_passed = fapi_sample::perform_validation(config, cell_counters);

        // Write captured messages to file
        if (file_writer.has_value()) {
            RT_LOGC_INFO(
                    ran::fapi::FapiComponent::FapiSample,
                    "Writing {} captured messages ({} bytes) to file",
                    file_writer->get_message_count(),
                    file_writer->get_buffer_size_bytes());

            try {
                file_writer->flush_to_file();
                RT_LOGC_INFO(
                        ran::fapi::FapiComponent::FapiSample,
                        "Capture file written successfully: {}",
                        config.capture_file_path);
            } catch (const std::exception &e) {
                RT_LOGC_ERROR(
                        ran::fapi::FapiComponent::FapiSample,
                        "Failed to write capture file: {}",
                        e.what());
            }
        }

        // FapiState destructor handles NVIPC cleanup
        return validation_passed ? EXIT_SUCCESS : EXIT_FAILURE;

    } catch (const std::exception &e) {
        std::cerr << std::format("Unhandled exception: {}\n", e.what());
        return EXIT_FAILURE;
    } catch (...) {
        std::cerr << "Unknown exception occurred\n";
        return EXIT_FAILURE;
    }
}

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
