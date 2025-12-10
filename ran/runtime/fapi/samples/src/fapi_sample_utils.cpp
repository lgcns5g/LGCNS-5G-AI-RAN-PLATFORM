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
 * @file fapi_sample_utils.cpp
 * @brief Implementation of utility functions for FAPI sample application
 */

#include <algorithm>
#include <array>
#include <cstdlib>
#include <exception>
#include <format>
#include <string_view>
#include <thread>

#include <nv_ipc.h>
#include <quill/LogMacros.h>
#include <scf_5g_fapi.h>
#include <yaml.hpp>

#include <CLI/CLI.hpp>

#include "fapi/fapi_log.hpp"
#include "fapi/fapi_utils.hpp"
#include "fapi_sample_utils.hpp"
#include "internal_use_only/config.hpp"
#include "log/components.hpp"
#include "log/rt_log.hpp"
#include "log/rt_log_macros.hpp"
#include "task/task_log.hpp"
#include "task/task_utils.hpp"

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

namespace fapi_sample {

using namespace std::chrono_literals;

std::size_t parse_cell_count(const std::filesystem::path &launch_pattern_file) {
    try {
        yaml::file_parser launch_parser(launch_pattern_file.c_str());
        yaml::document launch_doc = launch_parser.next_document();
        const yaml::node launch_root = launch_doc.root();
        yaml::node cell_configs = launch_root["Cell_Configs"];
        return cell_configs.length();
    } catch (const std::exception &e) {
        RT_LOGC_ERROR(
                ran::fapi::FapiComponent::FapiSample,
                "Failed to parse cell count from launch pattern: {}",
                e.what());
        return 0;
    }
}

std::optional<std::size_t> parse_test_slots(const std::filesystem::path &test_mac_config_file) {
    try {
        yaml::file_parser test_mac_parser(test_mac_config_file.c_str());
        yaml::document test_mac_doc = test_mac_parser.next_document();
        const yaml::node test_mac_root = test_mac_doc.root();
        const std::size_t test_slots_value = test_mac_root["test_slots"].as<std::size_t>();

        // Return nullopt for 0 (unlimited), otherwise return the value
        return test_slots_value > 0 ? std::optional<std::size_t>{test_slots_value} : std::nullopt;
    } catch (const std::exception &e) {
        RT_LOGC_ERROR(
                ran::fapi::FapiComponent::FapiSample,
                "Failed to parse test_slots from test_mac config: {}",
                e.what());
        return std::nullopt;
    }
}

std::optional<std::vector<MessageCounts>> parse_expected_counts(
        const std::filesystem::path &launch_pattern_file,
        const std::filesystem::path &test_mac_config_file) {
    try {
        // Parse launch pattern file to get Cell_Configs pattern filename
        yaml::file_parser launch_parser(launch_pattern_file.c_str());
        yaml::document launch_doc = launch_parser.next_document();
        const yaml::node launch_root = launch_doc.root();

        // Parse test_mac config file to get test_slots
        yaml::file_parser test_mac_parser(test_mac_config_file.c_str());
        yaml::document test_mac_doc = test_mac_parser.next_document();
        const yaml::node test_mac_root = test_mac_doc.root();

        const std::size_t test_slots = test_mac_root["test_slots"].as<std::size_t>();

        // Parse SCHED section to determine pattern length and slots with data
        yaml::node sched = launch_root["SCHED"];
        const std::size_t pattern_length = sched.length();
        if (pattern_length == 0) {
            RT_LOGC_ERROR(
                    ran::fapi::FapiComponent::FapiSample,
                    "SCHED section not found or empty in launch pattern");
            return std::nullopt;
        }

        // Count slots with non-empty configurations in the pattern
        std::size_t slots_with_data{};
        for (std::size_t slot_idx{}; slot_idx < pattern_length; ++slot_idx) {
            const yaml::node slot_entry = sched[slot_idx];

            // Check if config is non-empty (has cell configurations)
            if (slot_entry["config"].length() > 0) {
                ++slots_with_data;
            }
        }

        if (slots_with_data == 0) {
            RT_LOGC_ERROR(
                    ran::fapi::FapiComponent::FapiSample,
                    "No slots with data found in SCHED section");
            return std::nullopt;
        }

        // Validate test_slots is divisible by pattern_length to ensure accurate counting
        if (test_slots % pattern_length != 0) {
            RT_LOGC_ERROR(
                    ran::fapi::FapiComponent::FapiSample,
                    "test_slots ({}) must be evenly divisible by pattern_length ({})",
                    test_slots,
                    pattern_length);
            return std::nullopt;
        }

        // Calculate expected message count: (test_slots / pattern_length) * slots_with_data
        const std::size_t slot_pattern_cycles = test_slots / pattern_length;
        const std::size_t expected_msg_count = slot_pattern_cycles * slots_with_data;

        RT_LOGC_DEBUG(
                ran::fapi::FapiComponent::FapiSample,
                "Launch pattern analysis: pattern_length={}, slots_with_data={}, "
                "test_slots={}, slot_pattern_cycles={}, expected_messages={}",
                pattern_length,
                slots_with_data,
                test_slots,
                slot_pattern_cycles,
                expected_msg_count);

        // Iterate through Cell_Configs to calculate expected counts per cell
        yaml::node cell_configs = launch_root["Cell_Configs"];
        const std::size_t num_cells = cell_configs.length();

        if (num_cells == 0) {
            RT_LOGC_ERROR(
                    ran::fapi::FapiComponent::FapiSample, "No cells found in Cell_Configs array");
            return std::nullopt;
        }

        std::vector<MessageCounts> expected_per_cell;
        expected_per_cell.reserve(num_cells);

        static constexpr std::array<std::string_view, 2> KNOWN_PATTERNS{"7201", "7204"};

        // Calculate expected counts for each cell based on its pattern
        for (std::size_t i{}; i < num_cells; ++i) {
            const std::string pattern_file = cell_configs[i].as<std::string>();

            const bool is_known_pattern = std::ranges::any_of(
                    KNOWN_PATTERNS, [&pattern_file](const std::string_view pattern) {
                        return pattern_file.find(pattern) != std::string::npos;
                    });

            MessageCounts expected{};

            // For 7201/7204 pattern: calculate based on actual pattern structure
            if (is_known_pattern) {
                expected.ul_tti_requests = expected_msg_count;
                expected.dl_tti_requests = 0;
                expected.slot_responses = expected_msg_count;

                RT_LOGC_DEBUG(
                        ran::fapi::FapiComponent::FapiSample,
                        "Cell {}: pattern=7201/7204, test_slots={}, slot_pattern_cycles={}, "
                        "expected UL_TTI={}, DL_TTI={}, SLOT_RESPONSE={}",
                        i,
                        test_slots,
                        slot_pattern_cycles,
                        expected.ul_tti_requests,
                        expected.dl_tti_requests,
                        expected.slot_responses);
            } else {
                RT_LOGC_ERROR(
                        ran::fapi::FapiComponent::FapiSample,
                        "Cell {}: Unknown pattern type in {}",
                        i,
                        pattern_file);
                return std::nullopt;
            }

            expected_per_cell.push_back(expected);
        }

        return expected_per_cell;

    } catch (const std::exception &e) {
        RT_LOGC_ERROR(
                ran::fapi::FapiComponent::FapiSample,
                "Failed to parse config files for validation: {}",
                e.what());
        return std::nullopt;
    }
}

bool validate_message_counts(const MessageCounts &expected, const MessageCounters &actual) {
    bool validation_passed{true};

    const std::size_t actual_ul_tti = actual.ul_tti_request_count.load();
    const std::size_t actual_dl_tti = actual.dl_tti_request_count.load();
    const std::size_t actual_slot_response = actual.slot_response_count.load();

    if (actual_ul_tti != expected.ul_tti_requests) {
        RT_LOGC_ERROR(
                ran::fapi::FapiComponent::FapiSample,
                "Validation FAILED: UL_TTI_REQUEST count mismatch. Expected={}, Actual={}",
                expected.ul_tti_requests,
                actual_ul_tti);
        validation_passed = false;
    } else {
        RT_LOGC_DEBUG(
                ran::fapi::FapiComponent::FapiSample,
                "Validation PASSED: UL_TTI_REQUEST count matches ({})",
                actual_ul_tti);
    }

    if (actual_dl_tti != expected.dl_tti_requests) {
        RT_LOGC_ERROR(
                ran::fapi::FapiComponent::FapiSample,
                "Validation FAILED: DL_TTI_REQUEST count mismatch. Expected={}, Actual={}",
                expected.dl_tti_requests,
                actual_dl_tti);
        validation_passed = false;
    } else {
        RT_LOGC_DEBUG(
                ran::fapi::FapiComponent::FapiSample,
                "Validation PASSED: DL_TTI_REQUEST count matches ({})",
                actual_dl_tti);
    }

    if (actual_slot_response != expected.slot_responses) {
        RT_LOGC_ERROR(
                ran::fapi::FapiComponent::FapiSample,
                "Validation FAILED: SLOT_RESPONSE count mismatch. Expected={}, Actual={}",
                expected.slot_responses,
                actual_slot_response);
        validation_passed = false;
    } else {
        RT_LOGC_DEBUG(
                ran::fapi::FapiComponent::FapiSample,
                "Validation PASSED: SLOT_RESPONSE count matches ({})",
                actual_slot_response);
    }

    return validation_passed;
}

bool perform_validation(
        const AppConfig &config, const std::vector<MessageCounters> &cell_counters) {
    if (!config.validate_message_counts) {
        return true;
    }

    const auto actual_counts = snapshot_counters(cell_counters, config.expected_cells);
    RT_LOGC_DEBUG(
            ran::fapi::FapiComponent::FapiSample,
            "Running validation: launch_pattern_file={}, test_mac_config_file={}, "
            "per_cell_counters={}",
            config.launch_pattern_file,
            config.test_mac_config_file,
            actual_counts);

    // Parse config files to get expected counts per cell
    const auto expected_per_cell_opt =
            parse_expected_counts(config.launch_pattern_file, config.test_mac_config_file);

    if (!expected_per_cell_opt.has_value()) {
        RT_LOGC_ERROR(
                ran::fapi::FapiComponent::FapiSample,
                "Validation failed due to config parse error");
        return false;
    }

    const auto &expected_per_cell = expected_per_cell_opt.value();

    if (expected_per_cell.size() != config.expected_cells) {
        RT_LOGC_ERROR(
                ran::fapi::FapiComponent::FapiSample,
                "Validation failed: expected {} cells but config has {}",
                config.expected_cells,
                expected_per_cell.size());
        return false;
    }

    bool all_cells_passed{true};

    // Validate each cell individually
    for (std::size_t i{}; i < config.expected_cells; ++i) {
        const auto &cell_counter = cell_counters[i];
        const auto &expected_for_cell = expected_per_cell[i];

        const bool cell_passed = validate_message_counts(expected_for_cell, cell_counter);

        if (!cell_passed) {
            RT_LOGC_ERROR(ran::fapi::FapiComponent::FapiSample, "Cell {} validation FAILED", i);
            all_cells_passed = false;
        } else {
            RT_LOGC_DEBUG(ran::fapi::FapiComponent::FapiSample, "Cell {} validation PASSED", i);
        }
    }

    if (all_cells_passed) {
        RT_LOGC_INFO(ran::fapi::FapiComponent::FapiSample, "=== ALL CELLS VALIDATION PASSED ===");
    } else {
        RT_LOGC_ERROR(
                ran::fapi::FapiComponent::FapiSample,
                "=== VALIDATION FAILED FOR ONE OR MORE CELLS ===");
    }

    return all_cells_passed;
}

void setup_logging() {
    framework::task::enable_sanitizer_compatibility();
    framework::log::Logger::set_level(framework::log::LogLevel::Debug);
    framework::log::register_component<ran::fapi::FapiComponent>(framework::log::LogLevel::Debug);
    framework::log::register_component<framework::task::TaskLog>(framework::log::LogLevel::Debug);
}

void setup_message_callbacks(
        ran::fapi::FapiState &fapi_state,
        std::vector<MessageCounters> &cell_counters,
        const std::size_t max_cells) {
    fapi_state.set_on_ul_tti_request(
            [&cell_counters, max_cells](
                    const uint16_t cell_id,
                    [[maybe_unused]] const scf_fapi_body_header_t &body_hdr,
                    [[maybe_unused]] const uint32_t body_len) {
                if (cell_id < max_cells) {
                    cell_counters[cell_id].ul_tti_request_count.fetch_add(
                            1, std::memory_order_relaxed);
                }
            });

    fapi_state.set_on_dl_tti_request(
            [&cell_counters, max_cells](
                    const uint16_t cell_id,
                    [[maybe_unused]] const scf_fapi_body_header_t &body_hdr,
                    [[maybe_unused]] const uint32_t body_len) {
                if (cell_id < max_cells) {
                    cell_counters[cell_id].dl_tti_request_count.fetch_add(
                            1, std::memory_order_relaxed);
                }
            });

    fapi_state.set_on_slot_response([&cell_counters, max_cells](
                                            const uint16_t cell_id,
                                            [[maybe_unused]] const scf_fapi_body_header_t &body_hdr,
                                            [[maybe_unused]] const uint32_t body_len) {
        if (cell_id < max_cells) {
            cell_counters[cell_id].slot_response_count.fetch_add(1, std::memory_order_relaxed);
        }
    });
}

ParseResult parse_arguments(const int argc, const char **argv) {
    using namespace std::chrono_literals;

    CLI::App app{std::format(
            "FAPI State Machine Sample - {} version {}",
            framework::cmake::project_name,
            framework::cmake::project_version)};

    AppConfig config{};

    std::size_t slot_interval_us{static_cast<std::size_t>(config.slot_interval.count())};
    app.add_option("--slot_interval_us", slot_interval_us, "Slot interval in microseconds")
            ->check(CLI::Range(250, 1000000));

    app.add_flag(
            "--validate",
            config.validate_message_counts,
            "Validate message counts against expected values");

    app.add_option(
            "--launch_pattern_file",
            config.launch_pattern_file,
            "Path to launch pattern YAML file (required)");

    app.add_option(
            "--test_mac_config_file",
            config.test_mac_config_file,
            "Path to test_mac config YAML file (required)");

    app.add_option(
            "--capture_file",
            config.capture_file_path,
            "Path to capture file for recording FAPI messages (optional)");

    app.set_version_flag(
            "--version",
            std::string{framework::cmake::project_version},
            "Show version information");

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
        const int exit_code = app.exit(e);
        return ParseResult{std::nullopt, exit_code};
    }

    // Check required arguments (after help/version handling)
    if (config.launch_pattern_file.empty()) {
        RT_LOGC_ERROR(
                ran::fapi::FapiComponent::FapiSample,
                "Missing required argument: --launch_pattern_file");
        return ParseResult{std::nullopt, EXIT_FAILURE};
    }

    if (config.test_mac_config_file.empty()) {
        RT_LOGC_ERROR(
                ran::fapi::FapiComponent::FapiSample,
                "Missing required argument: --test_mac_config_file");
        return ParseResult{std::nullopt, EXIT_FAILURE};
    }

    // Update config with parsed values
    config.slot_interval = std::chrono::microseconds{slot_interval_us};

    // Parse expected cell count from launch pattern file (required)
    const std::size_t cell_count = parse_cell_count(config.launch_pattern_file);
    if (cell_count == 0) {
        RT_LOGC_ERROR(
                ran::fapi::FapiComponent::FapiSample,
                "Failed to parse cell count from launch pattern file: {}",
                config.launch_pattern_file);
        return ParseResult{std::nullopt, EXIT_FAILURE};
    }

    // Validate cell count against FAPI maximum
    if (cell_count > ran::fapi::FapiState::InitParams::DEFAULT_MAX_CELLS) {
        RT_LOGC_ERROR(
                ran::fapi::FapiComponent::FapiSample,
                "Cell count exceeds maximum supported by FAPI. "
                "Got: {}, Maximum: {} (FapiState::InitParams::DEFAULT_MAX_CELLS)",
                cell_count,
                ran::fapi::FapiState::InitParams::DEFAULT_MAX_CELLS);
        return ParseResult{std::nullopt, EXIT_FAILURE};
    }

    config.expected_cells = cell_count;

    // Parse test_slots from test_mac config file
    config.test_slots = parse_test_slots(config.test_mac_config_file);

    RT_LOGC_INFO(ran::fapi::FapiComponent::FapiSample, "FAPI Sample config: {}", config);

    return ParseResult{config, EXIT_SUCCESS};
}

void process_rx_messages(ran::fapi::FapiState &fapi_state, std::atomic<bool> &running) {
    // Continuously process messages until shutdown
    while (running.load()) {
        nv_ipc_msg_t msg{};
        const int ret = fapi_state.receive_message(msg);

        if (ret < 0) {
            // No message available, sleep briefly
            static constexpr auto RX_IDLE_SLEEP = 100us;
            std::this_thread::sleep_for(RX_IDLE_SLEEP);
            continue;
        }

        // Process the message through FapiState
        const auto result = fapi_state.process_message(msg);
        if (result != SCF_ERROR_CODE_MSG_OK) {
            RT_LOGC_ERROR(
                    ran::fapi::FapiComponent::FapiSample,
                    "Failed to process message: error_code={}",
                    static_cast<int>(result));
        }

        // Release the message buffer
        fapi_state.release_message(msg);
    }
}

std::string create_nvipc_config() { return ran::fapi::create_default_nvipc_config("fapi_sample"); }

std::function<void()> make_slot_trigger_func(
        ran::fapi::FapiState &fapi_state,
        std::atomic<bool> &running,
        std::atomic<std::uint64_t> &slots_sent,
        const std::optional<std::size_t> test_slots) {
    const bool has_slot_limit = test_slots.has_value();
    const std::size_t slot_limit = has_slot_limit ? test_slots.value() : 0;

    return [&fapi_state, &running, &slots_sent, has_slot_limit, slot_limit]() {
        // Check if we've sent the requested number of slots
        if (has_slot_limit && slots_sent.load() >= slot_limit) {
            if (running.load()) {
                RT_LOGC_INFO(
                        ran::fapi::FapiComponent::FapiSample,
                        "Sent {} slots (limit reached), exiting slot loop",
                        slots_sent.load());
                running.store(false);
            }
            return;
        }

        // Check if all cells stopped (MAC disconnected or stopped cells)
        if (fapi_state.get_num_cells_running() == 0) {
            if (running.load()) {
                RT_LOGC_INFO(
                        ran::fapi::FapiComponent::FapiSample,
                        "All cells stopped, exiting slot loop");
                running.store(false);
            }
            return;
        }

        // Send SLOT.indication to first running cell
        if (fapi_state.send_slot_indication()) {
            slots_sent++;
        }

        // Increment slot counter
        fapi_state.increment_slot();
    };
}

std::vector<MessageCounts>
snapshot_counters(const std::vector<MessageCounters> &cell_counters, const std::size_t num_cells) {
    std::vector<MessageCounts> counts;
    counts.reserve(num_cells);
    for (std::size_t i{}; i < num_cells; ++i) {
        counts.emplace_back(
                cell_counters[i].ul_tti_request_count.load(std::memory_order_relaxed),
                cell_counters[i].dl_tti_request_count.load(std::memory_order_relaxed),
                cell_counters[i].slot_response_count.load(std::memory_order_relaxed));
    }
    return counts;
}

} // namespace fapi_sample

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
