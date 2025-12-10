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
 * @file task_sample.cpp
 * @brief Task framework demonstration application
 */

#include <atomic>  // for atomic
#include <chrono>  // for chrono::milliseconds, chrono::microseconds
#include <cstdlib> // for EXIT_SUCCESS, EXIT_FAILURE
#include <exception>
#include <format>   // for format
#include <iostream> // for cerr
#include <ratio>
#include <string> // for string
#include <system_error>
#include <thread> // for this_thread::sleep_for

#include <tl/expected.hpp> // for expected, unexpected

#include <CLI/CLI.hpp> // for App, Range, ParseError

#include "internal_use_only/config.hpp" // for project_name, project_version
#include "log/components.hpp"           // for register_component
#include "log/rt_log.hpp"               // for Logger, LogLevel
#include "log/rt_log_macros.hpp"        // for RT_LOG_INFO, RT_LOG_ERROR
#include "task/task_graph.hpp"          // for TaskGraph
#include "task/task_log.hpp"            // for TaskLog
#include "task/task_scheduler.hpp"      // for TaskScheduler
#include "task/timed_trigger.hpp"       // for TimedTrigger

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

namespace {
namespace ft = framework::task;

struct AppConfig {
    int interval_ms{10};
    std::size_t count{100};
    std::size_t workers{1};
};

/**
 * Setup logging
 */
void setup_logging() {

    framework::log::Logger::set_level(framework::log::LogLevel::Debug);
    framework::log::register_component<ft::TaskLog>(framework::log::LogLevel::Debug);
}

/**
 * Parse command line arguments
 *
 * @param[in] argc Number of command line arguments
 * @param[in] argv Array of command line argument strings
 * @return Parsed configuration on success, empty string if --help or --version shown, error message
 * on failure
 */
tl::expected<AppConfig, std::string> parse_arguments(const int argc, const char **argv) {
    CLI::App app{std::format(
            "Task Framework Demo - {} version {}",
            framework::cmake::project_name,
            framework::cmake::project_version)};

    AppConfig config{};
    app.add_option("-i,--interval_ms", config.interval_ms, "Task trigger interval in milliseconds")
            ->check(CLI::Range(1, 10000));
    app.add_option("-c,--count", config.count, "Number of task triggers to execute")
            ->check(CLI::Range(1, 100000));
    app.add_option("-w,--workers", config.workers, "Number of worker threads")
            ->check(CLI::Range(1, 128));

    app.set_version_flag(
            "--version",
            std::string{framework::cmake::project_version},
            "Show version information");

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
        const int exit_code = app.exit(e);
        if (exit_code == 0) {
            // Success codes (--help or --version) - return empty error string
            return tl::unexpected("");
        }
        // Actual error
        return tl::unexpected(std::format("Argument parsing failed: {}", e.what()));
    }

    RT_LOG_INFO(
            "Task Demo: interval={}ms, count={}, workers={}",
            config.interval_ms,
            config.count,
            config.workers);

    return config;
}

} // namespace

/**
 * Main application entry point
 *
 * Demonstrates the task framework: creates a task scheduler, defines a simple
 * task, schedules it with a timed trigger, and reports statistics.
 *
 * @param[in] argc Number of command line arguments
 * @param[in] argv Array of command line argument strings
 * @return EXIT_SUCCESS on successful completion, EXIT_FAILURE on error
 */
int main(int argc, const char **argv) {
    try {
        // example-begin task-sample-main-1
        using namespace framework::task;
        using namespace std::chrono_literals;

        setup_logging();

        const auto config = parse_arguments(argc, argv);
        if (!config.has_value()) {
            // Empty error string means --help or --version was shown (success)
            // Non-empty error string means parse error (failure)
            if (!config.error().empty()) {
                RT_LOG_ERROR("{}", config.error());
                return EXIT_FAILURE;
            }
            return EXIT_SUCCESS;
        }

        // Create task scheduler with configured worker threads
        auto scheduler = TaskScheduler::create().workers(config->workers).build();

        // Define a sample task that increments a counter
        std::atomic<int> task_counter{0};
        auto graph = TaskGraph::create("demo_graph")
                             .single_task("sample_task")
                             .function([&task_counter]() {
                                 task_counter++;
                                 std::this_thread::sleep_for(100us);
                             })
                             .build();

        // Schedule task at regular intervals
        auto trigger_function = [&scheduler, &graph]() { scheduler.schedule(graph); };
        const auto interval = std::chrono::milliseconds{config->interval_ms};
        auto trigger = TimedTrigger::create(trigger_function, interval)
                               .max_triggers(config->count)
                               .build();

        // Run and wait for completion
        if (const auto start_result = trigger.start(); start_result) {
            RT_LOG_ERROR("Failed to start trigger: {}", start_result.message());
            return EXIT_FAILURE;
        }
        trigger.wait_for_completion();

        // Cleanup and report
        scheduler.join_workers();
        RT_LOG_INFO("Complete: {} tasks executed", task_counter.load());
        trigger.print_summary();
        scheduler.print_monitor_stats();

        return EXIT_SUCCESS;
        // example-end task-sample-main-1
    } catch (const std::exception &e) {
        std::cerr << std::format("Unhandled exception: {}\n", e.what());
        return EXIT_FAILURE;
    } catch (...) {
        std::cerr << "Unknown exception occurred\n";
        return EXIT_FAILURE;
    }
}

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
