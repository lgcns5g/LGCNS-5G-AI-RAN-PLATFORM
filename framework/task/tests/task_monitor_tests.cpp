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
 * @file task_monitor_tests.cpp
 * @brief Unit tests for TaskMonitor class
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <ratio>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <gsl-lite/gsl-lite.hpp>
#include <gtest/gtest.h>

#include "task/task.hpp"
#include "task/task_category.hpp"
#include "task/task_errors.hpp"
#include "task/task_graph.hpp"
#include "task/task_monitor.hpp"
#include "task/task_scheduler.hpp"
#include "task/time.hpp"

namespace {
namespace ft = framework::task;

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

using namespace std::chrono_literals;

/**
 * Validate Chrome trace format for TaskMonitor events
 * @param[in] content Chrome trace file content
 * @param[in] expected_events Expected number of task execution events
 * @param[in] task_names Expected task names to find in trace
 * @param[in] graph_name Expected graph name (used as category)
 */
void validate_task_chrome_trace(
        const std::string &content,
        std::size_t expected_events,
        const std::vector<std::string> &task_names,
        const std::string &graph_name) {
    // Verify Chrome trace format structure
    EXPECT_NE(content.find(R"("traceEvents":[)"), std::string::npos);
    EXPECT_NE(content.find(R"("ph":"X")"), std::string::npos); // Duration event
    EXPECT_NE(content.find(R"("tid":)"), std::string::npos);   // Worker thread ID
    EXPECT_NE(content.find(R"("ts":)"), std::string::npos);    // Timestamp
    EXPECT_NE(content.find(R"("dur":)"), std::string::npos);   // Duration

    // Verify expected task names are present
    for (const auto &task_name : task_names) {
        EXPECT_NE(content.find(R"("name":")" + task_name + R"(")"), std::string::npos);
    }

    // Verify graph name as category
    EXPECT_NE(content.find(R"("cat":")" + graph_name + R"(")"), std::string::npos);

    // Count Chrome trace events (should match expected)
    std::size_t event_count = 0;
    std::size_t pos = 0;
    while ((pos = content.find(R"("ph":"X")", pos)) != std::string::npos) {
        event_count++;
        pos += 1;
    }
    EXPECT_EQ(event_count, expected_events);
}

TEST(TaskMonitor, Construction) { EXPECT_NO_THROW(const ft::TaskMonitor monitor{}); }

TEST(TaskMonitor, StartConfigurationMatrix) {
    struct StartTestCase {
        std::string name;
        std::optional<std::uint32_t> core_id;
        int expected_result; // 0 = success, != 0 = expect failure
        std::string description;
    };

    const std::vector<StartTestCase> test_cases = {
            {"Basic", std::nullopt, 0, "No core pinning, no RT priority - should always succeed"},
            {"CorePinning", std::optional<std::uint32_t>{0}, 0, "Core 0 pinning only"},
            {"InvalidCore",
             std::optional<std::uint32_t>{999999},
             -1,
             "Invalid high core ID - should fail due to invalid core"}};

    for (const auto &test_case : test_cases) {
        SCOPED_TRACE("Testing: " + test_case.name + " - " + test_case.description);

        ft::TaskMonitor monitor{};
        const std::error_code result = monitor.start(test_case.core_id);

        if (test_case.expected_result == 0) {
            EXPECT_TRUE(ft::is_task_success(result)) << "Expected success for " << test_case.name;
        } else {
            EXPECT_TRUE(result) << "Expected failure for " << test_case.name;
        }

        if (!result) {
            std::this_thread::sleep_for(500us);
            monitor.stop();
        }
    }
}

TEST(TaskMonitor, MultipleStartStopCycles) {
    ft::TaskMonitor monitor{};

    // Test multiple start/stop cycles
    for (int cycle = 0; cycle < 3; ++cycle) {
        const std::error_code result = monitor.start(std::nullopt);
        EXPECT_TRUE(ft::is_task_success(result));

        // Register and process a task in each cycle
        auto func = []() { return ft::TaskResult{}; };
        auto shared_task =
                ft::TaskBuilder(std::format("cycle_task_{}", cycle)).function(func).build_shared();
        EXPECT_TRUE(ft::is_task_success(monitor.register_task(ft::TaskHandle(shared_task))));

        std::this_thread::sleep_for(1ms);
        monitor.stop();

        // Small delay between cycles
        std::this_thread::sleep_for(1ms);
    }
}

TEST(TaskMonitor, StartStopWithoutTasks) {
    ft::TaskMonitor monitor{};

    // Test start/stop without registering any tasks
    const std::error_code result = monitor.start(std::nullopt);
    EXPECT_TRUE(ft::is_task_success(result));

    // Let it run idle for a bit
    std::this_thread::sleep_for(1ms);

    EXPECT_NO_THROW(monitor.stop());
}

TEST(TaskMonitor, TaskTimeoutScenarios) {
    ft::TaskMonitor monitor{};
    const std::error_code result = monitor.start(std::nullopt);
    EXPECT_TRUE(ft::is_task_success(result));

    const ft::Nanos base_time = ft::Time::now_ns();

    // Task with very short timeout (should timeout)
    auto func = []() { return ft::TaskResult{}; };
    auto short_timeout_task = ft::TaskBuilder("short_timeout")
                                      .function(func)
                                      .timeout(3ms)
                                      .scheduled_time(base_time)
                                      .build_shared();

    // Task with reasonable timeout
    auto normal_timeout_task = ft::TaskBuilder("normal_timeout")
                                       .function(func)
                                       .timeout(10ms)
                                       .scheduled_time(base_time)
                                       .build_shared();

    // Task with no timeout (0 = disabled)
    auto no_timeout_task = ft::TaskBuilder("no_timeout")
                                   .function(func)
                                   .timeout(0ns)
                                   .scheduled_time(base_time)
                                   .build_shared();

    EXPECT_TRUE(ft::is_task_success(monitor.register_task(ft::TaskHandle(short_timeout_task))));
    EXPECT_TRUE(ft::is_task_success(monitor.register_task(ft::TaskHandle(normal_timeout_task))));
    EXPECT_TRUE(ft::is_task_success(monitor.register_task(ft::TaskHandle(no_timeout_task))));

    // Start the tasks so timeout logic can trigger
    EXPECT_TRUE(ft::is_task_success(
            monitor.record_start(short_timeout_task->get_task_id(), 1, base_time)));
    EXPECT_TRUE(ft::is_task_success(
            monitor.record_start(normal_timeout_task->get_task_id(), 1, base_time)));
    EXPECT_TRUE(ft::is_task_success(
            monitor.record_start(no_timeout_task->get_task_id(), 1, base_time)));

    // Let timeout checking run - should timeout the short task
    std::this_thread::sleep_for(50ms);

    monitor.stop();

    // Verify timeout behavior
    EXPECT_TRUE(short_timeout_task->is_cancelled())
            << "Short timeout task should be cancelled due to 3ms timeout";
    EXPECT_FALSE(no_timeout_task->is_cancelled())
            << "No timeout task should not be cancelled (timeout disabled)";
}

TEST(TaskMonitor, BoundaryValueTests) {
    ft::TaskMonitor monitor{};
    const std::error_code result = monitor.start(std::nullopt);
    EXPECT_TRUE(ft::is_task_success(result));

    struct BoundaryTestCase {
        std::string category;
        std::string name;
        std::chrono::nanoseconds scheduled_time;
        std::chrono::nanoseconds start_time;
        std::chrono::nanoseconds end_time;
        std::chrono::milliseconds timeout;
        ft::WorkerId worker;
        std::string description;
    };

    const std::chrono::nanoseconds max_time{std::numeric_limits<int64_t>::max()};
    const ft::WorkerId max_worker = std::numeric_limits<ft::WorkerId>::max();

    const std::vector<BoundaryTestCase> test_cases = {
            // Time boundary cases
            {"Time", "zero_time", 0ns, 0ns, 0ns, 0ms, 1, "All zero timestamps"},
            {"Time", "max_time", max_time, max_time, max_time, 1ms, 1, "Maximum timestamp values"},
            {"Time",
             "negative_duration",
             2ms,
             2ms,
             1ms,
             0ms,
             1,
             "End before start (negative duration)"},

            // Worker ID boundary cases
            {"Worker", "worker_min", 0ns, 0ns, 1us, 0ms, 0, "Minimum worker ID"},
            {"Worker", "worker_max", 0ns, 0ns, 1us, 0ms, max_worker, "Maximum worker ID"},
            {"Worker", "worker_pow2_1", 0ns, 0ns, 1us, 0ms, 1, "Power of 2 worker ID: 1"},
            {"Worker", "worker_pow2_128", 0ns, 0ns, 1us, 0ms, 128, "Power of 2 worker ID: 128"},
    };

    for (const auto &test_case : test_cases) {
        SCOPED_TRACE("Testing " + test_case.category + ": " + test_case.description);

        auto func = []() { return ft::TaskResult{}; };
        auto shared_task = ft::TaskBuilder(test_case.name)
                                   .function(func)
                                   .timeout(test_case.timeout)
                                   .scheduled_time(test_case.scheduled_time)
                                   .build_shared();

        EXPECT_TRUE(ft::is_task_success(monitor.register_task(ft::TaskHandle(shared_task))));
        const auto task_id = shared_task->get_task_id();
        EXPECT_TRUE(ft::is_task_success(
                monitor.record_start(task_id, test_case.worker, test_case.start_time)));
        EXPECT_TRUE(ft::is_task_success(
                monitor.record_end(task_id, test_case.end_time, ft::TaskStatus::Completed)));
    }

    std::this_thread::sleep_for(1ms);
    monitor.stop();
}

TEST(TaskMonitor, TaskNameVariations) {
    ft::TaskMonitor monitor{};
    const std::error_code result = monitor.start(std::nullopt);
    EXPECT_TRUE(ft::is_task_success(result));

    struct NameTestCase {
        std::string name;
        std::string description;
    };

    const std::vector<NameTestCase> test_cases = {
            {"task with spaces", "Spaces"},
            {"task-with-hyphens", "Hyphens"},
            {"task_with_underscores", "Underscores"},
            {"task.with.dots", "Dots"},
            {"task/with/slashes", "Forward slashes"},
            {"task\\with\\backslashes", "Backslashes"},
            {"task\"with\"quotes", "Double quotes"},
            {"task'with'apostrophes", "Single quotes"},
            {"task@with@symbols#$%", "Special symbols"},
            {"task\twith\ttabs", "Tab characters"},
            {"task\nwith\nnewlines", "Newline characters"},
            {"duplicate", "Duplicate name (first)"},
            {"duplicate", "Duplicate name (second)"} // Tests overwrite behavior
    };

    for (const auto &test_case : test_cases) {
        SCOPED_TRACE("Testing name: " + test_case.description);

        auto func = []() { return ft::TaskResult{}; };
        auto shared_task = ft::TaskBuilder(test_case.name).function(func).build_shared();

        EXPECT_TRUE(ft::is_task_success(monitor.register_task(ft::TaskHandle(shared_task))));
        const auto task_id = shared_task->get_task_id();
        EXPECT_TRUE(ft::is_task_success(monitor.record_start(task_id, 1, ft::Nanos{0})));
        EXPECT_TRUE(ft::is_task_success(
                monitor.record_end(task_id, ft::Nanos{1000}, ft::TaskStatus::Completed)));
        EXPECT_TRUE(ft::is_task_success(monitor.cancel_task(task_id)));
    }

    std::this_thread::sleep_for(5ms);
    monitor.stop();
}

TEST(TaskBuilder, ValidationErrors) {
    // Test that TaskBuilder throws on empty name
    EXPECT_THROW(
            { ft::TaskBuilder("").function([]() { return ft::TaskResult{}; }).build(); },
            std::invalid_argument);

    // Test that TaskBuilder with no function still builds (empty function handled
    // by Task::execute)
    EXPECT_NO_THROW({
        ft::TaskBuilder("ValidName").build(); // No function set - should build successfully
    });
}

TEST(TaskMonitor, CancellationRequestProcessing) {
    ft::TaskMonitor monitor{};
    const std::error_code result = monitor.start(std::nullopt);
    EXPECT_TRUE(ft::is_task_success(result));

    // Create and register tasks
    std::vector<std::shared_ptr<ft::Task>> tasks;
    for (int i = 0; i < 5; ++i) {
        auto func = []() { return ft::TaskResult{}; };
        auto shared_task =
                ft::TaskBuilder(std::format("cancel_test_{}", i)).function(func).build_shared();
        tasks.emplace_back(shared_task);

        EXPECT_TRUE(ft::is_task_success(monitor.register_task(ft::TaskHandle(shared_task))));
    }

    // Test that cancellation requests are processed without errors
    EXPECT_TRUE(ft::is_task_success(monitor.cancel_task(tasks.at(0)->get_task_id())));
    EXPECT_TRUE(ft::is_task_success(monitor.cancel_task(tasks.at(2)->get_task_id())));
    EXPECT_TRUE(ft::is_task_success(monitor.cancel_task(tasks.at(4)->get_task_id())));

    // Test cancelling non-existent task (should not error)
    static constexpr auto INVALID_TASK_ID = std::numeric_limits<std::uint64_t>::max();
    EXPECT_TRUE(ft::is_task_success(monitor.cancel_task(INVALID_TASK_ID)));

    // Test cancelling same task multiple times (should not error)
    const auto task_0_id = tasks.at(0)->get_task_id();
    EXPECT_TRUE(ft::is_task_success(monitor.cancel_task(task_0_id)));
    EXPECT_TRUE(ft::is_task_success(monitor.cancel_task(task_0_id)));

    // Give monitor time to process cancellations
    std::this_thread::sleep_for(10ms);

    monitor.stop();
}

TEST(TaskMonitor, TaskRegistrationEdgeCases) {
    ft::TaskMonitor monitor{};
    const std::error_code result = monitor.start(std::nullopt);
    EXPECT_TRUE(ft::is_task_success(result));

    auto func = []() { return ft::TaskResult{}; };

    // Test very long task name
    auto long_name_task = ft::TaskBuilder(std::string(1000, 'X')).function(func).build_shared();
    EXPECT_TRUE(ft::is_task_success(monitor.register_task(ft::TaskHandle(long_name_task))));

    // Test task with same name (should overwrite)
    auto duplicate_task = ft::TaskBuilder("duplicate")
                                  .function(func)
                                  .timeout(200ns)
                                  .scheduled_time(100ns)
                                  .build_shared();
    EXPECT_TRUE(ft::is_task_success(monitor.register_task(ft::TaskHandle(duplicate_task))));
    EXPECT_TRUE(ft::is_task_success(
            monitor.register_task(ft::TaskHandle(duplicate_task)))); // Register again

    monitor.stop();
}

TEST(TaskMonitor, CancellationEdgeCases) {
    ft::TaskMonitor monitor{};
    const std::error_code result = monitor.start(std::nullopt);
    EXPECT_TRUE(ft::is_task_success(result));

    // Cancel non-existent task
    static constexpr auto NON_EXISTENT_TASK_ID = std::numeric_limits<std::uint64_t>::max();
    EXPECT_TRUE(ft::is_task_success(monitor.cancel_task(NON_EXISTENT_TASK_ID)));

    // Cancel task multiple times
    auto func = []() { return ft::TaskResult{}; };
    auto shared_task = ft::TaskBuilder("cancelme").function(func).build_shared();
    EXPECT_TRUE(ft::is_task_success(monitor.register_task(ft::TaskHandle(shared_task))));

    const auto task_id = shared_task->get_task_id();
    EXPECT_TRUE(ft::is_task_success(monitor.cancel_task(task_id)));
    EXPECT_TRUE(ft::is_task_success(monitor.cancel_task(task_id))); // Cancel again

    std::this_thread::sleep_for(1ms);
    monitor.stop();
}

TEST(TaskMonitor, RecordExecutionEdgeCases) {
    ft::TaskMonitor monitor{};
    const std::error_code result = monitor.start(std::nullopt);
    EXPECT_TRUE(ft::is_task_success(result));

    const ft::Nanos now = ft::Time::now_ns();

    // Record start for non-existent task
    static constexpr auto NON_EXISTENT_TASK_ID = std::numeric_limits<std::uint64_t>::max();
    EXPECT_TRUE(ft::is_task_success(monitor.record_start(NON_EXISTENT_TASK_ID, 1, now)));

    // Record end for non-existent task
    EXPECT_TRUE(ft::is_task_success(
            monitor.record_end(NON_EXISTENT_TASK_ID, now, ft::TaskStatus::Completed)));

    // Record execution with zero worker ID
    EXPECT_TRUE(ft::is_task_success(monitor.record_start(NON_EXISTENT_TASK_ID, 0, now)));

    // Record execution with very high worker ID
    EXPECT_TRUE(ft::is_task_success(monitor.record_start(
            NON_EXISTENT_TASK_ID, std::numeric_limits<ft::WorkerId>::max(), now)));

    std::this_thread::sleep_for(1ms);
    monitor.stop();
}

TEST(TaskMonitor, CompleteTaskWorkflow) {
    ft::TaskMonitor monitor{};
    const std::error_code result = monitor.start(std::nullopt);
    EXPECT_TRUE(ft::is_task_success(result));

    const ft::Nanos start_time = ft::Time::now_ns();
    const ft::Nanos end_time = start_time + 2ms; // 2ms later

    // Complete workflow: Register -> Start -> End
    auto func = []() { return ft::TaskResult{}; };
    auto shared_task = ft::TaskBuilder("workflow_task")
                               .function(func)
                               .category(ft::BuiltinTaskCategory::HighPriority)
                               .timeout(5ms)
                               .scheduled_time(start_time)
                               .build_shared();

    EXPECT_TRUE(ft::is_task_success(monitor.register_task(ft::TaskHandle(shared_task))));
    const auto task_id = shared_task->get_task_id();
    EXPECT_TRUE(ft::is_task_success(monitor.record_start(task_id, 42, start_time)));
    EXPECT_TRUE(
            ft::is_task_success(monitor.record_end(task_id, end_time, ft::TaskStatus::Completed)));

    std::this_thread::sleep_for(5ms);
    monitor.stop();
}

TEST(TaskMonitor, TaskPropertiesAndConfigMatrix) {
    struct TaskPropertyTest {
        std::string name;
        ft::TaskCategory category;
        ft::TaskStatus final_status;
        std::chrono::microseconds sleep_duration;
        std::string description;
    };

    const std::vector<TaskPropertyTest> test_cases = {
            // Task categories and statuses with different configurations
            {"default_completed",
             ft::TaskCategory{ft::BuiltinTaskCategory::Default},
             ft::TaskStatus::Completed,
             std::chrono::microseconds{10},
             "Default category, completed, no graph"},
            {"high_priority_failed",
             ft::TaskCategory{ft::BuiltinTaskCategory::HighPriority},
             ft::TaskStatus::Failed,
             std::chrono::microseconds{1},
             "High priority, failed, fast sleep"},
            {"io_cancelled",
             ft::TaskCategory{ft::BuiltinTaskCategory::IO},
             ft::TaskStatus::Cancelled,
             std::chrono::microseconds{50},
             "IO category, cancelled, slow sleep"},
            {"compute_completed",
             ft::TaskCategory{ft::BuiltinTaskCategory::Compute},
             ft::TaskStatus::Completed,
             std::chrono::microseconds{10},
             "Compute category, completed, tmp graph"},
            {"network_failed",
             ft::TaskCategory{ft::BuiltinTaskCategory::Network},
             ft::TaskStatus::Failed,
             std::chrono::microseconds{100},
             "Network category, failed, simple graph"},
            {"low_priority_completed",
             ft::TaskCategory{ft::BuiltinTaskCategory::LowPriority},
             ft::TaskStatus::Completed,
             std::chrono::microseconds{10},
             "Low priority, completed, long filename"}};

    for (const auto &test_case : test_cases) {
        SCOPED_TRACE("Testing: " + test_case.description);

        // Test graph disabled state first (if no graph file)
        ft::TaskMonitor monitor{};

        const std::error_code result = monitor.start(std::nullopt, test_case.sleep_duration);
        EXPECT_TRUE(ft::is_task_success(result)) << "Monitor should start successfully";

        const ft::Nanos base_time = ft::Time::now_ns();

        auto func = []() { return ft::TaskResult{}; };
        auto shared_task = ft::TaskBuilder(test_case.name)
                                   .function(func)
                                   .category(test_case.category)
                                   .scheduled_time(base_time)
                                   .build_shared();

        EXPECT_TRUE(ft::is_task_success(monitor.register_task(ft::TaskHandle(shared_task))));
        const auto task_id = shared_task->get_task_id();
        EXPECT_TRUE(ft::is_task_success(monitor.record_start(task_id, 1, base_time)));
        EXPECT_TRUE(ft::is_task_success(
                monitor.record_end(task_id, base_time + 500us, test_case.final_status)));

        // Give it time to process based on sleep duration
        const auto process_time = std::max(
                std::chrono::milliseconds{2},
                std::chrono::duration_cast<std::chrono::milliseconds>(
                        test_case.sleep_duration * 3));
        std::this_thread::sleep_for(process_time);

        monitor.stop();
    }
}

TEST(TaskMonitor, ComplexDependencies) {
    ft::TaskMonitor monitor{};
    const std::error_code result = monitor.start(std::nullopt);
    EXPECT_TRUE(ft::is_task_success(result));

    // Create dependency chain:
    // clang-format off
  // A -> B -> C -> D
    // clang-format on
    const std::vector<std::pair<std::string, std::vector<std::string>>> tasks = {
            {"task_A", {}},                            // No dependencies
            {"task_B", {"task_A"}},                    // Depends on A
            {"task_C", {"task_B"}},                    // Depends on B
            {"task_D", {"task_C"}},                    // Depends on C
            {"task_E", {"task_A", "task_B", "task_C"}} // Depends on multiple tasks
    };

    for (const auto &[task_name, deps] : tasks) {
        auto func = []() { return ft::TaskResult{}; };
        auto shared_task = ft::TaskBuilder(task_name).function(func).build_shared();

        EXPECT_TRUE(ft::is_task_success(monitor.register_task(ft::TaskHandle(shared_task))));
    }

    std::this_thread::sleep_for(1ms);

    monitor.stop();
}

TEST(TaskMonitor, EmptyAndSelfDependencies) {
    ft::TaskMonitor monitor{};
    const std::error_code result = monitor.start(std::nullopt);
    EXPECT_TRUE(ft::is_task_success(result));

    // Task with empty dependencies
    auto func = []() { return ft::TaskResult{}; };
    auto empty_deps = ft::TaskBuilder("empty_deps_task").function(func).build_shared();
    EXPECT_TRUE(ft::is_task_success(monitor.register_task(ft::TaskHandle(empty_deps))));

    // Task with self-dependency (should be handled gracefully)
    auto self_dep = ft::TaskBuilder("self_dep_task").function(func).build_shared();
    EXPECT_TRUE(ft::is_task_success(monitor.register_task(ft::TaskHandle(self_dep))));

    // Task with non-existent dependencies
    auto missing_deps = ft::TaskBuilder("missing_deps_task").function(func).build_shared();
    EXPECT_TRUE(ft::is_task_success(monitor.register_task(ft::TaskHandle(missing_deps))));

    std::this_thread::sleep_for(1ms);
    monitor.stop();
}

TEST(TaskMonitor, ManyTasks) {
    ft::TaskMonitor monitor{};
    const std::error_code result = monitor.start(std::nullopt);
    EXPECT_TRUE(ft::is_task_success(result));

    constexpr int NUM_TASKS = 100;
    const ft::Nanos base_time = ft::Time::now_ns();

    // Register many tasks quickly
    auto func = []() { return ft::TaskResult{}; };
    for (int i = 0; i < NUM_TASKS; ++i) {
        auto category =
                (i % 2 == 0) ? ft::BuiltinTaskCategory::Compute : ft::BuiltinTaskCategory::IO;
        auto shared_task = ft::TaskBuilder(std::format("stress_task_{}", i))
                                   .function(func)
                                   .category(category)
                                   .timeout(1ms)
                                   .scheduled_time(base_time + ft::Nanos{i * 1000})
                                   .build_shared();

        EXPECT_TRUE(ft::is_task_success(monitor.register_task(ft::TaskHandle(shared_task))));
    }

    // Let the monitor process events
    std::this_thread::sleep_for(10ms);
    monitor.stop();
}

TEST(TaskMonitor, ConcurrentOperations) {
    ft::TaskMonitor monitor{};
    const std::error_code result = monitor.start(std::nullopt);
    EXPECT_TRUE(ft::is_task_success(result));

    const ft::Nanos base_time = ft::Time::now_ns();

    // Simulate concurrent task registration, execution, and cancellation
    std::vector<std::string> task_names;
    task_names.reserve(20);
    for (int i = 0; i < 20; ++i) {
        task_names.push_back(std::format("concurrent_task_{}", i));
    }

    // Register tasks
    std::vector<std::shared_ptr<ft::Task>> tasks;
    for (const auto &name : task_names) {
        auto func = []() { return ft::TaskResult{}; };
        auto shared_task = ft::TaskBuilder(name)
                                   .function(func)
                                   .timeout(50ms)
                                   .scheduled_time(base_time)
                                   .build_shared();
        tasks.emplace_back(shared_task);
        EXPECT_TRUE(ft::is_task_success(monitor.register_task(ft::TaskHandle(shared_task))));
    }

    // Simulate execution records
    for (size_t i = 0; i < task_names.size(); ++i) {
        const auto task_id = tasks.at(i)->get_task_id();
        EXPECT_TRUE(ft::is_task_success(
                monitor.record_start(task_id, static_cast<ft::WorkerId>(i % 4), base_time)));
        if (i % 3 == 0) {
            // Cancel some tasks
            EXPECT_TRUE(ft::is_task_success(monitor.cancel_task(task_id)));
        } else {
            // Complete others
            EXPECT_TRUE(ft::is_task_success(monitor.record_end(
                    task_id,
                    base_time + 10us,
                    (i % 4 == 0) ? ft::TaskStatus::Failed : ft::TaskStatus::Completed)));
        }
    }

    std::this_thread::sleep_for(10ms);
    monitor.stop();
}

TEST(TaskBuilder, BuildShared) {
    static const std::string task_name = "test_task";

    // Test build_shared() creates valid shared_ptr
    auto shared_task = ft::TaskBuilder(task_name)
                               .function([]() { return ft::TaskResult{}; })
                               .timeout(1ms)
                               .build_shared();

    ASSERT_NE(shared_task, nullptr);
    EXPECT_EQ(shared_task->get_task_name(), task_name);
    EXPECT_EQ(shared_task->get_timeout_ns(), 1ms);

    // Test that we can create a TaskHandle from it
    const ft::TaskHandle handle(shared_task);
    EXPECT_EQ(handle->get_task_name(), task_name);
    EXPECT_EQ(handle->get_task_id(), shared_task->get_task_id());

    // Verify reference counting works
    EXPECT_EQ(shared_task.use_count(), 2); // shared_task + handle's internal copy
}

TEST(TaskMonitor, WriteStatsToFile) {
    // Test writing empty stats should fail gracefully
    const ft::TaskMonitor empty_monitor{};
    const std::string empty_file = "test_empty_stats.json";
    std::error_code result = empty_monitor.write_stats_to_file(empty_file);
    EXPECT_TRUE(result); // Should fail with no execution records

    // Test with actual TaskScheduler and real task execution
    const std::string stats_file = "test_simple_stats.json";

    // Use gsl_lite::finally to ensure cleanup
    auto cleanup = gsl_lite::finally([&empty_file, &stats_file]() {
        std::filesystem::remove(empty_file);
        std::filesystem::remove(stats_file);
    });

    // Create TaskScheduler with minimal configuration
    auto task_scheduler = ft::TaskScheduler::create()
                                  .workers(2) // Just 2 workers
                                  .auto_start()
                                  .build();

    // Create task graph more efficiently
    ft::TaskGraph graph("name");
    graph.register_task("simple_task").function([]() { std::this_thread::sleep_for(10us); }).add();

    graph.register_task("another_task").function([]() { std::this_thread::sleep_for(20us); }).add();

    // Build graph once for efficiency
    graph.build();

    // Schedule for 3 executions
    const std::array iterations{0, 1, 2};
    for ([[maybe_unused]] const auto _ : iterations) {
        task_scheduler.schedule(graph);
    }

    task_scheduler.join_workers();

    // Export stats to file
    result = task_scheduler.write_monitor_stats_to_file(stats_file);
    EXPECT_TRUE(ft::is_task_success(result)); // Should succeed

    // Verify file was created and contains data
    EXPECT_TRUE(std::filesystem::exists(stats_file));

    std::ifstream file(stats_file);
    ASSERT_TRUE(file.is_open());

    std::string line;
    std::vector<std::string> lines;
    while (std::getline(file, line)) {
        lines.push_back(line);
    }
    file.close();

    // Should have version header plus execution records
    EXPECT_GE(lines.size(), 2);

    // First line should be version header
    EXPECT_TRUE(lines[0].find(R"("version":"1.0")") != std::string::npos);

    // Count task occurrences in JSON and verify format
    int simple_task_count = 0;
    int another_task_count = 0;
    int graph_name_count = 0;

    for (size_t i = 1; i < lines.size(); ++i) {
        // Check for updated JSON keys
        if (lines[i].find(R"("task_name":"simple_task")") != std::string::npos) {
            simple_task_count++;
        }
        if (lines[i].find(R"("task_name":"another_task")") != std::string::npos) {
            another_task_count++;
        }
        if (lines[i].find(R"("graph_name":"name")") != std::string::npos) {
            graph_name_count++;
        }

        // Verify all required fields are present
        EXPECT_TRUE(lines[i].find(R"("task_name":)") != std::string::npos)
                << "Line " << i << " missing task_name field: " << lines[i];
        EXPECT_TRUE(lines[i].find(R"("graph_name":)") != std::string::npos)
                << "Line " << i << " missing graph_name field: " << lines[i];
        EXPECT_TRUE(lines[i].find(R"("dependency_generation":)") != std::string::npos)
                << "Line " << i << " missing dependency_generation field";
        EXPECT_TRUE(lines[i].find(R"("times_scheduled":)") != std::string::npos)
                << "Line " << i << " missing times_scheduled field";
        EXPECT_TRUE(lines[i].find(R"("worker":)") != std::string::npos)
                << "Line " << i << " missing worker field";
        EXPECT_TRUE(lines[i].find(R"("duration_ns":)") != std::string::npos)
                << "Line " << i << " missing duration_ns field";
        EXPECT_TRUE(lines[i].find(R"("status":)") != std::string::npos)
                << "Line " << i << " missing status field";
        EXPECT_TRUE(lines[i].find(R"("was_cancelled":)") != std::string::npos)
                << "Line " << i << " missing was_cancelled field";
    }

    // Should have executed each task the expected number of times
    EXPECT_EQ(simple_task_count, iterations.size());
    EXPECT_EQ(another_task_count, iterations.size());
    // Should have graph name for all task records (2 tasks * 3 iterations = 6
    // total)
    EXPECT_EQ(graph_name_count, 2 * iterations.size());

    // Also test Chrome trace format
    const std::string chrome_filename = "test_monitor_chrome_trace.json";
    std::filesystem::remove(chrome_filename);

    EXPECT_TRUE(ft::is_task_success(task_scheduler.write_chrome_trace_to_file(chrome_filename)));
    EXPECT_TRUE(std::filesystem::exists(chrome_filename));

    std::ifstream chrome_file(chrome_filename);
    ASSERT_TRUE(chrome_file.is_open());
    std::stringstream buffer;
    buffer << chrome_file.rdbuf();
    chrome_file.close();

    validate_task_chrome_trace(buffer.str(), 6, {"simple_task", "another_task"}, "name");
    std::filesystem::remove(chrome_filename);
}

TEST(TaskSchedulerBuilder, MaxExecutionRecordsConfiguration) {
    std::atomic<int> task_count{0};

    // Use a very small limit to force truncation
    auto task_scheduler = ft::TaskScheduler::create()
                                  .workers(2)
                                  .max_execution_records(5) // Very small limit to force truncation
                                  .manual_start()
                                  .build();

    EXPECT_NO_THROW(task_scheduler.start_workers());

    // Create task graph with a simple task
    ft::TaskGraph graph("truncation_test");
    graph.register_task("test_task")
            .function([&task_count]() {
                ++task_count;
                std::this_thread::sleep_for(100us);
            })
            .add();
    graph.build();

    // Schedule many executions to exceed limit and force truncation
    for (int i = 0; i < 20; ++i) {
        task_scheduler.schedule(graph);
    }

    // Wait for all tasks to complete
    task_scheduler.join_workers();

    // Should have executed many tasks
    EXPECT_GT(task_count.load(), 15);

    // Write stats to temporary file to check truncation
    const std::string temp_filename =
            (std::filesystem::temp_directory_path() / "test_taskscheduler_truncation.json")
                    .string();
    const std::error_code write_result = task_scheduler.write_monitor_stats_to_file(temp_filename);
    EXPECT_TRUE(ft::is_task_success(write_result));

    // Check that truncation warning is in the file
    std::ifstream file(temp_filename);
    ASSERT_TRUE(file.is_open());

    std::string line;
    bool found_truncation_warning = false;
    while (std::getline(file, line)) {
        if (line.find("execution_records_truncated") != std::string::npos) {
            found_truncation_warning = true;
            break;
        }
    }
    file.close();

    // Should have truncation warning due to tiny limit
    EXPECT_TRUE(found_truncation_warning);

    // Cleanup
    std::ignore = std::remove(temp_filename.c_str());
}

TEST(TaskSchedulerBuilder, AutoCalculateMaxRecords) {
    std::atomic<int> task_count{0};

    // Don't set max_execution_records - should auto-calculate to huge value
    auto task_scheduler = ft::TaskScheduler::create().workers(2).manual_start().build();

    EXPECT_NO_THROW(task_scheduler.start_workers());

    // Create task graph with a simple task
    ft::TaskGraph graph("auto_calc_test");
    graph.register_task("test_task")
            .function([&task_count]() {
                ++task_count;
                std::this_thread::sleep_for(50us);
            })
            .add();
    graph.build();

    // Schedule some tasks (not enough to trigger truncation with huge limit)
    for (int i = 0; i < 10; ++i) {
        task_scheduler.schedule(graph);
    }

    // Wait for tasks to complete
    task_scheduler.join_workers();

    // Should have executed all tasks
    EXPECT_EQ(task_count.load(), 10);

    // Write stats to temporary file to check no truncation
    const std::string temp_filename =
            (std::filesystem::temp_directory_path() / "test_taskscheduler_auto_calc.json").string();
    const std::error_code write_result = task_scheduler.write_monitor_stats_to_file(temp_filename);
    EXPECT_TRUE(ft::is_task_success(write_result));

    // Check that NO truncation warning is in the file
    std::ifstream file(temp_filename);
    ASSERT_TRUE(file.is_open());

    std::string content;
    std::string line;
    while (std::getline(file, line)) {
        content += line + "\n";
    }
    file.close();

    // Should NOT have truncation warning with huge auto-calculated limit
    EXPECT_TRUE(content.find("execution_records_truncated") == std::string::npos);

    // Cleanup
    std::ignore = std::remove(temp_filename.c_str());
}

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

} // namespace
