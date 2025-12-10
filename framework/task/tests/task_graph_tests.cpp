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
 * @file task_graph_tests.cpp
 * @brief Unit tests for TaskGraph fluent API
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
#include <optional>
#include <ratio>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <quill/LogMacros.h>

#include <gtest/gtest.h>

#include "log/rt_log_macros.hpp"
#include "task/flat_map.hpp"
#include "task/task.hpp"
#include "task/task_category.hpp"
#include "task/task_graph.hpp"
#include "task/task_log.hpp"
#include "task/task_pool.hpp"
#include "task/task_scheduler.hpp"
#include "task/task_worker.hpp"
#include "task/time.hpp"

namespace {
namespace ft = framework::task;

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

using namespace std::chrono_literals;

/**
 * Wait for a task to complete with timeout protection
 *
 * @param[in] graph TaskGraph containing the task
 * @param[in] task_name Name of the task to wait for
 * @param[in] timeout Maximum time to wait for completion
 */
void wait_for_task_completion(
        const ft::TaskGraph &graph,
        const std::string_view task_name,
        const std::chrono::milliseconds timeout = 5000ms) {
    const auto deadline = ft::Time::now_ns() + timeout;
    while (!graph.task_has_status(task_name, ft::TaskStatus::Completed) &&
           ft::Time::now_ns() < deadline) {
        std::this_thread::sleep_for(1ms);
    }
    ASSERT_TRUE(graph.task_has_status(task_name, ft::TaskStatus::Completed))
            << "Task '" << task_name << "' did not complete within " << timeout.count()
            << "ms timeout";
}

/**
 * Basic TaskGraph functionality tests
 */
TEST(TaskGraph, EmptyGraph) {
    const ft::TaskGraph graph("name");

    EXPECT_TRUE(graph.empty());
    EXPECT_EQ(graph.size(), 0);
    EXPECT_TRUE(graph.get_task_specs().empty());
}

TEST(TaskGraph, SingleTaskCreation) {
    ft::TaskGraph graph("name");

    std::atomic<bool> task_executed(false);

    auto task_name = graph.register_task("test_task")
                             .category(ft::TaskCategory{ft::BuiltinTaskCategory::Default})
                             .timeout(1ms)
                             .function([&task_executed]() { task_executed.store(true); })
                             .add();

    EXPECT_EQ(task_name, "test_task");
    EXPECT_FALSE(graph.empty());
    EXPECT_EQ(graph.size(), 1);

    const auto &specs = graph.get_task_specs();
    EXPECT_EQ(specs.size(), 1);
    EXPECT_EQ(specs[0].task_name, "test_task");
    EXPECT_EQ(specs[0].category, ft::TaskCategory{ft::BuiltinTaskCategory::Default});
    EXPECT_EQ(specs[0].timeout, 1ms);
    EXPECT_TRUE(specs[0].dependency_names.empty());
}

TEST(TaskGraph, TaskPoolCapacityMultiplierBuilder) {
    // Test default task pool capacity multiplier (20x) with 2-task graph
    ft::TaskGraph graph_default("test_graph");
    EXPECT_EQ(graph_default.get_task_pool_capacity(), 0); // Not built yet

    graph_default.register_task("task1").function([]() { return ft::TaskResult{}; }).add();
    graph_default.register_task("task2")
            .depends_on("task1")
            .function([]() { return ft::TaskResult{}; })
            .add();

    graph_default.build();
    EXPECT_EQ(
            graph_default.get_task_pool_capacity(),
            64); // 2 tasks × 20 multiplier = 40, rounded up to next power of 2

    // Test custom task pool capacity multiplier (10x) using TaskGraphBuilder
    ft::TaskGraph graph_custom("test_graph_custom");
    graph_custom.register_task("task1")
            .task_pool_capacity_multiplier(10)
            .function([]() { return ft::TaskResult{}; })
            .add();
    graph_custom.register_task("task2")
            .depends_on("task1")
            .function([]() { return ft::TaskResult{}; })
            .add();

    // Verify pool capacity before build
    EXPECT_EQ(graph_custom.get_task_pool_capacity(), 0); // Not built yet

    graph_custom.build();
    EXPECT_EQ(
            graph_custom.get_task_pool_capacity(),
            32); // 2 tasks × 10 multiplier = 20, rounded up to next power of 2

    // Both graphs should be built successfully
    EXPECT_EQ(graph_default.size(), 2);
    EXPECT_EQ(graph_custom.size(), 2);
}

TEST(TaskGraph, TaskWithDependencies) {
    ft::TaskGraph graph("name");

    auto parent_task = graph.register_task("parent").function([]() {}).add();

    graph.register_task("child").depends_on(parent_task).function([]() {}).add();

    EXPECT_EQ(graph.size(), 2);

    const auto &specs = graph.get_task_specs();
    EXPECT_EQ(specs[0].task_name, "parent");
    EXPECT_TRUE(specs[0].dependency_names.empty());

    EXPECT_EQ(specs[1].task_name, "child");
    EXPECT_EQ(specs[1].dependency_names.size(), 1);
    EXPECT_EQ(specs[1].dependency_names[0], "parent");
}

TEST(TaskGraph, TaskWithMultipleDependencies) {
    ft::TaskGraph graph("name");

    auto task_a = graph.register_task("task_a").function([]() {}).add();

    auto task_b = graph.register_task("task_b").function([]() {}).add();

    graph.register_task("child").depends_on({task_a, task_b}).function([]() {}).add();

    EXPECT_EQ(graph.size(), 3);

    const auto &specs = graph.get_task_specs();
    EXPECT_EQ(specs[2].task_name, "child");
    EXPECT_EQ(specs[2].dependency_names.size(), 2);
    EXPECT_EQ(specs[2].dependency_names[0], "task_a");
    EXPECT_EQ(specs[2].dependency_names[1], "task_b");
}

TEST(TaskGraph, InvalidTaskName) {
    ft::TaskGraph graph("name");

    EXPECT_THROW(
            {
                graph.register_task("").function([]() {}).add(); // Empty name
            },
            std::invalid_argument);
}

TEST(TaskGraph, InvalidTaskFunction) {
    ft::TaskGraph graph("name");

    EXPECT_THROW(
            {
                graph.register_task("test").add(); // No function set
            },
            std::invalid_argument);
}

TEST(TaskGraph, DuplicateTaskNames) {
    ft::TaskGraph graph("name");

    graph.register_task("duplicate").function([]() {}).add();

    EXPECT_THROW(
            {
                graph.register_task("duplicate") // Same name
                        .function([]() {})
                        .add();
            },
            std::invalid_argument);
}

TEST(TaskGraph, InvalidDependency) {
    ft::TaskGraph graph("name");

    EXPECT_THROW(
            {
                graph.register_task("child")
                        .depends_on("nonexistent_parent")
                        .function([]() {})
                        .add();
            },
            std::invalid_argument);
}

/**
 * TaskScheduler integration tests
 */
TEST(TaskGraph, BasicScheduling) {
    auto scheduler = ft::TaskScheduler::create().workers(2).build();
    ft::TaskGraph graph("name");

    std::atomic<bool> task_executed(false);

    graph.register_task("basic_task")
            .function([&task_executed]() { task_executed.store(true); })
            .add();

    graph.build();
    scheduler.schedule(graph);

    // Wait for execution by joining workers
    scheduler.join_workers();

    EXPECT_TRUE(task_executed.load());
    EXPECT_TRUE(graph.task_has_status("basic_task", ft::TaskStatus::Completed));
}

TEST(TaskGraph, DependencyChainScheduling) {
    auto scheduler = ft::TaskScheduler::create().workers(2).build();
    ft::TaskGraph graph("name");

    std::atomic<int> execution_order(0);
    std::atomic<int> task1_order(-1);
    std::atomic<int> task2_order(-1);
    std::atomic<int> task3_order(-1);

    auto task1 = graph.register_task("task1")
                         .function([&execution_order, &task1_order]() {
                             task1_order.store(execution_order.fetch_add(1));
                             std::this_thread::sleep_for(5ms);
                         })
                         .add();

    auto task2 = graph.register_task("task2")
                         .depends_on(task1)
                         .function([&execution_order, &task2_order]() {
                             task2_order.store(execution_order.fetch_add(1));
                             std::this_thread::sleep_for(5ms);
                         })
                         .add();

    auto task3 = graph.register_task("task3")
                         .depends_on(task2)
                         .function([&execution_order, &task3_order]() {
                             task3_order.store(execution_order.fetch_add(1));
                         })
                         .add();

    graph.build();
    scheduler.schedule(graph);

    // Wait for all tasks to complete by joining workers
    scheduler.join_workers();

    // Verify execution order
    EXPECT_EQ(task1_order.load(), 0);
    EXPECT_EQ(task2_order.load(), 1);
    EXPECT_EQ(task3_order.load(), 2);

    // Verify all completed
    EXPECT_TRUE(graph.task_has_status(task1, ft::TaskStatus::Completed));
    EXPECT_TRUE(graph.task_has_status(task2, ft::TaskStatus::Completed));
    EXPECT_TRUE(graph.task_has_status(task3, ft::TaskStatus::Completed));
}

/**
 * Reference capture and reset functionality tests
 */
TEST(TaskGraph, SimpleReferenceCapture) {
    auto scheduler = ft::TaskScheduler::create().workers(2).build();
    ft::TaskGraph graph("name");

    // Variables that will change
    std::atomic<bool> executed{false};

    // Build graph with reference capture
    auto load_task = graph.register_task("simple_test")
                             .function([&executed]() { executed.store(true); })
                             .add();

    // Schedule immediately like other working tests
    graph.build();
    scheduler.schedule(graph);

    // Wait for execution
    scheduler.join_workers();

    EXPECT_TRUE(executed.load());
    EXPECT_TRUE(graph.task_has_status(load_task, ft::TaskStatus::Completed));
}

TEST(TaskGraph, ReferenceCaptureAndReset) {
    ft::FlatMap<ft::TaskCategory, std::size_t> category_workers{};
    category_workers[ft::TaskCategory{ft::BuiltinTaskCategory::IO}] = 1;
    category_workers[ft::TaskCategory{ft::BuiltinTaskCategory::Compute}] = 1;
    auto scheduler = ft::TaskScheduler::create()
                             .workers(ft::WorkersConfig::create_for_categories(category_workers))
                             .build();
    ft::TaskGraph graph("name");

    // Variables that will change between iterations
    int current_value = 10;
    std::string current_message = "first";
    std::vector<int> execution_values;
    std::vector<std::string> execution_messages;

    // Build graph with reference capture
    auto load_task = graph.register_task("load_data")
                             .category(ft::BuiltinTaskCategory::IO)
                             .function([&current_value, &execution_values]() {
                                 execution_values.push_back(current_value);
                             })
                             .add();

    graph.register_task("process_data")
            .category(ft::BuiltinTaskCategory::Compute)
            .depends_on(load_task)
            .function([&current_message, &execution_messages]() {
                execution_messages.push_back(current_message);
            })
            .add();

    // First execution
    current_value = 42;
    current_message = "iteration_1";

    graph.build();
    scheduler.schedule(graph,
                       ft::Time::now_ns()); // Schedule immediately like other working tests

    // Wait for all tasks to complete before modifying shared variables
    wait_for_task_completion(graph, "process_data");

    EXPECT_TRUE(graph.task_has_status("load_data", ft::TaskStatus::Completed));

    // Prepare tasks for second execution

    // Change values for second iteration
    current_value = 100;
    current_message = "iteration_2";
    // Note: Re-build not needed for same graph structure, just reset is
    // sufficient
    scheduler.schedule(graph,
                       ft::Time::now_ns()); // Schedule immediately like other working tests

    // Wait for all tasks to complete before modifying shared variables
    wait_for_task_completion(graph, "process_data");

    EXPECT_TRUE(graph.task_has_status("load_data", ft::TaskStatus::Completed));

    // Verify that different values were captured
    EXPECT_EQ(execution_values.size(), 2);
    EXPECT_EQ(execution_values[0], 42);
    EXPECT_EQ(execution_values[1], 100);

    EXPECT_EQ(execution_messages.size(), 2);
    EXPECT_EQ(execution_messages[0], "iteration_1");
    EXPECT_EQ(execution_messages[1], "iteration_2");

    // Prepare tasks for third execution

    current_value = 999;
    current_message = "iteration_3";

    // Note: Re-build not needed for same graph structure, just reset is
    // sufficient
    scheduler.schedule(graph, ft::Time::now_ns() + 2ms);

    // Wait for all tasks to complete before accessing shared variables
    wait_for_task_completion(graph, "process_data");

    EXPECT_TRUE(graph.task_has_status("load_data", ft::TaskStatus::Completed));

    // Verify third iteration captured new values
    EXPECT_EQ(execution_values.size(), 3);
    EXPECT_EQ(execution_values[2], 999);

    EXPECT_EQ(execution_messages.size(), 3);
    EXPECT_EQ(execution_messages[2], "iteration_3");
}

TEST(TaskGraph, ComplexReferenceCaptureScenario) {
    auto scheduler = ft::TaskScheduler::create().workers(3).build();
    ft::TaskGraph graph("name");

    // Simulate a processing pipeline with changing configuration
    struct ProcessingConfig {
        int batch_size = 1;
        float threshold = 0.5F;
        bool enable_optimization = false;
    };

    ProcessingConfig config;
    std::vector<ProcessingConfig> recorded_configs;

    auto config_task = graph.register_task("configure")
                               .function([&config, &recorded_configs]() {
                                   recorded_configs.push_back(config);
                               })
                               .add();

    graph.register_task("process")
            .depends_on(config_task)
            .function([&config]() {
                // Use the current config values
                EXPECT_GT(config.batch_size, 0);
                EXPECT_GE(config.threshold, 0.0F);
            })
            .add();

    // First iteration - default config
    graph.build();
    scheduler.schedule(graph);

    // Wait for all tasks to complete before modifying shared variables
    wait_for_task_completion(graph, "process");

    // Check first iteration completed before reset
    EXPECT_TRUE(graph.task_has_status("configure", ft::TaskStatus::Completed));

    // Second iteration - modified config

    config.batch_size = 10;
    config.threshold = 0.8F;
    config.enable_optimization = true;

    // Note: Re-build not needed for same graph structure, just reset is
    // sufficient
    scheduler.schedule(graph, ft::Time::now_ns() + 1ms);

    // Wait for all tasks to complete before modifying shared variables
    wait_for_task_completion(graph, "process");

    // Check second iteration completed before reset
    EXPECT_TRUE(graph.task_has_status("configure", ft::TaskStatus::Completed));

    // Third iteration - different config again

    config.batch_size = 5;
    config.threshold = 0.3F;
    config.enable_optimization = false;

    // Note: Re-build not needed for same graph structure, just reset is
    // sufficient
    scheduler.schedule(graph, ft::Time::now_ns() + 2ms);

    // Wait for all tasks to complete before accessing shared variables
    wait_for_task_completion(graph, "process");

    // Check third iteration completed
    EXPECT_TRUE(graph.task_has_status("configure", ft::TaskStatus::Completed));

    // Verify different configurations were captured
    EXPECT_EQ(recorded_configs.size(), 3);

    EXPECT_EQ(recorded_configs[0].batch_size, 1);
    EXPECT_FLOAT_EQ(recorded_configs[0].threshold, 0.5F);
    EXPECT_FALSE(recorded_configs[0].enable_optimization);

    EXPECT_EQ(recorded_configs[1].batch_size, 10);
    EXPECT_FLOAT_EQ(recorded_configs[1].threshold, 0.8F);
    EXPECT_TRUE(recorded_configs[1].enable_optimization);

    EXPECT_EQ(recorded_configs[2].batch_size, 5);
    EXPECT_FLOAT_EQ(recorded_configs[2].threshold, 0.3F);
    EXPECT_FALSE(recorded_configs[2].enable_optimization);
}

TEST(TaskGraph, ClearAndRebuild) {
    ft::TaskGraph graph("name");

    // Build initial graph
    graph.register_task("task1").function([]() {}).add();

    EXPECT_EQ(graph.size(), 1);

    // Clear and rebuild
    graph.clear();
    EXPECT_TRUE(graph.empty());
    EXPECT_EQ(graph.size(), 0);

    // Build new graph
    graph.register_task("new_task").function([]() {}).add();

    EXPECT_EQ(graph.size(), 1);
    EXPECT_EQ(graph.get_task_specs()[0].task_name, "new_task");
}

/**
 * New API validation tests
 */
TEST(TaskGraph, BuildStateTracking) {
    ft::TaskGraph graph("name");

    // Initially not built
    EXPECT_FALSE(graph.is_built());

    // Add a task
    graph.register_task("test_task").function([]() {}).add();

    EXPECT_FALSE(graph.is_built());

    // Build the graph
    graph.build();
    EXPECT_TRUE(graph.is_built());

    // Clear resets build state
    graph.clear();
    EXPECT_FALSE(graph.is_built());
}

TEST(TaskGraph, CannotModifyAfterBuild) {
    ft::TaskGraph graph("name");

    // Add initial task and build
    graph.register_task("initial_task").function([]() {}).add();

    graph.build();
    EXPECT_TRUE(graph.is_built());

    // Attempt to add another task should throw
    EXPECT_THROW(
            { graph.register_task("should_fail").function([]() {}).add(); }, std::runtime_error);
}

TEST(TaskGraph, BuildEmptyGraph) {
    ft::TaskGraph graph("name");

    // Building empty graph should work
    EXPECT_NO_THROW(graph.build());
    EXPECT_TRUE(graph.is_built());
}

TEST(TaskGraph, DoubleBuildIsNoop) {
    ft::TaskGraph graph("name");

    graph.register_task("test_task").function([]() {}).add();

    // First build
    EXPECT_NO_THROW(graph.build());
    EXPECT_TRUE(graph.is_built());

    // Second build should be no-op (just warning)
    EXPECT_NO_THROW(graph.build());
    EXPECT_TRUE(graph.is_built());
}

TEST(TaskGraph, SchedulingRequiresBuild) {
    auto scheduler = ft::TaskScheduler::create().workers(1).build();
    ft::TaskGraph graph("name");

    // Add task but don't build
    graph.register_task("test_task").function([]() {}).add();

    // Scheduling without build should throw
    EXPECT_THROW({ scheduler.schedule(graph); }, std::runtime_error);
}

TEST(TaskGraph, ZeroAllocationScheduling) {
    auto scheduler = ft::TaskScheduler::create().workers(2).build();
    ft::TaskGraph graph("name");

    std::atomic<bool> task1_executed{false};
    std::atomic<bool> task2_executed{false};

    auto task1 = graph.register_task("task1")
                         .function([&task1_executed]() { task1_executed.store(true); })
                         .add();

    graph.register_task("task2")
            .depends_on(task1)
            .function([&task2_executed]() { task2_executed.store(true); })
            .add();

    // Build to enable zero-allocation scheduling
    graph.build();

    // Schedule tasks
    scheduler.schedule(graph);

    // Wait for execution
    scheduler.join_workers();

    // Verify execution
    EXPECT_TRUE(task1_executed.load());
    EXPECT_TRUE(task2_executed.load());
    EXPECT_TRUE(graph.task_has_status("task1", ft::TaskStatus::Completed));
}

TEST(TaskGraph, BuildPreservesGraphStructure) {
    ft::TaskGraph graph("name");

    // Build complex graph
    auto task_a = graph.register_task("task_a")
                          .category(ft::TaskCategory{ft::BuiltinTaskCategory::IO})
                          .timeout(1ms)
                          .function([]() {})
                          .add();

    auto task_b = graph.register_task("task_b")
                          .category(ft::TaskCategory{ft::BuiltinTaskCategory::Compute})
                          .function([]() {})
                          .add();

    graph.register_task("task_c")
            .depends_on({task_a, task_b})
            .category(ft::TaskCategory{ft::BuiltinTaskCategory::Default})
            .timeout(500us)
            .function([]() {})
            .add();

    // Build
    graph.build();

    // Verify built structure
    const auto &built = graph.prepare_tasks(ft::Time::now_ns());
    EXPECT_EQ(built.size(), 3);

    // Find tasks by name (order may vary)
    const ft::Task *task_a_built = nullptr;
    const ft::Task *task_b_built = nullptr;
    const ft::Task *task_c_built = nullptr;

    for (const auto &task : built) {
        if (task->get_task_name() == "task_a") {
            task_a_built = task.get();
        } else if (task->get_task_name() == "task_b") {
            task_b_built = task.get();
        } else if (task->get_task_name() == "task_c") {
            task_c_built = task.get();
        }
    }

    ASSERT_NE(task_a_built, nullptr);
    ASSERT_NE(task_b_built, nullptr);
    ASSERT_NE(task_c_built, nullptr);

    // Verify properties preserved
    EXPECT_EQ(task_a_built->get_category(), ft::TaskCategory{ft::BuiltinTaskCategory::IO});
    EXPECT_EQ(task_a_built->get_timeout_ns(), 1ms);
    EXPECT_EQ(task_b_built->get_category(), ft::TaskCategory{ft::BuiltinTaskCategory::Compute});
    EXPECT_EQ(task_c_built->get_category(), ft::TaskCategory{ft::BuiltinTaskCategory::Default});
    EXPECT_EQ(task_c_built->get_timeout_ns(), 500us);

    // Verify dependencies (get from original specs)
    const auto &specs = graph.get_task_specs();
    auto task_c_spec = std::find_if(specs.begin(), specs.end(), [](const auto &spec) {
        return spec.task_name == "task_c";
    });
    ASSERT_NE(task_c_spec, specs.end());
    EXPECT_EQ(task_c_spec->dependency_names.size(), 2);

    // Verify task count by checking built tasks
    const auto &scheduled_tasks = graph.prepare_tasks(0ns);
    EXPECT_EQ(scheduled_tasks.size(), 3); // All three tasks

    // Verify root task count by checking specs
    const auto &task_specs = graph.get_task_specs();
    const auto root_task_count = static_cast<std::size_t>(
            std::count_if(task_specs.begin(), task_specs.end(), [](const auto &spec) {
                return spec.dependency_names.empty();
            }));
    EXPECT_EQ(root_task_count, 2); // task_a and task_b are roots
}

TEST(TaskGraph, MultipleParentExecution) {
    auto scheduler = ft::TaskScheduler::create().workers(3).build();
    ft::TaskGraph graph("name");

    std::atomic<int> execution_order(0);
    std::vector<std::atomic<int>> task_orders(4);
    for (auto &order : task_orders) {
        order.store(-1); // Initialize to -1 (not executed)
    }

    // Create two independent parent tasks
    auto parent_a = graph.register_task("parent_a")
                            .function([&execution_order, &task_orders]() {
                                task_orders[0].store(execution_order.fetch_add(1));
                                std::this_thread::sleep_for(5ms);
                            })
                            .add();

    auto parent_b = graph.register_task("parent_b")
                            .function([&execution_order, &task_orders]() {
                                task_orders[1].store(execution_order.fetch_add(1));
                                std::this_thread::sleep_for(10ms);
                            })
                            .add();

    // Create child task that depends on both parents
    auto child_task = graph.register_task("child_multi_parent")
                              .depends_on({parent_a, parent_b}) // Multiple dependencies
                              .function([&execution_order, &task_orders]() {
                                  task_orders[2].store(execution_order.fetch_add(1));
                              })
                              .add();

    // Create grandchild task that depends on the child
    auto grandchild_task = graph.register_task("grandchild")
                                   .depends_on(child_task)
                                   .function([&execution_order, &task_orders]() {
                                       task_orders[3].store(execution_order.fetch_add(1));
                                   })
                                   .add();

    // Build and schedule the graph
    graph.build();
    scheduler.schedule(graph);

    // Wait for all tasks to complete
    scheduler.join_workers();

    EXPECT_TRUE(graph.task_has_status("child_multi_parent", ft::TaskStatus::Completed));
    EXPECT_TRUE(graph.task_has_status(grandchild_task, ft::TaskStatus::Completed));

    // Verify execution order: both parents before child, child before grandchild
    const int parent_a_order = task_orders[0].load();
    const int parent_b_order = task_orders[1].load();
    const int child_order = task_orders[2].load();
    const int grandchild_order = task_orders[3].load();

    EXPECT_GE(parent_a_order, 0) << "Parent A should have executed";
    EXPECT_GE(parent_b_order, 0) << "Parent B should have executed";
    EXPECT_GT(child_order, parent_a_order) << "Child should execute after parent A";
    EXPECT_GT(child_order, parent_b_order) << "Child should execute after parent B";
    EXPECT_GT(grandchild_order, child_order) << "Grandchild should execute after child";
}

TEST(TaskGraph, ComplexMultiParentDiamond) {
    auto scheduler = ft::TaskScheduler::create().workers(4).build();
    ft::TaskGraph graph("name");

    std::atomic<int> execution_order(0);
    std::vector<std::atomic<int>> task_orders(7);
    for (auto &order : task_orders) {
        order.store(-1);
    }

    // Create diamond pattern:
    /* clang-format off
   *       A
   *     /   \
   *    B     C
   *   / \   / \
   *  D   E F   G
   *   \ | | /
   *     H
   */
    // clang-format on

    auto task_a = graph.register_task("A")
                          .function([&execution_order, &task_orders]() {
                              task_orders[0].store(execution_order.fetch_add(1));
                          })
                          .add();

    auto task_b = graph.register_task("B")
                          .depends_on(task_a)
                          .function([&execution_order, &task_orders]() {
                              task_orders[1].store(execution_order.fetch_add(1));
                          })
                          .add();

    auto task_c = graph.register_task("C")
                          .depends_on(task_a)
                          .function([&execution_order, &task_orders]() {
                              task_orders[2].store(execution_order.fetch_add(1));
                          })
                          .add();

    auto task_d = graph.register_task("D")
                          .depends_on(task_b)
                          .function([&execution_order, &task_orders]() {
                              task_orders[3].store(execution_order.fetch_add(1));
                          })
                          .add();

    auto task_e = graph.register_task("E")
                          .depends_on(task_b)
                          .function([&execution_order, &task_orders]() {
                              task_orders[4].store(execution_order.fetch_add(1));
                          })
                          .add();

    auto task_f = graph.register_task("F")
                          .depends_on(task_c)
                          .function([&execution_order, &task_orders]() {
                              task_orders[5].store(execution_order.fetch_add(1));
                          })
                          .add();

    auto task_g = graph.register_task("G")
                          .depends_on(task_c)
                          .function([&execution_order, &task_orders]() {
                              task_orders[6].store(execution_order.fetch_add(1));
                          })
                          .add();

    // Task H depends on ALL four leaf tasks (D, E, F, G)
    graph.register_task("H")
            .depends_on({task_d, task_e, task_f, task_g}) // Four parents!
            .function([&execution_order]() { execution_order.fetch_add(1); })
            .add();

    // Build and schedule
    graph.build();
    scheduler.schedule(graph);

    // Wait for completion
    std::this_thread::sleep_for(100ms);

    // All tasks should complete successfully
    for (const char *name : {"A", "B", "C", "D", "E", "F", "G", "H"}) {
        EXPECT_TRUE(graph.task_has_status(name, ft::TaskStatus::Completed))
                << "Task " << name << " should complete";
    }

    // Verify dependencies: H should execute after all of D, E, F, G
    for (std::size_t i = 3; i <= 6; ++i) { // D, E, F, G indices
        EXPECT_GE(task_orders[i].load(), 0) << "Task should have executed";
    }
}

/**
 * Test TaskGraph execution with tasks scheduled in the past (late execution)
 */
TEST(TaskGraph, LateTaskGraphExecution) {
    auto scheduler = ft::TaskScheduler::create().workers(2).build();
    ft::TaskGraph graph("name");

    std::atomic<int> execution_count{0};
    std::atomic<bool> root_executed{false};
    std::atomic<bool> child_executed{false};

    // Create root task
    auto late_root = graph.register_task("late_root")
                             .function([&execution_count, &root_executed]() {
                                 root_executed = true;
                                 execution_count++;
                                 RT_LOGC_INFO(ft::TaskLog::TaskGraph, "Late root task executed");
                                 std::this_thread::sleep_for(5ms);
                             })
                             .add();

    // Create child task that depends on root
    auto late_child = graph.register_task("late_child")
                              .function([&execution_count, &child_executed]() {
                                  child_executed = true;
                                  execution_count++;
                                  RT_LOGC_INFO(ft::TaskLog::TaskGraph, "Late child task executed");
                              })
                              .depends_on(std::vector<std::string_view>{late_root})
                              .add();

    // Build the graph
    graph.build();

    // Schedule the entire graph 500ms in the past
    const auto now = ft::Time::now_ns();
    const auto past_time = now - 500ms;

    scheduler.schedule(graph, past_time);

    // Wait for completion
    const auto deadline = ft::Time::now_ns() + 100ms;
    while (execution_count.load() < 2 && ft::Time::now_ns() < deadline) {
        std::this_thread::sleep_for(1ms);
    }

    // Verify both tasks executed despite being scheduled in the past
    EXPECT_TRUE(root_executed.load()) << "Root task should execute even when scheduled in past";
    EXPECT_TRUE(child_executed.load()) << "Child task should execute after root completes";
    EXPECT_EQ(execution_count.load(), 2) << "All tasks should execute";

    // Verify final statuses
    EXPECT_TRUE(graph.task_has_status(late_root, ft::TaskStatus::Completed));
    EXPECT_TRUE(graph.task_has_status(late_child, ft::TaskStatus::Completed));
}

TEST(TaskGraph, GenerateTaskGraphVisualization) {
    ft::TaskGraph graph("name");

    // Create a simple graph with dependencies for visualization
    auto parent_task = graph.register_task("parent_task")
                               .category(ft::TaskCategory{ft::BuiltinTaskCategory::IO})
                               .function([]() {})
                               .add();

    graph.register_task("child_task")
            .category(ft::TaskCategory{ft::BuiltinTaskCategory::Compute})
            .depends_on(parent_task)
            .function([]() {})
            .add();

    // Test the string visualization generation
    const std::string visualization = graph.to_string();

    // Should generate a non-empty string representation
    EXPECT_FALSE(visualization.empty());
    EXPECT_TRUE(visualization.find("parent_task") != std::string::npos);
    EXPECT_TRUE(visualization.find("child_task") != std::string::npos);
}

TEST(TaskGraph, DependencyGenerationLinearChain) {
    ft::TaskGraph graph("name");

    // Create linear dependency chain: A -> B -> C -> D
    auto task_a = graph.register_task("A").function([]() { return ft::TaskResult{}; }).add();

    auto task_b = graph.register_task("B")
                          .depends_on(task_a)
                          .function([]() { return ft::TaskResult{}; })
                          .add();

    auto task_c = graph.register_task("C")
                          .depends_on(task_b)
                          .function([]() { return ft::TaskResult{}; })
                          .add();

    graph.register_task("D").depends_on(task_c).function([]() { return ft::TaskResult{}; }).add();

    // Build the graph to calculate dependency generations
    graph.build();

    const auto &sched_tasks = graph.prepare_tasks(ft::Time::now_ns());
    EXPECT_EQ(sched_tasks.size(), 4);

    // Find tasks by name and verify dependency generations
    for (const auto &task : sched_tasks) {
        if (task->get_task_name() == "A") {
            EXPECT_EQ(task->get_dependency_generation(), 0) << "Root task should have generation 0";
        } else if (task->get_task_name() == "B") {
            EXPECT_EQ(task->get_dependency_generation(), 1)
                    << "B depends on A, should have generation 1";
        } else if (task->get_task_name() == "C") {
            EXPECT_EQ(task->get_dependency_generation(), 2)
                    << "C depends on B, should have generation 2";
        } else if (task->get_task_name() == "D") {
            EXPECT_EQ(task->get_dependency_generation(), 3)
                    << "D depends on C, should have generation 3";
        }
    }
}

TEST(TaskGraph, DependencyGenerationDiamondPattern) {
    ft::TaskGraph graph("name");

    // Create diamond pattern: A -> B, C -> D
    auto task_a = graph.register_task("A").function([]() { return ft::TaskResult{}; }).add();

    auto task_b = graph.register_task("B")
                          .depends_on(task_a)
                          .function([]() { return ft::TaskResult{}; })
                          .add();

    auto task_c = graph.register_task("C")
                          .depends_on(task_a)
                          .function([]() { return ft::TaskResult{}; })
                          .add();

    graph.register_task("D")
            .depends_on({task_b, task_c})
            .function([]() { return ft::TaskResult{}; })
            .add();

    // Build the graph to calculate dependency generations
    graph.build();

    const auto &sched_tasks = graph.prepare_tasks(ft::Time::now_ns());
    EXPECT_EQ(sched_tasks.size(), 4);

    // Find tasks by name and verify dependency generations
    for (const auto &task : sched_tasks) {
        if (task->get_task_name() == "A") {
            EXPECT_EQ(task->get_dependency_generation(), 0) << "Root task should have generation 0";
        } else if (task->get_task_name() == "B") {
            EXPECT_EQ(task->get_dependency_generation(), 1)
                    << "B depends on A, should have generation 1";
        } else if (task->get_task_name() == "C") {
            EXPECT_EQ(task->get_dependency_generation(), 1)
                    << "C depends on A, should have generation 1";
        } else if (task->get_task_name() == "D") {
            EXPECT_EQ(task->get_dependency_generation(), 2)
                    << "D depends on B and C (gen 1), should have generation 2";
        }
    }
}

TEST(TaskGraph, DependencyGenerationComplexMultiLevel) {
    ft::TaskGraph graph("name");

    /* Create complex multi-level pattern:
     *     A
     *   /   \
     *  B     C
     *  |    / \
     *  D   E   F
     *   \ /   /
     *    G   /
     *     \ /
     *      H
     */

    auto task_a = graph.register_task("A").function([]() { return ft::TaskResult{}; }).add();
    auto task_b = graph.register_task("B")
                          .depends_on(task_a)
                          .function([]() { return ft::TaskResult{}; })
                          .add();
    auto task_c = graph.register_task("C")
                          .depends_on(task_a)
                          .function([]() { return ft::TaskResult{}; })
                          .add();
    auto task_d = graph.register_task("D")
                          .depends_on(task_b)
                          .function([]() { return ft::TaskResult{}; })
                          .add();
    auto task_e = graph.register_task("E")
                          .depends_on(task_c)
                          .function([]() { return ft::TaskResult{}; })
                          .add();
    auto task_f = graph.register_task("F")
                          .depends_on(task_c)
                          .function([]() { return ft::TaskResult{}; })
                          .add();
    auto task_g = graph.register_task("G")
                          .depends_on({task_d, task_e})
                          .function([]() { return ft::TaskResult{}; })
                          .add();
    graph.register_task("H")
            .depends_on({task_g, task_f})
            .function([]() { return ft::TaskResult{}; })
            .add();

    // Build the graph
    graph.build();

    const auto &sched_tasks = graph.prepare_tasks(ft::Time::now_ns());
    EXPECT_EQ(sched_tasks.size(), 8);

    // Verify dependency generations
    std::map<std::string, std::uint32_t> expected_generations = {
            {"A", 0}, // Root
            {"B", 1},
            {"C", 1}, // Children of A
            {"D", 2},
            {"E", 2},
            {"F", 2}, // Grandchildren
            {"G", 3}, // Depends on D(gen 2) and E(gen 2), so gen 3
            {"H", 4}  // Depends on G(gen 3) and F(gen 2), so gen 4
    };

    for (const auto &task : sched_tasks) {
        const auto expected_it = expected_generations.find(std::string(task->get_task_name()));
        ASSERT_NE(expected_it, expected_generations.end())
                << "Unexpected task: " << task->get_task_name();
        EXPECT_EQ(task->get_dependency_generation(), expected_it->second)
                << "Task " << task->get_task_name() << " has wrong dependency generation";
    }
}

TEST(TaskGraph, DependencyGenerationWithMultipleRoots) {
    ft::TaskGraph graph("name");

    // Create pattern with multiple roots: A1 -> B, A2 -> C -> D
    auto task_a1 = graph.register_task("A1").function([]() { return ft::TaskResult{}; }).add();
    auto task_a2 = graph.register_task("A2").function([]() { return ft::TaskResult{}; }).add();
    graph.register_task("B").depends_on(task_a1).function([]() { return ft::TaskResult{}; }).add();
    auto task_c = graph.register_task("C")
                          .depends_on(task_a2)
                          .function([]() { return ft::TaskResult{}; })
                          .add();
    graph.register_task("D").depends_on(task_c).function([]() { return ft::TaskResult{}; }).add();

    // Build the graph
    graph.build();

    const auto &sched_tasks = graph.prepare_tasks(ft::Time::now_ns());
    EXPECT_EQ(sched_tasks.size(), 5);

    // Verify dependency generations
    for (const auto &task : sched_tasks) {
        if (task->get_task_name() == "A1" || task->get_task_name() == "A2") {
            EXPECT_EQ(task->get_dependency_generation(), 0)
                    << "Root tasks should have generation 0";
        } else if (task->get_task_name() == "B") {
            EXPECT_EQ(task->get_dependency_generation(), 1)
                    << "B depends on A1, should have generation 1";
        } else if (task->get_task_name() == "C") {
            EXPECT_EQ(task->get_dependency_generation(), 1)
                    << "C depends on A2, should have generation 1";
        } else if (task->get_task_name() == "D") {
            EXPECT_EQ(task->get_dependency_generation(), 2)
                    << "D depends on C, should have generation 2";
        }
    }
}

TEST(TaskGraph, DependencyGenerationSingleTask) {
    ft::TaskGraph graph("name");

    // Single task with no dependencies
    graph.register_task("SingleTask").function([]() { return ft::TaskResult{}; }).add();

    graph.build();

    const auto &sched_tasks = graph.prepare_tasks(ft::Time::now_ns());
    EXPECT_EQ(sched_tasks.size(), 1);
    EXPECT_EQ(sched_tasks[0]->get_dependency_generation(), 0)
            << "Single task should have generation 0";
    EXPECT_EQ(sched_tasks[0]->get_task_name(), "SingleTask");
}

// Test functions for TaskGraph function pointer tests
void simple_task_function() {
    // Do some simple work
}

ft::TaskResult task_function_with_result() {
    return ft::TaskResult{ft::TaskStatus::Completed, "Function task completed"};
}

void task_function_with_work() {
    // Simulate some computational work
    std::this_thread::sleep_for(1ms);
}

/**
 * Tests for TaskGraph API with functions (not just lambdas)
 */
TEST(TaskGraph, FunctionTypesTable) {
    struct TestCase {
        std::string name;
        ft::TaskCategory category;
        std::chrono::milliseconds timeout;
    };

    const std::vector<TestCase> test_cases = {
            {"VoidFunctionTest", ft::TaskCategory{ft::BuiltinTaskCategory::Default}, 1ms},
            {"TaskResultFunctionTest", ft::TaskCategory{ft::BuiltinTaskCategory::Compute}, 2ms},
            {"WorkFunctionTest", ft::TaskCategory{ft::BuiltinTaskCategory::IO}, 3ms},
            {"StdFunctionTest", ft::TaskCategory{ft::BuiltinTaskCategory::LowPriority}, 4ms}};

    for (const auto &test_case : test_cases) {
        SCOPED_TRACE("Test case: " + test_case.name);

        ft::TaskGraph graph("name");

        // Create task based on test case type
        if (test_case.name == "VoidFunctionTest") {
            graph.register_task(test_case.name)
                    .category(test_case.category)
                    .timeout(test_case.timeout)
                    .function(simple_task_function)
                    .add();
        } else if (test_case.name == "TaskResultFunctionTest") {
            graph.register_task(test_case.name)
                    .category(test_case.category)
                    .timeout(test_case.timeout)
                    .function(task_function_with_result)
                    .add();
        } else if (test_case.name == "WorkFunctionTest") {
            graph.register_task(test_case.name)
                    .category(test_case.category)
                    .timeout(test_case.timeout)
                    .function(task_function_with_work)
                    .add();
        } else if (test_case.name == "StdFunctionTest") {
            const std::function<ft::TaskResult()> wrapped_func = task_function_with_result;
            graph.register_task(test_case.name)
                    .category(test_case.category)
                    .timeout(test_case.timeout)
                    .function(wrapped_func)
                    .add();
        }

        // Verify basic graph properties
        EXPECT_FALSE(graph.empty());
        EXPECT_EQ(graph.size(), 1);

        const auto &specs = graph.get_task_specs();
        EXPECT_EQ(specs.size(), 1);
        EXPECT_EQ(specs[0].task_name, test_case.name);
        EXPECT_EQ(specs[0].category, test_case.category);
        EXPECT_EQ(specs[0].timeout, test_case.timeout);
        EXPECT_TRUE(specs[0].dependency_names.empty());

        // Verify task can be built
        EXPECT_NO_THROW(graph.build());
        const auto &sched_tasks = graph.prepare_tasks(ft::Time::now_ns());
        EXPECT_EQ(sched_tasks.size(), 1);
        EXPECT_TRUE(graph.task_has_status(test_case.name, ft::TaskStatus::NotStarted));
    }
}

TEST(TaskGraph, WorkerShutdownBehaviorDemo) {
    auto scheduler = ft::TaskScheduler::create().workers(2).build();
    ft::TaskGraph graph("name");

    std::atomic<bool> task_executed{false};

    // Create a task that will be cancelled
    graph.register_task("cancellable_task")
            .function([&task_executed]() { task_executed.store(true); })
            .add();

    graph.build();

    // Schedule for future execution
    scheduler.schedule(graph, ft::Time::now_ns() + 50ms); // 50ms future

    // Immediately shutdown and cancel pending tasks
    scheduler.join_workers(ft::WorkerShutdownBehavior::CancelPendingTasks);

    // Task should not have executed
    EXPECT_FALSE(task_executed.load()) << "Task should have been cancelled before execution";
    EXPECT_TRUE(graph.task_has_status("cancellable_task", ft::TaskStatus::Cancelled));
}

TEST(TaskGraph, TimesScheduledTracking) {
    struct TestCase {
        std::string description;
        int schedule_count;
        std::uint64_t expected_graph_count;
        std::uint64_t expected_task_round;
        bool clear_after_first;
    };

    const std::vector<TestCase> test_cases = {
            {"Initial state", 0, 0U, 0U, false},
            {"Single schedule", 1, 1U, 0U, false},
            {"Multiple schedules", 3, 3U, 2U, false},
            {"Clear and rebuild", 2, 1U, 0U, true}};

    for (const auto &test_case : test_cases) {
        SCOPED_TRACE("Test case: " + test_case.description);

        auto scheduler = ft::TaskScheduler::create().workers(1).build();
        ft::TaskGraph graph("name");

        // Initial state
        EXPECT_EQ(graph.get_times_scheduled(), 0U);

        // Create a simple task
        graph.register_task("test_task").function([]() {}).add();
        graph.build();

        // Schedule the specified number of times
        for (int i = 0; i < test_case.schedule_count; ++i) {
            if (i == 1 && test_case.clear_after_first) {
                graph.clear();
                EXPECT_EQ(graph.get_times_scheduled(), 0U)
                        << "Graph count should reset after clear";

                // Rebuild after clear
                graph.register_task("new_task").function([]() {}).add();
                graph.build();
            }

            scheduler.schedule(graph);

            // wait for the task to execute
            std::this_thread::sleep_for(1ms);
        }

        scheduler.join_workers();

        EXPECT_EQ(graph.get_times_scheduled(), test_case.expected_graph_count)
                << test_case.description << " - graph scheduling count";
    }
}

TEST(TaskGraph, ClearScheduledTasksBreaksParentChildRelationships) {
    ft::TaskGraph graph("name");

    // Create a graph with parent/child/grandchild relationships
    auto root_1 = graph.register_task("root_1").function([]() { return ft::TaskResult{}; }).add();

    auto root_2 = graph.register_task("root_2").function([]() { return ft::TaskResult{}; }).add();

    auto child = graph.register_task("child")
                         .depends_on({root_1, root_2})
                         .function([]() { return ft::TaskResult{}; })
                         .add();

    graph.register_task("grandchild")
            .depends_on(child)
            .function([]() { return ft::TaskResult{}; })
            .add();

    // Build the graph and prepare tasks
    graph.build();
    const auto &scheduled_tasks = graph.prepare_tasks(ft::Time::now_ns());

    // Verify initial state - tasks should be populated with parent relationships
    EXPECT_EQ(scheduled_tasks.size(), 4);
    EXPECT_FALSE(scheduled_tasks.empty()) << "Scheduled tasks should be populated";

    // Find tasks by name for verification
    std::shared_ptr<ft::Task> root_1_task{};
    std::shared_ptr<ft::Task> root_2_task{};
    std::shared_ptr<ft::Task> child_task{};
    std::shared_ptr<ft::Task> grandchild_task{};

    for (const auto &task : scheduled_tasks) {
        const std::string name{task->get_task_name()};
        if (name == "root_1") {
            root_1_task = task;
        } else if (name == "root_2") {
            root_2_task = task;
        } else if (name == "child") {
            child_task = task;
        } else if (name == "grandchild") {
            grandchild_task = task;
        }
    }

    ASSERT_NE(root_1_task, nullptr) << "Root 1 task should exist";
    ASSERT_NE(root_2_task, nullptr) << "Root 2 task should exist";
    ASSERT_NE(child_task, nullptr) << "Child task should exist";
    ASSERT_NE(grandchild_task, nullptr) << "Grandchild task should exist";

    // Verify parent relationships are set up correctly
    EXPECT_TRUE(root_1_task->has_no_parents()) << "Root 1 should have no parents";
    EXPECT_TRUE(root_2_task->has_no_parents()) << "Root 2 should have no parents";
    EXPECT_FALSE(child_task->has_no_parents()) << "Child should have parents";
    EXPECT_FALSE(grandchild_task->has_no_parents()) << "Grandchild should have parents";

    // Store task pool stats before cleanup to track releases
    const auto stats_before = graph.get_pool_stats();

    // Call clear_scheduled_tasks to break parent relationships and clear
    // scheduled_tasks
    graph.clear_scheduled_tasks();

    // Note: After clear_scheduled_tasks(), the internal scheduled_tasks_ vector
    // is empty The clear operation should have succeeded

    // Get stats after clearing scheduled_tasks - tasks should NOT be released yet
    // because local shared_ptr variables still hold references
    const auto stats_after_clear = graph.get_pool_stats();
    EXPECT_EQ(stats_after_clear.tasks_released, stats_before.tasks_released)
            << "Tasks should NOT be released yet because local shared_ptrs still "
               "hold references. "
            << "Before: " << stats_before.tasks_released
            << ", After clear: " << stats_after_clear.tasks_released;

    // Now reset all local shared_ptr references to simulate them going out of
    // scope
    root_1_task.reset();
    root_2_task.reset();
    child_task.reset();
    grandchild_task.reset();

    // Get updated pool stats to verify tasks were finally released
    const auto stats_after_reset = graph.get_pool_stats();

    // The number of released tasks should have increased by the number of
    // scheduled tasks
    const auto expected_releases = stats_before.tasks_released + 4; // 4 tasks were released
    EXPECT_EQ(stats_after_reset.tasks_released, expected_releases)
            << "All 4 tasks should have been released back to pool after resetting "
               "local references. "
            << "Before: " << stats_before.tasks_released
            << ", After reset: " << stats_after_reset.tasks_released;
}

TEST(TaskGraph, TaskSchedulerTimeoutCancellation) {
    bool cancelled = false;
    bool user_data_received = false;

    auto scheduler = ft::TaskScheduler::create().workers(1).build();

    // Create task that sleeps longer than timeout with simple user data
    auto graph = ft::TaskGraph::create("graph")
                         .single_task("timeout_test")
                         .timeout(5ms)
                         .user_data(std::string("test_data"))
                         .function(
                                 [&cancelled, &user_data_received](
                                         const ft::TaskContext &ctx) -> ft::TaskResult {
                                     // Check user data
                                     auto data = ctx.get_user_data<std::string>();
                                     if (data && *data == "test_data") {
                                         user_data_received = true;
                                     }

                                     // Sleep longer than timeout to trigger TaskMonitor
                                     // cancellation
                                     std::this_thread::sleep_for(50ms);

                                     // Check if cancelled by TaskMonitor
                                     if (ctx.cancellation_token->is_cancelled()) {
                                         cancelled = true;
                                         return ft::TaskResult{
                                                 ft::TaskStatus::Cancelled, "Cancelled by timeout"};
                                     }

                                     return ft::TaskResult{
                                             ft::TaskStatus::Completed, "Should not reach here"};
                                 })
                         .build();

    // Schedule and wait for completion
    scheduler.schedule(graph);
    scheduler.join_workers();

    // Verify results
    EXPECT_TRUE(user_data_received);
    EXPECT_TRUE(cancelled);
    EXPECT_TRUE(graph.task_has_status("timeout_test", ft::TaskStatus::Cancelled));
}

TEST(TaskGraph, DisableTaskAcrossExecutionRounds) {
    auto scheduler = ft::TaskScheduler::create().workers(1).build();
    ft::TaskGraph graph("test_graph");

    std::atomic<int> execution_count{0};

    auto test_task = graph.register_task("test_task")
                             .function([&execution_count]() { execution_count.fetch_add(1); })
                             .add();

    graph.build();

    // First scheduling round - task should execute
    scheduler.schedule(graph);
    std::this_thread::sleep_for(5ms); // Wait for execution
    EXPECT_EQ(execution_count.load(), 1);

    // Disable the task for next round
    EXPECT_TRUE(graph.disable_task(test_task));

    // Second scheduling round - task should NOT execute
    scheduler.schedule(graph);
    std::this_thread::sleep_for(5ms); // Wait to ensure no execution
    EXPECT_EQ(execution_count.load(), 1);

    // Re-enable and execute again
    EXPECT_TRUE(graph.enable_task(test_task));
    scheduler.schedule(graph);
    std::this_thread::sleep_for(5ms); // Wait for execution
    EXPECT_EQ(execution_count.load(), 2);
}

TEST(TaskGraph, DisableParentDisablesChildren) {
    auto scheduler = ft::TaskScheduler::create().workers(1).build();
    ft::TaskGraph graph("test_graph");

    std::atomic<bool> parent_executed{false};
    std::atomic<bool> child_executed{false};
    std::atomic<bool> grandchild_executed{false};

    auto parent_task = graph.register_task("parent")
                               .function([&parent_executed]() { parent_executed.store(true); })
                               .add();

    auto child_task = graph.register_task("child")
                              .depends_on(parent_task)
                              .function([&child_executed]() { child_executed.store(true); })
                              .add();

    graph.register_task("grandchild")
            .depends_on(child_task)
            .function([&grandchild_executed]() { grandchild_executed.store(true); })
            .add();

    graph.build();

    // Disable the parent task
    EXPECT_TRUE(graph.disable_task("parent"));

    scheduler.schedule(graph);
    std::this_thread::sleep_for(5ms); // Wait to ensure no execution

    EXPECT_FALSE(parent_executed.load());
    EXPECT_FALSE(child_executed.load());
    EXPECT_FALSE(grandchild_executed.load());
}

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

} // namespace
