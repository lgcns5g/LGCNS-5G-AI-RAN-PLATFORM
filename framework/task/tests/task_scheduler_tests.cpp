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
 * @file task_scheduler_tests.cpp
 * @brief Unit tests for TaskScheduler class and related configuration
 * structures
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <ratio>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include <quill/LogMacros.h>

#include <gtest/gtest.h>

#include "log/rt_log_macros.hpp"
#include "task/flat_map.hpp"
#include "task/task.hpp"
#include "task/task_category.hpp"
#include "task/task_errors.hpp"
#include "task/task_graph.hpp"
#include "task/task_log.hpp"
#include "task/task_pool.hpp"
#include "task/task_scheduler.hpp"
#include "task/task_utils.hpp"
#include "task/task_worker.hpp"
#include "task/time.hpp"
#include "task/timed_trigger.hpp"

namespace {
namespace ft = framework::task;

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

using namespace std::chrono_literals;

/**
 * TaskScheduler construction and basic functionality
 */
TEST(TaskScheduler, DefaultConstruction) {
    // Default construction should work without throwing
    EXPECT_NO_THROW({ auto scheduler = ft::TaskScheduler::create().build(); });
}

TEST(TaskScheduler, CustomConstruction) {
    const ft::WorkersConfig workers_config(2); // 2 workers
    const std::chrono::microseconds worker_sleep(50);

    EXPECT_NO_THROW({
        auto scheduler = ft::TaskScheduler::create()
                                 .workers(workers_config)
                                 .task_readiness_tolerance(500us)
                                 .no_monitor_pinning()
                                 .worker_sleep(worker_sleep)
                                 .build();

        EXPECT_EQ(scheduler.get_workers_config().size(), 2);
    });
}

TEST(TaskScheduler, InvalidWorkerConfiguration) {
    // Create invalid config
    std::vector<ft::WorkerConfig> invalid_workers;
    ft::WorkerConfig bad_config;
    bad_config.categories.clear(); // Invalid: no categories
    invalid_workers.push_back(bad_config);

    const ft::WorkersConfig invalid_config(invalid_workers);

    // Should throw due to invalid configuration
    EXPECT_THROW(
            { auto scheduler = ft::TaskScheduler::create().workers(invalid_config).build(); },
            std::invalid_argument);
}

TEST(TaskScheduler, ConfigurableBlackoutThreshold) {
    EXPECT_NO_THROW({
        auto scheduler = ft::TaskScheduler::create()
                                 .workers(1)
                                 .worker_blackout_warn_threshold(500us)
                                 .build();
    });
}

TEST(TaskScheduler, SingleTaskLoop) {
    using namespace framework::task;
    auto scheduler = TaskScheduler::create().workers(1).build();

    std::atomic<int> count(0);
    auto graph = TaskGraph::create("graph")
                         .single_task("task_name")
                         .function([&count]() { ++count; })
                         .build();

    const std::array iterations{0, 1, 2};
    for ([[maybe_unused]] const auto _ : iterations) {
        scheduler.schedule(graph);
    }

    scheduler.join_workers();

    EXPECT_EQ(count.load(), iterations.size());
    EXPECT_TRUE(graph.task_has_status("task_name", ft::TaskStatus::Completed));
}

TEST(TaskScheduler, TaskSchedulingBasic) {
    auto scheduler = ft::TaskScheduler::create().workers(2).build();

    bool task_executed = false;
    const ft::Nanos execution_time = ft::Time::now_ns() + 1ms; // 1ms in future

    ft::TaskGraph graph("name");
    auto test_task = graph.register_task("test_task")
                             .timeout(5ms)
                             .function([&task_executed]() { task_executed = true; })
                             .add();

    graph.build();
    scheduler.schedule(graph, execution_time);

    EXPECT_TRUE(graph.task_has_status(test_task, ft::TaskStatus::NotStarted));

    // Wait for task execution by joining workers
    scheduler.join_workers();

    EXPECT_TRUE(task_executed);
    EXPECT_TRUE(graph.task_has_status(test_task, ft::TaskStatus::Completed));
}

TEST(TaskScheduler, TaskSchedulingWithArguments) {
    auto scheduler = ft::TaskScheduler::create().workers(1).build();

    std::atomic<int> result(0);
    const int expected_sum = 42;

    ft::TaskGraph graph("name");
    auto sum_task = graph.register_task("sum_task")
                            .function([&result]() {
                                const int a = 10;
                                const int b = 15;
                                const int c = 17;
                                result.store(a + b + c);
                            })
                            .add();

    graph.build();
    scheduler.schedule(graph);

    // Wait for execution by joining workers
    scheduler.join_workers();

    EXPECT_EQ(result.load(), expected_sum);
    EXPECT_TRUE(graph.task_has_status(sum_task, ft::TaskStatus::Completed));
}

TEST(TaskScheduler, TaskCategoryAssignment) {
    // Create workers for specific categories
    ft::FlatMap<ft::TaskCategory, std::size_t> category_assignment;
    category_assignment[ft::TaskCategory{ft::BuiltinTaskCategory::HighPriority}] = 1;
    category_assignment[ft::TaskCategory{ft::BuiltinTaskCategory::Compute}] = 1;
    category_assignment[ft::TaskCategory{ft::BuiltinTaskCategory::IO}] = 1;

    const auto workers_config = ft::WorkersConfig::create_for_categories(category_assignment);
    auto scheduler = ft::TaskScheduler::create().workers(workers_config).build();

    std::atomic<int> task_counter(0);

    // Schedule tasks in different categories
    const std::vector<ft::TaskCategory> categories = {
            ft::TaskCategory{ft::BuiltinTaskCategory::HighPriority},
            ft::TaskCategory{ft::BuiltinTaskCategory::Compute},
            ft::TaskCategory{ft::BuiltinTaskCategory::IO}};

    ft::TaskGraph graph("name");
    std::vector<std::string> task_names;

    for (std::size_t i = 0; i < categories.size(); ++i) {
        auto task_name = graph.register_task(std::format("category_task_{}", i))
                                 .category(categories[i])
                                 .function([&task_counter]() { task_counter.fetch_add(1); })
                                 .add();
        task_names.push_back(task_name);
    }

    graph.build();
    scheduler.schedule(graph);

    // Wait for all tasks to complete by joining workers
    scheduler.join_workers();

    EXPECT_EQ(task_counter.load(), 3);

    for (const auto &task_name : task_names) {
        EXPECT_TRUE(graph.task_has_status(task_name, ft::TaskStatus::Completed));
    }
}

TEST(TaskScheduler, TaskDependencyChain) {
    auto scheduler = ft::TaskScheduler::create().workers(2).build();

    std::atomic<int> execution_order(0);
    std::atomic<int> task1_order(0);
    std::atomic<int> task2_order(0);

    ft::TaskGraph graph("name");

    // Schedule parent task
    auto parent_task = graph.register_task("parent_task")
                               .function([&execution_order, &task1_order]() {
                                   task1_order.store(execution_order.fetch_add(1));
                                   std::this_thread::sleep_for(5ms); // Brief delay
                               })
                               .add();

    // Schedule dependent task
    auto child_task = graph.register_task("child_task")
                              .depends_on(parent_task) // Depends on parent
                              .function([&execution_order, &task2_order]() {
                                  task2_order.store(execution_order.fetch_add(1));
                              })
                              .add();

    graph.build();
    scheduler.schedule(graph);

    // Wait for completion by joining workers
    scheduler.join_workers();

    EXPECT_TRUE(graph.task_has_status(parent_task, ft::TaskStatus::Completed));
    EXPECT_TRUE(graph.task_has_status(child_task, ft::TaskStatus::Completed));

    // Child should execute after parent
    EXPECT_LT(task1_order.load(), task2_order.load());
}

TEST(TaskScheduler, TaskWithException) {
    auto scheduler = ft::TaskScheduler::create().workers(1).build();

    ft::TaskGraph graph("name");
    auto exception_task_name =
            graph.register_task("exception_task")
                    .function([]() { throw std::runtime_error("Test exception"); })
                    .add();

    graph.build();
    scheduler.schedule(graph);

    // Wait for execution to complete by joining workers
    scheduler.join_workers();

    EXPECT_TRUE(graph.task_has_status(exception_task_name, ft::TaskStatus::Failed));
}

TEST(TaskScheduler, ConcurrentTaskScheduling) {
    // Create workers that can handle all categories
    std::vector<ft::WorkerConfig> worker_configs;
    const std::vector<ft::TaskCategory> all_categories = {
            ft::TaskCategory{ft::BuiltinTaskCategory::Default},
            ft::TaskCategory{ft::BuiltinTaskCategory::HighPriority},
            ft::TaskCategory{ft::BuiltinTaskCategory::LowPriority},
            ft::TaskCategory{ft::BuiltinTaskCategory::IO},
            ft::TaskCategory{ft::BuiltinTaskCategory::Compute},
            ft::TaskCategory{ft::BuiltinTaskCategory::Network},
            ft::TaskCategory{ft::BuiltinTaskCategory::Message}};

    worker_configs.reserve(4);
    for (int i = 0; i < 4; ++i) {
        worker_configs.push_back(ft::WorkerConfig::create_for_categories(all_categories));
    }

    auto scheduler = ft::TaskScheduler::create().workers(ft::WorkersConfig{worker_configs}).build();

    constexpr int NUM_TASKS = 50;
    std::atomic<int> completed_tasks(0);

    ft::TaskGraph graph("name");
    std::vector<std::string> task_names;

    // Schedule many tasks concurrently
    for (int i = 0; i < NUM_TASKS; ++i) {
        auto task_name =
                graph.register_task(std::format("concurrent_task_{}", i))
                        .category(static_cast<ft::BuiltinTaskCategory>(
                                i % 7)) // Distribute across all 7 built-in categories
                        .function([&completed_tasks, i]() {
                            completed_tasks.fetch_add(1);
                            // Small variable delay to create realistic load
                            std::this_thread::sleep_for(std::chrono::microseconds(10 + (i % 20)));
                        })
                        .add();
        task_names.push_back(task_name);
    }

    graph.build();
    scheduler.schedule(graph);

    // Wait for all tasks to complete (longer timeout for concurrent execution)
    std::this_thread::sleep_for(200ms);

    EXPECT_EQ(completed_tasks.load(), NUM_TASKS);

    // Verify all tasks completed successfully
    for (const auto &task_name : task_names) {
        EXPECT_TRUE(graph.task_has_status(task_name, ft::TaskStatus::Completed));
    }
}

TEST(TaskScheduler, ParentTaskCompletion) {
    auto scheduler = ft::TaskScheduler::create().workers(2).build();

    std::atomic<bool> parent_completed(false);
    std::atomic<bool> child_executed(false);

    ft::TaskGraph graph("name");

    // Schedule parent task that takes some time
    auto parent_task = graph.register_task("parent_task")
                               .function([&parent_completed]() {
                                   std::this_thread::sleep_for(20ms);
                                   parent_completed.store(true);
                               })
                               .add();

    // Schedule child task immediately but dependent on parent
    auto child_task = graph.register_task("child_task")
                              .depends_on(parent_task) // Depends on parent completion
                              .function([&child_executed, &parent_completed]() {
                                  // Child should only execute after parent completes
                                  EXPECT_TRUE(parent_completed.load())
                                          << "Child executed before parent completed";
                                  child_executed.store(true);
                              })
                              .add();

    graph.build();
    scheduler.schedule(graph);

    // Wait for both to complete
    std::this_thread::sleep_for(50ms);

    EXPECT_TRUE(graph.task_has_status(parent_task, ft::TaskStatus::Completed));
    EXPECT_TRUE(graph.task_has_status(child_task, ft::TaskStatus::Completed));
    EXPECT_TRUE(parent_completed.load());
    EXPECT_TRUE(child_executed.load());
}

TEST(TaskScheduler, ParentTaskFailurePropagation) {
    auto scheduler = ft::TaskScheduler::create().workers(2).build();

    std::atomic<bool> child_executed(false);

    ft::TaskGraph graph("name");

    // Schedule parent task that will fail
    auto parent_task_name =
            graph.register_task("failing_parent")
                    .function([]() { throw std::runtime_error("Parent task failed"); })
                    .add();

    // Schedule child task dependent on parent
    auto child_task_name = graph.register_task("dependent_child")
                                   .depends_on(parent_task_name)
                                   .function([&child_executed]() { child_executed.store(true); })
                                   .add();

    graph.build();
    scheduler.schedule(graph);

    // Wait for execution to complete
    scheduler.join_workers();

    // Parent should fail, child should be cancelled due to parent failure
    EXPECT_TRUE(graph.task_has_status(parent_task_name, ft::TaskStatus::Failed));
    EXPECT_TRUE(graph.task_has_status(child_task_name, ft::TaskStatus::Cancelled));
    EXPECT_FALSE(child_executed.load()) << "Child should not execute when parent fails";
}

TEST(TaskScheduler, MultipleChildrenSameParent) {
    auto scheduler = ft::TaskScheduler::create().workers(3).build();

    std::atomic<int> children_executed(0);
    std::atomic<bool> parent_done(false);

    ft::TaskGraph graph("name");

    // Schedule parent task
    auto parent_task_name = graph.register_task("shared_parent")
                                    .function([&parent_done]() {
                                        std::this_thread::sleep_for(15ms);
                                        parent_done.store(true);
                                    })
                                    .add();

    // Schedule multiple children depending on same parent
    std::vector<std::string> child_names;
    for (int i = 0; i < 5; ++i) {
        auto child_name = graph.register_task(std::format("child_{}", i))
                                  .depends_on(parent_task_name)
                                  .function([&children_executed, &parent_done, i]() {
                                      EXPECT_TRUE(parent_done.load())
                                              << "Child " << i << " executed before parent";
                                      children_executed.fetch_add(1);
                                  })
                                  .add();
        child_names.push_back(child_name);
    }

    graph.build();
    scheduler.schedule(graph);

    // Wait for all to complete
    scheduler.join_workers();

    EXPECT_TRUE(graph.task_has_status("shared_parent", ft::TaskStatus::Completed));
    EXPECT_EQ(children_executed.load(), 5);

    // All children should complete
    for (int i = 0; i < 5; ++i) {
        EXPECT_TRUE(graph.task_has_status(std::format("child_{}", i), ft::TaskStatus::Completed));
    }
}

TEST(TaskScheduler, DependencyChainLong) {
    auto scheduler = ft::TaskScheduler::create().workers(2).build();

    std::atomic<int> execution_order(0);
    std::vector<std::atomic<int>> task_orders(5);
    for (auto &order : task_orders) {
        order.store(-1); // Initialize to -1 (not executed)
    }

    // Create chain:
    // clang-format off
  // task0 -> task1 -> task2 -> task3 -> task4
    // clang-format on
    ft::TaskGraph graph("name");
    std::string previous_task_name;
    std::vector<std::string> task_names;

    for (int i = 0; i < 5; ++i) {
        std::vector<std::string_view> deps;
        if (!previous_task_name.empty()) {
            deps.emplace_back(previous_task_name);
        }
        auto task_name = graph.register_task(std::format("chain_task_{}", i))
                                 .category(ft::BuiltinTaskCategory::Default)
                                 .function([&execution_order, &task_orders, i]() {
                                     task_orders[static_cast<std::size_t>(i)].store(
                                             execution_order.fetch_add(1));
                                     std::this_thread::sleep_for(2ms); // Small delay
                                 })
                                 .depends_on(deps)
                                 .add();

        task_names.push_back(task_name);
        previous_task_name = task_name; // Next task will depend on this one
    }

    graph.build();
    scheduler.schedule(graph);

    // Wait for chain to complete
    scheduler.join_workers();

    // All tasks should complete successfully
    for (const auto &task_name : task_names) {
        EXPECT_TRUE(graph.task_has_status(task_name, ft::TaskStatus::Completed));
    }

    // Verify execution order
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(task_orders[static_cast<std::size_t>(i)].load(), i)
                << "Task " << i << " executed out of order";
    }
}

TEST(TaskScheduler, ParentStatusEdgeCases) {
    auto scheduler = ft::TaskScheduler::create().workers(2).build();

    std::atomic<bool> child_executed(false);
    std::atomic<bool> cancelled_child_executed(false);

    // Create first graph with a parent that completes and a child
    ft::TaskGraph completed_graph("completed_graph");

    auto completed_parent_name = completed_graph.register_task("completed_parent")
                                         .function([]() {
                                             // This parent completes successfully
                                         })
                                         .add();

    auto child_of_completed_name =
            completed_graph.register_task("child_of_completed")
                    .depends_on(completed_parent_name)
                    .function([&child_executed]() { child_executed.store(true); })
                    .add();

    // Create second graph with a parent that fails and a child
    ft::TaskGraph cancelled_graph("cancelled_graph");

    auto cancelled_parent_name = cancelled_graph.register_task("cancelled_parent")
                                         .function([]() {
                                             // This parent fails
                                             throw std::runtime_error("Parent cancelled");
                                         })
                                         .add();

    auto child_of_cancelled_name = cancelled_graph.register_task("child_of_cancelled")
                                           .depends_on(cancelled_parent_name)
                                           .function([&cancelled_child_executed]() {
                                               cancelled_child_executed.store(true);
                                           })
                                           .add();

    completed_graph.build();
    cancelled_graph.build();

    scheduler.schedule(completed_graph);
    scheduler.schedule(cancelled_graph);

    // Wait for execution to complete
    scheduler.join_workers();

    // Child with completed parent should execute
    EXPECT_TRUE(
            completed_graph.task_has_status(child_of_completed_name, ft::TaskStatus::Completed));
    EXPECT_TRUE(child_executed.load());

    // Child with cancelled parent should be cancelled
    EXPECT_TRUE(
            cancelled_graph.task_has_status(child_of_cancelled_name, ft::TaskStatus::Cancelled));
    EXPECT_FALSE(cancelled_child_executed.load());
}

TEST(TaskScheduler, ComplexDependencyGraph) {
    auto scheduler = ft::TaskScheduler::create().workers(4).build();

    std::atomic<int> execution_count(0);

    // Create dependency graph:
    // clang-format off
  /*     A
   *   /   \
   *  B     C
   *   \   /
   *     D
   */
    // clang-format on

    ft::TaskGraph graph("name");

    auto task_a_name = graph.register_task("taskA")
                               .function([&execution_count]() {
                                   execution_count.fetch_add(1);
                                   std::this_thread::sleep_for(10ms);
                               })
                               .add();

    auto task_b_name = graph.register_task("taskB")
                               .depends_on(task_a_name) // Depends on A
                               .function([&execution_count]() {
                                   execution_count.fetch_add(1);
                                   std::this_thread::sleep_for(10ms);
                               })
                               .add();

    auto task_c_name = graph.register_task("taskC")
                               .depends_on(task_a_name) // Depends on A
                               .function([&execution_count]() {
                                   execution_count.fetch_add(1);
                                   std::this_thread::sleep_for(10ms);
                               })
                               .add();

    // Now TaskScheduler supports multiple parents, so D depends on both B and C
    auto task_d_name = graph.register_task("taskD")
                               .depends_on({task_b_name, task_c_name}) // Depends on both B and C
                               .function([&execution_count]() { execution_count.fetch_add(1); })
                               .add();

    graph.build();
    scheduler.schedule(graph);

    // Wait for all tasks
    scheduler.join_workers();

    // All should complete
    EXPECT_TRUE(graph.task_has_status(task_a_name, ft::TaskStatus::Completed));
    EXPECT_TRUE(graph.task_has_status(task_b_name, ft::TaskStatus::Completed));
    EXPECT_TRUE(graph.task_has_status(task_c_name, ft::TaskStatus::Completed));
    EXPECT_TRUE(graph.task_has_status(task_d_name, ft::TaskStatus::Completed));
    EXPECT_EQ(execution_count.load(), 4);
}

TEST(TaskScheduler, TimingPrecision) {
    auto scheduler = ft::TaskScheduler::create().workers(1).build();

    std::atomic<ft::Nanos> execution_time(0ns);

    ft::TaskGraph graph("name");
    auto timing_task_name =
            graph.register_task("timing_task")
                    .function([&execution_time]() { execution_time.store(ft::Time::now_ns()); })
                    .add();

    graph.build();

    const auto start_time = ft::Time::now_ns();
    const auto scheduled_delay = 2ms;
    scheduler.schedule(graph, start_time + scheduled_delay);

    // Wait for execution
    scheduler.join_workers();

    EXPECT_TRUE(graph.task_has_status(timing_task_name, ft::TaskStatus::Completed));

    const ft::Nanos actual_delay = execution_time.load() - start_time;

    // Should execute at roughly the right time (within reasonable tolerance for
    // system scheduling)
    EXPECT_GE(actual_delay.count(), scheduled_delay.count());
    EXPECT_LT(actual_delay.count(),
              scheduled_delay.count() + 10000000); // +10ms tolerance
}

TEST(TaskScheduler, MultipleParentsExecution) {
    auto scheduler = ft::TaskScheduler::create().workers(4).build();

    std::atomic<int> execution_order(0);
    std::vector<std::atomic<int>> task_orders(4);
    for (auto &order : task_orders) {
        order.store(-1); // Initialize to -1 (not executed)
    }

    ft::TaskGraph graph("name");

    // Create two independent parent tasks
    auto parent1_name = graph.register_task("parent1")
                                .function([&execution_order, &task_orders]() {
                                    task_orders[0].store(execution_order.fetch_add(1));
                                    std::this_thread::sleep_for(10ms);
                                })
                                .add();

    auto parent2_name = graph.register_task("parent2")
                                .function([&execution_order, &task_orders]() {
                                    task_orders[1].store(execution_order.fetch_add(1));
                                    std::this_thread::sleep_for(15ms);
                                })
                                .add();

    // Create child task that depends on BOTH parents
    auto child_name = graph.register_task("child_multi_parent")
                              .depends_on({parent1_name, parent2_name}) // Depends on BOTH parents
                              .function([&execution_order, &task_orders]() {
                                  task_orders[2].store(execution_order.fetch_add(1));
                              })
                              .add();

    // Create another child that depends on the multi-parent child
    auto grandchild_name = graph.register_task("grandchild")
                                   .depends_on(child_name) // Depends on child
                                   .function([&execution_order, &task_orders]() {
                                       task_orders[3].store(execution_order.fetch_add(1));
                                   })
                                   .add();

    graph.build();
    scheduler.schedule(graph);

    // Wait for all tasks to complete
    scheduler.join_workers();

    // Verify all tasks completed successfully
    EXPECT_TRUE(graph.task_has_status(parent1_name, ft::TaskStatus::Completed));
    EXPECT_TRUE(graph.task_has_status(parent2_name, ft::TaskStatus::Completed));
    EXPECT_TRUE(graph.task_has_status(child_name, ft::TaskStatus::Completed));
    EXPECT_TRUE(graph.task_has_status(grandchild_name, ft::TaskStatus::Completed));

    // Verify execution order: both parents should execute before child
    // Child should execute before grandchild
    const int parent1_order = task_orders[0].load();
    const int parent2_order = task_orders[1].load();
    const int child_order = task_orders[2].load();
    const int grandchild_order = task_orders[3].load();

    EXPECT_GE(parent1_order, 0) << "Parent1 should have executed";
    EXPECT_GE(parent2_order, 0) << "Parent2 should have executed";
    EXPECT_GT(child_order, parent1_order) << "Child should execute after parent1";
    EXPECT_GT(child_order, parent2_order) << "Child should execute after parent2";
    EXPECT_GT(grandchild_order, child_order) << "Grandchild should execute after child";
}

TEST(TaskScheduler, MultipleParentsWithFailure) {
    auto scheduler = ft::TaskScheduler::create().workers(3).build();

    std::atomic<bool> child_executed(false);
    std::atomic<bool> good_parent_executed(false);

    ft::TaskGraph graph("name");

    // Create one parent that will succeed
    auto good_parent_name = graph.register_task("good_parent")
                                    .function([&good_parent_executed]() {
                                        std::this_thread::sleep_for(5ms);
                                        good_parent_executed.store(true);
                                    })
                                    .add();

    // Create one parent that will fail
    auto bad_parent_name = graph.register_task("bad_parent")
                                   .function([]() { throw std::runtime_error("Parent failed"); })
                                   .add();

    // Create child that depends on both parents
    auto child_name = graph.register_task("child_with_failing_parent")
                              .depends_on({good_parent_name, bad_parent_name}) // Depends on both
                              .function([&child_executed]() { child_executed.store(true); })
                              .add();

    graph.build();
    scheduler.schedule(graph);

    // Wait for execution to complete
    scheduler.join_workers();

    // Good parent should complete, bad parent should fail
    EXPECT_TRUE(graph.task_has_status(good_parent_name, ft::TaskStatus::Completed));
    EXPECT_TRUE(graph.task_has_status(bad_parent_name, ft::TaskStatus::Failed));

    // Child should be cancelled because one parent failed
    EXPECT_TRUE(graph.task_has_status(child_name, ft::TaskStatus::Cancelled));
    EXPECT_FALSE(child_executed.load()) << "Child should not execute when any parent fails";
    EXPECT_TRUE(good_parent_executed.load()) << "Good parent should have executed";
}

/**
 * Test task execution behavior when tasks are scheduled in the past (late
 * tasks)
 */
TEST(TaskScheduler, LateTaskExecution) {
    auto scheduler = ft::TaskScheduler::create().build();

    // Get current time and schedule tasks in the past
    const auto now = ft::Time::now_ns();
    const auto past_time = now - 100ms; // 100ms ago
    const auto recent_past = now - 1ms; // 1ms ago (within threshold)
    const auto far_past = now - 1s;     // 1 second ago (very late)

    std::atomic<int> execution_count{0};
    std::atomic<bool> past_task_executed{false};
    std::atomic<bool> recent_past_executed{false};
    std::atomic<bool> far_past_executed{false};

    // Create three separate graphs for tasks scheduled at different times
    ft::TaskGraph past_graph("past_graph");
    auto past_task_name = past_graph.register_task("past_task")
                                  .function([&]() {
                                      past_task_executed = true;
                                      execution_count++;
                                  })
                                  .add();

    ft::TaskGraph recent_graph("recent_graph");
    auto recent_task_name = recent_graph.register_task("recent_past_task")
                                    .function([&]() {
                                        recent_past_executed = true;
                                        execution_count++;
                                    })
                                    .add();

    ft::TaskGraph far_graph("far_graph");
    auto far_task_name = far_graph.register_task("far_past_task")
                                 .function([&]() {
                                     far_past_executed = true;
                                     execution_count++;
                                 })
                                 .add();

    // Schedule tasks in the past
    past_graph.build();
    recent_graph.build();
    far_graph.build();

    scheduler.schedule(past_graph, past_time);
    scheduler.schedule(recent_graph, recent_past);
    scheduler.schedule(far_graph, far_past);

    // Wait for all tasks to complete
    const auto deadline = ft::Time::now_ns() + 100ms; // 100ms timeout
    while (execution_count.load() < 3 && ft::Time::now_ns() < deadline) {
        std::this_thread::sleep_for(1ms);
    }

    // Verify all late tasks executed
    EXPECT_TRUE(past_task_executed.load()) << "Task scheduled 100ms in past should execute";
    EXPECT_TRUE(recent_past_executed.load()) << "Task scheduled 1ms in past should execute";
    EXPECT_TRUE(far_past_executed.load()) << "Task scheduled 1s in past should execute";
    EXPECT_EQ(execution_count.load(), 3) << "All late tasks should execute";

    // Verify final statuses
    EXPECT_TRUE(past_graph.task_has_status(past_task_name, ft::TaskStatus::Completed));
    EXPECT_TRUE(recent_graph.task_has_status(recent_task_name, ft::TaskStatus::Completed));
    EXPECT_TRUE(far_graph.task_has_status(far_task_name, ft::TaskStatus::Completed));
}

/**
 * Test that tasks scheduled slightly before current time (within threshold)
 * execute immediately
 */
TEST(TaskScheduler, EarlyTaskWithinThreshold) {
    auto scheduler = ft::TaskScheduler::create().build();

    // Schedule a task 500μs in the future (within default 1ms threshold)
    const auto now = ft::Time::now_ns();
    const auto near_future = now + 500us; // 500μs ahead

    std::atomic<bool> task_executed{false};
    std::chrono::steady_clock::time_point execution_start;

    ft::TaskGraph graph("name");
    auto early_task_name = graph.register_task("early_task")
                                   .function([&]() {
                                       execution_start = std::chrono::steady_clock::now();
                                       task_executed = true;
                                   })
                                   .add();

    graph.build();
    scheduler.schedule(graph, near_future);

    // Record when we expect the task to be ready
    const auto test_start = std::chrono::steady_clock::now();

    // Wait for task completion
    const auto deadline = ft::Time::now_ns() + 50ms; // 50ms timeout
    while (!task_executed.load() && ft::Time::now_ns() < deadline) {
        std::this_thread::sleep_for(100us);
    }

    EXPECT_TRUE(task_executed.load()) << "Task within threshold should execute early";
    EXPECT_TRUE(graph.task_has_status(early_task_name, ft::TaskStatus::Completed));

    // Task should execute quickly since it's within threshold
    const auto execution_delay = execution_start - test_start;
    const auto delay_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(execution_delay).count();
    EXPECT_LT(delay_ms, 10) << "Task within threshold should execute within 10ms";
}

TEST(TaskScheduler, DependencyGenerationOrdering) {
    auto scheduler = ft::TaskScheduler::create().workers(2).build();

    std::atomic<int> execution_order(0);
    std::vector<std::atomic<int>> task_orders(5);
    for (auto &order : task_orders) {
        order.store(-1); // Initialize to -1 (not executed)
    }

    ft::TaskGraph graph("name");

    // clang-format off
  // Create a complex dependency graph to test generation-based ordering:
  /*     A (gen 0)
   *   /   \
   *  B     C (gen 1)
   *  |     |
   *  D     E (gen 2)
   *   \   /
   *    F   (gen 3)
   */
    // clang-format on

    auto task_a_name = graph.register_task("taskA")
                               .function([&execution_order, &task_orders]() {
                                   task_orders[0].store(execution_order.fetch_add(1));
                                   std::this_thread::sleep_for(5ms);
                                   return ft::TaskResult{};
                               })
                               .add();

    auto task_b_name = graph.register_task("taskB")
                               .depends_on(task_a_name)
                               .function([&execution_order, &task_orders]() {
                                   task_orders[1].store(execution_order.fetch_add(1));
                                   std::this_thread::sleep_for(5ms);
                                   return ft::TaskResult{};
                               })
                               .add();

    auto task_c_name = graph.register_task("taskC")
                               .depends_on(task_a_name)
                               .function([&execution_order, &task_orders]() {
                                   task_orders[2].store(execution_order.fetch_add(1));
                                   std::this_thread::sleep_for(5ms);
                                   return ft::TaskResult{};
                               })
                               .add();

    auto task_d_name = graph.register_task("taskD")
                               .depends_on(task_b_name)
                               .function([&execution_order, &task_orders]() {
                                   task_orders[3].store(execution_order.fetch_add(1));
                                   std::this_thread::sleep_for(5ms);
                                   return ft::TaskResult{};
                               })
                               .add();

    auto task_e_name = graph.register_task("taskE")
                               .depends_on(task_c_name)
                               .function([&execution_order, &task_orders]() {
                                   task_orders[4].store(execution_order.fetch_add(1));
                                   std::this_thread::sleep_for(5ms);
                                   return ft::TaskResult{};
                               })
                               .add();

    auto task_f_name =
            graph.register_task("taskF")
                    .depends_on({task_d_name, task_e_name})
                    .function([&execution_order]() {
                        execution_order.fetch_add(
                                1); // Don't care about F's exact order, just that it's last
                        return ft::TaskResult{};
                    })
                    .add();

    graph.build();

    // Verify dependency generations are calculated correctly
    auto &sched_tasks = graph.prepare_tasks(ft::Time::now_ns());

    for (const auto &task : sched_tasks) {
        if (task->get_task_name() == "taskA") {
            EXPECT_EQ(task->get_dependency_generation(), 0) << "A should have generation 0";
        } else if (task->get_task_name() == "taskB" || task->get_task_name() == "taskC") {
            EXPECT_EQ(task->get_dependency_generation(), 1) << "B and C should have generation 1";
        } else if (task->get_task_name() == "taskD" || task->get_task_name() == "taskE") {
            EXPECT_EQ(task->get_dependency_generation(), 2) << "D and E should have generation 2";
        } else if (task->get_task_name() == "taskF") {
            EXPECT_EQ(task->get_dependency_generation(), 3) << "F should have generation 3";
        }
    }

    // Schedule all tasks at the same time - dependency generation should ensure
    // proper ordering
    scheduler.schedule(graph);

    // Wait for all tasks to complete
    scheduler.join_workers();

    // Verify dependency-based execution order
    const int order_a = task_orders[0].load();
    const int order_b = task_orders[1].load();
    const int order_c = task_orders[2].load();
    const int order_d = task_orders[3].load();
    const int order_e = task_orders[4].load();

    // A must execute first (generation 0)
    EXPECT_GE(order_a, 0) << "Task A should execute";

    // B and C must execute after A (generation 1)
    EXPECT_GT(order_b, order_a) << "Task B should execute after A";
    EXPECT_GT(order_c, order_a) << "Task C should execute after A";

    // D must execute after B (generation 2)
    EXPECT_GT(order_d, order_b) << "Task D should execute after B";

    // E must execute after C (generation 2)
    EXPECT_GT(order_e, order_c) << "Task E should execute after C";

    // Verify all task statuses
    EXPECT_TRUE(graph.task_has_status(task_a_name, ft::TaskStatus::Completed));
    EXPECT_TRUE(graph.task_has_status(task_b_name, ft::TaskStatus::Completed));
    EXPECT_TRUE(graph.task_has_status(task_c_name, ft::TaskStatus::Completed));
    EXPECT_TRUE(graph.task_has_status(task_d_name, ft::TaskStatus::Completed));
    EXPECT_TRUE(graph.task_has_status(task_e_name, ft::TaskStatus::Completed));
    EXPECT_TRUE(graph.task_has_status(task_f_name, ft::TaskStatus::Completed));
}

TEST(TaskScheduler, WorkerShutdownBehaviors) {
    // Test CancelPendingTasks behavior
    {
        auto scheduler = ft::TaskScheduler::create().workers(2).build();
        std::atomic<int> tasks_executed{0};

        ft::TaskGraph graph("name");

        // Create tasks that would take time to execute
        for (int i = 0; i < 5; ++i) {
            graph.register_task(std::format("cancel_task_{}", i))
                    .function([&tasks_executed]() {
                        tasks_executed.fetch_add(1);
                        std::this_thread::sleep_for(10ms);
                    })
                    .add();
        }

        graph.build();

        // Schedule tasks immediately
        scheduler.schedule(graph);

        // Wait very briefly, then cancel pending tasks
        std::this_thread::sleep_for(1ms);

        // Join with cancelling pending tasks - should not wait for completion
        scheduler.join_workers(ft::WorkerShutdownBehavior::CancelPendingTasks);

        // Most tasks should have been cancelled before execution
        EXPECT_LT(tasks_executed.load(), 5)
                << "Some tasks should have been cancelled before execution";
    }

    // Test FinishPendingTasks behavior (default)
    {
        auto scheduler = ft::TaskScheduler::create().workers(2).build();
        std::atomic<int> graceful_tasks_executed{0};

        ft::TaskGraph graph2("graph2");
        for (int i = 0; i < 3; ++i) {
            graph2.register_task(std::format("graceful_task_{}", i))
                    .function(
                            [&graceful_tasks_executed]() { graceful_tasks_executed.fetch_add(1); })
                    .add();
        }

        graph2.build();
        scheduler.schedule(graph2);

        // Join with default behavior (finish pending tasks)
        scheduler.join_workers(); // Uses default FinishPendingTasks

        // All tasks should complete
        EXPECT_EQ(graceful_tasks_executed.load(), 3);
        for (int i = 0; i < 3; ++i) {
            EXPECT_TRUE(graph2.task_has_status(
                    std::format("graceful_task_{}", i), ft::TaskStatus::Completed))
                    << "Task graceful_task_" << i << " should have completed gracefully";
        }
    }
}

TEST(TaskScheduler, StressTestMultipleWorkersWithDependencies) {
    // Create workers with core pinning if hardware allows
    const std::uint32_t max_cores = std::thread::hardware_concurrency();
    static constexpr std::uint32_t TRIGGER_CORE = 30;
    constexpr std::uint32_t CORE_OFFSET = TRIGGER_CORE + 1;
    constexpr std::uint32_t TASKS_PER_GRAPH = 5;
    constexpr std::uint32_t NUM_WORKERS = TASKS_PER_GRAPH;
    static constexpr int RT_PRIORITY = 95;

    std::vector<ft::WorkerConfig> worker_configs;
    const std::vector<ft::TaskCategory> all_categories = {
            ft::TaskCategory{ft::BuiltinTaskCategory::Default},
            ft::TaskCategory{ft::BuiltinTaskCategory::HighPriority},
            ft::TaskCategory{ft::BuiltinTaskCategory::LowPriority},
            ft::TaskCategory{ft::BuiltinTaskCategory::IO},
            ft::TaskCategory{ft::BuiltinTaskCategory::Compute},
            ft::TaskCategory{ft::BuiltinTaskCategory::Network},
            ft::TaskCategory{ft::BuiltinTaskCategory::Message}};

    // Only use core pinning if we have enough cores
    const bool enable_pinning = (CORE_OFFSET + NUM_WORKERS) <= max_cores;

    for (std::uint32_t i = 0; i < NUM_WORKERS; ++i) {
        if (enable_pinning) {
            worker_configs.push_back(ft::WorkerConfig::create_pinned_rt(
                    i + CORE_OFFSET, RT_PRIORITY, all_categories));
        } else {
            worker_configs.push_back(ft::WorkerConfig::create_rt_only(RT_PRIORITY, all_categories));
        }
    }

    static constexpr std::uint32_t MONITOR_CORE = 0;
    auto scheduler_builder = ft::TaskScheduler::create()
                                     .workers(ft::WorkersConfig{worker_configs})
                                     .max_tasks_per_category(TASKS_PER_GRAPH * 8U)
                                     .monitor_core(MONITOR_CORE);

    auto scheduler = scheduler_builder.build();

    std::atomic<uint32_t> total_task_executions{0};
    uint32_t total_graphs_scheduled{0};

    // Create simple task graph: 3 roots -> 1 child -> 1 grandchild (5 total
    // tasks). Use 80x pool capacity multiplier to handle peak concurrent load
    // during stress test with potential task buildup from scheduling variations
    static constexpr std::size_t POOL_CAPACITY_MULTIPLIER = 80;
    ft::TaskGraph graph("name", POOL_CAPACITY_MULTIPLIER);
    std::vector<std::string> root_names;

    // Create 3 root tasks (no dependencies)
    const std::vector<ft::TaskCategory> root_categories = {
            ft::TaskCategory{ft::BuiltinTaskCategory::Default},
            ft::TaskCategory{ft::BuiltinTaskCategory::IO},
            ft::TaskCategory{ft::BuiltinTaskCategory::Compute}};

    constexpr auto NUM_CHILDREN = 2;
    for (std::size_t i = 0; i < TASKS_PER_GRAPH - NUM_CHILDREN; ++i) {
        auto root_name =
                graph.register_task(std::format("root_{}", i + 1))
                        .category(root_categories[i])
                        .function([&total_task_executions, delay_us = 100 + 50 * i]() {
                            total_task_executions.fetch_add(1);
                            std::this_thread::sleep_for(std::chrono::microseconds(delay_us));
                        })
                        .add();
        root_names.push_back(root_name);
    }

    // Create child task that depends on all roots
    auto child =
            graph.register_task("child")
                    .category(ft::TaskCategory{ft::BuiltinTaskCategory::HighPriority})
                    .depends_on(std::vector<std::string_view>(root_names.begin(), root_names.end()))
                    .function([&total_task_executions]() {
                        total_task_executions.fetch_add(1);
                        std::this_thread::sleep_for(100us);
                    })
                    .add();

    // Create grandchild task that depends on child
    graph.register_task("grandchild")
            .category(ft::BuiltinTaskCategory::LowPriority)
            .depends_on(child)
            .function([&total_task_executions]() {
                total_task_executions.fetch_add(1);
                std::this_thread::sleep_for(50us);
            })
            .add();

    graph.build();

    const auto test_duration_seconds = std::chrono::seconds(3);
    static constexpr auto SCHEDULE_INTERVAL_MS = 2;

    // Create callback for scheduling graphs at regular intervals
    auto schedule_callback = [&scheduler, &graph, &total_graphs_scheduled]() {
        const auto execution_start = ft::Time::now_ns() + 100us;
        scheduler.schedule(graph, execution_start);
        ++total_graphs_scheduled;
    };

    // Create TimedTrigger with scheduling callback
    auto trigger = ft::TimedTrigger::create(
                           schedule_callback, std::chrono::milliseconds{SCHEDULE_INTERVAL_MS})
                           .pin_to_core(TRIGGER_CORE)
                           .with_stats_core(MONITOR_CORE)
                           .with_rt_priority(90) // Lower priority than workers but still RT
                           .build();

    const auto test_start_time = std::chrono::steady_clock::now();

    // Start the trigger to begin scheduling
    EXPECT_TRUE(ft::is_task_success(trigger.start()));

    // Wait for test duration
    std::this_thread::sleep_for(test_duration_seconds);

    // Stop the trigger
    trigger.stop();

    const auto expected_duration_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(test_duration_seconds).count();
    const auto actual_duration = std::chrono::steady_clock::now() - test_start_time;
    const auto duration_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(actual_duration).count();

    // Calculate derived metrics
    const double graphs_per_second = static_cast<double>(total_graphs_scheduled) /
                                     (static_cast<double>(duration_ms) / 1000.0);
    const double tasks_per_second = static_cast<double>(total_task_executions.load()) /
                                    (static_cast<double>(duration_ms) / 1000.0);

    RT_LOGC_DEBUG(ft::TaskLog::TaskScheduler, "Stress Test Results:");
    RT_LOGC_DEBUG(
            ft::TaskLog::TaskScheduler,
            "  Test Duration: {} ms (expected {} ms)",
            duration_ms,
            expected_duration_ms);
    RT_LOGC_DEBUG(
            ft::TaskLog::TaskScheduler,
            "  Total Task Executions: {}",
            total_task_executions.load());
    RT_LOGC_DEBUG(
            ft::TaskLog::TaskScheduler, "  Total Graphs Scheduled: {}", total_graphs_scheduled);
    RT_LOGC_DEBUG(
            ft::TaskLog::TaskScheduler,
            "  Tasks per Graph: {} ({} roots + 1 child + 1 grandchild)",
            TASKS_PER_GRAPH,
            TASKS_PER_GRAPH - NUM_CHILDREN);
    RT_LOGC_DEBUG(ft::TaskLog::TaskScheduler, "  Monitor: core pinned {}", MONITOR_CORE);
    RT_LOGC_DEBUG(
            ft::TaskLog::TaskScheduler,
            "  Trigger: core pinned {}, interval {}ms",
            TRIGGER_CORE,
            SCHEDULE_INTERVAL_MS);
    if (enable_pinning) {
        RT_LOGC_DEBUG(
                ft::TaskLog::TaskScheduler,
                "  Workers: {} (core pinned {}-{}, RT priority {})",
                NUM_WORKERS,
                CORE_OFFSET,
                CORE_OFFSET + NUM_WORKERS - 1,
                RT_PRIORITY);
    } else {
        RT_LOGC_DEBUG(
                ft::TaskLog::TaskScheduler,
                "  Workers: {} (core unpinned, RT priority {})",
                NUM_WORKERS,
                RT_PRIORITY);
    }
    RT_LOGC_DEBUG(ft::TaskLog::TaskScheduler, "  Graphs per Second: {:.1f}", graphs_per_second);
    RT_LOGC_DEBUG(ft::TaskLog::TaskScheduler, "  Tasks per Second: {:.1f}", tasks_per_second);

    scheduler.join_workers();

    // Verify stress test ran for at least 99% of expected duration (allowing some
    // tolerance)
    EXPECT_GE(static_cast<double>(duration_ms), static_cast<double>(expected_duration_ms) * 0.99)
            << "Stress test should run for at least 99% of expected duration";

    const auto expected_graphs_completed =
            test_duration_seconds.count() * 1000 / SCHEDULE_INTERVAL_MS;
    EXPECT_GE(
            static_cast<double>(total_graphs_scheduled),
            static_cast<double>(expected_graphs_completed) * 0.90)
            << "Should execute at least 90% of expected iterations";
    EXPECT_EQ(total_task_executions.load(), total_graphs_scheduled * TASKS_PER_GRAPH);

    scheduler.print_monitor_stats();
    EXPECT_TRUE(ft::is_task_success(scheduler.write_monitor_stats_to_file(
            "stress_test_stats.json", ft::TraceWriteMode::Overwrite)));
    EXPECT_TRUE(ft::is_task_success(
            trigger.write_stats_to_file("stress_test_stats.json", ft::TraceWriteMode::Append)));

    // Write task execution trace first, then append trigger trace
    EXPECT_TRUE(ft::is_task_success(scheduler.write_chrome_trace_to_file(
            "stress_test_trace.json", ft::TraceWriteMode::Overwrite)));
    EXPECT_EQ(
            trigger.write_chrome_trace_to_file(
                    "stress_test_trace.json", ft::TraceWriteMode::Append),
            0);

    graph.clear_scheduled_tasks();

    // Verify no heap allocations occurred - all tasks should come from
    // preallocated pool
    const auto pool_stats = graph.get_pool_stats();
    EXPECT_EQ(pool_stats.new_tasks_created, 0)
            << "No heap allocations should occur during stress test (pool should be "
               "sufficient). "
            << "Pool hits: " << pool_stats.pool_hits
            << ", New tasks created: " << pool_stats.new_tasks_created
            << ", Hit rate: " << pool_stats.hit_rate_percent() << "%";

    // Verify task release tracking - each task should be released after execution
    const auto expected_releases = total_graphs_scheduled * TASKS_PER_GRAPH;
    EXPECT_EQ(pool_stats.tasks_released, expected_releases)
            << "Each task should be released back to pool after execution. "
            << "Expected releases: " << expected_releases
            << ", Actual releases: " << pool_stats.tasks_released
            << " (difference: " << (expected_releases - pool_stats.tasks_released) << ")";
}

TEST(TaskScheduler, ManualWorkerStartup) {
    auto scheduler = ft::TaskScheduler::create()
                             .workers(2)
                             .task_readiness_tolerance(100ns)
                             .no_monitor_pinning()
                             .worker_sleep(10us)
                             .manual_start()
                             .build();

    std::atomic<bool> executed{false};
    ft::TaskGraph graph("name");
    graph.register_task("test")
            .function([&executed]() -> ft::TaskResult {
                executed.store(true);
                return ft::TaskResult{ft::TaskStatus::Completed};
            })
            .add();
    graph.build();

    scheduler.schedule(graph);
    std::this_thread::sleep_for(5ms);
    EXPECT_FALSE(executed.load());

    scheduler.start_workers();
    std::this_thread::sleep_for(10ms);
    EXPECT_TRUE(executed.load());

    scheduler.join_workers();
}

TEST(TaskScheduler, ProcessedTaskCleanup) {
    auto scheduler = ft::TaskScheduler::create()
                             .workers(1)
                             .task_readiness_tolerance(100us)
                             .no_monitor_pinning()
                             .worker_sleep(10us)
                             .manual_start()
                             .build();

    std::atomic<int> execution_count{0};
    ft::TaskGraph graph("name");
    graph.register_task("cancelled")
            .function([&execution_count]() -> ft::TaskResult {
                execution_count.fetch_add(1);
                return ft::TaskResult{ft::TaskStatus::Completed};
            })
            .add();
    graph.register_task("completed")
            .function([&execution_count]() -> ft::TaskResult {
                execution_count.fetch_add(1);
                return ft::TaskResult{ft::TaskStatus::Completed};
            })
            .add();
    graph.register_task("running")
            .function([&execution_count]() -> ft::TaskResult {
                execution_count.fetch_add(1);
                return ft::TaskResult{ft::TaskStatus::Completed};
            })
            .add();
    graph.register_task("failed")
            .function([&execution_count]() -> ft::TaskResult {
                execution_count.fetch_add(1);
                return ft::TaskResult{ft::TaskStatus::Completed};
            })
            .add();
    graph.build();

    // Schedule all tasks for future so they sit in queue
    scheduler.schedule(graph, ft::Time::now_ns() + 10ms); // 10ms future

    // Set each task to processed state while queued
    EXPECT_TRUE(graph.set_task_status("cancelled", ft::TaskStatus::Cancelled));
    EXPECT_TRUE(graph.set_task_status("completed", ft::TaskStatus::Completed));
    EXPECT_TRUE(graph.set_task_status("running", ft::TaskStatus::Running));
    EXPECT_TRUE(graph.set_task_status("failed", ft::TaskStatus::Failed));

    // Start workers - they should find all processed tasks and clean them up
    scheduler.start_workers();
    std::this_thread::sleep_for(50ms);

    EXPECT_EQ(execution_count.load(), 0); // None should execute
    scheduler.join_workers();
    EXPECT_EQ(execution_count.load(), 0); // Still none after joining
}

TEST(TaskScheduler, MultipleSchedulingWithTimeoutsAndPoolValidation) {
    // Create task scheduler with sufficient workers for parallel task execution
    auto scheduler = ft::TaskScheduler::create()
                             .workers(4)
                             .task_readiness_tolerance(100us)
                             .no_monitor_pinning()
                             .worker_sleep(10us)
                             .build();

    std::atomic<int> long_task_executions{0};
    std::atomic<int> short_task_executions{0};
    std::atomic<std::uint64_t> first_scheduling_round{0};
    std::atomic<std::uint64_t> last_scheduling_round{0};

    ft::TaskGraph graph("name");

    // Task that sleeps 30ms but times out after 20ms
    graph.register_task("timeout_task")
            .timeout(20ms)
            .function([&long_task_executions]() -> ft::TaskResult {
                long_task_executions.fetch_add(1);
                std::this_thread::sleep_for(30ms);
                return ft::TaskResult{ft::TaskStatus::Completed};
            })
            .add();

    // Tasks that sleep 15-19ms to allow multiple instances to queue/run
    // simultaneously
    std::vector<std::string> medium_task_names;
    for (int i = 0; i < 3; ++i) {
        auto task_name =
                graph.register_task(std::format("medium_task_{}", i))
                        .timeout(100ms)
                        .function([&short_task_executions, i]() -> ft::TaskResult {
                            short_task_executions.fetch_add(1);
                            std::this_thread::sleep_for(std::chrono::milliseconds(15 + i * 2));
                            return ft::TaskResult{ft::TaskStatus::Completed};
                        })
                        .add();
        medium_task_names.push_back(task_name);
    }

    graph.build();

    // Schedule same graph 5 times with 5ms intervals (tasks take 15-30ms)
    const uint64_t num_scheduling_rounds = 5;
    std::vector<std::uint64_t> scheduling_rounds;

    for (uint64_t round = 0; round < num_scheduling_rounds; ++round) {
        const ft::Nanos execution_time = ft::Time::now_ns() + 1ms;
        const std::uint64_t round_before = graph.get_times_scheduled();

        scheduler.schedule(graph, execution_time);

        const std::uint64_t round_after = graph.get_times_scheduled();
        scheduling_rounds.push_back(round_after);

        if (round == 0) {
            first_scheduling_round.store(round_after);
        }
        if (round == num_scheduling_rounds - 1) {
            last_scheduling_round.store(round_after);
        }

        EXPECT_EQ(round_after, round_before + 1) << "Execution round should increment";
        std::this_thread::sleep_for(5ms);
    }

    std::this_thread::sleep_for(100ms);

    scheduler.join_workers();

    // Verify schedling rounds
    EXPECT_EQ(scheduling_rounds.size(), static_cast<std::size_t>(num_scheduling_rounds));
    for (std::size_t i = 1; i < scheduling_rounds.size(); ++i) {
        EXPECT_EQ(scheduling_rounds[i], scheduling_rounds[i - 1] + 1);
    }

    // Verify multiple task instances executed
    const int total_executions = long_task_executions.load() + short_task_executions.load();
    EXPECT_GE(total_executions, num_scheduling_rounds)
            << "Long: " << long_task_executions.load()
            << ", Medium: " << short_task_executions.load();

    EXPECT_GT(long_task_executions.load(), 0) << "Long tasks should start";
    EXPECT_GT(short_task_executions.load(), 0) << "Medium tasks should complete";

    // Verify task pool cleanup
    graph.clear_scheduled_tasks();
    const auto pool_stats = graph.get_pool_stats();
    const auto expected_releases = num_scheduling_rounds * 4; // 4 tasks per round

    EXPECT_EQ(pool_stats.tasks_released, expected_releases) << "All tasks should return to pool";
    EXPECT_GT(pool_stats.pool_hits, 0U) << "Pool should be reused";

    // Verify no dangling references
    const std::uint64_t total_acquisitions = pool_stats.pool_hits + pool_stats.new_tasks_created;
    EXPECT_EQ(total_acquisitions, pool_stats.tasks_released)
            << "Acquisitions=" << total_acquisitions << ", releases=" << pool_stats.tasks_released;
}

TEST(TaskScheduler, ChromeTraceAppendMode) {
    // Simple test to verify Chrome trace append mode works correctly
    auto task_scheduler = ft::TaskScheduler::create().workers(2).auto_start().build();

    // Create simple task graph
    ft::TaskGraph graph("append_test");
    graph.register_task("test_task").function([]() { std::this_thread::sleep_for(100us); }).add();
    graph.build();

    // Schedule a few tasks
    for (int i = 0; i < 3; ++i) {
        task_scheduler.schedule(graph);
        std::this_thread::sleep_for(10ms);
    }

    task_scheduler.join_workers();

    const std::string chrome_filename = "test_append_chrome_trace.json";
    std::filesystem::remove(chrome_filename);

    // Write task traces first (OVERWRITE mode)
    EXPECT_TRUE(ft::is_task_success(task_scheduler.write_chrome_trace_to_file(
            chrome_filename, ft::TraceWriteMode::Overwrite)));

    // Create a simple trigger for append test
    std::atomic<int> trigger_count{0};
    auto trigger = ft::TimedTrigger::create(
                           [&trigger_count]() { trigger_count++; }, std::chrono::milliseconds{5})
                           .build();

    EXPECT_TRUE(ft::is_task_success(trigger.start()));
    std::this_thread::sleep_for(25ms); // ~5 trigger events
    trigger.stop();

    // Append trigger traces (APPEND mode)
    EXPECT_EQ(trigger.write_chrome_trace_to_file(chrome_filename, ft::TraceWriteMode::Append), 0);

    // Verify the combined file exists and has valid JSON structure
    EXPECT_TRUE(std::filesystem::exists(chrome_filename));

    std::ifstream chrome_file(chrome_filename);
    ASSERT_TRUE(chrome_file.is_open());

    std::stringstream buffer;
    buffer << chrome_file.rdbuf();
    const std::string content = buffer.str();
    chrome_file.close();

    // Basic JSON validation
    EXPECT_NE(content.find(R"({"traceEvents":[)"), std::string::npos);
    EXPECT_NE(content.find("]}"), std::string::npos); // Proper closing

    // Should end with ]}\n (account for trailing newline)
    const std::string expected_ending = "]}\n";
    EXPECT_GE(content.length(), expected_ending.length());
    EXPECT_EQ(content.substr(content.length() - expected_ending.length()), expected_ending);

    // Verify both task and trigger events are present
    EXPECT_NE(content.find(R"("name":"test_task")"), std::string::npos);
    EXPECT_NE(content.find(R"("name":"Trigger_)"), std::string::npos);
    EXPECT_NE(content.find(R"("cat":"append_test")"),
              std::string::npos); // Task category
    EXPECT_NE(content.find(R"("cat":"TimedTrigger")"),
              std::string::npos); // Trigger category

    // Count events - should have task events + trigger events
    std::size_t event_count = 0;
    std::size_t pos = 0;
    while ((pos = content.find(R"("ph":"X")", pos)) != std::string::npos) {
        event_count++;
        pos += 1;
    }
    EXPECT_GE(event_count, 6);  // At least 3 tasks + ~3 triggers
    EXPECT_LE(event_count, 15); // But not too many

    // Clean up
    std::filesystem::remove(chrome_filename);
}

TEST(TaskScheduler, TaskComparatorIgnoresDependencyGenerationAcrossGraphs) {
    auto scheduler = ft::TaskScheduler::create().workers(4).build();

    ft::TaskGraph graph_a("GraphA");
    ft::TaskGraph graph_b("GraphB");

    std::atomic<int> execution_order{0};
    std::unordered_map<std::string, int> task_orders;
    task_orders["ParentA"] = -1;
    task_orders["ChildA"] = -1;
    task_orders["ParentB"] = -1;
    task_orders["ChildB"] = -1;

    auto parent_a = graph_a.register_task("ParentA")
                            .function([&execution_order, &task_orders]() {
                                task_orders["ParentA"] = execution_order.fetch_add(1);
                                std::this_thread::sleep_for(1ms);
                            })
                            .add();

    graph_a.register_task("ChildA")
            .depends_on(parent_a)
            .function([&execution_order, &task_orders]() {
                task_orders["ChildA"] = execution_order.fetch_add(1);
                std::this_thread::sleep_for(1ms);
            })
            .add();

    auto parent_b = graph_b.register_task("ParentB")
                            .function([&execution_order, &task_orders]() {
                                task_orders["ParentB"] = execution_order.fetch_add(1);
                                std::this_thread::sleep_for(1ms);
                            })
                            .add();

    graph_b.register_task("ChildB")
            .depends_on(parent_b)
            .function([&execution_order, &task_orders]() {
                task_orders["ChildB"] = execution_order.fetch_add(1);
                std::this_thread::sleep_for(1ms);
            })
            .add();

    graph_a.build();
    graph_b.build();

    // Schedule GraphA earlier than GraphB to test cross-graph tie-breaking
    const ft::Nanos earlier_time = ft::Time::now_ns() + 1ms;
    const ft::Nanos later_time = earlier_time + 500us;

    scheduler.schedule(graph_a, earlier_time);
    scheduler.schedule(graph_b, later_time);
    scheduler.join_workers();

    // Verify all tasks executed
    for (const auto &[name, order] : task_orders) {
        EXPECT_NE(order, -1) << "Task " << name << " should have executed";
    }

    // Verify dependency ordering within each graph
    EXPECT_LT(task_orders["ParentA"], task_orders["ChildA"]);
    EXPECT_LT(task_orders["ParentB"], task_orders["ChildB"]);

    // Key test: ChildA (gen=1, earlier time) should execute before ParentB
    // (gen=0, later time) This proves scheduled time takes precedence over
    // dependency generation across graphs
    EXPECT_LT(task_orders["ChildA"], task_orders["ParentB"])
            << "ChildA should execute before ParentB due to earlier scheduled time";

    RT_LOGC_DEBUG(
            ft::TaskLog::TaskScheduler,
            "Task execution orders: ParentA={}, ChildA={}, ParentB={}, ChildB={}",
            task_orders["ParentA"],
            task_orders["ChildA"],
            task_orders["ParentB"],
            task_orders["ChildB"]);
}

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

} // namespace
