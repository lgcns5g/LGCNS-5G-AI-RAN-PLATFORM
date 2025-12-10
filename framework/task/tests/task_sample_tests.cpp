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
 * @file task_sample_tests.cpp
 * @brief Sample tests for task library documentation
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <system_error>
#include <thread>
#include <tuple>
#include <vector>

#include <wise_enum_detail.h>
#include <wise_enum_generated.h>

#include <gtest/gtest.h>

#include "task/memory_trigger.hpp"
#include "task/task.hpp"
#include "task/task_category.hpp"
#include "task/task_graph.hpp"
#include "task/task_pool.hpp"
#include "task/task_scheduler.hpp"
#include "task/task_worker.hpp"
#include "task/timed_trigger.hpp"

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

namespace {

using namespace std::chrono_literals;

// Define custom task categories at namespace scope
// example-begin custom-categories-1
DECLARE_TASK_CATEGORIES(MyAppCategories, DataProcessing, NetworkIO, Rendering);
// example-end custom-categories-1

TEST(TaskSampleTests, BasicTask) {
    using namespace framework::task;

    // example-begin basic-task-1
    // Create a simple task that executes a function
    std::atomic<int> counter{0};

    auto task = TaskBuilder("simple_task")
                        .function([&counter]() -> TaskResult {
                            counter++;
                            return TaskResult{TaskStatus::Completed};
                        })
                        .build_shared();

    // Execute the task
    const auto result = task->execute();
    // example-end basic-task-1

    EXPECT_TRUE(result.is_success());
    EXPECT_EQ(counter.load(), 1);
}

TEST(TaskSampleTests, TaskWithTimeout) {
    using namespace framework::task;

    // example-begin task-timeout-1
    auto task = TaskBuilder("timed_task")
                        .function([]() -> TaskResult {
                            std::this_thread::sleep_for(50ms);
                            return TaskResult{TaskStatus::Completed};
                        })
                        .timeout(100ms)
                        .build_shared();
    // example-end task-timeout-1

    const auto result = task->execute();
    EXPECT_TRUE(result.is_success());
}

TEST(TaskSampleTests, TaskWithCancellation) {
    using namespace framework::task;

    // example-begin task-cancellation-1
    // Define a cancellable work function
    auto cancellable_work = [](const TaskContext &ctx) -> TaskResult {
        for (int i = 0; i < 10; i++) {
            if (ctx.cancellation_token->is_cancelled()) {
                return TaskResult{TaskStatus::Cancelled, "Cancelled by user"};
            }
            std::this_thread::sleep_for(10ms);
        }
        return TaskResult{TaskStatus::Completed};
    };

    auto task = TaskBuilder("cancellable_task").function(cancellable_work).build_shared();
    // example-end task-cancellation-1

    // Execute in a separate thread and cancel during execution
    std::atomic<bool> started{false};
    std::thread worker([&]() {
        started.store(true);
        std::ignore = task->execute();
    });

    // Wait for task to start, then cancel it
    while (!started.load()) {
        std::this_thread::yield();
    }
    std::this_thread::sleep_for(5ms);
    task->cancel();

    worker.join();
    EXPECT_EQ(task->status(), TaskStatus::Cancelled);
}

TEST(TaskSampleTests, SimpleTaskGraph) {
    using namespace framework::task;

    // example-begin simple-graph-1
    std::atomic<int> counter{0};

    // Create a task graph with a single task
    auto graph = TaskGraph::create("simple_graph")
                         .single_task("increment")
                         .function([&counter]() { counter++; })
                         .build();
    // example-end simple-graph-1

    EXPECT_FALSE(graph.empty());
    EXPECT_EQ(graph.size(), 1);
}

TEST(TaskSampleTests, TaskGraphWithDependencies) {
    using namespace framework::task;

    // example-begin graph-dependencies-1
    std::atomic<int> step{0};

    TaskGraph graph("dependency_graph");

    // Create tasks with dependencies
    auto grandparent =
            graph.register_task("grandparent").function([&step]() { step.store(1); }).add();

    auto parent = graph.register_task("parent")
                          .depends_on(grandparent)
                          .function([&step]() { step.store(2); })
                          .add();

    graph.register_task("child").depends_on(parent).function([&step]() { step.store(3); }).add();

    // Build the graph to finalize dependencies
    graph.build();
    // example-end graph-dependencies-1

    EXPECT_EQ(graph.size(), 3);
    EXPECT_TRUE(graph.is_built());
}

TEST(TaskSampleTests, BasicScheduler) {
    using namespace framework::task;

    // example-begin basic-scheduler-1
    // Create a scheduler with 2 worker threads
    auto scheduler = TaskScheduler::create().workers(2).build();

    std::atomic<int> counter{0};

    auto graph = TaskGraph::create("scheduled_graph")
                         .single_task("work")
                         .function([&counter]() { counter++; })
                         .build();

    // Schedule the graph for execution
    scheduler.schedule(graph);

    // Wait for workers to complete
    scheduler.join_workers();
    // example-end basic-scheduler-1

    EXPECT_EQ(counter.load(), 1);
}

TEST(TaskSampleTests, TaskMonitorUsage) {
    using namespace framework::task;

    // example-begin task-monitor-1
    // Configure scheduler with monitor on dedicated core
    auto scheduler = TaskScheduler::create().workers(4).monitor_core(0).build();

    std::atomic<int> completed_count{0};
    std::atomic<int> timeout_count{0};

    TaskGraph graph("monitored_graph");

    // Normal task
    graph.register_task("fast_task").function([&completed_count]() { completed_count++; }).add();

    // Task with timeout that will exceed and be cancelled by the task monitor
    graph.register_task("timeout_task")
            .timeout(10ms)
            .function([&timeout_count](const TaskContext &ctx) -> TaskResult {
                std::this_thread::sleep_for(50ms);
                if (ctx.cancellation_token->is_cancelled()) {
                    return TaskResult{TaskStatus::Cancelled, "Task timed out"};
                }
                timeout_count++;
                return TaskResult{TaskStatus::Completed};
            })
            .add();

    graph.build();

    // Execute tasks
    scheduler.schedule(graph);
    scheduler.join_workers();

    // Print execution statistics
    scheduler.print_monitor_stats();

    // Export Chrome trace for visualization
    const auto result = scheduler.write_chrome_trace_to_file("trace.json");
    // example-end task-monitor-1

    EXPECT_EQ(completed_count.load(), 1);
    EXPECT_EQ(timeout_count.load(), 0); // Task should be cancelled before incrementing
    EXPECT_EQ(result.value(), 0);
}

TEST(TaskSampleTests, SchedulerWithCategories) {
    using namespace framework::task;

    // example-begin scheduler-categories-1
    // Configure workers for specific task categories
    std::vector<WorkerConfig> configs;

    // Worker 0: High priority tasks
    configs.push_back(
            WorkerConfig::create_for_categories({TaskCategory{BuiltinTaskCategory::HighPriority}}));

    // Worker 1: Default tasks
    configs.push_back(
            WorkerConfig::create_for_categories({TaskCategory{BuiltinTaskCategory::Default}}));

    auto scheduler = TaskScheduler::create().workers(WorkersConfig{configs}).build();
    // example-end scheduler-categories-1

    EXPECT_EQ(scheduler.get_workers_config().size(), 2);
}

TEST(TaskSampleTests, WorkerWithCoreAffinity) {
    using namespace framework::task;

    // example-begin worker-affinity-1
    const auto num_cores = std::thread::hardware_concurrency();
    if (num_cores >= 4) {
        // Pin worker to specific CPU core
        std::vector<WorkerConfig> configs;
        configs.push_back(WorkerConfig::create_pinned(
                2, // Core ID
                {TaskCategory{BuiltinTaskCategory::Default}}));

        auto scheduler = TaskScheduler::create().workers(WorkersConfig{configs}).build();
    }
    // example-end worker-affinity-1
}

TEST(TaskSampleTests, TimedTriggerBasic) {
    using namespace framework::task;

    // example-begin timed-trigger-1
    std::atomic<int> tick_count{0};

    // Create a periodic trigger
    auto trigger = TimedTrigger::create(
                           [&tick_count]() { tick_count++; },
                           10ms) // Trigger every 10ms
                           .max_triggers(5)
                           .build();

    // Start the trigger
    if (const auto err = trigger.start(); !err) {
        trigger.wait_for_completion();
    }
    // example-end timed-trigger-1

    EXPECT_GE(tick_count.load(), 5);
}

TEST(TaskSampleTests, MemoryTriggerBasic) {
    using namespace framework::task;

    // example-begin memory-trigger-1
    auto memory = std::make_shared<std::atomic<int>>(0);
    std::atomic<int> trigger_count{0};
    std::atomic<int> value_delta{0};

    // Monitor memory location for changes
    auto trigger = MemoryTrigger<int>::create(
                           memory,
                           [&trigger_count, &value_delta](int old_val, int new_val) {
                               trigger_count++;
                               value_delta.store(new_val - old_val);
                           })
                           .with_notification_strategy(NotificationStrategy::Polling)
                           .with_polling_interval(1ms)
                           .build();

    if (const auto err = trigger.start(); !err) {
        // Change the memory value
        memory->store(42);
        std::this_thread::sleep_for(10ms);
        trigger.stop();
    }
    // example-end memory-trigger-1

    EXPECT_GE(trigger_count.load(), 1);
    EXPECT_EQ(value_delta.load(), 42);
}

TEST(TaskSampleTests, TaskPoolUsage) {
    using namespace framework::task;

    // example-begin task-pool-1
    // Create a task pool with initial capacity
    auto pool = TaskPool::create(
            100, // Initial pool size
            8,   // Max task parents
            64,  // Max task name length
            32); // Max graph name length

    // Acquire tasks from pool
    auto task1 = pool->acquire_task("task1", "graph1");
    auto task2 = pool->acquire_task("task2", "graph1");

    // Check pool statistics
    const auto stats = pool->get_stats();
    // example-end task-pool-1

    EXPECT_GE(stats.total_tasks_served(), 2);
}

TEST(TaskSampleTests, CustomTaskCategories) {
    using namespace framework::task;

    // example-begin custom-categories-2
    std::atomic<int> counter{0};

    // Use custom task category with TaskGraph
    auto graph = TaskGraph::create("categorized_graph")
                         .single_task("process_data")
                         .category(TaskCategory{MyAppCategories::DataProcessing})
                         .function([&counter]() { counter++; })
                         .build();
    // example-end custom-categories-2

    EXPECT_FALSE(graph.empty());
}

} // namespace

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
