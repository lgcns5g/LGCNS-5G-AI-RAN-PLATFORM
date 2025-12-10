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
 * @file task_tests.cpp
 * @brief Unit tests for Task class, TaskResult, and CancellationToken
 */

#include <atomic>
#include <chrono>
#include <compare>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <ratio>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "task/task.hpp"
#include "task/task_category.hpp"
#include "task/time.hpp"

namespace {
namespace ft = framework::task;

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

using namespace std::chrono_literals;

/**
 * Basic functionality tests for TaskResult
 */
TEST(TaskResult, DefaultConstructor) {
    const ft::TaskResult result{};

    EXPECT_EQ(result.status, ft::TaskStatus::Completed);
    EXPECT_TRUE(result.message.empty());
    EXPECT_TRUE(result.is_success());
}

TEST(TaskResult, ParameterizedConstructor) {
    const ft::TaskResult success_result{ft::TaskStatus::Completed, "Success message"};
    EXPECT_EQ(success_result.status, ft::TaskStatus::Completed);
    EXPECT_EQ(success_result.message, "Success message");
    EXPECT_TRUE(success_result.is_success());

    const ft::TaskResult failure_result{ft::TaskStatus::Failed, "Error occurred"};
    EXPECT_EQ(failure_result.status, ft::TaskStatus::Failed);
    EXPECT_EQ(failure_result.message, "Error occurred");
    EXPECT_FALSE(failure_result.is_success());
}

TEST(TaskResult, IsSuccess) {
    EXPECT_TRUE(ft::TaskResult{ft::TaskStatus::Completed}.is_success());
    EXPECT_FALSE(ft::TaskResult{ft::TaskStatus::Failed}.is_success());
    EXPECT_FALSE(ft::TaskResult{ft::TaskStatus::Cancelled}.is_success());
    EXPECT_FALSE(ft::TaskResult{ft::TaskStatus::Running}.is_success());
    EXPECT_FALSE(ft::TaskResult{ft::TaskStatus::NotStarted}.is_success());
}

/**
 * Basic functionality tests for CancellationToken
 */
TEST(CancellationToken, DefaultState) {
    const ft::CancellationToken token{};

    EXPECT_FALSE(token.is_cancelled());
}

TEST(CancellationToken, CancelAndReset) {
    ft::CancellationToken token{};

    // Initial state
    EXPECT_FALSE(token.is_cancelled());

    // After cancellation
    token.cancel();
    EXPECT_TRUE(token.is_cancelled());

    // After reset (CancellationToken reset is internal to prepare_for_reuse)
    token.reset();
    EXPECT_FALSE(token.is_cancelled());

    // Multiple cancel calls
    token.cancel();
    token.cancel();
    EXPECT_TRUE(token.is_cancelled());
}

TEST(CancellationToken, ThreadSafety) {
    ft::CancellationToken token{};
    std::atomic<bool> thread_saw_cancellation{false};

    std::thread checker([&token, &thread_saw_cancellation]() {
        while (!token.is_cancelled()) {
            std::this_thread::yield();
        }
        thread_saw_cancellation.store(true);
    });

    // Give checker thread time to start
    std::this_thread::sleep_for(1ms);

    // Cancel token - checker thread should detect it
    token.cancel();

    checker.join();

    EXPECT_TRUE(thread_saw_cancellation.load());
    EXPECT_TRUE(token.is_cancelled());
}

TEST(Task, BasicTaskConstruction) {
    auto func = []() { return ft::TaskResult{}; };
    const ft::Task task = ft::TaskBuilder("TestTask")
                                  .function(func)
                                  .category(ft::TaskCategory{ft::BuiltinTaskCategory::HighPriority})
                                  .timeout(5us)
                                  .scheduled_time(1us)
                                  .build();

    EXPECT_EQ(task.get_task_name(), "TestTask");
    EXPECT_EQ(task.get_scheduled_time(), 1us);
    EXPECT_EQ(task.get_timeout_ns(), 5us);
    EXPECT_EQ(task.get_category(), ft::TaskCategory{ft::BuiltinTaskCategory::HighPriority});
    EXPECT_EQ(task.get_dependency_generation(), 0); // Default generation
}

TEST(Task, IsReady) {
    auto func = []() { return ft::TaskResult{}; };
    const ft::Task task = ft::TaskBuilder("TestTask").function(func).scheduled_time(1us).build();

    // Task should be ready when current time >= scheduled time - threshold
    EXPECT_TRUE(task.is_ready(1us, 0ns)); // Exact time
    EXPECT_TRUE(task.is_ready(1.5us,
                              0ns)); // After scheduled time
    EXPECT_TRUE(task.is_ready(900ns,
                              200ns)); // Within threshold (900 + 200 >= 1000)
    EXPECT_FALSE(task.is_ready(800ns,
                               100ns)); // Outside threshold (800 + 100 < 1000)
}

TEST(Task, IsReadyNoScheduledTime) {
    auto func = []() { return ft::TaskResult{}; };
    const ft::Task task = ft::TaskBuilder("TestTask").function(func).scheduled_time(0ns).build();

    // Task with no scheduled time should always be ready
    EXPECT_TRUE(task.is_ready(0ns, 0ns));
    EXPECT_TRUE(task.is_ready(1us, 0ns));
    EXPECT_TRUE(task.is_ready(5us, 100ns));
}

TEST(Task, BasicExecution) {
    auto func = []() { return ft::TaskResult{ft::TaskStatus::Completed, "Success"}; };
    const ft::Task task = ft::TaskBuilder("TestTask").function(func).build();

    const ft::TaskResult result = task.execute();

    EXPECT_EQ(result.status, ft::TaskStatus::Completed);
    EXPECT_EQ(result.message, "Success");
    EXPECT_TRUE(result.is_success());

    // Check task status was updated
    EXPECT_EQ(task.status(), ft::TaskStatus::Completed);
}

TEST(Task, ExecutionWithException) {
    auto func = []() -> ft::TaskResult { throw std::runtime_error("Test exception"); };
    const ft::Task task = ft::TaskBuilder("FailingTask").function(func).build();

    const ft::TaskResult result = task.execute();

    EXPECT_EQ(result.status, ft::TaskStatus::Failed);
    EXPECT_TRUE(result.message.find("Exception: Test exception") != std::string::npos);
    EXPECT_FALSE(result.is_success());

    // Check task status was updated
    EXPECT_EQ(task.status(), ft::TaskStatus::Failed);
}

TEST(Task, ExecutionWithNoFunction) {
    const std::function<ft::TaskResult()> empty_func{}; // Empty function
    const ft::Task task = ft::TaskBuilder("NoFunctionTask").function(empty_func).build();

    const ft::TaskResult result = task.execute();

    EXPECT_EQ(result.status, ft::TaskStatus::Failed);
    EXPECT_EQ(result.message, "Task has no function to execute");
    EXPECT_FALSE(result.is_success());
}

TEST(Task, ExecutionAlreadyRunning) {
    auto func = []() { return ft::TaskResult{ft::TaskStatus::Completed}; };
    const ft::Task task = ft::TaskBuilder("RunningTask").function(func).build();

    // Set the task to running state before testing
    task.set_status(ft::TaskStatus::Running);

    const ft::TaskResult result = task.execute();

    EXPECT_EQ(result.status, ft::TaskStatus::Failed);
    EXPECT_EQ(result.message, "Task not in runnable state");
    EXPECT_FALSE(result.is_success());
}

TEST(Task, ResetTask) {
    auto func = []() { return ft::TaskResult{}; };
    ft::Task task = ft::TaskBuilder("ResetTask").function(func).build();

    // Set task to completed state and cancel it before testing reset
    task.set_status(ft::TaskStatus::Completed);
    task.cancel();

    // Reset task using prepare_for_reuse
    task.prepare_for_reuse("reset_task", "graph_name", []() { return ft::TaskResult{}; });

    EXPECT_EQ(task.status(), ft::TaskStatus::NotStarted);
    EXPECT_FALSE(task.is_cancelled());
}

TEST(Task, CancelTask) {
    auto func = []() { return ft::TaskResult{}; };
    const ft::Task task = ft::TaskBuilder("CancelTask").function(func).build();

    task.cancel();

    EXPECT_TRUE(task.is_cancelled());
    EXPECT_EQ(task.status(), ft::TaskStatus::Cancelled);
}

TEST(Task, CancelRunningTask) {
    auto func = []() { return ft::TaskResult{}; };
    const ft::Task task = ft::TaskBuilder("CancelRunningTask").function(func).build();

    // Set task to running state before testing cancellation
    task.set_status(ft::TaskStatus::Running);

    task.cancel();

    EXPECT_TRUE(task.is_cancelled());
    // Status should still be Running for running tasks, but cancelled
    // The status() method should return Cancelled due to cancel_token
    EXPECT_EQ(task.status(), ft::TaskStatus::Cancelled);
}

TEST(Task, TaskWithCancellation) {
    std::mutex task_mutex{};
    std::condition_variable task_cv{};
    bool task_started{false};

    const ft::Task task =
            ft::TaskBuilder("CancellableTask")
                    .function([&task_mutex, &task_cv, &task_started]() {
                        // Signal that task has started
                        {
                            const std::lock_guard<std::mutex> lock(task_mutex);
                            task_started = true;
                        }
                        task_cv.notify_one();

                        // Simulate some work
                        for (int i = 0; i < 100; ++i) {
                            std::this_thread::sleep_for(10us);
                        }

                        return ft::TaskResult{ft::TaskStatus::Completed, "Task completed"};
                    })
                    .build();

    // Start task execution in separate thread
    std::thread executor([&task]() { std::ignore = task.execute(); });

    // Wait for task to start using condition variable
    {
        std::unique_lock<std::mutex> lock(task_mutex);
        task_cv.wait(lock, [&task_started] { return task_started; });
    }

    // Cancel the task after it has started
    task.cancel();

    // Wait for execution to complete
    executor.join();

    // Verify cancellation worked
    EXPECT_TRUE(task_started);
    EXPECT_TRUE(task.is_cancelled());
}

TEST(Task, CopyAndMove) {
    auto func = []() { return ft::TaskResult{}; };
    ft::Task original_task =
            ft::TaskBuilder("CopyMoveTask").function(func).timeout(0ns).scheduled_time(2us).build();

    // Test copy constructor
    ft::Task copied{original_task};
    EXPECT_EQ(copied.get_task_name(), "CopyMoveTask");
    EXPECT_EQ(copied.get_scheduled_time(), 2us);

    // Test copy assignment
    const ft::Task assigned = original_task;
    EXPECT_EQ(assigned.get_task_name(), "CopyMoveTask");
    EXPECT_EQ(assigned.get_scheduled_time(), 2us);

    // Test move constructor
    const ft::Task moved{std::move(original_task)};
    EXPECT_EQ(moved.get_task_name(), "CopyMoveTask");
    EXPECT_EQ(moved.get_scheduled_time(), 2us);

    // Test move assignment
    const ft::Task move_assigned = std::move(copied);
    EXPECT_EQ(move_assigned.get_task_name(), "CopyMoveTask");
    EXPECT_EQ(move_assigned.get_scheduled_time(), 2us);
}

TEST(Task, Status) {
    auto func = []() { return ft::TaskResult{}; };
    const ft::Task task = ft::TaskBuilder("Status").function(func).build();

    // Test initial status
    EXPECT_EQ(task.status(), ft::TaskStatus::NotStarted);
    EXPECT_FALSE(task.is_cancelled());

    // Test set_status()
    task.set_status(ft::TaskStatus::Running);
    EXPECT_EQ(task.status(), ft::TaskStatus::Running);

    // Test is_cancelled() and status() with cancellation
    task.cancel();
    EXPECT_TRUE(task.is_cancelled());
    EXPECT_EQ(
            task.status(),
            ft::TaskStatus::Cancelled); // Should return Cancelled due to
                                        // cancel token
}

TEST(Task, GetParentStatus) {
    struct TestCase {
        std::string name;
        std::shared_ptr<ft::Task> parent_task;
        bool expect_has_value;
        ft::TaskStatus expected_status;
    };

    // Create parent tasks with different statuses using TaskBuilder
    auto parent_func = []() { return ft::TaskResult{}; };
    auto parent_not_started =
            ft::TaskBuilder("ParentNotStarted").function(parent_func).build_shared();

    auto parent_completed = ft::TaskBuilder("ParentCompleted").function(parent_func).build_shared();
    parent_completed->set_status(ft::TaskStatus::Completed);

    auto parent_cancelled = ft::TaskBuilder("ParentCancelled").function(parent_func).build_shared();
    parent_cancelled->set_status(ft::TaskStatus::Cancelled);

    const std::vector<TestCase> test_cases = {
            {"NoParent", nullptr, false, ft::TaskStatus::NotStarted},
            {"ParentNotStarted", parent_not_started, true, ft::TaskStatus::NotStarted},
            {"ParentCompleted", parent_completed, true, ft::TaskStatus::Completed},
            {"ParentCancelled", parent_cancelled, true, ft::TaskStatus::Cancelled}};

    for (const auto &test_case : test_cases) {
        SCOPED_TRACE("Test case: " + test_case.name);

        auto func = []() { return ft::TaskResult{}; };
        // Use TaskBuilder with proper parent relationships
        auto builder = ft::TaskBuilder(test_case.name).function(func);
        if (test_case.parent_task) {
            builder.depends_on(test_case.parent_task);
        }
        const ft::Task task = builder.build();
        const auto has_no_parents = task.has_no_parents();

        EXPECT_EQ(has_no_parents, !test_case.expect_has_value) << "Test case: " << test_case.name;
        if (test_case.expect_has_value) {
            // Test that the parent matches the expected status
            const auto parent_matches_expected = task.any_parent_matches(
                    [expected = test_case.expected_status](ft::TaskStatus status) {
                        return status == expected;
                    });
            EXPECT_TRUE(parent_matches_expected) << "Test case: " << test_case.name;
        }
    }
}

TEST(Task, TaskTimeComparatorWithDependencyGeneration) {
    auto func = []() { return ft::TaskResult{}; };

    // Create tasks with different dependency generations but same scheduled time
    const ft::Nanos same_time{1000};
    const ft::Task root_task =
            ft::TaskBuilder("Root").function(func).scheduled_time(same_time).build();
    // Create child task that depends on root task (generation will be 1)
    auto child_task_shared = ft::TaskBuilder("Child")
                                     .function(func)
                                     .scheduled_time(same_time)
                                     .depends_on(std::make_shared<ft::Task>(root_task))
                                     .build_shared();
    const ft::Task child_task = *child_task_shared;

    // Create grandchild task that depends on child task (generation will be 2)
    const ft::Task grandchild_task = ft::TaskBuilder("Grandchild")
                                             .function(func)
                                             .scheduled_time(same_time)
                                             .depends_on(child_task_shared)
                                             .build();

    // TaskTimeComparator should prioritize lower dependency generation
    // (Priority queue is a max heap, so operator() returns true if a should be
    // scheduled after b)
    const auto comparator = [](const ft::Task &a, const ft::Task &b) {
        // Primary: Lower dependency generation has higher priority
        if (a.get_dependency_generation() != b.get_dependency_generation()) {
            return a.get_dependency_generation() > b.get_dependency_generation();
        }
        // Secondary: Earlier scheduled time has higher priority
        if (a.get_scheduled_time() != b.get_scheduled_time()) {
            return a.get_scheduled_time() > b.get_scheduled_time();
        }
        // Tie-breaker: Lexicographic order by name for deterministic behavior
        return a.get_task_name() > b.get_task_name();
    };

    // Root should come before child
    EXPECT_FALSE(comparator(root_task, child_task))
            << "Root should have higher priority than child";
    EXPECT_TRUE(comparator(child_task, root_task)) << "Child should have lower priority than root";

    // Child should come before grandchild
    EXPECT_FALSE(comparator(child_task, grandchild_task))
            << "Child should have higher priority than grandchild";
    EXPECT_TRUE(comparator(grandchild_task, child_task))
            << "Grandchild should have lower priority than child";

    // Root should come before grandchild
    EXPECT_FALSE(comparator(root_task, grandchild_task))
            << "Root should have higher priority than grandchild";
    EXPECT_TRUE(comparator(grandchild_task, root_task))
            << "Grandchild should have lower priority than root";

    // Test with different scheduled times: earlier time should win when
    // generations are same
    // Create children that depend on root task (both will have generation 1)
    auto root_shared = std::make_shared<ft::Task>(root_task);
    const ft::Task early_child = ft::TaskBuilder("EarlyChild")
                                         .function(func)
                                         .scheduled_time(500ns)
                                         .depends_on(root_shared)
                                         .build();
    const ft::Task late_child = ft::TaskBuilder("LateChild")
                                        .function(func)
                                        .scheduled_time(1.5us)
                                        .depends_on(root_shared)
                                        .build();

    EXPECT_FALSE(comparator(early_child, late_child))
            << "Earlier child should have higher priority";
    EXPECT_TRUE(comparator(late_child, early_child)) << "Later child should have lower priority";
}

// Test functions for function pointer tests
ft::TaskResult test_function_success() {
    return ft::TaskResult{ft::TaskStatus::Completed, "Function completed successfully"};
}

ft::TaskResult test_function_failure() {
    return ft::TaskResult{ft::TaskStatus::Failed, "Function failed"};
}

ft::TaskResult test_function_with_work() {
    // Simulate some work
    std::this_thread::sleep_for(1ms);
    return ft::TaskResult{ft::TaskStatus::Completed, "Work completed"};
}

// Test function object/functor
struct TestFunctor {
    std::string message;

    explicit TestFunctor(std::string msg) : message(std::move(msg)) {}

    ft::TaskResult operator()() const { return ft::TaskResult{ft::TaskStatus::Completed, message}; }
};

/**
 * Tests for Task API with functions (not just lambdas)
 */
TEST(Task, FunctionTypesTable) {
    struct TestCase {
        std::string name;
        std::function<ft::TaskResult()> func;
        ft::TaskCategory category;
        std::chrono::milliseconds timeout;
        std::chrono::microseconds scheduled_time;
        ft::TaskStatus expected_status;
        std::string expected_message;
        bool check_timing;
    };

    const std::vector<TestCase> test_cases = {
            {"FunctionPointerSuccess",
             test_function_success,
             ft::TaskCategory{ft::BuiltinTaskCategory::Default},
             0ms,
             0us,
             ft::TaskStatus::Completed,
             "Function completed successfully",
             false},
            {"FunctionPointerFailure",
             test_function_failure,
             ft::TaskCategory{ft::BuiltinTaskCategory::Default},
             0ms,
             0us,
             ft::TaskStatus::Failed,
             "Function failed",
             false},
            {"FunctionObjectTest",
             TestFunctor{"Functor test message"},
             ft::TaskCategory{ft::BuiltinTaskCategory::Compute},
             1ms,
             500us,
             ft::TaskStatus::Completed,
             "Functor test message",
             false},
            {"StdFunctionWrapper",
             std::function<ft::TaskResult()>{test_function_success},
             ft::TaskCategory{ft::BuiltinTaskCategory::HighPriority},
             2ms,
             1000us,
             ft::TaskStatus::Completed,
             "Function completed successfully",
             false},
            {"WorkFunctionWithTiming",
             test_function_with_work,
             ft::TaskCategory{ft::BuiltinTaskCategory::IO},
             5ms,
             2000us,
             ft::TaskStatus::Completed,
             "Work completed",
             true}};

    for (const auto &test_case : test_cases) {
        SCOPED_TRACE("Test case: " + test_case.name);

        const ft::Task task = ft::TaskBuilder(test_case.name)
                                      .function(test_case.func)
                                      .category(test_case.category)
                                      .timeout(test_case.timeout)
                                      .scheduled_time(test_case.scheduled_time)
                                      .build();

        // Verify task properties
        EXPECT_EQ(task.get_task_name(), test_case.name);
        EXPECT_EQ(task.get_category(), test_case.category);
        EXPECT_EQ(task.get_timeout_ns(), std::chrono::duration_cast<ft::Nanos>(test_case.timeout));
        EXPECT_EQ(
                task.get_scheduled_time(),
                std::chrono::duration_cast<ft::Nanos>(test_case.scheduled_time));

        // Verify execution
        const auto start_time = std::chrono::steady_clock::now();
        const ft::TaskResult result = task.execute();
        const auto end_time = std::chrono::steady_clock::now();

        EXPECT_EQ(result.status, test_case.expected_status);
        EXPECT_EQ(result.message, test_case.expected_message);
        EXPECT_EQ(result.is_success(), test_case.expected_status == ft::TaskStatus::Completed);
        EXPECT_EQ(task.status(), test_case.expected_status);

        // Check timing for work functions
        if (test_case.check_timing) {
            const auto duration =
                    std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            EXPECT_GE(duration.count(), 1) << "Work function should take measurable time";
        }
    }
}

/**
 * Tests for new Task methods added for TaskPool support
 */
TEST(Task, ReserveParentCapacity) {
    auto task = ft::TaskBuilder("ReserveTest")
                        .function([]() { return ft::TaskResult{}; })
                        .build_shared();

    // Should not throw and should work with any capacity
    task->reserve_parent_capacity(10);
    task->reserve_parent_capacity(0);
    task->reserve_parent_capacity(100);
}

TEST(Task, ClearParentTasks) {
    auto parent1 =
            ft::TaskBuilder("Parent1").function([]() { return ft::TaskResult{}; }).build_shared();

    auto parent2 =
            ft::TaskBuilder("Parent2").function([]() { return ft::TaskResult{}; }).build_shared();

    auto child = ft::TaskBuilder("Child")
                         .function([]() { return ft::TaskResult{}; })
                         .depends_on(parent1)
                         .depends_on(parent2)
                         .build_shared();

    // Should have parents initially
    EXPECT_FALSE(child->has_no_parents());

    // Clear parents
    child->clear_parent_tasks();

    // Should have no parents after clearing
    EXPECT_TRUE(child->has_no_parents());
}

TEST(Task, PrepareForReuse) {
    auto original_task = ft::TaskBuilder("OriginalTask")
                                 .function([]() {
                                     return ft::TaskResult{ft::TaskStatus::Completed, "original"};
                                 })
                                 .timeout(1us)
                                 .scheduled_time(2us)
                                 .graph_name("graph_name")
                                 .build_shared();

    // Execute original task to change state
    auto result = original_task->execute();
    EXPECT_TRUE(result.is_success());
    EXPECT_EQ(original_task->status(), ft::TaskStatus::Completed);

    // Prepare for reuse with new configuration
    const std::string new_name = "ReusedTask";
    const std::string new_graph_name = "new_graph_name";
    auto new_func = []() { return ft::TaskResult{ft::TaskStatus::Completed, "reused"}; };

    original_task->prepare_for_reuse(
            new_name,
            new_graph_name,
            new_func,
            ft::TaskCategory{ft::BuiltinTaskCategory::Default},
            ft::Nanos{3000},
            ft::Nanos{4000});

    // Verify task was reset and configured
    EXPECT_EQ(original_task->get_task_name(), new_name);
    EXPECT_EQ(original_task->get_graph_name(), new_graph_name);
    EXPECT_EQ(original_task->status(), ft::TaskStatus::NotStarted);
    EXPECT_EQ(original_task->get_timeout_ns(), 3us);
    EXPECT_EQ(original_task->get_scheduled_time(), 4us);
    EXPECT_TRUE(original_task->has_no_parents());

    // Verify new function works
    auto reuse_result = original_task->execute();
    EXPECT_TRUE(reuse_result.is_success());
    EXPECT_EQ(reuse_result.message, "reused");
}

TEST(Task, PrepareForReuseFunctionTypeSwitching) {
    auto task_result_no_params = []() {
        return ft::TaskResult{ft::TaskStatus::Completed, "no_params"};
    };

    auto task_result_with_context = []([[maybe_unused]] const ft::TaskContext &ctx) {
        return ft::TaskResult{ft::TaskStatus::Completed, "with_context"};
    };

    auto void_no_params = []() {
        // Void function - should be wrapped to return TaskResult
    };

    auto void_with_context = []([[maybe_unused]] const ft::TaskContext &ctx) {
        // Void function with context - should be wrapped
    };

    auto final_task_result = []() {
        return ft::TaskResult{ft::TaskStatus::Completed, "back_to_taskresult"};
    };

    struct TestCase {
        std::function<void()> setup_func;
        std::string expected_message;
        ft::TaskStatus expected_status;
        std::string description;
    };

    auto task = ft::TaskBuilder("SwitchingTask").function(task_result_no_params).build_shared();

    const std::vector<TestCase> test_cases = {
            {[]() { /* Initial function already set */ },
             "no_params",
             ft::TaskStatus::Completed,
             "Initial TaskResult function (no parameters)"},
            {[task, &task_result_with_context]() {
                 task->prepare_for_reuse("SwitchingTask", "graph", task_result_with_context);
             },
             "with_context",
             ft::TaskStatus::Completed,
             "TaskResult function with TaskContext parameter"},
            {[task, &void_no_params]() {
                 task->prepare_for_reuse("SwitchingTask", "graph", void_no_params);
             },
             "",
             ft::TaskStatus::Completed,
             "Void function (no parameters) - auto-wrapped"},
            {[task, &void_with_context]() {
                 task->prepare_for_reuse("SwitchingTask", "graph", void_with_context);
             },
             "",
             ft::TaskStatus::Completed,
             "Void function with context - auto-wrapped"},
            {[task, &final_task_result]() {
                 task->prepare_for_reuse("SwitchingTask", "graph", final_task_result);
             },
             "back_to_taskresult",
             ft::TaskStatus::Completed,
             "Final TaskResult function (no parameters)"}};

    for (std::size_t i = 0; i < test_cases.size(); ++i) {
        const auto &test_case = test_cases[i];

        // Setup the function for this test case
        test_case.setup_func();

        // Execute and verify results
        auto result = task->execute();
        EXPECT_TRUE(result.is_success()) << "Test case " << i << ": " << test_case.description;
        EXPECT_EQ(result.status, test_case.expected_status)
                << "Test case " << i << ": " << test_case.description;

        if (!test_case.expected_message.empty()) {
            EXPECT_EQ(result.message, test_case.expected_message)
                    << "Test case " << i << ": " << test_case.description;
        }
    }
}

TEST(Task, PrepareForReuseNullFunctionHandling) {
    // Define original function for task initialization
    const auto original_func = []() {
        return ft::TaskResult{ft::TaskStatus::Completed, "original"};
    };

    auto task = ft::TaskBuilder("NullFunctionTask").function(original_func).build_shared();

    // Test data for null function handling
    struct NullFunctionTest {
        ft::TaskFunction null_func;
        std::string description;
    };

    const std::vector<NullFunctionTest> null_tests = {
            {ft::TaskFunction{std::function<ft::TaskResult()>{}},
             "Null std::function<TaskResult()>"},
            {ft::TaskFunction{std::function<ft::TaskResult(const ft::TaskContext &)>{}},
             "Null std::function<TaskResult(const TaskContext&)>"}};

    for (std::size_t i = 0; i < null_tests.size(); ++i) {
        const auto &test = null_tests[i];

        // Verify the function is null before testing
        const bool is_null = std::visit([](const auto &func) { return !func; }, test.null_func);
        EXPECT_TRUE(is_null) << "Test case " << i << ": " << test.description;

        // This should not crash and should provide a default function
        task->prepare_for_reuse("NullFunctionTask", "graph", test.null_func);

        // Execute should work with default function
        auto result = task->execute();
        EXPECT_TRUE(result.is_success()) << "Test case " << i << ": " << test.description;
        EXPECT_EQ(result.status, ft::TaskStatus::Completed)
                << "Test case " << i << ": " << test.description;
    }
}

TEST(Task, FunctionDetectsCancellation) {
    bool cancelled = false;
    bool user_data_correct = false;
    bool bad_cast_detected = false;
    std::atomic<bool> task_started = false;

    struct UserData {
        int num{};
    };
    auto user_data = std::make_shared<UserData>(UserData{.num = 42});

    auto task_func = [&task_started, &cancelled, &user_data_correct, &bad_cast_detected](
                             const ft::TaskContext &ctx) -> ft::TaskResult {
        task_started = true;

        // First try wrong type to test bad_any_cast handling
        auto wrong_type_data = ctx.get_user_data<int>();
        bad_cast_detected = !wrong_type_data.has_value(); // Should fail due to type mismatch

        // Then retrieve as the correct type we stored: std::shared_ptr<UserData>
        auto retrieved_data = ctx.get_user_data<std::shared_ptr<UserData>>();
        user_data_correct = retrieved_data.has_value() && (*retrieved_data)->num == 42;
        if (user_data_correct) {
            (*retrieved_data)->num = 43; // Modify through shared_ptr
        }

        // Wait some time for the task to be cancelled before checking for
        // cancellation
        std::this_thread::sleep_for(1ms);

        // Check if task can detect the cancellation
        if (ctx.cancellation_token->is_cancelled()) {
            cancelled = true;
            return ft::TaskResult{ft::TaskStatus::Cancelled, "Function detected cancellation"};
        }

        return ft::TaskResult{ft::TaskStatus::Completed, "Work completed"};
    };

    // Create and execute task with user data in a separate thread
    auto task = ft::TaskBuilder("cancellation_test")
                        .function(task_func)
                        .user_data(user_data)
                        .build_shared();
    ft::TaskResult result{};
    std::thread exec_thread([&task, &result]() { result = task->execute(); });
    while (!task_started) {
        std::this_thread::sleep_for(50us);
    }
    task->cancel(); // Immediately cancel the task
    exec_thread.join();

    EXPECT_TRUE(bad_cast_detected);
    EXPECT_TRUE(user_data_correct);
    EXPECT_EQ(user_data->num, 43) << "Should see modifications made inside task function";
    EXPECT_TRUE(cancelled);
    EXPECT_EQ(result.status, ft::TaskStatus::Cancelled);
}

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

} // namespace
