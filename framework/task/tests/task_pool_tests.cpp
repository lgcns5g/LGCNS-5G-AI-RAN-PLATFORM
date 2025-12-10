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
 * @file task_pool_tests.cpp
 * @brief Unit tests for TaskPool class
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <format>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "task/task.hpp"
#include "task/task_pool.hpp"

namespace {

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

using namespace framework::task;

/**
 * Test fixture for TaskPool tests
 */
class TaskPoolTest : public ::testing::Test {
protected:
    // No setup needed - each test creates its own pool
};

TEST_F(TaskPoolTest, ConstructorWithDefaults) {
    // Test that factory function works with default parameters
    auto default_pool = TaskPool::create();
    EXPECT_NE(default_pool, nullptr);
    EXPECT_GT(default_pool->capacity(), 0);
}

TEST_F(TaskPoolTest, ConstructorWithCustomSize) {
    // Test factory function with custom pool size
    const std::size_t custom_size = 32;
    auto custom_pool = TaskPool::create(custom_size);
    EXPECT_NE(custom_pool, nullptr);
    EXPECT_GE(custom_pool->capacity(), custom_size);
}

TEST_F(TaskPoolTest, AcquireTaskBasic) {
    // Test basic task acquisition
    auto pool = TaskPool::create();
    const std::string task_name = "test_task";
    const std::string graph_name = "test_graph";
    auto task = pool->acquire_task(task_name, graph_name);

    ASSERT_NE(task, nullptr);
    EXPECT_EQ(task->get_task_name(), task_name);
    EXPECT_EQ(task->status(), TaskStatus::NotStarted);
}

TEST_F(TaskPoolTest, TaskReturnToPoolOnDestruct) {
    // Test that tasks are returned to pool when shared_ptr is destroyed
    auto pool = TaskPool::create();
    const std::string task_name = "pool_return_test";
    auto initial_stats = pool->get_stats();

    {
        auto task = pool->acquire_task(task_name, "graph_name");
        ASSERT_NE(task, nullptr);

        // Task should be acquired from pre-filled pool
        auto stats_after_acquire = pool->get_stats();
        EXPECT_GT(stats_after_acquire.pool_hits, initial_stats.pool_hits);
    }
    // Task should be returned to pool here when shared_ptr destructs

    // Acquire another task - should reuse the returned one
    auto task2 = pool->acquire_task("reused_task", "graph_name");
    ASSERT_NE(task2, nullptr);

    auto final_stats = pool->get_stats();
    EXPECT_GT(final_stats.pool_hits,
              initial_stats.pool_hits + 1); // Should have reused
}

TEST_F(TaskPoolTest, PoolExhaustionFallbackToHeap) {
    // Test behavior when pool is exhausted
    const std::size_t small_pool_size = 2;
    auto small_pool = TaskPool::create(small_pool_size);

    std::vector<std::shared_ptr<Task>> tasks{};
    tasks.reserve(small_pool_size + 2); // More than pool size

    // Acquire more tasks than pool capacity
    for (std::size_t i = 0; i < small_pool_size + 2; ++i) {
        auto task = small_pool->acquire_task(std::format("heap_test_{}", i), "graph_name");
        ASSERT_NE(task, nullptr);
        tasks.push_back(std::move(task));
    }

    // Should have fallen back to heap allocation
    auto stats = small_pool->get_stats();
    EXPECT_GT(stats.new_tasks_created, 0);

    // Test released counter tracking - reset tasks to return them to pool
    const auto initial_released = stats.tasks_released;
    const auto num_tasks = tasks.size();
    tasks.clear(); // This will trigger return_to_pool for all tasks

    auto final_stats = small_pool->get_stats();
    EXPECT_EQ(final_stats.tasks_released, initial_released + num_tasks);
}

TEST_F(TaskPoolTest, TaskReuse) {
    // Test that tasks are properly reset for reuse
    auto pool = TaskPool::create();
    const std::string first_name = "first_task";
    const std::string second_name = "second_task";

    std::shared_ptr<Task> first_task{};

    {
        first_task = pool->acquire_task(first_name, "graph_name");

        // Execute the task to change its state
        auto result = first_task->execute();
        EXPECT_TRUE(result.is_success());
        EXPECT_EQ(first_task->status(), TaskStatus::Completed);
    }
    // Task returned to pool

    // Acquire new task - should reuse the same Task object
    auto second_task = pool->acquire_task(second_name, "graph_name");
    ASSERT_NE(second_task, nullptr);

    // Should be reset to initial state
    EXPECT_EQ(second_task->status(), TaskStatus::NotStarted);
    EXPECT_EQ(second_task->get_task_name(), second_name);

    // Could be same Task object reused (task_id might be same)
    // This is implementation dependent
}

TEST_F(TaskPoolTest, MaxTaskParentsReserve) {
    // Test that parent_statuses vector has reserved capacity
    const std::size_t max_parents = 16;
    auto pool_with_reserve = TaskPool::create(DEFAULT_POOL_SIZE, max_parents);

    auto task = pool_with_reserve->acquire_task("reserve_test", "graph_name");
    ASSERT_NE(task, nullptr);

    // Create parent tasks and add them
    std::vector<std::shared_ptr<Task>> parents{};
    for (std::size_t i = 0; i < max_parents / 2; ++i) {
        auto parent = pool_with_reserve->acquire_task(std::format("parent_{}", i), "graph_name");
        task->add_parent_task(parent);
        parents.push_back(std::move(parent));
    }

    // Task should still function properly with parents
    EXPECT_FALSE(task->has_no_parents());
}

TEST_F(TaskPoolTest, ConcurrentAcquisition) {
    // Test thread safety of concurrent task acquisition
    auto pool = TaskPool::create(128); // Larger pool for concurrent access
    const std::size_t num_threads = 4;
    const std::size_t tasks_per_thread = 10;
    std::atomic<std::size_t> successful_acquisitions{0};
    std::vector<std::thread> threads{};

    for (std::size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back([&pool, t, &successful_acquisitions]() {
            for (std::size_t i = 0; i < tasks_per_thread; ++i) {
                auto task =
                        pool->acquire_task(std::format("thread_{}_task_{}", t, i), "graph_name");
                if (task) {
                    successful_acquisitions.fetch_add(1);

                    // Simulate some work
                    using namespace std::chrono_literals;
                    std::this_thread::sleep_for(10us);
                }
            }
        });
    }

    // Wait for all threads to complete
    for (auto &thread : threads) {
        thread.join();
    }

    // All acquisitions should have succeeded
    EXPECT_EQ(successful_acquisitions.load(), num_threads * tasks_per_thread);
}

TEST_F(TaskPoolTest, StatsTracking) {
    // Test that statistics are properly tracked
    auto pool = TaskPool::create();
    auto initial_stats = pool->get_stats();

    // Acquire from pre-filled pool
    auto task1 = pool->acquire_task("stats_test_1", "graph_name");
    auto stats_1 = pool->get_stats();
    EXPECT_GT(stats_1.pool_hits, initial_stats.pool_hits);
    EXPECT_EQ(stats_1.new_tasks_created, initial_stats.new_tasks_created);

    task1.reset(); // Return to pool

    // Reuse from pool
    auto task2 = pool->acquire_task("stats_test_2", "graph_name");
    auto stats_2 = pool->get_stats();
    EXPECT_GT(stats_2.pool_hits, stats_1.pool_hits);
    EXPECT_EQ(stats_2.new_tasks_created, initial_stats.new_tasks_created);

    // Test utility methods
    EXPECT_EQ(stats_2.total_tasks_served(), stats_2.pool_hits + stats_2.new_tasks_created);
    EXPECT_GT(stats_2.hit_rate_percent(), 0.0);
    EXPECT_LE(stats_2.hit_rate_percent(), 100.0);

    // Verify released counter incremented when task1 was returned to pool
    EXPECT_GT(stats_2.tasks_released, initial_stats.tasks_released);
    EXPECT_EQ(stats_2.tasks_released, initial_stats.tasks_released + 1);
}

TEST_F(TaskPoolTest, CapacityReporting) {
    // Test that capacity is reported correctly
    const std::size_t requested_size = 32;
    auto sized_pool = TaskPool::create(requested_size);

    const std::size_t capacity = sized_pool->capacity();

    // Capacity should be at least the requested size (might be larger due to
    // power-of-2 rounding)
    EXPECT_GE(capacity, requested_size);

    // Should be a power of 2 (BoundedQueue requirement)
    EXPECT_EQ(capacity & (capacity - 1), 0);
}

TEST_F(TaskPoolTest, TaskFunctionExecution) {
    // Test that pooled tasks can execute custom functions properly
    auto pool = TaskPool::create();
    std::atomic<int> counter{0};

    // Set custom function on the pooled task
    auto task = pool->acquire_task("function_test", "graph_name");
    task->prepare_for_reuse("function_test", "graph_name", [&counter]() {
        counter.fetch_add(1);
        return TaskResult{TaskStatus::Completed, "Counter incremented"};
    });

    // Execute the task
    auto result = task->execute();
    EXPECT_TRUE(result.is_success());
    EXPECT_EQ(counter.load(), 1);
}

TEST_F(TaskPoolTest, MultiplePoolsIndependent) {
    // Test that multiple pools operate independently
    auto pool1 = TaskPool::create(16);
    auto pool2 = TaskPool::create(32);

    auto task1 = pool1->acquire_task("pool1_task", "graph_name");
    auto task2 = pool2->acquire_task("pool2_task", "graph_name");

    ASSERT_NE(task1, nullptr);
    ASSERT_NE(task2, nullptr);
    EXPECT_NE(task1.get(), task2.get()); // Different Task objects

    auto stats1 = pool1->get_stats();
    auto stats2 = pool2->get_stats();

    // Stats should be independent
    EXPECT_GT(stats1.pool_hits, 0);
    EXPECT_GT(stats2.pool_hits, 0);
}

TEST_F(TaskPoolTest, EdgeCaseEmptyName) {
    // Test acquiring task with empty name
    auto pool = TaskPool::create();
    auto task = pool->acquire_task("", "");
    ASSERT_NE(task, nullptr);
    EXPECT_EQ(task->get_task_name(), "");
}

TEST_F(TaskPoolTest, EdgeCaseVeryLongName) {
    // Test acquiring task with very long name
    auto pool = TaskPool::create();
    const std::string long_name(1000, 'x');
    auto task = pool->acquire_task(long_name, "graph_name");
    ASSERT_NE(task, nullptr);
    EXPECT_EQ(task->get_task_name(), long_name);
    EXPECT_EQ(task->get_graph_name(), "graph_name");
}

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

} // namespace
