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

#include <algorithm>  // for fill_n, max
#include <atomic>     // for atomic
#include <chrono>     // for microseconds, nanoseconds
#include <cstddef>    // for size_t
#include <cstdint>    // for int64_t
#include <functional> // for function
#include <string>     // for string, allocator
#include <thread>     // for thread, sleep_for, yield
#include <vector>     // for vector

#include <benchmark/benchmark.h> // for State, BENCHMARK, Benchmark, BENCHM...
#include <quill/LogMacros.h>     // for QUILL_LOG_INFO, QUILL_LOG_TRACE_L1
#include <wise_enum_detail.h>    // for WISE_ENUM_IMPL_IIF_0
#include <wise_enum_generated.h> // for WISE_ENUM_IMPL_LOOP_4

#include "log/components.hpp" // for LogLevel, register_component, DECLA...
#include "log/rt_log.hpp"     // for RT_LOG_INFO, RT_LOGC_INFO, LoggerCo...
#include "log/rt_log_macros.hpp"

// Define benchmark-specific components and events
DECLARE_LOG_COMPONENT(BenchComponent, Core, Performance, Monitor, ThreadPool);

DECLARE_LOG_EVENT(BenchEvent, AppStart, BenchmarkStart, BenchmarkEnd, MeasurementPoint);

namespace {

namespace fl = ::framework::log;

// Global constants used across multiple functions
constexpr int64_t BURST_SIZE_1000 = 1000; //!< Number of messages per burst benchmark
constexpr int64_t PAUSE_DURATION_MICROSECONDS =
        250; //!< Pause duration between benchmark iterations
constexpr int64_t MESSAGES_PER_THREAD_DEFAULT =
        100; //!< Default messages per thread in multi-threaded benchmarks

/**
 * Configure logger with optimized settings for benchmarking
 *
 * Sets up the logger to output to /dev/null with specified settings
 * to minimize I/O overhead during performance measurements.
 *
 * @param[in] enable_timestamps Whether to include timestamps in log output
 * @param[in] min_level Minimum log level to process
 */
void configure_benchmark_logger(
        bool enable_timestamps = true, fl::LogLevel min_level = fl::LogLevel::Info) {
    fl::Logger::configure(
            fl::LoggerConfig::file("/dev/null", min_level)
                    .with_cpu_affinity(1)                                     // Pin to specific CPU
                    .with_backend_sleep_duration(std::chrono::nanoseconds(0)) // No backend sleep
                    .with_timestamps(enable_timestamps));

    // Register components with DEBUG level to ensure they're active
    fl::register_component<BenchComponent>(fl::LogLevel::Debug);
}

/**
 * Common benchmark function for different logging types
 *
 * Executes a logging function repeatedly and measures performance,
 * with optional message bursts and timing controls.
 *
 * @param[in,out] state Google Benchmark state object
 * @param[in] log_func Logging function to benchmark
 * @param[in] messages_per_iteration Number of messages to log per iteration
 */
void common_logging(
        benchmark::State &state,
        const std::function<void(int64_t, const std::string &)> &log_func,
        const int64_t messages_per_iteration = 1) {
    const std::string test_message = "Test log message for benchmarking";
    const auto pause_duration = std::chrono::microseconds(PAUSE_DURATION_MICROSECONDS);
    int64_t counter = 0;

    for (auto _ : state) { // NOLINT
        for (int64_t i = 0; i < messages_per_iteration; ++i) {
            log_func(counter++, test_message);
        }

        if (messages_per_iteration > 1) {
            state.PauseTiming();
            std::this_thread::sleep_for(pause_duration);
            state.ResumeTiming();
        }
    }

    state.SetItemsProcessed(state.iterations() * messages_per_iteration);
    fl::Logger::flush();
}

/**
 * Multi-threaded benchmark function
 *
 * Creates multiple threads that log concurrently to measure
 * thread safety and performance under concurrent load.
 *
 * @param[in,out] state Google Benchmark state object
 * @param[in] log_func Logging function to benchmark
 * @param[in] num_threads Number of concurrent threads to create
 * @param[in] messages_per_thread Number of messages each thread should log
 */
void common_multi_threaded_logging(
        benchmark::State &state,
        std::function<void(int64_t, const std::string &)> log_func,
        const int64_t num_threads,
        const int64_t messages_per_thread) {
    constexpr int64_t THREAD_COUNTER_OFFSET = 10000;
    const std::string test_message = "Multi-threaded test message";

    for (auto _ : state) { // NOLINT
        std::vector<std::thread> threads;
        threads.reserve(static_cast<std::size_t>(num_threads));
        std::atomic<bool> start_flag{false};

        for (int64_t thread_id = 0; thread_id < num_threads; ++thread_id) {
            threads.emplace_back(
                    [&log_func, &test_message, &start_flag, messages_per_thread, thread_id]() {
                        while (!start_flag.load()) {
                            std::this_thread::yield();
                        }

                        int64_t local_counter = thread_id * THREAD_COUNTER_OFFSET;
                        for (int64_t i = 0; i < messages_per_thread; ++i) {
                            log_func(local_counter++, test_message);
                        }
                    });
        }

        start_flag.store(true);

        for (auto &thread : threads) {
            thread.join();
        }

        state.PauseTiming();
        std::this_thread::sleep_for(std::chrono::microseconds(PAUSE_DURATION_MICROSECONDS));
        state.ResumeTiming();
    }

    state.SetItemsProcessed(state.iterations() * num_threads * messages_per_thread);
    fl::Logger::flush();
}

// Single message benchmarks

/**
 * Benchmark basic RT_LOG_INFO macro performance
 *
 * @param[in,out] state Google Benchmark state object
 */
void basic_logging(benchmark::State &state) {
    configure_benchmark_logger(true, fl::LogLevel::Info);
    common_logging(state, [](int64_t counter, const std::string &msg) {
        RT_LOG_INFO("Benchmark message #{}: {}", counter, msg);
    });
}

/**
 * Benchmark component-based logging performance
 *
 * @param[in,out] state Google Benchmark state object
 */
void component_logging(benchmark::State &state) {
    configure_benchmark_logger(true, fl::LogLevel::Info);
    common_logging(state, [](int64_t counter, const std::string &msg) {
        RT_LOGC_INFO(BenchComponent::Core, "Benchmark message #{}: {}", counter, msg);
    });
}

/**
 * Benchmark event-based logging performance
 *
 * @param[in,out] state Google Benchmark state object
 */
void event_logging(benchmark::State &state) {
    configure_benchmark_logger(true, fl::LogLevel::Info);
    common_logging(state, [](int64_t counter, const std::string &msg) {
        RT_LOGE_INFO(BenchEvent::BenchmarkStart, "Benchmark message #{}: {}", counter, msg);
    });
}

/**
 * Benchmark combined component and event logging performance
 *
 * @param[in,out] state Google Benchmark state object
 */
void component_event_logging(benchmark::State &state) {
    configure_benchmark_logger(true, fl::LogLevel::Info);
    common_logging(state, [](int64_t counter, const std::string &msg) {
        RT_LOGEC_INFO(
                BenchComponent::Core,
                BenchEvent::BenchmarkStart,
                "Benchmark message #{}: {}",
                counter,
                msg);
    });
}

/**
 * Benchmark logging performance without timestamp formatting
 *
 * @param[in,out] state Google Benchmark state object
 */
void basic_logging_no_timestamps(benchmark::State &state) {
    configure_benchmark_logger(false, fl::LogLevel::Info);
    common_logging(state, [](int64_t counter, const std::string &msg) {
        RT_LOG_INFO("Benchmark message #{}: {}", counter, msg);
    });
}

void basic_logging_filtered(benchmark::State &state) {
    configure_benchmark_logger(true, fl::LogLevel::Warn);
    common_logging(state, [](int64_t counter, const std::string &msg) {
        RT_LOG_INFO("Benchmark message #{}: {}", counter, msg);
    });
}

void trace_logging_filtered(benchmark::State &state) {
    configure_benchmark_logger(true, fl::LogLevel::Info);
    common_logging(state, [](int64_t counter, const std::string &msg) {
        RT_LOG_TRACE_L1("Benchmark message #{}: {}", counter, msg);
    });
}

void trace_component_logging_filtered(benchmark::State &state) {
    configure_benchmark_logger(true, fl::LogLevel::Info);
    common_logging(state, [](int64_t counter, const std::string &msg) {
        RT_LOGC_TRACE_L1(BenchComponent::Core, "Benchmark message #{}: {}", counter, msg);
    });
}

/**
 * Benchmark logging with parameter count variations
 *
 * Tests performance impact of different numbers of format parameters.
 *
 * @param[in,out] state Google Benchmark state object with parameter range
 */
void parameterized_logging(benchmark::State &state) {
    configure_benchmark_logger(true, fl::LogLevel::Info);
    const int64_t num_params = state.range(0);

    common_logging(state, [num_params](int64_t counter, const std::string &msg) {
        // NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
        switch (num_params) {
        case 0:
            RT_LOG_INFO("Benchmark message");
            break;
        case 1:
            RT_LOG_INFO("Benchmark message {}", counter);
            break;
        case 2:
            RT_LOG_INFO("Benchmark message {} {}", counter, msg);
            break;
        case 3:
            RT_LOG_INFO("Benchmark message {} {} {}", counter, msg, "param3");
            break;
        case 4:
            RT_LOG_INFO("Benchmark message {} {} {} {}", counter, msg, "param3", 42);
            break;
        case 5:
            RT_LOG_INFO("Benchmark message {} {} {} {} {}", counter, msg, "param3", 42, 3.14);
            break;
        case 6:
            RT_LOG_INFO(
                    "Benchmark message {} {} {} {} {} {}", counter, msg, "param3", 42, 3.14, true);
            break;
        case 7:
            RT_LOG_INFO(
                    "Benchmark message {} {} {} {} {} {} {}",
                    counter,
                    msg,
                    "param3",
                    42,
                    3.14,
                    true,
                    "param7");
            break;
        case 8:
            RT_LOG_INFO(
                    "Benchmark message {} {} {} {} {} {} {} {}",
                    counter,
                    msg,
                    "param3",
                    42,
                    3.14,
                    true,
                    "param7",
                    999);
            break;
        case 9:
            RT_LOG_INFO(
                    "Benchmark message {} {} {} {} {} {} {} {} {}",
                    counter,
                    msg,
                    "param3",
                    42,
                    3.14,
                    true,
                    "param7",
                    999,
                    2.71);
            break;
        case 10:
            RT_LOG_INFO(
                    "Benchmark message {} {} {} {} {} {} {} {} {} {}",
                    counter,
                    msg,
                    "param3",
                    42,
                    3.14,
                    true,
                    "param7",
                    999,
                    2.71,
                    false);
            break;
        default:
            RT_LOG_INFO("Benchmark message {} {}", counter, msg);
            break;
        }
        // NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
    });
}

void parameterized_component_logging(benchmark::State &state) {
    configure_benchmark_logger(true, fl::LogLevel::Info);
    const int64_t num_params = state.range(0);

    common_logging(state, [num_params](int64_t counter, const std::string &msg) {
        // NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
        switch (num_params) {
        case 0:
            RT_LOGC_INFO(BenchComponent::Core, "Benchmark message");
            break;
        case 1:
            RT_LOGC_INFO(BenchComponent::Core, "Benchmark message {}", counter);
            break;
        case 2:
            RT_LOGC_INFO(BenchComponent::Core, "Benchmark message {} {}", counter, msg);
            break;
        case 3:
            RT_LOGC_INFO(
                    BenchComponent::Core, "Benchmark message {} {} {}", counter, msg, "param3");
            break;
        case 4:
            RT_LOGC_INFO(
                    BenchComponent::Core,
                    "Benchmark message {} {} {} {}",
                    counter,
                    msg,
                    "param3",
                    42);
            break;
        case 5:
            RT_LOGC_INFO(
                    BenchComponent::Core,
                    "Benchmark message {} {} {} {} {}",
                    counter,
                    msg,
                    "param3",
                    42,
                    3.14);
            break;
        default:
            RT_LOGC_INFO(BenchComponent::Core, "Benchmark message {} {}", counter, msg);
            break;
        }
        // NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
    });
}

// Burst benchmarks

/**
 * Benchmark burst logging with 1000 messages per iteration
 *
 * @param[in,out] state Google Benchmark state object
 */
void burst_basic_logging(benchmark::State &state) {
    configure_benchmark_logger(true, fl::LogLevel::Info);
    common_logging(
            state,
            [](int64_t counter, const std::string &msg) {
                RT_LOG_INFO("Burst message #{}: {}", counter, msg);
            },
            BURST_SIZE_1000);
}

void burst_component_logging(benchmark::State &state) {
    configure_benchmark_logger(true, fl::LogLevel::Info);
    common_logging(
            state,
            [](int64_t counter, const std::string &msg) {
                RT_LOGC_INFO(BenchComponent::Core, "Burst message #{}: {}", counter, msg);
            },
            BURST_SIZE_1000);
}

void burst_event_logging(benchmark::State &state) {
    configure_benchmark_logger(true, fl::LogLevel::Info);
    common_logging(
            state,
            [](int64_t counter, const std::string &msg) {
                RT_LOGE_INFO(BenchEvent::BenchmarkStart, "Burst message #{}: {}", counter, msg);
            },
            BURST_SIZE_1000);
}

void burst_component_event_logging(benchmark::State &state) {
    configure_benchmark_logger(true, fl::LogLevel::Info);
    common_logging(
            state,
            [](int64_t counter, const std::string &msg) {
                RT_LOGEC_INFO(
                        BenchComponent::Core,
                        BenchEvent::BenchmarkStart,
                        "Burst message #{}: {}",
                        counter,
                        msg);
            },
            BURST_SIZE_1000);
}

// Multi-threaded benchmarks

/**
 * Benchmark multi-threaded basic logging performance
 *
 * @param[in,out] state Google Benchmark state object with thread count range
 */
void multi_threaded_basic_logging(benchmark::State &state) {
    configure_benchmark_logger(true, fl::LogLevel::Info);
    const int64_t num_threads = state.range(0);
    common_multi_threaded_logging(
            state,
            [](int64_t counter, const std::string &msg) {
                RT_LOG_INFO("MT message #{}: {}", counter, msg);
            },
            num_threads,
            MESSAGES_PER_THREAD_DEFAULT);
}

void multi_threaded_component_logging(benchmark::State &state) {
    configure_benchmark_logger(true, fl::LogLevel::Info);
    const int64_t num_threads = state.range(0);
    common_multi_threaded_logging(
            state,
            [](int64_t counter, const std::string &msg) {
                RT_LOGC_INFO(BenchComponent::Core, "MT message #{}: {}", counter, msg);
            },
            num_threads,
            MESSAGES_PER_THREAD_DEFAULT);
}

void multi_threaded_burst_logging(benchmark::State &state) {
    configure_benchmark_logger(true, fl::LogLevel::Info);
    const int64_t num_threads = state.range(0);
    // NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
    constexpr int64_t MESSAGES_PER_THREAD_BURST = 1000;
    // NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
    common_multi_threaded_logging(
            state,
            [](int64_t counter, const std::string &msg) {
                RT_LOG_INFO("MT burst #{}: {}", counter, msg);
            },
            num_threads,
            MESSAGES_PER_THREAD_BURST);
}

} // anonymous namespace

// Register benchmarks

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

// Burst benchmarks
BENCHMARK(burst_basic_logging);
BENCHMARK(burst_component_logging);
BENCHMARK(burst_event_logging);
BENCHMARK(burst_component_event_logging);

// Single message benchmarks
BENCHMARK(basic_logging);
BENCHMARK(component_logging);
BENCHMARK(event_logging);
BENCHMARK(component_event_logging);

// Variants
BENCHMARK(basic_logging_no_timestamps);
BENCHMARK(basic_logging_filtered);
BENCHMARK(trace_logging_filtered);
BENCHMARK(trace_component_logging_filtered);

// Parameterized benchmarks
BENCHMARK(parameterized_logging)->DenseRange(0, 10);
BENCHMARK(parameterized_component_logging)->DenseRange(1, 5);

// Multi-threaded benchmarks
BENCHMARK(multi_threaded_basic_logging)->DenseRange(1, 8);
BENCHMARK(multi_threaded_component_logging)->DenseRange(2, 4);
BENCHMARK(multi_threaded_burst_logging)->DenseRange(2, 4);

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

BENCHMARK_MAIN();
