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

#ifndef FRAMEWORK_TASK_NVTX_HPP
#define FRAMEWORK_TASK_NVTX_HPP

#include <atomic>
#include <string_view>

#include <nvtx3/nvToolsExt.h>

#include "log/rt_log_macros.hpp"

namespace framework::task {
/**
 * Singleton NVTX profiling manager
 *
 * Automatically detects if nsys or ncu profiling is active by examining
 * the parent process chain on first access and configures NVTX accordingly.
 * Thread-safe singleton implementation.
 */
class Nvtx {
public:
    /**
     * Check if NVTX profiling is currently enabled
     * @return True if NVTX profiling is enabled
     */
    [[nodiscard]] static bool is_enabled();

    /**
     * NVTX profiling statistics
     */
    struct Stats {
        std::uint64_t total_functions{};    //!< Total function calls
        std::uint64_t resolved_functions{}; //!< Successfully resolved function names
        std::uint64_t fallback_functions{}; //!< Functions that used fallback formatting
        std::size_t cache_entries{};        //!< Number of entries in function cache
        std::uint64_t cache_attempts{};     //!< Total cache lookup attempts
        std::uint64_t cache_hits{};         //!< Cache hits
        std::uint64_t cache_misses{};       //!< Cache misses
        double hit_ratio{};                 //!< Cache hit ratio percentage
    };

    /**
     * Get NVTX profiling statistics
     * @return Statistics structure with current values
     */
    [[nodiscard]] static Stats get_stats();

    /**
     * Get function name from cache or resolve it (for C function access)
     * @param[in] func Function pointer
     * @return Function name or nullptr if not found
     */
    [[nodiscard]] static const char *get_function_name(void *func);

    /**
     * Increment function call counters (for C function access)
     * @param[in] resolved True if function was resolved from symbols
     */
    static void increment_counters(bool resolved);

    // Non-copyable, non-movable
    Nvtx(const Nvtx &) = delete;
    Nvtx &operator=(const Nvtx &) = delete;
    Nvtx(Nvtx &&) = delete;
    Nvtx &operator=(Nvtx &&) = delete;

private:
    /// Private constructor for singleton
    Nvtx();

    /// Destructor
    ~Nvtx() = default;

    /**
     * Get the singleton instance
     * @return Reference to the singleton Nvtx instance
     */
    [[nodiscard]] static Nvtx &get_instance();

    /// Get reference to thread-local function cache
    [[nodiscard]] static auto &get_function_cache();

    // Atomic flags and counters
    std::atomic<bool> nvtx_enabled_{false};
    std::atomic<std::uint64_t> count_{0};
    std::atomic<std::uint64_t> resolved_count_{0};
    std::atomic<std::uint64_t> fallback_count_{0};
};

/**
 * RAII wrapper for NVTX profiling ranges
 *
 * Creates a scoped profiling range that automatically ends when destroyed.
 * Used for performance profiling with NVIDIA Nsight tools.
 */
class NvtxScopedRange final {
public:
    /**
     * Constructor - starts profiling range
     * @param[in] name Range name for profiler display
     */
    explicit NvtxScopedRange(const char *name);

    /// Destructor - ends profiling range
    ~NvtxScopedRange();

    // Non-copyable, non-movable
    NvtxScopedRange(const NvtxScopedRange &) = delete;
    NvtxScopedRange &operator=(const NvtxScopedRange &) = delete;
    NvtxScopedRange(NvtxScopedRange &&) = delete;
    NvtxScopedRange &operator=(NvtxScopedRange &&) = delete;
};

// Instrumentation macros
#ifdef NVTX_ENABLE
// NOLINTBEGIN(cppcoreguidelines-macro-usage)
#define NVTX_CONCAT_IMPL(x, y) x##y
#define NVTX_CONCAT(x, y) NVTX_CONCAT_IMPL(x, y)
#define NVTX_RANGE(name)                                                                           \
    const framework::task::NvtxScopedRange NVTX_CONCAT(nvtx_scope_, __LINE__)(name)
#define NVTX_FUNCTION() NVTX_RANGE(__FUNCTION__)
// NOLINTEND(cppcoreguidelines-macro-usage)
#else
#define NVTX_RANGE(name)
#define NVTX_FUNCTION()
#endif

} // namespace framework::task

/// @cond HIDE_FROM_DOXYGEN
// Must be in global namespace for quill to find it
// cppcheck-suppress functionStatic
RT_LOGGABLE_DEFERRED_FORMAT(
        framework::task::Nvtx::Stats,
        "Total functions: {} (Resolved: {}, Fallback: {}), Cache entries: {}, "
        "Cache stats: {} attempts, {} hits, {} misses, {:.2f}% hit rate",
        obj.total_functions,
        obj.resolved_functions,
        obj.fallback_functions,
        obj.cache_entries,
        obj.cache_attempts,
        obj.cache_hits,
        obj.cache_misses,
        obj.hit_ratio)
/// @endcond

#endif // FRAMEWORK_TASK_NVTX_HPP
