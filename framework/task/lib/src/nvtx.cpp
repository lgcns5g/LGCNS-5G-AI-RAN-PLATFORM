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

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <format>
#include <fstream>
#include <string>

#include <nvtx3/nvToolsExt.h>
#include <quill/LogMacros.h>
#include <unistd.h>

#include "log/rt_log_macros.hpp"
#include "task/function_cache.hpp"
#include "task/nvtx.hpp"
#include "task/task_log.hpp"

namespace {
namespace ft = framework::task;

/**
 * Check if current process is running under nsys or ncu profiler
 *
 * Checks the parent process command line to detect profiler usage.
 *
 * @return True if running under nsys or ncu, false otherwise
 */
[[nodiscard]] __attribute__((no_instrument_function)) bool is_running_under_profiler() {
    try {
        std::ifstream cmdline(std::format("/proc/{}/cmdline", getppid()));
        std::string parent_cmd;
        if (std::getline(cmdline, parent_cmd)) {
            return parent_cmd.find("nsys") != std::string::npos ||
                   parent_cmd.find("ncu") != std::string::npos;
        }
        return false;
    } catch (const std::exception &e) {
        RT_LOGC_ERROR(ft::TaskLog::TaskNvtx, "Failed to check parent process: {}", e.what());
        return false;
    }
}

// Helper function to push a range with color cycling
__attribute__((no_instrument_function)) void range_push(const char *name) {
    // Color definitions for ranges
    static constexpr std::array<std::uint32_t, 7> COLORS = {
            0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff};
    static thread_local std::size_t color_id = 0;

    nvtxEventAttributes_t attribute{};
    attribute.version = NVTX_VERSION;
    attribute.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attribute.colorType = NVTX_COLOR_ARGB;
    attribute.color = COLORS.at(color_id);
    color_id = (color_id + 1) % COLORS.size();
    attribute.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attribute.message.ascii = (name != nullptr) ? name : "Unknown";
    nvtxRangePushEx(&attribute);
}
} // namespace

namespace framework::task {

__attribute__((no_instrument_function)) auto &Nvtx::get_function_cache() {
    static constexpr std::size_t FUNCTION_CACHE_SIZE = 2048;    //!< Function cache size
    static constexpr std::size_t FUNCTION_CACHE_FULL_PCT = 90;  //!< Cache full percentage
    static constexpr std::size_t FUNCTION_CACHE_EVICT_PCT = 25; //!< Cache eviction percentage

    static thread_local ft::FunctionCache cache{
            FUNCTION_CACHE_SIZE, FUNCTION_CACHE_FULL_PCT, FUNCTION_CACHE_EVICT_PCT};
    return cache;
}

// Singleton implementation with Meyers pattern
__attribute__((no_instrument_function)) Nvtx &Nvtx::get_instance() {
    static Nvtx instance{};
    return instance;
}

__attribute__((no_instrument_function)) Nvtx::Nvtx() {
#ifdef NVTX_ENABLE
    if (is_running_under_profiler()) {
        RT_LOGC_DEBUG(
                ft::TaskLog::TaskNvtx,
                "Profiler (nsys or ncu) detected in parent process chain, "
                "enabling NVTX");
        nvtx_enabled_.store(true, std::memory_order_release);
    } else {
        RT_LOGC_DEBUG(
                ft::TaskLog::TaskNvtx,
                "No profiler detected in parent process chain, NVTX disabled");
        nvtx_enabled_.store(false, std::memory_order_release);
    }
#else
    nvtx_enabled_.store(false, std::memory_order_release);
#endif
}

__attribute__((no_instrument_function)) bool Nvtx::is_enabled() {
#ifdef NVTX_ENABLE
    return get_instance().nvtx_enabled_.load(std::memory_order_acquire);
#else
    return false; // Always disabled when NVTX_ENABLE is not defined
#endif
}

__attribute__((no_instrument_function)) Nvtx::Stats Nvtx::get_stats() {
    auto &instance = get_instance();
    auto &cache = get_function_cache();

    return Stats{
            .total_functions = instance.count_.load(),
            .resolved_functions = instance.resolved_count_.load(),
            .fallback_functions = instance.fallback_count_.load(),
            .cache_entries = cache.size(),
            .cache_attempts = cache.get_cache_attempts(),
            .cache_hits = cache.get_cache_hits(),
            .cache_misses = cache.get_cache_misses(),
            .hit_ratio = cache.get_hit_ratio()};
}

__attribute__((no_instrument_function)) const char *Nvtx::get_function_name(void *func) {
    auto &cache = get_function_cache();

    // Check cache first
    const char *cached_name = cache.get(func);

    if (cached_name != nullptr) {
        return cached_name;
    }

    // Not found, resolve and add to cache with demangling
    cache.add_with_demangling(func);

    // Try to get the name again after adding
    // Returns nullptr if demangling failed, caller handles fallback
    return cache.get(func);
}

__attribute__((no_instrument_function)) void Nvtx::increment_counters(const bool resolved) {
    auto &instance = get_instance();
    instance.count_.fetch_add(1, std::memory_order_relaxed);
    if (resolved) {
        instance.resolved_count_.fetch_add(1, std::memory_order_relaxed);
    } else {
        instance.fallback_count_.fetch_add(1, std::memory_order_relaxed);
    }
}

__attribute__((no_instrument_function)) NvtxScopedRange::NvtxScopedRange(const char *name) {
#ifdef NVTX_ENABLE
    range_push(name);
#endif
}

__attribute__((no_instrument_function)) NvtxScopedRange::~NvtxScopedRange() {
#ifdef NVTX_ENABLE
    nvtxRangePop();
#endif
}

} // namespace framework::task

// Implementation using thread-local function cache
extern "C" {
// NOLINTBEGIN(bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp)
__attribute__((no_instrument_function)) void
// NOLINTNEXTLINE(readability-identifier-naming)
__cyg_profile_func_enter(void *func, [[maybe_unused]] void *caller) {
    if (!ft::Nvtx::is_enabled()) {
        return;
    }

    // Try to get function name from cache or resolve it
    const char *name = ft::Nvtx::get_function_name(func);

    if (name != nullptr) {
        // Found or resolved successfully
        range_push(name);
        ft::Nvtx::increment_counters(true);
    } else {
        // Fallback to address format if demangling failed
        static constexpr std::size_t FALLBACK_BUFFER_SIZE =
                128; //!< Buffer size for "func_0x[address]" format (sufficient for
                     //!< 64-bit addresses)
        static thread_local std::array<char, FALLBACK_BUFFER_SIZE> buffer{};

        auto result = std::format_to_n(
                buffer.begin(),
                buffer.size() - 1,
                "func_0x{:x}",
                // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
                reinterpret_cast<std::uintptr_t>(func));
        // Null terminate at the actual written position
        *result.out = '\0';

        range_push(buffer.data());
        ft::Nvtx::increment_counters(false);
    }
}

__attribute__((no_instrument_function)) void
// NOLINTNEXTLINE(readability-identifier-naming)
__cyg_profile_func_exit([[maybe_unused]] void *func, [[maybe_unused]] void *caller) {
    if (!ft::Nvtx::is_enabled()) {
        return;
    }
    nvtxRangePop();
}
// NOLINTEND(bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp)
}
