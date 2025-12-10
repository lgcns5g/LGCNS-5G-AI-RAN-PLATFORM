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

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <string_view>
#include <utility>

#include <cxxabi.h>
#include <dlfcn.h>
#include <parallel_hashmap/phmap.h>

#include <gsl-lite/gsl-lite.hpp>

#include "task/flat_map.hpp"
#include "task/function_cache.hpp"

namespace framework::task {

// FunctionCache implementation
FunctionCache::FunctionCache(
        std::size_t max_size, std::size_t full_percentage, std::size_t evict_percentage)
        : cache_(max_size, GrowthStrategy::Evict) {
    // Configure eviction percentages for the cache
    cache_.set_eviction_percentages(full_percentage, evict_percentage);
}

const char *FunctionCache::get(void *addr) {
    cache_attempts_.fetch_add(1, std::memory_order_relaxed);

    auto it = cache_.find(addr);
    if (it != cache_.end()) {
        cache_hits_.fetch_add(1, std::memory_order_relaxed);
        return it->second.c_str();
    }

    cache_misses_.fetch_add(1, std::memory_order_relaxed);
    return nullptr;
}

void FunctionCache::add_with_demangling(void *addr) {
    Dl_info info;
    if (dladdr(addr, &info) == 0 || info.dli_sname == nullptr) {
        return;
    }

    // Get reference to the fixed string we'll store
    NameString &name_entry = cache_[addr]; // Creates entry if doesn't exist

    // SAFE approach: Let __cxa_demangle allocate its own buffer
    // to avoid realloc() being called on our std::array buffer
    int status = 0;
    gsl_lite::owner<char *> demangled = abi::__cxa_demangle(
            info.dli_sname,
            nullptr, // Let __cxa_demangle allocate
            nullptr, // We don't need the size
            &status);

    if (status == 0 && demangled != nullptr) {
        // Success: copy demangled name to our fixed buffer
        name_entry = demangled;
        // NOLINTNEXTLINE(cppcoreguidelines-no-malloc,hicpp-no-malloc,cppcoreguidelines-owning-memory)
        std::free(demangled); // Must free the allocated buffer
    } else {
        // Demangling failed: store original mangled name
        name_entry = info.dli_sname;
    }
}

void FunctionCache::add(void *addr, std::string_view name_str) {
    cache_[addr] = NameString(name_str);
}

void FunctionCache::add(void *addr, const char *name_str) {
    if (name_str != nullptr) {
        cache_[addr] = NameString(std::string_view{name_str});
    } else {
        cache_[addr] = NameString{};
    }
}

void FunctionCache::clear() { cache_.clear(); }

std::size_t FunctionCache::size() const { return cache_.size(); }

std::uint64_t FunctionCache::get_cache_hits() const {
    return cache_hits_.load(std::memory_order_relaxed);
}

std::uint64_t FunctionCache::get_cache_misses() const {
    return cache_misses_.load(std::memory_order_relaxed);
}

std::uint64_t FunctionCache::get_cache_attempts() const {
    return cache_attempts_.load(std::memory_order_relaxed);
}

void FunctionCache::evict_percentage(std::size_t percentage) {
    // Validation is handled by FlatMap::evict_percentage
    cache_.evict_percentage(percentage);
}

double FunctionCache::get_hit_ratio() const {
    const auto attempts = cache_attempts_.load(std::memory_order_relaxed);
    const auto hits = cache_hits_.load(std::memory_order_relaxed);
    return attempts > 0 ? (static_cast<double>(hits) / static_cast<double>(attempts)) * 100.0 : 0.0;
}

} // namespace framework::task
