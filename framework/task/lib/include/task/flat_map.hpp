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

#ifndef FRAMEWORK_TASK_FLAT_MAP_HPP
#define FRAMEWORK_TASK_FLAT_MAP_HPP

#include <stdexcept>
#include <string>
#include <string_view>

#include <parallel_hashmap/phmap.h>

#include <wise_enum.h>

#include "log/rt_log_macros.hpp"
#include "task/task_log.hpp"

namespace framework::task {

/**
 * Growth strategy when FlatMap reaches capacity threshold
 */
enum class GrowthStrategy {
    Allocate, //!< Allow underlying container to grow (current default behavior)
    Evict,    //!< Evict entries when threshold reached (prevents growth)
    Throw     //!< Throw exception when threshold would be exceeded
};

} // namespace framework::task

// WISE_ENUM_ADAPT must be called at global namespace scope for ADL to work
WISE_ENUM_ADAPT(framework::task::GrowthStrategy, Allocate, Evict, Throw)

namespace framework::task {

namespace detail {
/**
 * Validate that a percentage value is in the valid range (1-100)
 *
 * @param percentage The percentage value to validate
 * @param param_name The parameter name for error messages
 * @throws std::invalid_argument if percentage is not between 1 and 100
 */
inline void validate_percentage(const std::size_t percentage, const std::string_view param_name) {
    static constexpr std::size_t MAX_PERCENTAGE = 100;
    if (percentage == 0 || percentage > MAX_PERCENTAGE) {
        throw std::invalid_argument(std::string(param_name) + " must be between 1 and 100");
    }
}

/**
 * Validate that a size value is greater than zero
 *
 * @param size The size value to validate
 * @param param_name The parameter name for error messages
 * @throws std::invalid_argument if size is zero
 */
inline void validate_size(const std::size_t size, const std::string_view param_name) {
    if (size == 0) {
        throw std::invalid_argument(std::string(param_name) + " must be greater than 0");
    }
}
} // namespace detail

/**
 * High-performance flat hash map with configurable growth strategies
 *
 * Uses phmap::flat_hash_map internally with configurable behavior when the
 * container reaches a threshold: Allocate (allow growth), Evict (remove
 * entries), or Throw (exception on overflow).
 */
template <typename Key, typename Value> class FlatMap final {
private:
    phmap::flat_hash_map<Key, Value> map_{};
    std::size_t max_size_{};
    std::size_t eviction_threshold_{};
    std::size_t eviction_percentage_{};
    GrowthStrategy growth_strategy_{};

    void handle_capacity() {
        switch (growth_strategy_) {
        case GrowthStrategy::Allocate:
            // Allow underlying container to grow - no threshold check needed
            break;
        case GrowthStrategy::Evict:
            if (map_.size() >= eviction_threshold_) {
                evict_percentage(eviction_percentage_);
            }
            break;
        case GrowthStrategy::Throw:
            if (map_.size() >= max_size_) {
                throw std::length_error("FlatMap maximum capacity exceeded");
            }
            break;
        default:
            throw std::logic_error("Invalid growth strategy");
        }
    }

    static constexpr std::size_t DEFAULT_MAX_SIZE = 64;
    static constexpr std::size_t DEFAULT_EVICTION_THRESHOLD_PERCENT = 90;
    static constexpr std::size_t DEFAULT_EVICTION_PERCENTAGE = 25;
    static constexpr std::size_t PERCENTAGE_DIVISOR = 100;

public:
    /**
     * Constructor
     *
     * @param max_size Maximum number of entries for capacity calculations (must
     * be > 0)
     * @param growth_strategy Strategy to use when capacity limits are reached:
     *                       - Allocate: Allow container to grow (default)
     *                       - Evict: Remove entries when threshold reached (use
     * set_eviction_percentages())
     *                       - Throw: Throw exception when max_size would be
     * exceeded
     */
    explicit FlatMap(
            const std::size_t max_size = DEFAULT_MAX_SIZE,
            const GrowthStrategy growth_strategy = GrowthStrategy::Allocate)
            : max_size_(max_size), eviction_threshold_(
                                           (max_size * DEFAULT_EVICTION_THRESHOLD_PERCENT) /
                                           PERCENTAGE_DIVISOR),  // Default 90%
              eviction_percentage_(DEFAULT_EVICTION_PERCENTAGE), // Default 25%
              growth_strategy_(growth_strategy) {
        detail::validate_size(max_size, "max_size");
        map_.reserve(max_size_);
    }

    /**
     * Evict a percentage of entries from the container
     *
     * @param percentage Percentage (1-100) of entries to remove
     */
    void evict_percentage(std::size_t percentage) {
        detail::validate_percentage(percentage, "percentage");
        if (map_.empty()) {
            return;
        }

        const std::size_t target_removals = (map_.size() * percentage) / PERCENTAGE_DIVISOR;
        if (target_removals == 0) {
            return; // avoid division by zero
        }
        const std::size_t skip_factor = map_.size() / target_removals;

        if (skip_factor == 0) {
            return;
        }

        std::size_t removed_count = 0;
        std::size_t counter = 0;

        // Iterate and remove every skip_factor-th element
        auto it = map_.begin();
        while (it != map_.end() && removed_count < target_removals) {
            if (counter % skip_factor == 0) {
                it = map_.erase(it); // erase returns next iterator
                ++removed_count;
            } else {
                ++it;
            }
            ++counter;
        }

        if (removed_count > 0) {
            RT_LOGC_DEBUG(
                    TaskLog::FlatMap,
                    "FlatMap: Evicted {} entries ({}%). Remaining: {}",
                    removed_count,
                    percentage,
                    map_.size());
        }
    }

    /**
     * Access element with automatic insertion if key doesn't exist
     * @param[in] key Key to access
     * @return Reference to the value associated with key
     */
    [[nodiscard]] Value &operator[](const Key &key) {
        if (const auto it = map_.find(key); it != map_.end()) {
            // Key exists, return existing value without capacity check
            return it->second;
        }

        // Key doesn't exist, check capacity before insertion
        handle_capacity();
        return map_[key]; // This will insert default-constructed value
    }

    /**
     * Insert key-value pair
     * @param[in] value Key-value pair to insert
     * @return Iterator to inserted element and success flag
     */
    std::pair<typename phmap::flat_hash_map<Key, Value>::iterator, bool>
    insert(const std::pair<Key, Value> &value) {
        auto it = map_.find(value.first);
        if (it != map_.end()) {
            // Key already exists, return existing iterator with false flag
            return {it, false};
        }

        // Key doesn't exist, check capacity before insertion
        handle_capacity();
        return map_.insert(value);
    }

    /**
     * Emplace element with in-place construction
     * @param[in] args Arguments for constructing the element
     * @return Iterator to inserted element and success flag
     */
    template <typename... Args>
    std::pair<typename phmap::flat_hash_map<Key, Value>::iterator, bool> emplace(Args &&...args) {
        handle_capacity();
        return map_.emplace(std::forward<Args>(args)...);
    }

    /**
     * Find element by key (const version)
     * @param[in] key Key to find
     * @return Iterator to element if found
     */
    [[nodiscard]] auto find(const Key &key) const -> decltype(map_.find(key)) {
        return map_.find(key);
    }

    /**
     * Find element by key (non-const version)
     * @param[in] key Key to find
     * @return Iterator to element if found
     */
    [[nodiscard]] auto find(const Key &key) -> decltype(map_.find(key)) { return map_.find(key); }

    /**
     * Get const iterator to beginning
     * @return Const iterator to first element
     */
    [[nodiscard]] auto begin() const -> decltype(map_.begin()) { return map_.begin(); }

    /**
     * Get const iterator to end
     * @return Const iterator past last element
     */
    [[nodiscard]] auto end() const -> decltype(map_.end()) { return map_.end(); }

    /**
     * Get iterator to beginning
     * @return Iterator to first element
     */
    [[nodiscard]] auto begin() -> decltype(map_.begin()) { return map_.begin(); }

    /**
     * Get iterator to end
     * @return Iterator past last element
     */
    [[nodiscard]] auto end() -> decltype(map_.end()) { return map_.end(); }

    /**
     * Get number of elements
     * @return Number of elements in map
     */
    [[nodiscard]] std::size_t size() const { return map_.size(); }

    /**
     * Check if map is empty
     * @return True if map is empty
     */
    [[nodiscard]] bool empty() const { return map_.empty(); }

    /**
     * Get current capacity
     * @return Current capacity of underlying container
     */
    [[nodiscard]] std::size_t capacity() const { return map_.capacity(); }

    /**
     * Get maximum allowed size
     * @return Maximum number of elements allowed
     */
    [[nodiscard]] std::size_t max_size() const { return max_size_; }

    /**
     * Get current growth strategy
     * @return Current growth strategy
     */
    [[nodiscard]] GrowthStrategy growth_strategy() const { return growth_strategy_; }

    /**
     * Remove element by key
     * @param[in] key Key to remove
     */
    void erase(const Key &key) { map_.erase(key); }

    /// Clear all elements
    void clear() { map_.clear(); }

    /**
     * Get const reference to underlying map
     * @return Const reference to underlying phmap::flat_hash_map
     */
    [[nodiscard]] const phmap::flat_hash_map<Key, Value> &underlying() const { return map_; }

    /**
     * Get reference to underlying map
     * @return Reference to underlying phmap::flat_hash_map
     */
    phmap::flat_hash_map<Key, Value> &underlying() { return map_; }

    /**
     * Set maximum size
     * @param[in] new_max_size Maximum number of elements
     */
    void set_max_size(std::size_t new_max_size) {
        detail::validate_size(new_max_size, "new_max_size");

        // Calculate current full_percentage before updating max_size_
        const std::size_t current_full_percentage =
                (eviction_threshold_ * PERCENTAGE_DIVISOR) / max_size_;

        max_size_ = new_max_size;

        // Recalculate eviction_threshold_ using the same percentage but new
        // max_size_
        eviction_threshold_ = (max_size_ * current_full_percentage) / PERCENTAGE_DIVISOR;

        map_.reserve(new_max_size);
    }

    /**
     * Set both eviction percentages for the Evict strategy
     * @param[in] full_percentage Percentage full (1-100) at which eviction
     * triggers
     * @param[in] evict_percentage Percentage (1-100) of entries to evict when
     * threshold reached
     * @throws std::invalid_argument if percentages are invalid
     */
    void set_eviction_percentages(std::size_t full_percentage, std::size_t evict_percentage) {
        detail::validate_percentage(full_percentage, "full_percentage");
        detail::validate_percentage(evict_percentage, "evict_percentage");

        eviction_threshold_ = (max_size_ * full_percentage) / PERCENTAGE_DIVISOR;
        eviction_percentage_ = evict_percentage;
    }

    /**
     * Set growth strategy
     * @param[in] strategy Growth strategy to use when capacity limits are reached
     * @throws std::invalid_argument if changing to Evict strategy with
     *         currently invalid parameters
     */
    void set_growth_strategy(GrowthStrategy strategy) {
        // Validate parameters if switching to Evict strategy
        if (strategy == GrowthStrategy::Evict && growth_strategy_ != GrowthStrategy::Evict) {
            // Validate threshold percentage when switching to Evict
            const std::size_t current_full_percentage =
                    (eviction_threshold_ * PERCENTAGE_DIVISOR) / max_size_;
            detail::validate_percentage(current_full_percentage, "full_percentage");
            detail::validate_percentage(eviction_percentage_, "evict_percentage");
        }

        growth_strategy_ = strategy;
    }
};

} // namespace framework::task

#endif // FRAMEWORK_TASK_FLAT_MAP_HPP
