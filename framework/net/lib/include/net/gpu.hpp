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

#ifndef FRAMEWORK_NET_GPU_HPP
#define FRAMEWORK_NET_GPU_HPP

#include <cstdint>
#include <memory>
#include <string>

#include "net/doca_types.hpp"
#include "net/net_export.hpp"

namespace framework::net {

/**
 * Strongly-typed wrapper for GPU device identifiers
 *
 * Prevents accidental usage of raw integers. Non-negative values are
 * enforced at compile-time by using uint32_t constructor parameter.
 */
class GpuDeviceId final {
public:
    /**
     * Create a GPU device identifier
     *
     * @param[in] id CUDA device ID (inherently non-negative)
     */
    explicit constexpr GpuDeviceId(const uint32_t id) noexcept : device_id_{static_cast<int>(id)} {}

    /**
     * Get the underlying device ID value
     *
     * @return Device ID as integer
     */
    [[nodiscard]] constexpr int value() const noexcept { return device_id_; }

    /**
     * Equality comparison
     *
     * @param[in] other GPU device ID to compare against
     * @return True if device IDs are equal, false otherwise
     */
    [[nodiscard]] constexpr bool operator==(const GpuDeviceId &other) const noexcept = default;

private:
    int device_id_{}; //!< Underlying device ID
};

/**
 * RAII wrapper for DOCA GPU device management
 *
 * Provides automatic resource management for DOCA GPU devices,
 * including PCI bus ID resolution and device initialization.
 */
class Gpu final {
public:
    /**
     * Create and initialize a DOCA GPU device
     *
     * @param[in] gpu_device_id CUDA device ID to initialize
     * @throws std::runtime_error if GPU device creation fails
     * @throws std::invalid_argument if GPU device ID is invalid
     */
    explicit NET_EXPORT Gpu(GpuDeviceId gpu_device_id);

    /**
     * Get the PCI bus ID for this GPU device
     *
     * @return PCI bus ID string (e.g., "0000:3b:00.0")
     */
    [[nodiscard]] NET_EXPORT const std::string &pci_bus_id() const noexcept;

    /**
     * Get the underlying DOCA GPU device pointer
     *
     * @return Pointer to the DOCA GPU device
     */
    [[nodiscard]] NET_EXPORT doca_gpu *get() const noexcept;

private:
    /**
     * Custom deleter for doca_gpu that calls doca_close_cuda_device
     */
    struct GpuDeleter {
        void operator()(doca_gpu *gpu) const noexcept;
    };

    std::unique_ptr<doca_gpu, GpuDeleter> gpu_dev_; //!< DOCA GPU device
    std::string pci_bus_id_;                        //!< Cached PCI bus ID
};

} // namespace framework::net

#endif // FRAMEWORK_NET_GPU_HPP
