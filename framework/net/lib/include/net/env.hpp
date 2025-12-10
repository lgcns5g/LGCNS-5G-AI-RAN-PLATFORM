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

#ifndef FRAMEWORK_NET_ENV_HPP
#define FRAMEWORK_NET_ENV_HPP

#include <cstdint>
#include <memory>
#include <variant>

#include "net/dpdk_types.hpp"
#include "net/gpu.hpp"
#include "net/net_export.hpp"
#include "net/nic.hpp"

namespace framework::net {

/**
 * Configuration structure for environment initialization
 */
struct NET_EXPORT EnvConfig final {
    DpdkConfig dpdk_config;       //!< DPDK configuration parameters
    NicConfig nic_config;         //!< NIC configuration parameters
    GpuDeviceId gpu_device_id{0}; //!< GPU device ID to use
};

/**
 * RAII wrapper for DPDK and DOCA environment management
 *
 * Provides automatic resource management for the entire networking environment,
 * including DPDK EAL initialization, GPU device setup, and NIC management.
 * Validates system requirements before initialization.
 */
class Env final {
public:
    /**
     * Create and initialize the networking environment
     *
     * Performs the following initialization steps:
     * 1. Validates CUDA device count and GPU device ID
     * 2. Validates Mellanox NICs availability and configuration
     * 3. Initializes DPDK EAL
     * 4. Creates and initializes GPU device
     * 5. Creates and initializes NIC with queues
     *
     * @param[in] config Environment configuration parameters
     * @throws std::runtime_error if initialization fails
     * @throws std::invalid_argument if configuration is invalid
     */
    explicit NET_EXPORT Env(const EnvConfig &config);

    /**
     * Check if GPU device is available
     *
     * @return true if GPU device was created, false if running in DPDK-only mode
     */
    [[nodiscard]] NET_EXPORT bool has_gpu() const noexcept;

    /**
     * Get the GPU device
     *
     * @return Reference to the GPU device
     * @throws std::runtime_error if no GPU device was created (DPDK-only mode)
     */
    [[nodiscard]] NET_EXPORT const Gpu &gpu() const;

    /**
     * Get the NIC device
     *
     * @return Reference to the NIC device
     */
    [[nodiscard]] NET_EXPORT const Nic &nic() const noexcept;

    /**
     * Get the DPDK configuration used
     *
     * @return Reference to the DPDK configuration
     */
    [[nodiscard]] NET_EXPORT const DpdkConfig &dpdk_config() const noexcept;

    /**
     * Check if the environment is properly initialized
     *
     * @return true if environment is initialized, false otherwise
     */
    [[nodiscard]] NET_EXPORT bool is_initialized() const noexcept;

private:
    // Custom deleter for DPDK EAL cleanup
    struct DpdkEalDeleter {
        void operator()(std::monostate *ptr) const noexcept;
    };

    DpdkConfig dpdk_config_; //!< Stored DPDK configuration

    // IMPORTANT: Member declaration order matters for RAII cleanup
    // Members are destroyed in reverse order of declaration, ensuring:
    // 1. nic_ destroyed first (depends on DPDK)
    // 2. gpu_ destroyed second (depends on DPDK)
    // 3. dpdk_eal_guard_ destroyed last (calls dpdk_cleanup_eal())
    std::unique_ptr<std::monostate, DpdkEalDeleter>
            dpdk_eal_guard_;   //!< RAII guard for DPDK EAL cleanup
    std::unique_ptr<Gpu> gpu_; //!< GPU device
    std::unique_ptr<Nic> nic_; //!< NIC device
};

} // namespace framework::net

#endif // FRAMEWORK_NET_ENV_HPP
