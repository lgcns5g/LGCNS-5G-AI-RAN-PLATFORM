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

#include <algorithm>
#include <format>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <variant>
#include <vector>

#include <driver_types.h>
#include <quill/LogMacros.h>

#include <cuda_runtime.h>

#include "log/rt_log_macros.hpp"
#include "net/details/dpdk_utils.hpp"
#include "net/doca_rxq.hpp"
#include "net/doca_txq.hpp"
#include "net/dpdk_txq.hpp"
#include "net/dpdk_types.hpp"
#include "net/env.hpp"
#include "net/gpu.hpp"
#include "net/mempool.hpp"
#include "net/net_log.hpp"
#include "net/nic.hpp"

namespace {

/// Validate CUDA device count and GPU device ID
void validate_cuda_devices(const framework::net::GpuDeviceId gpu_device_id) {
    int device_count{};
    const cudaError_t cuda_result = cudaGetDeviceCount(&device_count);
    if (cuda_result != cudaSuccess) {
        framework::net::log_and_throw(
                framework::net::Net::NetGpu,
                "Failed to get CUDA device count: {}",
                cudaGetErrorString(cuda_result));
    }

    if (device_count == 0) {
        framework::net::log_and_throw(
                framework::net::Net::NetGpu,
                "No CUDA devices available. At least one CUDA device is required.");
    }

    if (gpu_device_id.value() >= device_count) {
        framework::net::log_and_throw<std::invalid_argument>(
                framework::net::Net::NetGpu,
                "GPU device ID {} is out of range. Available devices: 0-{}",
                gpu_device_id.value(),
                device_count - 1);
    }

    RT_LOGC_DEBUG(
            framework::net::Net::NetGpu,
            "CUDA validation passed - {} devices available, using device {}",
            device_count,
            gpu_device_id.value());
}

/// Validate Mellanox NICs availability and configuration
void validate_mellanox_nics(const framework::net::NicConfig &nic_config) {
    const auto available_nics = framework::net::discover_mellanox_nics();
    if (available_nics.empty()) {
        framework::net::log_and_throw(
                framework::net::Net::NetDoca,
                "No Mellanox NICs available. At least one Mellanox NIC is required.");
    }

    // Check if the configured NIC PCI address exists in the discovered NICs
    const auto it =
            std::find(available_nics.begin(), available_nics.end(), nic_config.nic_pcie_addr);
    if (it == available_nics.end()) {
        framework::net::log_and_throw<std::invalid_argument>(
                framework::net::Net::NetDoca,
                "NIC with PCI address '{}' not found in available Mellanox NICs",
                nic_config.nic_pcie_addr);
    }

    RT_LOGC_DEBUG(
            framework::net::Net::NetDoca,
            "Mellanox NIC validation passed - {} NICs available, using {}",
            available_nics.size(),
            nic_config.nic_pcie_addr);
}

/// Validate DPDK configuration parameters
void validate_dpdk_config(const framework::net::DpdkConfig &dpdk_config) {
    if (dpdk_config.file_prefix.empty()) {
        framework::net::log_and_throw<std::invalid_argument>(
                framework::net::Net::NetDpdk,
                "DPDK file_prefix cannot be empty. A unique file prefix is required to avoid "
                "conflicts with other DPDK applications.");
    }
}

} // namespace

namespace framework::net {
// Implementation of custom deleter for DPDK EAL
void Env::DpdkEalDeleter::operator()(std::monostate *ptr) const noexcept {
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    delete ptr;

    RT_LOGC_DEBUG(Net::NetDpdk, "Cleaning up DPDK EAL");
    if (const auto cleanup_result = dpdk_cleanup_eal(); cleanup_result != DpdkErrc::Success) {
        RT_LOGC_ERROR(
                Net::NetDpdk, "Failed to cleanup DPDK EAL: {}", get_error_name(cleanup_result));
    }
}

Env::Env(const EnvConfig &config) : dpdk_config_{config.dpdk_config} {
    // Check if DOCA queues are configured to determine if GPU is needed
    const bool has_doca_queues = !config.nic_config.doca_txq_configs.empty() ||
                                 !config.nic_config.doca_rxq_configs.empty();

    // Step 1: Validate CUDA devices (only if GPU is needed)
    if (has_doca_queues) {
        validate_cuda_devices(config.gpu_device_id);
    }

    // Step 2: Validate Mellanox NICs
    validate_mellanox_nics(config.nic_config);

    // Step 3: Validate DPDK configuration
    validate_dpdk_config(config.dpdk_config);

    // Step 4: Initialize DPDK EAL
    RT_LOGC_DEBUG(Net::NetDpdk, "Initializing DPDK EAL");
    const auto dpdk_result = dpdk_init_eal(dpdk_config_);
    if (dpdk_result) {
        log_and_throw(
                Net::NetDpdk, "Failed to initialize DPDK EAL: {}", get_error_name(dpdk_result));
    }
    // Create RAII guard for DPDK EAL cleanup (will auto-cleanup on destruction)
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    dpdk_eal_guard_.reset(new std::monostate{});

    // Step 5: Create GPU device (only if DOCA queues are configured)
    if (has_doca_queues) {
        RT_LOGC_DEBUG(Net::NetGpu, "Creating GPU device {}", config.gpu_device_id.value());
        gpu_ = std::make_unique<Gpu>(config.gpu_device_id);
    } else {
        RT_LOGC_DEBUG(Net::NetDpdk, "Skipping GPU creation - DPDK-only mode (no DOCA queues)");
    }

    // Step 6: Create NIC device with queues
    RT_LOGC_DEBUG(Net::NetDoca, "Creating NIC device {}", config.nic_config.nic_pcie_addr);
    nic_ = std::make_unique<Nic>(
            config.nic_config, gpu_ ? std::optional{gpu_->get()} : std::nullopt);

    if (has_doca_queues) {
        RT_LOGC_INFO(
                Net::NetDoca,
                "Environment initialized successfully - GPU: {}, NIC: {}, "
                "DOCA TX Queues: {}, DOCA RX Queues: {}, DPDK TX Queues: {}",
                config.gpu_device_id.value(),
                nic_->pci_address(),
                nic_->doca_tx_queues().size(),
                nic_->doca_rx_queues().size(),
                nic_->dpdk_tx_queues().size());
    } else {
        RT_LOGC_INFO(
                Net::NetDpdk,
                "Environment initialized successfully - DPDK-only mode, NIC: {}, "
                "DPDK TX Queues: {}, Mempools: {}",
                nic_->pci_address(),
                nic_->dpdk_tx_queues().size(),
                nic_->mempools().size());
    }
}

bool Env::has_gpu() const noexcept { return gpu_ != nullptr; }

const Gpu &Env::gpu() const {
    if (!gpu_) {
        log_and_throw(Net::NetGpu, "GPU device not available - running in DPDK-only mode");
    }
    return *gpu_;
}

const Nic &Env::nic() const noexcept { return *nic_; }

const DpdkConfig &Env::dpdk_config() const noexcept { return dpdk_config_; }

bool Env::is_initialized() const noexcept { return dpdk_eal_guard_ != nullptr && nic_ != nullptr; }

} // namespace framework::net
