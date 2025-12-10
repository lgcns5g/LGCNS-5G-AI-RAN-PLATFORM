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
#include <cstddef>
#include <cstdint>
#include <exception>
#include <format>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include <doca_ctx.h>
#include <doca_dev.h>
#include <doca_error.h>
#include <quill/LogMacros.h>
#include <tl/expected.hpp>

#include <wise_enum.h>

#include "log/rt_log_macros.hpp"
#include "net/details/doca_utils.hpp"
#include "net/details/dpdk_utils.hpp"
#include "net/doca_rxq.hpp"
#include "net/doca_txq.hpp"
#include "net/dpdk_txq.hpp"
#include "net/dpdk_types.hpp"
#include "net/mempool.hpp"
#include "net/net_log.hpp"
#include "net/nic.hpp"

namespace framework::net {

namespace {

/**
 * Check RDMA capabilities for NIC
 *
 * Validates that the NIC supports RDMA operations required for DOCA GPU NetIO.
 * Logs appropriate debug/error messages based on capability status.
 *
 * @param[in] ddev DOCA device (NIC) to check
 * @param[in] nic_pcie_addr NIC PCIe address for logging
 */
void check_rdma_capabilities(doca_dev *ddev, const std::string_view nic_pcie_addr) {
    // Check NIC RDMA support
    const auto nic_rdma_result = doca_is_rdma_supported(ddev);
    if (nic_rdma_result.has_value()) {
        if (nic_rdma_result.value()) {
            RT_LOGC_DEBUG(Net::NetDoca, "NIC {} supports RDMA operations", nic_pcie_addr);
        } else {
            RT_LOGC_ERROR(
                    Net::NetDoca,
                    "NIC {} does not support RDMA operations (required for DOCA GPU NetIO)",
                    nic_pcie_addr);
        }
    } else {
        RT_LOGC_ERROR(
                Net::NetDoca,
                "Failed to check NIC {} RDMA capability: {}",
                nic_pcie_addr,
                nic_rdma_result.error());
    }
}

/**
 * Configure DPDK port settings and perform hardware initialization
 *
 * @param[in] dpdk_port_id DPDK port identifier
 * @param[in] pci_address PCI address of the NIC
 * @param[in] config NIC configuration parameters
 */
void configure_dpdk_port_settings(
        const std::uint16_t dpdk_port_id, const std::string &pci_address, const NicConfig &config) {
    // Configure DPDK port settings
    // Set RX queues to 0 since we use DOCA RX queues, but configure TX queues for
    // DPDK
    static constexpr auto NO_RX_QUEUES = 0U;
    const auto num_tx_queues = static_cast<std::uint16_t>(config.dpdk_txq_configs.size());
    const auto port_state = dpdk_try_configure_port(
            dpdk_port_id, NO_RX_QUEUES, num_tx_queues, config.enable_accurate_send_scheduling);
    if (port_state == DpdkPortState::ConfigureError) {
        const std::string state_str(::wise_enum::to_string(port_state));
        log_and_throw(Net::NetDpdk, "Failed to configure DPDK port: {}", state_str);
    }

    // Tune virtual memory settings
    if (auto vm_result = dpdk_try_tune_virtual_memory(); vm_result) {
        log_and_throw(
                Net::NetDpdk,
                "Failed to tune virtual memory settings: {}",
                get_error_name(vm_result));
    }

    // Check PCIe maximum read request size
    if (auto pcie_result = dpdk_check_pcie_max_read_request_size(pci_address, config.pcie_mrrs);
        pcie_result) {
        // Only throw on actual errors, not on value mismatches (which just generate
        // warnings)
        if (pcie_result != make_error_code(DpdkErrc::PcieVerifyMismatch)) {
            log_and_throw(
                    Net::NetDpdk,
                    "Failed to check PCIe MRRS for {}: {}",
                    pci_address,
                    get_error_name(pcie_result));
        }
        // pcie_verify_mismatch is not a failure, just a warning that was already
        // logged
    }

    // Disable Ethernet flow control
    if (auto fc_result = dpdk_disable_ethernet_flow_control(dpdk_port_id); fc_result) {
        log_and_throw(
                Net::NetDpdk,
                "Failed to disable flow control for port {}: {}",
                dpdk_port_id,
                get_error_name(fc_result));
    }

    // Set port MTU
    if (auto mtu_result = dpdk_set_port_mtu(dpdk_port_id, config.max_mtu_size); mtu_result) {
        log_and_throw(
                Net::NetDpdk,
                "Failed to set MTU {} for port {}: {}",
                config.max_mtu_size,
                dpdk_port_id,
                get_error_name(mtu_result));
    }

    // Log link information
    if (auto link_result = dpdk_log_link_info(dpdk_port_id); link_result) {
        log_and_throw(
                Net::NetDpdk,
                "Failed to log link info for port {}: {}",
                dpdk_port_id,
                get_error_name(link_result));
    }

    // Check if link is up
    if (auto link_status_result = dpdk_is_link_status_up(dpdk_port_id); link_status_result) {
        log_and_throw(
                Net::NetDpdk,
                "Link is down for dpdk port {} ({}): {}",
                dpdk_port_id,
                pci_address,
                get_error_name(link_status_result));
    }

    // Enable flow rule isolation
    if (auto isolation_result = dpdk_enable_flow_rule_isolation(dpdk_port_id); isolation_result) {
        log_and_throw(
                Net::NetDpdk,
                "Failed to enable flow rule isolation for port {}: {}",
                dpdk_port_id,
                get_error_name(isolation_result));
    }
}

/**
 * Validate NIC configuration parameters (basic validation only)
 *
 * Performs basic parameter validation before attempting to open the DOCA device.
 * This ensures that invalid parameters fail fast with appropriate exception types.
 *
 * @param[in] config NIC configuration to validate
 * @param[in] gpu_dev Optional GPU device
 * @throws std::invalid_argument if configuration is invalid
 */
void validate_nic_config_basic(const NicConfig &config, const std::optional<doca_gpu *> &gpu_dev) {
    // GPU device is only required if DOCA queues are configured
    const bool has_doca_queues =
            !config.doca_txq_configs.empty() || !config.doca_rxq_configs.empty();
    if (has_doca_queues && !gpu_dev.has_value()) {
        log_and_throw<std::invalid_argument>(
                Net::NetDoca, "GPU device must be provided when DOCA queues are configured");
    }

    if (config.nic_pcie_addr.empty()) {
        log_and_throw<std::invalid_argument>(Net::NetDoca, "NIC PCIe address cannot be empty");
    }

    if (config.max_mtu_size == 0) {
        log_and_throw<std::invalid_argument>(
                Net::NetDoca, "Maximum MTU size must be greater than zero");
    }

    if (config.doca_txq_configs.empty() && config.doca_rxq_configs.empty() &&
        config.dpdk_txq_configs.empty()) {
        log_and_throw<std::invalid_argument>(
                Net::NetDoca, "At least one TX or RX queue configuration must be provided");
    }
}

/**
 * Create DPDK TX queues for the NIC
 *
 * @param[in] dpdk_port_id DPDK port ID
 * @param[in] configs DPDK TX queue configurations
 * @return Vector of created DPDK TX queues
 * @throws std::runtime_error if queue creation fails
 */
std::vector<DpdkTxQueue>
create_dpdk_tx_queues(std::uint16_t dpdk_port_id, const std::vector<DpdkTxQConfig> &configs) {
    std::vector<DpdkTxQueue> queues;
    queues.reserve(configs.size());

    std::uint16_t txq_id = 0;
    for (const auto &config : configs) {
        try {
            queues.emplace_back(dpdk_port_id, txq_id, config);
        } catch (const std::exception &e) {
            log_and_throw(Net::NetDpdk, "Failed to create DPDK TX queue {}: {}", txq_id, e.what());
        }
        ++txq_id;
    }

    return queues;
}

/**
 * Create DOCA TX queues for the NIC
 *
 * @param[in] configs DOCA TX queue configurations
 * @param[in] gpu_dev GPU device for DOCA operations
 * @param[in] ddev DOCA device
 * @return Vector of created DOCA TX queues
 * @throws std::runtime_error if queue creation fails
 */
std::vector<DocaTxQ> create_doca_tx_queues(
        const std::vector<DocaTxQConfig> &configs, doca_gpu *gpu_dev, doca_dev *ddev) {
    std::vector<DocaTxQ> queues;
    queues.reserve(configs.size());

    for (const auto &config : configs) {
        try {
            queues.emplace_back(config, gpu_dev, ddev);
        } catch (const std::exception &e) {
            log_and_throw(Net::NetDoca, "Failed to create DOCA TX queue: {}", e.what());
        }
    }

    return queues;
}

/**
 * Create DOCA RX queues for the NIC
 *
 * @param[in] configs DOCA RX queue configurations
 * @param[in] gpu_dev GPU device for DOCA operations
 * @param[in] ddev DOCA device
 * @return Vector of created DOCA RX queues
 * @throws std::runtime_error if queue creation fails
 */
std::vector<DocaRxQ> create_doca_rx_queues(
        const std::vector<DocaRxQConfig> &configs, doca_gpu *gpu_dev, doca_dev *ddev) {
    std::vector<DocaRxQ> queues;
    queues.reserve(configs.size());

    for (const auto &config : configs) {
        try {
            queues.emplace_back(config, gpu_dev, ddev);
        } catch (const std::exception &e) {
            log_and_throw(Net::NetDoca, "Failed to create DOCA RX queue: {}", e.what());
        }
    }

    return queues;
}

/**
 * Create mempools for the NIC
 *
 * @param[in] dpdk_port_id DPDK port ID
 * @param[in] configs Mempool configurations
 * @return Vector of created mempools
 * @throws std::runtime_error if mempool creation fails
 */
std::vector<Mempool>
create_mempools(std::uint16_t dpdk_port_id, const std::vector<MempoolConfig> &configs) {
    std::vector<Mempool> mempools;
    mempools.reserve(configs.size());

    for (const auto &config : configs) {
        try {
            mempools.emplace_back(dpdk_port_id, config);
        } catch (const std::exception &e) {
            log_and_throw(Net::NetDpdk, "Failed to create mempool '{}': {}", config.name, e.what());
        }
    }

    return mempools;
}

} // anonymous namespace

void Nic::NicDeleter::operator()(doca_dev *ddev) const noexcept {
    if (ddev != nullptr) {
        std::uint16_t port_id{};
        if (const auto port_result = doca_get_dpdk_port_id(ddev, &port_id);
            port_result != DOCA_SUCCESS) {
            RT_LOGC_ERROR(
                    Net::NetDoca,
                    "Failed to get DPDK port ID: {}",
                    doca_error_get_descr(port_result));
        }
        if (auto stop_result = dpdk_stop_eth_dev(port_id); stop_result != DpdkErrc::Success) {
            RT_LOGC_ERROR(
                    Net::NetDpdk, "Failed to stop DPDK port: {}", get_error_name(stop_result));
        }

        const doca_error_t result = doca_close_device(ddev);
        if (result != DOCA_SUCCESS) {
            RT_LOGC_ERROR(
                    Net::NetDoca, "Failed to close DOCA device: {}", doca_error_get_descr(result));
        }
    }
}

Nic::Nic(const NicConfig &config, const std::optional<doca_gpu *> gpu_dev)
        : pci_address_(config.nic_pcie_addr) {
    // Validate basic configuration parameters first
    validate_nic_config_basic(config, gpu_dev);

    // Initialize DOCA device
    doca_dev *raw_ddev = nullptr;
    doca_error_t result = doca_open_and_probe_device(pci_address_, &raw_ddev);
    if (result != DOCA_SUCCESS) {
        log_and_throw(
                Net::NetDoca,
                "Failed to open and probe DOCA device: {}",
                doca_error_get_descr(result));
    }

    if (raw_ddev == nullptr) {
        log_and_throw(Net::NetDoca, "DOCA device creation returned null pointer");
    }

    // Transfer ownership to unique_ptr
    ddev_.reset(raw_ddev);

    // Check RDMA capabilities if DOCA GPU NetIO is being used
    const bool has_doca_queues =
            !config.doca_txq_configs.empty() || !config.doca_rxq_configs.empty();
    if (has_doca_queues && gpu_dev.has_value()) {
        check_rdma_capabilities(ddev_.get(), pci_address_);
    }

    // Get MAC address
    result = doca_get_mac_addr_from_pci(pci_address_, mac_address_);
    if (result != DOCA_SUCCESS) {
        log_and_throw(Net::NetDoca, "Failed to get MAC address: {}", doca_error_get_descr(result));
    }

    // Get DPDK port ID
    result = doca_get_dpdk_port_id(ddev_.get(), &dpdk_port_id_);
    if (result != DOCA_SUCCESS) {
        log_and_throw(Net::NetDoca, "Failed to get DPDK port ID: {}", doca_error_get_descr(result));
    }

    // Detect device type for CX-6 devices
    const auto cx6_result = is_device_cx6(ddev_.get());
    is_cx6_device_ = (cx6_result.has_value() && cx6_result.value());

    // Validate Mellanox driver
    if (auto validate_result = dpdk_validate_mellanox_driver(dpdk_port_id_); validate_result) {
        log_and_throw(
                Net::NetDpdk,
                "Failed to validate Mellanox driver for port {}: {}",
                dpdk_port_id_,
                get_error_name(validate_result));
    }

    // Configure DPDK port settings and perform hardware initialization
    configure_dpdk_port_settings(dpdk_port_id_, pci_address_, config);

    // Create all queues and mempools
    dpdk_tx_queues_ = create_dpdk_tx_queues(dpdk_port_id_, config.dpdk_txq_configs);

    // Only create DOCA queues if GPU device is available
    if (gpu_dev.has_value()) {
        doca_tx_queues_ =
                create_doca_tx_queues(config.doca_txq_configs, gpu_dev.value(), ddev_.get());

        doca_rx_queues_ =
                create_doca_rx_queues(config.doca_rxq_configs, gpu_dev.value(), ddev_.get());
    }

    mempools_ = create_mempools(dpdk_port_id_, config.mempool_configs);

    // Start the port
    if (auto start_result = dpdk_start_eth_dev(dpdk_port_id_); start_result) {
        log_and_throw(Net::NetDpdk, "Failed to start DPDK port: {}", get_error_name(start_result));
    }

    RT_LOGC_DEBUG(
            Net::NetDoca,
            "NIC initialized successfully - PCI: {}, MAC: {}, DPDK Port: {}, "
            "DOCA TX Queues: {}, DOCA RX Queues: {}, DPDK TX Queues: {}, Mempools: "
            "{}",
            pci_address_,
            mac_address_.to_string(),
            dpdk_port_id_,
            doca_tx_queues_.size(),
            doca_rx_queues_.size(),
            dpdk_tx_queues_.size(),
            mempools_.size());
}

const std::string &Nic::pci_address() const noexcept { return pci_address_; }

const MacAddress &Nic::mac_address() const noexcept { return mac_address_; }

std::uint16_t Nic::dpdk_port_id() const noexcept { return dpdk_port_id_; }

doca_dev *Nic::get() const noexcept { return ddev_.get(); }

bool Nic::is_cx6_device() const noexcept { return is_cx6_device_; }

tl::expected<bool, std::string> Nic::is_rdma_supported() const noexcept {
    return doca_is_rdma_supported(ddev_.get());
}

const std::vector<DocaTxQ> &Nic::doca_tx_queues() const noexcept { return doca_tx_queues_; }

const std::vector<DocaRxQ> &Nic::doca_rx_queues() const noexcept { return doca_rx_queues_; }

const DocaTxQ &Nic::doca_tx_queue(const std::size_t index) const {
    if (index >= doca_tx_queues_.size()) {
        log_and_throw<std::out_of_range>(
                Net::NetDoca,
                "DOCA TX queue index {} is out of range (size: {})",
                index,
                doca_tx_queues_.size());
    }
    return doca_tx_queues_[index];
}

const DocaRxQ &Nic::doca_rx_queue(const std::size_t index) const {
    if (index >= doca_rx_queues_.size()) {
        log_and_throw<std::out_of_range>(
                Net::NetDoca,
                "DOCA RX queue index {} is out of range (size: {})",
                index,
                doca_rx_queues_.size());
    }
    return doca_rx_queues_[index];
}

const std::vector<DpdkTxQueue> &Nic::dpdk_tx_queues() const noexcept { return dpdk_tx_queues_; }

const DpdkTxQueue &Nic::dpdk_tx_queue(const std::size_t index) const {
    if (index >= dpdk_tx_queues_.size()) {
        log_and_throw<std::out_of_range>(
                Net::NetDpdk,
                "DPDK TX queue index {} is out of range (size: {})",
                index,
                dpdk_tx_queues_.size());
    }
    return dpdk_tx_queues_[index];
}

const std::vector<Mempool> &Nic::mempools() const noexcept { return mempools_; }

const Mempool &Nic::mempool(const std::size_t index) const {
    if (index >= mempools_.size()) {
        log_and_throw<std::out_of_range>(
                Net::NetDpdk,
                "Mempool index {} is out of range (size: {})",
                index,
                mempools_.size());
    }
    return mempools_[index];
}

std::error_code Nic::send(
        const std::size_t dpdk_txq_id,
        const std::size_t mempool_id,
        const std::span<const std::span<const std::uint8_t>> messages,
        const EthernetHeader &eth_header,
        const std::uint32_t max_retry_count) const {
    if (dpdk_txq_id >= dpdk_tx_queues_.size()) {
        log_and_throw<std::out_of_range>(
                Net::NetDpdk,
                "DPDK TX queue index {} is out of range (size: {})",
                dpdk_txq_id,
                dpdk_tx_queues_.size());
    }

    if (mempool_id >= mempools_.size()) {
        log_and_throw<std::out_of_range>(
                Net::NetDpdk,
                "Mempool index {} is out of range (size: {})",
                mempool_id,
                mempools_.size());
    }

    return dpdk_tx_queues_[dpdk_txq_id].send(
            messages, eth_header, mempools_[mempool_id].dpdk_mempool(), max_retry_count);
}

std::error_code Nic::send_mbufs(
        const std::size_t dpdk_txq_id,
        const std::span<rte_mbuf *> mbufs,
        const std::uint32_t max_retry_count) const {
    if (dpdk_txq_id >= dpdk_tx_queues_.size()) {
        log_and_throw<std::out_of_range>(
                Net::NetDpdk,
                "DPDK TX queue index {} is out of range (size: {})",
                dpdk_txq_id,
                dpdk_tx_queues_.size());
    }

    return dpdk_tx_queues_[dpdk_txq_id].send_mbufs(mbufs, max_retry_count);
}

} // namespace framework::net
