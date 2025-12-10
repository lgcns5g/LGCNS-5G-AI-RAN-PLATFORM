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

#ifndef FRAMEWORK_NET_DOCA_UTILS_HPP
#define FRAMEWORK_NET_DOCA_UTILS_HPP

#include <string>
#include <string_view>

#include <doca_argp.h>
#include <doca_buf.h>
#include <doca_buf_inventory.h>
#include <doca_dev.h>
#include <doca_error.h>
#include <doca_eth_rxq.h>
#include <doca_eth_txq.h>
#include <doca_gpunetio.h>
#include <doca_mmap.h>
#include <tl/expected.hpp>

#include <cuda_runtime.h>

#include "log/rt_log_macros.hpp"
#include "net/details/dpdk_utils.hpp"
#include "net/doca_types.hpp"
#include "net/dpdk_types.hpp"
#include "net/net_export.hpp"

namespace framework::net {

/**
 * @brief Log DOCA SDK and runtime version information
 */
NET_EXPORT void doca_log_versions();

/**
 * @brief Initialize DOCA logging backends
 *
 * @param[out] sdk_log SDK log backend (optional, can be nullptr)
 * @return DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
[[nodiscard]] NET_EXPORT doca_error_t doca_init_logging(doca_log_backend **sdk_log);

/**
 * @brief Open a DOCA device according to a given PCI address
 *
 * @param[in] pci_addr PCI address
 * @param[out] retval Pointer to doca_dev struct, nullptr if not found
 * @return DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
[[nodiscard]] NET_EXPORT doca_error_t
doca_open_device_with_pci(const std::string_view pci_addr, doca_dev **retval);

/**
 * @brief Open and probe a DOCA network device
 *
 * @param[in] nic_pcie_addr Network card PCIe address
 * @param[out] ddev DOCA device
 * @return DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
[[nodiscard]] NET_EXPORT doca_error_t
doca_open_and_probe_device(std::string_view nic_pcie_addr, doca_dev **ddev);

/**
 * @brief Close a DOCA device
 *
 * @param[in] ddev DOCA device to close
 * @return DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
[[nodiscard]] NET_EXPORT doca_error_t doca_close_device(doca_dev *ddev);

/**
 * @brief Check if device is a CX-6 device
 *
 * @param[in] ddev DOCA device
 * @return True if CX-6 device, error message on failure
 */
[[nodiscard]] NET_EXPORT tl::expected<bool, std::string> is_device_cx6(doca_dev *ddev);

/**
 * @brief Open and initialize a DOCA GPU device
 *
 * @param[in] gpu_pcie_addr GPU PCIe address
 * @param[out] gpu_dev DOCA GPU device
 * @return DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
[[nodiscard]] NET_EXPORT doca_error_t
doca_open_cuda_device(const std::string_view gpu_pcie_addr, doca_gpu **gpu_dev);

/**
 * @brief Close and destroy a DOCA GPU device
 *
 * @param[in] gpu_dev DOCA GPU device to destroy
 * @return DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
[[nodiscard]] NET_EXPORT doca_error_t doca_close_cuda_device(doca_gpu *gpu_dev);

/**
 * @brief Convert CUDA device ID to PCI bus ID
 *
 * @param[in] cuda_device_id CUDA device ID
 * @return PCI bus ID string (e.g., "0000:3b:00.0")
 */
[[nodiscard]] NET_EXPORT std::string doca_device_id_to_pci_bus_id(int cuda_device_id);

/**
 * @brief Get DPDK port ID for a DOCA device
 *
 * @param[in] dev_input DOCA device
 * @param[out] port_id DPDK port ID
 * @return DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
[[nodiscard]] NET_EXPORT doca_error_t doca_get_dpdk_port_id(doca_dev *dev_input, uint16_t *port_id);

/**
 * @brief Get MAC address from PCIe address
 *
 * Searches through all available DOCA devices to find the one with the
 * specified PCIe address and retrieves its MAC address.
 *
 * @param[in] pci_addr PCIe address string (e.g., "0000:3a:00.0")
 * @param[out] mac_addr Buffer to store the 6-byte MAC address
 * @return DOCA_SUCCESS on success
 * @return DOCA_ERROR_INVALID_VALUE if parameters are nullptr
 * @return DOCA_ERROR_NOT_FOUND if no device with specified PCIe address is
 * found
 * @return Other DOCA errors on device enumeration or MAC retrieval failures
 */
[[nodiscard]] NET_EXPORT doca_error_t
doca_get_mac_addr_from_pci(const std::string_view pci_addr, MacAddress &mac_addr);

/**
 * @brief Create DOCA Ethernet RX queue for GPU
 *
 * @param[in] rxq DOCA Eth RX queue handler
 * @param[in] gpu_dev DOCA GPUNetIO device
 * @param[in] ddev DOCA device
 * @param[in] max_pkt_num Maximum number of packets in queue
 * @param[in] max_pkt_size Maximum packet size in bytes
 * @param[in] sem_items Semaphore configuration (if not set, no semaphore created)
 * @return DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
[[nodiscard]] NET_EXPORT doca_error_t doca_create_rxq(
        struct DocaRxQParams *rxq,
        doca_gpu *gpu_dev,
        doca_dev *ddev,
        uint32_t max_pkt_num,
        uint32_t max_pkt_size,
        const std::optional<DocaSemItems> &sem_items);

/**
 * @brief Destroy DOCA Ethernet RX queue for GPU
 *
 * @param[in] rxq DOCA Eth RX queue handler
 * @return DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
[[nodiscard]] NET_EXPORT doca_error_t doca_destroy_rxq(struct DocaRxQParams *rxq);

/**
 * @brief Cleanup DPDK flow rule to avoid reference issues during RX queue
 * destruction
 *
 * @param[in] rxq DOCA Eth RX queue handler containing flow rule
 * @return DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
[[nodiscard]] NET_EXPORT doca_error_t doca_destroy_flow_rule(struct DocaRxQParams *rxq);

/**
 * Set up DPDK flow rules to direct packets to GPU receive queue
 *
 * Configures DPDK flow rules to match packets with specific source and
 * destination MAC addresses and specified EtherType, directing them to the GPU
 * receive queue. Optionally matches VLAN TCI when provided.
 *
 * @param[in] rxq Receive queue structure with GPU queue info
 * @param[in] nic_pcie_addr NIC PCIe address for receiver MAC lookup
 * @param[in] sender_mac_addr Sender MAC address
 * @param[in] ether_type EtherType value to match in flow rule
 * @param[in] vlan_tci Optional VLAN TCI value (if not set, no VLAN matching)
 * @return DOCA_SUCCESS on success
 * @return DOCA_ERROR_BAD_STATE on DPDK configuration or flow rule failures
 * @return DOCA_ERROR_NOT_FOUND if DPDK port ID cannot be found
 * @return Other DOCA errors on MAC address retrieval failures
 */
[[nodiscard]] NET_EXPORT doca_error_t doca_create_flow_rule(
        struct DocaRxQParams *rxq,
        const std::string_view nic_pcie_addr,
        const MacAddress &sender_mac_addr,
        uint16_t ether_type,
        const std::optional<uint16_t> &vlan_tci = std::nullopt);

/**
 * @brief Create DOCA Ethernet TX queue for GPU
 *
 * @param[in] txq DOCA Eth TX queue handler
 * @param[in] gpu_dev DOCA GPUNetIO device
 * @param[in] ddev DOCA device
 * @param[in] pkt_size Packet size to send
 * @param[in] pkt_num Number of packets
 * @param[in] max_sq_descr_num Maximum number of send queue descriptors
 * @param[in] nic_pcie_addr NIC PCIe address for source MAC lookup
 * @param[in] dest_mac_addr Destination MAC address string in format
 * "XX:XX:XX:XX:XX:XX"
 * @param[in] ether_type EtherType value to use in packet headers
 * @param[in] vlan_tci Optional VLAN tag control information; when set,
 *                     packets are tagged with the provided 802.1Q VLAN TCI
 * @return DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
[[nodiscard]] NET_EXPORT doca_error_t doca_create_txq(
        struct DocaTxQParams *txq,
        doca_gpu *gpu_dev,
        doca_dev *ddev,
        uint32_t pkt_size,
        uint32_t pkt_num,
        uint32_t max_sq_descr_num,
        const std::string_view nic_pcie_addr,
        const MacAddress &dest_mac_addr,
        uint16_t ether_type,
        const std::optional<uint16_t> &vlan_tci = std::nullopt);

/**
 * @brief Destroy DOCA Ethernet TX queue for GPU
 *
 * @param[in] txq DOCA Eth TX queue handler
 * @return DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
[[nodiscard]] NET_EXPORT doca_error_t doca_destroy_txq(struct DocaTxQParams *txq);

/**
 * @brief Check if size meets GDRCopy requirements
 *
 * @param[in] size Size to check
 * @return true if size is GDRCopy compatible, false otherwise
 */
[[nodiscard]] NET_EXPORT bool doca_is_gdrcopy_compatible_size(size_t size) noexcept;

/**
 * @brief Align size to GPU page boundary for GDRCopy compatibility
 *
 * @param[in] size Original size
 * @return Size aligned to GPU page boundary
 */
[[nodiscard]] NET_EXPORT size_t doca_align_to_gpu_page(size_t size) noexcept;

/**
 * @brief Check if NIC device supports RDMA operations
 *
 * Verifies that the DOCA device can perform RDMA write operations, which is
 * required for GPU-accelerated RDMA datapath.
 *
 * @param[in] ddev DOCA device to check
 * @return Pair of (error_code, is_supported) - first is DOCA_SUCCESS on success,
 * second is true if RDMA is supported
 */
[[nodiscard]] NET_EXPORT tl::expected<bool, std::string> doca_is_rdma_supported(doca_dev *ddev);

} // namespace framework::net

#endif
