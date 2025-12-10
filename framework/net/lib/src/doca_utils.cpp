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
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <format>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include <doca_buf_array.h>
#include <doca_ctx.h>
#include <doca_dev.h>
#include <doca_dpdk.h>
#include <doca_error.h>
#include <doca_eth_rxq.h>
#include <doca_eth_txq.h>
#include <doca_eth_txq_gpu_data_path.h>
#include <doca_gpunetio.h>
#include <doca_log.h>
#include <doca_mmap.h>
#include <doca_pe.h>
#include <doca_rdma.h>
#include <doca_types.h>
#include <doca_version.h>
#include <driver_types.h>
#include <gdrapi.h>
#include <quill/LogMacros.h>
#include <rte_build_config.h>
#include <rte_byteorder.h>
#include <rte_ethdev.h>
#include <rte_ether.h>
#include <rte_flow.h>
#include <tl/expected.hpp>

#include <gsl-lite/gsl-lite.hpp>

#include <cuda_runtime.h>

#include "log/rt_log_macros.hpp"
#include "net/details/doca_utils.hpp"
#include "net/details/dpdk_utils.hpp"
#include "net/doca_types.hpp"
#include "net/dpdk_types.hpp"
#include "net/net_log.hpp"

namespace {

// NOLINTBEGIN(cppcoreguidelines-macro-usage,cppcoreguidelines-avoid-do-while,clang-diagnostic-gnu-zero-variadic-macro-arguments)

/* Error handling macros to reduce duplication */
/**
 * Check DOCA function result and return on error with logging
 *
 * @param[in] call DOCA function call
 * @param[in] msg Error message to log
 */
#define DOCA_RETURN_ON_ERR(call, msg)                                                              \
    do {                                                                                           \
        const doca_error_t _result = (call);                                                       \
        if (_result != DOCA_SUCCESS) {                                                             \
            RT_LOGC_ERROR(Net::NetDoca, msg ": {}", doca_error_get_descr(_result));                \
            return _result;                                                                        \
        }                                                                                          \
    } while (0)

/**
 * Check DOCA function result and return DOCA_ERROR_BAD_STATE on error with
 * logging Used in cleanup/destroy functions where specific error propagation is
 * not needed
 *
 * @param[in] call DOCA function call
 * @param[in] msg Error message to log
 */
#define DOCA_RETURN_BAD_STATE_ON_ERR(call, msg)                                                    \
    do {                                                                                           \
        const doca_error_t _result = (call);                                                       \
        if (_result != DOCA_SUCCESS) {                                                             \
            RT_LOGC_ERROR(Net::NetDoca, msg ": {}", doca_error_get_descr(_result));                \
            return DOCA_ERROR_BAD_STATE;                                                           \
        }                                                                                          \
    } while (0)

/**
 * Log error and return specified DOCA error code
 * Generic macro for immediate error return with logging
 *
 * @param[in] error_code DOCA error code to return
 * @param[in] msg Error message to log
 */
#define DOCA_RETURN_ERROR(error_code, msg)                                                         \
    do {                                                                                           \
        RT_LOGC_ERROR(Net::NetDoca, msg);                                                          \
        return (error_code);                                                                       \
    } while (0)

/**
 * Check condition and return specified DOCA error code if true
 * Generic conditional macro with condition logging
 *
 * @param[in] condition Condition to check (if true, return error)
 * @param[in] error_code DOCA error code to return if condition is true
 * @param[in] msg Error message to log
 */
#define DOCA_RETURN_ERROR_IF(condition, error_code, msg)                                           \
    do {                                                                                           \
        if (condition) {                                                                           \
            RT_LOGC_ERROR(Net::NetDoca, "({}) " msg, #condition);                                  \
            return (error_code);                                                                   \
        }                                                                                          \
    } while (0)

/**
 * Convenience macros for common error types
 * Built on top of the generic DOCA_RETURN_ERROR macro
 */
#define DOCA_RETURN_INVALID_VALUE(msg) DOCA_RETURN_ERROR(DOCA_ERROR_INVALID_VALUE, msg)
#define DOCA_RETURN_DRIVER_ERROR(msg) DOCA_RETURN_ERROR(DOCA_ERROR_DRIVER, msg)
#define DOCA_RETURN_NOT_FOUND(msg) DOCA_RETURN_ERROR(DOCA_ERROR_NOT_FOUND, msg)

/**
 * Convenience conditional macros for common error types
 * Built on top of the generic DOCA_RETURN_ERROR_IF macro
 */
#define DOCA_RETURN_INVALID_VALUE_IF(condition, msg)                                               \
    DOCA_RETURN_ERROR_IF(condition, DOCA_ERROR_INVALID_VALUE, msg)
#define DOCA_RETURN_DRIVER_ERROR_IF(condition, msg)                                                \
    DOCA_RETURN_ERROR_IF(condition, DOCA_ERROR_DRIVER, msg)
#define DOCA_RETURN_NOT_FOUND_IF(condition, msg)                                                   \
    DOCA_RETURN_ERROR_IF(condition, DOCA_ERROR_NOT_FOUND, msg)

} // namespace

namespace framework::net {

void doca_log_versions() {
    RT_LOGC_INFO(
            Net::NetDoca,
            "DOCA Versions SDK: {}, Runtime: {}",
            doca_version(),
            doca_version_runtime());
}

doca_error_t doca_init_logging(doca_log_backend **sdk_log) {
    /* Register a logger backend */
    DOCA_RETURN_ON_ERR(doca_log_backend_create_standard(), "Failed to create standard log backend");

    /* Register a logger backend for internal SDK errors and warnings */
    if (sdk_log != nullptr) {
        DOCA_RETURN_ON_ERR(
                doca_log_backend_create_with_file_sdk(stderr, sdk_log),
                "Failed to create SDK log backend");
        DOCA_RETURN_ON_ERR(
                doca_log_backend_set_sdk_level(*sdk_log, DOCA_LOG_LEVEL_WARNING),
                "Failed to set SDK log level");
    }

    RT_LOGC_DEBUG(Net::NetDoca, "DOCA logging initialized successfully");

    return DOCA_SUCCESS;
}

doca_error_t doca_open_device_with_pci(const std::string_view pci_addr, doca_dev **retval) {
    doca_devinfo **dev_list = nullptr;
    uint32_t nb_devs = 0;
    *retval = nullptr;

    DOCA_RETURN_ON_ERR(
            doca_devinfo_create_list(&dev_list, &nb_devs), "Failed to load doca devices list");

    const auto dev_list_sp = std::span<doca_devinfo *>(dev_list, nb_devs);
    for (auto *dev : dev_list_sp) {
        uint8_t is_addr_equal = 0;
        auto res = doca_devinfo_is_equal_pci_addr(dev, pci_addr.data(), &is_addr_equal);
        if (res == DOCA_SUCCESS && is_addr_equal != 0) {
            /* if device can be opened */
            res = doca_dev_open(dev, retval);
            if (res == DOCA_SUCCESS) {
                doca_devinfo_destroy_list(dev_list);
                return res;
            }
        }
    }

    RT_LOGC_WARN(Net::NetDoca, "Matching device not found ({})", pci_addr);
    doca_devinfo_destroy_list(dev_list);
    return DOCA_ERROR_NOT_FOUND;
}

doca_error_t doca_open_and_probe_device(const std::string_view nic_pcie_addr, doca_dev **ddev) {
    DOCA_RETURN_INVALID_VALUE_IF(nic_pcie_addr.empty(), "PCI address cannot be empty");
    if (nic_pcie_addr.size() >= DOCA_DEVINFO_PCI_ADDR_SIZE) {
        RT_LOGC_ERROR(
                Net::NetDoca,
                "PCI address '{}' too long (max {} characters)",
                nic_pcie_addr,
                DOCA_DEVINFO_PCI_ADDR_SIZE - 1);
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (const auto res = doca_open_device_with_pci(nic_pcie_addr.data(), ddev);
        res != DOCA_SUCCESS) {
        RT_LOGC_ERROR(
                Net::NetDoca, "Failed to open NIC device with PCI address '{}'", nic_pcie_addr);
        return res;
    }

    if (*ddev == nullptr) {
        RT_LOGC_ERROR(
                Net::NetDoca, "Failed to open NIC device with PCI address '{}'", nic_pcie_addr);
        return DOCA_ERROR_NOT_FOUND;
    }

    // Build device arguments with tx_pp for CX-6 devices
    std::string devargs = "txq_inline_max=0";
    const auto cx6_result = is_device_cx6(*ddev);
    if (cx6_result.has_value() && cx6_result.value()) {
        devargs = "tx_pp=500,txq_inline_max=0";
        RT_LOGC_INFO(
                Net::NetDoca, "CX-6 device detected, adding tx_pp=500ns for NIC {}", nic_pcie_addr);
    }

    RT_LOGC_DEBUG(
            Net::NetDoca, "Probing DOCA device '{}' with devargs: {}", nic_pcie_addr, devargs);

    DOCA_RETURN_ON_ERR(
            doca_dpdk_port_probe(*ddev, devargs.c_str()), "doca_dpdk_port_probe returned");

    RT_LOGC_DEBUG(
            Net::NetDoca,
            "Successfully initialized DOCA device with PCI address '{}'",
            nic_pcie_addr);

    return DOCA_SUCCESS;
}

doca_error_t doca_close_device(doca_dev *ddev) {
    DOCA_RETURN_INVALID_VALUE_IF(ddev == nullptr, "Invalid DOCA device parameter");

    DOCA_RETURN_ON_ERR(doca_dev_close(ddev), "Failed to close DOCA device");

    RT_LOGC_DEBUG(Net::NetDoca, "DOCA device closed successfully");
    return DOCA_SUCCESS;
}

tl::expected<bool, std::string> is_device_cx6(doca_dev *ddev) {
    if (ddev == nullptr) {
        return tl::unexpected("Device is null");
    }

    // Check device capabilities for CX-6 detection
    enum doca_eth_wait_on_time_type wait_on_time_mode {};
    const auto cap_result = doca_eth_txq_cap_get_wait_on_time_offload_supported(
            doca_dev_as_devinfo(ddev), &wait_on_time_mode);

    if (cap_result != DOCA_SUCCESS) {
        return tl::unexpected(std::format(
                "Failed to check CX-6 capability: {}", doca_error_get_descr(cap_result)));
    }

    const bool is_cx6 = (wait_on_time_mode == DOCA_ETH_WAIT_ON_TIME_TYPE_DPDK);
    return is_cx6;
}

doca_error_t doca_open_cuda_device(const std::string_view gpu_pcie_addr, doca_gpu **gpu_dev) {
    DOCA_RETURN_INVALID_VALUE_IF(
            gpu_pcie_addr.empty() || gpu_pcie_addr.size() >= DOCA_DEVINFO_PCI_ADDR_SIZE,
            "Invalid GPU PCI address parameter");

    /* In a multi-GPU system, ensure CUDA refers to the right GPU device */
    int cuda_id = 0;
    auto cuda_ret = cudaDeviceGetByPCIBusId(&cuda_id, gpu_pcie_addr.data());
    if (cuda_ret != cudaSuccess) {
        RT_LOGC_ERROR(
                Net::NetGpu,
                "Failed to get CUDA device PCI bus ID {} ({})",
                gpu_pcie_addr,
                cudaGetErrorString(cuda_ret));
        return DOCA_ERROR_INVALID_VALUE;
    }

    cuda_ret = cudaSetDevice(cuda_id);
    if (cuda_ret != cudaSuccess) {
        RT_LOGC_ERROR(
                Net::NetGpu,
                "Failed to set CUDA device {} ({})",
                cuda_id,
                cudaGetErrorString(cuda_ret));
        return DOCA_ERROR_INVALID_VALUE;
    }

    DOCA_RETURN_ON_ERR(
            doca_gpu_create(gpu_pcie_addr.data(), gpu_dev), "Failed to create DOCA GPU device");
    DOCA_RETURN_ERROR_IF(
            *gpu_dev == nullptr, DOCA_ERROR_NOT_FOUND, "Failed to create DOCA GPU device");

    RT_LOGC_INFO(Net::NetGpu, "CUDA device {} initialized for GPU {}", cuda_id, gpu_pcie_addr);

    return DOCA_SUCCESS;
}

doca_error_t doca_close_cuda_device(doca_gpu *gpu_dev) {
    DOCA_RETURN_INVALID_VALUE_IF(gpu_dev == nullptr, "Invalid GPU device parameter");
    DOCA_RETURN_ON_ERR(doca_gpu_destroy(gpu_dev), "Failed to destroy GPU device");

    RT_LOGC_DEBUG(Net::NetGpu, "DOCA GPU device destroyed successfully");
    return DOCA_SUCCESS;
}

std::string doca_device_id_to_pci_bus_id(const int cuda_device_id) {
    static constexpr size_t PCI_BUS_ID_BUFFER_SIZE = 32;
    std::array<char, PCI_BUS_ID_BUFFER_SIZE> buffer{};
    const auto cuda_ret = cudaDeviceGetPCIBusId(buffer.data(), buffer.size(), cuda_device_id);
    if (cuda_ret != cudaSuccess) {
        RT_LOGC_ERROR(
                Net::NetGpu,
                "Failed to get CUDA device PCI bus ID ({})",
                cudaGetErrorString(cuda_ret));
        return std::string{};
    }
    std::string pci_bus_id(buffer.data());
    std::transform(pci_bus_id.begin(), pci_bus_id.end(), pci_bus_id.begin(), [](unsigned char c) {
        return std::tolower(c);
    });

    RT_LOGC_DEBUG(Net::NetGpu, "CUDA device {} PCI bus ID: {}", cuda_device_id, pci_bus_id);
    return pci_bus_id;
}

doca_error_t doca_create_rxq(
        struct DocaRxQParams *rxq,
        doca_gpu *gpu_dev,
        doca_dev *ddev,
        const uint32_t max_pkt_num,
        const uint32_t max_pkt_size,
        const std::optional<DocaSemItems> &sem_items) {
    doca_error_t result{};
    uint32_t cyclic_buffer_size{};

    DOCA_RETURN_INVALID_VALUE_IF(
            rxq == nullptr || gpu_dev == nullptr || ddev == nullptr,
            "Invalid input parameters for RX queue creation");

    rxq->gpu_dev = gpu_dev;
    rxq->ddev = ddev;

    rxq->has_sem_items = false;
    rxq->sem_items = DocaSemItems{};
    if (sem_items.has_value()) {
        const auto valid_item_args = sem_items->num_items > 0 && sem_items->item_size > 0;
        rxq->has_sem_items = valid_item_args;
        if (valid_item_args) {
            rxq->sem_items = *sem_items;
        } else {
            RT_LOGC_WARN(
                    Net::NetDoca,
                    "Semaphore items provided but invalid - num_items={}, item_size={}",
                    sem_items->num_items,
                    sem_items->item_size);
        }
    }

    // Setup scope guard for automatic cleanup on failure
    bool cleanup_needed = true;
    auto cleanup_guard = gsl_lite::finally([rxq, &cleanup_needed] {
        if (cleanup_needed) {
            const auto cleanup_result = doca_destroy_rxq(rxq);
            if (cleanup_result != DOCA_SUCCESS) {
                RT_LOGC_ERROR(
                        Net::NetDoca,
                        "Failed to cleanup RX queue during error handling: {}",
                        doca_error_get_descr(cleanup_result));
            }
        }
    });

    RT_LOGC_INFO(
            Net::NetDoca,
            "Creating DOCA Ethernet receive queue - max_pkt_num={}, max_pkt_size={}",
            max_pkt_num,
            max_pkt_size);

    DOCA_RETURN_ON_ERR(
            doca_eth_rxq_create(rxq->ddev, max_pkt_num, max_pkt_size, &(rxq->eth_rxq_cpu)),
            "Failed to create DOCA Ethernet RX queue");

    DOCA_RETURN_ON_ERR(
            doca_eth_rxq_set_type(rxq->eth_rxq_cpu, DOCA_ETH_RXQ_TYPE_CYCLIC),
            "Failed to set RX queue type to cyclic");

    DOCA_RETURN_ON_ERR(
            doca_eth_rxq_estimate_packet_buf_size(
                    DOCA_ETH_RXQ_TYPE_CYCLIC,
                    0 /* rate */,
                    0 /* pkt_max_time */,
                    max_pkt_size,
                    max_pkt_num,
                    0 /* log_max_lro_pkt_sz */,
                    0 /* head_size */,
                    0 /* tail_size */,
                    &cyclic_buffer_size),
            "Failed to estimate packet buffer size");

    DOCA_RETURN_ON_ERR(doca_mmap_create(&rxq->pkt_buff_mmap), "Failed to create memory mapping");

    DOCA_RETURN_ON_ERR(
            doca_mmap_add_dev(rxq->pkt_buff_mmap, rxq->ddev),
            "Failed to add device to memory mapping");

    result = doca_gpu_mem_alloc(
            rxq->gpu_dev,
            cyclic_buffer_size,
            GPU_PAGE_SIZE,
            DOCA_GPU_MEM_TYPE_CPU_GPU,
            &rxq->gpu_pkt_addr,
            &rxq->cpu_pkt_addr);
    if (result != DOCA_SUCCESS || rxq->gpu_pkt_addr == nullptr) {
        RT_LOGC_ERROR(
                Net::NetDoca, "Failed to allocate GPU memory: {}", doca_error_get_descr(result));
        return result != DOCA_SUCCESS ? result : DOCA_ERROR_NO_MEMORY;
    }

    rxq->max_pkt_size = max_pkt_size;
    rxq->max_pkt_num = max_pkt_num;

    // Map GPU memory buffer used to receive packets with DMABuf
    result = doca_gpu_dmabuf_fd(
            rxq->gpu_dev, rxq->gpu_pkt_addr, cyclic_buffer_size, &(rxq->dmabuf_fd));
    if (result != DOCA_SUCCESS) {
        RT_LOGC_INFO(
                Net::NetDoca,
                "Mapping RX buffer ({} size {}B) with nvidia-peermem mode",
                static_cast<void *>(rxq->gpu_pkt_addr),
                cyclic_buffer_size);

        // If failed, use nvidia-peermem legacy method
        DOCA_RETURN_ON_ERR(
                doca_mmap_set_memrange(rxq->pkt_buff_mmap, rxq->gpu_pkt_addr, cyclic_buffer_size),
                "Failed to set memory range for mapping");
    } else {
        RT_LOGC_INFO(
                Net::NetDoca,
                "Mapping RX buffer ({} size {}B dmabuf fd {}) with dmabuf mode",
                static_cast<void *>(rxq->gpu_pkt_addr),
                cyclic_buffer_size,
                rxq->dmabuf_fd);

        DOCA_RETURN_ON_ERR(
                doca_mmap_set_dmabuf_memrange(
                        rxq->pkt_buff_mmap,
                        rxq->dmabuf_fd,
                        rxq->gpu_pkt_addr,
                        0 /* dmabuf_offset */,
                        cyclic_buffer_size),
                "Failed to set dmabuf memory range for mapping");
    }

    DOCA_RETURN_ON_ERR(
            doca_mmap_set_permissions(rxq->pkt_buff_mmap, DOCA_ACCESS_FLAG_LOCAL_READ_WRITE),
            "Failed to set memory mapping permissions");

    DOCA_RETURN_ON_ERR(doca_mmap_start(rxq->pkt_buff_mmap), "Failed to start memory mapping");

    DOCA_RETURN_ON_ERR(
            doca_eth_rxq_set_pkt_buf(
                    rxq->eth_rxq_cpu, rxq->pkt_buff_mmap, 0 /* mmap_offset */, cyclic_buffer_size),
            "Failed to set packet buffer for RX queue");

    rxq->eth_rxq_ctx = doca_eth_rxq_as_doca_ctx(rxq->eth_rxq_cpu);
    if (rxq->eth_rxq_ctx == nullptr) {
        RT_LOGC_ERROR(Net::NetDoca, "Failed to get DOCA context from RX queue");
        return DOCA_ERROR_UNEXPECTED;
    }

    DOCA_RETURN_ON_ERR(
            doca_ctx_set_datapath_on_gpu(rxq->eth_rxq_ctx, rxq->gpu_dev),
            "Failed to set datapath on GPU");

    DOCA_RETURN_ON_ERR(doca_ctx_start(rxq->eth_rxq_ctx), "Failed to start DOCA context");

    DOCA_RETURN_ON_ERR(
            doca_eth_rxq_get_gpu_handle(rxq->eth_rxq_cpu, &(rxq->eth_rxq_gpu)),
            "Failed to get GPU handle for RX queue");

    // Get the DOCA flow queue ID for later use in flow rules
    DOCA_RETURN_ON_ERR(
            doca_eth_rxq_get_flow_queue_id(rxq->eth_rxq_cpu, &rxq->dpdk_queue_idx),
            "Failed to get flow queue ID");
    RT_LOGC_DEBUG(
            Net::NetDoca,
            "DOCA Ethernet RX queue created successfully - Flow queue ID: {}",
            rxq->dpdk_queue_idx);

    // Create semaphore only if configured
    if (rxq->has_sem_items) {
        DOCA_RETURN_ON_ERR(
                doca_gpu_semaphore_create(rxq->gpu_dev, &rxq->sem_cpu),
                "Failed to create DOCA GPU semaphore");

        DOCA_RETURN_ON_ERR(
                doca_gpu_semaphore_set_items_num(rxq->sem_cpu, rxq->sem_items.num_items),
                "Failed to set semaphore items number");

        DOCA_RETURN_ON_ERR(
                doca_gpu_semaphore_set_memory_type(rxq->sem_cpu, DOCA_GPU_MEM_TYPE_GPU),
                "Failed to set semaphore memory type on GPU");

        DOCA_RETURN_ON_ERR(
                doca_gpu_semaphore_set_custom_info(
                        rxq->sem_cpu, rxq->sem_items.item_size, DOCA_GPU_MEM_TYPE_CPU_GPU),
                "Failed to set semaphore custom info");

        DOCA_RETURN_ON_ERR(
                doca_gpu_semaphore_start(rxq->sem_cpu), "Failed to start DOCA GPU semaphore");

        DOCA_RETURN_ON_ERR(
                doca_gpu_semaphore_get_gpu_handle(rxq->sem_cpu, &rxq->sem_gpu),
                "Failed to get GPU handle for semaphore");

        RT_LOGC_DEBUG(
                Net::NetDoca,
                "DOCA GPU semaphore created successfully with {} items of size {} bytes",
                rxq->sem_items.num_items,
                rxq->sem_items.item_size);
    } else {
        RT_LOGC_DEBUG(Net::NetDoca, "Skipping semaphore creation (not configured)");
    }

    // Dismiss the cleanup guard since we succeeded
    cleanup_needed = false;
    return DOCA_SUCCESS;
}

doca_error_t doca_destroy_rxq(struct DocaRxQParams *rxq) {
    DOCA_RETURN_INVALID_VALUE_IF(
            rxq == nullptr, "Invalid input parameter for RX queue destruction");

    RT_LOGC_DEBUG(Net::NetDoca, "Destroying DOCA Ethernet receive queue");

    // First cleanup DPDK flow rule to avoid reference issues
    if (const auto result = doca_destroy_flow_rule(rxq); result != DOCA_SUCCESS) {
        RT_LOGC_WARN(
                Net::NetDoca,
                "Failed to cleanup DPDK flow rule, continuing with destruction: {}",
                doca_error_get_descr(result));
    }

    if (rxq->eth_rxq_ctx != nullptr) {
        DOCA_RETURN_BAD_STATE_ON_ERR(
                doca_ctx_stop(rxq->eth_rxq_ctx), "Failed to stop DOCA context");
    }

    // Stop and destroy the semaphore if it exists
    if (rxq->sem_cpu != nullptr) {
        const auto stop_result = doca_gpu_semaphore_stop(rxq->sem_cpu);
        if (stop_result != DOCA_SUCCESS) {
            RT_LOGC_WARN(
                    Net::NetDoca,
                    "Failed to stop DOCA GPU semaphore: {}, continuing cleanup",
                    doca_error_get_descr(stop_result));
        }
        const auto destroy_result = doca_gpu_semaphore_destroy(rxq->sem_cpu);
        if (destroy_result != DOCA_SUCCESS) {
            RT_LOGC_WARN(
                    Net::NetDoca,
                    "Failed to destroy DOCA GPU semaphore: {}, continuing cleanup",
                    doca_error_get_descr(destroy_result));
        }
        rxq->sem_cpu = nullptr;
        rxq->sem_gpu = nullptr;
    }

    if (rxq->gpu_pkt_addr != nullptr) {
        DOCA_RETURN_BAD_STATE_ON_ERR(
                doca_gpu_mem_free(rxq->gpu_dev, rxq->gpu_pkt_addr), "Failed to free GPU memory");
    }

    if (rxq->eth_rxq_cpu != nullptr) {
        DOCA_RETURN_BAD_STATE_ON_ERR(
                doca_eth_rxq_destroy(rxq->eth_rxq_cpu), "Failed to destroy DOCA Ethernet RX queue");
    }

    if (rxq->pkt_buff_mmap != nullptr) {
        DOCA_RETURN_BAD_STATE_ON_ERR(
                doca_mmap_destroy(rxq->pkt_buff_mmap), "Failed to destroy memory mapping");
    }

    RT_LOGC_DEBUG(Net::NetDoca, "DOCA Ethernet receive queue destroyed successfully");

    return DOCA_SUCCESS;
}

doca_error_t doca_get_dpdk_port_id(doca_dev *dev_input, uint16_t *port_id) {
    doca_error_t result{};
    uint16_t dpdk_port_id{};

    DOCA_RETURN_INVALID_VALUE_IF(
            dev_input == nullptr || port_id == nullptr,
            "Invalid input parameters for DPDK port ID lookup");

    *port_id = RTE_MAX_ETHPORTS;

    std::array<char, DOCA_DEVINFO_PCI_ADDR_SIZE> pci_addr_input{};
    DOCA_RETURN_ON_ERR(
            doca_devinfo_get_pci_addr_str(doca_dev_as_devinfo(dev_input), pci_addr_input.data()),
            "Failed to get device PCI address");

    uint32_t valid_port_count{};
    for (dpdk_port_id = 0; dpdk_port_id < RTE_MAX_ETHPORTS; ++dpdk_port_id) {
        if (rte_eth_dev_is_valid_port(dpdk_port_id) != 0) {
            ++valid_port_count;
        }
    }
    RT_LOGC_DEBUG(
            Net::NetDpdk,
            "Looking for DPDK port for DOCA device with PCI address: {} "
            "out of {} valid DPDK ports",
            pci_addr_input.data(),
            valid_port_count);

    for (dpdk_port_id = 0; dpdk_port_id < RTE_MAX_ETHPORTS; ++dpdk_port_id) {
        // Search for the probed devices
        if (rte_eth_dev_is_valid_port(dpdk_port_id) == 0) {
            continue;
        }

        doca_dev *dev_local{};
        result = doca_dpdk_port_as_dev(dpdk_port_id, &dev_local);
        if (result != DOCA_SUCCESS) {
            RT_LOGC_DEBUG(
                    Net::NetDpdk,
                    "Failed to find DOCA device for DPDK port {}: {}",
                    dpdk_port_id,
                    doca_error_get_descr(result));
            continue;
        }
        // Setup scope guard for automatic cleanup of dev_local
        bool dev_cleanup_needed = true;
        auto dev_cleanup_guard = gsl_lite::finally([&dev_local, &dev_cleanup_needed] {
            if (dev_cleanup_needed && dev_local != nullptr) {
                const doca_error_t close_result = doca_dev_close(dev_local);
                if (close_result != DOCA_SUCCESS) {
                    RT_LOGC_DEBUG(
                            Net::NetDpdk,
                            "Failed to close temporary DOCA device: {}",
                            doca_error_get_descr(close_result));
                }
            }
        });

        // Get the PCI address of this port's device for comparison logging
        std::array<char, DOCA_DEVINFO_PCI_ADDR_SIZE> dev_pci_addr{};
        const doca_error_t pci_result =
                doca_devinfo_get_pci_addr_str(doca_dev_as_devinfo(dev_local), dev_pci_addr.data());
        if (pci_result != DOCA_SUCCESS) {
            RT_LOGC_DEBUG(
                    Net::NetDpdk,
                    "DPDK port {}: Could not get PCI address for comparison",
                    dpdk_port_id);
            continue;
        }

        uint8_t is_addr_equal{};
        result = doca_devinfo_is_equal_pci_addr(
                doca_dev_as_devinfo(dev_local), pci_addr_input.data(), &is_addr_equal);
        if (result != DOCA_SUCCESS) {
            RT_LOGC_DEBUG(
                    Net::NetDpdk,
                    "Failed to compare PCI address for DPDK port {}: {}",
                    dpdk_port_id,
                    doca_error_get_descr(result));
            continue;
        }

        if (is_addr_equal != 0) {
            *port_id = dpdk_port_id;
            RT_LOGC_DEBUG(
                    Net::NetDpdk,
                    "Found DPDK port {} for DOCA device {}",
                    dpdk_port_id,
                    pci_addr_input.data());
            dev_cleanup_needed = false; // Don't clean up the matching device
            break;
        }
    }

    DOCA_RETURN_NOT_FOUND_IF(*port_id == RTE_MAX_ETHPORTS, "No DPDK port matches the DOCA device");

    return DOCA_SUCCESS;
}

doca_error_t doca_get_mac_addr_from_pci(const std::string_view pci_addr, MacAddress &mac_addr) {
    doca_devinfo **dev_list = nullptr;
    uint32_t dev_count{};
    doca_error_t result{};
    bool found = false; // Will be modified in the loop

    DOCA_RETURN_INVALID_VALUE_IF(
            pci_addr.empty() || pci_addr.size() >= DOCA_DEVINFO_PCI_ADDR_SIZE,
            "Invalid input parameters for MAC address lookup");

    // Get list of available DOCA devices
    result = doca_devinfo_create_list(&dev_list, &dev_count);
    DOCA_RETURN_ON_ERR(result, "Failed to create device info list");

    RT_LOGC_DEBUG(
            Net::NetDoca,
            "Searching for MAC address of PCIe device: {} out of {} DOCA devices",
            pci_addr,
            dev_count);

    const auto dev_list_sp = std::span<doca_devinfo *>(dev_list, dev_count);

    // Search through all devices to find the one with matching PCIe address
    for (auto *dev : dev_list_sp) {
        std::string dev_pci_addr;
        static constexpr auto STRING_HANDLES_NULL_CHAR = 1;
        dev_pci_addr.resize(DOCA_DEVINFO_PCI_ADDR_SIZE - STRING_HANDLES_NULL_CHAR);

        result = doca_devinfo_get_pci_addr_str(dev, dev_pci_addr.data());
        if (result != DOCA_SUCCESS) {
            RT_LOGC_WARN(
                    Net::NetDoca,
                    "Failed to get PCI address for device: {}",
                    doca_error_get_descr(result));
            continue;
        }

        RT_LOGC_DEBUG(
                Net::NetDoca,
                "Comparing device PCI address {} with target {}",
                dev_pci_addr,
                pci_addr);

        // Compare PCIe addresses
        if (std::string_view{dev_pci_addr} == pci_addr) {
            // Found matching device, get its MAC address
            result = doca_devinfo_get_mac_addr(dev, mac_addr.bytes.data(), MacAddress::ADDRESS_LEN);
            if (result == DOCA_SUCCESS) {
                RT_LOGC_DEBUG(
                        Net::NetDoca,
                        "Found MAC address for {}: {}",
                        pci_addr,
                        mac_addr.to_string());
                found = true;
                break;
            } else {
                RT_LOGC_WARN(
                        Net::NetDoca,
                        "Found matching PCIe device but failed to get MAC address: {}",
                        doca_error_get_descr(result));
            }
        }
    }

    // Cleanup device list
    doca_devinfo_destroy_list(dev_list);

    DOCA_RETURN_NOT_FOUND_IF(!found, "No device found with the specified PCIe address");

    return DOCA_SUCCESS;
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
doca_error_t doca_create_flow_rule(
        struct DocaRxQParams *rxq,
        const std::string_view nic_pcie_addr,
        const MacAddress &sender_mac_addr,
        const uint16_t ether_type,
        const std::optional<uint16_t> &vlan_tci) {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    rte_flow_attr attr{};
    attr.group = 0;
    attr.priority = 0;
    attr.ingress = 1;
    attr.egress = 0;
    attr.transfer = 0;

    DOCA_RETURN_INVALID_VALUE_IF(
            rxq == nullptr || nic_pcie_addr.empty(),
            "Invalid input parameters for flow rule creation");

    // Get the DPDK port ID for this device
    uint16_t port_id{};
    const int ret_port = doca_get_dpdk_port_id(rxq->ddev, &port_id);
    DOCA_RETURN_NOT_FOUND_IF(ret_port != 0, "Failed to get DPDK port ID");

    // Check if port is started, and start it if needed
    rte_eth_dev_info dev_info{};
    int ret = rte_eth_dev_info_get(port_id, &dev_info);
    DOCA_RETURN_ERROR_IF(ret != 0, DOCA_ERROR_BAD_STATE, "Failed to get device info for DPDK port");

    // Disable Ethernet flow control to prevent packet drops
    rte_eth_fc_conf fc_conf{};
    ret = rte_eth_dev_flow_ctrl_get(port_id, &fc_conf);
    if (ret == 0) {
        fc_conf.mode = RTE_ETH_FC_NONE;
        ret = rte_eth_dev_flow_ctrl_set(port_id, &fc_conf);
        if (ret == 0) {
            RT_LOGC_DEBUG(Net::NetDpdk, "Disabled Ethernet flow control on port {}", port_id);
        } else {
            RT_LOGC_WARN(
                    Net::NetDpdk, "Failed to disable flow control on port {}: {}", port_id, ret);
        }
    } else {
        RT_LOGC_WARN(
                Net::NetDpdk, "Failed to get flow control config on port {}: {}", port_id, ret);
    }

    // Get MAC addresses
    MacAddress receiver_mac{};

    // Get receiver's MAC address (this NIC's MAC address)
    DOCA_RETURN_ON_ERR(
            doca_get_mac_addr_from_pci(nic_pcie_addr, receiver_mac),
            "Failed to get receiver MAC address from PCIe address");

    // Set up Ethernet matching for packets
    rte_flow_item_eth eth_spec{};
    rte_flow_item_eth eth_mask{};

    // NOLINTBEGIN(cppcoreguidelines-pro-type-union-access,cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)
    // Match packets destined to this NIC (receiver's MAC address)
    static constexpr uint8_t FULL_MASK_BYTE = 0xFF;
    static constexpr uint16_t FULL_MASK_SHORT = 0xFFFF;
    static constexpr uint16_t VLAN_ID_MASK = 0x0FFF;

    std::copy_n(receiver_mac.bytes.data(), RTE_ETHER_ADDR_LEN, eth_spec.dst.addr_bytes);
    std::fill_n(eth_mask.dst.addr_bytes, RTE_ETHER_ADDR_LEN, FULL_MASK_BYTE);

    // Match packets from the sender's MAC address
    std::copy_n(sender_mac_addr.bytes.data(), RTE_ETHER_ADDR_LEN, eth_spec.src.addr_bytes);
    std::fill_n(eth_mask.src.addr_bytes, RTE_ETHER_ADDR_LEN, FULL_MASK_BYTE);

    // Set EtherType based on VLAN configuration
    const bool is_vlan = vlan_tci.has_value();
    eth_spec.type = rte_cpu_to_be_16(is_vlan ? RTE_ETHER_TYPE_VLAN : ether_type);
    eth_mask.type = FULL_MASK_SHORT;
    // NOLINTEND(cppcoreguidelines-pro-type-union-access,cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)

    const uint16_t vlan_id = is_vlan ? static_cast<uint16_t>(*vlan_tci & VLAN_ID_MASK) : 0;

    // Set up VLAN matching if VLAN TCI is provided
    rte_flow_item_vlan vlan_spec{};
    rte_flow_item_vlan vlan_mask{};
    if (is_vlan) {
        // NOLINTBEGIN(cppcoreguidelines-pro-type-union-access)
        // TCI format: [3 bits priority | 1 bit DEI | 12 bits VLAN ID]
        // Extract only VLAN ID (12 bits), ignore priority and DEI
        vlan_spec.tci = rte_cpu_to_be_16(vlan_id);
        vlan_mask.tci = rte_cpu_to_be_16(VLAN_ID_MASK); // Match only 12-bit VLAN ID

        // Match the inner EtherType inside the VLAN tag
        vlan_spec.inner_type = rte_cpu_to_be_16(ether_type);
        vlan_mask.inner_type = FULL_MASK_SHORT;
        // NOLINTEND(cppcoreguidelines-pro-type-union-access)
    }

    // Set up action to queue packets to our GPU queue
    rte_flow_action_queue queue{};
    uint16_t flow_queue_id{};
    const doca_error_t flow_result =
            doca_eth_rxq_get_flow_queue_id(rxq->eth_rxq_cpu, &flow_queue_id);
    if (flow_result == DOCA_SUCCESS) {
        queue.index = flow_queue_id;
        RT_LOGC_DEBUG(Net::NetDpdk, "Using DOCA flow queue ID: {}", flow_queue_id);
    } else {
        RT_LOGC_WARN(
                Net::NetDpdk,
                "Could not get flow queue ID, using DPDK queue ID: {}",
                rxq->dpdk_queue_idx);
        queue.index = rxq->dpdk_queue_idx;
    }

    const std::array<rte_flow_action, 2> actions{
            {{.type = RTE_FLOW_ACTION_TYPE_QUEUE, .conf = &queue},
             {.type = RTE_FLOW_ACTION_TYPE_END, .conf = nullptr}}};

    // Prepare ETH, VLAN and END items
    const rte_flow_item eth_item{
            .type = RTE_FLOW_ITEM_TYPE_ETH, .spec = &eth_spec, .last = nullptr, .mask = &eth_mask};
    const rte_flow_item vlan_item{
            .type = RTE_FLOW_ITEM_TYPE_VLAN,
            .spec = &vlan_spec,
            .last = nullptr,
            .mask = &vlan_mask};
    const rte_flow_item end_item{
            .type = RTE_FLOW_ITEM_TYPE_END, .spec = nullptr, .last = nullptr, .mask = nullptr};
    // Build pattern array with eth_item and end_item; override middle if VLAN is requested
    std::array<rte_flow_item, 3> patterns{{
            eth_item,
            is_vlan ? vlan_item : end_item,
            end_item,
    }};

    // Validate the flow rule first
    rte_flow_error err{};
    ret = rte_flow_validate(port_id, &attr, patterns.data(), actions.data(), &err);
    DOCA_RETURN_ERROR_IF(ret != 0, DOCA_ERROR_BAD_STATE, "Failed to validate DPDK flow rule");

    // We must start the port in order to create the flow rule
    if (auto start_result = dpdk_start_eth_dev(port_id); start_result) {
        RT_LOGC_ERROR(Net::NetDpdk, "Failed to start DPDK port: {}", get_error_name(start_result));
        return DOCA_ERROR_BAD_STATE;
    }

    // Create the flow rule
    rte_flow *flow = rte_flow_create(port_id, &attr, patterns.data(), actions.data(), &err);
    if (flow == nullptr) {
        RT_LOGC_ERROR(
                Net::NetDpdk,
                "Failed to create DPDK flow rule: {}",
                err.message != nullptr ? err.message : "unknown error");
        return DOCA_ERROR_DRIVER;
    }

    // Store flow rule and port ID for cleanup
    rxq->dpdk_flow_rule = flow;
    rxq->dpdk_port_id = port_id;

    // Log with VLAN ID displayed as "(not set)" or "(<id>)"
    const std::string vlan_id_str = is_vlan ? std::to_string(vlan_id) : std::string("not set");
    const uint16_t outer_eth_type = is_vlan ? RTE_ETHER_TYPE_VLAN : ether_type;
    const std::string outer_eth_type_str = std::format("{:#x}", outer_eth_type);
    const std::string inner_eth_type_str =
            is_vlan ? std::format("{:#x}", ether_type) : std::string("not set");
    RT_LOGC_INFO(
            Net::NetDpdk,
            "Successfully created RX flow rule for GPU queue {}: "
            "Dest MAC {}, "
            "Source MAC {}, "
            "Outer EtherType {}, "
            "Inner EtherType ({}), "
            "VLAN ID ({})",
            queue.index,
            receiver_mac.to_string(),
            sender_mac_addr.to_string(),
            outer_eth_type_str,
            inner_eth_type_str,
            vlan_id_str);

    return DOCA_SUCCESS;
}

doca_error_t doca_destroy_flow_rule(struct DocaRxQParams *rxq) {
    rte_flow_error err{};

    if (rxq == nullptr || rxq->dpdk_flow_rule == nullptr) {
        return DOCA_SUCCESS; // Nothing to cleanup
    }

    RT_LOGC_DEBUG(Net::NetDpdk, "Destroying DPDK flow rule on port {}", rxq->dpdk_port_id);

    const int ret = rte_flow_destroy(rxq->dpdk_port_id, rxq->dpdk_flow_rule, &err);
    if (ret != 0) {
        RT_LOGC_ERROR(
                Net::NetDpdk,
                "Failed to destroy DPDK flow rule: {}",
                err.message != nullptr ? err.message : "unknown error");
        return DOCA_ERROR_DRIVER;
    }

    rxq->dpdk_flow_rule = nullptr;
    RT_LOGC_DEBUG(Net::NetDpdk, "DPDK flow rule destroyed successfully");

    return DOCA_SUCCESS;
}

bool doca_is_gdrcopy_compatible_size(const size_t size) noexcept {
    return size >= GPU_PAGE_SIZE && (size & GPU_PAGE_OFFSET) == 0;
}

size_t doca_align_to_gpu_page(const size_t size) noexcept {
    if (size < GPU_PAGE_SIZE) {
        return GPU_PAGE_SIZE;
    }

    // Check if already aligned
    if (doca_is_gdrcopy_compatible_size(size)) {
        return size;
    }

    // Align to next boundary
    return (size + GPU_PAGE_OFFSET) & GPU_PAGE_MASK;
}

tl::expected<bool, std::string> doca_is_rdma_supported(doca_dev *ddev) {
    if (ddev == nullptr) {
        return tl::unexpected("Device is null");
    }

    const auto *devinfo = doca_dev_as_devinfo(ddev);

    // Check if device supports RDMA read operations
    const auto read_result = doca_rdma_cap_task_read_is_supported(devinfo);
    if (read_result != DOCA_SUCCESS) {
        const std::string error_msg = std::format(
                "RDMA read operations not supported on device: {}",
                doca_error_get_descr(read_result));
        RT_LOGC_DEBUG(Net::NetDoca, "{}", error_msg);
        return tl::unexpected(error_msg);
    }

    // Check if device supports RDMA write operations
    const auto write_result = doca_rdma_cap_task_write_is_supported(devinfo);
    if (write_result != DOCA_SUCCESS) {
        const std::string error_msg = std::format(
                "RDMA write operations not supported on device: {}",
                doca_error_get_descr(write_result));
        RT_LOGC_DEBUG(Net::NetDoca, "{}", error_msg);
        return tl::unexpected(error_msg);
    }

    RT_LOGC_INFO(Net::NetDoca, "RDMA read and write operations supported on NIC device");
    return true;
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
doca_error_t doca_create_txq(
        struct DocaTxQParams *txq,
        doca_gpu *gpu_dev,
        doca_dev *ddev,
        const uint32_t pkt_size,
        const uint32_t pkt_num,
        const uint32_t max_sq_descr_num,
        const std::string_view nic_pcie_addr,
        const MacAddress &dest_mac_addr,
        const uint16_t ether_type,
        const std::optional<uint16_t> &vlan_tci) {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    doca_error_t result{};
    uint32_t buffer_size{};
    MacAddress source_mac{};
    cudaError_t cuda_result{};
    std::vector<uint8_t> cpu_pkt_buffer{};

    DOCA_RETURN_INVALID_VALUE_IF(
            txq == nullptr || gpu_dev == nullptr || ddev == nullptr || nic_pcie_addr.empty(),
            "Invalid input parameters for TX queue creation");

    txq->gpu_dev = gpu_dev;
    txq->ddev = ddev;
    txq->pkt_size = pkt_size;
    txq->num_packets = pkt_num;
    txq->inflight_sends = max_sq_descr_num / 2;

    // Setup scope guard for automatic cleanup on failure
    bool cleanup_needed = true;
    auto cleanup_guard = gsl_lite::finally([txq, &cleanup_needed] {
        if (cleanup_needed) {
            const auto cleanup_result = doca_destroy_txq(txq);
            if (cleanup_result != DOCA_SUCCESS) {
                RT_LOGC_ERROR(
                        Net::NetDoca,
                        "Failed to cleanup TX queue during error handling: {}",
                        doca_error_get_descr(cleanup_result));
            }
        }
    });

    const uint32_t original_buffer_size = txq->num_packets * pkt_size;
    buffer_size = static_cast<uint32_t>(doca_align_to_gpu_page(original_buffer_size));

    if (buffer_size != original_buffer_size) {
        RT_LOGC_INFO(
                Net::NetDoca,
                "GPU buffer alignment: {} bytes -> {} bytes (GDRCopy requires "
                "{}KB pages)",
                original_buffer_size,
                buffer_size,
                GPU_PAGE_SIZE / 1024U);
    } else {
        RT_LOGC_DEBUG(
                Net::NetDoca, "GPU buffer size {} bytes already GDRCopy compatible", buffer_size);
    }

    RT_LOGC_INFO(
            Net::NetDoca,
            "Creating DOCA Ethernet transmit queue - max_sq_descr_num={}, "
            "pkt_size={}, pkt_num={}",
            max_sq_descr_num,
            pkt_size,
            pkt_num);

    DOCA_RETURN_ON_ERR(
            doca_eth_txq_create(txq->ddev, max_sq_descr_num, &(txq->eth_txq_cpu)),
            "Failed to create DOCA Ethernet TX queue");

    DOCA_RETURN_ON_ERR(
            doca_eth_txq_set_l3_chksum_offload(txq->eth_txq_cpu, 1),
            "Failed to set L3 checksum offload");

    DOCA_RETURN_ON_ERR(
            doca_eth_txq_set_l4_chksum_offload(txq->eth_txq_cpu, 1),
            "Failed to set L4 checksum offload");

    // Application can check Txq completions on the GPU. By default, it can be
    // done by CPU.
    DOCA_RETURN_ON_ERR(
            doca_eth_txq_gpu_set_completion_on_gpu(txq->eth_txq_cpu),
            "Failed to set completion on GPU");

    txq->eth_txq_ctx = doca_eth_txq_as_doca_ctx(txq->eth_txq_cpu);
    if (txq->eth_txq_ctx == nullptr) {
        RT_LOGC_ERROR(Net::NetDoca, "Failed to get DOCA context from TX queue");
        return DOCA_ERROR_UNEXPECTED;
    }

    DOCA_RETURN_ON_ERR(
            doca_ctx_set_datapath_on_gpu(txq->eth_txq_ctx, txq->gpu_dev),
            "Failed to set datapath on GPU");

    // Create progress engine
    DOCA_RETURN_ON_ERR(doca_pe_create(&txq->eth_txq_pe), "Failed to create progress engine");

    // Connect progress engine to context
    DOCA_RETURN_ON_ERR(
            doca_pe_connect_ctx(txq->eth_txq_pe, txq->eth_txq_ctx),
            "Failed to connect progress engine to TX queue context");

    DOCA_RETURN_ON_ERR(doca_ctx_start(txq->eth_txq_ctx), "Failed to start DOCA context");

    DOCA_RETURN_ON_ERR(
            doca_eth_txq_get_gpu_handle(txq->eth_txq_cpu, &(txq->eth_txq_gpu)),
            "Failed to get GPU handle for TX queue");

    DOCA_RETURN_ON_ERR(doca_mmap_create(&txq->pkt_buff_mmap), "Failed to create memory mapping");

    DOCA_RETURN_ON_ERR(
            doca_mmap_add_dev(txq->pkt_buff_mmap, txq->ddev),
            "Failed to add device to memory mapping");

    result = doca_gpu_mem_alloc(
            txq->gpu_dev,
            buffer_size,
            GPU_PAGE_SIZE,
            DOCA_GPU_MEM_TYPE_GPU,
            &txq->gpu_pkt_addr,
            nullptr);
    if (result != DOCA_SUCCESS || txq->gpu_pkt_addr == nullptr) {
        RT_LOGC_ERROR(
                Net::NetDoca, "Failed to allocate GPU memory: {}", doca_error_get_descr(result));
        return result != DOCA_SUCCESS ? result : DOCA_ERROR_NO_MEMORY;
    }

    // Allocate CPU buffer for packet preparation using RAII vector
    try {
        cpu_pkt_buffer.resize(buffer_size, 0);
    } catch (const std::bad_alloc &e) {
        RT_LOGC_ERROR(
                Net::NetDoca, "Failed to allocate CPU memory for packet preparation: {}", e.what());
        return DOCA_ERROR_NO_MEMORY;
    }

    // Get source MAC address from NIC PCIe address
    result = doca_get_mac_addr_from_pci(nic_pcie_addr, source_mac);
    if (result != DOCA_SUCCESS) {
        RT_LOGC_ERROR(
                Net::NetDoca,
                "Failed to get source MAC address from NIC PCIe address: {}",
                doca_error_get_descr(result));
        return result;
    }

    RT_LOGC_INFO(
            Net::NetDoca,
            "TX MAC addresses - Source: {}, Destination: {}",
            source_mac.to_string(),
            dest_mac_addr.to_string());

    // Create test packets with Ethernet headers (optionally with 802.1Q VLAN tag)
    for (uint32_t idx = 0; idx < txq->num_packets; ++idx) {
        const std::size_t pkt_offset = static_cast<std::size_t>(idx) * pkt_size;

        // Create a span for this packet's memory for safe access
        const auto packet_span = std::span<uint8_t>(cpu_pkt_buffer).subspan(pkt_offset, pkt_size);

        // Ethernet header setup at byte level using spans (no reinterpret_cast
        // needed)
        static constexpr auto TWICE_ETHER_ADDR_LEN =
                static_cast<std::size_t>(2U) * MacAddress::ADDRESS_LEN;
        static constexpr auto VLAN_HEADER_BYTES = static_cast<std::size_t>(4U); // TPID+TCI
        const bool add_vlan = vlan_tci.has_value();
        const std::size_t eth_header_size =
                TWICE_ETHER_ADDR_LEN + (add_vlan ? VLAN_HEADER_BYTES : 0U) + sizeof(uint16_t);

        // Set destination MAC address using subspan
        auto dest_mac_span = packet_span.subspan(0, MacAddress::ADDRESS_LEN);
        std::copy(dest_mac_addr.bytes.begin(), dest_mac_addr.bytes.end(), dest_mac_span.begin());

        // Set source MAC address using subspan
        auto src_mac_span = packet_span.subspan(MacAddress::ADDRESS_LEN, MacAddress::ADDRESS_LEN);
        std::copy(source_mac.bytes.begin(), source_mac.bytes.end(), src_mac_span.begin());

        std::size_t cursor = TWICE_ETHER_ADDR_LEN;
        if (add_vlan) {
            // Insert 802.1Q TPID 0x8100 and VLAN TCI (both big-endian)
            auto tpid_span = packet_span.subspan(cursor, sizeof(uint16_t));
            const uint16_t tpid_be = __builtin_bswap16(RTE_ETHER_TYPE_VLAN);
            std::memcpy(tpid_span.data(), &tpid_be, sizeof(uint16_t));
            cursor += sizeof(uint16_t);

            auto tci_span = packet_span.subspan(cursor, sizeof(uint16_t));
            const uint16_t tci_be = __builtin_bswap16(vlan_tci.value());
            std::memcpy(tci_span.data(), &tci_be, sizeof(uint16_t));
            cursor += sizeof(uint16_t);
        }

        // Write EtherType (inner type if VLAN is present)
        auto ethertype_span = packet_span.subspan(cursor, sizeof(uint16_t));
        const uint16_t ethertype_be = __builtin_bswap16(ether_type);
        std::memcpy(ethertype_span.data(), &ethertype_be, sizeof(uint16_t));

        // Zero the payload past the ethernet header
        const auto payload_span = packet_span.subspan(eth_header_size);
        std::memset(payload_span.data(), 0, payload_span.size());
    }

    cuda_result =
            cudaMemcpy(txq->gpu_pkt_addr, cpu_pkt_buffer.data(), buffer_size, cudaMemcpyDefault);
    DOCA_RETURN_DRIVER_ERROR_IF(cuda_result != cudaSuccess, "CUDA memory copy failed");

    // Map GPU memory buffer used to transmit packets with DMABuf
    result = doca_gpu_dmabuf_fd(txq->gpu_dev, txq->gpu_pkt_addr, buffer_size, &(txq->dmabuf_fd));
    if (result != DOCA_SUCCESS) {
        RT_LOGC_INFO(
                Net::NetDoca,
                "Mapping transmit queue buffer ({} size {}B) with "
                "nvidia-peermem mode",
                static_cast<void *>(txq->gpu_pkt_addr),
                buffer_size);

        // If failed, use nvidia-peermem legacy method
        DOCA_RETURN_ON_ERR(
                doca_mmap_set_memrange(txq->pkt_buff_mmap, txq->gpu_pkt_addr, buffer_size),
                "Failed to set memory range for mapping");
    } else {
        RT_LOGC_INFO(
                Net::NetDoca,
                "Mapping transmit queue buffer ({} size {}B dmabuf fd {}) "
                "with dmabuf mode",
                static_cast<void *>(txq->gpu_pkt_addr),
                buffer_size,
                txq->dmabuf_fd);

        DOCA_RETURN_ON_ERR(
                doca_mmap_set_dmabuf_memrange(
                        txq->pkt_buff_mmap,
                        txq->dmabuf_fd,
                        txq->gpu_pkt_addr,
                        0 /* dmabuf_offset */,
                        buffer_size),
                "Failed to set dmabuf memory range for mapping");
    }

    DOCA_RETURN_ON_ERR(
            doca_mmap_set_permissions(txq->pkt_buff_mmap, DOCA_ACCESS_FLAG_LOCAL_READ_WRITE),
            "Failed to set memory mapping permissions");

    DOCA_RETURN_ON_ERR(doca_mmap_start(txq->pkt_buff_mmap), "Failed to start memory mapping");

    DOCA_RETURN_ON_ERR(
            doca_buf_arr_create(txq->num_packets, &txq->buf_arr), "Failed to create buffer array");

    DOCA_RETURN_ON_ERR(
            doca_buf_arr_set_target_gpu(txq->buf_arr, txq->gpu_dev),
            "Failed to set buffer array target GPU");

    DOCA_RETURN_ON_ERR(
            doca_buf_arr_set_params(txq->buf_arr, txq->pkt_buff_mmap, txq->pkt_size, 0),
            "Failed to set buffer array parameters");

    DOCA_RETURN_ON_ERR(doca_buf_arr_start(txq->buf_arr), "Failed to start buffer array");

    DOCA_RETURN_ON_ERR(
            doca_buf_arr_get_gpu_handle(txq->buf_arr, &(txq->buf_arr_gpu)),
            "Failed to get buffer array GPU handle");

    RT_LOGC_DEBUG(Net::NetDoca, "DOCA Ethernet TX queue created successfully");

    // Dismiss the cleanup guard since we succeeded
    cleanup_needed = false;
    return DOCA_SUCCESS;
}

doca_error_t doca_destroy_txq(struct DocaTxQParams *txq) {
    DOCA_RETURN_INVALID_VALUE_IF(
            txq == nullptr, "Invalid input parameter for TX queue destruction");

    RT_LOGC_DEBUG(Net::NetDoca, "Destroying DOCA Ethernet transmit queue");

    if (txq->eth_txq_ctx != nullptr) {
        DOCA_RETURN_BAD_STATE_ON_ERR(
                doca_ctx_stop(txq->eth_txq_ctx), "Failed to stop DOCA context");
    }

    if (txq->gpu_pkt_addr != nullptr) {
        DOCA_RETURN_BAD_STATE_ON_ERR(
                doca_gpu_mem_free(txq->gpu_dev, txq->gpu_pkt_addr), "Failed to free GPU memory");
    }

    if (txq->eth_txq_cpu != nullptr) {
        DOCA_RETURN_BAD_STATE_ON_ERR(
                doca_eth_txq_destroy(txq->eth_txq_cpu), "Failed to destroy DOCA Ethernet TX queue");
    }

    if (txq->eth_txq_pe != nullptr) {
        DOCA_RETURN_BAD_STATE_ON_ERR(
                doca_pe_destroy(txq->eth_txq_pe), "Failed to destroy progress engine");
    }

    if (txq->buf_arr != nullptr) {
        DOCA_RETURN_BAD_STATE_ON_ERR(
                doca_buf_arr_destroy(txq->buf_arr), "Failed to destroy buffer array");
    }

    if (txq->pkt_buff_mmap != nullptr) {
        DOCA_RETURN_BAD_STATE_ON_ERR(
                doca_mmap_destroy(txq->pkt_buff_mmap), "Failed to destroy memory mapping");
    }

    RT_LOGC_DEBUG(Net::NetDoca, "DOCA Ethernet transmit queue destroyed successfully");

    return DOCA_SUCCESS;
}

// NOLINTEND(cppcoreguidelines-macro-usage,cppcoreguidelines-avoid-do-while,clang-diagnostic-gnu-zero-variadic-macro-arguments)

} // namespace framework::net
