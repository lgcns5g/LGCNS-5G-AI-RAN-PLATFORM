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

#ifndef RAN_FRONTHAUL_ORDER_KERNEL_DESCRIPTORS_HPP
#define RAN_FRONTHAUL_ORDER_KERNEL_DESCRIPTORS_HPP

/**
 * @file order_kernel_descriptors.hpp
 * @brief Kernel descriptor structures for OrderKernelModule
 *
 * Defines static and dynamic descriptor structures based on OrderKernelConfigParamsT
 * from test_oran_order_kernel.cpp (lines 267-345)
 */

#include <array>
#include <chrono>
#include <cstdint>

// Forward declarations for DOCA types (avoid full DOCA header dependency)
struct doca_gpu_eth_rxq;
struct doca_gpu_semaphore_gpu;

namespace ran::fronthaul {

/**
 * @brief DOCA semaphore info structure for order kernel
 *
 * This structure is passed through DOCA GPUNetIO semaphores to communicate
 * packet metadata from the NIC to the GPU order kernel.
 */
struct DocaOrderSemInfo final {
    std::uint32_t pkts{}; //!< Number of packets received
};

/**
 * Convert duration to nanoseconds at compile-time
 *
 * @tparam Rep Arithmetic type representing the number of ticks
 * @tparam Period std::ratio representing the tick period
 * @param[in] duration Duration to convert
 * @return Duration value in nanoseconds
 */
template <typename Rep, typename Period>
constexpr std::uint64_t to_nanoseconds(std::chrono::duration<Rep, Period> duration) {
    return static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count());
}

/// Maximum number of cells that can be processed per slot
inline constexpr std::size_t ORDER_KERNEL_MAX_CELLS_PER_SLOT = 1;

// ============================================================================
// Network Configuration Constants
// ============================================================================

// Code reviewers - should these go into a separate file?

/// DPDK core for network operations (default: core 0 for receiver)
inline constexpr std::uint32_t DEFAULT_DPDK_CORE_ID = 0;

/// O-RAN eCPRI EtherType used by production O-RUs
inline constexpr std::uint16_t ORAN_ORU_ETHER_TYPE = 0xaefe;

// ============================================================================
// Default Timeout Configuration Constants
// ============================================================================
// These values are taken from test_oran_order_kernel.cpp (lines 479-484)
// and serve as sensible defaults for production use.

/// Default timeout when no packets received (6 milliseconds)
inline constexpr std::uint64_t DEFAULT_TIMEOUT_NO_PKT_NS =
        to_nanoseconds(std::chrono::milliseconds(6));

/// Default timeout for first packet reception (1500 microseconds)
inline constexpr std::uint64_t DEFAULT_TIMEOUT_FIRST_PKT_NS =
        to_nanoseconds(std::chrono::microseconds(1'500));

/// Default log interval for timeout messages (1 second)
inline constexpr std::uint64_t DEFAULT_TIMEOUT_LOG_INTERVAL_NS =
        to_nanoseconds(std::chrono::seconds(1));

/// Default timeout log enable flag (1=enabled, 0=disabled)
inline constexpr std::uint8_t DEFAULT_TIMEOUT_LOG_ENABLE = 1;

/// Default maximum packets to receive per kernel call
inline constexpr std::uint32_t DEFAULT_MAX_RX_PKTS = 512;

/// Default RX packet timeout between packets (100 microseconds)
inline constexpr std::uint64_t DEFAULT_RX_PKTS_TIMEOUT_NS =
        to_nanoseconds(std::chrono::microseconds(100));

/**
 * @brief Static kernel parameters (set once during setup)
 *
 * Parameters in this structure are initialized during module setup and remain
 * constant throughout the pipeline's lifetime. This includes DOCA objects,
 * cell configuration, buffer pointers, and GDRCopy device addresses.
 *
 * Based on OrderKernelConfigParamsT from test_oran_order_kernel.cpp
 */
struct OrderKernelStaticDescriptor final {
    // ========================================================================
    // DOCA Objects (pointers to GPU-accessible DOCA structures)
    // ========================================================================

    //! DOCA RX queue device pointers (from DOCA GPUNetIO)
    std::array<doca_gpu_eth_rxq *, ORDER_KERNEL_MAX_CELLS_PER_SLOT> rxq_info_gpu{};

    //! DOCA semaphore GPU objects for packet ordering
    std::array<doca_gpu_semaphore_gpu *, ORDER_KERNEL_MAX_CELLS_PER_SLOT> sem_gpu{};

    //! Semaphore item count (must be power of 2)
    std::array<std::uint32_t, ORDER_KERNEL_MAX_CELLS_PER_SLOT> sem_order_num{};

    // ========================================================================
    // Cell Configuration
    // ========================================================================

    //! Cell identifier (0 for single-cell)
    std::array<int, ORDER_KERNEL_MAX_CELLS_PER_SLOT> cell_id{};

    //! Compression method (1=BFP)
    std::array<int, ORDER_KERNEL_MAX_CELLS_PER_SLOT> comp_meth{};

    //! BFP bit width (14 for BFP14)
    std::array<int, ORDER_KERNEL_MAX_CELLS_PER_SLOT> bit_width{};

    //! RU type (2=FXCN O-RU specific handling)
    std::array<int, ORDER_KERNEL_MAX_CELLS_PER_SLOT> ru_type{};

    //! Beta scaling factor for BFP decompression (0.000244 for BFP14)
    std::array<float, ORDER_KERNEL_MAX_CELLS_PER_SLOT> beta{};

    // ========================================================================
    // Control (GDRCopy-backed device pointers)
    // ========================================================================

    //! PHY start signal (CPU → GPU via GDRCopy)
    std::array<std::uint32_t *, ORDER_KERNEL_MAX_CELLS_PER_SLOT> start_cuphy_d{};

    //! Kernel exit condition status (GPU → CPU via GDRCopy)
    std::array<std::uint32_t *, ORDER_KERNEL_MAX_CELLS_PER_SLOT> order_kernel_exit_cond_d{};

    //! Last RX semaphore index (device memory)
    std::array<std::uint32_t *, ORDER_KERNEL_MAX_CELLS_PER_SLOT> last_sem_idx_rx_h{};

    //! Last order semaphore index (device memory)
    std::array<std::uint32_t *, ORDER_KERNEL_MAX_CELLS_PER_SLOT> last_sem_idx_order_h{};

    // ========================================================================
    // PUSCH Configuration
    // ========================================================================

    //! Antenna port (eAxC ID) mapping (GDRCopy-backed)
    std::array<std::uint16_t *, ORDER_KERNEL_MAX_CELLS_PER_SLOT> pusch_e_ax_c_map{};

    //! Number of antenna ports (4 for 4x4 MIMO)
    std::array<std::uint32_t, ORDER_KERNEL_MAX_CELLS_PER_SLOT> pusch_e_ax_c_num{};

    //! PUSCH output buffer (device memory)
    std::array<std::uint8_t *, ORDER_KERNEL_MAX_CELLS_PER_SLOT> pusch_buffer{};

    //! PRBs per slot (273 for 100MHz bandwidth)
    std::array<std::uint32_t, ORDER_KERNEL_MAX_CELLS_PER_SLOT> pusch_prb_x_slot{};

    //! PRBs per symbol
    std::array<std::uint32_t, ORDER_KERNEL_MAX_CELLS_PER_SLOT> pusch_prb_x_symbol{};

    //! PRBs per symbol per antenna
    std::array<std::uint32_t, ORDER_KERNEL_MAX_CELLS_PER_SLOT> pusch_prb_x_symbol_x_antenna{};

    //! PRB stride in bytes (273 * 48 bytes for 100MHz BW)
    std::array<std::uint32_t, ORDER_KERNEL_MAX_CELLS_PER_SLOT> pusch_prb_stride{};

    //! Ordered PRB counter (GDRCopy-backed for CPU visibility)
    std::array<std::uint32_t *, ORDER_KERNEL_MAX_CELLS_PER_SLOT> pusch_ordered_prbs{};

    // ========================================================================
    // PRACH Configuration
    // ========================================================================

    //! PRACH antenna port mapping (GDRCopy-backed)
    std::array<std::uint16_t *, ORDER_KERNEL_MAX_CELLS_PER_SLOT> prach_e_ax_c_map{};

    //! Number of PRACH antenna ports
    std::array<std::uint32_t, ORDER_KERNEL_MAX_CELLS_PER_SLOT> prach_e_ax_c_num{};

    //! PRACH FDM occasion 0 buffer
    std::array<std::uint8_t *, ORDER_KERNEL_MAX_CELLS_PER_SLOT> prach_buffer_0{};

    //! PRACH FDM occasion 1 buffer
    std::array<std::uint8_t *, ORDER_KERNEL_MAX_CELLS_PER_SLOT> prach_buffer_1{};

    //! PRACH FDM occasion 2 buffer
    std::array<std::uint8_t *, ORDER_KERNEL_MAX_CELLS_PER_SLOT> prach_buffer_2{};

    //! PRACH FDM occasion 3 buffer
    std::array<std::uint8_t *, ORDER_KERNEL_MAX_CELLS_PER_SLOT> prach_buffer_3{};

    //! PRACH PRBs per slot
    std::array<std::uint32_t, ORDER_KERNEL_MAX_CELLS_PER_SLOT> prach_prb_x_slot{};

    //! PRACH PRBs per symbol
    std::array<std::uint32_t, ORDER_KERNEL_MAX_CELLS_PER_SLOT> prach_prb_x_symbol{};

    //! PRACH PRBs per symbol per antenna
    std::array<std::uint32_t, ORDER_KERNEL_MAX_CELLS_PER_SLOT> prach_prb_x_symbol_x_antenna{};

    //! PRACH PRB stride in bytes
    std::array<std::uint32_t, ORDER_KERNEL_MAX_CELLS_PER_SLOT> prach_prb_stride{};

    //! PRACH ordered PRB counter (GDRCopy-backed)
    std::array<std::uint32_t *, ORDER_KERNEL_MAX_CELLS_PER_SLOT> prach_ordered_prbs{};

    // ========================================================================
    // SRS Configuration
    // ========================================================================

    //! SRS antenna port mapping (GDRCopy-backed)
    std::array<std::uint16_t *, ORDER_KERNEL_MAX_CELLS_PER_SLOT> srs_e_ax_c_map{};

    //! Number of SRS antenna ports
    std::array<std::uint32_t, ORDER_KERNEL_MAX_CELLS_PER_SLOT> srs_e_ax_c_num{};

    //! SRS output buffer
    std::array<std::uint8_t *, ORDER_KERNEL_MAX_CELLS_PER_SLOT> srs_buffer{};

    //! SRS PRBs per slot
    std::array<std::uint32_t, ORDER_KERNEL_MAX_CELLS_PER_SLOT> srs_prb_x_slot{};

    //! SRS PRB stride in bytes
    std::array<std::uint32_t, ORDER_KERNEL_MAX_CELLS_PER_SLOT> srs_prb_stride{};

    //! SRS ordered PRB counter (GDRCopy-backed)
    std::array<std::uint32_t *, ORDER_KERNEL_MAX_CELLS_PER_SLOT> srs_ordered_prbs{};

    //! SRS start symbol index
    std::array<std::uint8_t, ORDER_KERNEL_MAX_CELLS_PER_SLOT> srs_start_sym{};

    // ========================================================================
    // Sub-slot Processing
    // ========================================================================

    //! Symbol ordering done signal array
    std::uint32_t *sym_ord_done_sig_arr{nullptr};

    //! Symbol ordering done mask array
    std::uint32_t *sym_ord_done_mask_arr{nullptr};

    //! PUSCH PRB-to-symbol mapping
    std::uint32_t *pusch_prb_symbol_map_d{nullptr};

    //! Number of order cells symbol mask array
    std::uint32_t *num_order_cells_sym_mask_arr{nullptr};

    // ========================================================================
    // Timing Windows (static configuration from system parameters)
    // ========================================================================

    //! Early packet threshold (Ta4_min in ORAN spec, nanoseconds)
    std::array<std::uint64_t, ORDER_KERNEL_MAX_CELLS_PER_SLOT> ta4_min_ns{};

    //! Late packet threshold (Ta4_max in ORAN spec, nanoseconds)
    std::array<std::uint64_t, ORDER_KERNEL_MAX_CELLS_PER_SLOT> ta4_max_ns{};

    //! Slot duration in nanoseconds (e.g., 500us for 30kHz SCS)
    std::array<std::uint64_t, ORDER_KERNEL_MAX_CELLS_PER_SLOT> slot_duration{};

    // ========================================================================
    // Timeout Configuration (static configuration from system parameters)
    // ========================================================================

    //! Timeout if no packets received (nanoseconds, default 6 seconds)
    std::uint64_t timeout_no_pkt_ns{DEFAULT_TIMEOUT_NO_PKT_NS};

    //! Timeout for first packet (nanoseconds, default 1500 microseconds)
    std::uint64_t timeout_first_pkt_ns{DEFAULT_TIMEOUT_FIRST_PKT_NS};

    //! Log interval for timeout messages (nanoseconds, default 1 second)
    std::uint64_t timeout_log_interval_ns{DEFAULT_TIMEOUT_LOG_INTERVAL_NS};

    //! Enable timeout logging (1=enable, 0=disable)
    std::uint8_t timeout_log_enable{DEFAULT_TIMEOUT_LOG_ENABLE};

    //! Maximum packets to receive per call (default 100)
    std::uint32_t max_rx_pkts{DEFAULT_MAX_RX_PKTS};

    //! RX packet timeout (nanoseconds, default 100 microseconds)
    std::uint64_t rx_pkts_timeout_ns{DEFAULT_RX_PKTS_TIMEOUT_NS};

    // ========================================================================
    // Packet Tracing (optional)
    // ========================================================================

    //! Packet timestamps per symbol
    std::array<std::uint64_t *, ORDER_KERNEL_MAX_CELLS_PER_SLOT> rx_packets_ts{};

    //! Packet counts per symbol
    std::array<std::uint32_t *, ORDER_KERNEL_MAX_CELLS_PER_SLOT> rx_packets_count{};

    //! Byte counts per symbol
    std::array<std::uint32_t *, ORDER_KERNEL_MAX_CELLS_PER_SLOT> rx_bytes_count{};

    //! Earliest packet timestamp
    std::array<std::uint64_t *, ORDER_KERNEL_MAX_CELLS_PER_SLOT> rx_packets_ts_earliest{};

    //! Latest packet timestamp
    std::array<std::uint64_t *, ORDER_KERNEL_MAX_CELLS_PER_SLOT> rx_packets_ts_latest{};

    //! Next slot packet timestamps
    std::array<std::uint64_t *, ORDER_KERNEL_MAX_CELLS_PER_SLOT> next_slot_rx_packets_ts{};

    //! Next slot packet counts
    std::array<std::uint32_t *, ORDER_KERNEL_MAX_CELLS_PER_SLOT> next_slot_rx_packets_count{};

    //! Next slot byte counts
    std::array<std::uint32_t *, ORDER_KERNEL_MAX_CELLS_PER_SLOT> next_slot_rx_bytes_count{};

    // ========================================================================
    // PCAP (Phase 2 - optional, placeholders)
    // ========================================================================

    //! PCAP capture buffer
    std::array<std::uint8_t *, ORDER_KERNEL_MAX_CELLS_PER_SLOT> pcap_buffer{};

    //! PCAP timestamp buffer
    std::array<std::uint8_t *, ORDER_KERNEL_MAX_CELLS_PER_SLOT> pcap_buffer_ts{};

    //! PCAP buffer index
    std::array<std::uint32_t *, ORDER_KERNEL_MAX_CELLS_PER_SLOT> pcap_buffer_index{};

    // ========================================================================
    // Other Control
    // ========================================================================

    //! Barrier flag for synchronization (Phase 2)
    int *barrier_flag{nullptr};

    //! Completion flag (Phase 2)
    std::array<std::uint8_t *, ORDER_KERNEL_MAX_CELLS_PER_SLOT> done_shared{};

    //! Next slot PRB count channel 1 (Phase 2)
    std::array<std::uint32_t *, ORDER_KERNEL_MAX_CELLS_PER_SLOT> next_slot_num_prb_ch1{};

    //! Next slot PRB count channel 2 (Phase 2)
    std::array<std::uint32_t *, ORDER_KERNEL_MAX_CELLS_PER_SLOT> next_slot_num_prb_ch2{};
};

/**
 * @brief Dynamic kernel parameters (can change per iteration)
 *
 * Parameters in this structure are updated every slot/subframe via configure_io().
 * This includes timing parameters, packet statistics, and timeout configuration.
 *
 * Strategy: When unsure if a parameter is static or dynamic → make it dynamic (safer).
 */
struct OrderKernelDynamicDescriptor final {
    // ========================================================================
    // ORAN Timing (changes every slot)
    // ========================================================================

    //! Current frame ID (0-255)
    std::uint8_t frame_id{0};

    //! Current subframe ID (0-9, 10 subframes per frame)
    std::uint8_t subframe_id{0};

    //! Current slot ID (varies by numerology, e.g., 0-1 for 30kHz SCS)
    std::uint8_t slot_id{0};

    // ========================================================================
    // Per-Slot Timing (updated every slot)
    // ========================================================================

    //! Slot start time in nanoseconds (system timestamp)
    std::array<std::uint64_t, ORDER_KERNEL_MAX_CELLS_PER_SLOT> slot_start{};

    // ========================================================================
    // Packet Statistics (GDRCopy-backed device pointers)
    // ========================================================================

    //! Early packet count (current slot, GPU writes via GDRCopy)
    std::array<std::uint32_t *, ORDER_KERNEL_MAX_CELLS_PER_SLOT> early_rx_packets{};

    //! On-time packet count (current slot)
    std::array<std::uint32_t *, ORDER_KERNEL_MAX_CELLS_PER_SLOT> on_time_rx_packets{};

    //! Late packet count (current slot)
    std::array<std::uint32_t *, ORDER_KERNEL_MAX_CELLS_PER_SLOT> late_rx_packets{};

    //! Early packet count (next slot)
    std::array<std::uint32_t *, ORDER_KERNEL_MAX_CELLS_PER_SLOT> next_slot_early_rx_packets{};

    //! On-time packet count (next slot)
    std::array<std::uint32_t *, ORDER_KERNEL_MAX_CELLS_PER_SLOT> next_slot_on_time_rx_packets{};

    //! Late packet count (next slot)
    std::array<std::uint32_t *, ORDER_KERNEL_MAX_CELLS_PER_SLOT> next_slot_late_rx_packets{};

    //! Dropped packet count (always used, even when tracing disabled)
    std::array<std::uint32_t *, ORDER_KERNEL_MAX_CELLS_PER_SLOT> rx_packets_dropped_count{};

    //! Last timeout error timestamp
    std::array<std::uint64_t *, ORDER_KERNEL_MAX_CELLS_PER_SLOT>
            order_kernel_last_timeout_error_time{};
};

} // namespace ran::fronthaul

#endif // RAN_FRONTHAUL_ORDER_KERNEL_DESCRIPTORS_HPP
