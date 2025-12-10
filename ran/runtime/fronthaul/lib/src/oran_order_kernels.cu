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

/**
 * @file oran_order_kernels.cu
 * @brief CUDA kernels for ORAN order processing
 */

#include <cstdint>
#include <cstdio>

#include <doca_buf.h>
#include <doca_buf_inventory.h>

#include <cuda_fp16.h>
#include <doca_gpunetio_dev_buf.cuh>
#include <doca_gpunetio_dev_eth_rxq.cuh>
#include <doca_gpunetio_dev_eth_txq.cuh>
#include <doca_gpunetio_dev_sem.cuh>

#include "aerial-fh-driver/oran.hpp"
#include "fronthaul/oran_order_kernels.hpp"
#include "fronthaul/order_kernel_descriptors.hpp"
#include "gpu_blockFP.h" //Compression Decompression repo
#include "gpu_fixed.h"   //Compression Decompression repo
#include "log/rt_log_macros.hpp"
#include "net/doca_types.hpp"
#include "net/net_log.hpp"

enum class UserDataCompressionMethod {
    NO_COMPRESSION = 0b0000,
    BLOCK_FLOATING_POINT = 0b0001,
    BLOCK_SCALING = 0b0010,
    U_LAW = 0b0011,
    MODULATION_COMPRESSION = 0b0100,
    BFP_SELECTIVE_RE_SENDING = 0b0101,
    MOD_COMPR_SELECTIVE_RE_SENDING = 0b0110,
    RESERVED = 0b0111,
};
#define ORDER_KERNEL_PUSCH_ONLY (0)
#define ORDER_KERNEL_SRS_ENABLE (1)
#define ORDER_KERNEL_PUSCH (1)
#define ORDER_KERNEL_SRS (2)
#define ORDER_KERNEL_SRS_AND_PUSCH (ORDER_KERNEL_SRS | ORDER_KERNEL_PUSCH)
#define PRB_SIZE_16F PRB_SIZE(16)

// The ping-pong order kernels use a single CTA to both receive and process the packets,
// alternating between reading and processing in a ping-pong fashion.
#define ORDER_KERNEL_PINGPONG_NUM_THREADS (320)
#define ORDER_KERNEL_PINGPONG_SRS_NUM_THREADS (1024)
[[maybe_unused]] static constexpr uint32_t ORAN_PRACH_B4_SYMBOLS_X_SLOT = 12;
// static constexpr uint32_t ORAN_SRS_SYMBOLS_X_SLOT = 14;
[[maybe_unused]] static constexpr uint32_t ORAN_SRS_PRBS_X_PORT_X_SYMBOL = 272;
static constexpr uint32_t ORAN_PUSCH_SYMBOLS_X_SLOT = 14;
static constexpr uint32_t ORAN_MAX_SYMBOLS = 14;
[[maybe_unused]] static constexpr uint32_t ORAN_MAX_SRS_SYMBOLS = 6;

/**
 * @brief Packet tracing information structure
 *
 * Used for collecting packet timing and statistics data during kernel execution.
 */
struct order_kernel_pkt_tracing_info final {
    uint32_t **rx_packets_count{};
    uint32_t **rx_bytes_count{};
    uint32_t **next_slot_rx_packets_count{};
    uint32_t **next_slot_rx_bytes_count{};
    uint64_t **rx_packets_ts_earliest{};
    uint64_t **rx_packets_ts_latest{};
    uint64_t **rx_packets_ts{};
    uint64_t **next_slot_rx_packets_ts{};
};

/**
 * PUSCH Symbol RX state
 */
typedef enum _cuphySymbolRxState {
    SYM_RX_NOT_DONE = 0, /// OFDM symbol to be received
    SYM_RX_DONE = 1,     /// OFDM symbol received succesfully
    SYM_RX_TIMEOUT = 2,  /// OFDM symbol reception timed out
    SYM_RX_ERROR = 3,    /// Error during OFDM symbol reception
    SYM_RX_MAX
} cuphySymbolRxState_t;

static constexpr uint32_t ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM =
        100; // Setting to 100 as a safety net value considering we receive 64 SRS packets per
             // symbol with 64TR config

enum order_kernel_exit_code {
    ORDER_KERNEL_RUNNING = 0,
    ORDER_KERNEL_EXIT_PRB = 1,
    ORDER_KERNEL_EXIT_ERROR_LEGACY = 2,
    ORDER_KERNEL_EXIT_TIMEOUT_RX_PKT = 3,
    ORDER_KERNEL_EXIT_TIMEOUT_NO_PKT = 4,
    ORDER_KERNEL_EXIT_ERROR1 = 5,
    ORDER_KERNEL_EXIT_ERROR2 = 6,
    ORDER_KERNEL_EXIT_ERROR3 = 7,
    ORDER_KERNEL_EXIT_ERROR4 = 8,
    ORDER_KERNEL_EXIT_ERROR5 = 9,
    ORDER_KERNEL_EXIT_ERROR6 = 10,
    ORDER_KERNEL_EXIT_ERROR7 = 11,
};

static constexpr uint32_t ORAN_FRAME_WRAP = 256;
static constexpr uint32_t ORAN_SUBFRAME_WRAP = 10;
static constexpr uint32_t ORAN_SLOT_WRAP = 2;
static constexpr uint32_t ORAN_SLOTS_PER_FRAME = ORAN_SUBFRAME_WRAP * ORAN_SLOT_WRAP;

namespace {

// Determines the number of slots between the specified ORAN details (ORAN details 2 - ORAN details
// 1) Result is in the range of [-2560, 2560] Positive means ORAN details 2 is later in time than
// ORAN details 1 Negative means ORAN details 2 is earlier in time than ORAN details 1
inline __device__ int32_t calculate_slot_difference(
        uint8_t frameId1,
        uint8_t frameId2,
        uint8_t subframeId1,
        uint8_t subframeId2,
        uint8_t slotId1,
        uint8_t slotId2) {
    // Calculate frame difference accounting for wrap-around
    // Using (ORAN_FRAME_WRAP/2) ensures we get the shortest path around the wrap
    int32_t frame_diff = ((frameId2 - frameId1 + (ORAN_FRAME_WRAP / 2)) % ORAN_FRAME_WRAP) -
                         (ORAN_FRAME_WRAP / 2);

    // Calculate absolute slot positions within their respective frames
    int32_t slot_count1 = (ORAN_SLOT_WRAP * subframeId1 + slotId1);
    int32_t slot_count2 = (ORAN_SLOT_WRAP * subframeId2 + slotId2);

    // Calculate slot difference based on frame difference
    int32_t slot_diff = slot_count2 - slot_count1;

    // Combine frame and slot differences
    return frame_diff * ORAN_SLOTS_PER_FRAME + slot_diff;
}

__device__ __forceinline__ uint16_t
get_eaxc_index(uint16_t *eAxC_map, int eAxC_num, uint16_t eAxC_id) {
    for (int i = 0; i < eAxC_num; i++) {
        if (eAxC_map[i] == eAxC_id)
            return i;
    }

    return 0;
}

__device__ __forceinline__ unsigned long long __globaltimer() {
    unsigned long long globaltimer;
    // 64-bit GPU global nanosecond timer
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(globaltimer));
    return globaltimer;
}

} // namespace

// ============================================================================
// Unified kernel implementation with descriptor-based interface
// ============================================================================
// Note: This kernel is placed in the ran::fronthaul namespace (not anonymous)
// so it can be linked from order_kernel_module.cpp

namespace ran::fronthaul {

/**
 * @brief Unified order kernel with descriptor-based interface
 *
 * This kernel receives ORAN packets, orders them by timing, decompresses BFP data,
 * and writes PRBs to PUSCH/PRACH/SRS buffers. Takes static and dynamic descriptors
 * as parameters instead of dozens of individual parameters.
 *
 * Define launch bounds to support 2 CTAs per SM, even though we typically allocate one SM
 * per CTA. This is intended to minimize the possibility of the order kernel being SM-starved.
 * We would typically prefer to run two CTAs on a single SM rather than delay the launch of
 * one or more CTAs.
 *
 * @tparam ok_tb_enable Test bench mode enable flag
 * @tparam ul_rx_pkt_tracing_level Packet tracing verbosity level
 * @tparam srs_enable SRS processing mode (0=PUSCH only, 1=SRS only, 3=PUSCH+SRS)
 * @tparam NUM_THREADS Number of threads per CTA
 * @tparam NUM_CTAS_PER_SM Number of CTAs per SM for launch bounds
 * @param[in] static_desc Static descriptor with configuration that doesn't change per iteration
 * @param[in] dynamic_desc Dynamic descriptor with per-iteration parameters (frame/slot IDs, timing,
 * stats)
 */
template <
        bool ok_tb_enable,
        uint8_t ul_rx_pkt_tracing_level,
        uint8_t srs_enable,
        int NUM_THREADS,
        int NUM_CTAS_PER_SM>
__global__ void __launch_bounds__(NUM_THREADS, NUM_CTAS_PER_SM)
        order_kernel_doca_single_subSlot_pingpong(
                const ran::fronthaul::OrderKernelStaticDescriptor *static_desc,
                const ran::fronthaul::OrderKernelDynamicDescriptor *dynamic_desc) {
    // Unpack static descriptor fields
    struct doca_gpu_eth_rxq **doca_rxq =
            const_cast<doca_gpu_eth_rxq **>(static_desc->rxq_info_gpu.data());
    struct doca_gpu_semaphore_gpu **sem_gpu =
            const_cast<doca_gpu_semaphore_gpu **>(static_desc->sem_gpu.data());
    const uint16_t *sem_order_num =
            reinterpret_cast<const uint16_t *>(static_desc->sem_order_num.data());

    const int *cell_id = static_desc->cell_id.data();
    const int *ru_type = static_desc->ru_type.data();

    uint32_t **start_cuphy_d = const_cast<uint32_t **>(static_desc->start_cuphy_d.data());
    uint32_t **exit_cond_d = const_cast<uint32_t **>(static_desc->order_kernel_exit_cond_d.data());
    uint32_t **last_sem_idx_rx_h = const_cast<uint32_t **>(static_desc->last_sem_idx_rx_h.data());
    uint32_t **last_sem_idx_order_h =
            const_cast<uint32_t **>(static_desc->last_sem_idx_order_h.data());
    const int *comp_meth = static_desc->comp_meth.data();
    const int *bit_width = static_desc->bit_width.data();
    const float *beta = static_desc->beta.data();
    constexpr int prb_size = PRB_SIZE_16F;

    const uint32_t timeout_no_pkt_ns = static_desc->timeout_no_pkt_ns;
    const uint32_t timeout_first_pkt_ns = static_desc->timeout_first_pkt_ns;
    const uint32_t timeout_log_interval_ns = static_desc->timeout_log_interval_ns;
    const uint8_t timeout_log_enable = static_desc->timeout_log_enable;
    const uint32_t max_rx_pkts = static_desc->max_rx_pkts;
    const uint32_t rx_pkts_timeout_ns = static_desc->rx_pkts_timeout_ns;

    // Unpack dynamic descriptor fields
    const uint8_t frameId = dynamic_desc->frame_id;
    const uint8_t subframeId = dynamic_desc->subframe_id;
    const uint8_t slotId = dynamic_desc->slot_id;

    uint32_t **early_rx_packets = const_cast<uint32_t **>(dynamic_desc->early_rx_packets.data());
    uint32_t **on_time_rx_packets =
            const_cast<uint32_t **>(dynamic_desc->on_time_rx_packets.data());
    uint32_t **late_rx_packets = const_cast<uint32_t **>(dynamic_desc->late_rx_packets.data());
    uint32_t **next_slot_early_rx_packets =
            const_cast<uint32_t **>(dynamic_desc->next_slot_early_rx_packets.data());
    uint32_t **next_slot_on_time_rx_packets =
            const_cast<uint32_t **>(dynamic_desc->next_slot_on_time_rx_packets.data());
    uint32_t **next_slot_late_rx_packets =
            const_cast<uint32_t **>(dynamic_desc->next_slot_late_rx_packets.data());
    uint64_t *slot_start = const_cast<uint64_t *>(dynamic_desc->slot_start.data());
    uint64_t *ta4_min_ns = const_cast<uint64_t *>(static_desc->ta4_min_ns.data());
    uint64_t *ta4_max_ns = const_cast<uint64_t *>(static_desc->ta4_max_ns.data());
    uint64_t *slot_duration = const_cast<uint64_t *>(static_desc->slot_duration.data());
    uint64_t **order_kernel_last_timeout_error_time =
            const_cast<uint64_t **>(dynamic_desc->order_kernel_last_timeout_error_time.data());

    // Packet tracing info (disabled)
    order_kernel_pkt_tracing_info pkt_tracing_info{};
    pkt_tracing_info.rx_packets_ts = nullptr;
    pkt_tracing_info.rx_packets_count = nullptr;
    pkt_tracing_info.rx_bytes_count = nullptr;
    pkt_tracing_info.rx_packets_ts_earliest = nullptr;
    pkt_tracing_info.rx_packets_ts_latest = nullptr;
    pkt_tracing_info.next_slot_rx_packets_ts = nullptr;
    pkt_tracing_info.next_slot_rx_packets_count = nullptr;
    pkt_tracing_info.next_slot_rx_bytes_count = nullptr;

    uint32_t **rx_packets_dropped_count =
            const_cast<uint32_t **>(dynamic_desc->rx_packets_dropped_count.data());

    // Sub-slot processing
    uint32_t *sym_ord_done_sig_arr = static_desc->sym_ord_done_sig_arr;
    uint32_t *sym_ord_done_mask_arr = static_desc->sym_ord_done_mask_arr;
    uint32_t *pusch_prb_symbol_map = static_desc->pusch_prb_symbol_map_d;
    uint32_t *num_order_cells_sym_mask_arr = static_desc->num_order_cells_sym_mask_arr;
    constexpr uint8_t pusch_prb_non_zero = 1;

    // PUSCH
    uint16_t **pusch_eAxC_map = const_cast<uint16_t **>(static_desc->pusch_e_ax_c_map.data());
    uint32_t *pusch_eAxC_num = const_cast<uint32_t *>(static_desc->pusch_e_ax_c_num.data());
    uint8_t **pusch_buffer = const_cast<uint8_t **>(static_desc->pusch_buffer.data());
    uint32_t *pusch_prb_x_slot = const_cast<uint32_t *>(static_desc->pusch_prb_x_slot.data());
    constexpr int pusch_symbols_x_slot = 14;
    uint32_t *pusch_prb_x_port_x_symbol =
            const_cast<uint32_t *>(static_desc->pusch_prb_x_symbol_x_antenna.data());
    uint32_t **pusch_ordered_prbs = const_cast<uint32_t **>(static_desc->pusch_ordered_prbs.data());

    // PRACH
    uint16_t **prach_eAxC_map = const_cast<uint16_t **>(static_desc->prach_e_ax_c_map.data());
    uint32_t *prach_eAxC_num = const_cast<uint32_t *>(static_desc->prach_e_ax_c_num.data());
    uint8_t **prach_buffer_0 = const_cast<uint8_t **>(static_desc->prach_buffer_0.data());
    uint8_t **prach_buffer_1 = const_cast<uint8_t **>(static_desc->prach_buffer_1.data());
    uint8_t **prach_buffer_2 = const_cast<uint8_t **>(static_desc->prach_buffer_2.data());
    uint8_t **prach_buffer_3 = const_cast<uint8_t **>(static_desc->prach_buffer_3.data());
    constexpr uint16_t prach_section_id_0 = 2048;
    constexpr uint16_t prach_section_id_1 = 2049;
    constexpr uint16_t prach_section_id_2 = 2050;
    constexpr uint16_t prach_section_id_3 = 2051;
    uint32_t *prach_prb_x_slot = const_cast<uint32_t *>(static_desc->prach_prb_x_slot.data());
    constexpr int prach_symbols_x_slot = 12;
    uint32_t *prach_prb_x_port_x_symbol =
            const_cast<uint32_t *>(static_desc->prach_prb_x_symbol_x_antenna.data());
    uint32_t **prach_ordered_prbs = const_cast<uint32_t **>(static_desc->prach_ordered_prbs.data());

    // SRS
    uint16_t **srs_eAxC_map = const_cast<uint16_t **>(static_desc->srs_e_ax_c_map.data());
    uint32_t *srs_eAxC_num = const_cast<uint32_t *>(static_desc->srs_e_ax_c_num.data());
    uint8_t **srs_buffer = const_cast<uint8_t **>(static_desc->srs_buffer.data());
    uint32_t *srs_prb_x_slot = const_cast<uint32_t *>(static_desc->srs_prb_x_slot.data());
    constexpr int srs_symbols_x_slot = 14;
    uint32_t *srs_prb_x_port_x_symbol = const_cast<uint32_t *>(static_desc->srs_prb_stride.data());
    uint32_t **srs_ordered_prbs = const_cast<uint32_t **>(static_desc->srs_ordered_prbs.data());
    uint8_t *srs_start_sym = const_cast<uint8_t *>(static_desc->srs_start_sym.data());

    [[maybe_unused]] constexpr uint8_t num_order_cells = 1;

    // Test bench (disabled)
    uint8_t **tb_fh_buf = nullptr;
    constexpr uint32_t max_pkt_size = 0;
    uint32_t *rx_pkt_num_slot = nullptr;

    // ========================================================================
    // Kernel implementation begins here
    // ========================================================================
    int cell_idx = blockIdx.x;
    uint32_t cell_idx_mask = (0x1 << cell_idx);

    const unsigned long long kernel_start = __globaltimer();
    int prb_x_slot;
    // Restart from last semaphore item
    int sem_idx_rx = (int)(*(*(last_sem_idx_rx_h + cell_idx)));
    int sem_idx_order = (int)(*(*(last_sem_idx_order_h + cell_idx)));
    int last_sem_idx_order = (int)(*(*(last_sem_idx_order_h + cell_idx)));
    if constexpr (srs_enable == ORDER_KERNEL_SRS_ENABLE) {
        prb_x_slot = srs_prb_x_slot[cell_idx];
    } else if constexpr (srs_enable == ORDER_KERNEL_PUSCH_ONLY) {
        prb_x_slot = pusch_prb_x_slot[cell_idx] + prach_prb_x_slot[cell_idx];
    } else {
        prb_x_slot =
                pusch_prb_x_slot[cell_idx] + prach_prb_x_slot[cell_idx] + srs_prb_x_slot[cell_idx];
    }

    __shared__ uint32_t rx_pkt_num;
    __shared__ uint64_t rx_buf_idx;
    __shared__ uint32_t done_shared_sh;
    __shared__ uint32_t early_rx_packets_count_sh;
    __shared__ uint32_t on_time_rx_packets_count_sh;
    __shared__ uint32_t late_rx_packets_count_sh;
    __shared__ uint32_t next_slot_early_rx_packets_count_sh;
    __shared__ uint32_t next_slot_on_time_rx_packets_count_sh;
    __shared__ uint32_t next_slot_late_rx_packets_count_sh;
    __shared__ uint32_t pusch_prb_symbol_ordered[ORAN_PUSCH_SYMBOLS_X_SLOT];
    __shared__ uint32_t pusch_prb_symbol_ordered_done[ORAN_PUSCH_SYMBOLS_X_SLOT];
    __shared__ uint32_t rx_packets_dropped_count_sh;
    // These shared memory arrays are only referenced when template parameter
    // ul_rx_pkt_tracing_level is non-zero. When ul_rx_pkt_tracing_level is 0, the compiler will not
    // allocate this static shared memory.
    __shared__ uint32_t rx_packets_count_sh[ORAN_MAX_SYMBOLS];
    __shared__ uint32_t rx_bytes_count_sh[ORAN_MAX_SYMBOLS];
    __shared__ uint32_t next_slot_rx_packets_count_sh[ORAN_MAX_SYMBOLS];
    __shared__ uint32_t next_slot_rx_bytes_count_sh[ORAN_MAX_SYMBOLS];
    __shared__ uint64_t rx_packets_ts_earliest_sh[ORAN_MAX_SYMBOLS];
    __shared__ uint64_t rx_packets_ts_latest_sh[ORAN_MAX_SYMBOLS];
    __shared__ uint64_t rx_packets_ts_sh[ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM * ORAN_MAX_SYMBOLS];
    __shared__ uint64_t
            next_slot_rx_packets_ts_sh[ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM * ORAN_MAX_SYMBOLS];

    // Cell specific (de-reference from host pinned memory once)
    uint32_t *pusch_ordered_prbs_cell = *(pusch_ordered_prbs + cell_idx);
    uint32_t *prach_ordered_prbs_cell = *(prach_ordered_prbs + cell_idx);
    uint32_t *srs_ordered_prbs_cell = *(srs_ordered_prbs + cell_idx);
    uint32_t *exit_cond_d_cell = *(exit_cond_d + cell_idx);
    uint32_t *last_sem_idx_rx_h_cell = *(last_sem_idx_rx_h + cell_idx);
    uint32_t *last_sem_idx_order_h_cell = *(last_sem_idx_order_h + cell_idx);
    struct doca_gpu_eth_rxq *doca_rxq_cell = *(doca_rxq + cell_idx);
    struct doca_gpu_semaphore_gpu *sem_gpu_cell = *(sem_gpu + cell_idx);
    int pusch_prb_x_slot_cell = pusch_prb_x_slot[cell_idx];
    int prach_prb_x_slot_cell = prach_prb_x_slot[cell_idx];

    uint64_t slot_start_cell = slot_start[cell_idx];
    uint64_t ta4_min_ns_cell = ta4_min_ns[cell_idx];
    uint64_t ta4_max_ns_cell = ta4_max_ns[cell_idx];
    uint64_t slot_duration_cell = slot_duration[cell_idx];
    uint32_t *early_rx_packets_cell = *(early_rx_packets + cell_idx);
    uint32_t *on_time_rx_packets_cell = *(on_time_rx_packets + cell_idx);
    uint32_t *late_rx_packets_cell = *(late_rx_packets + cell_idx);
    uint32_t *next_slot_early_rx_packets_cell = *(next_slot_early_rx_packets + cell_idx);
    uint32_t *next_slot_on_time_rx_packets_cell = *(next_slot_on_time_rx_packets + cell_idx);
    uint32_t *next_slot_late_rx_packets_cell = *(next_slot_late_rx_packets + cell_idx);
    uint32_t *rx_packets_dropped_count_cell = *(rx_packets_dropped_count + cell_idx);

    uint8_t *pusch_buffer_cell = *(pusch_buffer + cell_idx);
    uint16_t *pusch_eAxC_map_cell = *(pusch_eAxC_map + cell_idx);
    int pusch_eAxC_num_cell = pusch_eAxC_num[cell_idx];
    uint32_t pusch_prb_x_port_x_symbol_cell = pusch_prb_x_port_x_symbol[cell_idx];
    uint8_t *prach_buffer_0_cell = *(prach_buffer_0 + cell_idx);
    uint8_t *prach_buffer_1_cell = *(prach_buffer_1 + cell_idx);
    uint8_t *prach_buffer_2_cell = *(prach_buffer_2 + cell_idx);
    uint8_t *prach_buffer_3_cell = *(prach_buffer_3 + cell_idx);
    uint16_t *prach_eAxC_map_cell = *(prach_eAxC_map + cell_idx);
    int prach_eAxC_num_cell = prach_eAxC_num[cell_idx];
    uint32_t prach_prb_x_port_x_symbol_cell = prach_prb_x_port_x_symbol[cell_idx];

    uint8_t *srs_buffer_cell = *(srs_buffer + cell_idx);
    uint16_t *srs_eAxC_map_cell = *(srs_eAxC_map + cell_idx);
    int srs_eAxC_num_cell = srs_eAxC_num[cell_idx];
    uint32_t srs_prb_x_port_x_symbol_cell = srs_prb_x_port_x_symbol[cell_idx];
    uint8_t srs_start_sym_cell = srs_start_sym[cell_idx];
    int srs_prb_x_slot_cell = srs_prb_x_slot[cell_idx];

    const int ru_type_cell = ru_type[cell_idx];
    const int comp_meth_cell = comp_meth[cell_idx];
    const int bit_width_cell = bit_width[cell_idx];
    const float beta_cell = beta[cell_idx];
    const uint16_t sem_order_num_cell = sem_order_num[cell_idx];

    uint32_t *pusch_prb_symbol_map_cell =
            pusch_prb_symbol_map + (ORAN_PUSCH_SYMBOLS_X_SLOT * cell_idx);

    const uint32_t tid = threadIdx.x;
    const uint32_t laneId = threadIdx.x % 32;
    [[maybe_unused]] const uint32_t nwarps = blockDim.x / 32;

    if (tid == 0) {
        if constexpr (srs_enable == ORDER_KERNEL_SRS_ENABLE) {
            DOCA_GPUNETIO_VOLATILE(srs_ordered_prbs_cell[0]) = 0;
        } else if constexpr (srs_enable == ORDER_KERNEL_PUSCH_ONLY) {
            DOCA_GPUNETIO_VOLATILE(pusch_ordered_prbs_cell[0]) = 0;
            DOCA_GPUNETIO_VOLATILE(prach_ordered_prbs_cell[0]) = 0;
            DOCA_GPUNETIO_VOLATILE(pusch_prb_symbol_ordered[tid]) = 0;
            DOCA_GPUNETIO_VOLATILE(pusch_prb_symbol_ordered_done[tid]) = 0;
        } else {
            DOCA_GPUNETIO_VOLATILE(srs_ordered_prbs_cell[0]) = 0;
            DOCA_GPUNETIO_VOLATILE(pusch_ordered_prbs_cell[0]) = 0;
            DOCA_GPUNETIO_VOLATILE(prach_ordered_prbs_cell[0]) = 0;
            DOCA_GPUNETIO_VOLATILE(pusch_prb_symbol_ordered[tid]) = 0;
            DOCA_GPUNETIO_VOLATILE(pusch_prb_symbol_ordered_done[tid]) = 0;
        }
        DOCA_GPUNETIO_VOLATILE(done_shared_sh) = 1;

        early_rx_packets_count_sh = DOCA_GPUNETIO_VOLATILE(*next_slot_early_rx_packets_cell);
        on_time_rx_packets_count_sh = DOCA_GPUNETIO_VOLATILE(*next_slot_on_time_rx_packets_cell);
        late_rx_packets_count_sh = DOCA_GPUNETIO_VOLATILE(*next_slot_late_rx_packets_cell);
        next_slot_early_rx_packets_count_sh = 0;
        next_slot_late_rx_packets_count_sh = 0;
        next_slot_on_time_rx_packets_count_sh = 0;
        rx_packets_dropped_count_sh = 0;
    } else if (tid < ORAN_PUSCH_SYMBOLS_X_SLOT) {
        if constexpr (
                srs_enable == ORDER_KERNEL_PUSCH_ONLY || srs_enable == ORDER_KERNEL_SRS_AND_PUSCH) {
            DOCA_GPUNETIO_VOLATILE(pusch_prb_symbol_ordered[tid]) = 0;
            DOCA_GPUNETIO_VOLATILE(pusch_prb_symbol_ordered_done[tid]) = 0;
        }
    }

    __syncthreads();

    uint64_t *order_kernel_last_timeout_error_time_cell =
            order_kernel_last_timeout_error_time[cell_idx];
    const uint16_t compressed_prb_size = (bit_width_cell == BFP_NO_COMPRESSION) ? PRB_SIZE_16F
                                         : (bit_width_cell == BFP_COMPRESSION_14_BITS)
                                                 ? PRB_SIZE_14F
                                                 : PRB_SIZE_9F;
    uint8_t first_packet_received = 0;
    unsigned long long first_packet_received_time = 0;

    uint32_t early_rx_packets_count = 0;
    uint32_t late_rx_packets_count = 0;
    uint32_t on_time_rx_packets_count = 0;
    uint32_t next_slot_early_rx_packets_count = 0;
    uint32_t next_slot_late_rx_packets_count = 0;
    uint32_t next_slot_on_time_rx_packets_count = 0;
    uint32_t packets_dropped_count = 0;

    // Only referenced by tid 0
    uint32_t rx_pkt_num_total{0};

    __shared__ bool sh_have_data_to_process;
    __shared__ uint32_t sh_next_pkt_ind;

    [[maybe_unused]] static constexpr int WARP_SIZE = 32;
    uint32_t warp_pusch_ordered_prbs_cell = 0;
    uint32_t warp_prach_ordered_prbs_cell = 0;
    uint32_t warp_srs_ordered_prbs_cell = 0;
    __shared__ uint32_t smem_pusch_prach_ordered_prbs_cell;
    __shared__ uint32_t smem_srs_ordered_prbs_cell;
    if (tid == 0) {
        if constexpr (srs_enable == ORDER_KERNEL_SRS_ENABLE) {
            smem_srs_ordered_prbs_cell = *srs_ordered_prbs_cell;
        } else if constexpr (srs_enable == ORDER_KERNEL_PUSCH_ONLY) {
            smem_pusch_prach_ordered_prbs_cell =
                    *pusch_ordered_prbs_cell + *prach_ordered_prbs_cell;
        } else {
            smem_srs_ordered_prbs_cell = *srs_ordered_prbs_cell;
            smem_pusch_prach_ordered_prbs_cell =
                    *pusch_ordered_prbs_cell + *prach_ordered_prbs_cell;
        }
    }
    __syncthreads();

    if (ul_rx_pkt_tracing_level) {
        if (tid < ORAN_MAX_SYMBOLS) {
            rx_packets_count_sh[tid] = DOCA_GPUNETIO_VOLATILE(
                    pkt_tracing_info.next_slot_rx_packets_count[cell_idx][tid]);
            rx_bytes_count_sh[tid] = DOCA_GPUNETIO_VOLATILE(
                    pkt_tracing_info.next_slot_rx_bytes_count[cell_idx][tid]);
            rx_packets_ts_earliest_sh[tid] = 0xFFFFFFFFFFFFFFFFLLU;
            rx_packets_ts_latest_sh[tid] = 0;
            next_slot_rx_packets_count_sh[tid] = 0;
            next_slot_rx_bytes_count_sh[tid] = 0;
        }
        __syncthreads();
        const int max_pkt_idx = ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM * ORAN_MAX_SYMBOLS;
        for (uint32_t pkt_idx = tid; pkt_idx < max_pkt_idx; pkt_idx += blockDim.x) {
            uint32_t symbol_idx = pkt_idx / ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM;
            uint64_t *next_slot_rx_packets_ts_cell =
                    *(pkt_tracing_info.next_slot_rx_packets_ts + cell_idx);
            rx_packets_ts_sh[pkt_idx] =
                    DOCA_GPUNETIO_VOLATILE(next_slot_rx_packets_ts_cell[pkt_idx]);
            __threadfence_block();
            if (rx_packets_ts_sh[pkt_idx] != 0)
                atomicMin(
                        (unsigned long long *)&rx_packets_ts_earliest_sh[symbol_idx],
                        (unsigned long long)rx_packets_ts_sh[pkt_idx]);
            atomicMax(
                    (unsigned long long *)&rx_packets_ts_latest_sh[symbol_idx],
                    (unsigned long long)rx_packets_ts_sh[pkt_idx]);
            __threadfence_block();
            next_slot_rx_packets_ts_sh[pkt_idx] = 0;
        }
        __syncthreads();
        if (tid < ORAN_MAX_SYMBOLS) {
            if (rx_packets_ts_earliest_sh[tid] == 0)
                rx_packets_ts_earliest_sh[tid] = 0xFFFFFFFFFFFFFFFFLLU;
        }
    }

    // For the packets that are already available in semaphores before the first receive
    // call, we do not want to record packet stats because they were recoreded when the
    // packets were initially read.
    bool record_packet_stats = false;

    // Breaks when *exit_cond_d_cell != ORDER_KERNEL_RUNNING
    while (1) {

        // Breaks when no more packet buffers are ready to process. This loop may process buffers
        // stored in semaphores from a previous order kernel invocation.
        while (1) {
            uint32_t warp_pusch_prach_ordered_prbs_cell_this_burst = 0;
            uint32_t warp_srs_ordered_prbs_cell_this_burst = 0;

            if (tid == 0) {
                const doca_error_t ret = doca_gpu_dev_semaphore_get_packet_info_status(
                        sem_gpu_cell,
                        sem_idx_order,
                        DOCA_GPU_SEMAPHORE_STATUS_READY,
                        &rx_pkt_num,
                        &rx_buf_idx);
                if constexpr (ok_tb_enable) {
                    rx_pkt_num = rx_pkt_num_slot[cell_idx];
                }
                sh_have_data_to_process =
                        (ret != DOCA_ERROR_NOT_FOUND) && (rx_pkt_num > 0) &&
                        DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell) == ORDER_KERNEL_RUNNING;
                sh_next_pkt_ind = 0;
            }

            __syncthreads();

            if (!sh_have_data_to_process) {
                break;
            }

            // Breaks when no more packets are ready for processing in the current buffer.
            // Each warp handles one packet at a time, with packets dynamically scheduled to
            // warps using atomicAdd().
            while (1) {
                uint32_t pkt_idx;
                if (laneId == 0) {
                    pkt_idx = atomicAdd(&sh_next_pkt_ind, 1);
                }
                pkt_idx = __shfl_sync(0xffffffff, pkt_idx, 0, 32);

                if (pkt_idx >= rx_pkt_num) {
                    break;
                }
                uint8_t *pkt_thread = NULL;
                uint64_t rx_timestamp{0};
                if constexpr (ok_tb_enable) {
                    uint8_t *tb_fh_buf_cell = *(tb_fh_buf + cell_idx);
                    pkt_thread = (uint8_t *)(tb_fh_buf_cell + ((pkt_idx)*max_pkt_size));
                } else {
                    struct doca_gpu_buf *buf_ptr;
                    doca_gpu_dev_eth_rxq_get_buf(doca_rxq_cell, rx_buf_idx + pkt_idx, &buf_ptr);
                    uintptr_t rx_pkt_addr;
                    doca_gpu_dev_buf_get_addr(buf_ptr, &rx_pkt_addr);
                    doca_gpu_dev_eth_rxq_get_buf_timestamp(
                            doca_rxq_cell, rx_buf_idx + pkt_idx, &rx_timestamp);
                    pkt_thread = (uint8_t *)rx_pkt_addr;
                }

                uint8_t frameId_pkt = oran_umsg_get_frame_id(pkt_thread);
                uint8_t subframeId_pkt = oran_umsg_get_subframe_id(pkt_thread);
                uint8_t slotId_pkt = oran_umsg_get_slot_id(pkt_thread);
                uint8_t symbol_id_pkt = oran_umsg_get_symbol_id(pkt_thread);
                const uint8_t seq_id_pkt = oran_get_sequence_id(pkt_thread);
                const uint16_t ecpri_payload_length = oran_umsg_get_ecpri_payload(pkt_thread);

                // Calculate slot difference to determine if packet belongs to current slot
                int32_t full_slot_diff = calculate_slot_difference(
                        frameId, frameId_pkt, subframeId, subframeId_pkt, slotId, slotId_pkt);

                if (full_slot_diff > 0) {
                    if (record_packet_stats && laneId == 0) {
                        const uint64_t packet_early_thres =
                                slot_start_cell + slot_duration_cell + ta4_min_ns_cell +
                                (slot_duration_cell * symbol_id_pkt / ORAN_MAX_SYMBOLS);
                        const uint64_t packet_late_thres =
                                slot_start_cell + slot_duration_cell + ta4_max_ns_cell +
                                (slot_duration_cell * symbol_id_pkt / ORAN_MAX_SYMBOLS);
                        if (rx_timestamp < packet_early_thres)
                            next_slot_early_rx_packets_count++;
                        else if (rx_timestamp > packet_late_thres)
                            next_slot_late_rx_packets_count++;
                        else
                            next_slot_on_time_rx_packets_count++;
                        if constexpr (ul_rx_pkt_tracing_level) {
                            uint32_t next_slot_rx_packets_ts_idx =
                                    atomicAdd(&next_slot_rx_packets_count_sh[symbol_id_pkt], 1);
                            atomicAdd(
                                    &next_slot_rx_bytes_count_sh[symbol_id_pkt],
                                    ORAN_ETH_HDR_SIZE + ecpri_payload_length);
                            __threadfence_block();
                            next_slot_rx_packets_ts_idx +=
                                    ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM * symbol_id_pkt;
                            next_slot_rx_packets_ts_sh[next_slot_rx_packets_ts_idx] = rx_timestamp;
                        }
                    }
                    if (laneId == 0) {
                        atomicCAS(&done_shared_sh, 1, 0);
                    }
                } else if (full_slot_diff == 0) {
                    /* if this is the right slot, order & decompress */
                    uint8_t *section_buf = oran_umsg_get_first_section_buf(pkt_thread);
                    // 4 bytes for ecpriPcid, ecprSeqid, ecpriEbit, ecpriSubSeqid
                    uint16_t current_length = 4 + sizeof(oran_umsg_iq_hdr);
                    uint16_t num_sections = 0;
                    bool sanity_check = (current_length < ecpri_payload_length);
                    if (ecpri_hdr_sanity_check(pkt_thread) == false) {
                        printf("ERROR malformatted eCPRI header... block %d thread %d\n",
                               blockIdx.x,
                               threadIdx.x);
                        // break;
                    }
                    const uint16_t startPRB_offset_idx_0 = 0;
                    const uint16_t startPRB_offset_idx_1 = 0;
                    const uint16_t startPRB_offset_idx_2 = 0;
                    const uint16_t startPRB_offset_idx_3 = 0;
                    while (current_length < ecpri_payload_length) {
                        if (current_length + ORAN_IQ_UNCOMPRESSED_SECTION_OVERHEAD >=
                            ecpri_payload_length) {
                            sanity_check = false;
                            break;
                        }

                        uint16_t num_prb = oran_umsg_get_num_prb_from_section_buf(section_buf);
                        const uint16_t section_id =
                                oran_umsg_get_section_id_from_section_buf(section_buf);
                        const uint16_t start_prb =
                                oran_umsg_get_start_prb_from_section_buf(section_buf);

                        if (num_prb == 0)
                            num_prb = ORAN_MAX_PRB_X_SLOT;
                        const uint16_t prb_buffer_size = compressed_prb_size * num_prb;

                        // WAR added for FXN O-RU to pass. Will remove it when new FW is applied to
                        // fix the erronous ecpri payload length
                        if (ru_type_cell != SINGLE_SECT_MODE &&
                            current_length + prb_buffer_size +
                                            ORAN_IQ_UNCOMPRESSED_SECTION_OVERHEAD >
                                    ecpri_payload_length) {
                            sanity_check = false;
                            break;
                        }
                        uint8_t *pkt_offset_ptr =
                                section_buf + ORAN_IQ_UNCOMPRESSED_SECTION_OVERHEAD;
                        uint8_t *gbuf_offset_ptr;
                        uint8_t *gbuf_offset_ptr_srs = NULL;
                        uint8_t *buffer;

                        if constexpr (srs_enable == ORDER_KERNEL_SRS_ENABLE) {
                            buffer = srs_buffer_cell;
                            gbuf_offset_ptr =
                                    buffer + oran_srs_get_offset_from_hdr(
                                                     pkt_thread,
                                                     (uint16_t)get_eaxc_index(
                                                             srs_eAxC_map_cell,
                                                             srs_eAxC_num_cell,
                                                             oran_umsg_get_flowid(pkt_thread)),
                                                     srs_symbols_x_slot,
                                                     srs_prb_x_port_x_symbol_cell,
                                                     prb_size,
                                                     start_prb,
                                                     srs_start_sym_cell);
                        } else {
                            // Check if section_id is NOT a PRACH section (i.e., it's PUSCH)
                            if (section_id != prach_section_id_0 &&
                                section_id != prach_section_id_1 &&
                                section_id != prach_section_id_2 &&
                                section_id != prach_section_id_3) {

                                if (srs_enable == ORDER_KERNEL_PUSCH_ONLY) {
                                    buffer = pusch_buffer_cell;
                                    gbuf_offset_ptr =
                                            buffer +
                                            oran_get_offset_from_hdr(
                                                    pkt_thread,
                                                    (uint16_t)get_eaxc_index(
                                                            pusch_eAxC_map_cell,
                                                            pusch_eAxC_num_cell,
                                                            oran_umsg_get_flowid(pkt_thread)),
                                                    pusch_symbols_x_slot,
                                                    pusch_prb_x_port_x_symbol_cell,
                                                    prb_size,
                                                    start_prb);
                                } else if (srs_enable == ORDER_KERNEL_SRS_AND_PUSCH) {
                                    buffer = pusch_buffer_cell;
                                    gbuf_offset_ptr =
                                            buffer +
                                            oran_get_offset_from_hdr(
                                                    pkt_thread,
                                                    (uint16_t)get_eaxc_index(
                                                            pusch_eAxC_map_cell,
                                                            pusch_eAxC_num_cell,
                                                            oran_umsg_get_flowid(pkt_thread)),
                                                    pusch_symbols_x_slot,
                                                    pusch_prb_x_port_x_symbol_cell,
                                                    prb_size,
                                                    start_prb);
                                    if (symbol_id_pkt == srs_start_sym_cell) {
                                        buffer = srs_buffer_cell;
                                        gbuf_offset_ptr_srs =
                                                buffer +
                                                oran_srs_get_offset_from_hdr(
                                                        pkt_thread,
                                                        (uint16_t)get_eaxc_index(
                                                                srs_eAxC_map_cell,
                                                                srs_eAxC_num_cell,
                                                                oran_umsg_get_flowid(pkt_thread)),
                                                        srs_symbols_x_slot,
                                                        srs_prb_x_port_x_symbol_cell,
                                                        prb_size,
                                                        start_prb,
                                                        srs_start_sym_cell);
                                    }
                                }
                            } else {
                                if (section_id == prach_section_id_0)
                                    buffer = prach_buffer_0_cell;
                                else if (section_id == prach_section_id_1)
                                    buffer = prach_buffer_1_cell;
                                else if (section_id == prach_section_id_2)
                                    buffer = prach_buffer_2_cell;
                                else if (section_id == prach_section_id_3)
                                    buffer = prach_buffer_3_cell;
                                gbuf_offset_ptr =
                                        buffer + oran_get_offset_from_hdr(
                                                         pkt_thread,
                                                         (uint16_t)get_eaxc_index(
                                                                 prach_eAxC_map_cell,
                                                                 prach_eAxC_num_cell,
                                                                 oran_umsg_get_flowid(pkt_thread)),
                                                         prach_symbols_x_slot,
                                                         prach_prb_x_port_x_symbol_cell,
                                                         prb_size,
                                                         start_prb);

                                /* prach_buffer_x_cell is populated based on number of PRACH PDU's,
                                   hence the index can be used as "Frequency domain occasion index"
                                    and mutiplying with num_prb i.e. NRARB=12 (NumRB's (PRACH
                                   SCS=30kHz) for each FDM ocassion) will yeild the corrosponding
                                   PRB start for each Frequency domain index Note: WIP for a more
                                   generic approach to caluclate and pass the startRB from the
                                   cuPHY-CP */
                                if (section_id == prach_section_id_0)
                                    gbuf_offset_ptr -= startPRB_offset_idx_0;
                                else if (section_id == prach_section_id_1)
                                    gbuf_offset_ptr -= startPRB_offset_idx_1;
                                else if (section_id == prach_section_id_2)
                                    gbuf_offset_ptr -= startPRB_offset_idx_2;
                                else if (section_id == prach_section_id_3)
                                    gbuf_offset_ptr -= startPRB_offset_idx_3;
                            }
                        }

                        if (comp_meth_cell ==
                            static_cast<uint8_t>(
                                    ::UserDataCompressionMethod::BLOCK_FLOATING_POINT)) {
                            if (bit_width_cell ==
                                BFP_NO_COMPRESSION) // BFP with 16 bits is a special case and uses
                                                    // FP16, so copy the values
                            {
                                for (int index_copy = laneId; index_copy < (num_prb * prb_size);
                                     index_copy += 32)
                                    gbuf_offset_ptr[index_copy] = pkt_offset_ptr[index_copy];
                            } else {
                                decompress_scale_blockFP<false>(
                                        (unsigned char *)pkt_offset_ptr,
                                        (__half *)gbuf_offset_ptr,
                                        beta_cell,
                                        num_prb,
                                        bit_width_cell,
                                        (int)(threadIdx.x & 31),
                                        32);

                                if (srs_enable == ORDER_KERNEL_SRS_AND_PUSCH &&
                                    buffer == srs_buffer_cell) { // Copy it to both places. Could
                                                                 // also create a function with two
                                                                 // output pointers.
                                    decompress_scale_blockFP<false>(
                                            (unsigned char *)pkt_offset_ptr,
                                            (__half *)gbuf_offset_ptr_srs,
                                            beta_cell,
                                            num_prb,
                                            bit_width_cell,
                                            (int)(threadIdx.x & 31),
                                            32);
                                }
                            }
                        } else // aerial_fh::UserDataCompressionMethod::NO_COMPRESSION
                        {
                            decompress_scale_fixed<false>(
                                    pkt_offset_ptr,
                                    (__half *)gbuf_offset_ptr,
                                    beta_cell,
                                    num_prb,
                                    bit_width_cell,
                                    (int)(threadIdx.x & 31),
                                    32);
                        }
                        if (laneId == 0) {
                            if constexpr (srs_enable == ORDER_KERNEL_SRS_ENABLE) {
                                warp_srs_ordered_prbs_cell += num_prb;
                                warp_srs_ordered_prbs_cell_this_burst += num_prb;
                            } else {
                                // Check if section_id is NOT a PRACH section (i.e., it's PUSCH)
                                if (section_id != prach_section_id_0 &&
                                    section_id != prach_section_id_1 &&
                                    section_id != prach_section_id_2 &&
                                    section_id != prach_section_id_3) {
                                    uint32_t tot_pusch_prb_symbol_ordered = atomicAdd(
                                            &pusch_prb_symbol_ordered[symbol_id_pkt], num_prb);
                                    tot_pusch_prb_symbol_ordered += num_prb;
                                    if (pusch_prb_non_zero &&
                                        tot_pusch_prb_symbol_ordered >=
                                                pusch_prb_symbol_map_cell[symbol_id_pkt] &&
                                        DOCA_GPUNETIO_VOLATILE(
                                                pusch_prb_symbol_ordered_done[symbol_id_pkt]) ==
                                                0) {
                                        DOCA_GPUNETIO_VOLATILE(
                                                pusch_prb_symbol_ordered_done[symbol_id_pkt]) = 1;
                                        atomicOr(
                                                &sym_ord_done_mask_arr[symbol_id_pkt],
                                                cell_idx_mask);
                                        if (DOCA_GPUNETIO_VOLATILE(
                                                    sym_ord_done_mask_arr[symbol_id_pkt]) ==
                                                    num_order_cells_sym_mask_arr[symbol_id_pkt] &&
                                            DOCA_GPUNETIO_VOLATILE(
                                                    sym_ord_done_sig_arr[symbol_id_pkt]) ==
                                                    (uint32_t)SYM_RX_NOT_DONE) {
                                            DOCA_GPUNETIO_VOLATILE(
                                                    sym_ord_done_sig_arr[symbol_id_pkt]) =
                                                    (uint32_t)SYM_RX_DONE;
                                        }
                                    }
                                    warp_pusch_ordered_prbs_cell += num_prb;
                                    if (buffer == srs_buffer_cell) {
                                        warp_srs_ordered_prbs_cell += num_prb;
                                    }
                                } else {
                                    warp_prach_ordered_prbs_cell += num_prb;
                                }
                                warp_pusch_prach_ordered_prbs_cell_this_burst += num_prb;
                            }
                        }

                        current_length += prb_buffer_size + ORAN_IQ_UNCOMPRESSED_SECTION_OVERHEAD;
                        section_buf = pkt_offset_ptr + prb_buffer_size;
                        ++num_sections;
                        if (num_sections > ORAN_MAX_PRB_X_SLOT) {
                            printf("Invalid U-Plane packet, num_sections %d > 273 for Cell %d "
                                   "F%dS%dS%d\n",
                                   num_sections,
                                   (blockIdx.x / 2),
                                   frameId_pkt,
                                   subframeId_pkt,
                                   slotId_pkt);
                            break;
                        }
                    }
                    if (record_packet_stats && laneId == 0) {
                        const uint64_t packet_early_thres =
                                slot_start_cell + ta4_min_ns_cell +
                                (slot_duration_cell * symbol_id_pkt / ORAN_MAX_SYMBOLS);
                        const uint64_t packet_late_thres =
                                slot_start_cell + ta4_max_ns_cell +
                                (slot_duration_cell * symbol_id_pkt / ORAN_MAX_SYMBOLS);
                        if (rx_timestamp < packet_early_thres)
                            early_rx_packets_count++;
                        else if (rx_timestamp > packet_late_thres)
                            late_rx_packets_count++;
                        else
                            on_time_rx_packets_count++;
                        if constexpr (ul_rx_pkt_tracing_level) {
                            uint32_t rx_packets_ts_idx =
                                    atomicAdd(&rx_packets_count_sh[symbol_id_pkt], 1);
                            atomicAdd(
                                    &rx_bytes_count_sh[symbol_id_pkt],
                                    ORAN_ETH_HDR_SIZE + ecpri_payload_length);
                            __threadfence_block();
                            rx_packets_ts_idx += ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM * symbol_id_pkt;
                            rx_packets_ts_sh[rx_packets_ts_idx] = rx_timestamp;
                            atomicMin(
                                    (unsigned long long *)&rx_packets_ts_earliest_sh[symbol_id_pkt],
                                    (unsigned long long)rx_timestamp);
                            atomicMax(
                                    (unsigned long long *)&rx_packets_ts_latest_sh[symbol_id_pkt],
                                    (unsigned long long)rx_timestamp);
                            __threadfence_block();
                        }
                    }
                    if (!sanity_check) {
                        printf("ERROR uplane pkt sanity check failed, it could be erroneous BFP, "
                               "numPrb or ecpri payload len, or other reasons... block %d thread "
                               "%d\n",
                               blockIdx.x,
                               threadIdx.x);
                        atomicCAS(exit_cond_d_cell, ORDER_KERNEL_RUNNING, ORDER_KERNEL_EXIT_ERROR7);
                        break;
                    }
                } else // Drop packets that are in the past slots
                {
                    if (record_packet_stats && laneId == 0) {
                        packets_dropped_count++;
                    }
                }
            }

            if (laneId == 0) {
                uint32_t old_prb_count, num_prb_added;
                if constexpr (srs_enable == ORDER_KERNEL_SRS_ENABLE) {
                    num_prb_added = warp_srs_ordered_prbs_cell_this_burst;
                    old_prb_count = atomicAdd(&smem_srs_ordered_prbs_cell, num_prb_added);
                } else {
                    num_prb_added = warp_pusch_prach_ordered_prbs_cell_this_burst;
                    old_prb_count = atomicAdd(&smem_pusch_prach_ordered_prbs_cell, num_prb_added);
                }

                if (old_prb_count < prb_x_slot && old_prb_count + num_prb_added >= prb_x_slot) {
                    atomicCAS(exit_cond_d_cell, ORDER_KERNEL_RUNNING, ORDER_KERNEL_EXIT_PRB);
                }
            }
            __syncthreads();

            if (tid == 0 && DOCA_GPUNETIO_VOLATILE(done_shared_sh) == 1) {
                doca_gpu_dev_semaphore_set_status(
                        sem_gpu_cell, last_sem_idx_order, DOCA_GPU_SEMAPHORE_STATUS_DONE);
                last_sem_idx_order = (last_sem_idx_order + 1) & (sem_order_num_cell - 1);
            }

            sem_idx_order = (sem_idx_order + 1) & (sem_order_num_cell - 1);
        }
        if (tid == 0) {
            const unsigned long long current_time = __globaltimer();
            if (first_packet_received &&
                ((current_time - first_packet_received_time) > timeout_first_pkt_ns)) {
                if constexpr (srs_enable == ORDER_KERNEL_SRS_ENABLE) {
                    if (timeout_log_enable) {
                        if ((current_time -
                             DOCA_GPUNETIO_VOLATILE(*order_kernel_last_timeout_error_time_cell)) >
                            timeout_log_interval_ns) {
                            printf("%d Cell %d Order kernel sem_idx_rx %d last_sem_idx_rx %d "
                                   "sem_idx_order %d last_sem_idx_order %d SRS PRBs %d/%d.  First "
                                   "packet received timeout after %d ns F%dS%dS%d done = %d "
                                   "current_time=%llu,last_timeout_log_time=%llu,total_rx_pkts=%"
                                   "d\n",
                                   __LINE__,
                                   cell_idx,
                                   sem_idx_rx,
                                   *last_sem_idx_rx_h_cell,
                                   sem_idx_order,
                                   *last_sem_idx_order_h_cell,
                                   DOCA_GPUNETIO_VOLATILE(srs_ordered_prbs_cell[0]),
                                   srs_prb_x_slot_cell,
                                   timeout_first_pkt_ns,
                                   frameId,
                                   subframeId,
                                   slotId,
                                   DOCA_GPUNETIO_VOLATILE(done_shared_sh),
                                   current_time,
                                   DOCA_GPUNETIO_VOLATILE(
                                           *order_kernel_last_timeout_error_time_cell),
                                   rx_pkt_num_total);
                            DOCA_GPUNETIO_VOLATILE(*order_kernel_last_timeout_error_time_cell) =
                                    current_time;
                        }
                    }
                } else {
                    if (timeout_log_enable) {
                        if ((current_time -
                             DOCA_GPUNETIO_VOLATILE(*order_kernel_last_timeout_error_time_cell)) >
                            timeout_log_interval_ns) {
                            printf("%d Cell %d Order kernel sem_idx_rx %d last_sem_idx_rx %d "
                                   "sem_idx_order %d last_sem_idx_order %d PUSCH PRBs %d/%d PRACH "
                                   "PRBs %d/%d.  First packet received timeout after %d ns "
                                   "F%dS%dS%d done = %d "
                                   "current_time=%llu,last_timeout_log_time=%llu,total_rx_pkts=%"
                                   "d\n",
                                   __LINE__,
                                   cell_idx,
                                   sem_idx_rx,
                                   *last_sem_idx_rx_h_cell,
                                   sem_idx_order,
                                   *last_sem_idx_order_h_cell,
                                   DOCA_GPUNETIO_VOLATILE(pusch_ordered_prbs_cell[0]),
                                   pusch_prb_x_slot_cell,
                                   DOCA_GPUNETIO_VOLATILE(prach_ordered_prbs_cell[0]),
                                   prach_prb_x_slot_cell,
                                   timeout_first_pkt_ns,
                                   frameId,
                                   subframeId,
                                   slotId,
                                   DOCA_GPUNETIO_VOLATILE(done_shared_sh),
                                   current_time,
                                   DOCA_GPUNETIO_VOLATILE(
                                           *order_kernel_last_timeout_error_time_cell),
                                   rx_pkt_num_total);
                            DOCA_GPUNETIO_VOLATILE(*order_kernel_last_timeout_error_time_cell) =
                                    current_time;
                        }
                    }
                    if (pusch_prb_non_zero) {
                        for (uint32_t idx = 0; idx < ORAN_PUSCH_SYMBOLS_X_SLOT; idx++) {
                            DOCA_GPUNETIO_VOLATILE(sym_ord_done_sig_arr[idx]) =
                                    (uint32_t)SYM_RX_TIMEOUT;
                        }
                    }
                }
                atomicCAS(exit_cond_d_cell, ORDER_KERNEL_RUNNING, ORDER_KERNEL_EXIT_TIMEOUT_RX_PKT);
            } else if (
                    (!first_packet_received) &&
                    ((current_time - kernel_start) > timeout_no_pkt_ns)) {
                if constexpr (srs_enable == ORDER_KERNEL_SRS_ENABLE) {
                    if (timeout_log_enable) {
                        if ((current_time -
                             DOCA_GPUNETIO_VOLATILE(*order_kernel_last_timeout_error_time_cell)) >
                            timeout_log_interval_ns) {
                            printf("%d Cell %d Order kernel sem_idx_rx %d last_sem_idx_rx %d "
                                   "sem_idx_order %d last_sem_idx_order %d SRS PRBs %d/%d. No "
                                   "packet received timeout after %d ns F%dS%dS%d done = %d "
                                   "current_time=%llu,last_timeout_log_time=%llu\n",
                                   __LINE__,
                                   cell_idx,
                                   sem_idx_rx,
                                   *last_sem_idx_rx_h_cell,
                                   sem_idx_order,
                                   *last_sem_idx_order_h_cell,
                                   DOCA_GPUNETIO_VOLATILE(srs_ordered_prbs_cell[0]),
                                   srs_prb_x_slot_cell,
                                   timeout_no_pkt_ns,
                                   frameId,
                                   subframeId,
                                   slotId,
                                   DOCA_GPUNETIO_VOLATILE(done_shared_sh),
                                   current_time,
                                   DOCA_GPUNETIO_VOLATILE(
                                           *order_kernel_last_timeout_error_time_cell));
                            DOCA_GPUNETIO_VOLATILE(*order_kernel_last_timeout_error_time_cell) =
                                    current_time;
                        }
                    }
                } else {
                    if (timeout_log_enable) {
                        if ((current_time -
                             DOCA_GPUNETIO_VOLATILE(*order_kernel_last_timeout_error_time_cell)) >
                            timeout_log_interval_ns) {
                            printf("%d Cell %d Order kernel sem_idx_rx %d last_sem_idx_rx %d "
                                   "sem_idx_order %d last_sem_idx_order %d PUSCH PRBs %d/%d PRACH "
                                   "PRBs %d/%d. No packet received timeout after %d ns F%dS%dS%d "
                                   "done = %d current_time=%llu,last_timeout_log_time=%llu\n",
                                   __LINE__,
                                   cell_idx,
                                   sem_idx_rx,
                                   *last_sem_idx_rx_h_cell,
                                   sem_idx_order,
                                   *last_sem_idx_order_h_cell,
                                   DOCA_GPUNETIO_VOLATILE(pusch_ordered_prbs_cell[0]),
                                   pusch_prb_x_slot_cell,
                                   DOCA_GPUNETIO_VOLATILE(prach_ordered_prbs_cell[0]),
                                   prach_prb_x_slot_cell,
                                   timeout_no_pkt_ns,
                                   frameId,
                                   subframeId,
                                   slotId,
                                   DOCA_GPUNETIO_VOLATILE(done_shared_sh),
                                   current_time,
                                   DOCA_GPUNETIO_VOLATILE(
                                           *order_kernel_last_timeout_error_time_cell));
                            DOCA_GPUNETIO_VOLATILE(*order_kernel_last_timeout_error_time_cell) =
                                    current_time;
                        }
                    }
                    if (pusch_prb_non_zero) {
                        for (uint32_t idx = 0; idx < ORAN_PUSCH_SYMBOLS_X_SLOT; idx++) {
                            DOCA_GPUNETIO_VOLATILE(sym_ord_done_sig_arr[idx]) =
                                    (uint32_t)SYM_RX_TIMEOUT;
                        }
                    }
                }
                atomicCAS(exit_cond_d_cell, ORDER_KERNEL_RUNNING, ORDER_KERNEL_EXIT_TIMEOUT_NO_PKT);
            }
            DOCA_GPUNETIO_VOLATILE(rx_pkt_num) = 0;
        }

        __syncthreads();

        if (DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell) != ORDER_KERNEL_RUNNING)
            break;

        __syncthreads();

        {
            doca_error_t ret = doca_gpu_dev_eth_rxq_receive_block(
                    doca_rxq_cell, max_rx_pkts, rx_pkts_timeout_ns, &rx_pkt_num, &rx_buf_idx);
            /* If any thread returns receive error, the whole execution stops */
            if (ret != DOCA_SUCCESS) {
                doca_gpu_dev_semaphore_set_status(
                        sem_gpu_cell, sem_idx_rx, DOCA_GPU_SEMAPHORE_STATUS_ERROR);
                atomicCAS(exit_cond_d_cell, ORDER_KERNEL_RUNNING, ORDER_KERNEL_EXIT_ERROR1);
                printf("Exit from rx kernel block %d threadIdx %d ret %d sem_idx_rx %d\n",
                       blockIdx.x,
                       threadIdx.x,
                       ret,
                       sem_idx_rx);
            } else {
                if (rx_pkt_num > 0) {
                    if (tid == 0) {
                        doca_gpu_dev_semaphore_set_packet_info(
                                sem_gpu_cell,
                                sem_idx_rx,
                                DOCA_GPU_SEMAPHORE_STATUS_READY,
                                rx_pkt_num,
                                rx_buf_idx);
                        rx_pkt_num_total += rx_pkt_num;
                        if (first_packet_received == 0) {
                            first_packet_received = 1;
                            first_packet_received_time = __globaltimer();
                        }
                    }
                }
            }
        }

        __syncthreads();

        if (DOCA_GPUNETIO_VOLATILE(rx_pkt_num) == 0) {
            continue;
        }

        // For newly read packets, we want to record packet stats the first time
        // we process them.
        record_packet_stats = true;
        sem_idx_rx = (sem_idx_rx + 1) & (sem_order_num_cell - 1);

        __syncthreads();
    }

    // Only laneId == 0 threads should have non-zero packet stat counts
    if (early_rx_packets_count)
        atomicAdd(&early_rx_packets_count_sh, early_rx_packets_count);
    if (late_rx_packets_count)
        atomicAdd(&late_rx_packets_count_sh, late_rx_packets_count);
    if (on_time_rx_packets_count)
        atomicAdd(&on_time_rx_packets_count_sh, on_time_rx_packets_count);
    if (next_slot_early_rx_packets_count)
        atomicAdd(&next_slot_early_rx_packets_count_sh, next_slot_early_rx_packets_count);
    if (next_slot_late_rx_packets_count)
        atomicAdd(&next_slot_late_rx_packets_count_sh, next_slot_late_rx_packets_count);
    if (next_slot_on_time_rx_packets_count)
        atomicAdd(&next_slot_on_time_rx_packets_count_sh, next_slot_on_time_rx_packets_count);
    if (packets_dropped_count)
        atomicAdd(&rx_packets_dropped_count_sh, packets_dropped_count);
    __syncthreads();

    if constexpr (ul_rx_pkt_tracing_level) {
        __syncthreads();

        if (tid < ORAN_MAX_SYMBOLS) {
            pkt_tracing_info.rx_packets_count[cell_idx][tid] = rx_packets_count_sh[tid];
            pkt_tracing_info.rx_bytes_count[cell_idx][tid] = rx_bytes_count_sh[tid];
            pkt_tracing_info.next_slot_rx_packets_count[cell_idx][tid] =
                    next_slot_rx_packets_count_sh[tid];
            pkt_tracing_info.next_slot_rx_bytes_count[cell_idx][tid] =
                    next_slot_rx_bytes_count_sh[tid];
            pkt_tracing_info.rx_packets_ts_earliest[cell_idx][tid] = rx_packets_ts_earliest_sh[tid];
            pkt_tracing_info.rx_packets_ts_latest[cell_idx][tid] = rx_packets_ts_latest_sh[tid];
        }

        __syncthreads();
        const int max_pkt_idx = ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM * ORAN_MAX_SYMBOLS;
        uint64_t *rx_packets_ts_cell = pkt_tracing_info.rx_packets_ts[cell_idx];
        uint64_t *next_slot_rx_packets_ts_cell = pkt_tracing_info.next_slot_rx_packets_ts[cell_idx];
        for (uint32_t pkt_idx = tid; pkt_idx < max_pkt_idx; pkt_idx += blockDim.x) {
            DOCA_GPUNETIO_VOLATILE(rx_packets_ts_cell[pkt_idx]) = rx_packets_ts_sh[pkt_idx];
            DOCA_GPUNETIO_VOLATILE(next_slot_rx_packets_ts_cell[pkt_idx]) =
                    next_slot_rx_packets_ts_sh[pkt_idx];
        }
        __syncthreads();
    }

    if (laneId == 0) {
        if constexpr (srs_enable == ORDER_KERNEL_SRS_ENABLE) {
            atomicAdd(srs_ordered_prbs_cell, warp_srs_ordered_prbs_cell);
        } else {
            atomicAdd(pusch_ordered_prbs_cell, warp_pusch_ordered_prbs_cell);
            atomicAdd(prach_ordered_prbs_cell, warp_prach_ordered_prbs_cell);
            if (srs_enable == ORDER_KERNEL_SRS_AND_PUSCH) {
                atomicAdd(srs_ordered_prbs_cell, warp_srs_ordered_prbs_cell);
            }
        }
    }

    if (tid == 0) {
        DOCA_GPUNETIO_VOLATILE(*last_sem_idx_rx_h_cell) = sem_idx_rx;
        DOCA_GPUNETIO_VOLATILE(*early_rx_packets_cell) = early_rx_packets_count_sh;
        DOCA_GPUNETIO_VOLATILE(*on_time_rx_packets_cell) = on_time_rx_packets_count_sh;
        DOCA_GPUNETIO_VOLATILE(*late_rx_packets_cell) = late_rx_packets_count_sh;
        DOCA_GPUNETIO_VOLATILE(*next_slot_early_rx_packets_cell) =
                next_slot_early_rx_packets_count_sh;
        DOCA_GPUNETIO_VOLATILE(*next_slot_on_time_rx_packets_cell) =
                next_slot_on_time_rx_packets_count_sh;
        DOCA_GPUNETIO_VOLATILE(*next_slot_late_rx_packets_cell) =
                next_slot_late_rx_packets_count_sh;
        DOCA_GPUNETIO_VOLATILE(*rx_packets_dropped_count_cell) = rx_packets_dropped_count_sh;
        DOCA_GPUNETIO_VOLATILE(*last_sem_idx_order_h_cell) = last_sem_idx_order;
        uint32_t *start_cuphy_d_cell = *(start_cuphy_d + cell_idx);
        DOCA_GPUNETIO_VOLATILE(*start_cuphy_d_cell) = 1;
    }
}

// Explicit template instantiation for the configuration used by OrderKernelModule
// This ensures the CUDA device linker can resolve the template instantiation when
// linking the test executable. Without this, the template is only instantiated in
// order_kernel_module.cpp (a .cpp file), and the device linker cannot see it.
template __global__ void
        __launch_bounds__(320, 1) order_kernel_doca_single_subSlot_pingpong<false, 0, 0, 320, 1>(
                const OrderKernelStaticDescriptor *static_desc,
                const OrderKernelDynamicDescriptor *dynamic_desc);

} // namespace ran::fronthaul
