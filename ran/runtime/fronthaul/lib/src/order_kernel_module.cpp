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

// OrderKernelModule implementation for ORAN UL Receiver

#include <algorithm>
#include <any>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <format>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <driver_types.h>
#include <quill/LogMacros.h>

#include <gsl-lite/gsl-lite.hpp>

#include <cuda.h>      // CUDA driver API for graph operations
#include <cuda_fp16.h> // For __half type
#include <cuda_runtime_api.h>

#include "fronthaul/fronthaul_log.hpp"
#include "fronthaul/oran_order_kernels.hpp"
#include "fronthaul/order_kernel_descriptors.hpp"
#include "fronthaul/order_kernel_module.hpp"
#include "log/components.hpp"
#include "log/rt_log_macros.hpp"
#include "memory/gdrcopy_buffer.hpp"
#include "memory/unique_ptr_utils.hpp"
#include "net/doca_types.hpp"
#include "oran/cplane_types.hpp"
#include "pipeline/igraph.hpp"
#include "pipeline/igraph_node_provider.hpp"
#include "pipeline/istream_executor.hpp"
#include "pipeline/kernel_descriptor_accessor.hpp"
#include "pipeline/kernel_launch_config.hpp"
#include "pipeline/types.hpp"
#include "tensor/data_types.hpp"
#include "tensor/tensor_info.hpp"
#include "utils/core_log.hpp"
#include "utils/error_macros.hpp"

namespace ran::fronthaul {

// Namespace aliases for cleaner code
namespace pipeline = framework::pipeline;
namespace memory = framework::memory;
namespace tensor = framework::tensor;

// ============================================================================
// Memory Management Constants
// ============================================================================

/// Maximum number of antenna ports (eAxC IDs) per slot
inline constexpr std::uint32_t MAX_ANTENNA_PORTS_PER_SLOT = 4;

/// PUSCH resource configuration for 100MHz bandwidth
inline constexpr int PUSCH_NUM_PRB = 273;                      //!< Number of PRBs
inline constexpr int NUM_ANTENNA_PORTS = 4;                    //!< Number of antenna ports
inline constexpr int PUSCH_RE_PER_PRB = 12;                    //!< Resource elements per PRB
inline constexpr std::uint32_t ORAN_PUSCH_SYMBOLS_X_SLOT = 14; //!< PUSCH symbols per slot

/// PUSCH tensor dimensions for TensorInfo (FP16 complex, real/imag interleaved)
inline constexpr std::size_t PUSCH_REAL_IMAG = 2; //!< Real + imaginary components
/// Total number of FP16 elements in PUSCH buffer (273 × 12 × 14 × 4 × 2 = 366,912)
inline constexpr std::size_t PUSCH_NUM_ELEMENTS = static_cast<std::size_t>(PUSCH_NUM_PRB) *
                                                  PUSCH_RE_PER_PRB * ORAN_PUSCH_SYMBOLS_X_SLOT *
                                                  NUM_ANTENNA_PORTS * PUSCH_REAL_IMAG;

/// Total size of PUSCH buffer in bytes (366,912 × 2 = 733,824 bytes)
inline constexpr std::size_t PUSCH_SIZE_BYTES = PUSCH_NUM_ELEMENTS * sizeof(__half);

// ============================================================================
// Construction
// ============================================================================

OrderKernelModule::OrderKernelModule(std::string instance_id, const StaticParams &params)
        : instance_id_(std::move(instance_id)), execution_mode_(params.execution_mode),
          timing_params_(params.timing), eaxc_ids_(params.eaxc_ids), gdr_handle_(params.gdr_handle),
          doca_rxq_params_(params.doca_rxq_params) {

    RT_LOGC_INFO(
            FronthaulKernels::OrderModule,
            "OrderKernelModule: Constructing instance '{}', execution_mode={}",
            instance_id_,
            execution_mode_ == pipeline::ExecutionMode::Graph ? "Graph" : "Stream");

    // Validate preconditions
    gsl_Expects(gdr_handle_ != nullptr);
    gsl_Expects(doca_rxq_params_ != nullptr);

    // Validate timing parameters
    gsl_Expects(timing_params_.ta4_min_ns < timing_params_.ta4_max_ns);
    gsl_Expects(timing_params_.ta4_max_ns <= timing_params_.slot_duration_ns);

    RT_LOGC_DEBUG(
            FronthaulKernels::OrderModule,
            "OrderKernelModule '{}': Timing params - slot_duration={}ns, ta4_min={}ns, "
            "ta4_max={}ns",
            instance_id_,
            timing_params_.slot_duration_ns,
            timing_params_.ta4_min_ns,
            timing_params_.ta4_max_ns);

    RT_LOGC_DEBUG(
            FronthaulKernels::OrderModule,
            "OrderKernelModule '{}': Constructor complete (Phase 1 skeleton)",
            instance_id_);
}

// ============================================================================
// Interface Access
// ============================================================================

pipeline::IStreamExecutor *OrderKernelModule::as_stream_executor() { return this; }

pipeline::IGraphNodeProvider *OrderKernelModule::as_graph_node_provider() { return this; }

// ============================================================================
// Port Introspection (called during pipeline construction)
// ============================================================================

std::vector<std::string> OrderKernelModule::get_input_port_names() const {
    return {"doca_objects"};
}

std::vector<std::string> OrderKernelModule::get_output_port_names() const {
    return {"pusch"}; // Phase 2: Add "prach", "srs"
}

std::vector<tensor::TensorInfo>
OrderKernelModule::get_input_tensor_info(std::string_view port_name) const {
    if (port_name != "doca_objects") {
        const std::string error_msg = std::format(
                "OrderKernelModule '{}': Unknown input port '{}'", instance_id_, port_name);
        RT_LOGC_ERROR(FronthaulKernels::OrderModule, "{}", error_msg);
        throw std::invalid_argument(error_msg);
    }

    // Single tensor containing pointer to DOCA objects (opaque pointer, 8 bytes for 64-bit ptr)
    constexpr std::size_t PTR_SIZE_BYTES = 8;
    return {tensor::TensorInfo(tensor::TensorInfo::DataType::TensorR8U, {PTR_SIZE_BYTES})};
}

std::vector<tensor::TensorInfo>
OrderKernelModule::get_output_tensor_info(std::string_view port_name) const {
    if (port_name != "pusch") {
        const std::string error_msg = std::format(
                "OrderKernelModule '{}': Unknown output port '{}'", instance_id_, port_name);
        RT_LOGC_ERROR(FronthaulKernels::OrderModule, "{}", error_msg);
        throw std::invalid_argument(error_msg);
    }

    // PUSCH buffer: FP16 complex (real/imag interleaved)
    // Element count and size defined at top of file
    return {tensor::TensorInfo(tensor::TensorInfo::DataType::TensorR16F, {PUSCH_NUM_ELEMENTS})};
}

// ============================================================================
// Memory Configuration (called before setup())
// ============================================================================

pipeline::InputPortMemoryCharacteristics
OrderKernelModule::get_input_memory_characteristics(std::string_view port_name) const {
    if (port_name == "doca_objects") {
        // DOCA objects are external (stable pointers, don't change)
        return pipeline::InputPortMemoryCharacteristics{
                .requires_fixed_address_for_zero_copy = true};
    }

    const std::string error_msg =
            std::format("OrderKernelModule '{}': Unknown input port '{}'", instance_id_, port_name);
    RT_LOGC_ERROR(FronthaulKernels::OrderModule, "{}", error_msg);
    throw std::invalid_argument(error_msg);
}

pipeline::OutputPortMemoryCharacteristics
OrderKernelModule::get_output_memory_characteristics(std::string_view port_name) const {
    if (port_name == "pusch") {
        // Output buffer allocated once in setup_memory(), never changes
        return pipeline::OutputPortMemoryCharacteristics{
                .provides_fixed_address_for_zero_copy = true};
    }

    const std::string error_msg = std::format(
            "OrderKernelModule '{}': Unknown output port '{}'", instance_id_, port_name);
    RT_LOGC_ERROR(FronthaulKernels::OrderModule, "{}", error_msg);
    throw std::invalid_argument(error_msg);
}

pipeline::ModuleMemoryRequirements OrderKernelModule::get_requirements() const {
    pipeline::ModuleMemoryRequirements reqs{};

    // Return actual memory requirements
    constexpr std::size_t GPU_MEMORY_ALIGNMENT = 256;

    // Kernel descriptor sizes (framework-managed)
    reqs.static_kernel_descriptor_bytes = sizeof(OrderKernelStaticDescriptor);
    reqs.dynamic_kernel_descriptor_bytes = sizeof(OrderKernelDynamicDescriptor);

    // Device tensor memory: NOT using framework allocation (managed directly by module)
    // Design choice: For single-module pipeline, we manage device memory directly via cudaMalloc()
    // in setup_memory() for simpler ownership and easier debugging. Framework allocation is better
    // suited for multi-module pipelines where batch allocation reduces fragmentation.
    // Memory managed directly:
    //   - PUSCH buffer (~40MB): cudaMalloc(&d_pusch_buffer_, PUSCH_BUFFER_SIZE)
    //   - Semaphore indices (8 bytes): cudaMalloc(&d_last_sem_idx_rx/order_, sizeof(uint32_t))
    reqs.device_tensor_bytes = 0;

    reqs.alignment = GPU_MEMORY_ALIGNMENT;

    RT_LOGC_DEBUG(
            FronthaulKernels::OrderMemory,
            "OrderKernelModule '{}': Memory requirements - static_desc={}B, dynamic_desc={}B, "
            "device_tensors={}B (self-managed)",
            instance_id_,
            reqs.static_kernel_descriptor_bytes,
            reqs.dynamic_kernel_descriptor_bytes,
            reqs.device_tensor_bytes);

    return reqs;
}

// ============================================================================
// Setup (one-time initialization)
// ============================================================================

void OrderKernelModule::setup_memory(const pipeline::ModuleMemorySlice &memory_slice) {
    RT_LOGC_INFO(
            FronthaulKernels::OrderMemory,
            "OrderKernelModule '{}': setup_memory() - Phase 2 implementation",
            instance_id_);

    mem_slice_ = memory_slice;

    // ========================================================================
    // Allocate GDRCopy buffers
    // ========================================================================
    // Pattern from test_oran_order_kernel.cpp lines 117-184

    RT_LOGC_DEBUG(
            FronthaulKernels::OrderMemory,
            "OrderKernelModule '{}': Allocating GDRCopy buffers",
            instance_id_);

    try {
        // Buffer 0: exit_cond (uint32_t control flag)
        gdr_buffers_[ExitCond] =
                std::make_unique<memory::GpinnedBuffer>(gdr_handle_, sizeof(std::uint32_t));
        *static_cast<std::uint32_t *>(gdr_buffers_[ExitCond]->get_host_addr()) = 0;

        // Buffer 1: start_cuphy (uint32_t processing flag)
        gdr_buffers_[StartCuphy] =
                std::make_unique<memory::GpinnedBuffer>(gdr_handle_, sizeof(std::uint32_t));
        *static_cast<std::uint32_t *>(gdr_buffers_[StartCuphy]->get_host_addr()) = 0;

        // Buffer 2: early_rx_packets (uint32_t counter)
        gdr_buffers_[EarlyRxPackets] =
                std::make_unique<memory::GpinnedBuffer>(gdr_handle_, sizeof(std::uint32_t));
        *static_cast<std::uint32_t *>(gdr_buffers_[EarlyRxPackets]->get_host_addr()) = 0;

        // Buffer 3: on_time_rx_packets (uint32_t counter)
        gdr_buffers_[OnTimeRxPackets] =
                std::make_unique<memory::GpinnedBuffer>(gdr_handle_, sizeof(std::uint32_t));
        *static_cast<std::uint32_t *>(gdr_buffers_[OnTimeRxPackets]->get_host_addr()) = 0;

        // Buffer 4: late_rx_packets (uint32_t counter)
        gdr_buffers_[LateRxPackets] =
                std::make_unique<memory::GpinnedBuffer>(gdr_handle_, sizeof(std::uint32_t));
        *static_cast<std::uint32_t *>(gdr_buffers_[LateRxPackets]->get_host_addr()) = 0;

        // Buffer 5: eAxC_map (4 uint16_t eAxC ID mapping)
        // This maps packet flow IDs to antenna indices for buffer offset calculation
        gdr_buffers_[EaxcMap] = std::make_unique<memory::GpinnedBuffer>(
                gdr_handle_, MAX_ANTENNA_PORTS_PER_SLOT * sizeof(std::uint16_t));

        // Populate eAxC map with flow IDs from RU config
        auto *eaxc_map = static_cast<std::uint16_t *>(gdr_buffers_[EaxcMap]->get_host_addr());
        const std::size_t num_eaxc_ids =
                std::min(eaxc_ids_.size(), static_cast<std::size_t>(MAX_ANTENNA_PORTS_PER_SLOT));
        for (std::size_t i{}; i < num_eaxc_ids; ++i) {
            // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
            eaxc_map[i] = eaxc_ids_[i];
        }

        RT_LOGC_DEBUG(
                FronthaulKernels::OrderMemory,
                "OrderKernelModule '{}': eAxC map configured from config: {}",
                instance_id_,
                eaxc_ids_);

        // Buffer 6: ordered_prbs_pusch (uint32_t PRB count)
        gdr_buffers_[OrderedPrbsPusch] =
                std::make_unique<memory::GpinnedBuffer>(gdr_handle_, sizeof(std::uint32_t));
        *static_cast<std::uint32_t *>(gdr_buffers_[OrderedPrbsPusch]->get_host_addr()) = 0;

        // Buffer 7: rx_packets_dropped (uint32_t counter)
        gdr_buffers_[RxPacketsDropped] =
                std::make_unique<memory::GpinnedBuffer>(gdr_handle_, sizeof(std::uint32_t));
        *static_cast<std::uint32_t *>(gdr_buffers_[RxPacketsDropped]->get_host_addr()) = 0;

        // Buffer 8: ordered_prbs_prach (uint32_t PRB count)
        gdr_buffers_[OrderedPrbsPrach] =
                std::make_unique<memory::GpinnedBuffer>(gdr_handle_, sizeof(std::uint32_t));
        *static_cast<std::uint32_t *>(gdr_buffers_[OrderedPrbsPrach]->get_host_addr()) = 0;

        // Buffer 9: ordered_prbs_srs (uint32_t PRB count)
        gdr_buffers_[OrderedPrbsSrs] =
                std::make_unique<memory::GpinnedBuffer>(gdr_handle_, sizeof(std::uint32_t));
        *static_cast<std::uint32_t *>(gdr_buffers_[OrderedPrbsSrs]->get_host_addr()) = 0;

        // Buffer 10: next_slot_early_rx_packets (uint32_t counter)
        gdr_buffers_[NextSlotEarlyRxPackets] =
                std::make_unique<memory::GpinnedBuffer>(gdr_handle_, sizeof(std::uint32_t));
        *static_cast<std::uint32_t *>(gdr_buffers_[NextSlotEarlyRxPackets]->get_host_addr()) = 0;

        // Buffer 11: next_slot_on_time_rx_packets (uint32_t counter)
        gdr_buffers_[NextSlotOnTimeRxPackets] =
                std::make_unique<memory::GpinnedBuffer>(gdr_handle_, sizeof(std::uint32_t));
        *static_cast<std::uint32_t *>(gdr_buffers_[NextSlotOnTimeRxPackets]->get_host_addr()) = 0;

        // Buffer 12: next_slot_late_rx_packets (uint32_t counter)
        gdr_buffers_[NextSlotLateRxPackets] =
                std::make_unique<memory::GpinnedBuffer>(gdr_handle_, sizeof(std::uint32_t));
        *static_cast<std::uint32_t *>(gdr_buffers_[NextSlotLateRxPackets]->get_host_addr()) = 0;

        // Buffer 13: order_kernel_last_timeout_error_time (uint64_t timestamp)
        gdr_buffers_[LastTimeoutErrorTime] =
                std::make_unique<memory::GpinnedBuffer>(gdr_handle_, sizeof(std::uint64_t));
        *static_cast<std::uint64_t *>(gdr_buffers_[LastTimeoutErrorTime]->get_host_addr()) = 0;

        // Buffer 14: sym_ord_done_sig_arr (14 × uint32_t symbol completion signals)
        gdr_buffers_[SymOrdDoneSigArr] = std::make_unique<memory::GpinnedBuffer>(
                gdr_handle_, ORAN_PUSCH_SYMBOLS_X_SLOT * sizeof(std::uint32_t));
        std::memset(
                gdr_buffers_[SymOrdDoneSigArr]->get_host_addr(),
                0,
                ORAN_PUSCH_SYMBOLS_X_SLOT * sizeof(std::uint32_t));

        // Buffer 15: sym_ord_done_mask_arr (14 × uint32_t symbol completion masks)
        gdr_buffers_[SymOrdDoneMaskArr] = std::make_unique<memory::GpinnedBuffer>(
                gdr_handle_, ORAN_PUSCH_SYMBOLS_X_SLOT * sizeof(std::uint32_t));
        std::memset(
                gdr_buffers_[SymOrdDoneMaskArr]->get_host_addr(),
                0,
                ORAN_PUSCH_SYMBOLS_X_SLOT * sizeof(std::uint32_t));

        // Buffer 16: pusch_prb_symbol_map_d (14 × uint32_t PRB-to-symbol mapping)
        gdr_buffers_[PuschPrbSymbolMap] = std::make_unique<memory::GpinnedBuffer>(
                gdr_handle_, ORAN_PUSCH_SYMBOLS_X_SLOT * sizeof(std::uint32_t));
        std::memset(
                gdr_buffers_[PuschPrbSymbolMap]->get_host_addr(),
                0,
                ORAN_PUSCH_SYMBOLS_X_SLOT * sizeof(std::uint32_t));

        // Buffer 17: num_order_cells_sym_mask_arr (14 × uint32_t cell masks per symbol)
        gdr_buffers_[NumOrderCellsSymMask] = std::make_unique<memory::GpinnedBuffer>(
                gdr_handle_, ORAN_PUSCH_SYMBOLS_X_SLOT * sizeof(std::uint32_t));
        std::memset(
                gdr_buffers_[NumOrderCellsSymMask]->get_host_addr(),
                0,
                ORAN_PUSCH_SYMBOLS_X_SLOT * sizeof(std::uint32_t));

        RT_LOGC_DEBUG(
                FronthaulKernels::OrderMemory,
                "OrderKernelModule '{}': All {} GDRCopy buffers allocated and initialized",
                instance_id_,
                static_cast<std::size_t>(NumGdrBuffers));

    } catch (const std::exception &e) {
        const std::string error_msg = std::format(
                "OrderKernelModule '{}': Failed to allocate GDRCopy buffers: {}",
                instance_id_,
                e.what());
        RT_LOGC_ERROR(FronthaulKernels::OrderMemory, "{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    // ========================================================================
    // Allocate device memory for descriptors and buffers
    // ========================================================================

    RT_LOGC_DEBUG(
            FronthaulKernels::OrderMemory,
            "OrderKernelModule '{}': Allocating device memory for descriptors and buffers",
            instance_id_);

    // Allocate and initialize PUSCH buffer on device using RAII
    // PUSCH buffers will not change across multiple iterations, so we can allocate them once in
    // setup_memory().
    d_pusch_buffer_ = memory::make_unique_device<std::uint8_t>(PUSCH_SIZE_BYTES);
    FRAMEWORK_CUDA_RUNTIME_CHECK_THROW(cudaMemset(d_pusch_buffer_.get(), 0, PUSCH_SIZE_BYTES));

    // Allocate and initialize last_sem_idx_rx semaphore index using RAII
    d_last_sem_idx_rx_ = memory::make_unique_device<std::uint32_t>(1);
    FRAMEWORK_CUDA_RUNTIME_CHECK_THROW(
            cudaMemset(d_last_sem_idx_rx_.get(), 0, sizeof(std::uint32_t)));

    // Allocate and initialize last_sem_idx_order semaphore index using RAII
    d_last_sem_idx_order_ = memory::make_unique_device<std::uint32_t>(1);
    FRAMEWORK_CUDA_RUNTIME_CHECK_THROW(
            cudaMemset(d_last_sem_idx_order_.get(), 0, sizeof(std::uint32_t)));

    // ========================================================================
    // Create kernel descriptor accessor (framework pattern)
    // ========================================================================

    RT_LOGC_DEBUG(
            FronthaulKernels::OrderMemory,
            "OrderKernelModule '{}': Creating kernel descriptor accessor",
            instance_id_);

    // Create kernel descriptor accessor from memory slice
    kernel_desc_mgr_ = std::make_unique<pipeline::KernelDescriptorAccessor>(memory_slice);

    // Create static kernel parameters in pinned memory using placement new
    h_static_desc_ = &kernel_desc_mgr_->create_static_param<OrderKernelStaticDescriptor>(0);

    // Create dynamic kernel parameters in pinned memory
    h_dynamic_desc_ = &kernel_desc_mgr_->create_dynamic_param<OrderKernelDynamicDescriptor>(0);

    // Get device pointers for kernel parameters
    d_static_desc_ = kernel_desc_mgr_->get_static_device_ptr<OrderKernelStaticDescriptor>(0);
    d_dynamic_desc_ = kernel_desc_mgr_->get_dynamic_device_ptr<OrderKernelDynamicDescriptor>(0);

    RT_LOGC_DEBUG(
            FronthaulKernels::OrderMemory,
            "OrderKernelModule '{}': Kernel descriptors initialized - "
            "h_static={}, d_static={}, h_dynamic={}, d_dynamic={}",
            instance_id_,
            static_cast<const void *>(h_static_desc_),
            static_cast<const void *>(d_static_desc_),
            static_cast<const void *>(h_dynamic_desc_),
            static_cast<const void *>(d_dynamic_desc_));

    // ========================================================================
    // Populate static descriptor (GDRCopy + device buffers + DOCA handles)
    // ========================================================================
    // Note: DOCA handles are now available from constructor (StaticParams), so we can
    // populate everything in setup_memory() without waiting for set_inputs()

    RT_LOGC_DEBUG(
            FronthaulKernels::OrderMemory,
            "OrderKernelModule '{}': Populating static descriptor with GDRCopy, device pointers, "
            "and DOCA handles",
            instance_id_);

    // Single-cell configuration (ORDER_KERNEL_MAX_CELLS_PER_SLOT = 1)
    static constexpr std::size_t CELL_IDX = 0;

    // Control signals (GDRCopy-backed)
    h_static_desc_->order_kernel_exit_cond_d[CELL_IDX] =
            static_cast<std::uint32_t *>(gdr_buffers_[ExitCond]->get_device_addr());
    h_static_desc_->start_cuphy_d[CELL_IDX] =
            static_cast<std::uint32_t *>(gdr_buffers_[StartCuphy]->get_device_addr());

    // PUSCH configuration
    h_static_desc_->pusch_buffer[CELL_IDX] = d_pusch_buffer_.get();
    h_static_desc_->pusch_e_ax_c_map[CELL_IDX] =
            static_cast<std::uint16_t *>(gdr_buffers_[EaxcMap]->get_device_addr());
    h_static_desc_->pusch_ordered_prbs[CELL_IDX] =
            static_cast<std::uint32_t *>(gdr_buffers_[OrderedPrbsPusch]->get_device_addr());

    // PUSCH resource configuration (273 PRBs for 100MHz bandwidth, 4 antenna ports, 14
    // symbols/slot)
    // pusch_prb_x_slot counts total PRB sections across all antennas and symbols in a slot
    h_static_desc_->pusch_prb_x_slot[CELL_IDX] =
            PUSCH_NUM_PRB * NUM_ANTENNA_PORTS * ORAN_PUSCH_SYMBOLS_X_SLOT;
    h_static_desc_->pusch_prb_x_symbol[CELL_IDX] = PUSCH_NUM_PRB;
    h_static_desc_->pusch_prb_x_symbol_x_antenna[CELL_IDX] = PUSCH_NUM_PRB;
    h_static_desc_->pusch_prb_stride[CELL_IDX] = PUSCH_NUM_PRB;
    h_static_desc_->pusch_e_ax_c_num[CELL_IDX] = NUM_ANTENNA_PORTS;

    RT_LOGC_DEBUG(
            FronthaulKernels::OrderMemory,
            "OrderKernelModule '{}': PUSCH configured - PRBs/symbol={}, antenna_ports={}, "
            "symbols/slot={}, total_PRBs/slot={}",
            instance_id_,
            PUSCH_NUM_PRB,
            NUM_ANTENNA_PORTS,
            ORAN_PUSCH_SYMBOLS_X_SLOT,
            PUSCH_NUM_PRB * NUM_ANTENNA_PORTS * ORAN_PUSCH_SYMBOLS_X_SLOT);

    // Compression parameters for U-plane (from RU emulator config)
    // comp_meth: 1 = Block Floating Point (BFP)
    // bit_width: 9 bits per IQ sample (RU emulator should be configured to match)
    // beta: Calculated scaling factor for BFP decompression
    // ru_type: 3 = OTHER_MODE (standard O-RU behavior)
    static constexpr int COMP_METHOD_BFP = 1;
    static constexpr int BIT_WIDTH_9 = 9;
    static constexpr int RU_TYPE_OTHER = 3;

    // BFP scaling parameters (standard configuration)
    static constexpr int EXPONENT_UL = 4;
    static constexpr int FS_OFFSET_UL = 0;
    static constexpr float MAX_AMP_UL = 65504.0F; // Max FP16 value

    // Calculate beta for BFP decompression:
    // sqrt_fs0 = 2^(ul_bit_width-1) * 2^(2^exponent_ul - 1)
    // fs = sqrt_fs0^2 * 2^(-fs_offset_ul)
    // beta_ul = max_amp_ul / sqrt(fs)
    const float sqrt_fs0 = ::powf(2.0F, static_cast<float>(BIT_WIDTH_9 - 1)) *
                           ::powf(2.0F, ::powf(2.0F, static_cast<float>(EXPONENT_UL)) - 1.0F);
    const float fs = sqrt_fs0 * sqrt_fs0 * ::powf(2.0F, -1.0F * static_cast<float>(FS_OFFSET_UL));
    const float beta_ul = MAX_AMP_UL / ::sqrtf(fs);

    h_static_desc_->comp_meth[CELL_IDX] = COMP_METHOD_BFP;
    h_static_desc_->bit_width[CELL_IDX] = BIT_WIDTH_9;
    h_static_desc_->beta[CELL_IDX] = beta_ul;
    h_static_desc_->ru_type[CELL_IDX] = RU_TYPE_OTHER;

    RT_LOGC_DEBUG(
            FronthaulKernels::OrderMemory,
            "OrderKernelModule '{}': Compression configured - method={} (BFP), bit_width={}, "
            "beta={:.10f}, ru_type={}",
            instance_id_,
            COMP_METHOD_BFP,
            BIT_WIDTH_9,
            beta_ul,
            RU_TYPE_OTHER);

    // PRACH configuration
    h_static_desc_->prach_ordered_prbs[CELL_IDX] =
            static_cast<std::uint32_t *>(gdr_buffers_[OrderedPrbsPrach]->get_device_addr());

    // PRACH resource configuration (disabled - set to 0)
    h_static_desc_->prach_e_ax_c_map[CELL_IDX] = nullptr;
    h_static_desc_->prach_e_ax_c_num[CELL_IDX] = 0;
    h_static_desc_->prach_buffer_0[CELL_IDX] = nullptr;
    h_static_desc_->prach_buffer_1[CELL_IDX] = nullptr;
    h_static_desc_->prach_buffer_2[CELL_IDX] = nullptr;
    h_static_desc_->prach_buffer_3[CELL_IDX] = nullptr;
    h_static_desc_->prach_prb_x_slot[CELL_IDX] = 0;
    h_static_desc_->prach_prb_x_symbol[CELL_IDX] = 0;
    h_static_desc_->prach_prb_x_symbol_x_antenna[CELL_IDX] = 0;
    h_static_desc_->prach_prb_stride[CELL_IDX] = 0;

    // SRS configuration
    h_static_desc_->srs_ordered_prbs[CELL_IDX] =
            static_cast<std::uint32_t *>(gdr_buffers_[OrderedPrbsSrs]->get_device_addr());

    // SRS resource configuration (disabled - set to 0)
    h_static_desc_->srs_e_ax_c_map[CELL_IDX] = nullptr;
    h_static_desc_->srs_e_ax_c_num[CELL_IDX] = 0;
    h_static_desc_->srs_buffer[CELL_IDX] = nullptr;
    h_static_desc_->srs_prb_x_slot[CELL_IDX] = 0;
    h_static_desc_->srs_prb_stride[CELL_IDX] = 0;
    h_static_desc_->srs_start_sym[CELL_IDX] = 0;

    // Sub-slot processing / symbol ordering arrays (GDRCopy-backed)
    h_static_desc_->sym_ord_done_sig_arr =
            static_cast<std::uint32_t *>(gdr_buffers_[SymOrdDoneSigArr]->get_device_addr());
    h_static_desc_->sym_ord_done_mask_arr =
            static_cast<std::uint32_t *>(gdr_buffers_[SymOrdDoneMaskArr]->get_device_addr());
    h_static_desc_->pusch_prb_symbol_map_d =
            static_cast<std::uint32_t *>(gdr_buffers_[PuschPrbSymbolMap]->get_device_addr());
    h_static_desc_->num_order_cells_sym_mask_arr =
            static_cast<std::uint32_t *>(gdr_buffers_[NumOrderCellsSymMask]->get_device_addr());

    // Semaphore indices (device memory)
    h_static_desc_->last_sem_idx_rx_h[CELL_IDX] = d_last_sem_idx_rx_.get();
    h_static_desc_->last_sem_idx_order_h[CELL_IDX] = d_last_sem_idx_order_.get();

    // Timing parameters (static configuration from system parameters)
    h_static_desc_->ta4_min_ns[CELL_IDX] = timing_params_.ta4_min_ns;
    h_static_desc_->ta4_max_ns[CELL_IDX] = timing_params_.ta4_max_ns;
    h_static_desc_->slot_duration[CELL_IDX] = timing_params_.slot_duration_ns;

    // Timeout configuration (static configuration, using defaults)
    h_static_desc_->timeout_no_pkt_ns = DEFAULT_TIMEOUT_NO_PKT_NS;
    h_static_desc_->timeout_first_pkt_ns = DEFAULT_TIMEOUT_FIRST_PKT_NS;
    h_static_desc_->timeout_log_interval_ns = DEFAULT_TIMEOUT_LOG_INTERVAL_NS;
    h_static_desc_->timeout_log_enable = DEFAULT_TIMEOUT_LOG_ENABLE;
    h_static_desc_->max_rx_pkts = DEFAULT_MAX_RX_PKTS;
    h_static_desc_->rx_pkts_timeout_ns = DEFAULT_RX_PKTS_TIMEOUT_NS;

    // DOCA GPU handles (infrastructure objects from constructor)
    h_static_desc_->rxq_info_gpu[CELL_IDX] = doca_rxq_params_->eth_rxq_gpu;
    h_static_desc_->sem_gpu[CELL_IDX] = doca_rxq_params_->sem_gpu;
    h_static_desc_->sem_order_num[CELL_IDX] = doca_rxq_params_->sem_items.num_items;

    // Initialize dynamic descriptor with GDRCopy packet statistics pointers
    h_dynamic_desc_->early_rx_packets[CELL_IDX] =
            static_cast<std::uint32_t *>(gdr_buffers_[EarlyRxPackets]->get_device_addr());
    h_dynamic_desc_->on_time_rx_packets[CELL_IDX] =
            static_cast<std::uint32_t *>(gdr_buffers_[OnTimeRxPackets]->get_device_addr());
    h_dynamic_desc_->late_rx_packets[CELL_IDX] =
            static_cast<std::uint32_t *>(gdr_buffers_[LateRxPackets]->get_device_addr());
    h_dynamic_desc_->next_slot_early_rx_packets[CELL_IDX] =
            static_cast<std::uint32_t *>(gdr_buffers_[NextSlotEarlyRxPackets]->get_device_addr());
    h_dynamic_desc_->next_slot_on_time_rx_packets[CELL_IDX] =
            static_cast<std::uint32_t *>(gdr_buffers_[NextSlotOnTimeRxPackets]->get_device_addr());
    h_dynamic_desc_->next_slot_late_rx_packets[CELL_IDX] =
            static_cast<std::uint32_t *>(gdr_buffers_[NextSlotLateRxPackets]->get_device_addr());
    h_dynamic_desc_->rx_packets_dropped_count[CELL_IDX] =
            static_cast<std::uint32_t *>(gdr_buffers_[RxPacketsDropped]->get_device_addr());
    h_dynamic_desc_->order_kernel_last_timeout_error_time[CELL_IDX] =
            static_cast<std::uint64_t *>(gdr_buffers_[LastTimeoutErrorTime]->get_device_addr());

    // Configure kernel launch parameters (used for both stream and graph mode)
    RT_LOGC_INFO(
            FronthaulKernels::OrderMemory,
            "OrderKernelModule '{}': Configuring kernel launch parameters",
            instance_id_);

    constexpr int NUM_CELLS = 1;
    constexpr int NUM_THREADS_PER_BLOCK = 320; // ORDER_KERNEL_PINGPONG_NUM_THREADS
    constexpr int NUM_CTAS_PER_SM = 1;

    // Setup kernel function pointer with template parameters:
    // - ok_tb_enable = false (no test bench)
    // - ul_rx_pkt_tracing_level = 0 (no packet tracing)
    // - srs_enable = 0 (PUSCH only, ORDER_KERNEL_PUSCH_ONLY)
    // - NUM_THREADS = 320
    // - NUM_CTAS_PER_SM = 1
    setup_kernel_function(
            kernel_config_,
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
            reinterpret_cast<const void *>(&order_kernel_doca_single_subSlot_pingpong<
                                           false,
                                           0,
                                           0, // ORDER_KERNEL_PUSCH_ONLY
                                           NUM_THREADS_PER_BLOCK,
                                           NUM_CTAS_PER_SM>));

    // Setup kernel dimensions
    setup_kernel_dimensions(
            kernel_config_, dim3(NUM_CELLS, 1, 1), dim3(NUM_THREADS_PER_BLOCK, 1, 1));

    // Setup kernel arguments (descriptor pointers)
    setup_kernel_arguments(kernel_config_, d_static_desc_, d_dynamic_desc_);

    RT_LOGC_INFO(
            FronthaulKernels::OrderMemory,
            "OrderKernelModule '{}': setup_memory() complete - allocated {} GDRCopy buffers, PUSCH "
            "buffer ({}B), semaphore indices, fully populated static/dynamic descriptors, and "
            "configured kernel launch parameters (grid=1, block=320)",
            instance_id_,
            static_cast<std::size_t>(NumGdrBuffers),
            PUSCH_SIZE_BYTES);
}

void OrderKernelModule::set_inputs([[maybe_unused]] std::span<const pipeline::PortInfo> inputs) {
    // OrderKernelModule has no dynamic inputs from other modules
    // DOCA infrastructure objects are provided via StaticParams (constructor), not set_inputs()
    // This is intentionally a no-op: DOCA objects are stable infrastructure with fixed addresses,
    // not inter-module data flow
    RT_LOGC_DEBUG(
            FronthaulKernels::OrderModule,
            "OrderKernelModule '{}': set_inputs() - no-op (DOCA params provided via constructor)",
            instance_id_);
}

void OrderKernelModule::warmup(cudaStream_t stream) {
    // Copy static descriptor to device
    // Static descriptor contains PUSCH buffer pointer and DOCA handles set during setup_memory()
    // This must be done after setup_memory() completes and before first kernel launch

    // Verify buffer pointer before copy
    const void *expected_pusch_ptr = d_pusch_buffer_.get();
    const void *descriptor_pusch_ptr = h_static_desc_->pusch_buffer[0];

    RT_LOGC_INFO(
            FronthaulKernels::OrderModule,
            "OrderKernelModule '{}': warmup() - Copying static descriptor to device",
            instance_id_);
    RT_LOGC_INFO(
            FronthaulKernels::OrderModule,
            "  Expected PUSCH ptr: {}, Descriptor PUSCH ptr: {}, Match: {}",
            expected_pusch_ptr,
            descriptor_pusch_ptr,
            expected_pusch_ptr == descriptor_pusch_ptr ? "YES" : "NO");
    RT_LOGC_INFO(
            FronthaulKernels::OrderModule,
            "  PUSCH config: prb_x_symbol_x_antenna={} (used by kernel)",
            h_static_desc_->pusch_prb_x_symbol_x_antenna[0]);

    if (expected_pusch_ptr != descriptor_pusch_ptr) {
        RT_LOGC_ERROR(
                FronthaulKernels::OrderModule,
                "OrderKernelModule '{}': CRITICAL - PUSCH buffer pointer mismatch!",
                instance_id_);
        throw std::runtime_error("PUSCH buffer pointer mismatch in static descriptor");
    }

    kernel_desc_mgr_->copy_static_descriptors_to_device(stream);

    // Synchronize to ensure static descriptor is on device before first kernel launch
    FRAMEWORK_CUDA_RUNTIME_CHECK_THROW(cudaStreamSynchronize(stream));

    RT_LOGC_INFO(
            FronthaulKernels::OrderModule,
            "OrderKernelModule '{}': warmup() - Static descriptor copied and verified successfully",
            instance_id_);
}

// ============================================================================
// Per-Iteration Configuration
// ============================================================================

void OrderKernelModule::configure_io(
        const pipeline::DynamicParams &params, [[maybe_unused]] cudaStream_t stream) {
    // Extract ORAN slot timing from module-specific parameters
    const auto timing = std::any_cast<ran::oran::OranSlotTiming>(params.module_specific_params);

    // Use timing from FAPI replay (already in ORAN format with 3GPP to ORAN conversion)
    h_dynamic_desc_->frame_id = timing.frame_id;
    h_dynamic_desc_->subframe_id = timing.subframe_id;
    h_dynamic_desc_->slot_id = timing.slot_id;

    // Reset exit condition to ORDER_KERNEL_RUNNING (0) for new kernel launch
    // This is critical: exit_cond persists across kernel launches (GDRCopy buffer)
    // and must be reset before each kernel to prevent immediate exit
    *static_cast<std::uint32_t *>(gdr_buffers_[ExitCond]->get_host_addr()) = 0;

    // Reset ordered PRBs counter for this slot
    // This counter accumulates across kernel execution and must be reset per slot
    *static_cast<std::uint32_t *>(gdr_buffers_[OrderedPrbsPusch]->get_host_addr()) = 0;
    *static_cast<std::uint32_t *>(gdr_buffers_[OrderedPrbsPrach]->get_host_addr()) = 0;
    *static_cast<std::uint32_t *>(gdr_buffers_[OrderedPrbsSrs]->get_host_addr()) = 0;

    // Set per-slot timing (slot start timestamp)
    constexpr std::size_t CELL_IDX = 0; // Single cell index

    h_dynamic_desc_->slot_start[CELL_IDX] = 0;

    RT_LOGC_DEBUG(
            FronthaulKernels::OrderModule,
            "OrderKernelModule '{}': configure_io() - Updated dynamic descriptor: "
            "frame={}, subframe={}, slot={}",
            instance_id_,
            h_dynamic_desc_->frame_id,
            h_dynamic_desc_->subframe_id,
            h_dynamic_desc_->slot_id);

    // Note: Pipeline will call memory_mgr_->copy_all_dynamic_descriptors_to_device(stream)
    // after all modules' configure_io() calls complete
}

std::vector<pipeline::PortInfo> OrderKernelModule::get_outputs() const {
    // Return PUSCH buffer pointer
    RT_LOGC_INFO(
            FronthaulKernels::OrderModule,
            "OrderKernelModule '{}': get_outputs() - returning PUSCH buffer (ptr={}, elements={}, "
            "size={} bytes, kernel_buffer_ptr={})",
            instance_id_,
            static_cast<const void *>(d_pusch_buffer_.get()),
            PUSCH_NUM_ELEMENTS,
            PUSCH_SIZE_BYTES,
            static_cast<const void *>(h_static_desc_->pusch_buffer[0]));

    // Create TensorInfo with element count and populate cached size field
    tensor::TensorInfo pusch_tensor_info(
            tensor::TensorInfo::DataType::TensorR16F, {PUSCH_NUM_ELEMENTS});
    pusch_tensor_info.set_size_bytes(PUSCH_SIZE_BYTES);

    return {pipeline::PortInfo{
            .name = "pusch",
            .tensors = {{
                    .device_ptr = d_pusch_buffer_.get(),
                    .tensor_info = pusch_tensor_info,
            }}}};
}

// ============================================================================
// Execution - Stream Mode
// ============================================================================

void OrderKernelModule::execute(cudaStream_t stream) {

    RT_LOGC_DEBUG(
            FronthaulKernels::OrderKernel,
            "OrderKernelModule '{}': execute() - Launching "
            "order_kernel_doca_single_subSlot_pingpong on stream {}, PUSCH buffer={}",
            instance_id_,
            static_cast<void *>(stream),
            static_cast<const void *>(d_pusch_buffer_.get()));

    // Launch kernel using framework's kernel_config_.launch() method
    const CUresult launch_err = kernel_config_.launch(stream);

    if (launch_err != CUDA_SUCCESS) {
        const char *error_str = nullptr;
        cuGetErrorString(launch_err, &error_str);
        RT_LOGC_ERROR(
                FronthaulKernels::OrderKernel,
                "OrderKernelModule '{}': Kernel launch failed: {}",
                instance_id_,
                error_str != nullptr ? error_str : "unknown error");
        throw std::runtime_error(
                std::string("Kernel launch failed: ") +
                (error_str != nullptr ? error_str : "unknown error"));
    }

    RT_LOGC_DEBUG(
            FronthaulKernels::OrderKernel,
            "OrderKernelModule '{}': Kernel launched successfully",
            instance_id_);
}

// ============================================================================
// Execution - Graph Mode
// ============================================================================

std::span<const CUgraphNode> OrderKernelModule::add_node_to_graph(
        gsl_lite::not_null<pipeline::IGraph *> graph, std::span<const CUgraphNode> deps) {
    RT_LOGC_DEBUG(
            FronthaulKernels::OrderKernel,
            "OrderKernelModule '{}': Adding kernel node to graph with {} dependencies",
            instance_id_,
            deps.size());

    // Add kernel node using kernel params from kernel_config_
    kernel_node_ = graph->add_kernel_node(deps, kernel_config_.get_kernel_params());

    RT_LOGC_DEBUG(
            FronthaulKernels::OrderKernel,
            "OrderKernelModule '{}': Kernel node added: {}",
            instance_id_,
            static_cast<void *>(kernel_node_));

    return {&kernel_node_, 1};
}

void OrderKernelModule::update_graph_node_params(
        CUgraphExec exec, [[maybe_unused]] const pipeline::DynamicParams &params) {
    const auto &kernel_params = kernel_config_.get_kernel_params();
    FRAMEWORK_CUDA_DRIVER_CHECK_THROW(
            cuGraphExecKernelNodeSetParams(exec, kernel_node_, &kernel_params));

    RT_LOGC_DEBUG(
            FronthaulKernels::OrderKernel,
            "OrderKernelModule '{}': Graph node params updated",
            instance_id_);
}

// ============================================================================
// Kernel Results Access
// ============================================================================

OrderKernelModule::OrderKernelResults OrderKernelModule::read_kernel_results() const {
    OrderKernelResults results{};

    // Read exit condition from GDRCopy memory (CPU-visible GPU memory)
    const auto *exit_cond_ptr =
            static_cast<const std::uint32_t *>(gdr_buffers_[ExitCond]->get_host_addr());
    results.exit_condition = *exit_cond_ptr;

    // Read PRB counts from GDRCopy memory
    const auto *pusch_prbs_ptr =
            static_cast<const std::uint32_t *>(gdr_buffers_[OrderedPrbsPusch]->get_host_addr());
    results.pusch_ordered_prbs = *pusch_prbs_ptr;

    const auto *prach_prbs_ptr =
            static_cast<const std::uint32_t *>(gdr_buffers_[OrderedPrbsPrach]->get_host_addr());
    results.prach_ordered_prbs = *prach_prbs_ptr;

    const auto *srs_prbs_ptr =
            static_cast<const std::uint32_t *>(gdr_buffers_[OrderedPrbsSrs]->get_host_addr());
    results.srs_ordered_prbs = *srs_prbs_ptr;

    // Get expected PRBs from kernel configuration (matches actual kernel setup)
    static constexpr std::size_t CELL_IDX = 0;
    results.expected_prbs = h_static_desc_->pusch_prb_x_slot[CELL_IDX];

    return results;
}

} // namespace ran::fronthaul
