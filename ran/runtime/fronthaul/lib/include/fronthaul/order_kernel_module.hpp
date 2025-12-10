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

#ifndef RAN_FRONTHAUL_ORDER_KERNEL_MODULE_HPP
#define RAN_FRONTHAUL_ORDER_KERNEL_MODULE_HPP

// clang-format off
#include <driver_types.h>
#include <gsl-lite/gsl-lite.hpp>
#include <cuda.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "memory/gdrcopy_buffer.hpp"
#include "pipeline/iallocation_info_provider.hpp"
#include "pipeline/igraph.hpp"
#include "pipeline/igraph_node_provider.hpp"
#include "pipeline/imodule.hpp"
#include "pipeline/istream_executor.hpp"
#include "pipeline/kernel_descriptor_accessor.hpp"
#include "pipeline/kernel_launch_config.hpp"
#include "tensor/tensor_info.hpp"
#include "pipeline/types.hpp"
#include "memory/unique_ptr_utils.hpp"
#include "fronthaul/order_kernel_descriptors.hpp"
#include "log/rt_log_macros.hpp"

// clang-format on

// Forward declarations
namespace framework::net {
struct DocaRxQParams;
}

namespace ran::fronthaul {

/**
 * OrderKernelModule - ORAN UL Receiver Order Kernel
 *
 * Wraps the order_kernel_doca_single_subSlot_pingpong CUDA kernel for ORAN packet
 * reception, ordering, and decompression.
 *
 * Configuration:
 * - 1 input port: "doca_objects" (DOCA RX queue parameters, zero-copy)
 * - 1 output port: "pusch" (PUSCH IQ data buffer pointer)
 * - Custom CUDA kernel for packet processing on GPU
 * - GDRCopy memory for NIC↔GPU direct access
 */
class OrderKernelModule final : public framework::pipeline::IModule,
                                public framework::pipeline::IAllocationInfoProvider,
                                public framework::pipeline::IGraphNodeProvider,
                                public framework::pipeline::IStreamExecutor {
public:
    /**
     * Timing window parameters for ORAN packet processing
     *
     * These values correspond to the ORAN timing specification and are typically
     * loaded from YAML configuration (Ta4_min_ns, Ta4_max_ns in cuphycontroller config).
     */
    struct TimingParams final {
        /// Default slot duration for 30kHz SCS (500 microseconds)
        static constexpr std::uint64_t DEFAULT_SLOT_DURATION_NS = 500'000;

        /// Default Ta4 early window (50 microseconds)
        static constexpr std::uint64_t DEFAULT_TA4_MIN_NS = 50'000;

        /// Default Ta4 late window (450 microseconds)
        static constexpr std::uint64_t DEFAULT_TA4_MAX_NS = 450'000;

        std::uint64_t slot_duration_ns{
                DEFAULT_SLOT_DURATION_NS}; //!< Slot duration in nanoseconds (default: 500us for
                                           //!< 30kHz SCS)
        std::uint64_t ta4_min_ns{
                DEFAULT_TA4_MIN_NS}; //!< Ta4 early window - packets before slot_start + ta4_min
                                     //!< are early (default: 50us)
        std::uint64_t ta4_max_ns{
                DEFAULT_TA4_MAX_NS}; //!< Ta4 late window - packets after slot_start + ta4_max are
                                     //!< late (default: 450us)
    };

    /**
     * Static parameters for module construction
     */
    struct StaticParams final {
        framework::pipeline::ExecutionMode execution_mode{
                framework::pipeline::ExecutionMode::Stream}; //!< Pipeline execution mode
                                                             //!< (default: Stream)
        gdr_t gdr_handle{nullptr}; //!< Non-owning GDRCopy handle (gdr_t is already a pointer)
        const framework::net::DocaRxQParams *doca_rxq_params{
                nullptr};      //!< DOCA RX queue parameters (must not be null)
        TimingParams timing{}; //!< ORAN timing windows (Ta4 early/late thresholds, slot duration)
        std::vector<std::uint16_t> eaxc_ids{0, 1, 2, 3}; //!< UL eAxC IDs for antenna ports
    };

    /**
     * Construct module with instance ID and parameters
     *
     * @param[in] instance_id Unique identifier for this module instance
     * @param[in] params Static configuration parameters
     * @pre params.gdr_handle must not be null
     * @pre params.doca_rxq_params must not be null
     * @throws gsl::fail_fast if preconditions are violated
     */
    explicit OrderKernelModule(std::string instance_id, const StaticParams &params);

    ~OrderKernelModule() override = default;

    // Non-copyable, non-movable
    OrderKernelModule(const OrderKernelModule &) = delete;
    OrderKernelModule &operator=(const OrderKernelModule &) = delete;
    OrderKernelModule(OrderKernelModule &&) = delete;
    OrderKernelModule &operator=(OrderKernelModule &&) = delete;

    // ========================================================================
    // IModule Interface - Identification
    // ========================================================================

    /**
     * Get module type identifier
     *
     * @return Module type string "order_kernel_module"
     */
    [[nodiscard]] std::string_view get_type_id() const override { return "order_kernel_module"; }

    /**
     * Get module instance identifier
     *
     * @return Instance ID provided at construction
     */
    [[nodiscard]] std::string_view get_instance_id() const override { return instance_id_; }

    /**
     * Get stream executor interface
     *
     * @return Pointer to IStreamExecutor interface (this module)
     */
    [[nodiscard]] framework::pipeline::IStreamExecutor *as_stream_executor() override;

    /**
     * Get graph node provider interface
     *
     * @return Pointer to IGraphNodeProvider interface (this module)
     */
    [[nodiscard]] framework::pipeline::IGraphNodeProvider *as_graph_node_provider() override;

    // ========================================================================
    // IModule Interface - Port Introspection
    // ========================================================================

    /**
     * Get input port names
     *
     * @return Vector containing "doca_objects"
     */
    [[nodiscard]] std::vector<std::string> get_input_port_names() const override;

    /**
     * Get output port names
     *
     * @return Vector containing "pusch"
     */
    [[nodiscard]] std::vector<std::string> get_output_port_names() const override;

    /**
     * Get input tensor information for specified port
     *
     * @param[in] port_name Input port name
     * @return Vector of tensor info for the port
     */
    [[nodiscard]] std::vector<framework::tensor::TensorInfo>
    get_input_tensor_info(std::string_view port_name) const override;

    /**
     * Get output tensor information for specified port
     *
     * @param[in] port_name Output port name
     * @return Vector of tensor info for the port
     */
    [[nodiscard]] std::vector<framework::tensor::TensorInfo>
    get_output_tensor_info(std::string_view port_name) const override;

    // ========================================================================
    // IModule & IAllocationInfoProvider - Memory Configuration
    // ========================================================================

    /**
     * Get input memory characteristics for specified port
     *
     * @param[in] port_name Input port name
     * @return Memory characteristics for the port
     */
    [[nodiscard]] framework::pipeline::InputPortMemoryCharacteristics
    get_input_memory_characteristics(std::string_view port_name) const override;

    /**
     * Get output memory characteristics for specified port
     *
     * @param[in] port_name Output port name
     * @return Memory characteristics for the port
     */
    [[nodiscard]] framework::pipeline::OutputPortMemoryCharacteristics
    get_output_memory_characteristics(std::string_view port_name) const override;

    /**
     * Get module memory requirements
     *
     * @return Memory requirements for descriptors and buffers
     */
    [[nodiscard]] framework::pipeline::ModuleMemoryRequirements get_requirements() const override;

    // ========================================================================
    // IModule Interface - Setup Phase
    // ========================================================================

    /**
     * Allocate and initialize module memory
     *
     * @param[in] memory_slice Memory slice allocated by framework
     */
    void setup_memory(const framework::pipeline::ModuleMemorySlice &memory_slice) override;

    /**
     * Configure input port connections.
     *
     * @param[in] inputs Input port information containing DOCA objects
     */
    void set_inputs(std::span<const framework::pipeline::PortInfo> inputs) override;

    /**
     * Perform warmup operations
     *
     * @param[in] stream CUDA stream for warmup operations
     */
    void warmup(cudaStream_t stream) override;

    // ========================================================================
    // IModule Interface - Per-Iteration Configuration
    // ========================================================================

    /**
     * Configure I/O for current iteration
     *
     * @param[in] params Dynamic parameters for this iteration
     * @param[in] stream CUDA stream for async operations during configuration
     */
    void
    configure_io(const framework::pipeline::DynamicParams &params, cudaStream_t stream) override;

    /**
     * Get output port information
     *
     * @return Vector of output port info
     */
    [[nodiscard]] std::vector<framework::pipeline::PortInfo> get_outputs() const override;

    // ========================================================================
    // IStreamExecutor Interface - Stream Mode Execution
    // ========================================================================

    /**
     * Execute kernel in stream mode
     *
     * @param[in] stream CUDA stream for kernel execution
     */
    void execute(cudaStream_t stream) override;

    // ========================================================================
    // IGraphNodeProvider Interface - Graph Mode Execution
    // ========================================================================

    /**
     * Add order kernel node to CUDA graph.
     *
     * @param[in] graph Graph interface for node creation
     * @param[in] deps Dependency nodes that must complete before this node
     * @return Span of created graph node handle (single order kernel node)
     */
    [[nodiscard]] std::span<const CUgraphNode> add_node_to_graph(
            gsl_lite::not_null<framework::pipeline::IGraph *> graph,
            std::span<const CUgraphNode> deps) override;

    /**
     * Update graph node parameters
     *
     * @param[in] exec CUDA graph executable handle
     * @param[in] params Dynamic parameters for update
     */
    void update_graph_node_params(
            CUgraphExec exec, const framework::pipeline::DynamicParams &params) override;

    // ========================================================================
    // OrderKernelModule Specific Interface - Kernel Results Access
    // ========================================================================

    /**
     * Kernel execution results structure
     */
    struct OrderKernelResults final {
        std::uint32_t exit_condition{};     //!< Kernel exit condition code
        std::uint32_t pusch_ordered_prbs{}; //!< Number of PUSCH PRBs processed
        std::uint32_t prach_ordered_prbs{}; //!< Number of PRACH PRBs processed
        std::uint32_t srs_ordered_prbs{};   //!< Number of SRS PRBs processed
        std::uint32_t expected_prbs{};      //!< Expected total PRBs for this slot
    };

    /**
     * Read kernel execution results from GDRCopy memory
     *
     * This method reads the kernel results directly from CPU-visible GPU memory
     * without requiring GPU synchronization. Should be called after kernel
     * execution completes.
     *
     * @return OrderKernelResults structure with current values
     */
    [[nodiscard]] OrderKernelResults read_kernel_results() const;

private:
    // Module identification
    std::string instance_id_; //!< Module instance identifier
    framework::pipeline::ExecutionMode
            execution_mode_;     //!< Pipeline execution mode (Stream or Graph)
    TimingParams timing_params_; //!< ORAN timing windows (Ta4 early/late thresholds, slot duration)
    std::vector<std::uint16_t> eaxc_ids_;              //!< UL eAxC IDs for antenna ports
    framework::pipeline::ModuleMemorySlice mem_slice_; //!< Assigned memory slice

    // GDRCopy handle (non-owning)
    gdr_t gdr_handle_{nullptr}; //!< Non-owning GDRCopy handle (gdr_t is already a pointer)

    // GDRCopy buffers (CPU-visible GPU memory for NIC direct access)
    // Indices match order in test_oran_order_kernel.cpp lines 238-265
    enum GdrBufferIndex : std::size_t {
        ExitCond = 0,                 //!< exit_cond: uint32_t kernel control flag
        StartCuphy = 1,               //!< start_cuphy: uint32_t processing flag
        EarlyRxPackets = 2,           //!< early_rx_packets: uint32_t counter
        OnTimeRxPackets = 3,          //!< on_time_rx_packets: uint32_t counter
        LateRxPackets = 4,            //!< late_rx_packets: uint32_t counter
        EaxcMap = 5,                  //!< eAxC_map: 4 × uint16_t array mapping
        OrderedPrbsPusch = 6,         //!< ordered_prbs_pusch: uint32_t PRB count
        RxPacketsDropped = 7,         //!< rx_packets_dropped: uint32_t counter
        OrderedPrbsPrach = 8,         //!< ordered_prbs_prach: uint32_t PRB count
        OrderedPrbsSrs = 9,           //!< ordered_prbs_srs: uint32_t PRB count
        NextSlotEarlyRxPackets = 10,  //!< next_slot_early_rx_packets: uint32_t counter
        NextSlotOnTimeRxPackets = 11, //!< next_slot_on_time_rx_packets: uint32_t counter
        NextSlotLateRxPackets = 12,   //!< next_slot_late_rx_packets: uint32_t counter
        LastTimeoutErrorTime = 13,    //!< order_kernel_last_timeout_error_time: uint64_t timestamp
        SymOrdDoneSigArr = 14,  //!< sym_ord_done_sig_arr: 14 × uint32_t symbol completion signals
        SymOrdDoneMaskArr = 15, //!< sym_ord_done_mask_arr: 14 × uint32_t symbol completion masks
        PuschPrbSymbolMap = 16, //!< pusch_prb_symbol_map_d: 14 × uint32_t PRB-to-symbol mapping
        NumOrderCellsSymMask =
                17,        //!< num_order_cells_sym_mask_arr: 14 × uint32_t cell masks per symbol
        NumGdrBuffers = 18 //!< Total count of GDR buffers
    };
    std::array<std::unique_ptr<framework::memory::GpinnedBuffer>, NumGdrBuffers> gdr_buffers_;

    // Device memory pointers (GPU-only memory)
    // PUSCH buffers will not change across multiple iterations, so we can allocate them once in
    // setup_memory(). Using UniqueDevicePtr for automatic RAII cleanup.
    framework::memory::UniqueDevicePtr<std::uint8_t> d_pusch_buffer_; //!< Device PUSCH IQ buffer
    // These 2 are real inputs to the kernel.
    framework::memory::UniqueDevicePtr<std::uint32_t>
            d_last_sem_idx_rx_; //!< Device semaphore index (RX)
    framework::memory::UniqueDevicePtr<std::uint32_t>
            d_last_sem_idx_order_; //!< Device semaphore index (Order)

    // Kernel descriptor management (framework pattern)
    std::unique_ptr<framework::pipeline::KernelDescriptorAccessor>
            kernel_desc_mgr_;                             //!< Kernel descriptor accessor
    OrderKernelStaticDescriptor *h_static_desc_{nullptr}; //!< Static params in pinned memory (CPU)
    OrderKernelDynamicDescriptor *h_dynamic_desc_{
            nullptr};                                     //!< Dynamic params in pinned memory (CPU)
    OrderKernelStaticDescriptor *d_static_desc_{nullptr}; //!< Static params on device (GPU)
    OrderKernelDynamicDescriptor *d_dynamic_desc_{nullptr}; //!< Dynamic params on device (GPU)

    // Input data pointers (set by set_inputs, configured once)
    const framework::net::DocaRxQParams *doca_rxq_params_{
            nullptr}; //!< DOCA RX queue parameters pointer

    // CUDA graph execution state
    CUgraphNode kernel_node_{
            nullptr}; //!< CUDA kernel graph node handle (used in add_node_to_graph)
    // DUAL means 2 parameters: static and dynamic params
    framework::pipeline::DualKernelLaunchConfig kernel_config_; //!< Kernel launch configuration
};

} // namespace ran::fronthaul

/// @cond HIDE_FROM_DOXYGEN
/**
 * Enable logging support for OrderKernelResults struct
 *
 * Registers OrderKernelResults with the logging framework to enable
 * direct logging of kernel execution results.
 *
 * Note: This must be in the global namespace so Quill's formatter can detect it.
 */
// cppcheck-suppress functionStatic
RT_LOGGABLE_DEFERRED_FORMAT(
        ran::fronthaul::OrderKernelModule::OrderKernelResults,
        "PUSCH={}/{}, PRACH={}, SRS={}, Exit={}",
        obj.pusch_ordered_prbs,
        obj.expected_prbs,
        obj.prach_ordered_prbs,
        obj.srs_ordered_prbs,
        obj.exit_condition)
/// @endcond

#endif // RAN_FRONTHAUL_ORDER_KERNEL_MODULE_HPP
