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

#ifndef RAN_FRONTHAUL_FRONTHAUL_HPP
#define RAN_FRONTHAUL_FRONTHAUL_HPP

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <system_error>
#include <vector>

#include <gdrapi.h>

#include <cuda_runtime.h>

#include "fronthaul/fronthaul_export.hpp"
#include "fronthaul/order_kernel_pipeline.hpp"
#include "fronthaul/uplane_config.hpp"
#include "log/rt_log_macros.hpp"
#include "memory/gdrcopy_buffer.hpp"
#include "net/doca_rxq.hpp"
#include "net/doca_txq.hpp"
#include "net/dpdk_txq.hpp"
#include "net/dpdk_types.hpp"
#include "net/env.hpp"
#include "net/mempool.hpp"
#include "oran/cplane_types.hpp"
#include "oran/dpdk_buf.hpp"
#include "oran/fapi_to_cplane.hpp"
#include "oran/numerology.hpp"
#include "pipeline/ipipeline_output_provider.hpp"
#include "utils/cuda_stream.hpp"

namespace ran::fronthaul {

/**
 * Fronthaul statistics
 *
 * @note This struct contains a snapshot of statistics at a point in time.
 *       The Fronthaul class maintains atomic counters internally and
 *       get_stats() returns a consistent snapshot.
 */
struct FRONTHAUL_EXPORT FronthaulStats final {
    std::uint64_t requests_sent{};    //!< Total requests sent (one per cell transmission)
    std::uint64_t packets_sent{};     //!< Total packets transmitted
    std::uint64_t send_errors{};      //!< Total send errors encountered
    double avg_packets_per_request{}; //!< Average packets per request
};

/**
 * Packet send time calculation input parameters
 */
struct FRONTHAUL_EXPORT PacketSendTimeParams final {
    std::chrono::nanoseconds t0{};            //!< Time for system frame 0, subframe 0, slot 0
    std::chrono::nanoseconds tai_offset{};    //!< TAI offset for time synchronization
    std::uint64_t absolute_slot{};            //!< Absolute slot number being processed
    std::chrono::nanoseconds slot_period{};   //!< Slot period
    std::uint32_t slot_ahead{};               //!< Number of slots processing is ahead of real-time
    std::chrono::nanoseconds t1a_max_cp_ul{}; //!< T1a max window for uplink C-plane
    std::chrono::nanoseconds actual_start{};  //!< Actual processing start time (current time)
};

/**
 * Packet send time calculation result
 */
struct FRONTHAUL_EXPORT PacketSendTimeResult final {
    std::chrono::nanoseconds expected_start{}; //!< Expected slot start time
    std::chrono::nanoseconds actual_start{};   //!< Actual processing start time
    std::chrono::nanoseconds time_delta{};     //!< Delta between actual and expected
    std::chrono::nanoseconds threshold{};      //!< Timing threshold
    std::chrono::nanoseconds start_tx{};       //!< Calculated packet transmission time
    bool exceeds_threshold{};                  //!< True if delta exceeds threshold
};

/**
 * Calculate packet send time for a slot
 *
 * This function computes when C-plane packets should be transmitted to the NIC,
 * accounting for processing advance time, T1a timing windows, and TAI offset.
 *
 * This is exposed for unit testing purposes.
 *
 * @param[in] params Input parameters for packet send time calculation
 * @return Packet send time calculation results
 */
FRONTHAUL_EXPORT [[nodiscard]] PacketSendTimeResult
calculate_packet_send_time(const PacketSendTimeParams &params);

/**
 * Create packet header template for ORAN C-Plane messages
 *
 * @param[in] src_mac Source MAC address (from NIC)
 * @param[in] dest_mac Destination MAC address
 * @param[in] vlan_tci VLAN tag control information
 * @param[in] enhanced_antenna_carrier Enhanced antenna carrier ID (encodes cell and antenna port)
 * @return Packet header template ready for ORAN flow
 */
FRONTHAUL_EXPORT [[nodiscard]] ran::oran::PacketHeaderTemplate create_packet_header_template(
        const framework::net::MacAddress &src_mac,
        const framework::net::MacAddress &dest_mac,
        std::uint16_t vlan_tci,
        std::uint16_t enhanced_antenna_carrier);

/**
 * Configuration for fronthaul library
 */
struct FRONTHAUL_EXPORT FronthaulConfig final {
    static constexpr std::uint16_t DEFAULT_MTU = 1514; //!< Default MTU size in bytes

    // Network configuration (owns all DPDK/NIC settings for C-Plane)
    framework::net::EnvConfig net_config{}; //!< Network environment configuration

    // Cell configuration (destination MAC addresses and VLANs)
    std::vector<framework::net::MacAddress> cell_dest_macs; //!< Destination MAC addresses per cell
    std::vector<std::uint16_t> cell_vlan_tcis;              //!< VLAN TCI per cell

    // ORAN parameters
    ran::oran::OranNumerology numerology{
            //!< ORAN numerology configuration
            ran::oran::from_scs(ran::oran::SubcarrierSpacing::Scs30Khz)};
    std::uint32_t num_antenna_ports{4}; //!< Number of antenna ports
    std::uint16_t mtu{DEFAULT_MTU};     //!< Maximum transmission unit size

    // Timing parameters for C-plane transmission
    std::uint32_t slot_ahead{1};              //!< Slots to process ahead
    std::uint64_t t1a_max_cp_ul_ns{};         //!< T1a max window for uplink C-plane
    std::uint64_t t1a_min_cp_ul_ns{};         //!< T1a min window for uplink C-plane
    std::uint64_t tx_cell_start_offset_ns{0}; //!< Optional per-cell offset

    // GPS/TAI timing (optional)
    std::int64_t gps_alpha{0}; //!< GPS alpha timing parameter
    std::int64_t gps_beta{0};  //!< GPS beta timing parameter

    // U-Plane configuration
    UPlaneConfig uplane_config{}; //!< U-Plane Order Kernel pipeline configuration
};

/**
 * Fronthaul library main class
 *
 * Manages ORAN fronthaul operations including:
 * - Converting FAPI to ORAN C-Plane messages
 * - Transmitting C-Plane packets via DPDK
 * - Processing U-Plane packets via Order Kernel pipeline
 *
 * This class is stateless - all timing and request data comes from caller.
 * Use send_ul_cplane() to transmit C-Plane messages for each cell/slot.
 * Use process_uplane() to execute the Order Kernel pipeline.
 *
 * Implements IPipelineOutputProvider to expose Order Kernel output addresses
 * for zero-copy integration with downstream pipelines.
 */
class FRONTHAUL_EXPORT Fronthaul final : public framework::pipeline::IPipelineOutputProvider {
public:
    /**
     * Construct fronthaul library
     *
     * Sets up network environment and creates ORAN flows.
     * All setup happens in constructor - ready to use immediately.
     *
     * @param[in] config Fronthaul configuration
     * @throws std::runtime_error if setup fails
     * @throws std::invalid_argument if configuration is invalid
     */
    explicit Fronthaul(const FronthaulConfig &config);

    /**
     * Destructor
     *
     * Cleans up resources including CUDA stream if U-Plane was configured.
     */
    ~Fronthaul() override = default;

    // Non-copyable, non-movable (contains std::atomic members)
    Fronthaul(const Fronthaul &) = delete;
    Fronthaul &operator=(const Fronthaul &) = delete;
    Fronthaul(Fronthaul &&) = delete;
    Fronthaul &operator=(Fronthaul &&) = delete;

    /**
     * Send uplink C-Plane messages for a cell
     *
     * Converts FAPI UL_TTI_REQUEST to ORAN C-Plane messages and transmits them.
     * The request must already have sfn and slot fields updated to match the
     * desired transmission timing.
     *
     * @param[in] request FAPI UL_TTI_REQUEST message with updated timing fields
     * @param[in] body_len Size of FAPI message request body (excluding body header)
     * @param[in] cell_id Cell identifier (index into cell_dest_macs)
     * @param[in] absolute_slot Absolute slot number for timing calculation
     * @param[in] t0 Time for system frame 0, subframe 0, slot 0
     * @param[in] tai_offset TAI offset for time synchronization
     */
    void send_ul_cplane(
            const scf_fapi_ul_tti_req_t &request,
            std::size_t body_len,
            std::uint16_t cell_id,
            std::uint64_t absolute_slot,
            std::chrono::nanoseconds t0,
            std::chrono::nanoseconds tai_offset);

    /**
     * Process U-Plane for the current slot
     *
     * Executes the Order Kernel pipeline to receive and process U-Plane packets.
     * Timing is passed by value to avoid threading issues (U-plane processing is asynchronous).
     *
     * @param[in] timing ORAN slot timing (frame, subframe, slot)
     */
    void process_uplane(ran::oran::OranSlotTiming timing);

    /**
     * Get configuration
     * @return Reference to fronthaul configuration
     */
    [[nodiscard]] const FronthaulConfig &config() const noexcept { return config_; }

    /**
     * Get fronthaul statistics
     * @return Current statistics snapshot
     */
    [[nodiscard]] FronthaulStats get_stats() const noexcept;

    /**
     * Order kernel accumulated statistics across all slots
     */
    struct OrderKernelStatistics final {
        std::uint64_t total_pusch_prbs{};    //!< Accumulated PUSCH PRBs across all slots
        std::uint64_t total_prach_prbs{};    //!< Accumulated PRACH PRBs across all slots
        std::uint64_t total_srs_prbs{};      //!< Accumulated SRS PRBs across all slots
        std::uint64_t total_expected_prbs{}; //!< Accumulated expected PRBs across all slots
        std::uint64_t slots_processed{};     //!< Number of U-Plane slots processed
    };

    /**
     * Reset statistics counters to zero
     *
     * Resets all statistics counters (requests_sent, packets_sent, send_errors) to zero.
     * Useful for test scenarios where fresh statistics are needed.
     */
    void reset_stats() noexcept;

    /**
     * Read accumulated kernel statistics from U-Plane processing
     *
     * Retrieves accumulated PRB counts across all processed slots.
     *
     * @return OrderKernelStatistics with accumulated counts
     * @throws std::runtime_error if U-Plane is not initialized
     */
    [[nodiscard]] OrderKernelStatistics read_kernel_statistics() const;

    /**
     * Get Order Kernel pipeline pointer
     *
     * Provides non-owning access to the Order Kernel pipeline for integration with Driver.
     * The pipeline remains owned by Fronthaul and must not be deleted by the caller.
     *
     * @return Non-owning pointer to OrderKernelPipeline, or nullptr if U-Plane not initialized
     */
    [[nodiscard]] OrderKernelPipeline *get_order_kernel_pipeline() const noexcept {
        return uplane_resources_ ? uplane_resources_->pipeline.get() : nullptr;
    }

    /**
     * Get Order Kernel output addresses
     *
     * Implements IPipelineOutputProvider::get_order_kernel_outputs().
     *
     * Provides access to the stable output buffer addresses captured after Order Kernel
     * warmup. These addresses can be used for zero-copy data passing to downstream pipelines
     * (e.g., PUSCH pipeline).
     *
     * The addresses are captured once after Order Kernel initialization and remain valid
     * throughout the Fronthaul lifetime.
     *
     * @return Span of PortInfo describing Order Kernel outputs, or empty span if U-Plane not
     * initialized
     */
    [[nodiscard]] std::span<const framework::pipeline::PortInfo>
    get_order_kernel_outputs() const noexcept override {
        return uplane_resources_
                       ? std::span<const framework::pipeline::
                                           PortInfo>{uplane_resources_->order_kernel_outputs}
                       : std::span<const framework::pipeline::PortInfo>{};
    }

private:
    /**
     * U-Plane processing resources bundle
     *
     */
    struct UPlaneResources final {
        std::unique_ptr<OrderKernelPipeline> pipeline;
        framework::utils::CudaStream stream;
        framework::memory::UniqueGdrHandle gdr_handle;
        std::uint64_t absolute_slot{0};
        OrderKernelStatistics stats{}; //!< Accumulated kernel statistics

        //! Order Kernel output addresses (captured after warmup, stable)
        std::vector<framework::pipeline::PortInfo> order_kernel_outputs;
    };
    // ========================================================================
    // C-Plane Members
    // ========================================================================

    // Configuration
    FronthaulConfig config_{};

    // Network environment (owns DPDK, NIC, mempools)
    std::unique_ptr<framework::net::Env> net_env_;

    // ORAN flows (one per cell per antenna port) - manages packet headers and
    // sequence IDs Indexed as: oran_flows_[cell_index][antenna_port_index]
    std::vector<std::vector<std::unique_ptr<ran::oran::SimpleOranFlow>>> oran_flows_;

    // ORAN peers (one per cell per antenna port) - manages timestamps
    // Indexed as: oran_peers_[cell_index][antenna_port_index]
    std::vector<std::vector<std::unique_ptr<ran::oran::SimpleOranPeer>>> oran_peers_;

    // Reusable buffers for C-plane message conversion (avoids per-slot
    // allocation)
    ran::oran::PrbChunks prb_chunks_;
    std::vector<ran::oran::OranCPlaneMsgInfo> cplane_infos_;

    // Reusable buffers for packet transmission (avoids per-packet allocation)
    std::vector<rte_mbuf *> mbuf_pool_;
    std::vector<ran::oran::MBuf> mbuf_wrapper_pool_;

    // Statistics
    std::atomic<std::uint64_t> total_packets_sent_{0};
    std::atomic<std::uint64_t> total_requests_sent_{0};
    std::atomic<std::uint64_t> total_send_errors_{0};

    // ========================================================================
    // U-Plane Members
    // ========================================================================

    // U-Plane resources (REQUIRED - unique_ptr for deferred construction only)
    std::unique_ptr<UPlaneResources> uplane_resources_;

    // ========================================================================
    // Private Methods
    // ========================================================================

    // Send C-plane messages for a cell
    std::error_code send_cplane_messages(
            std::span<ran::oran::OranCPlaneMsgInfo> cplane_infos, std::size_t cell_index);

    // Initialize U-Plane pipeline (called from constructor if config provided)
    void initialize_uplane(const UPlaneConfig &uplane_config);
};

} // namespace ran::fronthaul

/// @cond HIDE_FROM_DOXYGEN
// Must be in global namespace for quill to find it
// cppcheck-suppress functionStatic
RT_LOGGABLE_DEFERRED_FORMAT(
        ran::fronthaul::FronthaulStats,
        "Requests sent: {}, Packets sent: {}, Send errors: {}, Avg packets/request: {:.2f}",
        obj.requests_sent,
        obj.packets_sent,
        obj.send_errors,
        obj.avg_packets_per_request)

// cppcheck-suppress functionStatic
RT_LOGGABLE_DEFERRED_FORMAT(
        ran::fronthaul::Fronthaul::OrderKernelStatistics,
        "Slots processed: {}, Total PUSCH PRBs (antennas x symbols x slots): {} (expected: {}), "
        "Total PRACH PRBs: {}, Total SRS PRBs: {}",
        obj.slots_processed,
        obj.total_pusch_prbs,
        obj.total_expected_prbs,
        obj.total_prach_prbs,
        obj.total_srs_prbs)
/// @endcond

#endif // RAN_FRONTHAUL_FRONTHAUL_HPP
