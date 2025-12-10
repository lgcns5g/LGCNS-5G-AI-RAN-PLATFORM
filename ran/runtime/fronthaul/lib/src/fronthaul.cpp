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
#include <any>
#include <atomic>
#include <chrono>
#include <compare>
#include <cstdint>
#include <cstring>
#include <exception>
#include <format>
#include <limits>
#include <memory>
#include <ratio>
#include <span>
#include <stdexcept>
#include <system_error>
#include <utility>
#include <vector>

#include <aerial-fh-driver/oran.hpp>
#include <gdrapi.h>
#include <quill/LogMacros.h>
#include <rte_ether.h>
#include <rte_mbuf.h>
#include <rte_mbuf_core.h>
#include <scf_5g_fapi.h>

#include <gsl-lite/gsl-lite.hpp>

#include <cuda_runtime.h>

#include "fronthaul/fronthaul.hpp"
#include "fronthaul/fronthaul_log.hpp"
#include "fronthaul/order_kernel_factories.hpp"
#include "fronthaul/order_kernel_module.hpp"
#include "fronthaul/order_kernel_pipeline.hpp"
#include "fronthaul/uplane_config.hpp"
#include "log/rt_log_macros.hpp"
#include "memory/gdrcopy_buffer.hpp"
#include "net/details/dpdk_utils.hpp"
#include "net/doca_rxq.hpp"
#include "net/doca_types.hpp"
#include "net/dpdk_types.hpp"
#include "net/env.hpp"
#include "net/gpu.hpp"
#include "net/mempool.hpp"
#include "net/nic.hpp"
#include "oran/cplane_message.hpp"
#include "oran/cplane_types.hpp"
#include "oran/cplane_utils.hpp"
#include "oran/dpdk_buf.hpp"
#include "oran/fapi_to_cplane.hpp"
#include "oran/numerology.hpp"
#include "pipeline/types.hpp"
#include "task/time.hpp"
#include "tensor/data_types.hpp"
#include "tensor/tensor_info.hpp"
#include "utils/cuda_stream.hpp"
#include "utils/error_macros.hpp"

namespace ran::fronthaul {

ran::oran::PacketHeaderTemplate create_packet_header_template(
        const framework::net::MacAddress &src_mac,
        const framework::net::MacAddress &dest_mac,
        const std::uint16_t vlan_tci,
        const std::uint16_t enhanced_antenna_carrier) {

    ran::oran::PacketHeaderTemplate header{};

    // Ethernet header
    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)
    std::ranges::copy(src_mac.bytes, header.eth.src_addr.addr_bytes);
    std::ranges::copy(dest_mac.bytes, header.eth.dst_addr.addr_bytes);
    // NOLINTEND(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)
    header.eth.ether_type = ran::oran::cpu_to_be_16(RTE_ETHER_TYPE_VLAN);

    // VLAN header
    header.vlan.vlan_tci = ran::oran::cpu_to_be_16(vlan_tci);
    header.vlan.eth_proto = ran::oran::cpu_to_be_16(ETHER_TYPE_ECPRI);

    // eCPRI header template (fixed fields)
    // NOLINTBEGIN(cppcoreguidelines-pro-type-union-access)
    header.ecpri.ecpriVersion = ORAN_DEF_ECPRI_VERSION;
    header.ecpri.ecpriReserved = ORAN_DEF_ECPRI_RESERVED;
    header.ecpri.ecpriConcatenation = ORAN_ECPRI_CONCATENATION_NO;
    header.ecpri.ecpriMessage = ECPRI_MSG_TYPE_RTC; // C-Plane
    header.ecpri.ecpriPcid = ran::oran::cpu_to_be_16(enhanced_antenna_carrier);
    header.ecpri.ecpriEbit = 1;     // End bit (last/only packet)
    header.ecpri.ecpriSubSeqid = 0; // No sub-sequencing
    // NOLINTEND(cppcoreguidelines-pro-type-union-access)

    return header;
}

PacketSendTimeResult calculate_packet_send_time(const PacketSendTimeParams &params) {
    PacketSendTimeResult result{};

    // Calculate expected start time for this slot
    const auto iabsolute_slot = static_cast<std::int64_t>(params.absolute_slot);
    const auto islot_ahead = static_cast<std::int64_t>(params.slot_ahead);
    const auto iabsolute_slot_ahead = iabsolute_slot - islot_ahead;
    result.expected_start = params.t0 + params.slot_period * iabsolute_slot_ahead;
    result.actual_start = params.actual_start;

    // Calculate threshold: (slot_period * slot_ahead) - t1a_max_cp_ul
    result.threshold = params.slot_period * islot_ahead - params.t1a_max_cp_ul;

    // Calculate time delta
    result.time_delta = result.actual_start - result.expected_start;

    // Threshold violation only occurs when processing late (positive delta) AND
    // exceeding threshold
    result.exceeds_threshold = (result.time_delta > std::chrono::nanoseconds{0}) &&
                               (result.time_delta > result.threshold);

    // Calculate transmission time:
    // expected_start: when the slot actually starts
    // + threshold: adds back slot_ahead advance minus t1a window
    // + tai_offset: GPS/TAI synchronization
    result.start_tx = result.expected_start + result.threshold + params.tai_offset;

    return result;
}

// NOLINTNEXTLINE(modernize-pass-by-value)
Fronthaul::Fronthaul(const FronthaulConfig &config) : config_(config) {
    // 1. Validate configuration
    if (config_.cell_dest_macs.empty()) {
        throw std::invalid_argument("At least one cell MAC address required");
    }
    if (config_.cell_vlan_tcis.size() != config_.cell_dest_macs.size()) {
        throw std::invalid_argument(std::format(
                "VLAN TCI count ({}) must match cell MAC count ({})",
                config_.cell_vlan_tcis.size(),
                config_.cell_dest_macs.size()));
    }

    RT_LOGC_INFO(
            FronthaulLog::FronthaulGeneral,
            "Initializing fronthaul: {} cells, {}kHz SCS, slot_period={}ns",
            config_.cell_dest_macs.size(),
            ran::oran::to_khz(config_.numerology.subcarrier_spacing),
            config_.numerology.slot_period_ns);

    // 2. Create network environment (initializes DPDK, NIC, mempools)
    try {
        net_env_ = std::make_unique<framework::net::Env>(config_.net_config);
    } catch (const std::exception &e) {
        throw std::runtime_error(std::format("Failed to create network environment: {}", e.what()));
    }

    RT_LOGC_INFO(
            FronthaulLog::FronthaulNetwork,
            "Network environment initialized: NIC MAC={}",
            net_env_->nic().mac_address().to_string());

    // 3. Create ORAN flows and peers (one per cell per antenna port) with packet
    // header templates
    oran_flows_.reserve(config_.cell_dest_macs.size());
    oran_peers_.reserve(config_.cell_dest_macs.size());
    for (std::size_t cell_idx = 0; cell_idx < config_.cell_dest_macs.size(); ++cell_idx) {
        const auto &dest_mac = config_.cell_dest_macs[cell_idx];

        // Create one flow and peer per antenna port for this cell
        std::vector<std::unique_ptr<ran::oran::SimpleOranFlow>> cell_flows{};
        std::vector<std::unique_ptr<ran::oran::SimpleOranPeer>> cell_peers{};
        cell_flows.reserve(config_.num_antenna_ports);
        cell_peers.reserve(config_.num_antenna_ports);

        for (std::size_t ap_idx = 0; ap_idx < config_.num_antenna_ports; ++ap_idx) {
            // Create enhanced antenna carrier ID combining cell and antenna port
            // Format: (cell_idx << 8) | ap_idx to encode both in 16 bits
            // Ensure cell_idx fits in 8 bits
            if (cell_idx > std::numeric_limits<std::uint8_t>::max()) {
                throw std::invalid_argument(std::format(
                        "Cell index {} exceeds maximum of {}",
                        cell_idx,
                        std::numeric_limits<std::uint8_t>::max()));
            }
            const auto enhanced_antenna_carrier =
                    static_cast<std::uint16_t>((static_cast<unsigned>(cell_idx) << 8U) | ap_idx);

            const std::uint16_t cell_vlan = config_.cell_vlan_tcis.at(cell_idx);

            const auto &src_mac = net_env_->nic().mac_address();
            auto header_template = create_packet_header_template(
                    src_mac, dest_mac, cell_vlan, enhanced_antenna_carrier);

            RT_LOGC_DEBUG(
                    FronthaulLog::FronthaulNetwork,
                    "[PACKET_HEADER] Cell {} Port {}: src_mac={} dst_mac={} vlan_tci=0x{:04X} "
                    "eAxC=0x{:04X}",
                    cell_idx,
                    ap_idx,
                    src_mac.to_string(),
                    dest_mac.to_string(),
                    cell_vlan,
                    enhanced_antenna_carrier);

            cell_flows.push_back(std::make_unique<ran::oran::SimpleOranFlow>(header_template));
            cell_peers.push_back(std::make_unique<ran::oran::SimpleOranPeer>());
        }

        oran_flows_.push_back(std::move(cell_flows));
        oran_peers_.push_back(std::move(cell_peers));

        RT_LOGC_DEBUG(
                FronthaulLog::FronthaulNetwork,
                "Created {} ORAN flows and peers for cell {}: dest={}",
                config_.num_antenna_ports,
                cell_idx,
                dest_mac.to_string());
    }

    // 4. Pre-allocate C-plane message buffer to avoid per-slot allocations
    // Estimate: max PDUs per request * antenna ports * max symbols per PDU
    // Use 14 (max symbols per slot) for conservative allocation
    static constexpr std::size_t MAX_SYMBOLS_PER_PDU = 14;
    static constexpr std::size_t MAX_PDUS_PER_REQUEST = 16;
    const std::size_t estimated_messages =
            MAX_PDUS_PER_REQUEST * config_.num_antenna_ports * MAX_SYMBOLS_PER_PDU;
    cplane_infos_.reserve(estimated_messages);

    // 5. Pre-allocate packet buffers to avoid per-packet allocations
    // Estimate: assume ~3 packets per message on average (accounting for MTU fragmentation)
    static constexpr std::size_t AVG_PACKETS_PER_MESSAGE = 3;
    const std::size_t estimated_packets = estimated_messages * AVG_PACKETS_PER_MESSAGE;
    mbuf_pool_.reserve(estimated_packets);
    mbuf_wrapper_pool_.reserve(estimated_packets);

    RT_LOGC_INFO(
            FronthaulLog::FronthaulGeneral,
            "Fronthaul C-Plane initialized successfully: {} cells, reserved {} message slots, {} "
            "packet "
            "slots",
            config_.cell_dest_macs.size(),
            estimated_messages,
            estimated_packets);

    // 6. Initialize U-Plane pipeline (only if DOCA RX queues are configured)
    if (!net_env_->nic().doca_rx_queues().empty()) {
        initialize_uplane(config_.uplane_config);
    } else {
        RT_LOGC_INFO(
                FronthaulLog::FronthaulGeneral,
                "Skipping U-Plane initialization - no DOCA RX queues configured (C-Plane only "
                "mode)");
    }
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
void Fronthaul::send_ul_cplane(
        const scf_fapi_ul_tti_req_t &request,
        const std::size_t body_len,
        const std::uint16_t cell_id,
        const std::uint64_t absolute_slot,
        const std::chrono::nanoseconds t0,
        const std::chrono::nanoseconds tai_offset) {
    // NOLINTEND(bugprone-easily-swappable-parameters)

    // Validate cell_id
    if (cell_id >= config_.cell_dest_macs.size()) {
        RT_LOGC_ERROR(
                FronthaulLog::FronthaulGeneral,
                "Invalid cell_id {}: only {} cells configured",
                cell_id,
                config_.cell_dest_macs.size());
        return;
    }

    // Skip empty requests
    if (request.num_pdus == 0) {
        RT_LOGC_DEBUG(
                FronthaulLog::FronthaulGeneral, "Skipping empty request for cell {}", cell_id);
        return;
    }

    // Calculate packet send timing from parameters
    const PacketSendTimeParams timing_params{
            .t0 = t0,
            .tai_offset = tai_offset,
            .absolute_slot = absolute_slot,
            .slot_period = std::chrono::nanoseconds{config_.numerology.slot_period_ns},
            .slot_ahead = config_.slot_ahead,
            .t1a_max_cp_ul = std::chrono::nanoseconds{config_.t1a_max_cp_ul_ns},
            .actual_start = framework::task::Time::now_ns()};
    const auto timing = calculate_packet_send_time(timing_params);

    // Check timing threshold
    if (timing.exceeds_threshold) {
        RT_LOGC_WARN(
                FronthaulLog::FronthaulTiming,
                "Processing slot {} too late: delta={}ns exceeds threshold={}ns",
                absolute_slot,
                timing.time_delta.count(),
                timing.threshold.count());
    }

    const auto start_tx_ns = static_cast<std::uint64_t>(timing.start_tx.count());

    RT_LOGC_DEBUG(
            FronthaulLog::FronthaulGeneral,
            "Sending C-plane for cell {}, sfn={}, slot={}, start_tx={}ns",
            cell_id,
            request.sfn,
            request.slot,
            start_tx_ns);

    // Setup TX windows for this slot
    ran::oran::OranTxWindows tx_windows{};
    tx_windows.tx_window_start = start_tx_ns;
    tx_windows.tx_window_bfw_start = start_tx_ns + config_.tx_cell_start_offset_ns;
    tx_windows.tx_window_end = start_tx_ns + (config_.t1a_max_cp_ul_ns - config_.t1a_min_cp_ul_ns);

    // Convert FAPI to C-Plane (reuses pre-allocated member buffers)
    const std::error_code result = ran::oran::convert_ul_tti_request_to_cplane(
            request,
            body_len,
            static_cast<std::uint16_t>(config_.num_antenna_ports),
            config_.numerology,
            tx_windows,
            prb_chunks_,
            cplane_infos_);

    if (result || cplane_infos_.empty()) {
        RT_LOGC_WARN(
                FronthaulLog::FronthaulGeneral,
                "Failed to convert FAPI to C-Plane for cell {}",
                cell_id);
        return;
    }

    RT_LOGC_DEBUG(
            FronthaulLog::FronthaulGeneral,
            "Generated {} C-plane messages for cell {}",
            cplane_infos_.size(),
            cell_id);

    // Send C-plane messages for this cell
    const auto send_result = send_cplane_messages(cplane_infos_, cell_id);
    if (send_result) {
        total_send_errors_.fetch_add(1, std::memory_order_relaxed);
        RT_LOGC_ERROR(
                FronthaulLog::FronthaulNetwork,
                "Failed to send C-plane messages for cell {}: {}",
                cell_id,
                framework::net::get_error_name(send_result));
    } else {
        total_requests_sent_.fetch_add(1, std::memory_order_relaxed);
    }
}

std::error_code Fronthaul::send_cplane_messages(
        std::span<ran::oran::OranCPlaneMsgInfo> cplane_infos, const std::size_t cell_index) {

    if (cell_index >= oran_flows_.size()) {
        return std::make_error_code(std::errc::invalid_argument);
    }

    if (oran_flows_[cell_index].empty()) {
        return std::make_error_code(std::errc::invalid_argument);
    }

    // Count required packets
    const std::size_t packet_count = ran::oran::count_cplane_packets(cplane_infos, config_.mtu);
    if (packet_count == 0) {
        return {};
    }

    // Warn if packet count exceeds pre-allocated capacity
    if (packet_count > mbuf_pool_.capacity()) {
        RT_LOGC_WARN(
                FronthaulLog::FronthaulNetwork,
                "Packet count {} exceeds pre-allocated capacity {} - will allocate",
                packet_count,
                mbuf_pool_.capacity());
    }

    // Prepare pre-allocated buffers for this transmission
    mbuf_pool_.resize(packet_count);
    mbuf_wrapper_pool_.clear();

    // Get mempool and allocate mbufs
    auto *mempool = net_env_->nic().mempool(0).dpdk_mempool();
    const int alloc_result =
            rte_pktmbuf_alloc_bulk(mempool, mbuf_pool_.data(), static_cast<unsigned>(packet_count));
    if (alloc_result != 0) {
        RT_LOGC_ERROR(FronthaulLog::FronthaulNetwork, "Failed to allocate {} mbufs", packet_count);
        return std::make_error_code(std::errc::not_enough_memory);
    }

    // Wrap mbufs as MBufs
    mbuf_wrapper_pool_.reserve(packet_count);
    for (auto *mbuf : mbuf_pool_) {
        // cppcheck-suppress useStlAlgorithm
        mbuf_wrapper_pool_.emplace_back(mbuf);
    }

    // Prepare C-Plane packets for all messages
    std::uint16_t packets_prepared = 0;
    for (auto &cplane_info : cplane_infos) {
        const std::size_t msg_packet_count =
                ran::oran::count_cplane_packets(std::span{&cplane_info, 1}, config_.mtu);

        const std::size_t total_after_msg = packets_prepared + msg_packet_count;
        if (total_after_msg > packet_count) {
            RT_LOGC_ERROR(
                    FronthaulLog::FronthaulNetwork,
                    "Packet count mismatch: expected total={}, already prepared={}, attempting to "
                    "add={}, would overflow by {}",
                    packet_count,
                    packets_prepared,
                    msg_packet_count,
                    total_after_msg - packet_count);
            break;
        }

        // Get buffer slice for this message
        const auto msg_buffers =
                std::span{mbuf_wrapper_pool_}.subspan(packets_prepared, msg_packet_count);

        // Select flow based on cell and antenna port
        const std::size_t ap_idx = cplane_info.ap_idx;
        if (ap_idx >= oran_flows_[cell_index].size()) {
            RT_LOGC_ERROR(FronthaulLog::FronthaulNetwork, "Invalid antenna port index: {}", ap_idx);
            continue;
        }

        // Prepare packets - uses flow for packet templates/sequence IDs and peer
        // for timestamps
        const auto prepared = ran::oran::prepare_cplane_message(
                cplane_info,
                *oran_flows_[cell_index][ap_idx],
                *oran_peers_[cell_index][ap_idx],
                msg_buffers,
                config_.mtu);

        packets_prepared += prepared;
    }

    // Validate that actual preparation matches prediction
    if (packets_prepared != packet_count) {
        RT_LOGC_WARN(
                FronthaulLog::FronthaulNetwork,
                "Packet count prediction mismatch: predicted={}, actually prepared={}",
                packet_count,
                packets_prepared);
    }

    // Send via DPDK
    const std::span<rte_mbuf *> mbuf_span{mbuf_pool_.data(), packets_prepared};
    const auto send_result = framework::net::dpdk_eth_send_mbufs(
            mbuf_span, 0 /*queue_id*/, net_env_->nic().dpdk_port_id());

    if (send_result) {
        RT_LOGC_ERROR(
                FronthaulLog::FronthaulNetwork,
                "DPDK send failed for cell {}: {}",
                cell_index,
                framework::net::get_error_name(send_result));
    } else {
        total_packets_sent_.fetch_add(packets_prepared, std::memory_order_relaxed);
        RT_LOGC_DEBUG(
                FronthaulLog::FronthaulNetwork,
                "Sent {} C-plane packets for cell {}",
                packets_prepared,
                cell_index);
    }

    return send_result;
}

FronthaulStats Fronthaul::get_stats() const noexcept {
    const auto requests_sent = total_requests_sent_.load(std::memory_order_relaxed);
    const auto packets_sent = total_packets_sent_.load(std::memory_order_relaxed);
    const auto send_errors = total_send_errors_.load(std::memory_order_relaxed);

    const double avg_packets_per_request =
            requests_sent > 0
                    ? static_cast<double>(packets_sent) / static_cast<double>(requests_sent)
                    : 0.0;

    const FronthaulStats stats{
            .requests_sent = requests_sent,
            .packets_sent = packets_sent,
            .send_errors = send_errors,
            .avg_packets_per_request = avg_packets_per_request};

    return stats;
}

void Fronthaul::reset_stats() noexcept {
    total_requests_sent_.store(0, std::memory_order_relaxed);
    total_packets_sent_.store(0, std::memory_order_relaxed);
    total_send_errors_.store(0, std::memory_order_relaxed);
}

Fronthaul::OrderKernelStatistics Fronthaul::read_kernel_statistics() const {
    if (!uplane_resources_) {
        throw std::runtime_error("Cannot read kernel statistics: U-Plane not initialized");
    }

    return uplane_resources_->stats;
}

void Fronthaul::initialize_uplane(const UPlaneConfig &uplane_config) {

    gsl_Expects(net_env_ != nullptr);

    RT_LOGC_INFO(
            FronthaulLog::FronthaulGeneral,
            "Initializing U-Plane: NIC={}, RU MAC={}, GPU={}",
            config_.net_config.nic_config.nic_pcie_addr,
            config_.cell_dest_macs[0].to_string(),
            config_.net_config.gpu_device_id.value());

    // 1. Create GDRCopy handle for CPU-visible GPU memory
    auto gdr = framework::memory::make_unique_gdr_handle();

    // 2. Build Order Kernel pipeline via factory using shared net_env_
    const auto &doca_rxq = net_env_->nic().doca_rx_queue(0);
    const framework::net::DocaRxQParams *doca_params = doca_rxq.params();

    auto pipeline_factory = std::make_unique<ran::fronthaul::OrderKernelPipelineFactory>();
    pipeline_factory->set_doca_params(doca_params);

    framework::pipeline::PipelineSpec pipeline_spec{};
    pipeline_spec.execution_mode = framework::pipeline::ExecutionMode::Stream;

    const ran::fronthaul::OrderKernelModule::StaticParams module_params{
            .execution_mode = framework::pipeline::ExecutionMode::Stream,
            .gdr_handle = gdr.get(),
            .doca_rxq_params = doca_params,
            .timing =
                    {
                            .slot_duration_ns = uplane_config.slot_duration_ns,
                            .ta4_min_ns = uplane_config.ta4_min_ns,
                            .ta4_max_ns = uplane_config.ta4_max_ns,
                    },
            .eaxc_ids = uplane_config.eaxc_ids,
    };

    const framework::pipeline::ModuleSpec module_spec{framework::pipeline::ModuleCreationInfo{
            .module_type = "order_kernel_module",
            .instance_id = "order_kernel_0",
            .init_params = module_params,
    }};
    pipeline_spec.modules.emplace_back(module_spec);

    std::unique_ptr<ran::fronthaul::OrderKernelPipeline> pipeline;
    try {
        pipeline = pipeline_factory->create_order_kernel_pipeline(
                "order_kernel_pipeline_0", pipeline_spec);
    } catch (const std::exception &e) {
        RT_LOGC_ERROR(
                FronthaulLog::FronthaulGeneral, "Failed to create U-Plane pipeline: {}", e.what());
        uplane_resources_ = nullptr;
        throw;
    }

    // 3. Create bundle with all resources atomically using make_unique
    uplane_resources_ = std::make_unique<UPlaneResources>(UPlaneResources{
            .pipeline = std::move(pipeline),
            .stream = {}, // Default construct CudaStream
            .gdr_handle = std::move(gdr),
            .absolute_slot = 0,
            .stats = {},                // Initialize statistics to zero
            .order_kernel_outputs = {}, // Will be populated after warmup
    });

    // 4. Setup and warmup pipeline (now using bundled resources)
    try {
        uplane_resources_->pipeline->setup();
        uplane_resources_->pipeline->warmup(uplane_resources_->stream.get());

        // 5. Capture output addresses after warmup (stable for pipeline lifetime)
        uplane_resources_->order_kernel_outputs = uplane_resources_->pipeline->get_outputs();
        RT_LOGC_INFO(
                FronthaulLog::FronthaulGeneral,
                "Order Kernel output addresses captured: {} ports",
                uplane_resources_->order_kernel_outputs.size());
    } catch (const std::exception &e) {
        RT_LOGC_ERROR(
                FronthaulLog::FronthaulGeneral,
                "Failed to setup/warmup U-Plane pipeline: {}",
                e.what());
        uplane_resources_ = nullptr;
        throw;
    }

    RT_LOGC_INFO(
            FronthaulLog::FronthaulGeneral, "U-Plane Order Kernel pipeline initialized and ready");
}

void Fronthaul::process_uplane(const ran::oran::OranSlotTiming timing) {
    // Early return if U-Plane is not configured (C-Plane only mode)
    if (!uplane_resources_) {
        RT_LOGC_DEBUG(
                FronthaulLog::FronthaulGeneral,
                "U-Plane processing skipped - not configured (C-Plane only mode)");
        return;
    }

    // Use gsl_lite::finally to ensure absolute_slot is always incremented
    auto &absolute_slot = uplane_resources_->absolute_slot;
    const auto slot_increment = gsl_lite::finally([&absolute_slot]() { ++absolute_slot; });

    // Pass OranSlotTiming as module-specific parameter
    const framework::pipeline::DynamicParams dynamic_params{
            .module_specific_params = std::any{timing},
    };

    // Configure pipeline I/O
    std::vector<framework::pipeline::PortInfo> external_outputs(1);

    try {
        std::vector<framework::pipeline::PortInfo> external_inputs{};
        uplane_resources_->pipeline->configure_io(
                dynamic_params, external_inputs, external_outputs, uplane_resources_->stream.get());
    } catch (const std::exception &e) {
        RT_LOGC_ERROR(
                FronthaulLog::FronthaulGeneral,
                "Failed to configure U-Plane pipeline I/O: {}",
                e.what());
        return;
    }

    // Execute pipeline (launches GPU kernels asynchronously)
    // No stream synchronization needed here - the next slot's configure_io() call
    // includes a synchronization barrier, ensuring this slot's GPU work completes
    // before the next slot updates descriptors. This allows CPU/GPU overlap.
    try {
        uplane_resources_->pipeline->execute_stream(uplane_resources_->stream.get());
    } catch (const std::exception &e) {
        RT_LOGC_ERROR(
                FronthaulLog::FronthaulGeneral, "Failed to execute U-Plane pipeline: {}", e.what());
        return;
    }

    // Synchronize to ensure kernel completes before reading results from GDRCopy memory
    std::uint32_t slot_ordered_prbs = 0;
    std::uint32_t slot_expected_prbs = 0;
    try {
        FRAMEWORK_CUDA_RUNTIME_CHECK_THROW(cudaStreamSynchronize(uplane_resources_->stream.get()));
        const auto kernel_results = uplane_resources_->pipeline->read_kernel_results();
        RT_LOGC_DEBUG(FronthaulLog::FronthaulGeneral, "[KERNEL_EXIT] Cell=0 | {}", kernel_results);

        // Save slot-specific counts before accumulating
        slot_ordered_prbs = kernel_results.pusch_ordered_prbs;
        slot_expected_prbs = kernel_results.expected_prbs;

        // Accumulate PRB counts and expected PRBs across slots
        uplane_resources_->stats.total_pusch_prbs += kernel_results.pusch_ordered_prbs;
        uplane_resources_->stats.total_prach_prbs += kernel_results.prach_ordered_prbs;
        uplane_resources_->stats.total_srs_prbs += kernel_results.srs_ordered_prbs;
        uplane_resources_->stats.total_expected_prbs += kernel_results.expected_prbs;
        ++uplane_resources_->stats.slots_processed;
    } catch (const std::exception &e) {
        RT_LOGC_WARN(FronthaulLog::FronthaulGeneral, "Failed to read kernel results: {}", e.what());
    }

    // Log execution
    if (!external_outputs.empty() && !external_outputs[0].tensors.empty()) {
        const auto &pusch_port = external_outputs[0];
        const auto *pusch_buffer = pusch_port.tensors[0].device_ptr;
        const auto pusch_size = pusch_port.tensors[0].tensor_info.get_total_size_in_bytes();
        const auto data_type = pusch_port.tensors[0].tensor_info.get_type();

        RT_LOGC_DEBUG(
                FronthaulLog::FronthaulGeneral,
                "U-Plane slot {} executed: PUSCH buffer={}, size={} bytes, type={}, "
                "ordered_prbs={}/{} expected",
                uplane_resources_->absolute_slot,
                pusch_buffer,
                pusch_size,
                framework::tensor::nv_get_data_type_string(data_type),
                slot_ordered_prbs,
                slot_expected_prbs);
    }
}

} // namespace ran::fronthaul
