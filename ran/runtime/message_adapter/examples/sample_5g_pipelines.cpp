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
#include <bit>
#include <climits>
#include <cmath>
#include <optional> // for optional
#include <sstream>  // for basic_stringstream
#include <stdexcept>
#include <tuple>
#include <unordered_map> // for unordered_map
#include <utility>       // for move

#include <quill/LogMacros.h>     // for QUILL_LOG_ERROR, QUILL_LOG_INFO
#include <scf_5g_fapi.h>         // for scf_fapi_body_header_t, scf_fapi_h...
#include <wise_enum_detail.h>    // for WISE_ENUM_IMPL_IIF_0
#include <wise_enum_generated.h> // for WISE_ENUM_IMPL_LOOP_4, WISE_ENUM_I...

#include <gsl-lite/gsl-lite.hpp>

#include "cell.hpp"
#include "driver/pusch_pipeline_context.hpp" // for PuschPipelineContext, PuschHos...
#include "log/components.hpp"                // For component/event declarations
#include "log/rt_log.hpp"                    // For logger configuration
#include "log/rt_log_macros.hpp"             // For logging macros
#include "message_adapter/phy_stats.hpp"     // for PhyStats
#include "pipeline_interface.hpp"            // for PhyMacMsgDesc
#include "pusch/pusch_defines.hpp"
#include "ran_common.hpp"
#include "sample_5g_pipelines.hpp"

namespace ran::message_adapter {

using ran::fapi_5g::Cell;
using ran::fapi_5g::FapiStateT;

namespace common = ran::common;
namespace driver = ran::driver;

//! Logging component for 5G FAPI sample pipelines
DECLARE_LOG_COMPONENT(Sample5gPipelines, Core, Fapi5g);

//! FAPI event logging identifiers for message processing
DECLARE_LOG_EVENT(
        FapiEvent,
        ConfigRequest,
        ConfigResponse,
        StartRequest,
        StopRequest,
        UlTtiRequest,
        DlTtiRequest,
        SlotIndication,
        SlotResponse,
        ErrorIndication,
        RcvdMessage,
        CrcIndication,
        RxDataIndication,
        PuschNoiseVarIndication);

/** FAPI error event logging identifiers */
DECLARE_LOG_EVENT(
        ErrorEvent,
        InvalidParam,
        InvalidState,
        InvalidConfig,
        InvalidCellid,
        InvalidData,
        InvalidMessage);

Sample5GPipeline::Sample5GPipeline(const InitParams &params)
        : max_cells_(params.max_cells), ipc_(params.ipc), output_provider_(params.output_provider),
          phy_stats_(params.max_cells), on_graph_schedule_(params.on_graph_schedule) {

    static constexpr auto MAX_CELLS = sizeof(decltype(active_cell_bitmap_)::value_type) * CHAR_BIT;
    for (auto &slot : slot_accumulation_) {
        slot.messages.reserve(MAX_CELLS);
    }

    // slot_accumulation_.done already initialized to false by member initializer {}

    // Make sure we fit all cells in the bitmap
    gsl_Expects(max_cells_ <= MAX_CELLS);

    if (ipc_ == nullptr) {
        throw std::invalid_argument("Sample5GPipeline: ipc parameter cannot be nullptr");
    }

    if (output_provider_ == nullptr) {
        throw std::invalid_argument(
                "Sample5GPipeline: output_provider parameter cannot be nullptr");
    }

    RT_LOGC_INFO(Sample5gPipelines::Core, "Sample5GPipeline created with max_cells={}", max_cells_);

    // Initialize cell_id_map with invalid IDs
    cell_id_map_.fill(common::INVALID_CELL_ID);

    // Configure logging
    framework::log::Logger::configure(
            framework::log::LoggerConfig::console(framework::log::LogLevel::Debug));

    framework::log::register_component<Sample5gPipelines>(
            {{Sample5gPipelines::Core, framework::log::LogLevel::Debug},
             {Sample5gPipelines::Fapi5g, framework::log::LogLevel::Debug}});

    // Reset counters
    processed_messages_ = 0;
    total_errors_ = 0;
}

void Sample5GPipeline::process_msg(nv_ipc_msg_t &msg) {
    // ipc_ is set in constructor and used for sending responses

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    auto *hdr = reinterpret_cast<scf_fapi_header_t *>(msg.msg_buf);
    if (hdr->handle_id != msg.cell_id || hdr->message_count != 1) {
        RT_LOGEC_ERROR(
                Sample5gPipelines::Fapi5g, ErrorEvent::InvalidMessage, "Invalid message pointer");
        total_errors_++;
    }

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast,cppcoreguidelines-pro-bounds-pointer-arithmetic)
    void *ptr = ((reinterpret_cast<uint8_t *>(msg.msg_buf)) + sizeof(scf_fapi_header_t));
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    scf_fapi_body_header_t &body_hdr = *(reinterpret_cast<scf_fapi_body_header_t *>(ptr));

    const uint32_t head_len = sizeof(scf_fapi_header_t) + sizeof(scf_fapi_body_header_t);
    const uint32_t body_len = body_hdr.length;
    if (static_cast<uint32_t>(msg.msg_len) != head_len + body_len) {
        RT_LOGEC_ERROR(
                Sample5gPipelines::Fapi5g,
                ErrorEvent::InvalidMessage,
                "msg_len is not set correctly: cell_id= {} msg_id={} msg_len={} head_len={} "
                "body_len={}",
                msg.cell_id,
                msg.msg_id,
                msg.msg_len,
                head_len,
                body_len);
        total_errors_++;
    }

    const uint16_t type_id = body_hdr.type_id;
    RT_LOGEC_DEBUG(Sample5gPipelines::Fapi5g, FapiEvent::RcvdMessage, "Message ID:{}", msg.msg_id);

    // Validate cell_id for all messages
    const auto cell_id = static_cast<uint16_t>(msg.cell_id);
    if (cell_id >= common::NUM_CELLS_SUPPORTED) {
        RT_LOGEC_ERROR(
                Sample5gPipelines::Fapi5g,
                ErrorEvent::InvalidCellid,
                "Invalid cell_id={} >= NUM_CELLS_SUPPORTED={}",
                cell_id,
                common::NUM_CELLS_SUPPORTED);
        ipc_->rx_release(ipc_, &msg);
        total_errors_++;
        return;
    }

    // For non-CONFIG_REQUEST messages, also check if cell exists
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
    if (type_id != SCF_FAPI_CONFIG_REQUEST && !cells_[cell_id]) {
        RT_LOGEC_ERROR(
                Sample5gPipelines::Fapi5g,
                ErrorEvent::InvalidCellid,
                "Cell {} not configured for message type {}",
                cell_id,
                type_id);
        ipc_->rx_release(ipc_, &msg);
        total_errors_++;
        return;
    }

    switch (type_id) {
    case SCF_FAPI_CONFIG_REQUEST: {
        const scf_fapi_error_codes_t ret = process_config_request(
                // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
                reinterpret_cast<scf_fapi_config_request_msg_t &>(body_hdr),
                static_cast<uint16_t>(msg.cell_id),
                body_len); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        std::ignore = send_cell_config_response(static_cast<uint16_t>(msg.cell_id), ret);
    } break;
    case SCF_FAPI_START_REQUEST: {
        const scf_fapi_error_codes_t ret =
                process_start_request(static_cast<uint16_t>(msg.cell_id));

        if (ret != SCF_ERROR_CODE_MSG_OK) {
            std::ignore = send_error_indication(
                    static_cast<uint16_t>(msg.cell_id), SCF_FAPI_START_REQUEST, ret);
        }
        // Note: process_start_request handles slot counter init and callback invocation
    } break;
    case SCF_FAPI_UL_TTI_REQUEST: {
        std::ignore = process_ul_tti_request(
                // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
                reinterpret_cast<scf_fapi_ul_tti_req_t &>(body_hdr),
                static_cast<uint16_t>(msg.cell_id));
    } break;
    case SCF_FAPI_DL_TTI_REQUEST: {
        std::ignore = process_dl_tti_request(
                // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
                reinterpret_cast<scf_fapi_dl_tti_req_t &>(body_hdr),
                static_cast<uint16_t>(msg.cell_id));
    } break;

    case SCF_FAPI_STOP_REQUEST: {
        const scf_fapi_error_codes_t ret = process_stop_request(static_cast<uint16_t>(msg.cell_id));
        if (ret != SCF_ERROR_CODE_MSG_OK) {
            std::ignore = send_error_indication(
                    static_cast<uint16_t>(msg.cell_id), SCF_FAPI_STOP_REQUEST, ret);
        }
    } break;

    case SCF_FAPI_SLOT_RESPONSE: {
        process_slot_response(
                // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
                reinterpret_cast<scf_fapi_slot_rsp_t &>(body_hdr),
                static_cast<uint16_t>(msg.cell_id));
    } break;

    case SCF_FAPI_SLOT_INDICATION: {
        const uint32_t packed = sfn_slot_packed_.load(std::memory_order_acquire);
        driver_.reset_slot_status(unpack_slot(packed));
    } break;

    default:
        RT_LOGC_ERROR(Sample5gPipelines::Fapi5g, "Ignoring message ID: {}", type_id);
        break;
    }
    ipc_->rx_release(ipc_, &msg);
    processed_messages_++;
}

std::string Sample5GPipeline::get_status() const {
    std::stringstream strm;
    strm << "Sample5GPipeline Status:\n"
         << "  Processed Messages: " << processed_messages_ << "\n"
         << "  Total Errors: " << total_errors_ << "\n"
         << "  Success Rate: "
         << (processed_messages_ > 0
                     ? (100.0 * (processed_messages_ - total_errors_) / processed_messages_)
                     : 0.0)
         << "%";

    return strm.str();
}

std::size_t Sample5GPipeline::get_num_cells_running() const {
    // Count number of bits set in active_cell_bitmap (each bit = one running cell)
    const auto bitmap = active_cell_bitmap_.load(std::memory_order_acquire);
    return static_cast<std::size_t>(std::popcount(bitmap));
}

void Sample5GPipeline::process_slot_response(
        const scf_fapi_slot_rsp_t &slot_response, const uint16_t cell_id) {
    // cell_id already validated in process_message

    /* Optional: Validate slot response matches current slot
    const uint32_t packed = sfn_slot_packed_.load(std::memory_order_acquire);
    if (slot_response.sfn != unpack_sfn(packed) || slot_response.slot != unpack_slot(packed)) {
         RT_LOGEC_ERROR(
                 Sample5gPipelines::Fapi5g,
                 FapiEvent::SlotResponse,
                 "Slot response mismatch: expected sfn={} slot={}, received sfn={} slot={}",
                 unpack_sfn(packed),
                 unpack_slot(packed),
                 slot_response.sfn,
                 slot_response.slot);
         return;
     }*/

    RT_LOGEC_DEBUG(
            Sample5gPipelines::Fapi5g,
            FapiEvent::SlotResponse,
            "Slot response received for cell_id={} sfn={} slot={}",
            cell_id,
            slot_response.sfn,
            slot_response.slot);

    // Process slot response in driver - check if all active cells have responded
    // Note: Driver expects uint32_t but we use uint64_t for up to 64 cells support
    const auto bitmap = active_cell_bitmap_.load(std::memory_order_acquire);
    const bool slot_completed = driver_.process_slot_response(
            slot_response.slot, cell_id, gsl_lite::narrow_cast<std::uint32_t>(bitmap));

    if (slot_completed) {
        RT_LOGEC_INFO(
                Sample5gPipelines::Fapi5g,
                FapiEvent::SlotResponse,
                "Slot {} completed - all active cells have responded, {} messages accumulated",
                slot_response.slot,
                // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
                slot_accumulation_[slot_response.slot].messages.size());

        // Mark accumulation complete for this slot
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
        slot_accumulation_[slot_response.slot].done.store(true, std::memory_order_release);

        const ran::fapi::SlotInfo current_slot_info{
                slot_response.sfn,
                slot_response.slot,
        };

        // Trigger uplink graph scheduling with accumulated messages
        if (on_graph_schedule_) {
            RT_LOGEC_INFO(
                    Sample5gPipelines::Fapi5g,
                    FapiEvent::SlotResponse,
                    "Invoking graph schedule callback for slot {}",
                    slot_response.slot);
            on_graph_schedule_(current_slot_info);
        }
    }
}

scf_fapi_error_codes_t Sample5GPipeline::process_config_request(
        scf_fapi_config_request_msg_t &config_request,
        const uint16_t cell_id,
        const uint32_t body_len) {

    uint32_t consumed_body_len = 0;

    // cell_id already validated in process_message
    if (num_cells_configured_ == common::NUM_CELLS_SUPPORTED) {
        RT_LOGEC_ERROR(
                Sample5gPipelines::Fapi5g,
                FapiEvent::ConfigRequest,
                "Cell capacity exceeded. num_cells_configured={}",
                num_cells_configured_);
        return SCF_ERROR_CODE_MSG_INVALID_CONFIG;
    }

    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
    if (cells_[cell_id] &&
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
        cells_[cell_id]->fapi_state.load(std::memory_order_acquire) != FapiStateT::FapiStateIdle) {
        RT_LOGEC_ERROR(
                Sample5gPipelines::Fapi5g,
                FapiEvent::ConfigRequest,
                "cell_id={} : FAPI state is not idle. Cannot process config request",
                cell_id);
        return SCF_ERROR_CODE_MSG_INVALID_STATE;
    }

    uint8_t *body_ptr = &config_request.msg_body.tlvs[0];
    auto tlvs = config_request.msg_body.num_tlvs;
    consumed_body_len += sizeof(tlvs);

    RT_LOGEC_INFO(
            Sample5gPipelines::Fapi5g,
            FapiEvent::ConfigRequest,
            "Processing CONFIG_REQUEST for cell_id={} num TLVs={}",
            cell_id,
            tlvs);

    common::PhyParams ul_params;
    uint32_t phy_cell_id = common::INVALID_CELL_ID;
    while (tlvs != 0 && consumed_body_len < body_len) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        auto *hdr = reinterpret_cast<scf_fapi_tl_t *>(body_ptr);
        switch (hdr->tag) {
        case CONFIG_TLV_NUM_RX_ANT:
            ul_params.num_rx_ant = hdr->AsValue<uint16_t>();
            if (ul_params.num_rx_ant != common::MAX_ANTENNAS) {
                RT_LOGEC_ERROR(
                        Sample5gPipelines::Fapi5g,
                        FapiEvent::ConfigRequest,
                        "Number of receive antennas={} not supported. Only {} antennas are "
                        "supported",
                        ul_params.num_rx_ant,
                        common::MAX_ANTENNAS);
                return SCF_ERROR_CODE_MSG_INVALID_CONFIG;
            }
            RT_LOGC_DEBUG(Sample5gPipelines::Fapi5g, "num_rx_ant_={}", ul_params.num_rx_ant);
            break;
        case CONFIG_TLV_PHY_CELL_ID:
            phy_cell_id = hdr->AsValue<uint16_t>();
            RT_LOGC_DEBUG(Sample5gPipelines::Fapi5g, "phy_cell_id={}", phy_cell_id);
            break;
        case CONFIG_TLV_UL_BANDWIDTH:
            ul_params.bandwidth = hdr->AsValue<uint16_t>();
            if (ul_params.bandwidth != common::BANDWIDTH_SUPPORTED_MHZ) {
                RT_LOGEC_ERROR(
                        Sample5gPipelines::Fapi5g,
                        FapiEvent::ConfigRequest,
                        "Bandwidth={} not supported. Only 100MHz is supported",
                        ul_params.bandwidth);
                return SCF_ERROR_CODE_MSG_INVALID_CONFIG;
            }
            ul_params.num_prb = common::NUM_PRBS_SUPPORTED;
            RT_LOGC_DEBUG(Sample5gPipelines::Fapi5g, "ul_bandwidth_={}", ul_params.bandwidth);
            break;
        default:
            const uint16_t tag = hdr->tag; // Copy creates aligned local variable
            RT_LOGC_DEBUG(Sample5gPipelines::Fapi5g, "Ignoring TLV tag: {}", tag);
            break;
        }
        // Round up TLV length to 4-byte boundary according to specs
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        body_ptr += sizeof(scf_fapi_tl_t) +
                    static_cast<size_t>(
                            ((hdr->length + 3) / 4) *
                            4); // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        tlvs--;
        consumed_body_len += sizeof(scf_fapi_tl_t) +
                             static_cast<size_t>(
                                     ((hdr->length + 3) / 4) *
                                     4); // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        if (consumed_body_len > body_len) {
            RT_LOGEC_ERROR(
                    Sample5gPipelines::Fapi5g,
                    FapiEvent::ConfigRequest,
                    "Malformed Config Request causing buffer overrun while reading TLVs. Read "
                    "tlvs={} consumed_body_len={} body_len={}",
                    consumed_body_len,
                    body_len,
                    tlvs);
            return SCF_ERROR_CODE_MSG_INVALID_CONFIG;
        }
    }

    // Create Cell if it doesn't exist
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
    if (!cells_[cell_id]) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
        cells_[cell_id] = std::make_unique<Cell>(phy_cell_id, ul_params);
    }
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
    cells_[cell_id]->fapi_state.store(FapiStateT::FapiStateConfigured, std::memory_order_release);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
    cell_id_map_[num_cells_configured_] = cell_id;
    num_cells_configured_++;

    RT_LOGEC_INFO(
            Sample5gPipelines::Fapi5g,
            FapiEvent::ConfigRequest,
            "FAPI state set to FapiStateConfigured for cell_id={}",
            cell_id);

    const std::string execution_mode = "Stream";

    // Create UL indication callback
    const auto ul_callback = [this](const std::size_t slot) { send_ul_indications(slot); };

    // Get Order Kernel output addresses from output provider (captured after warmup)
    const auto order_kernel_outputs = output_provider_->get_order_kernel_outputs();
    if (order_kernel_outputs.empty()) {
        RT_LOGEC_ERROR(
                Sample5gPipelines::Fapi5g,
                FapiEvent::ConfigRequest,
                "Order Kernel outputs not available from output provider - ensure pipeline is "
                "initialized");
        return SCF_ERROR_CODE_MSG_INVALID_CONFIG;
    }

    RT_LOGC_INFO(
            Sample5gPipelines::Fapi5g,
            "Retrieved Order Kernel output addresses from output provider: {} ports",
            order_kernel_outputs.size());

    driver_.create_pusch_pipeline(ul_params, execution_mode, ul_callback, order_kernel_outputs);

    return SCF_ERROR_CODE_MSG_OK;
}

bool Sample5GPipeline::process_ul_tti_request(
        const scf_fapi_ul_tti_req_t &ul_tti_request, const uint16_t cell_id) {
    RT_LOGEC_INFO(
            Sample5gPipelines::Fapi5g,
            FapiEvent::UlTtiRequest,
            "sfn={} slot={} cell_id={} num_pdus={} num_ue_groups={}",
            ul_tti_request.sfn,
            ul_tti_request.slot,
            cell_id,
            ul_tti_request.num_pdus,
            ul_tti_request.ngroup);

    // Check if cell is in running state
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
    const FapiStateT current_state = cells_[cell_id]->fapi_state.load(std::memory_order_acquire);
    if (current_state != FapiStateT::FapiStateRunning) {
        RT_LOGEC_ERROR(
                Sample5gPipelines::Fapi5g,
                FapiEvent::UlTtiRequest,
                "Cell {} not in running state (current state={})",
                cell_id,
                static_cast<int>(current_state));
        return false;
    }

    // Get an available host input resource
    auto maybe_host_buffers = driver_.pusch_pipeline_context.get_host_buffers();
    if (!maybe_host_buffers.has_value()) {
        RT_LOGEC_ERROR(
                Sample5gPipelines::Fapi5g,
                FapiEvent::UlTtiRequest,
                "No host input resources available for sfn={} slot={}",
                ul_tti_request.sfn,
                ul_tti_request.slot);
        return false;
    }

    // Unpack the resource index and pointer
    auto [host_buffers_index, host_buffers] = maybe_host_buffers.value();

    RT_LOGEC_DEBUG(
            Sample5gPipelines::Fapi5g,
            FapiEvent::UlTtiRequest,
            "Allocated host buffers resource {} for sfn={} slot={}",
            host_buffers_index,
            ul_tti_request.sfn,
            ul_tti_request.slot);

    // Save host input index for this slot (pipeline index will be set later)
    driver_.pusch_pipeline_context.save_host_buffers_index(ul_tti_request.slot, host_buffers_index);

    // Get Cell to retrieve ulPhyParams
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
    const auto &cell = cells_[cell_id];

    uint32_t offset = 0;
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    const auto *data = reinterpret_cast<const uint8_t *>(ul_tti_request.payload);
    // Process PUSCH PDUs
    for (std::size_t i = 0; i < ul_tti_request.num_pdus; ++i) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast,cppcoreguidelines-pro-bounds-pointer-arithmetic)
        const auto &pdu = *(reinterpret_cast<const scf_fapi_generic_pdu_info_t *>(data + offset));

        switch (pdu.pdu_type) {
        case UL_TTI_PDU_TYPE_PUSCH: {
            const auto &pdu_dat =
                    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast,cppcoreguidelines-pro-bounds-pointer-arithmetic)
                    *reinterpret_cast<const scf_fapi_pusch_pdu_t *>(&pdu.pdu_config[0]);
            if (!driver_.pusch_pipeline_context.prepare_input_data(
                        ul_tti_request.sfn,
                        cell_id,
                        pdu_dat,
                        host_buffers->inputs.pusch_inputs,
                        cell->ul_phy_params)) {
                RT_LOGEC_ERROR(
                        Sample5gPipelines::Fapi5g,
                        FapiEvent::UlTtiRequest,
                        "Failed to prepare input data for PUSCH PDU {}",
                        i);
                driver_.pusch_pipeline_context.release_host_buffers(host_buffers_index);
                driver_.pusch_pipeline_context.clear_slot_resources(ul_tti_request.slot);
                return false;
            }
            break;
        }
        default:
            RT_LOGEC_ERROR(
                    Sample5gPipelines::Fapi5g,
                    FapiEvent::UlTtiRequest,
                    "Unsupported PDU type {} in UL_TTI_REQUEST",
                    pdu.pdu_type);
            // Skip unsupported PDU types
            break;
        }
        offset += pdu.pdu_size;
    }

    host_buffers->inputs.pusch_inputs.ue_group_idx_index = ul_tti_request.ngroup;
    // NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast,cppcoreguidelines-pro-bounds-pointer-arithmetic)
    for (std::size_t i = 0; i < ul_tti_request.ngroup; ++i) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        const uint8_t num_ues = *(data + offset); // number of UEs in each group
        offset += sizeof(uint8_t);
        for (std::size_t j = 0; j < num_ues; ++j) {
            host_buffers->inputs.pusch_inputs.ue_group_idx_map[i][j] =
                    *(reinterpret_cast<const int8_t *>(data + offset));
            offset += sizeof(uint8_t);
        }
    }
    // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast,cppcoreguidelines-pro-bounds-pointer-arithmetic)

    // Accumulate message for C-plane processing
    // If we already have messages, check if this is a new slot (compare both SFN and slot)
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
    if (!slot_accumulation_[ul_tti_request.slot].messages.empty()) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        const auto *first_msg_hdr = reinterpret_cast<const scf_fapi_body_header_t *>(
                // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
                slot_accumulation_[ul_tti_request.slot].messages[0].msg_data.data());
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        const auto *first_msg = reinterpret_cast<const scf_fapi_ul_tti_req_t *>(first_msg_hdr);

        // Check both SFN and slot to detect transitions across frame boundaries
        if (first_msg->sfn != ul_tti_request.sfn || first_msg->slot != ul_tti_request.slot) {
            // New slot detected - clear previous slot's messages
            RT_LOGEC_DEBUG(
                    Sample5gPipelines::Fapi5g,
                    FapiEvent::UlTtiRequest,
                    "New slot detected (prev={}:{}, curr={}:{}), clearing {} accumulated messages",
                    first_msg->sfn,
                    first_msg->slot,
                    ul_tti_request.sfn,
                    ul_tti_request.slot,
                    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
                    slot_accumulation_[ul_tti_request.slot].messages.size());
            // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
            slot_accumulation_[ul_tti_request.slot].messages.clear();
        }
    }

    // Deep copy message for C-plane processing
    ran::fapi::CapturedFapiMessage captured;
    captured.cell_id = cell_id;
    captured.msg_id = SCF_FAPI_UL_TTI_REQUEST;

    // Calculate total message length (body header + body content)
    // NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast,cppcoreguidelines-pro-bounds-pointer-arithmetic)
    const auto *body_start = reinterpret_cast<const uint8_t *>(&ul_tti_request);
    const uint32_t total_msg_len =
            sizeof(scf_fapi_body_header_t) +
            reinterpret_cast<const scf_fapi_body_header_t *>(&ul_tti_request)->length;
    // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast,cppcoreguidelines-pro-bounds-pointer-arithmetic)

    // Deep copy message data
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    captured.msg_data.assign(body_start, body_start + total_msg_len);

    // UL_TTI_REQUEST has no separate data buffer (leave data_buf empty)

    // Add to accumulation vector
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
    slot_accumulation_[ul_tti_request.slot].messages.push_back(std::move(captured));

    RT_LOGEC_DEBUG(
            Sample5gPipelines::Fapi5g,
            FapiEvent::UlTtiRequest,
            "Accumulated UL_TTI_REQUEST: cell_id={} sfn={} slot={} (total messages: {})",
            cell_id,
            ul_tti_request.sfn,
            ul_tti_request.slot,
            // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
            slot_accumulation_[ul_tti_request.slot].messages.size());

    return true;
}

// cppcheck-suppress functionStatic
// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
bool Sample5GPipeline::process_dl_tti_request(
        const scf_fapi_dl_tti_req_t &dl_tti_request, const uint16_t cell_id) {
    RT_LOGEC_INFO(
            Sample5gPipelines::Fapi5g,
            FapiEvent::DlTtiRequest,
            "sfn={} slot={} cell_id={}",
            dl_tti_request.sfn,
            dl_tti_request.slot,
            cell_id);
    return true;
}

template <typename T>
static scf_fapi_body_header_t *
add_scf_fapi_hdr(nv_ipc_msg_t &msg, int msg_id, uint16_t cell_id, bool data) {
    scf_fapi_header_t *hdr = nullptr;
    if (data) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        hdr = reinterpret_cast<scf_fapi_header_t *>(msg.data_buf);
    } else {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        hdr = reinterpret_cast<scf_fapi_header_t *>(msg.msg_buf);
    }

    hdr->message_count = 1;
    hdr->handle_id = static_cast<uint8_t>(cell_id);

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    auto *body = reinterpret_cast<scf_fapi_body_header_t *>(hdr->payload);
    body->type_id = static_cast<uint16_t>(msg_id);
    body->length = static_cast<uint32_t>(sizeof(T) - sizeof(scf_fapi_body_header_t));

    msg.msg_id = msg_id;
    msg.cell_id = static_cast<int32_t>(cell_id);
    msg.msg_len = static_cast<int32_t>(
            sizeof(scf_fapi_header_t) + sizeof(scf_fapi_body_header_t) + body->length);
    msg.data_len = 0;

    return body;
}

// cppcheck-suppress functionStatic
// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
bool Sample5GPipeline::send_cell_config_response(
        uint16_t cell_id, scf_fapi_error_codes_t error_code) {
    PhyMacMsgDesc msg_desc;
    if (ipc_->tx_allocate(ipc_, &msg_desc, 0) < 0) {
        RT_LOGC_ERROR(Sample5gPipelines::Fapi5g, "Failed to allocate message descriptor");
        return false;
    }
    RT_LOGEC_INFO(
            Sample5gPipelines::Fapi5g,
            FapiEvent::ConfigResponse,
            "Send CONFIG.response: cell_id={} error_code={:#04X}",
            cell_id,
            static_cast<uint32_t>(error_code));
    auto *fapi = add_scf_fapi_hdr<scf_fapi_config_response_msg_t>(
            msg_desc, SCF_FAPI_CONFIG_RESPONSE, cell_id, false);

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    auto *rsp = reinterpret_cast<scf_fapi_config_response_msg_t *>(fapi);
    rsp->msg_body.error_code = error_code;
    rsp->msg_body.num_invalid_tlvs = 0;
    rsp->msg_body.num_idle_only_tlvs = 0;
    rsp->msg_body.num_running_only_tlvs = 0;
    rsp->msg_body.num_missing_tlvs = 0;

    msg_desc.msg_len = static_cast<int32_t>(
            fapi->length + sizeof(scf_fapi_body_header_t) + sizeof(scf_fapi_header_t));

    ipc_->tx_send_msg(ipc_, &msg_desc);
    ipc_->tx_tti_sem_post(ipc_);

    return true;
}

void Sample5GPipeline::send_slot_indications() {
    // Load current sfn/slot atomically (single writer, multiple readers)
    const uint32_t current_packed = sfn_slot_packed_.load(std::memory_order_acquire);
    const uint16_t current_sfn = unpack_sfn(current_packed);
    const uint16_t current_slot = unpack_slot(current_packed);

    RT_LOGC_INFO(
            Sample5gPipelines::Fapi5g,
            "send_slot_indications() called: sfn={} slot={} num_cells_configured={}",
            current_sfn,
            current_slot,
            num_cells_configured_);

    // Send slot indication to all running cells
    for (uint32_t i = 0; i < num_cells_configured_; i++) {
        const auto cell_id =
                cell_id_map_[i]; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index)

        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
        const bool cell_exists = (cells_[cell_id] != nullptr);
        const auto cell_state =
                // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
                cell_exists ? cells_[cell_id]->fapi_state.load(std::memory_order_acquire)
                            : FapiStateT::FapiStateIdle;

        RT_LOGC_INFO(
                Sample5gPipelines::Fapi5g,
                "Checking cell {}: exists={} state={} (running={})",
                cell_id,
                cell_exists,
                static_cast<int>(cell_state),
                static_cast<int>(FapiStateT::FapiStateRunning));

        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
        if (cells_[cell_id] &&
            // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
            cells_[cell_id]->fapi_state.load(std::memory_order_acquire) ==
                    FapiStateT::FapiStateRunning) {

            PhyMacMsgDesc msg_desc;
            if (ipc_->tx_allocate(ipc_, &msg_desc, 0) < 0) {
                RT_LOGC_ERROR(Sample5gPipelines::Fapi5g, "Failed to allocate message descriptor");
                continue; // Skip this cell, try others
            }

            auto *fapi = add_scf_fapi_hdr<scf_fapi_slot_ind_t>(
                    msg_desc, SCF_FAPI_SLOT_INDICATION, static_cast<uint16_t>(cell_id), false);
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
            auto *rsp = reinterpret_cast<scf_fapi_slot_ind_t *>(fapi);

            // Apply L2 timing advance before sending SLOT.indication
            // The MAC operates L2_TIMING_ADVANCE slots ahead of the PHY for scheduling
            uint16_t advanced_sfn = current_sfn;
            uint16_t advanced_slot = current_slot + L2_TIMING_ADVANCE;
            if (advanced_slot >= common::NUM_SLOTS_PER_SF) {
                advanced_sfn++;
                advanced_slot -= common::NUM_SLOTS_PER_SF;
                if (advanced_sfn >= common::NUM_SFNS_PER_FRAME) {
                    advanced_sfn -= common::NUM_SFNS_PER_FRAME;
                }
            }

            // Reset accumulation done marker for new slot
            // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
            slot_accumulation_[advanced_slot].done.store(false, std::memory_order_release);

            RT_LOGEC_DEBUG(
                    Sample5gPipelines::Fapi5g,
                    FapiEvent::SlotIndication,
                    "Send SLOT.indication: cell_id={} sfn={} slot={} (current: sfn={} slot={}, "
                    "advance={})",
                    cell_id,
                    advanced_sfn,
                    advanced_slot,
                    current_sfn,
                    current_slot,
                    L2_TIMING_ADVANCE);

            RT_LOGC_INFO(
                    Sample5gPipelines::Fapi5g,
                    "Sending SLOT.indication: cell_id={} sfn={} slot={}",
                    cell_id,
                    advanced_sfn,
                    advanced_slot);

            rsp->sfn = advanced_sfn;
            rsp->slot = advanced_slot;
            ipc_->tx_send_msg(ipc_, &msg_desc);
            ipc_->tx_tti_sem_post(ipc_);
        }
    }

    // Advance slot counter atomically (no compare-and-swap needed - single writer)
    uint16_t next_sfn = current_sfn;
    uint16_t next_slot = current_slot + 1;
    if (next_slot == common::NUM_SLOTS_PER_SF) {
        next_sfn++;
        next_slot = 0;
        if (next_sfn == common::NUM_SFNS_PER_FRAME) {
            next_sfn = 0;
            // Increment SFN wrap counter for absolute slot calculation
            sfn_wrap_counter_.fetch_add(1, std::memory_order_release);
        }
    }
    sfn_slot_packed_.store(pack_sfn_slot(next_sfn, next_slot), std::memory_order_release);
}

scf_fapi_error_codes_t Sample5GPipeline::process_start_request(const uint16_t cell_id) {
    // cell_id already validated in process_message

    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
    if (cells_[cell_id] &&
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
        cells_[cell_id]->fapi_state.load(std::memory_order_acquire) ==
                FapiStateT::FapiStateConfigured) {

        // Check if this is the first cell starting (no cells currently running)
        const bool no_cells_running = (active_cell_bitmap_.load(std::memory_order_acquire) == 0);

        // Transition cell to running state (atomic)
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
        cells_[cell_id]->fapi_state.store(FapiStateT::FapiStateRunning, std::memory_order_release);

        // Set bit for this cell in the bitmap (atomic read-modify-write)
        active_cell_bitmap_.fetch_or(1ULL << cell_id, std::memory_order_acq_rel);

        const uint64_t new_bitmap = active_cell_bitmap_.load(std::memory_order_acquire);

        RT_LOGEC_INFO(
                Sample5gPipelines::Fapi5g,
                FapiEvent::StartRequest,
                "FAPI state set to FapiStateRunning for cell_id={}, active_cell_bitmap=0x{:X}",
                cell_id,
                new_bitmap);

        // If this is the first cell starting, initialize slot counter
        if (no_cells_running) {
            // Initialize slot counter atomically
            sfn_slot_packed_.store(pack_sfn_slot(0, 0), std::memory_order_release);
        }

        return SCF_ERROR_CODE_MSG_OK;
    } else {
        RT_LOGC_ERROR(
                Sample5gPipelines::Fapi5g,
                "FAPI is not in configured state. Cannot process start request for cell_id={}",
                cell_id);
        return SCF_ERROR_CODE_MSG_INVALID_STATE;
    }
}

scf_fapi_error_codes_t Sample5GPipeline::process_stop_request(const uint16_t cell_id) {
    // cell_id already validated in process_message

    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
    if (cells_[cell_id] &&
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
        cells_[cell_id]->fapi_state.load(std::memory_order_acquire) ==
                FapiStateT::FapiStateRunning) {

        // Transition cell to stopped state (atomic)
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
        cells_[cell_id]->fapi_state.store(FapiStateT::FapiStateStopped, std::memory_order_release);

        // Clear bit for this cell in the bitmap (atomic read-modify-write)
        active_cell_bitmap_.fetch_and(~(1ULL << cell_id), std::memory_order_acq_rel);

        const uint64_t new_bitmap = active_cell_bitmap_.load(std::memory_order_acquire);

        RT_LOGEC_INFO(
                Sample5gPipelines::Fapi5g,
                FapiEvent::StopRequest,
                "FAPI state set to FapiStateStopped for cell_id={}, active_cell_bitmap=0x{:X}",
                cell_id,
                new_bitmap);
        return SCF_ERROR_CODE_MSG_OK;
    } else {
        RT_LOGC_ERROR(
                Sample5gPipelines::Fapi5g,
                "FAPI is not in running state. Cannot process stop request for cell_id={}",
                cell_id);
        return SCF_ERROR_CODE_MSG_INVALID_STATE;
    }
}

bool Sample5GPipeline::send_error_indication(
        uint16_t cell_id, scf_fapi_message_id_e msg_id, scf_fapi_error_codes_t error_code) const {
    PhyMacMsgDesc msg_desc;
    if (ipc_->tx_allocate(ipc_, &msg_desc, 0) < 0) {
        RT_LOGC_ERROR(Sample5gPipelines::Fapi5g, "Failed to allocate message descriptor");
        return false;
    }
    RT_LOGEC_INFO(
            Sample5gPipelines::Fapi5g,
            FapiEvent::ErrorIndication,
            "Send ERROR.indication: cell_id={} msg_id={} error_code={:#04X}",
            cell_id,
            static_cast<uint32_t>(msg_id),
            static_cast<uint32_t>(error_code));
    auto *fapi = add_scf_fapi_hdr<scf_fapi_error_ind_t>(
            msg_desc, SCF_FAPI_ERROR_INDICATION, cell_id, false);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    auto *rsp = reinterpret_cast<scf_fapi_error_ind_t *>(fapi);

    // Load current sfn/slot atomically
    const uint32_t packed = sfn_slot_packed_.load(std::memory_order_acquire);
    rsp->sfn = unpack_sfn(packed);
    rsp->slot = unpack_slot(packed);
    rsp->msg_id = msg_id;
    rsp->err_code = error_code;
    ipc_->tx_send_msg(ipc_, &msg_desc);
    ipc_->tx_tti_sem_post(ipc_);
    return true;
}

void Sample5GPipeline::send_ul_indications(const std::size_t slot) {
    // Validate IPC context
    if (ipc_ == nullptr) {
        RT_LOGC_ERROR(
                Sample5gPipelines::Core, "Cannot send UL indications: IPC context not initialized");
        return;
    }

    // Retrieve the slot resources
    const driver::PuschSlotResources &slot_resources =
            driver_.pusch_pipeline_context.get_slot_resources(slot);

    // Check if host buffers are allocated
    if (slot_resources.host_buffers_index < 0) {
        RT_LOGEC_ERROR(
                Sample5gPipelines::Fapi5g,
                FapiEvent::CrcIndication,
                "Slot {}: No host buffers allocated",
                slot);
        return;
    }

    const driver::PuschHostInput &host_input =
            driver_.pusch_pipeline_context.get_host_input_by_index(
                    static_cast<std::size_t>(slot_resources.host_buffers_index));
    const auto &inputs = host_input.pusch_inputs;

    const driver::PuschHostOutput &host_output =
            driver_.pusch_pipeline_context.get_host_output_by_index(
                    static_cast<std::size_t>(slot_resources.host_buffers_index));
    const auto &outputs = host_output.pusch_outputs;

    RT_LOGEC_INFO(
            Sample5gPipelines::Fapi5g,
            FapiEvent::CrcIndication,
            "Send CRC indications: slot={} num_crcs={}",
            slot,
            inputs.ue_params_index);

    uint32_t offset = 0;
    uint32_t num_crc_success = 0;

    PhyMacMsgDesc msg_desc;
    if (ipc_->tx_allocate(ipc_, &msg_desc, 0) < 0) {
        RT_LOGC_ERROR(Sample5gPipelines::Fapi5g, "Failed to allocate message descriptor");
        return;
    }
    auto *fapi = add_scf_fapi_hdr<scf_fapi_crc_ind_t>(
            msg_desc, SCF_FAPI_CRC_INDICATION, inputs.ue_params[0].cell_id, false);

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    auto *rsp = reinterpret_cast<scf_fapi_crc_ind_t *>(fapi);
    rsp->sfn = inputs.ue_params[0].sfn;
    rsp->slot = static_cast<std::uint16_t>(slot);
    rsp->num_crcs = static_cast<std::uint16_t>(inputs.ue_params_index);

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    auto *next = reinterpret_cast<std::uint8_t *>(rsp->crc_info);

    for (std::size_t i = 0; i < inputs.ue_params_index; ++i) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        auto *crc_info = reinterpret_cast<scf_fapi_crc_info_t *>(next);

        const uint32_t crc_value = outputs.tb_crcs[i];
        const auto &ue_param = inputs.ue_params[i];

        crc_info->rnti = ue_param.rnti;
        crc_info->harq_id = ue_param.harq_process_id;
        crc_info->handle = ue_param.handle;
        crc_info->tb_crc_status = crc_value == 0 ? 0 : 1;
        crc_info->num_cb = 0;
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)
        next = crc_info->cb_crc_status;
        if (crc_value == 0) {
            num_crc_success++;
        }

        RT_LOGEC_INFO(
                Sample5gPipelines::Fapi5g,
                FapiEvent::CrcIndication,
                "UE param[{}]: cell_id={} rnti={} harq_id={} handle={} crc={}",
                i,
                ue_param.cell_id,
                ue_param.rnti,
                ue_param.harq_process_id,
                ue_param.handle,
                crc_value == 0 ? "PASS" : "FAIL");

        // UL_CQI, TA and RSSI for FAPI 10.02 support
        // NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers,cppcoreguidelines-pro-type-reinterpret-cast)
        auto *end = reinterpret_cast<scf_fapi_crc_end_info_t *>(next);
        end->timing_advance = 31;
        end->rssi = 0xFFFF;

        const float adjusted_snr = outputs.post_eq_sinr_db[i];
        const auto clamped_adjusted_snr = std::clamp(adjusted_snr, -64.0F, 63.0F);
        end->ul_cqi = static_cast<uint8_t>(std::round(clamped_adjusted_snr + 64.0F) * 2.0F);
        // NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers,cppcoreguidelines-pro-type-reinterpret-cast)

        RT_LOGC_INFO(
                Sample5gPipelines::Fapi5g,
                "UL_CQI: UE param[{}]: raw snr={} adjusted snr={} ul_cqi={}",
                i,
                adjusted_snr,
                clamped_adjusted_snr,
                end->ul_cqi);

        offset += sizeof(scf_fapi_crc_info_t) + sizeof(scf_fapi_crc_end_info_t);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        next += sizeof(scf_fapi_crc_end_info_t);
    }

    fapi->length = offset;
    msg_desc.msg_len = static_cast<std::int32_t>(
            fapi->length + sizeof(scf_fapi_body_header_t) + sizeof(scf_fapi_header_t));
    ipc_->tx_send_msg(ipc_, &msg_desc);
    ipc_->tx_tti_sem_post(ipc_);
    if (num_crc_success != 0) {

        send_rx_data_indication(slot, host_input, host_output);
        send_ul_noise_var_indication(slot, host_input, host_output);
    }
}
void Sample5GPipeline::send_rx_data_indication(
        const std::size_t slot,
        const driver::PuschHostInput &host_input,
        const driver::PuschHostOutput &host_output) {
    RT_LOGEC_INFO(
            Sample5gPipelines::Fapi5g,
            FapiEvent::RxDataIndication,
            "send_rx_data_indication(): Sending RX data indication");

    const auto &inputs = host_input.pusch_inputs;
    const auto &outputs = host_output.pusch_outputs;

    PhyMacMsgDesc msg_desc;
    msg_desc.data_pool = NV_IPC_MEMPOOL_CPU_DATA;
    if (ipc_->tx_allocate(ipc_, &msg_desc, 0) < 0) {
        RT_LOGC_ERROR(Sample5gPipelines::Fapi5g, "Failed to allocate message descriptor");
        return;
    }

    scf_fapi_body_header_t *hdr_ptr = add_scf_fapi_hdr<scf_fapi_rx_data_ind_t>(
            msg_desc, SCF_FAPI_RX_DATA_INDICATION, inputs.ue_params[0].cell_id, false);

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    auto &indication = *reinterpret_cast<scf_fapi_rx_data_ind_t *>(hdr_ptr);
    indication.sfn = inputs.ue_params[0].sfn;
    indication.slot = static_cast<std::uint16_t>(slot);
    indication.num_pdus = static_cast<std::uint16_t>(inputs.ue_params_index);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    auto *next = reinterpret_cast<uint8_t *>(indication.pdus);
    msg_desc.data_len = 0;
    uint32_t offset = 0;

    // NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast,cppcoreguidelines-pro-bounds-pointer-arithmetic,cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
    for (std::size_t i = 0; i < inputs.ue_params_index; ++i) {
        auto &ulsch_pdu = *(reinterpret_cast<scf_fapi_rx_data_pdu_t *>(next));
        ulsch_pdu.rnti = inputs.ue_params[i].rnti;
        ulsch_pdu.handle = inputs.ue_params[i].handle;
        ulsch_pdu.harq_id = inputs.ue_params[i].harq_process_id;

        ulsch_pdu.rssi = 0xFFFF;
        ulsch_pdu.timing_advance = 31;
        const float adjusted_snr = outputs.post_eq_sinr_db[i];
        const auto clamped_adjusted_snr = std::clamp(adjusted_snr, -64.0F, 63.0F);
        ulsch_pdu.ul_cqi = static_cast<uint8_t>(std::round(clamped_adjusted_snr + 64.0F) * 2.0F);

        if (outputs.tb_crcs[i] == 0) {
            // CRC PASS
            ulsch_pdu.pdu_len = inputs.ue_params[i].tb_size;
            // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
            msg_desc.data_len += static_cast<std::int32_t>(ulsch_pdu.pdu_len);
        } else {
            // CRC FAIL
            ulsch_pdu.pdu_len = 0;

            // Record CRC failure for this cell
            phy_stats_.record_crc_failure(inputs.ue_params[i].cell_id);
        }

        offset += sizeof(scf_fapi_rx_data_pdu_t);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)
        next = ulsch_pdu.pdu;
    }
    // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast,cppcoreguidelines-pro-bounds-pointer-arithmetic,cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

    hdr_ptr->length = offset;
    msg_desc.msg_len = static_cast<std::int32_t>(
            hdr_ptr->length + sizeof(scf_fapi_header_t) + sizeof(scf_fapi_body_header_t));
    std::copy(
            outputs.tb_payloads[0],
            // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
            outputs.tb_payloads[0] + msg_desc.data_len,
            static_cast<std::uint8_t *>(msg_desc.data_buf));
    ipc_->tx_send_msg(ipc_, &msg_desc);
    ipc_->tx_tti_sem_post(ipc_);
}

void Sample5GPipeline::send_ul_noise_var_indication(
        const std::size_t slot,
        const driver::PuschHostInput &host_input,
        const driver::PuschHostOutput &host_output) {

    const auto &inputs = host_input.pusch_inputs;
    const auto &outputs = host_output.pusch_outputs;

    RT_LOGEC_INFO(
            Sample5gPipelines::Fapi5g,
            FapiEvent::PuschNoiseVarIndication,
            "send_ul_noise_var_indication(): Sending UL noise variance indication for slot {} num "
            "ues={}",
            slot,
            inputs.ue_params_index);

    PhyMacMsgDesc msg_desc;
    if (ipc_->tx_allocate(ipc_, &msg_desc, 0) < 0) {
        RT_LOGC_ERROR(Sample5gPipelines::Fapi5g, "Failed to allocate message descriptor");
        return;
    }

    scf_fapi_body_header_t *hdr_ptr = add_scf_fapi_hdr<scf_fapi_rx_measurement_ind_t>(
            msg_desc, SCF_FAPI_RX_PE_NOISE_VARIANCE_INDICATION, inputs.ue_params[0].cell_id, false);

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    auto &indication = *reinterpret_cast<scf_fapi_rx_measurement_ind_t *>(hdr_ptr);
    indication.sfn = inputs.ue_params[0].sfn;
    indication.slot = static_cast<std::uint16_t>(slot);
    indication.num_meas = static_cast<std::uint16_t>(inputs.ue_params_index);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    auto *next = reinterpret_cast<uint8_t *>(indication.meas_info);
    uint32_t offset = 0;
    for (std::size_t i = 0; i < inputs.ue_params_index; ++i) {

        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        scf_fapi_meas_t &pn_meas_info = *(reinterpret_cast<scf_fapi_meas_t *>(next));
        pn_meas_info.handle = inputs.ue_params[i].handle;
        pn_meas_info.rnti = inputs.ue_params[i].rnti;
        float pn_dbm = outputs.post_eq_noise_var_db[i];
        // NOLINTBEGIN(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers,readability-avoid-nested-conditional-operator)
        pn_dbm = (pn_dbm < -152.0F) ? -152.0F : ((pn_dbm > 0.0F) ? 0.0F : pn_dbm);
        pn_meas_info.meas = static_cast<std::uint16_t>(std::round((pn_dbm + 152.0F) * 10.0F));
        // NOLINTEND(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers,readability-avoid-nested-conditional-operator)
        next += sizeof(scf_fapi_meas_t); // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        offset += sizeof(scf_fapi_meas_t);
        // Copy packed field value to avoid taking reference to unaligned member
        const std::uint16_t pn_meas_value = pn_meas_info.meas;
        RT_LOGC_INFO(
                Sample5gPipelines::Fapi5g,
                "send_ul_noise_var_indication(): UE param[{}]: rnti={} raw pn={} pn={}",
                i,
                inputs.ue_params[i].rnti,
                pn_dbm,
                pn_meas_value);
    }

    hdr_ptr->length = offset;
    msg_desc.msg_len = static_cast<std::int32_t>(
            hdr_ptr->length + sizeof(scf_fapi_header_t) + sizeof(scf_fapi_body_header_t));
    ipc_->tx_send_msg(ipc_, &msg_desc);
    ipc_->tx_tti_sem_post(ipc_);
}

ran::fapi::SlotInfo Sample5GPipeline::get_current_slot() const {
    // Atomically load the packed SFN/slot and unpack it
    const auto packed = sfn_slot_packed_.load(std::memory_order_acquire);
    return {.sfn = unpack_sfn(packed), .slot = unpack_slot(packed)};
}

std::span<const ran::fapi::CapturedFapiMessage>
Sample5GPipeline::get_accumulated_ul_tti_msgs(const std::uint16_t slot) const {
    // Sanity check - warn if accumulation not marked complete
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
    if (!slot_accumulation_[slot].done.load(std::memory_order_acquire)) {
        RT_LOGC_ERROR(
                Sample5gPipelines::Fapi5g,
                "get_accumulated_ul_tti_msgs() called before slot accumulation complete for slot "
                "{} - "
                "possible premature access!",
                slot);
        return {};
    }

    // Return non-owning view of accumulated messages
    // If on_graph_schedule_ callback is not set, this feature is disabled
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
    return slot_accumulation_[slot].messages;
}

std::uint64_t
Sample5GPipeline::get_current_absolute_slot(const ran::fapi::SlotInfo slot_info) const noexcept {
    const std::uint64_t sfn_wraps = sfn_wrap_counter_.load(std::memory_order_acquire);
    const std::uint64_t total_frames = sfn_wraps * common::NUM_SFNS_PER_FRAME + slot_info.sfn;
    return total_frames * common::NUM_SLOTS_PER_SF + slot_info.slot;
}

void Sample5GPipeline::launch_pipelines(const std::size_t slot) {
    // Delegate to Driver to launch PUSCH pipelines
    driver_.launch_pipelines(slot);
}

} // namespace ran::message_adapter
