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
 * @file fapi_state.cpp
 * @brief Implementation of FapiState FAPI message processor
 *
 * Implements NVIPC initialization, FAPI message processing, per-cell state
 * machine transitions, and message transmission for 5G NR PHY-MAC interface.
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <iterator>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nv_ipc.h>
#include <nv_ipc.hpp>
#include <quill/LogMacros.h>
#include <scf_5g_fapi.h>
#include <unistd.h>
#include <yaml.hpp>

#include <gsl-lite/gsl-lite.hpp>
#include <wise_enum.h>

#include "fapi/fapi_buffer.hpp"
#include "fapi/fapi_log.hpp"
#include "fapi/fapi_state.hpp"
#include "log/rt_log_macros.hpp"

namespace ran::fapi {

// Anonymous namespace for internal helper functions
namespace {

/**
 * Convert FAPI message ID to string representation
 *
 * @param[in] msg_id FAPI message ID
 * @return String name of the message type
 */
[[nodiscard]] std::string_view fapi_message_id_to_string(const int32_t msg_id) noexcept {
    switch (msg_id) {
    case SCF_FAPI_CONFIG_REQUEST:
        return "CONFIG.request";
    case SCF_FAPI_CONFIG_RESPONSE:
        return "CONFIG.response";
    case SCF_FAPI_START_REQUEST:
        return "START.request";
    case SCF_FAPI_STOP_REQUEST:
        return "STOP.request";
    case SCF_FAPI_STOP_INDICATION:
        return "STOP.indication";
    case SCF_FAPI_ERROR_INDICATION:
        return "ERROR.indication";
    case SCF_FAPI_DL_TTI_REQUEST:
        return "DL_TTI.request";
    case SCF_FAPI_UL_TTI_REQUEST:
        return "UL_TTI.request";
    case SCF_FAPI_SLOT_INDICATION:
        return "SLOT.indication";
    case SCF_FAPI_UL_DCI_REQUEST:
        return "UL_DCI.request";
    case SCF_FAPI_TX_DATA_REQUEST:
        return "TX_DATA.request";
    case SCF_FAPI_SLOT_RESPONSE:
        return "SLOT.response";
    default:
        return "UNKNOWN";
    }
}

/**
 * Create unique temporary YAML file path
 *
 * @param prefix File name prefix
 * @return Unique temporary file path using PID and timestamp
 */
std::filesystem::path create_unique_temp_yaml_file(const std::string &prefix) {
    const std::filesystem::path temp_dir = std::filesystem::temp_directory_path();
    const auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    return temp_dir / std::format("{}_{}_{}_.yaml", prefix, ::getpid(), now);
}

/**
 * Initialize NVIPC from config file path
 */
nv_ipc_t *init_nvipc_from_file(const std::string &config_file) {
    if (!std::filesystem::exists(config_file)) {
        RT_LOGC_ERROR(FapiComponent::FapiCore, "Config file not found: {}", config_file);
        return nullptr;
    }

    try {
        yaml::file_parser parser(config_file.c_str());
        yaml::document doc = parser.next_document();
        const yaml::node root = doc.root();
        yaml::node transport_node = root["transport"];

        nv_ipc_config_t config{};
        const nv_ipc_module_t module_type = NV_IPC_MODULE_PHY;
        nv_ipc_parse_yaml_node(&config, &transport_node, module_type);

        nv_ipc_t *ipc = create_nv_ipc_interface(&config);
        if (ipc == nullptr) {
            RT_LOGC_ERROR(
                    FapiComponent::FapiCore,
                    "Failed to create NVIPC interface from config file: {}",
                    config_file);
        }

        return ipc;
    } catch (const std::exception &e) {
        RT_LOGC_ERROR(
                FapiComponent::FapiCore,
                "Failed to initialize NVIPC from config file {}: {}",
                config_file,
                e.what());
        return nullptr;
    }
}

/**
 * Initialize NVIPC from YAML string
 */
nv_ipc_t *init_nvipc_from_string(const std::string &yaml_string) {
    if (yaml_string.empty()) {
        RT_LOGC_ERROR(FapiComponent::FapiCore, "YAML config string is empty");
        return nullptr;
    }

    // YAML library only supports file parsing, so write to temp file
    // Use unique file name to avoid collisions between concurrent processes
    const std::filesystem::path temp_file = create_unique_temp_yaml_file("fapi_nvipc_config");

    // Setup cleanup guard to ensure temp file is removed
    const auto cleanup = gsl_lite::finally([&temp_file] { std::filesystem::remove(temp_file); });

    try {
        // Write string to temp file
        std::ofstream ofs(temp_file);
        if (!ofs) {
            RT_LOGC_ERROR(
                    FapiComponent::FapiCore,
                    "Failed to create temp file for YAML string config: {}",
                    temp_file.string());
            return nullptr;
        }
        ofs << yaml_string;
        ofs.close();

        // Parse from temp file
        yaml::file_parser parser(temp_file.c_str());
        yaml::document doc = parser.next_document();
        const yaml::node root = doc.root();
        yaml::node transport_node = root["transport"];

        nv_ipc_config_t config{};
        const nv_ipc_module_t module_type = NV_IPC_MODULE_PHY;
        nv_ipc_parse_yaml_node(&config, &transport_node, module_type);

        nv_ipc_t *ipc = create_nv_ipc_interface(&config);
        if (ipc == nullptr) {
            RT_LOGC_ERROR(
                    FapiComponent::FapiCore,
                    "Failed to create NVIPC interface from YAML string config");
        }

        return ipc;
    } catch (const std::exception &e) {
        RT_LOGC_ERROR(
                FapiComponent::FapiCore,
                "Failed to initialize NVIPC from YAML string config: {}",
                e.what());
        return nullptr;
    }
}

/**
 * Validate cell ID is within bounds
 */
bool is_valid_cell_id(const uint16_t cell_id, const std::size_t max_cells) {
    return cell_id < max_cells;
}

// clang-format off
/**
 * Validate state transition is allowed by state machine
 *
 * @param[in] from Current state
 * @param[in] to Target state
 * @return true if transition is valid, false otherwise
 */
bool is_valid_transition(const FapiStateT from, const FapiStateT to) {
    using enum FapiStateT;
    switch (from) {
    case FapiStateIdle:       return to == FapiStateConfigured;
    case FapiStateConfigured: return to == FapiStateConfigured || to == FapiStateRunning;
    case FapiStateRunning:    return to == FapiStateStopped;
    case FapiStateStopped:    return to == FapiStateConfigured;
    default: return false;
    }
}
// clang-format on

/**
 * Parameters for building FAPI message headers
 */
struct FapiHeaderParams {
    uint16_t msg_id;    //!< FAPI message type ID
    uint16_t cell_id;   //!< Cell identifier
    uint32_t body_size; //!< Size of message body payload
};

/**
 * Validate FAPI message header
 *
 * @param[in] hdr FAPI message header to validate
 * @param[in] expected_cell_id Expected cell ID from message
 * @return Error code indicating validation result
 */
scf_fapi_error_codes_t
validate_message_header(const scf_fapi_header_t &hdr, const int32_t expected_cell_id) {
    // Copy values before logging so they don't reference the nvipc message buffer
    const auto handle_id = hdr.handle_id;
    const auto message_count = hdr.message_count;

    if (handle_id != expected_cell_id) {
        RT_LOGC_ERROR(
                FapiComponent::FapiMsg,
                "Header handle_id={} does not match cell_id={}",
                handle_id,
                expected_cell_id);
        return SCF_ERROR_CODE_MSG_INVALID_CONFIG;
    }

    if (message_count != 1) {
        RT_LOGC_ERROR(
                FapiComponent::FapiMsg, "Invalid message_count={}, expected 1", message_count);
        return SCF_ERROR_CODE_MSG_INVALID_CONFIG;
    }

    return SCF_ERROR_CODE_MSG_OK;
}

/**
 * Validate FAPI message length
 *
 * @param[in] actual_msg_len Actual message length from nv_ipc_msg_t
 * @param[in] body_hdr Body header containing expected body length
 * @param[in] cell_id Cell ID for error logging
 * @param[in] msg_id Message ID for error logging
 * @return Error code indicating validation result
 */
scf_fapi_error_codes_t validate_message_length(
        const int32_t actual_msg_len,
        const scf_fapi_body_header_t &body_hdr,
        const int32_t cell_id,
        const int32_t msg_id) {
    const uint32_t head_len = sizeof(scf_fapi_header_t) + sizeof(scf_fapi_body_header_t);
    const uint32_t body_len = body_hdr.length;
    const uint32_t expected_len = head_len + body_len;

    if (static_cast<uint32_t>(actual_msg_len) != expected_len) {
        RT_LOGC_ERROR(
                FapiComponent::FapiMsg,
                "Invalid msg_len: cell_id={} msg_id={} msg_len={} expected={}",
                cell_id,
                msg_id,
                actual_msg_len,
                expected_len);
        return SCF_ERROR_CODE_MSG_INVALID_CONFIG;
    }

    return SCF_ERROR_CODE_MSG_OK;
}

/**
 * Build FAPI message header
 */
scf_fapi_body_header_t *build_fapi_header(nv_ipc_msg_t &msg, const FapiHeaderParams &params) {
    // Precondition: cell_id must fit in uint8_t (FAPI spec limitation)
    static constexpr uint16_t MAX_FAPI_CELL_ID = 256;
    gsl_Expects(params.cell_id < MAX_FAPI_CELL_ID);

    auto *hdr = assume_cast<scf_fapi_header_t>(msg.msg_buf);
    hdr->message_count = 1;
    hdr->handle_id = static_cast<uint8_t>(params.cell_id);

    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)
    auto *body = assume_cast<scf_fapi_body_header_t>(hdr->payload);
    body->type_id = params.msg_id;
    body->length = params.body_size;

    msg.msg_id = static_cast<int32_t>(params.msg_id);
    msg.cell_id = static_cast<int32_t>(params.cell_id);
    msg.msg_len = static_cast<int32_t>(
            sizeof(scf_fapi_header_t) + sizeof(scf_fapi_body_header_t) + params.body_size);
    msg.data_len = 0;

    return body;
}

} // anonymous namespace

std::string FapiState::extract_nvipc_prefix(const std::string &yaml_content) {
    // YAML library only supports file parsing, so write to temp file
    // Use unique file name to avoid collisions between concurrent processes
    const std::filesystem::path temp_file = create_unique_temp_yaml_file("fapi_prefix_extract");

    // Setup cleanup guard to ensure temp file is removed
    const auto cleanup = gsl_lite::finally([&temp_file] { std::filesystem::remove(temp_file); });

    try {
        // Write YAML string to temp file
        std::ofstream ofs(temp_file);
        if (!ofs) {
            RT_LOGC_WARN(
                    FapiComponent::FapiCore,
                    "Failed to create temp file for prefix extraction: {}",
                    temp_file.string());
            return "";
        }
        ofs << yaml_content;
        ofs.close();

        // Parse YAML and extract prefix
        yaml::file_parser parser(temp_file.c_str());
        yaml::document doc = parser.next_document();
        const yaml::node root = doc.root();
        const yaml::node transport_node = root["transport"];
        const yaml::node shm_config_node = transport_node["shm_config"];
        yaml::node prefix_node = shm_config_node["prefix"];
        const std::string prefix = prefix_node.as<std::string>();

        RT_LOGC_DEBUG(FapiComponent::FapiCore, "Extracted NVIPC prefix: '{}'", prefix);

        return prefix;
    } catch (const std::exception &e) {
        RT_LOGC_WARN(
                FapiComponent::FapiCore, "Failed to extract NVIPC prefix from YAML: {}", e.what());
        return "";
    }
}

void FapiState::NvIpcDeleter::operator()(nv_ipc_t *ipc) const noexcept {
    if (ipc != nullptr) {
        RT_LOGC_INFO(FapiComponent::FapiCore, "Destroying NVIPC interface");
        ipc->ipc_destroy(ipc);
    }

    // Clean up /dev/shm files created by NVIPC
    if (prefix.empty()) {
        return;
    }

    RT_LOGC_DEBUG(FapiComponent::FapiCore, "Cleaning up /dev/shm files for prefix '{}'", prefix);
    const std::filesystem::path shm_dir = "/dev/shm";
    try {
        std::size_t files_removed{0};
        for (const auto &entry : std::filesystem::directory_iterator(shm_dir)) {
            const std::string filename = entry.path().filename().string();
            if (filename.find(prefix) != std::string::npos) {
                RT_LOGC_DEBUG(
                        FapiComponent::FapiCore,
                        "Removing /dev/shm file: {}",
                        entry.path().string());
                std::filesystem::remove(entry.path());
                ++files_removed;
            }
        }
        RT_LOGC_DEBUG(
                FapiComponent::FapiCore,
                "Cleaned up {} /dev/shm file(s) for prefix '{}'",
                files_removed,
                prefix);
    } catch (const std::exception &e) {
        // Log but don't fail if cleanup fails
        RT_LOGC_ERROR(
                FapiComponent::FapiCore,
                "Failed to clean up /dev/shm for prefix '{}': {}",
                prefix,
                e.what());
    }
}

FapiState::FapiState(const InitParams &params) : params_(params), cells_(params.max_cells) {
    RT_LOGC_INFO(
            FapiComponent::FapiCore,
            "Initializing FapiState: max_cells={}, max_sfn={}, max_slot={}",
            params_.max_cells,
            params_.max_sfn,
            params_.max_slot);

    // Determine YAML content source and extract prefix
    std::string yaml_content;
    if (!params_.nvipc_config_file.empty()) {
        RT_LOGC_DEBUG(
                FapiComponent::FapiCore,
                "Initializing NVIPC from file: {}",
                params_.nvipc_config_file);
        std::ifstream ifs(params_.nvipc_config_file);
        if (!ifs) {
            throw std::runtime_error(
                    "Failed to open NVIPC config file: " + params_.nvipc_config_file);
        }
        // The code above already validates if ifs is invalid, so we can safely ignore the warning
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnull-dereference"
        yaml_content =
                std::string(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
#pragma GCC diagnostic pop
    } else if (!params_.nvipc_config_string.empty()) {
        RT_LOGC_DEBUG(FapiComponent::FapiCore, "Initializing NVIPC from string");
        yaml_content = params_.nvipc_config_string;
    } else {
        throw std::runtime_error(
                "Either nvipc_config_file or nvipc_config_string must be provided");
    }

    // Extract NVIPC prefix for cleanup
    nvipc_prefix_ = extract_nvipc_prefix(yaml_content);

    // Initialize NVIPC
    // cppcheck-suppress constVariablePointer
    nv_ipc_t *raw_ipc{};
    if (!params_.nvipc_config_file.empty()) {
        raw_ipc = init_nvipc_from_file(params_.nvipc_config_file);
        if (raw_ipc == nullptr) {
            throw std::runtime_error(std::format(
                    "Failed to initialize NVIPC from file: {}", params_.nvipc_config_file));
        }
    } else {
        raw_ipc = init_nvipc_from_string(params_.nvipc_config_string);
        if (raw_ipc == nullptr) {
            throw std::runtime_error("Failed to initialize NVIPC from string");
        }
    }

    // Create unique_ptr with custom deleter that knows the prefix
    ipc_ = std::unique_ptr<nv_ipc_t, NvIpcDeleter>(raw_ipc, NvIpcDeleter{nvipc_prefix_});
    RT_LOGC_DEBUG(FapiComponent::FapiCore, "FapiState initialized successfully");
}

scf_fapi_error_codes_t FapiState::process_message(nv_ipc_msg_t &msg) {
    // Parse FAPI message header
    auto *hdr = assume_cast<scf_fapi_header_t>(msg.msg_buf);
    const scf_fapi_error_codes_t header_result = validate_message_header(*hdr, msg.cell_id);
    if (header_result != SCF_ERROR_CODE_MSG_OK) {
        return header_result;
    }

    // Parse body header
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)
    auto &body_hdr = *assume_cast<scf_fapi_body_header_t>(hdr->payload);

    // Validate message length
    const scf_fapi_error_codes_t length_result =
            validate_message_length(msg.msg_len, std::as_const(body_hdr), msg.cell_id, msg.msg_id);
    if (length_result != SCF_ERROR_CODE_MSG_OK) {
        return length_result;
    }

    const uint16_t type_id = body_hdr.type_id;
    const auto cell_id = static_cast<uint16_t>(msg.cell_id);
    const uint32_t body_len = body_hdr.length;

    RT_LOGC_DEBUG(
            FapiComponent::FapiMsg, "Received message type_id={} cell_id={}", type_id, cell_id);

    // Call generic capture callback first (before routing)
    if (on_message_) {
        invoke_message_callback(cell_id, type_id, body_hdr, body_len, msg);
    }

    // Route message to appropriate handler
    switch (type_id) {
    case SCF_FAPI_CONFIG_REQUEST:
    case SCF_FAPI_START_REQUEST:
    case SCF_FAPI_STOP_REQUEST:
        return handle_config_start_stop_request(cell_id, type_id, body_hdr, body_len);

    case SCF_FAPI_UL_TTI_REQUEST:
        RT_LOGC_DEBUG(FapiComponent::FapiMsg, "UL_TTI_REQUEST for cell {}", cell_id);
        if (on_ul_tti_request_) {
            on_ul_tti_request_(cell_id, std::as_const(body_hdr), body_len);
        }
        return SCF_ERROR_CODE_MSG_OK;

    case SCF_FAPI_DL_TTI_REQUEST:
        RT_LOGC_DEBUG(FapiComponent::FapiMsg, "DL_TTI_REQUEST for cell {}", cell_id);
        if (on_dl_tti_request_) {
            on_dl_tti_request_(cell_id, std::as_const(body_hdr), body_len);
        }
        return SCF_ERROR_CODE_MSG_OK;

    case SCF_FAPI_SLOT_RESPONSE:
        process_slot_response(cell_id, std::as_const(body_hdr), body_len);
        if (on_slot_response_) {
            on_slot_response_(cell_id, std::as_const(body_hdr), body_len);
        }
        return SCF_ERROR_CODE_MSG_OK;

    default:
        RT_LOGC_WARN(FapiComponent::FapiMsg, "Ignoring message type_id={}", type_id);
        return SCF_ERROR_CODE_MSG_INVALID_CONFIG;
    }
}

void FapiState::invoke_message_callback(
        const uint16_t cell_id,
        const uint16_t type_id,
        const scf_fapi_body_header_t &body_hdr,
        const uint32_t body_len,
        const nv_ipc_msg_t &msg) const {
    const auto *body_start = assume_cast<uint8_t>(&body_hdr);
    const uint32_t total_msg_len = sizeof(scf_fapi_body_header_t) + body_len;

    const FapiMessageData msg_data{
            .cell_id = cell_id,
            .msg_id = static_cast<scf_fapi_message_id_e>(type_id),
            .msg_buf = std::span<const uint8_t>{body_start, total_msg_len},
            .data_buf =
                    (msg.data_buf != nullptr) && msg.data_len > 0
                            ? std::span<
                                      const uint8_t>{assume_cast<uint8_t>(msg.data_buf), gsl_lite::narrow_cast<std::size_t>(msg.data_len)}
                            : std::span<const uint8_t>{}};
    on_message_(msg_data);
}

scf_fapi_error_codes_t FapiState::handle_config_start_stop_request(
        const uint16_t cell_id,
        const uint16_t type_id,
        scf_fapi_body_header_t &body_hdr,
        const uint32_t body_len) {
    switch (type_id) {
    case SCF_FAPI_CONFIG_REQUEST: {
        auto &config_req = assume_cast_ref<scf_fapi_config_request_msg_t>(body_hdr);
        const auto result = process_config_request(cell_id, config_req, body_len);
        if (!send_config_response(cell_id, result)) {
            RT_LOGC_ERROR(
                    FapiComponent::FapiMsg, "Failed to send CONFIG.response for cell {}", cell_id);
        }
        return result;
    }

    case SCF_FAPI_START_REQUEST: {
        const auto result = process_start_request(cell_id, std::as_const(body_hdr), body_len);
        if (result != SCF_ERROR_CODE_MSG_OK) {
            if (!send_error_indication(cell_id, SCF_FAPI_START_REQUEST, result)) {
                RT_LOGC_ERROR(
                        FapiComponent::FapiMsg,
                        "Failed to send ERROR.indication for cell {}",
                        cell_id);
            }
        }
        return result;
    }

    case SCF_FAPI_STOP_REQUEST: {
        const auto result = process_stop_request(cell_id, std::as_const(body_hdr), body_len);
        if (result == SCF_ERROR_CODE_MSG_OK) {
            if (!send_stop_indication(cell_id)) {
                RT_LOGC_ERROR(
                        FapiComponent::FapiMsg,
                        "Failed to send STOP.indication for cell {}",
                        cell_id);
            }
        } else {
            if (!send_error_indication(cell_id, SCF_FAPI_STOP_REQUEST, result)) {
                RT_LOGC_ERROR(
                        FapiComponent::FapiMsg,
                        "Failed to send ERROR.indication for cell {}",
                        cell_id);
            }
        }
        return result;
    }

    default:
        // Should never reach here as caller validates type_id
        RT_LOGC_ERROR(FapiComponent::FapiMsg, "Unexpected message type_id={}", type_id);
        return SCF_ERROR_CODE_MSG_INVALID_CONFIG;
    }
}

scf_fapi_error_codes_t FapiState::process_config_request(
        const uint16_t cell_id,
        scf_fapi_config_request_msg_t &config_msg,
        const uint32_t body_len) {
    if (!is_valid_cell_id(cell_id, params_.max_cells)) {
        RT_LOGC_ERROR(FapiComponent::FapiState, "Invalid cell_id={}", cell_id);
        return SCF_ERROR_CODE_MSG_INVALID_PHY_ID;
    }

    auto &cell = cells_[cell_id];

    // Validate state transition
    const FapiStateT current_state = cell.state.load();
    if (!is_valid_transition(current_state, FapiStateT::FapiStateConfigured)) {
        RT_LOGC_ERROR(
                FapiComponent::FapiState,
                "Invalid state transition: {} -> Configured for cell {}",
                ::wise_enum::to_string(current_state),
                cell_id);
        return SCF_ERROR_CODE_MSG_INVALID_STATE;
    }

    // Check capacity only for new configuration (not reconfiguration)
    if (current_state == FapiStateT::FapiStateIdle && num_cells_configured_ >= params_.max_cells) {
        RT_LOGC_ERROR(FapiComponent::FapiState, "Cell capacity exceeded");
        return SCF_ERROR_CODE_MSG_INVALID_CONFIG;
    }

    const bool is_reconfiguration = (current_state == FapiStateT::FapiStateConfigured);

    RT_LOGC_INFO(
            FapiComponent::FapiState,
            "Processing CONFIG.request for cell_id={} (current_state={}, is_reconfig={})",
            cell_id,
            ::wise_enum::to_string(current_state),
            is_reconfiguration);

    // Invoke callback if registered - callback can validate/reject config
    if (on_config_request_) {
        const auto &body_hdr = std::as_const(config_msg).msg_hdr;
        const scf_fapi_error_codes_t callback_result =
                on_config_request_(cell_id, body_hdr, body_len);
        if (callback_result != SCF_ERROR_CODE_MSG_OK) {
            RT_LOGC_WARN(
                    FapiComponent::FapiState,
                    "CONFIG.request callback rejected for cell_id={} with error={:#04X}",
                    cell_id,
                    static_cast<uint32_t>(callback_result));
            return callback_result;
        }
    }

    // Parse TLVs and extract configuration
    uint32_t consumed_body_len = sizeof(config_msg.msg_body.num_tlvs);
    uint16_t num_tlvs = config_msg.msg_body.num_tlvs;

    RT_LOGC_DEBUG(FapiComponent::FapiState, "Parsing {} TLVs", num_tlvs);

    // Create span for TLV area starting after num_tlvs field
    const std::span<std::byte> tlv_buffer =
            make_buffer_span(&config_msg.msg_body.tlvs[0], body_len - consumed_body_len);

    std::size_t tlv_offset{0};
    while (num_tlvs != 0 && consumed_body_len < body_len) {
        // Check if buffer has enough bytes for TLV header
        if (tlv_offset + sizeof(scf_fapi_tl_t) > tlv_buffer.size()) {
            RT_LOGC_ERROR(
                    FapiComponent::FapiState,
                    "Malformed CONFIG.request: insufficient bytes for TLV header");
            return SCF_ERROR_CODE_MSG_INVALID_CONFIG;
        }

        const std::span<std::byte> current_tlv_span = tlv_buffer.subspan(tlv_offset);
        auto *tlv_hdr = assume_cast<scf_fapi_tl_t>(current_tlv_span.data());

        // Calculate total TLV length (header + rounded payload)
        const std::size_t tlv_total_len =
                sizeof(scf_fapi_tl_t) + static_cast<std::size_t>(((tlv_hdr->length + 3) / 4) * 4);

        // Check if buffer has enough bytes for complete TLV (header + payload)
        if (tlv_offset + tlv_total_len > tlv_buffer.size()) {
            RT_LOGC_ERROR(
                    FapiComponent::FapiState,
                    "Malformed CONFIG.request: insufficient bytes for TLV payload");
            return SCF_ERROR_CODE_MSG_INVALID_CONFIG;
        }

        switch (tlv_hdr->tag) {
        case CONFIG_TLV_NUM_RX_ANT:
            cell.num_rx_ant = tlv_hdr->AsValue<uint16_t>();
            RT_LOGC_DEBUG(
                    FapiComponent::FapiState, "Cell {} num_rx_ant={}", cell_id, cell.num_rx_ant);
            break;
        case CONFIG_TLV_PHY_CELL_ID:
            cell.phy_cell_id = tlv_hdr->AsValue<uint16_t>();
            RT_LOGC_DEBUG(
                    FapiComponent::FapiState, "Cell {} phy_cell_id={}", cell_id, cell.phy_cell_id);
            break;
        default: {
            const uint16_t tag = tlv_hdr->tag;
            RT_LOGC_DEBUG(FapiComponent::FapiState, "Ignoring TLV tag: {}", tag);
        } break;
        }

        tlv_offset += tlv_total_len;
        consumed_body_len += tlv_total_len;
        num_tlvs--;

        if (consumed_body_len > body_len) {
            RT_LOGC_ERROR(
                    FapiComponent::FapiState, "Malformed CONFIG.request causing buffer overrun");
            return SCF_ERROR_CODE_MSG_INVALID_CONFIG;
        }
    }

    // Transition to configured
    cell.state.store(FapiStateT::FapiStateConfigured);
    cell.cell_id = cell_id;

    // Increment counter only on first-time configuration, not reconfiguration
    if (!is_reconfiguration) {
        num_cells_configured_++;
    }

    RT_LOGC_DEBUG(
            FapiComponent::FapiState,
            "Cell {} {}, total cells configured={}",
            cell_id,
            is_reconfiguration ? "reconfigured" : "configured",
            num_cells_configured_.load());

    return SCF_ERROR_CODE_MSG_OK;
}

scf_fapi_error_codes_t FapiState::process_start_request(
        const uint16_t cell_id, const scf_fapi_body_header_t &body_hdr, const uint32_t body_len) {
    if (!is_valid_cell_id(cell_id, params_.max_cells)) {
        return SCF_ERROR_CODE_MSG_INVALID_PHY_ID;
    }

    auto &cell = cells_[cell_id];

    // Validate state transition
    const FapiStateT current_state = cell.state.load();
    if (!is_valid_transition(current_state, FapiStateT::FapiStateRunning)) {
        RT_LOGC_ERROR(
                FapiComponent::FapiState,
                "Invalid state transition: {} -> Running for cell {}",
                ::wise_enum::to_string(current_state),
                cell_id);
        return SCF_ERROR_CODE_MSG_INVALID_STATE;
    }

    // Invoke callback if registered - callback can validate/reject start
    if (on_start_request_) {
        const scf_fapi_error_codes_t callback_result =
                on_start_request_(cell_id, body_hdr, body_len);
        if (callback_result != SCF_ERROR_CODE_MSG_OK) {
            RT_LOGC_WARN(
                    FapiComponent::FapiState,
                    "START.request callback rejected for cell_id={} with error={:#04X}",
                    cell_id,
                    static_cast<uint32_t>(callback_result));
            return callback_result;
        }
    }

    cell.state.store(FapiStateT::FapiStateRunning);
    num_cells_running_++;

    RT_LOGC_DEBUG(
            FapiComponent::FapiState,
            "Cell {} started, total running={}",
            cell_id,
            num_cells_running_.load());

    return SCF_ERROR_CODE_MSG_OK;
}

scf_fapi_error_codes_t FapiState::process_stop_request(
        const uint16_t cell_id, const scf_fapi_body_header_t &body_hdr, const uint32_t body_len) {
    if (!is_valid_cell_id(cell_id, params_.max_cells)) {
        return SCF_ERROR_CODE_MSG_INVALID_PHY_ID;
    }

    auto &cell = cells_[cell_id];

    // Validate state transition
    const FapiStateT current_state = cell.state.load();
    if (!is_valid_transition(current_state, FapiStateT::FapiStateStopped)) {
        RT_LOGC_ERROR(
                FapiComponent::FapiState,
                "Invalid state transition: {} -> Stopped for cell {}",
                ::wise_enum::to_string(current_state),
                cell_id);
        return SCF_ERROR_CODE_MSG_INVALID_STATE;
    }

    // Invoke callback if registered - callback can validate/reject stop
    if (on_stop_request_) {
        const scf_fapi_error_codes_t callback_result =
                on_stop_request_(cell_id, body_hdr, body_len);
        if (callback_result != SCF_ERROR_CODE_MSG_OK) {
            RT_LOGC_WARN(
                    FapiComponent::FapiState,
                    "STOP.request callback rejected for cell_id={} with error={:#04X}",
                    cell_id,
                    static_cast<uint32_t>(callback_result));
            return callback_result;
        }
    }

    cell.state.store(FapiStateT::FapiStateStopped);
    num_cells_running_--;

    RT_LOGC_DEBUG(
            FapiComponent::FapiState,
            "Cell {} stopped, total running={}",
            cell_id,
            num_cells_running_.load());

    return SCF_ERROR_CODE_MSG_OK;
}

void FapiState::process_slot_response(
        const uint16_t cell_id,
        const scf_fapi_body_header_t &body_hdr,
        const uint32_t body_len) const {
    if (!is_valid_cell_id(cell_id, params_.max_cells)) {
        RT_LOGC_ERROR(FapiComponent::FapiMsg, "Invalid cell_id={}", cell_id);
        return;
    }

    // Validate message length for slot response structure
    constexpr std::size_t EXPECTED_LEN =
            sizeof(scf_fapi_slot_rsp_t) - sizeof(scf_fapi_body_header_t);
    if (body_len != EXPECTED_LEN) {
        RT_LOGC_WARN(
                FapiComponent::FapiMsg,
                "SLOT_RESPONSE length mismatch: cell={} expected={} got={}",
                cell_id,
                EXPECTED_LEN,
                body_len);
    }

    // Body header is the start of slot_rsp_t structure
    const auto &slot_rsp = assume_cast_ref<scf_fapi_slot_rsp_t>(body_hdr);

    // Copy values before logging so they don't reference the nvipc message buffer
    const auto sfn = slot_rsp.sfn;
    const auto slot = slot_rsp.slot;

    RT_LOGC_DEBUG(
            FapiComponent::FapiMsg, "SLOT_RESPONSE: cell={} sfn={} slot={}", cell_id, sfn, slot);
}

void FapiState::increment_slot() {
    // Lock-free increment using compare-and-swap on packed slot representation
    static constexpr uint32_t SLOT_MASK = 0xFFFFU;
    static constexpr uint32_t SFN_SHIFT = 16U;

    const uint16_t max_slot = params_.max_slot;
    const uint16_t max_sfn = params_.max_sfn;

    uint32_t old_packed = current_slot_packed_.load(std::memory_order_relaxed);
    uint32_t new_packed{};

    // NOLINTNEXTLINE(cppcoreguidelines-avoid-do-while) - Standard CAS loop pattern
    do {
        // Unpack current values
        const uint16_t current_slot = old_packed & SLOT_MASK;
        const uint16_t current_sfn = (old_packed >> SFN_SHIFT) & SLOT_MASK;

        // Calculate new values with wraparound
        uint16_t new_slot = current_slot + 1;
        uint16_t new_sfn = current_sfn;

        if (new_slot >= max_slot) {
            new_slot = 0;
            new_sfn = current_sfn + 1;
            if (new_sfn >= max_sfn) {
                new_sfn = 0;
            }
        }

        // Pack new values
        new_packed = (static_cast<uint32_t>(new_sfn) << SFN_SHIFT) | new_slot;

        // CAS loop: retry if another thread modified current_slot_packed_
    } while (!current_slot_packed_.compare_exchange_weak(
            old_packed, new_packed, std::memory_order_release, std::memory_order_relaxed));
}

void FapiState::reset_slot() {
    // Atomically reset both sfn and slot to 0
    current_slot_packed_.store(0, std::memory_order_release);
}

SlotInfo FapiState::get_current_slot() const noexcept {
    // Atomically read packed slot and unpack
    static constexpr uint32_t SLOT_MASK = 0xFFFFU;
    static constexpr uint32_t SFN_SHIFT = 16U;

    const uint32_t packed = current_slot_packed_.load(std::memory_order_acquire);
    return SlotInfo{
            .sfn = static_cast<uint16_t>((packed >> SFN_SHIFT) & SLOT_MASK),
            .slot = static_cast<uint16_t>(packed & SLOT_MASK)};
}

bool FapiState::send_slot_indication() {
    // Atomically capture current slot to avoid torn reads
    const SlotInfo slot = get_current_slot();

    // Find the first running cell to send SLOT.indication
    // SLOT.indication is a time synchronization message sent once per slot, not per cell
    const auto it = std::find_if(cells_.begin(), cells_.end(), [](const CellConfig &cell) {
        return cell.state.load() == FapiStateT::FapiStateRunning;
    });

    // No running cells, cannot send
    if (it == cells_.end()) {
        return false;
    }

    const uint16_t target_cell_id = it->cell_id;

    nv_ipc_msg_t msg{};
    if (allocate_message(msg) < 0) {
        RT_LOGC_ERROR(FapiComponent::FapiMsg, "Failed to allocate SLOT.indication message");
        return false;
    }

    const FapiHeaderParams params{
            .msg_id = SCF_FAPI_SLOT_INDICATION,
            .cell_id = target_cell_id,
            .body_size = sizeof(scf_fapi_slot_ind_t) - sizeof(scf_fapi_body_header_t)};
    auto *body = build_fapi_header(msg, params);

    auto *slot_ind = assume_cast<scf_fapi_slot_ind_t>(body);
    slot_ind->sfn = slot.sfn;
    slot_ind->slot = slot.slot;

    if (send_message(msg) < 0) {
        return false;
    }
    ipc_->tx_tti_sem_post(ipc_.get());

    RT_LOGC_DEBUG(
            FapiComponent::FapiMsg,
            "Sent SLOT.indication: cell={}, sfn={}, slot={}",
            target_cell_id,
            slot.sfn,
            slot.slot);

    return true;
}

bool FapiState::send_config_response(
        const uint16_t cell_id, const scf_fapi_error_codes_t error_code) {
    nv_ipc_msg_t msg{};
    if (allocate_message(msg) < 0) {
        RT_LOGC_ERROR(FapiComponent::FapiMsg, "Failed to allocate message");
        return false;
    }

    const FapiHeaderParams params{
            .msg_id = SCF_FAPI_CONFIG_RESPONSE,
            .cell_id = cell_id,
            .body_size = sizeof(scf_fapi_config_response_msg_t) - sizeof(scf_fapi_body_header_t)};
    auto *body = build_fapi_header(msg, params);

    auto *rsp = assume_cast<scf_fapi_config_response_msg_t>(body);
    rsp->msg_body.error_code = error_code;
    rsp->msg_body.num_invalid_tlvs = 0;
    rsp->msg_body.num_idle_only_tlvs = 0;
    rsp->msg_body.num_running_only_tlvs = 0;
    rsp->msg_body.num_missing_tlvs = 0;

    if (send_message(msg) < 0) {
        return false;
    }
    ipc_->tx_tti_sem_post(ipc_.get());

    RT_LOGC_INFO(
            FapiComponent::FapiMsg,
            "Sent CONFIG.response: cell={}, error={:#04X}",
            cell_id,
            static_cast<uint32_t>(error_code));

    return true;
}

bool FapiState::send_stop_indication(const uint16_t cell_id) {
    nv_ipc_msg_t msg{};
    if (allocate_message(msg) < 0) {
        RT_LOGC_ERROR(FapiComponent::FapiMsg, "Failed to allocate message");
        return false;
    }

    // STOP.indication has no body, just header
    const FapiHeaderParams params{
            .msg_id = SCF_FAPI_STOP_INDICATION, .cell_id = cell_id, .body_size = 0};
    build_fapi_header(msg, params);

    if (send_message(msg) < 0) {
        return false;
    }
    ipc_->tx_tti_sem_post(ipc_.get());

    RT_LOGC_INFO(FapiComponent::FapiMsg, "Sent STOP.indication: cell={}", cell_id);

    return true;
}

bool FapiState::send_error_indication(
        const uint16_t cell_id,
        const scf_fapi_message_id_e msg_id,
        const scf_fapi_error_codes_t error_code) {
    // Atomically capture current slot to avoid torn reads
    const SlotInfo slot = get_current_slot();

    nv_ipc_msg_t msg{};
    if (allocate_message(msg) < 0) {
        RT_LOGC_ERROR(FapiComponent::FapiMsg, "Failed to allocate message");
        return false;
    }

    const FapiHeaderParams params{
            .msg_id = SCF_FAPI_ERROR_INDICATION,
            .cell_id = cell_id,
            .body_size = sizeof(scf_fapi_error_ind_t) - sizeof(scf_fapi_body_header_t)};
    auto *body = build_fapi_header(msg, params);

    auto *err_ind = assume_cast<scf_fapi_error_ind_t>(body);
    err_ind->sfn = slot.sfn;
    err_ind->slot = slot.slot;
    err_ind->msg_id = msg_id;
    err_ind->err_code = error_code;

    if (send_message(msg) < 0) {
        return false;
    }
    ipc_->tx_tti_sem_post(ipc_.get());

    RT_LOGC_INFO(
            FapiComponent::FapiMsg,
            "Sent ERROR.indication: cell={}, msg_id={}, error={:#04X}",
            cell_id,
            static_cast<uint32_t>(msg_id),
            static_cast<uint32_t>(error_code));

    return true;
}

FapiStateT FapiState::get_cell_state(const uint16_t cell_id) const noexcept {
    if (!is_valid_cell_id(cell_id, params_.max_cells)) {
        return FapiStateT::FapiStateIdle;
    }
    return cells_[cell_id].state.load();
}

std::size_t FapiState::get_num_cells_configured() const noexcept { return num_cells_configured_; }

std::size_t FapiState::get_num_cells_running() const noexcept { return num_cells_running_; }

int FapiState::allocate_message(nv_ipc_msg_t &msg) {
    static constexpr uint32_t OPTIONS = 0;
    const int result = ipc_->tx_allocate(ipc_.get(), &msg, OPTIONS);
    if (result >= 0) {
        // Acquire fence ensures buffer is clean after allocation from free pool
        std::atomic_thread_fence(std::memory_order_acquire);
    }
    return result;
}

int FapiState::receive_message(nv_ipc_msg_t &msg) {
    const int result = ipc_->rx_recv_msg(ipc_.get(), &msg);
    if (result >= 0) {
        // Acquire fence ensures all writes from sender are visible after receiving message
        std::atomic_thread_fence(std::memory_order_acquire);
    }
    return result;
}

void FapiState::release_message(nv_ipc_msg_t &msg) {
    // Release fence ensures all buffer reads are complete before returning to free pool
    std::atomic_thread_fence(std::memory_order_release);
    ipc_->rx_release(ipc_.get(), &msg);
}

int FapiState::send_message(nv_ipc_msg_t &msg) {
    // Release fence ensures all buffer writes are visible before sending
    std::atomic_thread_fence(std::memory_order_release);

    // Copy values before logging so they don't reference the nvipc message buffer
    const auto msg_id = msg.msg_id;
    const auto cell_id = msg.cell_id;

    const int result = ipc_->tx_send_msg(ipc_.get(), &msg);
    if (result < 0) {
        RT_LOGC_ERROR(
                FapiComponent::FapiMsg,
                "Failed to send {}: cell_id={}, msg_id={}",
                fapi_message_id_to_string(msg_id),
                cell_id,
                msg_id);
        ipc_->tx_release(ipc_.get(), &msg);
    }
    return result;
}

void FapiState::set_on_ul_tti_request(OnUlTtiRequestCallback callback) {
    on_ul_tti_request_ = std::move(callback);
}

void FapiState::set_on_dl_tti_request(OnDlTtiRequestCallback callback) {
    on_dl_tti_request_ = std::move(callback);
}

void FapiState::set_on_slot_response(OnSlotResponseCallback callback) {
    on_slot_response_ = std::move(callback);
}

void FapiState::set_on_config_request(OnConfigRequestCallback callback) {
    on_config_request_ = std::move(callback);
}

void FapiState::set_on_start_request(OnStartRequestCallback callback) {
    on_start_request_ = std::move(callback);
}

void FapiState::set_on_stop_request(OnStopRequestCallback callback) {
    on_stop_request_ = std::move(callback);
}

void FapiState::set_on_message(OnMessageCallback callback) { on_message_ = std::move(callback); }

} // namespace ran::fapi
