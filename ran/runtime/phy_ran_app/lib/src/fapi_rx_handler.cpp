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
 * @file fapi_rx_handler.cpp
 * @brief Implementation of FAPI RX message handler
 */

#include <atomic>
#include <chrono>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include <nv_ipc.h>
#include <nv_ipc.hpp>
#include <quill/LogMacros.h>
#include <unistd.h>
#include <yaml.hpp>

#include <gsl-lite/gsl-lite.hpp>

#include "ifapi_slot_info_provider.hpp"
#include "ipipeline_executor.hpp"
#include "islot_indication_sender.hpp"
#include "log/rt_log_macros.hpp"
#include "message_adapter/examples/sample_5g_pipelines.hpp"
#include "message_adapter/phy_stats.hpp"
#include "phy_ran_app/fapi_rx_handler.hpp"
#include "phy_ran_app/phy_ran_app_log.hpp"
#include "pipeline/ipipeline_output_provider.hpp"

namespace ran::phy_ran_app {

namespace {

/**
 * Create unique temporary YAML file path
 *
 * @param[in] prefix File name prefix
 * @return Unique temporary file path using PID and timestamp
 */
[[nodiscard]] std::filesystem::path create_unique_temp_yaml_file(const std::string &prefix) {
    const std::filesystem::path temp_dir = std::filesystem::temp_directory_path();
    const auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    return temp_dir / std::format("{}_{}_{}_.yaml", prefix, ::getpid(), now);
}

/**
 * Initialize NVIPC from config file path
 *
 * @param[in] config_file Path to YAML configuration file
 * @return nvIPC interface pointer, or nullptr on failure
 */
[[nodiscard]] nv_ipc_t *init_nvipc_from_file(const std::string &config_file) {
    if (!std::filesystem::exists(config_file)) {
        RT_LOGC_ERROR(PhyRanApp::FapiRx, "Config file not found: {}", config_file);
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
                    PhyRanApp::FapiRx,
                    "Failed to create NVIPC interface from config file: {}",
                    config_file);
        }

        return ipc;
    } catch (const std::exception &e) {
        RT_LOGC_ERROR(
                PhyRanApp::FapiRx,
                "Failed to initialize NVIPC from config file {}: {}",
                config_file,
                e.what());
        return nullptr;
    }
}

/**
 * Initialize NVIPC from YAML string
 *
 * @param[in] yaml_string YAML configuration as string
 * @return nvIPC interface pointer, or nullptr on failure
 */
[[nodiscard]] nv_ipc_t *init_nvipc_from_string(const std::string &yaml_string) {
    if (yaml_string.empty()) {
        RT_LOGC_ERROR(PhyRanApp::FapiRx, "YAML config string is empty");
        return nullptr;
    }

    // YAML library only supports file parsing, so write to temp file
    // Use unique file name to avoid collisions between concurrent processes
    const std::filesystem::path temp_file =
            create_unique_temp_yaml_file("phy_ran_app_nvipc_config");

    // Setup cleanup guard to ensure temp file is removed
    const auto cleanup = gsl_lite::finally([&temp_file] { std::filesystem::remove(temp_file); });

    try {
        // Write string to temp file
        std::ofstream ofs(temp_file);
        if (!ofs) {
            RT_LOGC_ERROR(
                    PhyRanApp::FapiRx,
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
                    PhyRanApp::FapiRx, "Failed to create NVIPC interface from YAML string config");
        }

        return ipc;
    } catch (const std::exception &e) {
        RT_LOGC_ERROR(
                PhyRanApp::FapiRx,
                "Failed to initialize NVIPC from YAML string config: {}",
                e.what());
        return nullptr;
    }
}

} // anonymous namespace

FapiRxHandler::FapiRxHandler(
        const InitParams &params,
        GraphScheduleCallback on_graph_schedule,
        framework::pipeline::IPipelineOutputProvider &output_provider)
        : pipeline_(nullptr) {
    // Create nvIPC endpoint
    if (!params.nvipc_config_string.empty()) {
        ipc_ = init_nvipc_from_string(params.nvipc_config_string);
    } else if (!params.nvipc_config_file.empty()) {
        ipc_ = init_nvipc_from_file(params.nvipc_config_file);
    } else {
        throw std::runtime_error("FapiRxHandler: No nvIPC config provided");
    }

    if (ipc_ == nullptr) {
        throw std::runtime_error("FapiRxHandler: Failed to create nvIPC interface");
    }

    RT_LOGC_INFO(PhyRanApp::FapiRx, "nvIPC interface created successfully");

    // Create Sample5GPipeline with pipeline output provider interface
    const ran::message_adapter::Sample5GPipeline::InitParams pipeline_params{
            .ipc = ipc_,
            .max_cells = params.max_cells,
            .on_graph_schedule = std::move(on_graph_schedule),
            .output_provider = &output_provider,
    };

    try {
        pipeline_ = std::make_unique<ran::message_adapter::Sample5GPipeline>(pipeline_params);
        RT_LOGC_INFO(PhyRanApp::FapiRx, "Sample5GPipeline created successfully");
    } catch (const std::exception &e) {
        // Cleanup nvIPC before rethrowing
        if (ipc_ != nullptr) {
            ipc_->ipc_destroy(ipc_);
            ipc_ = nullptr;
        }
        throw std::runtime_error(
                std::format("FapiRxHandler: Failed to create Sample5GPipeline: {}", e.what()));
    }
}

FapiRxHandler::~FapiRxHandler() {
    // Destroy pipeline first (may use nvIPC in cleanup)
    pipeline_.reset();

    // Close nvIPC endpoint
    if (ipc_ != nullptr) {
        RT_LOGC_DEBUG(PhyRanApp::FapiRx, "Closing nvIPC interface");
        ipc_->ipc_destroy(ipc_);
        ipc_ = nullptr;
    }
}

void FapiRxHandler::receive_and_process_messages(const std::atomic<bool> &running) {
    using namespace std::chrono_literals;

    RT_LOGC_INFO(PhyRanApp::FapiRx, "Starting FAPI RX message loop");

    constexpr auto RX_IDLE_SLEEP = 100us;

    // Continuously process messages until shutdown
    while (running.load(std::memory_order_acquire)) {
        nv_ipc_msg_t msg{};
        const int ret = ipc_->rx_recv_msg(ipc_, &msg);

        if (ret < 0) {
            // No message available, sleep briefly
            std::this_thread::sleep_for(RX_IDLE_SLEEP);
            continue;
        }

        // Process the message through Sample5GPipeline
        RT_LOGC_TRACE_L1(
                PhyRanApp::FapiRx,
                "Received FAPI message: cell_id={}, msg_id={}",
                msg.cell_id,
                msg.msg_id);

        try {
            pipeline_->process_msg(msg);
        } catch (const std::exception &e) {
            RT_LOGC_ERROR(
                    PhyRanApp::FapiRx,
                    "Exception processing FAPI message (cell_id={}, msg_id={}): {}",
                    msg.cell_id,
                    msg.msg_id,
                    e.what());
        }
    }

    RT_LOGC_INFO(PhyRanApp::FapiRx, "FAPI RX message loop terminated");
}

ran::message_adapter::ISlotIndicationSender *FapiRxHandler::get_slot_indication_sender() {
    if (pipeline_ != nullptr) {
        return pipeline_.get();
    }
    return nullptr;
}

ran::message_adapter::IFapiSlotInfoProvider *FapiRxHandler::get_slot_info_provider() {
    if (pipeline_ != nullptr) {
        return pipeline_.get();
    }
    return nullptr;
}

ran::message_adapter::IPipelineExecutor *FapiRxHandler::get_pipeline_executor() {
    if (pipeline_ != nullptr) {
        return pipeline_.get();
    }
    return nullptr;
}

std::size_t FapiRxHandler::get_num_cells_running() const {
    if (pipeline_ != nullptr) {
        return pipeline_->get_num_cells_running();
    }
    return 0;
}

const ran::message_adapter::PhyStats &FapiRxHandler::get_stats() const {
    gsl_Expects(pipeline_ != nullptr);
    return pipeline_->get_stats();
}

} // namespace ran::phy_ran_app
