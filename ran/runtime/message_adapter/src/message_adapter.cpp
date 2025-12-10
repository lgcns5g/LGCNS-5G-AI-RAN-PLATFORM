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

#include <chrono>        // for milliseconds
#include <cstdlib>       // for getenv
#include <exception>     // for exception
#include <filesystem>    // for path, operator/
#include <unordered_map> // for unordered_map

#include <nv_ipc.hpp>            // for nv_ipc_parse_yaml_node
#include <quill/LogMacros.h>     // for QUILL_LOG_INFO
#include <wise_enum_detail.h>    // for WISE_ENUM_IMPL_IIF_0
#include <wise_enum_generated.h> // for WISE_ENUM_IMPL_LOOP_2, WISE_ENUM_IM...
#include <yaml.hpp>

#include "log/components.hpp"    // for LogLevel, register_component, DECLA...
#include "log/rt_log.hpp"        // for Logger, LoggerConfig
#include "log/rt_log_macros.hpp" // for RT_LOGEC_INFO, RT_LOGC_INFO
#include "message_adapter.hpp"

namespace ran::message_adapter {
// Define components for logging
DECLARE_LOG_COMPONENT(MessageAdapterComponent, MessageAdapterCore, Nvipc);
// Define events for logging
DECLARE_LOG_EVENT(
        MessageAdapterEvent,
        NvipcInitComplete,
        NvipcInitError,
        MESSAGE_LOOP_START,
        MESSAGE_LOOP_START_ERROR,
        MESSAGE_LOOP_STOP,
        NvipcDestroyComplete);

/** NvIPC event logging identifiers */
DECLARE_LOG_EVENT(NvIpcEvent, NvipcRxMsg, NvipcTxMsg, NvipcRxError, NvipcTxError);

MessageAdapter::MessageAdapter(const std::string &config_file) // NOLINT(modernize-pass-by-value)
        : running_(false), stop_requested_(false), config_file_(config_file) {

    // Console logging with INFO level
    framework::log::Logger::configure(
            framework::log::LoggerConfig::console(framework::log::LogLevel::Info));

    // Or file logging
    // framework::log::Logger::configure(framework::log::LoggerConfig::file("app.log",
    // framework::log::LogLevel::Debug));

    framework::log::register_component<MessageAdapterComponent>(
            {{MessageAdapterComponent::MessageAdapterCore, framework::log::LogLevel::Info},
             {MessageAdapterComponent::Nvipc, framework::log::LogLevel::Debug}});
}

MessageAdapter::~MessageAdapter() {
    // Ensure proper cleanup when the object is destroyed
    if (running_.load()) {
        stop();
    }
}

bool MessageAdapter::start() {
    if (running_.load()) {
        RT_LOGC_INFO(
                MessageAdapterComponent::MessageAdapterCore, "MessageAdapter is already running");
        return false; // Already running
    }

    nv_ipc_config_t config;
    ipc_ = init_nv_ipc_interface(&config);
    if (ipc_ == nullptr) {
        RT_LOGEC_INFO(
                MessageAdapterComponent::MessageAdapterCore,
                MessageAdapterEvent::NvipcInitError,
                "init_nv_ipc_interface failed");
        return false;
    }

    stop_requested_.store(false);
    running_.store(true);

    // Create the thread with the message_loop function
    try {
        thread_ = std::thread([this]() { this->message_loop(); });
    } catch (const std::exception &e) {
        running_.store(false);
        RT_LOGEC_INFO(
                MessageAdapterComponent::MessageAdapterCore,
                MessageAdapterEvent::MESSAGE_LOOP_START_ERROR,
                "MessageAdapter::failed to start message loop: {}",
                e.what());
        return false;
    }
    RT_LOGEC_INFO(
            MessageAdapterComponent::MessageAdapterCore,
            MessageAdapterEvent::MESSAGE_LOOP_START,
            "MessageAdapter::message_loop started");
    return true;
}

nv_ipc_t *MessageAdapter::init_nv_ipc_interface(nv_ipc_config_t *config) {

    const char *framework_home_env =
            std::getenv("AERIAL_FRAMEWORK_ROOT"); // NOLINT(concurrency-mt-unsafe)
    if (framework_home_env == nullptr ||
        framework_home_env[0] == '\0') { // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        RT_LOGEC_INFO(
                MessageAdapterComponent::MessageAdapterCore,
                MessageAdapterEvent::NvipcInitError,
                "AERIAL_FRAMEWORK_ROOT is not set or empty");
        return nullptr;
    }

    const std::filesystem::path config_path = std::filesystem::path(framework_home_env) / "ran" /
                                              "runtime" / "message_adapter" / "config" /
                                              config_file_;
    const std::string config_file = config_path.string();
    try {
        yaml::file_parser file_parser(config_file.c_str());
        yaml::document doc = file_parser.next_document();
        const yaml::node node_config = doc.root();

        // Extract execution_mode from YAML (default to 0 if not present or invalid)
        try {
            yaml::node execution_mode_node = node_config["execution_mode"];
            execution_mode_ = execution_mode_node.as<int>();
            RT_LOGC_INFO(
                    MessageAdapterComponent::MessageAdapterCore,
                    "Loaded execution_mode from config: {}",
                    execution_mode_);
        } catch (const std::exception &ex) {
            RT_LOGC_INFO(
                    MessageAdapterComponent::MessageAdapterCore,
                    "execution_mode not found or invalid in config, using default: 0 (error: {})",
                    ex.what());
        }

        yaml::node transport_node = node_config["transport"];
        nv_ipc_module_t module_type = NV_IPC_MODULE_PHY; // NOLINT(misc-const-correctness)
        nv_ipc_parse_yaml_node(config, &transport_node, module_type);
    } catch (const std::exception &e) {
        RT_LOGEC_INFO(
                MessageAdapterComponent::MessageAdapterCore,
                MessageAdapterEvent::NvipcInitError,
                "Failed to parse config file: {}",
                e.what());
        return nullptr;
    }
    return create_nv_ipc_interface(config);
}

void MessageAdapter::stop() {
    if (!running_.load()) {
        RT_LOGC_INFO(MessageAdapterComponent::MessageAdapterCore, "MessageAdapter is not running");
        return; // Not running
    }

    stop_requested_.store(true);

    // Wait for the thread to finish if it's joinable
    if (thread_.joinable()) {
        thread_.join();
    }

    running_.store(false);
    RT_LOGEC_INFO(
            MessageAdapterComponent::MessageAdapterCore,
            MessageAdapterEvent::MESSAGE_LOOP_STOP,
            "MessageAdapter::message_loop stopped");
}

bool MessageAdapter::is_running() const noexcept { return running_.load(); }

void MessageAdapter::message_loop() {
    static constexpr int POLL_INTERVAL_MS = 10;

    while (!stop_requested_.load()) {
        nv_ipc_msg_t smsg;
        while (ipc_->rx_recv_msg(ipc_, &smsg) >= 0) {
            process_message(smsg);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(POLL_INTERVAL_MS));
    }
    // Cleanup when stopping
    running_.store(false);
    RT_LOGEC_INFO(
            MessageAdapterComponent::MessageAdapterCore,
            MessageAdapterEvent::MESSAGE_LOOP_STOP,
            "MessageAdapter::message_loop stopped");
    if (ipc_ != nullptr) {
        ipc_->ipc_destroy(ipc_);
        RT_LOGEC_INFO(
                MessageAdapterComponent::MessageAdapterCore,
                MessageAdapterEvent::NvipcDestroyComplete,
                "NVIPC context deleted");
        ipc_ = nullptr;
    }
}
} // namespace ran::message_adapter
