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

#include <chrono>   // for seconds
#include <iostream> // for char_traits, operator<<, basic_os...
#include <memory>   // for allocator, make_unique
#include <thread>   // for sleep_for

#include <nv_ipc.h> // for nv_ipc_msg_t, nv_ipc_t

#include "fapi/fapi_state.hpp"
#include "message_adapter.hpp"     // for MessageAdapter
#include "pipeline_interface.hpp"  // for PipelineInterface
#include "ran_common.hpp"          // for NUM_CELLS_SUPPORTED
#include "sample_5g_pipelines.hpp" // for Sample5GPipeline

/**
 * Adapter to bridge Sample5GPipeline with MessageAdapter's PipelineInterface
 *
 * Sample5GPipeline was refactored to implement IFapiMessageProcessor which has
 * process_msg(msg) - single parameter, using internal ipc_ pointer.
 *
 * MessageAdapter expects PipelineInterface which has process_msg(msg, ipc) - two parameters.
 *
 * This adapter inherits from both and forwards the two-parameter call to the one-parameter
 * version, ignoring the ipc parameter since Sample5GPipeline already has ipc_ internally.
 */
class Sample5GPipelineAdapter : public ran::message_adapter::Sample5GPipeline,
                                public ran::message_adapter::PipelineInterface {
public:
    /**
     * Constructor - forwards InitParams to Sample5GPipeline
     *
     * @param[in] params Initialization parameters (includes ipc pointer)
     */
    explicit Sample5GPipelineAdapter(
            const ran::message_adapter::Sample5GPipeline::InitParams &params)
            : Sample5GPipeline(params) {}

    // Bring Sample5GPipeline::process_msg(msg) into scope to avoid hiding overload
    using Sample5GPipeline::process_msg;

    /**
     * Implement PipelineInterface::process_msg(msg, ipc)
     *
     * Forwards to Sample5GPipeline::process_msg(msg), ignoring the ipc parameter
     * since Sample5GPipeline uses its internal ipc_ pointer from InitParams.
     *
     * @param[in,out] msg FAPI message to process
     * @param[in] ipc IPC interface pointer (ignored, using internal ipc_)
     */
    void process_msg(nv_ipc_msg_t &msg, [[maybe_unused]] nv_ipc_t *ipc) override {
        Sample5GPipeline::process_msg(msg);
    }
};

int main() {

    ran::message_adapter::MessageAdapter message_adapter("config.yaml");
    const bool result = message_adapter.start();
    if (!result) {
        std::cerr << "Failed to start message adapter" << '\n';
        return 1;
    }
    // NOTE: Sample5GPipeline now requires InitParams with callbacks for integration
    // with slot indication triggering and uplink graph scheduling.
    //
    // To make this functional, you would need:
    // 1. on_graph_schedule callback: Schedule uplink processing graph (C-plane → U-plane →
    // PUSCH)
    //    This requires TaskScheduler + TaskGraph integration (see phy_ran_app.cpp)
    //
    // Without graph scheduling, PUSCH pipelines won't execute, so this standalone app
    // cannot achieve parity with the original origin/main behavior.
    //
    // This is a placeholder to make the code compile. For functional FAPI + PUSCH
    // integration, use phy_ran_app instead.
    const ran::message_adapter::Sample5GPipeline::InitParams params{
            .ipc = message_adapter.get_ipc(),
            .max_cells = ran::common::NUM_CELLS_SUPPORTED,
            .on_graph_schedule = [](ran::fapi::SlotInfo) {} // No-op: requires TaskGraph integration
    };
    message_adapter.add_pipeline(std::make_unique<Sample5GPipelineAdapter>(params));
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}
