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

#ifndef RAN_MESSAGE_ADAPTER_HPP
#define RAN_MESSAGE_ADAPTER_HPP

#include <algorithm>
#include <atomic>  // for atomic
#include <memory>  // for unique_ptr
#include <mutex>   // for mutex, lock_guard
#include <string>  // for string
#include <thread>  // for thread
#include <utility> // for move
#include <vector>  // for vector

#include <nv_ipc.h> // for nv_ipc_t, nv_ipc_config_t, nv_ipc_...
#include <stddef.h> // for size_t

#include "pipeline_interface.hpp" // for PipelinePtr, PipelineInterface

namespace ran::message_adapter {

/**
 * MessageAdapter class for handling message processing in a separate thread
 *
 * This class creates a dedicated thread for message processing and provides
 * thread-safe operations for starting, stopping, and managing the message loop.
 */
class MessageAdapter {
public:
    /**
     * Constructor that creates a thread for message processing
     *
     * The thread is automatically joined when the object is destroyed
     * due to the use of std::jthread.
     *
     * @param[in] config_file Path to the configuration file
     */
    explicit MessageAdapter(const std::string &config_file);

    /**
     * Destructor - ensures proper cleanup of resources
     */
    ~MessageAdapter();

    // Prevent copying and moving
    MessageAdapter(const MessageAdapter &) = delete;
    MessageAdapter &operator=(const MessageAdapter &) = delete;
    MessageAdapter(MessageAdapter &&) = delete;
    MessageAdapter &operator=(MessageAdapter &&) = delete;

    /**
     * Start the message processing thread
     *
     * @return true if started successfully, false otherwise
     */
    [[nodiscard]] bool start();

    /**
     * Stop the message processing thread
     */
    void stop();

    /**
     * Check if the thread is currently running
     *
     * @return true if running, false otherwise
     */
    [[nodiscard]] bool is_running() const noexcept;

    /**
     * Get the execution mode from configuration
     *
     * @return Execution mode (0 = Stream, 1 = Graph)
     */
    [[nodiscard]] int get_execution_mode() const noexcept { return execution_mode_; }

    /**
     * Get the Ipc interface pointer
     *
     * @return Pointer to the NV Ipc interface
     */
    [[nodiscard]] nv_ipc_t *get_ipc() const noexcept { return ipc_; }

    /**
     * Initialize NV Ipc interface
     *
     * This function initializes the NV Ipc interface with the configuration
     * specified in the config file.
     *
     * @param[in,out] config Pointer to the nv_ipc_config_t structure to be configured
     * @return Pointer to the initialized NV Ipc interface, or nullptr if initialization fails
     */
    [[nodiscard]] nv_ipc_t *init_nv_ipc_interface(nv_ipc_config_t *config);

    /**
     * Add a pipeline for message processing
     *
     * This function adds a new pipeline to the processing chain. The pipeline
     * will be used to process incoming messages in the message loop.
     *
     * @param[in] new_pipeline Smart pointer to the new pipeline object
     */
    void add_pipeline(PipelinePtr new_pipeline) {
        const std::lock_guard<std::mutex> lock(pipelines_mutex_);
        pipelines_.push_back(std::move(
                new_pipeline)); // Add to pipeline chain
                                // RT_LOGC_INFO(MessageAdapterComponent::MessageAdapterCore,
                                // "Pipeline added " << pipelines_.size());
    }

    /**
     * Set the pipeline at a specific index
     *
     * This function sets the pipeline at the specified index. If the index
     * is out of bounds, the pipeline is added to the end.
     *
     * @param[in] index Index where to set the pipeline
     * @param[in] new_pipeline Smart pointer to the new pipeline object
     */
    void set_pipeline(size_t index, PipelinePtr new_pipeline) {
        const std::lock_guard<std::mutex> lock(pipelines_mutex_);
        if (index < pipelines_.size()) {
            pipelines_[index] = std::move(new_pipeline);
        } else {
            pipelines_.push_back(std::move(new_pipeline));
        }
    }

    /**
     * Process a single message using all pipelines
     *
     * This function delegates message processing to all pipelines in the chain.
     * If no pipelines are set, the message is ignored.
     *
     * @param[in,out] msg Message to be processed
     */
    void process_message(nv_ipc_msg_t &msg) {
        // Hold lock during processing to ensure thread safety
        // This is acceptable since message processing should be fast
        const std::lock_guard<std::mutex> lock(pipelines_mutex_);

        for (const auto &pipeline : pipelines_) {
            if (pipeline) {
                pipeline->process_msg(msg, ipc_);
            }
        }
    }

private:
    /**
     * Main message processing loop
     *
     * This function runs in the separate thread and handles
     * the message processing logic.
     */
    void message_loop();
    std::vector<PipelinePtr> pipelines_;
    mutable std::mutex pipelines_mutex_; ///< Mutex to protect pipelines_ access

    std::thread thread_;               ///< Thread for message processing
    std::atomic<bool> running_;        ///< Flag indicating if thread is running
    std::atomic<bool> stop_requested_; ///< Flag to request thread stop
    std::string config_file_;
    nv_ipc_t *ipc_{nullptr};
    int execution_mode_{}; ///< Execution mode (0 = Stream, 1 = Graph)
};

} // namespace ran::message_adapter

#endif // RAN_MESSAGE_ADAPTER_HPP
