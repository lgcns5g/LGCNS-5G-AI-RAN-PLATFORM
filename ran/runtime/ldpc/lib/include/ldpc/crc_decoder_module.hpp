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

#ifndef RAN_LDPC_CRC_DECODER_MODULE_HPP
#define RAN_LDPC_CRC_DECODER_MODULE_HPP

#include <any>
#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <cuphy.h>
#include <driver_types.h>

#include <gsl-lite/gsl-lite.hpp>

#include <cuda.h>

#include "ldpc/outer_rx_params.hpp"
#include "pipeline/iallocation_info_provider.hpp"
#include "pipeline/igraph.hpp"
#include "pipeline/igraph_node_provider.hpp"
#include "pipeline/imodule.hpp"
#include "pipeline/istream_executor.hpp"
#include "pipeline/kernel_descriptor_accessor.hpp"
#include "pipeline/types.hpp"
#include "ran_common.hpp"
#include "tensor/tensor_info.hpp"

namespace ran::ldpc {

/**
 * @class CrcDecoderModule
 * @brief CRC decoder module implementing IModule interface
 *
 * This module provides CRC (Cyclic Redundancy Check) decoding functionality
 * by wrapping existing cuPHY CRC capabilities. It decodes the CRC of the code
 * blocks and transport blocks and concatenates code blocks into transport blocks.
 */
class CrcDecoderModule final : public framework::pipeline::IModule,
                               public framework::pipeline::IAllocationInfoProvider,
                               public framework::pipeline::IStreamExecutor,
                               public framework::pipeline::IGraphNodeProvider {
public:
    /**
     * @brief Configuration parameters for CRC decoder module
     */
    struct StaticParams final {
        bool reverse_bytes{true}; //!< Reverse bytes in each word before computing the CRC
        std::size_t max_num_cbs_per_tb{
                ran::common::MAX_NUM_CBS_PER_TB};          //!< Maximum number of code blocks
                                                           //!< per transport block
        std::size_t max_num_tbs{ran::common::MAX_NUM_TBS}; //!< Maximum number of transport blocks
    };

    /**
     * @brief Constructor
     *
     * @param instance_id The instance identifier for this module
     * @param init_params Initialization parameters for module configuration
     * @throws std::invalid_argument if initialization parameters are invalid
     * @throws std::runtime_error if CRC decode object cannot be created
     */
    explicit CrcDecoderModule(std::string instance_id, const std::any &init_params);

    /**
     * @brief Destructor
     */
    ~CrcDecoderModule() override;

    /**
     * @brief Copy constructor (deleted)
     *
     * CrcDecoderModule manages CUDA resources and pipeline state that cannot be
     * safely copied. Each module instance must have unique identity and resource
     * ownership.
     */
    CrcDecoderModule(const CrcDecoderModule &) = delete;

    /**
     * @brief Move constructor (deleted)
     *
     */
    CrcDecoderModule(CrcDecoderModule &&) = delete;

    /**
     * @brief Copy assignment operator (deleted)
     *
     */
    CrcDecoderModule &operator=(const CrcDecoderModule &) = delete;

    /**
     * @brief Move assignment operator (deleted)
     *
     */
    CrcDecoderModule &operator=(CrcDecoderModule &&) = delete;

    /**
     * Get the type identifier
     *
     * @return The type ID as a string_view
     */
    [[nodiscard]] std::string_view get_type_id() const override { return "crc_decoder_module"; }

    /**
     * Get the instance identifier
     *
     * @return The instance ID as a string_view
     */
    [[nodiscard]] std::string_view get_instance_id() const override;

    /**
     * Perform one-time setup after memory allocation.
     *
     * @param memory_slice Memory slice allocated by PipelineMemoryManager
     */
    void setup_memory(const framework::pipeline::ModuleMemorySlice &memory_slice) override;

    /**
     * Get the input tensor information for a given port name.
     *
     * @param port_name The name of the port to get the tensor information for
     * @return Vector of tensor information for all tensors on this port
     * @throws std::invalid_argument if the port name is invalid
     */
    [[nodiscard]] std::vector<framework::tensor::TensorInfo>
    get_input_tensor_info(std::string_view port_name) const override;

    /**
     * Get the output tensor information for a given port name.
     *
     * @param port_name The name of the port to get the tensor information for
     * @return Vector of tensor information for all tensors on this port
     * @throws std::invalid_argument if the port name is invalid
     */
    [[nodiscard]] std::vector<framework::tensor::TensorInfo>
    get_output_tensor_info(std::string_view port_name) const override;

    /**
     * Get the names of all input ports
     *
     * @return A vector containing input port names
     */
    [[nodiscard]] std::vector<std::string> get_input_port_names() const override;

    /**
     * Get the names of all output ports
     *
     * @return A vector containing output port names
     */
    [[nodiscard]] std::vector<std::string> get_output_port_names() const override;

    /**
     * Set the inputs for the module.
     *
     * @param inputs Span of port information with device pointers to input data
     * @throws std::invalid_argument if required inputs are missing or port information
     * doesn't match with expected inputs
     */
    void set_inputs(std::span<const framework::pipeline::PortInfo> inputs) override;

    /**
     * Get the output port information.
     *
     * @return Vector of port information for all outputs
     * @note Device pointers are only valid after configure_io() has been called
     * @throws std::runtime_error if the module inputs/outputs are not configured
     */
    [[nodiscard]] std::vector<framework::pipeline::PortInfo> get_outputs() const override;

    /**
     * Configure I/O for the current iteration.
     *
     * Sets up module output tensor information and device pointers, and sets up the cuPHY CRC
     * decode object.
     *
     * @param[in] params Dynamic parameters for the current iteration
     * @param[in] stream CUDA stream for async operations during configuration
     * @throws std::invalid_argument if the dynamic parameters are invalid
     * @throws std::runtime_error if the module is not setup
     */
    void
    configure_io(const framework::pipeline::DynamicParams &params, cudaStream_t stream) override;

    /**
     * @brief Return the graph node provider for this module.
     * @return The graph node provider
     */
    [[nodiscard]] framework::pipeline::IGraphNodeProvider *as_graph_node_provider() override {
        return this;
    }

    /**
     * @brief Return the stream executor for this module.
     * @return The stream executor
     */
    [[nodiscard]] framework::pipeline::IStreamExecutor *as_stream_executor() override {
        return this;
    }

    /**
     * @brief Execute the module on the given stream.
     * @param stream The stream to execute the module on
     * @note All dynamic descriptors must have been copied to device before execution.
     */
    void execute(cudaStream_t stream) override;

    /**
     * @brief Get the memory requirements for the module.
     * @return The memory requirements
     * @throws std::runtime_error if failed to get workspace size for CRC check
     */
    [[nodiscard]] framework::pipeline::ModuleMemoryRequirements get_requirements() const override;

    /**
     * @brief Add node(s) to the graph
     * @param[in] graph The graph to add the node(s) to
     * @param[in] deps The dependencies of the node(s)
     * @return Span of created graph node handles (returns both CRC decode nodes)
     * @throws std::runtime_error if CUDA graph node creation fails
     */
    [[nodiscard]] std::span<const CUgraphNode> add_node_to_graph(
            gsl_lite::not_null<framework::pipeline::IGraph *> graph,
            const std::span<const CUgraphNode> deps) override;

    /**
     * @brief Update graph node parameters for dynamic tick changes
     *
     * This method enables dynamic updates to kernel launch parameters using
     * cuGraphExecKernelNodeSetParams. Modules can extract their specific
     * parameters from params.module_specific_params and update their graph
     * nodes accordingly (e.g., changing grid dimensions, shared memory size).
     *
     * @param exec The executable graph to update
     * @param params Dynamic parameters containing module-specific parameters
     * @throws std::runtime_error if cuGraphExecKernelNodeSetParams fails
     */
    void update_graph_node_params(
            CUgraphExec exec, const framework::pipeline::DynamicParams &params) override;

private:
    std::string instance_id_; //!< Module instance identifier

    // Static and dynamic parameters
    StaticParams static_params_;                              //!< Static parameters from init
    std::optional<PuschOuterRxParams> pusch_outer_rx_params_; //!< Transport block parameters

    // Port information
    std::vector<framework::pipeline::PortInfo> inputs_;  //!< Input port information
    std::vector<framework::pipeline::PortInfo> outputs_; //!< Output port information

    framework::pipeline::ModuleMemorySlice memory_slice_; //!< Allocated memory slice

    std::vector<std::size_t> tb_payload_offsets_; //!< Output transport block payload offsets per UE
    std::vector<std::size_t> cb_crc_offsets_;     //!< Output code block CRC offsets per UE

    // Internal state
    bool setup_complete_{false};                    //!< Whether setup is complete
    cuphyPuschRxCrcDecodeHndl_t crc_decode_hndl_{}; //!< CRC decode object handle
    std::array<cuphyPuschRxCrcDecodeLaunchCfg_t, 2>
            crc_launch_cfgs_; //!< CRC decode launch configurations
    std::unique_ptr<framework::pipeline::KernelDescriptorAccessor>
            kernel_desc_mgr_;                                  //!< Kernel descriptor manager
    std::array<CUgraphNode, 2> graph_nodes_{nullptr, nullptr}; //!< CUDA graph node handles
};

} // namespace ran::ldpc

#endif // RAN_LDPC_CRC_DECODER_MODULE_HPP
