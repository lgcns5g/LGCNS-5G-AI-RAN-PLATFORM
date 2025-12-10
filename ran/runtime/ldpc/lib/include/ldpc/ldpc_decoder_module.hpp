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

#ifndef RAN_LDPC_DECODER_MODULE_HPP
#define RAN_LDPC_DECODER_MODULE_HPP

#include <any>
#include <array>
#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <cuphy.h>
#include <driver_types.h>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wextern-c-compat"
#endif
#include <cuphy.hpp>
#include <ldpc/ldpc_api.hpp>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#include <wise_enum.h>

#include "outer_rx_params.hpp"
#include "pipeline/iallocation_info_provider.hpp"
#include "pipeline/igraph_node_provider.hpp"
#include "pipeline/imodule.hpp"
#include "pipeline/istream_executor.hpp"
#include "pipeline/types.hpp"
#include "ran_common.hpp"
#include "tensor/tensor_info.hpp"

namespace ran::ldpc {

/**
 * Method for determining maximum LDPC decoding iterations
 */
enum class LdpcMaxIterationsMethod : std::uint8_t {
    Fixed = 0, //!< Use fixed max_num_iterations value
    Lut = 1    //!< Use lookup table based on spectral efficiency
};

} // namespace ran::ldpc

// WISE_ENUM_ADAPT must be called at global namespace scope for ADL to work
WISE_ENUM_ADAPT(ran::ldpc::LdpcMaxIterationsMethod, Fixed, Lut)

namespace ran::ldpc {

/**
 * @class LdpcDecoderModule
 * @brief LDPC decoder module implementing IModule interface
 *
 * This module performs LDPC decoding on received LLRs (Log-Likelihood Ratios)
 * and outputs decoded bits.
 */
class LdpcDecoderModule final : public framework::pipeline::IModule,
                                public framework::pipeline::IAllocationInfoProvider,
                                public framework::pipeline::IStreamExecutor,
                                public framework::pipeline::IGraphNodeProvider {
public:
    /**
     * @brief Configuration (static)parameters for LDPC decoder module
     *
     */
    struct StaticParams final {
        float clamp_value{LDPC_CLAMP_VALUE};                 //!< Clamp value for LLRs
        std::size_t max_num_iterations{LDPC_MAX_ITERATIONS}; //!< Maximum number of
                                                             //!< decoding iterations (used when
                                                             //!< max_iterations_method is Fixed)
        std::size_t max_num_cbs_per_tb{
                ran::common::MAX_NUM_CBS_PER_TB};          //!< Maximum number of code blocks
                                                           //!< per transport block
        std::size_t max_num_tbs{ran::common::MAX_NUM_TBS}; //!< Maximum number of transport blocks
        float normalization_factor{LDPC_NORMALIZATION_FACTOR}; //!< Normalization factor for LLRs
        LdpcMaxIterationsMethod max_iterations_method{
                LdpcMaxIterationsMethod::Fixed}; //!< Method for determining max iterations
        std::size_t max_num_ldpc_het_configs{
                LDPC_MAX_HET_CONFIGS}; //!< Maximum number of heterogeneous LDPC
                                       //!< configurations (descriptor set capacity)
    };

    /**
     * @brief Constructor
     *
     * @param instance_id The instance identifier for this module
     * @param init_params Initialization parameters for module configuration
     * @throws std::invalid_argument if initialization parameters are invalid
     */
    explicit LdpcDecoderModule(std::string instance_id, const std::any &init_params);

    /**
     * Get the type identifier
     *
     * @return The type ID as a string_view
     */
    [[nodiscard]] std::string_view get_type_id() const override { return "ldpc_decoder_module"; }

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
     * @throws std::invalid_argument if the port name is invalid or port info not found
     */
    [[nodiscard]] std::vector<framework::tensor::TensorInfo>
    get_input_tensor_info(std::string_view port_name) const override;

    /**
     * Get the output tensor information for a given port name.
     *
     * @param port_name The name of the port to get the tensor information for
     * @return Vector of tensor information for all tensors on this port
     * @throws std::invalid_argument if the port name is invalid or port info not found
     * @throws std::runtime_error if called before setup completion
     */
    [[nodiscard]] std::vector<framework::tensor::TensorInfo>
    get_output_tensor_info(std::string_view port_name) const override;

    /**
     * Get the names of all input ports
     *
     * @return A vector containing "llrs"
     */
    [[nodiscard]] std::vector<std::string> get_input_port_names() const override;

    /**
     * Get the names of all output ports
     *
     * @return A vector containing "decoded_bits"
     */
    [[nodiscard]] std::vector<std::string> get_output_port_names() const override;

    /**
     * Set the inputs for the module.
     *
     * @param inputs Span of port information with device pointers to input data
     * @throws std::invalid_argument if required inputs are missing or port names
     * don't match expected inputs
     */
    void set_inputs(std::span<const framework::pipeline::PortInfo> inputs) override;

    /**
     * Get the output port information.
     *
     * @return Vector of port information for all outputs
     * @note Device pointers are only valid after configure_io() has been called
     * @throws std::runtime_error if called before configure_io() has been called
     */
    [[nodiscard]] std::vector<framework::pipeline::PortInfo> get_outputs() const override;

    /**
     * Configure I/O for the current iteration.
     *
     * @param[in] params Dynamic parameters for the current iteration
     * @param[in] stream CUDA stream for async operations during configuration
     * @throws std::invalid_argument if the dynamic parameters are invalid
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
     * @throws std::runtime_error if called before configure_io() has been called
     * @throws std::runtime_error if outer_rx parameters are not set
     */
    void execute(cudaStream_t stream) override;

    /**
     * Add LDPC decoder nodes to CUDA graph
     *
     * Creates kernel nodes for all LDPC descriptor batches and adds them to the graph.
     * Each descriptor batch becomes a separate kernel node in the graph.
     *
     * @param[in] graph Graph interface for node creation
     * @param[in] deps Dependency nodes that must complete before LDPC decoder nodes execute
     * @return Span of created graph node handles (all LDPC decoder nodes)
     * @throws std::runtime_error if configure_io() was not called before add_node_to_graph()
     */
    [[nodiscard]] std::span<const CUgraphNode> add_node_to_graph(
            gsl_lite::not_null<framework::pipeline::IGraph *> graph,
            std::span<const CUgraphNode> deps) override;

    /**
     * Update graph node parameters for dynamic iteration changes
     *
     * Updates kernel launch parameters in the executable graph. This is called
     * when the dynamic parameters change between iterations.
     *
     * @param[in] exec Graph executable to update
     * @param[in] params Dynamic parameters containing updated LDPC configurations
     * @throws std::runtime_error if cuGraphExecKernelNodeSetParams fails
     */
    void update_graph_node_params(
            CUgraphExec exec, const framework::pipeline::DynamicParams &params) override;

    /**
     * @brief Get the memory requirements for the module.
     * @return The memory requirements
     */
    [[nodiscard]] framework::pipeline::ModuleMemoryRequirements get_requirements() const override;

private:
    /**
     * Prepare LDPC decode descriptors by grouping TBs with identical configurations
     *
     * Groups transport blocks by (base_graph, lifting_size, num_parity_nodes) to enable
     * batched kernel launches. Each descriptor contains multiple TBs with the same config.
     */
    void prepare_ldpc_descriptors();

    /**
     * Calculate maximum LDPC iterations based on spectral efficiency
     *
     * Uses lookup table approach:
     * - High spectral efficiency (>7.2): 7 iterations
     * - Low spectral efficiency (<0.4): 20 iterations
     * - Medium spectral efficiency: 10 iterations
     *
     * @param[in] ldpc_params LDPC parameters containing code rate and other config
     * @param[in] mod_order Modulation order (QPSK, 16QAM, 64QAM, 256QAM)
     * @return Maximum number of iterations to use
     */
    [[nodiscard]] std::size_t static calculate_max_iterations_from_se(
            const LdpcParams &ldpc_params, ModulationOrder mod_order);

    std::string instance_id_; //!< Module instance identifier

    // Static and dynamic parameters
    StaticParams static_params_;                              //!< Static parameters from init
    std::optional<PuschOuterRxParams> pusch_outer_rx_params_; //!< Dynamic parameters from setup

    // Port information
    std::vector<framework::pipeline::PortInfo> inputs_;  //!< Input port information
    std::vector<framework::pipeline::PortInfo> outputs_; //!< Output port information

    framework::pipeline::ModuleMemorySlice memory_slice_; //!< Allocated memory slice

    // Internal state
    bool setup_complete_{false}; //!< Whether setup is complete

    ::cuphy::context ctx_;          //!< cuPHY context
    ::cuphy::LDPC_decoder decoder_; //!< LDPC decoder

    // Descriptor batching for efficient execution
    ::cuphy::LDPC_decode_desc_set ldpc_desc_set_; //!< Descriptor set for batching TBs
    std::vector<cuphyLDPCDecodeLaunchConfig_t> ldpc_launch_cfgs_; //!< Launch configs per descriptor

    // Graph mode support
    std::vector<CUgraphNode> ldpc_graph_nodes_; //!< Graph nodes for each LDPC descriptor batch
};

} // namespace ran::ldpc

#endif // RAN_LDPC_DECODER_MODULE_HPP
