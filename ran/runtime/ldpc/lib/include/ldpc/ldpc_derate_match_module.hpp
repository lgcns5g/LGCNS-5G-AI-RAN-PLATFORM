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

#ifndef RAN_LDPC_DERATE_MATCH_MODULE_HPP
#define RAN_LDPC_DERATE_MATCH_MODULE_HPP

#include <any>
#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <cuphy.h>
#include <cuphy.hpp>
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
 * @class LdpcDerateMatchModule
 * @brief LDPC derate match module implementing IModule interface
 *
 * This module performs LDPC derate matching on received LLRs (Log-Likelihood
 * Ratios) and outputs derate matched LLRs that can be fed into the LDPC
 * decoder. It supports both CUDA graph execution and direct stream execution
 * modes.
 */
class LdpcDerateMatchModule final : public framework::pipeline::IModule,
                                    public framework::pipeline::IAllocationInfoProvider,
                                    public framework::pipeline::IStreamExecutor,
                                    public framework::pipeline::IGraphNodeProvider {
public:
    /**
     * @brief Static parameters for LDPC derate match module
     */
    struct StaticParams final {
        bool enable_scrambling{true};                      //!< Enable/disable scrambling
        std::size_t max_num_tbs{ran::common::MAX_NUM_TBS}; //!< Maximum number of
                                                           //!< transport blocks
        std::size_t max_num_cbs_per_tb{ran::common::MAX_NUM_CBS_PER_TB}; //!< Maximum number
                                                                         //!< of code blocks
                                                                         //!< per transport
                                                                         //!< block
        std::size_t max_num_rm_llrs_per_cb{MAX_NUM_RM_LLRS_PER_CB};      //!< Maximum rate matching
                                                                         //!< LLRs per code block
        std::size_t max_num_ue_grps{ran::common::MAX_NUM_UE_GRPS};       //!< Maximum number
                                                                         //!< of user groups
    };

    /**
     * @brief Constructor
     *
     * @param instance_id The instance identifier for this module
     * @param init_params Initialization parameters for module configuration
     * @throws std::invalid_argument if the initialization parameters are invalid
     * @throws std::runtime_error if the LDPC derate match object cannot be created
     * @throws std::runtime_error if pinned host memory cannot be allocated
     */
    explicit LdpcDerateMatchModule(std::string instance_id, const std::any &init_params);

    /**
     * @brief Destructor
     */
    ~LdpcDerateMatchModule() override;

    /**
     * Copy constructor - deleted to prevent copying
     */
    LdpcDerateMatchModule(const LdpcDerateMatchModule &other) = delete;

    /**
     * Copy assignment operator - deleted to prevent copying
     */
    LdpcDerateMatchModule &operator=(const LdpcDerateMatchModule &other) = delete;

    /**
     * Move constructor - deleted to prevent moving
     */
    LdpcDerateMatchModule(LdpcDerateMatchModule &&other) = delete;

    /**
     * Move assignment operator - deleted to prevent moving
     */
    LdpcDerateMatchModule &operator=(LdpcDerateMatchModule &&other) = delete;

    /**
     * Get the type identifier
     *
     * @return The type ID as a string view
     */
    [[nodiscard]] std::string_view get_type_id() const override {
        return "ldpc_derate_match_module";
    }

    /**
     * Get the instance identifier
     *
     * @return The instance ID as a string view
     */
    [[nodiscard]] std::string_view get_instance_id() const override;

    /**
     * Perform one-time setup after memory allocation.
     *
     * @param memory_slice Memory slice allocated by PipelineMemoryManager
     */
    void setup_memory(const framework::pipeline::ModuleMemorySlice &memory_slice) override;

    /**
     * Get the input tensor information for a specified port.
     *
     * @param port_name The name of the input port
     * @return Vector of tensor information for all tensors on this port
     * @throws std::invalid_argument if the input port was not found
     */
    [[nodiscard]] std::vector<framework::tensor::TensorInfo>
    get_input_tensor_info(std::string_view port_name) const override;

    /**
     * Get the output tensor information for a specified port.
     *
     * @param port_name The name of the output port
     * @return Vector of tensor information for all tensors on this port
     * @throws std::runtime_error if the output tensor information has not been set
     * @throws std::invalid_argument if the output port was not found
     */
    [[nodiscard]] std::vector<framework::tensor::TensorInfo>
    get_output_tensor_info(std::string_view port_name) const override;

    /**
     * Get the names of all input ports.
     *
     * @return A vector of port names
     */
    [[nodiscard]] std::vector<std::string> get_input_port_names() const override;

    /**
     * Get the names of all output ports.
     *
     * @return A vector of port names
     */
    [[nodiscard]] std::vector<std::string> get_output_port_names() const override;

    /**
     * Set input connections for the module.
     *
     * @param inputs Span of port information with device pointers to input data
     * @throws std::invalid_argument if required inputs are missing or the inputs
     * don't match with the expected inputs
     */
    void set_inputs(std::span<const framework::pipeline::PortInfo> inputs) override;

    /**
     * Get output port information.
     *
     * Returns information about all output ports including their device pointers
     * and tensor metadata. This is used by the pipeline to route data between
     * modules.
     *
     * @return Vector of port information for all outputs
     * @throws std::runtime_error if the module is not setup
     */
    [[nodiscard]] std::vector<framework::pipeline::PortInfo> get_outputs() const override;

    /**
     * Configure I/O for the current iteration.
     *
     * @param[in] params Dynamic parameters for the current iteration
     * @param[in] stream CUDA stream for async operations during configuration
     * @throws std::invalid_argument if the dynamic parameters are invalid
     * @throws std::runtime_error if the underlying cuPHY module setup fails
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
     * @brief Cast the module to a IStreamExecutor
     * @return The casted module
     */
    [[nodiscard]] framework::pipeline::IStreamExecutor *as_stream_executor() override {
        return this;
    }

    /**
     * Execute the module on a given stream.
     *
     * @param stream CUDA stream to execute on
     * @note All dynamic descriptors must have been copied to device before execution.
     * @throws std::runtime_error if the module is not setup
     * @throws std::runtime_error if the kernel launch fails
     */
    void execute(cudaStream_t stream) override;

    /**
     * Get the memory requirements for the module.
     *
     * @return The memory requirements
     * @throws std::runtime_error if the workspace size cannot be determined
     */
    [[nodiscard]] framework::pipeline::ModuleMemoryRequirements get_requirements() const override;

    /**
     * @brief Add node(s) to the graph
     * @param[in] graph The graph to add the node(s) to
     * @param[in] deps The dependencies of the node(s)
     * @return Span of created graph node handle (single node)
     * @throws std::runtime_error if the CUDA graph node addition fails
     */
    [[nodiscard]] std::span<const CUgraphNode> add_node_to_graph(
            gsl_lite::not_null<framework::pipeline::IGraph *> graph,
            const std::span<const CUgraphNode> deps) override;

    /**
     * @brief Update graph node parameters for dynamic parameter changes
     *
     * @param exec The executable graph to update
     * @param params Dynamic parameters containing module-specific parameters
     * @throws std::runtime_error if cuGraphExecKernelNodeSetParams fails
     */
    void update_graph_node_params(
            CUgraphExec exec, const framework::pipeline::DynamicParams &params) override;

private:
    /**
     * Set the tensor descriptor.
     *
     * @param desc The tensor descriptor to set
     * @param tensor_info The tensor information to set the descriptor to
     * @throws std::invalid_argument if the tensor dimensions are invalid
     */
    static void set_tensor_descriptor(
            cuphy::tensor_desc &desc, const framework::tensor::TensorInfo &tensor_info);

    std::string instance_id_; //!< Module instance identifier

    // Static and dynamic parameters
    StaticParams config_; //!< Static parameters from init
    std::optional<PuschOuterRxParams>
            pusch_outer_rx_params_; //!< PUSCH receiver outer_rx parameters, populated in
                                    //!< configure_io

    // Port information
    std::vector<framework::pipeline::PortInfo> inputs_;  //!< Input port information
    std::vector<framework::pipeline::PortInfo> outputs_; //!< Output port information

    // Memory management
    std::unique_ptr<framework::pipeline::KernelDescriptorAccessor>
            kernel_desc_mgr_;                             //!< Kernel descriptor manager
    framework::pipeline::ModuleMemorySlice memory_slice_; //!< Allocated memory slice
    void **de_rm_output_{nullptr};                        //!< Output addresses for derate match

    bool setup_complete_{false}; //!< Whether setup is complete

    // Graph node handle (set during add_node_to_graph)
    CUgraphNode graph_node_{nullptr}; //!< CUDA graph node handle

    // For cuPHY API
    std::vector<cuphyTensorPrm_t> t_prm_llr_vec_;      //!< Tensor parameters for LLR input
    std::vector<cuphyTensorPrm_t> t_prm_llr_cdm1_vec_; //!< Tensor parameters for LLR output (CDM1)
    std::vector<cuphy::tensor_desc> t_desc_llr_vec_;   //!< Tensor descriptor for LLR input
    std::vector<cuphy::tensor_desc>
            t_desc_llr_cdm1_vec_; //!< Tensor descriptor for LLR output (CDM1)
    cuphyPuschRxRateMatchLaunchCfg_t
            kernel_launch_cfg_{};                 //!< LDPC derate match launch configuration
    cuphyPuschRxRateMatchHndl_t pusch_rm_hndl_{}; //!< LDPC derate match object handle
};

} // namespace ran::ldpc

#endif // RAN_LDPC_DERATE_MATCH_MODULE_HPP
