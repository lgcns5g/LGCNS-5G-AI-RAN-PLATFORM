# %% [raw] tags=["remove-cell"]
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# %% [markdown]
# # 7. Complete 5G NR PUSCH Pipeline
#
# This tutorial demonstrates how to implement signal processing pipelines
# using the Aerial Framework's interface-based, modular architecture. You'll
# learn how to compose flexible pipelines from reusable modules.
#
# The example implements a complete 5G NR PUSCH receiver pipeline with an
# inner receiver using a single TensorRT-based module and an outer receiver
# composed of three CUDA-based modules. The CUDA modules use the LDPC library
# from NVIDIA Aerial cuPHY. We show how to build and run the pipeline using
# `cmake` and `ctest`, and how to profile runtime performance using
# [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems).
#
# **Content of this tutorial:**
# 1. Implementing Modules
# 2. Implementing Pipelines
#     - 2.1. Pipeline Implementation
#     - 2.2. Pipeline Memory Management
#     - 2.3. CUDA Graph Building and Execution
# 3. Putting It Together - PUSCH Receiver Pipeline
#     - 3.1. Inner Receiver Module
#     - 3.2. Outer Receiver Modules
#     - 3.3. PUSCH Receiver Pipeline
# 4. Building and Running the Pipeline
# 5. Performance Benchmarking
#     - NVIDIA Nsight Systems profiling

# **Prerequisites:**
# - Completed the following tutorials:
#     - [Getting Started](getting_started.ipynb)
# - Recommended to complete before this tutorial:
#     - [PUSCH Receiver Lowering Tutorial](pusch_receiver_lowering_tutorial.ipynb)
#
# **Time:** ~30 minutes
#
# ---

# %% [markdown]
# ## 1. Implementing Modules
# **Modules** are the building blocks of **pipelines**. Each module
# executes a specific algorithm or set of algorithms to process data.
# A pipeline is composed of a directed graph of modules connected by
# named **ports**. An input port receives **tensors** from the previous
# module, while an output port sends tensors to the next module. The
# pipeline routes tensors between modules by connecting their ports.

# ![Module with Input and Output Ports](
#     ../../figures/generated/pipeline_module.drawio.svg)

# Modules are responsible for:
# - Managing their input and output ports and corresponding tensors
# - Managing their memory requirements
# - Building CUDA graph node(s) for graph-based execution
# - Configuring themselves for execution
# - Processing the data

# In addition to CUDA-based modules, modules can also be implemented using
# [TensorRT](https://developer.nvidia.com/tensorrt). TensorRT-based modules additionally:
# - Configure the TensorRT engine for execution
# - Build TensorRT graph node(s) for graph-based execution
# - Leverage TensorRT for optimized GPU processing

# Modules are implemented by inheriting from multiple interfaces,
# listed in the following:

# `IModule` interface: Each module needs to inherit from the `IModule`
# interface.
#
# <details>
# <summary><b>Click to expand:</b> <i><code>IModule</code> interface
# functions to be implemented</i></summary>
#
# ```cpp
# // Returns the type identifier of the module for identification.
# std::string_view get_type_id()
#
# // Returns the instance identifier of the module.
# std::string_view get_instance_id()
#
# // Called to setup the memory for the module. The memory slice is
# // allocated by the pipeline memory manager.
# void setup_memory(const ModuleMemorySlice &memory_slice)
#
# // Returns the input tensor information for the specified port.
# std::vector<tensor::TensorInfo> get_input_tensor_info(std::string_view port_name)
#
# // Returns the output tensor information for the specified port.
# std::vector<tensor::TensorInfo> get_output_tensor_info(std::string_view port_name)
#
# // Returns the names of all input ports.
# std::vector<std::string> get_input_port_names()
#
# // Returns the names of all output ports.
# std::vector<std::string> get_output_port_names()
#
# // Called by the pipeline to set the inputs for the module. The
# // inputs are typically the outputs of the previous module.
# void set_inputs(std::span<const PortInfo> inputs)
#
# // Returns the outputs for the module.
# std::vector<PortInfo> get_outputs()
#
# // Called by the pipeline to warmup the module for execution. Called
# // once after the first set_inputs() to perform any expensive
# // one-time initialization requiring input/output connection knowledge.
# void warmup(cudaStream_t stream)
#
# // Called by the pipeline to configure the module for execution with
# // the dynamic parameters for the current iteration.
# void configure_io(const DynamicParams &params, cudaStream_t stream)
# ```
#
# </details>
#
# `IAllocationInfoProvider` interface: For pipeline
# memory management, the modules need to inherit from the
# `IAllocationInfoProvider` interface.
#
# <details>
# <summary><b>Click to expand:</b>
# <i><code>IAllocationInfoProvider</code> interface functions
# to be implemented</i></summary>
#
# ```cpp
# // Returns the memory requirements for the module: output tensors,
# // descriptor, and workspace memory required internally.
# ModuleMemoryRequirements get_requirements()
# ```
# </details>
#
# `IStreamExecutor` interface: To support stream-based execution,
# the modules need to inherit from the `IStreamExecutor` interface.
#
# <details>
# <summary><b>Click to expand:</b> <i><code>IStreamExecutor</code>
# interface functions to be implemented</i></summary>
#
# ```cpp
# // Called by the pipeline to execute the module.
# void execute(cudaStream_t stream)
# ```
# </details>
#
# `IGraphNodeProvider` interface: To support graph-based execution,
# the modules need to inherit from the `IGraphNodeProvider`
# interface.
#
# <details>
# <summary><b>Click to expand:</b> <i><code>IGraphNodeProvider</code>
# interface functions to be implemented</i></summary>
#
# ```cpp
# // Called by the pipeline to add the nodes for the module to the
# // graph. The graph is typically the graph for the current iteration.
# void add_node_to_graph(IGraph &graph, std::span<const GraphNodeHandle> deps)
#
# // Called by the pipeline to update the parameters for the module in
# // the graph. The parameters are typically the dynamic parameters for
# // the current iteration.
# void update_graph_node_params(CUgraphExec exec, const DynamicParams &params)
# ```
# </details>
#
# ---
#
# **API Reference:**
# See the [Pipeline API Documentation](../../api/framework/pipeline.rst)
# for detailed interface definitions and usage examples.
#
# For more information about NVIDIA Aerial cuPHY and the CUDA-Accelerated RAN,
# see the [NVIDIA Aerial CUDA-Accelerated RAN documentation](https://docs.nvidia.com/aerial/cuda-accelerated-ran/latest/index.html).
#
# ---
#

# %% [markdown]
# ## 2. Implementing Pipelines
# Pipelines coordinate the execution of multiple modules and manage
# data flow between them. A pipeline creates modules, establishes
# connections between module ports, manages memory allocation, and
# orchestrates execution in either stream mode or graph mode.
#
# The pipelines are responsible for:
# - Creating and managing module instances
# - Establishing data flow connections between modules using
#   module router
# - Allocating and managing memory for all modules
# - Orchestrating module execution in correct order
# - Supporting both stream-based and CUDA graph-based execution
#   modes
#
# ### 2.1. Pipeline Implementation
#
# Pipelines are implemented by inheriting from the `IPipeline`
# interface. Upon construction, the pipeline creates the modules and
# establishes the data flow connections between the modules. The
# pipeline also allocates the memory for all modules.
#
# <details>
# <summary><b>Click to expand:</b> <i><code>IPipeline</code> interface
# functions to be implemented</i></summary>
#
# ```cpp
# // Returns the unique identifier for the pipeline instance.
# std::string_view get_pipeline_id()
#
# // Returns the number of external input tensors required by the
# // pipeline.
# std::size_t get_num_external_inputs()
#
# // Returns the number of external output tensors produced by the
# // pipeline.
# std::size_t get_num_external_outputs()
#
# // Performs one-time setup: creates modules, allocates memory,
# // establishes connections between modules.
# void setup()
#
# // Performs one-time warmup and initialization of all modules.
# // Called once after first configure_io() call. Expensive
# // operations like loading ML models and capturing CUDA graphs
# // happen here.
# void warmup(cudaStream_t stream)
#
# // Configures I/O and dynamic parameters for the current iteration.
# // Maps external inputs to first module's inputs and last module's
# // outputs to external outputs. Called before each execution.
# void configure_io(const DynamicParams &params,
#                   std::span<const PortInfo> external_inputs,
#                   std::span<PortInfo> external_outputs,
#                   cudaStream_t stream)
#
# // Executes the pipeline using stream-based execution. Modules are
# // executed sequentially on the stream.
# void execute_stream(cudaStream_t stream)
#
# // Executes the pipeline using CUDA graph-based execution. Launches
# // the pre-built CUDA graph.
# void execute_graph(cudaStream_t stream)
# ```
#
# </details>
#

# %% [markdown]
# ### 2.2. Pipeline Memory Management
#
# The pipeline provides centralized memory management to efficiently
# allocate and reuse GPU memory across all modules. This enables
# linear memory allocation (minimizes fragmentation) and provides
# predictable memory usage. Memory management works in three phases:
#
# **1. Requirements Collection** - Modules report their memory
#    needs via `IAllocationInfoProvider::get_requirements()`,
#    specifying their needs for descriptors, output tensors and
#    internal workspace memory with alignment constraints. Note that
#    this requires that the modules actually implement the
#    `IAllocationInfoProvider` interface.
#
# **2. Allocation** - The `PipelineMemoryManager` member object of
#    the pipeline analyzes all requirements, identifies memory reuse
#    opportunities, and allocates common CPU and GPU buffers for all
#    modules.
#
# **3. Distribution** - Memory slices are distributed to modules
#    via `setup_memory()`. Each module receives fixed memory
#    addresses for descriptors and output tensors and workspace that
#    remain constant throughout execution.
#
# ![Pipeline Memory Management](../../figures/generated/pipeline_memory_management.svg)
#

# %% [markdown]
# ### 2.3. CUDA Graph Building and Execution
#
# CUDA graphs enable the pipeline to launch all GPU operations with
# a single call, reducing kernel launch latency and CPU overhead.
# The `GraphManager` member object of the pipeline manages the graph
# construction and execution.
#
# There are four phases in the graph lifecycle:
#
# **1. Construction** - The `GraphManager` object is responsible
#    for constructing the graph. The `GraphManager` calls each
#    module implementing `IGraphNodeProvider` to add its nodes to
#    the graph via `add_node_to_graph()`.
#
# **2. Instantiation** - The graph structure is compiled into an
#    executable instance and uploaded to the GPU by calling
#    `instantiate_graph()` and `upload_graph()`. The graph is then
#    ready for execution.
#
# **3. Execution** - The entire pipeline launches with a single
#    `launch_graph()` call instead of multiple individual kernel
#    launches.
#
# **4. Update** - Dynamic parameters are updated via
#    `update_graph_node_params()` without rebuilding the graph
#    structure.
#
# ![Pipeline CUDA Graph Lifecycle](../../figures/generated/pipeline_cuda_graph.svg)
#
# **TensorRT Integration:** TensorRT operations are captured into
# a subgraph during warmup and added as a child graph node in the
# same way as the other modules.
#

# %% [markdown]
# ## 3. Putting It Together - PUSCH Receiver Pipeline

# In the following, we will walk through the implementation of the
# PUSCH receiver pipeline as an example of a complete signal processing pipeline.

# The PUSCH receiver pipeline is composed of four modules:
# - Inner receiver module implemented using TensorRT.
# - Outer receiver composed of three CUDA-based modules.
#   - LDPC rate matching module
#   - LDPC decoding module
#   - CRC decoding module
#
# The outer receiver is implemented using NVIDIA Aerial cuPHY library.

# ![PUSCH Receiver Pipeline](
#     ../../figures/generated/pusch_pipeline.drawio.svg)
#
# ---
#
# ### 3.1. Inner Receiver Module

# The inner receiver module is implemented using TensorRT.
# It performs the following operations:
# - Channel estimation
# - Noise covariance estimation
# - Equalization
# - Soft demapping
#
# The input to the inner receiver module is the received resource
# grid. The outputs are the log-likelihood ratios (LLRs),
# post-equalization noise variance and SINR estimates. For the detailed
# implementation of the TensorRT engine, see the
# [PUSCH Receiver Lowering Tutorial](pusch_receiver_lowering_tutorial.ipynb).
#
# Implementing a module based on TensorRT requires several key
# steps beyond the standard module interfaces. These are described
# below.
#
# **Note:** Additionally, modules based on TensorRT
# require a TRT engine file. The creation of that is beyond the
# scope of this tutorial.
#
# **1. Configure the TensorRT engine with input and output
#    parameters**
#    - Define the input tensor shapes, data types, and formats
#      that the TRT engine expects
#    - Configure output tensor specifications including shapes
#      and data types
#
# <details>
# <summary><b>Click to expand:</b> <i>Example: configuring
# TensorRT engine with input and output parameters</i></summary>
#
# ```cpp
#    const tensorrt::MLIRTensorParams xtf{
#            .name = "arg0", // Input XTF
#            .data_type = tensor::TensorR32F,
#            .rank = 4,
#            .dims = {num_rx_ant, num_ofdm_symbols, num_subcarriers, num_real_imag_interleaved}};
#
#    const tensorrt::MLIRTensorParams post_eq_noise_var_db{
#            .name = "result0", // Output post-equalizer noise var dB
#            .data_type = tensor::TensorR32F,
#            .rank = 1,
#            .dims = {ran::common::MAX_UES_PER_SLOT}};
#
#    const tensorrt::MLIRTensorParams post_eq_sinr_db{
#            .name = "result1", // Output post-equalizer SINR dB
#            .data_type = tensor::TensorR32F,
#            .rank = 1,
#            .dims = {ran::common::MAX_UES_PER_SLOT}};
#
#    const tensorrt::MLIRTensorParams llr{
#            .name = "result2", // Output LLRs
#            .data_type = tensor::TensorR16F,
#            .rank = 4,
#            .dims = {
#                    ran::common::OFDM_SYMBOLS_NORMAL_CYCLIC_PREFIX -
#                            ran::common::MAX_DMRS_OFDM_SYMBOLS,
#                    static_cast<std::size_t>(num_prb) * ran::common::NUM_SUBCARRIERS_PER_PRB,
#                    ran::common::MAX_UL_LAYERS,
#                    ran::common::MAX_QAM_BITS}};
#    const std::vector<tensorrt::MLIRTensorParams> inputs = {xtf};
#    const std::vector<tensorrt::MLIRTensorParams> outputs = {
#            post_eq_noise_var_db, post_eq_sinr_db, llr};
# ```
#
# </details>
#
# **2. Create the TensorRT Engine**
#    - The `TrtEngine` and `MLIRTrtEngine` classes are provided
#      by the Aerial Framework.
#      - The `TrtEngine` class is a wrapper around the TensorRT
#        engine.
#      - The `MLIRTrtEngine` class is a wrapper around the
#        TensorRT engine that provides a simplified interface. In this
#        tutorial, we use the `MLIRTrtEngine` class.
#    - The `MLIRTrtEngine` class requires an instance of the
#      TRT runtime implementing the `ITrtEngine` interface. It also requires
#      an instance of the `IPrePostTrtEngEnqueue` interface. This is used here to
#      capture the TensorRT engine operations into CUDA graphs
#      for graph-based pipeline execution.
#
# <details>
# <summary><b>Click to expand:</b> <i>Example: creating the TensorRT
# engine</i></summary>
#
# ```cpp
#    auto tensorrt_runtime =
#            std::make_unique<tensorrt::TrtEngine>(params.trt_engine_path, *trt_logger_);
#
#    // Create graph capturer based on execution mode
#    auto create_graph_capturer = [this]() -> std::unique_ptr<tensorrt::IPrePostTrtEngEnqueue> {
#        if (execution_mode_ == core::ExecutionMode::Graph) {
#            auto capturer = std::make_unique<tensorrt::CaptureStreamPrePostTrtEngEnqueue>();
#            graph_capturer_ = capturer.get();
#            return capturer;
#        }
#        auto capturer = std::make_unique<tensorrt::NullPrePostTrtEngEnqueue>();
#        graph_capturer_ = capturer.get();
#        return capturer;
#    };
#
#    // Create MLIR TRT engine with the runtime and capture helper (takes ownership of both)
#    trt_engine_ = std::make_unique<tensorrt::MLIRTrtEngine>(
#            inputs, outputs, std::move(tensorrt_runtime), create_graph_capturer());
# ```
# </details>
#
# **3. Graph capture for CUDA graph execution**
#    - TensorRT engine operations must be captured into CUDA
#      graphs for graph-based pipeline execution. Graph capture
#      enables:
#      - Lower latency by reducing CPU overhead
#      - Better GPU utilization through optimized kernel
#        scheduling
#      - Deterministic execution patterns
#    - The TensorRT engine operations are captured into CUDA
#      graphs during the `warmup()` call.
#    - The module's `add_node_to_graph()` implementation should
#      add TensorRT execution as graph node(s).
#
# <details>
# <summary><b>Click to expand:</b> <i>Example: capturing the
# TensorRT engine operations into CUDA graphs</i></summary>
#
# ```cpp
#
#    void InnerRxModule::warmup(cudaStream_t stream) {
#
#        // Configure TRT engine with FIXED tensor addresses (one-time setup)
#        // Use internal fixed buffers for all tensors
#        const std::vector<void *> input_buffers = {d_xtf_};
#        const std::vector<void *> output_buffers = {
#               d_post_eq_noise_var_db_, d_post_eq_sinr_db_, d_llr_};
#
#        RT_LOGC_INFO(PuschComponent::InnerRxModule,
#                     "'{}': Calling TRT engine setup()", instance_id_);
#        const utils::NvErrc setup_result = trt_engine_->setup(input_buffers, output_buffers);
#        if (setup_result != utils::NvErrc::Success) {
#            const std::string error_msg =
#                    std::format("InnerRxModule '{}': TRT engine setup() failed", instance_id_);
#            RT_LOGC_ERROR(PuschComponent::InnerRxModule, "{}", error_msg);
#           throw std::runtime_error(error_msg);
#        }
#
#        // Use provided stream for warmup/graph capture
#        // Note: TensorRT graph capture requires a non-default stream
#        // (cannot use cudaStreamDefault)
#        RT_LOGC_INFO(PuschComponent::InnerRxModule,
#                     "'{}': Calling TRT engine warmup()", instance_id_);
#        const utils::NvErrc warmup_result = trt_engine_->warmup(stream);
#        if (warmup_result != utils::NvErrc::Success) {
#            const std::string error_msg =
#                    std::format("InnerRxModule '{}': TRT engine warmup() failed", instance_id_);
#            RT_LOGC_ERROR(PuschComponent::InnerRxModule, "{}", error_msg);
#            throw std::runtime_error(error_msg);
#        }
#    }
# ```
# </details>
#
# <details>
# <summary><b>Click to expand:</b> <i>Example: adding the TensorRT
# engine to the pipeline graph</i></summary>
#
# ```cpp

#    std::span<const CUgraphNode> InnerRxModule::add_node_to_graph(
#            gsl_lite::not_null<core::IGraph *> graph, std::span<const CUgraphNode> deps) {
#
#        const auto *capturer =
#            dynamic_cast<const tensorrt::CaptureStreamPrePostTrtEngEnqueue *>(graph_capturer_);
#
#        ...
#
#        CUgraph trt_graph = capturer->get_graph();
#
#        ...
#
#        // Add TensorRT subgraph as child graph node and store the handle
#        trt_node_ = graph->add_child_graph_node(deps, trt_graph);
#
#        ...
#
#        return {&trt_node_, 1};
#    }
# ```
# </details>
#
# **4. Running the TensorRT Engine**
#    - Execute by calling the TRT engine's `run()` method
#    - Handle errors appropriately with proper logging and
#      exception handling
#    - Update dynamic parameters if needed for each iteration
#
# <details>
# <summary><b>Click to expand:</b> <i>Example: running the TRT
# engine</i></summary>
#
# ```cpp
#    const utils::NvErrc execute_result = trt_engine_->run(stream);
#    if (execute_result != utils::NvErrc::Success) {
#        const std::string error_msg =
#                std::format("InnerRxModule '{}': TRT engine run() failed", instance_id_);
#        RT_LOGC_ERROR(PuschComponent::InnerRxModule, "{}", error_msg);
#        throw std::runtime_error(error_msg);
#    }
# ```
# </details>
#

# A key aspect of TensorRT integration is that graph capture
# happens automatically during the `warmup()` call, and the captured
# graph can then be incorporated into the pipeline's CUDA graph for
# efficient graph-based execution.
#
# ---
#
# **API Reference:**
# See the [PUSCH API Documentation](../../api/ran/runtime/pusch.rst)
# for detailed module interface definitions and usage examples.
#
# ---
#
# ### 3.2. Outer Receiver Modules

# The outer receiver module is composed of three CUDA-based modules:
# - LDPC derate matching module
# - LDPC decoding module
# - CRC decoding module
#
# For purposes of this tutorial, we use the LDPC
# derate matching module as an example. Other modules are
# implemented similarly. All modules need to implement the interfaces
# listed in the first section. We focus here on the essential parts, and the
# rest of the interface implementation can be found in the files listed
# below.
#
# **1. Module Creation**
# All modules contain a module-specific `StaticParams` struct that
# defines the static configuration parameters for the module. This
# is passed to the constructor upon creation of the module.
#
# <details>
# <summary><b>Click to expand:</b>
# <i>Example: LDPC derate matching module static parameters</i></summary>
#
# ```cpp
#    struct StaticParams final {
#        bool enable_scrambling{true};                      //!< Enable/disable scrambling
#        std::size_t max_num_tbs{ran::common::MAX_NUM_TBS}; //!< Maximum number of
#                                                           //!< transport blocks
#        std::size_t max_num_cbs_per_tb{ran::common::MAX_NUM_CBS_PER_TB}; //!< Maximum number
#                                                                         //!< of code blocks
#                                                                         //!< per transport
#                                                                         //!< block
#        std::size_t max_num_rm_llrs_per_cb{MAX_NUM_RM_LLRS_PER_CB};      //!< Maximum rate matching
#                                                                         //!< LLRs per code block
#        std::size_t max_num_ue_grps{ran::common::MAX_NUM_UE_GRPS};       //!< Maximum number
#                                                                         //!< of user groups
#    };
# ```
# </details>
#

# All modules are constructed with the same pattern, i.e. they accept
# `std::any` as initialization parameters, and cast it to the module's
# specific parameters. The constructor then performs one-time
# initialization. This includes validating the initialization
# parameters, creating the needed cuPHY objects and pre-allocating host
# memory. The static parameters define maximum capacities and features
# that won't change between executions.
#
# <details>
# <summary><b>Click to expand:</b> <i>Example: LDPC derate matching module creation</i></summary>
#
# ```cpp
#    try {
#        config_ = std::any_cast<StaticParams>(init_params);
#    } catch (const std::bad_any_cast &e) {
#        const std::string error_message = "Invalid initialization parameters!";
#        RT_LOGC_ERROR(LdpcComponent::DerateMatch, "{}", error_message);
#        throw std::invalid_argument(error_message);
#    }
#
#    constexpr int FP_CONFIG = 3; // FP16 in, FP16 out
#    const cuphyStatus_t status = cuphyCreatePuschRxRateMatch(
#            &pusch_rm_hndl_, FP_CONFIG, static_cast<int>(config_.enable_scrambling));
#    if (status != CUPHY_STATUS_SUCCESS) {
#        const std::string error_message = "Failed to create LDPC derate match object!";
#        RT_LOGC_ERROR(LdpcComponent::DerateMatch, "{}", error_message);
#        throw std::runtime_error(error_message);
#    }
# ```
# </details>
#
# **2. Memory Management**
#
# As described in the first section, modules declare their memory needs via
# `get_requirements()`, and get in return the allocated memory slice
# in `setup_memory()`. In case of cuPHY components, the memory needs
# are obtained using cuPHY API calls. Additionally, the module needs
# memory for its own output tensors.
#
# <details>
# <summary><b>Click to expand:</b>
# <i>Example: LDPC derate matching module get_requirements()</i></summary>
#
# ```cpp
#    adsp::ModuleMemoryRequirements LdpcDerateMatchModule::get_requirements() const {
#        std::size_t dyn_descr_size_bytes{};
#        std::size_t dyn_descr_align_bytes{};
#        if (const cuphyStatus_t status =
#                    cuphyPuschRxRateMatchGetDescrInfo(&dyn_descr_size_bytes,
#                                                     &dyn_descr_align_bytes);
#            status != CUPHY_STATUS_SUCCESS) {
#            const std::string error_message = "Failed to get workspace size for LDPC derate match";
#            RT_LOGC_ERROR(LdpcComponent::DerateMatch, "{}", error_message);
#            throw std::runtime_error(error_message);
#        }
#
#        static constexpr std::size_t NUM_BYTES_PER_LLR = sizeof(__half);
#        const std::size_t n_bytes = NUM_BYTES_PER_LLR * config_.max_num_rm_llrs_per_cb *
#                                    config_.max_num_cbs_per_tb * config_.max_num_tbs;
#
#        adsp::ModuleMemoryRequirements req;
#        req.static_kernel_descriptor_bytes = 0;
#        req.dynamic_kernel_descriptor_bytes = dyn_descr_size_bytes;
#        req.device_tensor_bytes = n_bytes;
#        req.alignment = dyn_descr_align_bytes;
#        return req;
#    }
# ```
# </details>
#
# In `setup_memory()`, the allocated memory slice is just stored in
# a member variable. Additionally. the module creates a
# `KernelDescriptorAccessor` to manage the kernel descriptors.
#
# <details>
# <summary><b>Click to expand:</b> <i>Example: LDPC derate matching
# module setup_memory()</i></summary>
#
# ```cpp
#
#    void LdpcDerateMatchModule::setup_memory(const adsp::ModuleMemorySlice &memory_slice) {
#        memory_slice_ = memory_slice;
#
#        ...
#
#        kernel_desc_mgr_ = std::make_unique<adsp::KernelDescriptorAccessor>(memory_slice);
#    }
# ```
# </details>
#
# **3. configure_io()**
#
# Called before each execution, this method sets up the module's
# input/output tensors and kernel parameters for the current
# iteration. It receives dynamic parameters that describe the
# specific workload (e.g., number of transport blocks, code block
# configurations). Based on these parameters, the module configures
# output tensor shapes, sets up device pointers within its allocated
# memory slice, and prepares the kernel launch configuration.
#
# **4. execute()**
#
# This is the core execution method that launches the module's CUDA
# kernel(s) on the provided stream. After running `configure_io()`,
# all configuration is complete and the kernel launch parameters are
# ready. The module extracts the kernel configuration prepared
# during `configure_io()` and launches it on the stream.
#
# <details>
# <summary><b>Click to expand:</b> <i>Example: LDPC derate matching
# module execution code</i></summary>
#
# ```cpp
#    const CUDA_KERNEL_NODE_PARAMS &kernel_node_params_driver =
#            kernel_launch_cfg_.kernelNodeParamsDriver;
#    AERIAL_DSP_CUDA_DRIVER_CHECK_THROW(adsp::launch_kernel(kernel_node_params_driver, stream));
# ```
# </details>
#
# **5. add_node_to_graph()**
#
# For CUDA graph-based execution, this method adds the module's
# kernel as a node in the computational graph. It receives
# dependency nodes and returns its own node handle, allowing the
# framework to chain operations correctly. The module uses the same
# kernel launch configuration prepared during `configure_io()`, but
# instead of launching directly, it registers with the graph. The
# node handle is stored for later parameter updates via
# `update_graph_node_params()`.
#
# <details>
# <summary><b>Click to expand:</b> <i>Example: LDPC derate matching
# module add_node_to_graph()</i></summary>
#
# ```cpp
# std::span<const CUgraphNode> LdpcDerateMatchModule::add_node_to_graph(
#        gsl_lite::not_null<adsp::IGraph *> graph, const std::span<const CUgraphNode> deps) {
#
#    ...
#
#    // Add kernel node using kernel params from kernel_config_
#    graph_node_ = graph->add_kernel_node(deps, kernel_launch_cfg_.kernelNodeParamsDriver);
#    if (graph_node_ == nullptr) {
#        const std::string error_message = "Failed to add kernel node to graph";
#        RT_LOGC_ERROR(LdpcComponent::DerateMatch, "{}", error_message);
#        throw std::runtime_error(error_message);
#    }
#
#    ...
#
#    return {&graph_node_, 1};
# }
# ```
# </details>
#
# For more information and examples, see the implementation files
# of our LDPC modules.
#
# ---
#
# **API Reference:**
# See the [LDPC API Documentation](../../api/ran/runtime/ldpc.rst)
# for detailed module interface definitions and usage examples.
#
# ---
#
# ### 3.3. PUSCH Receiver Pipeline
#
# The PUSCH receiver pipeline brings together all the modules described above
# into a cohesive processing chain. The pipeline coordinates module
# execution, manages data flow between modules, and supports both
# stream-based and graph-based execution modes.
#
# **1. Module Construction**
#
# In the pipeline constructor, all four modules (inner receiver,
# derate match, LDPC decoder, CRC decoder) are created with their
# respective static parameters. The modules are stored in the
# pipeline and later used to establish data flow connections.
# We use module factory pattern to create the modules, and store them
# in the pipeline.
#
# <details>
# <summary><b>Click to expand:</b> <i>Example: PUSCH pipeline module construction</i></summary>
#
# ```cpp
#    // Map module types to their corresponding member unique_ptrs using reference_wrapper
#    std::unordered_map<std::string_view, std::reference_wrapper<std::unique_ptr<core::IModule>>>
#            module_map = {
#                    {"inner_rx_module", std::ref(inner_rx_module_)},
#                    {"ldpc_derate_match_module", std::ref(ldpc_derate_match_module_)},
#                    {"ldpc_decoder_module", std::ref(ldpc_decoder_module_)},
#                    {"crc_decoder_module", std::ref(crc_decoder_module_)}};

#    for (const auto &module_spec : spec.modules) {
#        const auto &module_info = module_spec.get();
#        auto module = module_factory_->create_module(
#                module_info.module_type, module_info.instance_id, module_info.init_params);
#        if (!module) {
#            const std::string error_msg =
#                    std::format("Failed to create module '{}'", module_info.instance_id);
#            RT_LOGEC_ERROR(
#                    PuschComponent::PuschPipeline,
#                    PuschPipelineEvent::CreateModules,
#                    "{}",
#                    error_msg);
#            throw std::runtime_error(error_msg);
#        }

#        try {
#            module_map.at(module_info.module_type).get() = std::move(module);
#        } catch (const std::out_of_range &) {
#            const std::string error_msg =
#                    std::format("Unknown module type '{}'", module_info.module_type);
#            RT_LOGC_ERROR(PuschComponent::PuschPipeline, "{}", error_msg);
#            throw std::runtime_error(error_msg);
#        }
#        RT_LOGC_INFO(
#                PuschComponent::PuschPipeline,
#                "Successfully created module '{}'",
#                module_info.instance_id);
#    }
# ```
# </details>
#
# **2. Memory Allocation**
#
# The pipeline queries memory requirements from all modules via
# `get_requirements()` and uses `PipelineMemoryManager` to allocate
# a single contiguous memory pool. Each module then receives its
# memory slice via `setup_memory()`, enabling efficient memory reuse
# across the pipeline.
#
# <details>
# <summary><b>Click to expand:</b> <i>Example: PUSCH pipeline memory allocation</i></summary>
#
# ```cpp
#    memory_mgr_ = core::PipelineMemoryManager::create_for_modules(modules_);
#
#    // Allocate memory slices for all modules
#    memory_mgr_->allocate_all_module_slices(modules_);
#
#    // Call setup_memory() on each module with their memory slice
#    for (auto *module : modules_) {
#        const auto slice = memory_mgr_->get_module_slice(module->get_instance_id());
#        module->setup_memory(slice);
#    }
# ```
# </details>
#
# **3. configure_io()**
#
# This method chains the modules together by connecting outputs of
# one module to inputs of the next. It also maps external inputs to the
# first module's inputs and the correct outputs to the pipeline's external
# outputs. Each module's `configure_io()` is called in sequence to prepare
# for execution.
#
# **4. execute_stream()**
#
# For stream-based execution, the pipeline takes advantage of the
# `IStreamExecutor` interface to call each module's
# `execute()` method sequentially on the provided CUDA stream.
#
# <details>
# <summary><b>Click to expand:</b> <i>Example: PUSCH pipeline stream execution</i></summary>
#
# ```cpp
#    for (auto *module : modules_) {
#        auto *stream_executor = module->as_stream_executor();
#        if (stream_executor != nullptr) {
#            stream_executor->execute(stream);
#        }
#    }
# ```
# </details>
#
# **5. build_graph()**
#
# This method constructs a CUDA graph by calling each module's
# `add_node_to_graph()` with appropriate dependencies.
#
# <details>
# <summary><b>Click to expand:</b> <i>Example: PUSCH pipeline graph construction</i></summary>
#
# ```cpp
#    // Create graph manager (constructor implicitly creates empty CUDA graph)
#    graph_manager_ = std::make_unique<core::GraphManager>();
#
#    // Add modules to graph in execution order
#    std::vector<CUgraphNode> prev_nodes;
#
#    for (auto *module : modules_) {
#        auto *graph_provider = module->as_graph_node_provider();
#        if (graph_provider == nullptr) {
#            const std::string error_msg = std::format(
#                    "PuschPipeline '{}': Module '{}' does not implement "
#                    "IGraphNodeProvider",
#                    pipeline_id_,
#                    std::string(module->get_instance_id()));
#            RT_LOGC_ERROR(PuschComponent::PuschPipeline, "{}", error_msg);
#            throw std::runtime_error(error_msg);
#        }
#        // Add module's node(s) to graph with dependencies on previous nodes
#        const auto nodes = graph_manager_->add_kernel_node(
#                gsl_lite::not_null<core::IGraphNodeProvider *>(graph_provider), prev_nodes);
#
#        // Current node(s) become dependencies for next module
#        prev_nodes.assign(nodes.begin(), nodes.end());
#    }
#
#    // Instantiate and upload graph
#    graph_manager_->instantiate_graph();
#    graph_manager_->upload_graph(stream);
#
# ```
# </details>
#
# **6. execute_graph()**
#
# For graph-based execution, the pipeline launches the pre-built
# CUDA graph with a single call. Dynamic parameters could still be
# updated via `update_graph_node_params()` before each launch, but in
# our case this is done in `configure_io()`.
#
# <details>
# <summary><b>Click to expand:</b> <i>Example: PUSCH pipeline graph execution</i></summary>
#
# ```cpp
#    auto *const exec = graph_manager_->get_exec();
#    const core::DynamicParams dummy_params{}; // We don't use params for parameter updates
#    for (auto *module : modules_) {
#        auto *graph_node_provider = module->as_graph_node_provider();
#        if (graph_node_provider != nullptr) {
#            graph_node_provider->update_graph_node_params(exec, dummy_params);
#        }
#    }
#
#    graph_manager_->launch_graph(stream);
# ```
# </details>
#
# ## 4. Building and Running the Pipeline
#
# The above pipeline is implemented in the `PuschPipeline` class.
# This section shows how to build and run it using `cmake` and `ctest`.

# %%
import os
import sys

# Import shared tutorial utilities from tutorial_utils.py (in the same
# directory) Contains helper functions for Docker container interaction
# and project navigation
from tutorial_utils import (
    build_cmake_target,
    check_container_running,
    configure_cmake,
    get_project_root,
    is_running_in_docker,
    run_container_command,
    show_output,
)

IN_DOCKER = is_running_in_docker()
PROJECT_ROOT = get_project_root()
CONTAINER_NAME = f"aerial-framework-base-{os.environ.get('USER', 'default')}"

print(f"Project root: {PROJECT_ROOT}")
if IN_DOCKER:
    print("✅ Running inside Docker container")
else:
    print(f"Running on host, will use container: {CONTAINER_NAME}")
    check_container_running(CONTAINER_NAME)
    print(f"✅ Container '{CONTAINER_NAME}' is running")
print("✅ Step 4a complete: Environment setup verified")

# %% [markdown]
# **Configure CMake preset:**

# %%
preset = "gcc-release"
build_dir = PROJECT_ROOT / "out" / "build" / preset

configure_cmake(build_dir, preset)
print("✅ Step 4b complete: CMake configured")

# %% [markdown]
# **Build PUSCH pipeline targets:**
#
# Build `pusch_all` which compiles the PUSCH pipeline, its components
# and the associated tests and benchmarks. Also build
# `py_ran_setup` which sets up the Python environment used to
# build the TensorRT engine for the inner receiver module.

# %%
# Build pusch tests and benchmark and TRT engine generation target
try:
    build_cmake_target(build_dir, ["pusch_all", "py_ran_setup", "sync_env_python"])
except RuntimeError as e:
    print(f"❌ Build failed: {e}")
    print("\nNote: Error message shows last few lines of output.")
    print("If build fails, enter the container to run commands manually and view full logs:")
    print("  docker exec -it aerial-framework-base-$USER bash -l")
    print(f"  cmake --build out/build/{preset} --target pusch_all py_ran_setup sync_env_python")
    sys.exit(1)
print("✅ Step 4c complete: PUSCH pipeline targets built")

# %% [markdown]
# **Run graph mode tests for the complete PUSCH pipeline:**
#
# Use `GTEST_FILTER` environment variable with ctest to run specific test subsets:
# - `GraphMode/*` - Graph execution mode tests
# - `StreamMode/*` - Stream execution mode tests
# - `PuschPipelineTest.*` - All PUSCH pipeline tests
#
# ```bash
# GTEST_FILTER=GraphMode/* ctest --preset -R pusch_tests -V
# ```
# Note: This cell displays only the last few lines of output. Run the
# command manually to see the full logs.

# %%
cmd = f"env GTEST_FILTER=GraphMode/* ctest --preset {preset} -R pusch_tests -V"
result = run_container_command(cmd, CONTAINER_NAME, cwd=PROJECT_ROOT)

show_output(result, lines=500)

if result.returncode == 0:
    print("✅ Tests completed successfully")
else:
    print("⚠️ Test execution completed with warnings")
    sys.exit(1)
print("✅ Step 4d complete: Graph mode tests executed")

# %% [markdown]
# ## 5. Performance Benchmarking
#
# The PUSCH pipeline includes Google Benchmark tests for performance measurement
# and NVIDIA Nsight Systems (nsys) support for detailed profiling.

# %% [markdown]
# **Run benchmarks for PUSCH pipeline and inner_rx module:**
#
# Benchmarks measure execution time and provide statistics including min, mean, median,
# p95, max, and standard deviation in microseconds.
#
# **Results are saved to:**
# - `out/build/gcc-release/benchmark_results/pusch_pipeline_bench_results.json`
# - `out/build/gcc-release/benchmark_results/inner_rx_module_bench_results.json`

# %%
print("Running PUSCH pipeline and inner receiver module benchmarks...")

cmd = f"ctest --preset {preset} -R 'pusch_pipeline_bench|inner_rx_module_bench' -V"
result = run_container_command(cmd, CONTAINER_NAME, cwd=PROJECT_ROOT)

show_output(result, lines=500)

if result.returncode == 0:
    print("✅ Benchmarks completed successfully")
else:
    print("⚠️ Benchmark execution completed with warnings")
    sys.exit(1)
print("✅ Step 5a complete: Benchmarks executed")

# %% [markdown]
# **Run NVIDIA Nsight Systems profiling:**
#
# Profile the benchmarks to analyze GPU performance with detailed kernel traces.
#
# **Reports are saved to:**
# - `out/build/gcc-release/nsys_results/pusch_pipeline.nsys-rep`
# - `out/build/gcc-release/nsys_results/inner_rx_module.nsys-rep`
#
# %%
print("Running nsys profiling for PUSCH pipeline and inner_rx module...")

cmd = f"ctest --preset {preset} -R 'pusch_pipeline_nsys|inner_rx_module_nsys' -V"
result = run_container_command(cmd, CONTAINER_NAME, cwd=PROJECT_ROOT)

show_output(result, lines=1000)

if result.returncode == 0:
    print("✅ Profiling completed successfully")
else:
    print("⚠️ Profiling completed with warnings")
    sys.exit(1)
print("✅ Step 5b complete: Nsight Systems profiling finished")

# %% [markdown]
# **Analyzing profiling results:**
#
# To analyze the nsys profiling reports with [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems):
#
# **If you are using a remote machine**, first copy the reports to your local machine:
# ```bash
# # From local machine, copy nsys reports from remote host
# scp -C "<remote-host>:<aerial-framework-host-dir>/out/build/gcc-release/nsys_results/*.nsys-rep" .
# ```
#
# **Then open in Nsight Systems GUI:**
# ```bash
# nsys-ui
# # Then use: File -> Open -> pusch_pipeline.nsys-rep
# # or: File -> Open -> inner_rx_module.nsys-rep
# ```
#
# **If running locally**, open the reports directly from the output directory.
#
# The nsys reports provide detailed performance analysis including:
# - CUDA kernel execution times and occupancy
# - Memory transfers between host and device
# - CUDA API calls and synchronization points
# - NVTX ranges for custom profiling markers
#
# ---
#
# ## Troubleshooting
#
# **Docker and Setup:**
# - **Not running in Docker:** Ensure the Docker container is running.
#   Check status: `docker ps | grep aerial-framework-base`;
#   restart if needed (see Getting Started tutorial).
#
# **Build Issues:**
# - **Configuration or build fails:** If CMake configure or build steps fail, enter the container
#   to run commands manually and view complete logs:
#   - Enter container: `docker exec -it aerial-framework-base-$USER bash -l`
#   - Configure: `cmake --preset gcc-release -DENABLE_CLANG_TIDY=OFF -DENABLE_IWYU=OFF`
#   - Build: `cmake --build out/build/gcc-release --target pusch_all py_ran_setup`
# - **View full build/test output:** If build or tests fail, enter the container to
#   run commands manually and view complete logs.
#
# **Environment Issues:**
# - **Python environment problems:** Clean and rebuild the RAN Python environment:
#   `rm -rf ran/py/.venv && cmake --build out/build/gcc-release -t py_ran_setup`
#
