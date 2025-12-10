# TensorRT + ReLU Pipeline Architecture

**Document Version:** 1.5
**Date:** October 8, 2025
**Status:** Reference Implementation

## Table of Contents

1. [Overview](#overview)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Execution Modes](#execution-modes)
4. [Module A - TensorRT Addition Engine](#module-a---tensorrt-addition-engine)
5. [Module B - CUDA ReLU Kernel](#module-b---cuda-relu-kernel)
6. [Memory Management](#memory-management)
7. [Stream Capture and Graph Execution](#stream-capture-and-graph-execution)
8. [Factory Pattern Integration](#factory-pattern-integration)
9. [Testing Strategy](#testing-strategy)
10. [Data Flow Diagrams](#data-flow-diagrams)
11. [Zero-Copy Optimization](#zero-copy-optimization)

---

## Overview

The Sample Pipeline demonstrates a complete two-module pipeline that chains a TensorRT engine with a custom CUDA kernel:

```text
External Input 0 ──┐
                   ├─→ ModuleA (TensorRT Add) ─→ ModuleB (CUDA ReLU) ─→ External Output
External Input 1 ──┘
```

**Pipeline Operation:**

- **ModuleA**: Performs element-wise addition using TensorRT: `output = input0 + input1`
- **ModuleB**: Applies ReLU activation using custom CUDA kernel: `output[i] = max(0, input[i])`
- **Combined**: Computes `ReLU(input0 + input1)`

**Execution Modes:**

- **Stream Mode**: Sequential module execution on a CUDA stream with flexible addressing
- **Graph Mode**: Pre-built CUDA graph executed as a single unit with fixed addressing (lower latency)

See [Execution Modes](#execution-modes) for detailed comparison and when to use each mode.

**Implementation Files:**

- [sample_module_a.hpp](../src/sample_module_a.hpp) / [sample_module_a.cpp](../src/sample_module_a.cpp)
- [sample_module_b.hpp](../src/sample_module_b.hpp) / [sample_module_b.cpp](../src/sample_module_b.cpp)
- [sample_pipeline.hpp](../src/sample_pipeline.hpp) / [sample_pipeline.cpp](../src/sample_pipeline.cpp)

---

## Pipeline Architecture

### Component Hierarchy

```text
SamplePipeline (IPipeline)
├── ModuleA (SampleModuleA)
│   ├── MLIRTrtEngine (TensorRT execution)
│   ├── CaptureStreamPrePostTrtEngEnqueue (CUDA graph capture)
│   └── TrtEngine (Engine loading)
└── ModuleB (SampleModuleB)
    └── sample_module_b_kernel (CUDA ReLU kernel)
```

### Module Interfaces

Both modules implement:

- **`IModule`**: Core module lifecycle (setup, configure_io, ports)
- **`IAllocationInfoProvider`**: Memory requirements reporting
- **`IStreamExecutor`**: Direct stream-based execution
- **`IGraphNodeProvider`**: CUDA graph node integration

### Pipeline Lifecycle

1. **(One-time) Construction**: Create pipeline and modules with configuration parameters
   - User calls: `SamplePipelineFactory::create_pipeline()`
   - Pipeline internally creates modules via `IModuleFactory`

2. **(One-time) Setup**: Allocate memory, initialize module data structures
   - User calls: `pipeline->setup()`
   - Pipeline internally:
     - Configures zero-copy modes based on execution mode
     - Creates `PipelineMemoryManager` and allocates device memory
     - Calls `module->setup_memory(slice)` on each module (internal implementation detail)
   - See [Memory Allocation Flow](#memory-allocation-flow) for detailed call hierarchy

3. **(Per-iteration) I/O Configuration**: Update connections and prepare for execution
   - User calls: `pipeline->configure_io(params, external_inputs, external_outputs, stream)`
   - Pipeline internally:
     - Routes inputs: `module->set_inputs()` on each module
     - Updates iteration state: `module->configure_io(params)` on each module
     - Copies dynamic descriptors to device
   - See [Data Flow Diagrams](#data-flow-diagrams) for complete sequence

4. **(One-time) Warmup**: Load engines, copy static descriptors, capture CUDA graphs
   - User calls: `pipeline->warmup(stream)` after first `configure_io()`
   - Pipeline internally copies static descriptors and calls `module->warmup(stream)` on each module
   - TensorRT graph capture requires non-default stream (provided by user)

5. **(One-time, Graph mode only) Graph Build**: Build pipeline-level CUDA graph
   - User calls: `pipeline->build_graph()` after `warmup()`
   - Combines module graphs with dependencies into single executable graph

6. **(Per-iteration) Execution**: Launch stream or graph
   - User calls: `pipeline->execute_stream(stream)` OR `pipeline->execute_graph(stream)`
   - Choose based on execution mode configured at construction

7. **(Per-iteration) Iteration**: Update data and repeat steps 3 + 6
   - Call `configure_io()` with new data, then `execute_stream()` or `execute_graph()`
   - In graph mode: skip `warmup()` and `build_graph()` (already done)

---

## Execution Modes

### Overview Execution Modes

The pipeline supports two execution modes that are **statically configured** at pipeline construction time via `PipelineSpec`. The mode cannot change during the pipeline's lifetime and affects memory allocation strategy and zero-copy optimization possibilities.

### ExecutionMode Enum

Defined in [core/types.hpp:151-166](../../../framework/core/lib/include/core/types.hpp#L151-L166):

```cpp
/**
 * @enum ExecutionMode
 * @brief Pipeline execution mode determining addressing and memory allocation strategy
 *
 * The execution mode is a static configuration set at pipeline construction time
 * and cannot change during the pipeline's lifetime.
 */
enum class ExecutionMode : std::uint8_t {
  Stream, //!< Stream mode: flexible addressing, supports dynamic set_tensor_address()
          //!< per iteration, enables zero-copy with dynamic upstream addresses
  Graph   //!< Graph mode: fixed addressing required for CUDA graph capture/replay,
          //!< zero-copy only possible with stable upstream addresses
};
```

### Mode Configuration

**Setting Execution Mode** in `PipelineSpec`:

```cpp
pipeline::PipelineSpec spec;
spec.pipeline_name = "SamplePipeline";
spec.execution_mode = pipeline::ExecutionMode::Graph;  // or ExecutionMode::Stream

// ... configure modules, connections, etc.
```

**Default Behavior**: Defaults to `ExecutionMode::Graph` for backward compatibility if not explicitly set.

### Stream Mode Details

**Characteristics**:

- **Addressing**: Flexible - modules can accept different tensor addresses per iteration
- **TRT Engine**: Uses `set_tensor_address()` to update input/output pointers dynamically
- **Memory Strategy**: Modules can skip input buffer allocation when zero-copy is possible
- **Zero-Copy**: Works with BOTH stable and dynamic upstream addresses
- **Graph Capture**: Not used (direct kernel launches)

**When to Use**:

- Development and debugging (easier to inspect individual operations)
- Pipelines with dynamic topology changes per iteration
- External inputs with changing addresses across ticks
- When maximum flexibility is needed

**Performance**:

- No graph build/instantiation cost
- Slightly higher latency than graph mode (H100 measured 11usec vs graph mode 7usec)

**Example Configuration**:

```cpp
spec.execution_mode = pipeline::ExecutionMode::Stream;

// Stream mode enables zero-copy for external inputs (if addresses change per iteration)
// Pipeline configures: module_a->set_connection_copy_mode("input0", ConnectionCopyMode::ZeroCopy)
```

### Graph Mode Details

**Characteristics**:

- **Addressing**: Fixed - tensor addresses must be stable before `warmup()` for CUDA graph capture
- **TRT Engine**: Tensor addresses captured during graph warmup, reused every iteration
- **Memory Strategy**: Modules MUST allocate input buffers if upstream addresses are dynamic
- **Zero-Copy**: Only possible if upstream provides fixed addresses
- **Graph Capture**: TRT operations captured as CUDA graph subgraph

**When to Use**:

- Production deployments requiring lowest latency
- Pipelines with fixed topology (no per-iteration changes)
- External inputs with fixed addresses
- When maximum performance is critical

**Performance**:

- Single graph launch
- One-time graph build cost during first iteration
- Lowest latency (H100 measured 7usec vs stream mode 11usec)

**Example Configuration**:

```cpp
spec.execution_mode = pipeline::ExecutionMode::Graph;

// Graph mode requires copy for external inputs (addresses not stable before warmup)
// Pipeline configures: module_a->set_connection_copy_mode("input0", ConnectionCopyMode::Copy)
```

### Mode Comparison Table

| Aspect | Stream Mode | Graph Mode |
|--------|-------------|------------|
| **Addressing** | Flexible (can change per iteration) | Fixed (captured at warmup) |
| **Setup** | None (always ready) | One-time `build_graph()` |
| **Launch** | Multiple kernel launches | Single graph launch |
| **Latency** | ~5-10 μs per module | ~1-2 μs total |
| **Flexibility** | Can change execution flow | Fixed execution flow |
| **Zero-Copy with Unstable Upstream** | ✅ Supported | ❌ Not possible (requires copy) |
| **Zero-Copy with Stable Upstream** | ✅ Supported | ✅ Supported |
| **TRT Configuration** | `set_tensor_address()` per iteration | Addresses captured at warmup |
| **Use Case** | Development, dynamic topologies | Production, low-latency |

### Mode Impact on Memory Allocation

**Stream Mode** ([sample_pipeline.cpp:126-137](../src/sample_pipeline.cpp#L126-L137)):

```cpp
if (execution_mode_ == pipeline::ExecutionMode::Stream) {
  // External inputs CAN be zero-copy
  // We'll use set_tensor_address() each iteration (addresses can change)
  module_a_typed->set_connection_copy_mode("input0", pipeline::ConnectionCopyMode::ZeroCopy);
  module_a_typed->set_connection_copy_mode("input1", pipeline::ConnectionCopyMode::ZeroCopy);

  // SampleModuleA.get_requirements() sees zero-copy flags
  // Result: Allocates ONLY 1 tensor (output), skips input allocation
  // Memory: 1 × tensor_bytes = 65,536 bytes (for tensor_size = 16384)
}
```

**Graph Mode** ([sample_pipeline.cpp:138-145](../src/sample_pipeline.cpp#L138-L145)):

> **Note**: This sample app uses `Copy` mode for graph execution. Applications with stable upstream tensor addresses can use `ZeroCopy` mode instead.

```cpp
else {
  // External inputs MUST be copy (need fixed addresses for capture)
  module_a_typed->set_connection_copy_mode("input0", pipeline::ConnectionCopyMode::Copy);
  module_a_typed->set_connection_copy_mode("input1", pipeline::ConnectionCopyMode::Copy);

  // SampleModuleA.get_requirements() sees copy mode flags
  // Result: Allocates 3 tensors (2 inputs + 1 output)
  // Memory: 3 × tensor_bytes = 196,608 bytes (for tensor_size = 16384)
}
```

**Memory Savings**: Stream mode saves ~131 KB (2 input buffers) when zero-copy is possible.

### Static vs Dynamic Mode Switching

**Design Decision**: ExecutionMode is **STATIC** (cannot change during pipeline lifetime)

**Rationale**:

- **Memory Allocation**: Buffer sizes determined at setup based on mode
- **Graph Capture**: Once captured, graph topology is fixed
- **Simplicity**: Avoids complex state management and error-prone mode transitions
- **Performance**: Eliminates runtime mode checks in hot paths

**Future Enhancement**: If dynamic mode switching is needed, would require:

1. Separate memory pools for each mode
2. Mode-specific warmup state tracking
3. Graph invalidation/rebuild on mode change
4. Runtime overhead to check current mode

**Current Status**: Not implemented (static mode only)

### Mode Selection Guidelines

**Choose Stream Mode when**:

- External inputs have changing addresses per iteration
- Pipeline topology changes dynamically
- Debugging or development phase
- Flexibility is more important than performance

**Choose Graph Mode when**:

- External inputs have fixed addresses
- Pipeline topology is fixed
- Production deployment
- Lowest latency is critical
- Can accept memory overhead for input buffer copies

---

## Module A - TensorRT Addition Engine

### TensorRT Engine Details

**Operation**: Element-wise addition
**MLIR Representation**: `result0 = arg0 + arg1`
**Engine Format**: Serialized TensorRT engine file (`.trtengine`)
**Tensor Names**:

- Input 0: `"arg0"`
- Input 1: `"arg1"`
- Output: `"result0"`

**Tensor Configuration**:

```cpp
const pipeline::MLIRTensorParams input0_params{
    .name = "arg0",
    .data_type = pipeline::TensorR32F,    // float32
    .rank = 1,                         // 1D tensor
    .dims = {tensor_size_}             // Configurable size
};
```

### Buffers Allocated (example)

ModuleA allocates **1-3 buffers** in device memory depending on ExecutionMode:

#### Graph Mode (3 buffers)

| Buffer | Type | Size | Purpose | Pointer |
|--------|------|------|---------|---------|
| `d_input0_` | `float*` | `tensor_size_ * sizeof(float)` | Fixed input 0 for TRT | Allocated from memory slice |
| `d_input1_` | `float*` | `tensor_size_ * sizeof(float)` | Fixed input 1 for TRT | Allocated from memory slice |
| `d_output_` | `float*` | `tensor_size_ * sizeof(float)` | TRT output | Allocated from memory slice |

**Memory Layout** (from `ModuleMemorySlice`):

```text
[d_input0_][d_input1_][d_output_]
|<-tensor_bytes->|<-tensor_bytes->|<-tensor_bytes->|
Total: 3 × tensor_bytes = 196,608 bytes (for tensor_size = 16384)
```

**Why 3 Buffers?**

- TensorRT graph capture requires **stable tensor addresses** before `warmup()`
- External input pointers may change between ticks (dynamic)
- Solution: Allocate fixed buffers, copy external inputs in `configure_io()`
- ConnectionCopyMode: **Copy** (external → fixed buffers)

#### Stream Mode (1 buffer)

| Buffer | Type | Size | Purpose | Pointer |
|--------|------|------|---------|---------|
| `d_output_` | `float*` | `tensor_size_ * sizeof(float)` | TRT output | Allocated from memory slice |
| `d_input0_` | `float*` | N/A | Points to external input | No allocation (zero-copy) |
| `d_input1_` | `float*` | N/A | Points to external input | No allocation (zero-copy) |

**Memory Layout** (from `ModuleMemorySlice`):

```text
[d_output_]
|<-tensor_bytes->|
Total: 1 × tensor_bytes = 65,536 bytes (for tensor_size = 16384)
Memory Savings: 131,072 bytes (2 input buffers skipped)
```

**Why Only 1 Buffer?**

- TRT in stream mode uses `set_tensor_address()` per iteration (flexible addressing)
- Can accept external input addresses directly (even if they change per iteration)
- Solution: Store external input pointers, use them directly
- ConnectionCopyMode: **ZeroCopy** (external → TRT uses directly)

**Current Implementation**: Graph mode only (stream mode with zero-copy not yet exercised in tests).

See [sample_module_a.cpp:162-176](../src/sample_module_a.cpp#L162-L176) for allocation code and [sample_module_a.cpp:508-533](../src/sample_module_a.cpp#L508-L533) for conditional allocation logic.

### Data Flow and Warmup

User case **External Inputs → Fixed Buffers**

1. **`set_inputs()`** ([sample_module_a.cpp:270-305](../src/sample_module_a.cpp#L270-L305))
   - Stores external input pointers: `external_input0_data_`, `external_input1_data_`
   - Lightweight operation - just saves pointer references
   - No heavy computation or device operations

2. **`warmup(cudaStream_t stream)`** ([sample_module_a.cpp:299-376](../src/sample_module_a.cpp#L299-L376))
   - **One-time initialization** called after `set_inputs()`
   - Receives CUDA stream from pipeline (cannot use `cudaStreamDefault`)
   - Validates that memory and connections are established
   - Configures TRT engine with fixed buffer addresses:

     ```cpp
     const std::vector<void*> input_buffers = {d_input0_, d_input1_};
     const std::vector<void*> output_buffers = {d_output_};
     trt_engine_->setup(input_buffers, output_buffers);
     ```

   - Calls `trt_engine_->warmup(stream)` to capture CUDA graph on provided stream
   - Synchronizes stream to ensure graph capture is complete
   - Sets `is_warmed_up_ = true`
   - Idempotent: Subsequent calls are no-ops
   - **No stream creation/destruction** - uses pipeline-provided stream

3. **`configure_io()`** ([sample_module_a.cpp:188-233](../src/sample_module_a.cpp#L188-L233))
   - In case of non ZeroCopy - it copies external inputs to fixed buffers:

     ```cpp
     cudaMemcpy(d_input0_, external_input0_data_, tensor_bytes_, cudaMemcpyDeviceToDevice);
     cudaMemcpy(d_input1_, external_input1_data_, tensor_bytes_, cudaMemcpyDeviceToDevice);
     ```

   - Uses **synchronous** `cudaMemcpy` (not stream-aware)
   - Ensures data is ready before execution
   - Called **every iteration** for new data

**Why Copy in `configure_io()`?**

- Works for **both** stream and graph modes
- Stream mode: `configure_io()` → `execute()` → TRT uses copied data
- Graph mode: `configure_io()` → `execute_graph()` → TRT child node uses copied data

**Why Separate `warmup()` from `set_inputs()`?**

- Makes expensive one-time initialization **explicit** in the API
- `set_inputs()` is lightweight - just routing/connectivity
- `warmup()` is expensive - engine loading, graph capture
- Clear separation of concerns: routing vs. initialization

### Engine Setup and Warmup

**TensorRT Stack Creation** ([sample_module_a.cpp:91-103](../src/sample_module_a.cpp#L91-L103)):

```cpp
// 1. Create TensorRT engine with engine file path
auto tensorrt_runtime = std::make_unique<pipeline::TrtEngine>(
    params.trt_engine_path, *trt_logger_);

// 2. Create graph capture helper for CUDA graph integration
auto graph_capturer =
    std::make_unique<pipeline::CaptureStreamPrePostTrtEngEnqueue>();
graph_capturer_ = graph_capturer.get(); // Keep non-owning pointer

// 3. Create MLIR TRT engine with the runtime and capture helper (takes ownership
// of both)
trt_engine_ = std::make_unique<pipeline::MLIRTrtEngine>(
    inputs, outputs, std::move(tensorrt_runtime), std::move(graph_capturer));
```

**Warmup Sequence** ([sample_module_a.cpp:299-376](../src/sample_module_a.cpp#L299-L376)):

Called via the explicit `warmup(cudaStream_t stream)` method after connections are established:

```cpp
void SampleModuleA::warmup(cudaStream_t stream) {
  // Skip if already warmed up (idempotent)
  if (is_warmed_up_) return;

  // Validate preconditions
  // - Memory must be allocated (d_input0_, d_input1_, d_output_ != nullptr)
  // - Connections must be established (external_input0/1_data_ != nullptr)

  // Configure TRT engine with FIXED tensor addresses
  const std::vector<void*> input_buffers = {d_input0_, d_input1_};
  const std::vector<void*> output_buffers = {d_output_};

  // Setup (loads engine to device)
  trt_engine_->setup(input_buffers, output_buffers);

  // Use pipeline-provided stream (cannot use cudaStreamDefault for TRT graph capture)
  // No stream creation needed - pipeline manages stream lifecycle
  trt_engine_->warmup(stream);

  // Synchronize to ensure graph capture is complete
  cudaStreamSynchronize(stream);

  is_warmed_up_ = true;
}
```

**What Happens During Warmup?**

- `CaptureStreamPrePostTrtEngEnqueue` begins stream capture
- TensorRT engine executes on warmup stream
- CUDA graph is captured containing all TRT operations
- Captured graph stored in `graph_capturer_` for later retrieval

### Graph Integration

**Adding TRT Subgraph to Pipeline Graph** ([sample_module_a.cpp:426-463](../src/sample_module_a.cpp#L426-L463)):

```cpp
CUgraphNode SampleModuleA::add_node_to_graph(
    gsl_lite::not_null<pipeline::IGraph*> graph,
    std::span<const CUgraphNode> deps)
{
    // Retrieve captured TensorRT graph from warmup
    cudaGraph_t trt_graph = graph_capturer_->get_graph();

    // Add as child graph node (TRT graph becomes subgraph of pipeline graph)
    trt_node_ = graph->add_child_graph_node(deps, trt_graph);

    return trt_node_;  // Module owns node handle
}
```

**Key Points:**

- TRT graph is added as a **child graph node** (not individual kernel nodes)
- Dependencies: Previous module's node (ensures sequential execution)
- Module stores `trt_node_` handle for parameter updates
- No parameter updates needed (TRT uses fixed addresses)

### Execution Modes Stream vs Graph

**Stream Mode** ([sample_module_a.cpp:406-424](../src/sample_module_a.cpp#L406-L424)):

```cpp
void SampleModuleA::execute(cudaStream_t stream) {
    // Input data already copied in configure_io()
    // Execute TensorRT inference
    const pipeline::NvErrc run_result = trt_engine_->run(stream);
}
```

**Graph Mode**:

- Pipeline graph contains TRT child graph
- `execute_graph()` launches entire pipeline graph
- TRT operations execute as part of graph
- No explicit `execute()` call on module

---

## Module B - CUDA ReLU Kernel

### Custom Kernel Implementation

**Kernel Function** ([sample_module_b_kernel.cu:17-29](../src/sample_module_b_kernel.cu#L17-L29)):

```cpp
__global__ void sample_module_b_kernel(
    const SampleModuleBStaticKernelParams* static_params,
    const SampleModuleBDynamicKernelParams* dynamic_params)
{
    const std::size_t idx =
        (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;

    if (idx < static_params->size) {
        static_params->output[idx] =
            fmaxf(0.0F, dynamic_params->input[idx]);  // ReLU: max(0, x)
    }
}
```

**Kernel Parameters** ([sample_module_b_kernel.cuh:18-35](../src/sample_module_b_kernel.cuh#L18-L35)):

```cpp
// Static parameters (don't change between executions)
struct SampleModuleBStaticKernelParams {
    float* output;      // Output buffer pointer
    std::size_t size;   // Number of elements
};

// Dynamic parameters (change per iteration)
struct SampleModuleBDynamicKernelParams {
    const float* input;  // Input buffer pointer (from upstream module)
};
```

**Kernel Launch Configuration** ([sample_module_b.hpp:169-172](../src/sample_module_b.hpp#L169-L172)):

```cpp
static constexpr unsigned int BLOCK_SIZE = 256;
std::size_t grid_size_ = (tensor_size_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
```

### Buffers Allocated

ModuleB allocates **1 output buffer** in device memory:

| Buffer | Type | Size | Purpose | Pointer |
|--------|------|------|---------|---------|
| `d_output_` | `float*` | `tensor_size_ * sizeof(float)` | ReLU output | Allocated from memory slice |

**Input Buffer**: ModuleB does **not** allocate an input buffer. Instead:

- Receives input pointer from ModuleA's output via `set_inputs()`
- Stores pointer in `d_input_` member variable
- No data copy required (direct pointer passing)

See [sample_module_b.cpp:47-62](../src/sample_module_b.cpp#L47-L62) for allocation code.

### Input Pointer Management

**`set_inputs()`** ([sample_module_b.cpp:173-199](../src/sample_module_b.cpp#L173-L199)):

```cpp
void SampleModuleB::set_inputs(const std::vector<pipeline::PortInfo>& inputs) {
    for (const auto& port : inputs) {
        if (port.name == "input") {
            d_input_ = port.tensors[0].device_ptr;  // Store pointer from ModuleA
        }
    }
}
```

**`configure_io()`** ([sample_module_b.cpp:121-136](../src/sample_module_b.cpp#L121-L136)):

```cpp
void SampleModuleB::configure_io(const pipeline::DynamicParams& params) {
    // Update dynamic kernel parameters with current input pointer
    dynamic_params_cpu_ptr_->input =
        reinterpret_cast<const float*>(d_input_);

    // Note: Pipeline will bulk-copy dynamic descriptors to device
}
```

**Data Flow**: External Input → ModuleA output → ModuleB input (no copy)

### Kernel Descriptors

ModuleB uses the **kernel descriptor indirection pattern** for graph mode support:

**Descriptor Setup** ([sample_module_b.cpp:64-118](../src/sample_module_b.cpp#L64-L118)):

```cpp
// 1. Create kernel descriptor accessor
kernel_desc_mgr_ =
    std::make_unique<pipeline::KernelDescriptorAccessor>(memory_slice);

// 2. Create static params in pinned memory (CPU)
static_params_cpu_ptr_ =
    &kernel_desc_mgr_->create_static_param<SampleModuleBStaticKernelParams>(0);
static_params_cpu_ptr_->output = d_output_;
static_params_cpu_ptr_->size = tensor_size_;

// 3. Create dynamic params in pinned memory (CPU)
dynamic_params_cpu_ptr_ =
    &kernel_desc_mgr_->create_dynamic_param<SampleModuleBDynamicKernelParams>(0);
dynamic_params_cpu_ptr_->input = nullptr;  // Updated in configure_io()

// 4. Get device pointers for kernel parameters
static_params_gpu_ptr_ =
    kernel_desc_mgr_->get_static_device_ptr<SampleModuleBStaticKernelParams>(0);
dynamic_params_gpu_ptr_ =
    kernel_desc_mgr_->get_dynamic_device_ptr<SampleModuleBDynamicKernelParams>(0);

// 5. Configure kernel launch parameters
pipeline::setup_kernel_function(kernel_config_,
    reinterpret_cast<const void*>(sample_module_b_kernel));
pipeline::setup_kernel_dimensions(kernel_config_,
    dim3(static_cast<unsigned int>(grid_size_), 1, 1),
    dim3(BLOCK_SIZE, 1, 1));
pipeline::setup_kernel_arguments(kernel_config_,
    static_params_gpu_ptr_, dynamic_params_gpu_ptr_);
```

**Descriptor Memory Layout**:

```text
CPU (Pinned)                          GPU (Device)
┌─────────────────────┐              ┌─────────────────────┐
│ static_params_cpu   │────────────▶│ static_params_gpu    │
│ - output: d_output_ │   Bulk copy  │ - output: d_output_ │
│ - size: tensor_size │   (once)     │ - size: tensor_size │
└─────────────────────┘              └─────────────────────┘

┌─────────────────────┐              ┌─────────────────────┐
│ dynamic_params_cpu  │────────────▶│ dynamic_params_gpu   │
│ - input: d_input_   │   Bulk copy  │ - input: d_input_   │
│   (updated per iteration)│   (per iteration) │   (updated)         │
└─────────────────────┘              └─────────────────────┘

Kernel receives: (static_params_gpu, dynamic_params_gpu)
```

**Why Indirection Pattern?**

- Graph mode: Kernel parameters are captured at graph creation time
- Cannot change parameter **values** after graph is instantiated
- Solution: Kernel receives **pointers to descriptors**
- Change descriptor **data** (via bulk copy) instead of kernel parameters
- `cuGraphExecKernelNodeSetParams()` forces re-read of descriptor pointers

### Graph Integration (example)

**Adding Kernel Node to Pipeline Graph** ([sample_module_b.cpp:247-270](../src/sample_module_b.cpp#L247-L270)):

```cpp
CUgraphNode SampleModuleB::add_node_to_graph(
    gsl_lite::not_null<pipeline::IGraph*> graph,
    std::span<const CUgraphNode> deps)
{
    // Add kernel node using configured kernel params
    kernel_node_ = graph->add_kernel_node(deps, kernel_config_.get_kernel_params());

    return std::span<const CUgraphNode>(&kernel_node_, 1);  // Return span of single node
}
```

**Updating Graph Node Parameters** ([sample_module_b.cpp:272-295](../src/sample_module_b.cpp#L272-L295)):

```cpp
void SampleModuleB::update_graph_node_params(
    CUgraphExec exec,
    const pipeline::DynamicParams& params)
{
    // Force CUDA to re-read indirection pointers
    // (even though addresses don't change, descriptor DATA changes via bulk copy)
    const auto& params = kernel_config_.get_kernel_params();
    cuGraphExecKernelNodeSetParams(exec, kernel_node_, &params);
}
```

**Key Points:**

- Kernel parameters point to **descriptor addresses** (not tensor addresses)
- Descriptor addresses never change (stable for graph mode)
- Descriptor **data** changes via pipeline's bulk copy operation
- `update_graph_node_params()` forces re-read of descriptors

### Execution Modes (example)

**Stream Mode** ([sample_module_b.cpp:240-245](../src/sample_module_b.cpp#L240-L245)):

```cpp
void SampleModuleB::execute(cudaStream_t stream) {
    launch_relu_kernel(stream);
}

void SampleModuleB::launch_relu_kernel(cudaStream_t stream) {
    const CUresult launch_err = kernel_config_.launch(stream);
}
```

**Graph Mode**:

- Pipeline graph contains kernel node
- Kernel receives descriptor pointers (captured at graph creation)
- Descriptors updated via bulk copy before graph launch
- `update_graph_node_params()` called to force descriptor re-read

---

## Memory Management

### Memory Architecture

The pipeline uses **single contiguous allocation** strategy via `PipelineMemoryManager`:

```text
Device Memory (Single cudaMalloc)
┌──────────────────────────────────────────────────────────────┐
│ ModuleA Slice              │ ModuleB Slice                   │
│ ┌────────┬────────┬─────┐  │ ┌────────┐                      │
│ │ input0 │ input1 │ out │  │ │ output │                      │
│ └────────┴────────┴─────┘  │ └────────┘                      │
└──────────────────────────────────────────────────────────────┘

Pinned Host Memory (cudaMallocHost)
┌──────────────────────────────────────────────────────────────┐
│ ModuleB Kernel Descriptors                                   │
│ ┌─────────────────┬─────────────────┐                        │
│ │ Static Params   │ Dynamic Params  │                        │
│ └─────────────────┴─────────────────┘                        │
└──────────────────────────────────────────────────────────────┘

Device Memory (Kernel Descriptors)
┌──────────────────────────────────────────────────────────────┐
│ ModuleB Kernel Descriptors (GPU copies)                      │
│ ┌─────────────────┬─────────────────┐                        │
│ │ Static Params   │ Dynamic Params  │                        │
│ └─────────────────┴─────────────────┘                        │
└──────────────────────────────────────────────────────────────┘
```

### Memory Requirements

**ModuleA** ([sample_module_a.cpp:508-533](../src/sample_module_a.cpp#L508-L533)):

```cpp
pipeline::ModuleMemoryRequirements SampleModuleA::get_requirements() const {
  pipeline::ModuleMemoryRequirements reqs{};

  // Calculate required tensors based on zero-copy configuration
  std::size_t num_tensors = 1;  // Always need output

  // Count inputs that need allocation (not zero-copy)
  if (!input0_upstream_is_stable_) {
    num_tensors++;  // Need to allocate input0
  }
  if (!input1_upstream_is_stable_) {
    num_tensors++;  // Need to allocate input1
  }

  reqs.device_tensor_bytes = tensor_bytes_ * num_tensors;
  reqs.alignment = MEMORY_ALIGNMENT;

  return reqs;
}

// Graph mode: num_tensors = 3 (2 inputs + 1 output) = 196,608 bytes
// Stream mode: num_tensors = 1 (1 output only)     = 65,536 bytes
```

**ModuleB** ([sample_module_b.cpp:220-236](../src/sample_module_b.cpp#L220-L236)):

```cpp
pipeline::ModuleMemoryRequirements reqs{};
reqs.static_kernel_descriptor_bytes = sizeof(SampleModuleBStaticKernelParams);
reqs.dynamic_kernel_descriptor_bytes = sizeof(SampleModuleBDynamicKernelParams);
reqs.device_tensor_bytes = tensor_bytes_;  // 1 output
reqs.alignment = 256;
```

**Total Pipeline Memory** (for tensor_size = 16384):

**Graph Mode**:

- ModuleA device tensors: `3 × 16384 × 4 = 196,608 bytes` (~192 KB)
- ModuleB device tensors: `1 × 16384 × 4 = 65,536 bytes` (~64 KB)
- ModuleB descriptors (CPU): `16 + 8 = 24 bytes`
- ModuleB descriptors (GPU): `16 + 8 = 24 bytes`
- **Total**: ~262 KB device memory + 24 bytes host memory

**Stream Mode (with zero-copy)**:

- ModuleA device tensors: `1 × 16384 × 4 = 65,536 bytes` (~64 KB)
- ModuleB device tensors: `1 × 16384 × 4 = 65,536 bytes` (~64 KB)
- ModuleB descriptors (CPU): `16 + 8 = 24 bytes`
- ModuleB descriptors (GPU): `16 + 8 = 24 bytes`
- **Total**: ~131 KB device memory + 24 bytes host memory
- **Savings**: ~131 KB (50% reduction in device memory)

### Memory Allocation Flow

**Pipeline Setup** ([sample_pipeline.cpp:111-140](../src/sample_pipeline.cpp#L111-L140)):

**Note**: Graph mode uses `Copy` in this example because upstream tensors are dynamic. If upstream tensors are stable (fixed addresses), `ModuleA` can use `ZeroCopy` even in graph mode.

```cpp
void SamplePipeline::setup() {
    // 1. Configure zero-copy based on execution mode
    // This must happen BEFORE get_requirements() is called
    // Note: set_connection_copy_mode() is part of IModule interface (no cast needed)
    if (execution_mode_ == pipeline::ExecutionMode::Stream) {
        // Stream mode: External inputs CAN be zero-copy
        module_a_->set_connection_copy_mode("input0", pipeline::ConnectionCopyMode::ZeroCopy);
        module_a_->set_connection_copy_mode("input1", pipeline::ConnectionCopyMode::ZeroCopy);
    } else {
        // Graph mode: External inputs MUST be copy (need fixed addresses)
        module_a_->set_connection_copy_mode("input0", pipeline::ConnectionCopyMode::Copy);
        module_a_->set_connection_copy_mode("input1", pipeline::ConnectionCopyMode::Copy);
    }

    // 2. Create memory manager and calculate requirements
    // get_requirements() now sees the connection mode configuration
    memory_mgr_ = pipeline::PipelineMemoryManager::create_for_modules(modules_);

    // 3. Allocate all memory slices (single cudaMalloc)
    memory_mgr_->allocate_all_module_slices(modules_);

    // 4. Distribute slices to modules
    for (auto* module : modules_) {
        const auto slice = memory_mgr_->get_module_slice(module->get_instance_id());
        module->setup_memory(slice);
    }

    // Note: Static descriptors copied in warmup() to ensure proper ordering
}
```

**I/O Configuration** ([sample_pipeline.cpp:151-189](../src/sample_pipeline.cpp#L151-L189)):

```cpp
void SamplePipeline::configure_io(..., cudaStream_t stream) {
    // 1. Route inputs (set input pointers)
    module_a_->set_inputs(external_inputs);
    module_b_->set_inputs(module_a_outputs);

    // 2. Call configure_io on modules (updates dynamic descriptors)
    for (auto* module : modules_) {
        module->configure_io(params);
    }

    // 3. Copy all dynamic descriptors to device (bulk operation)
    memory_mgr_->copy_all_dynamic_descriptors_to_device(stream);

    // 4. Synchronize to ensure descriptors ready before execution
    cudaStreamSynchronize(stream);
}
```

### Benefits of Single Allocation

1. **Reduced Fragmentation**: One large allocation instead of many small ones
2. **Better Locality**: Related tensors are close in memory
3. **Simplified Management**: Single free operation for entire pipeline
4. **Predictable Overhead**: Memory overhead known upfront
5. **Alignment Guarantees**: All slices properly aligned for GPU access

---

## Stream Capture and Graph Execution

### Stream Capture Timing

**When Does Capture Happen?**

- During explicit `warmup(stream)` call on ModuleA (called by pipeline)
- After memory allocation and tensor addresses are known
- After connections are established via `set_inputs()`
- Before any actual pipeline execution

**Capture Sequence** ([sample_module_a.cpp:299-376](../src/sample_module_a.cpp#L299-L376)):

```cpp
void SampleModuleA::warmup(cudaStream_t stream) {
    if (!is_warmed_up_) {
        // 1. Configure TRT with FIXED addresses
        const std::vector<void*> input_buffers = {d_input0_, d_input1_};
        const std::vector<void*> output_buffers = {d_output_};
        trt_engine_->setup(input_buffers, output_buffers);

        // 2. Use pipeline-provided stream (no stream creation)
        // TensorRT requires non-default stream for graph capture
        trt_engine_->warmup(stream);

        // 3. Synchronize to ensure capture complete
        cudaStreamSynchronize(stream);

        is_warmed_up_ = true;
    }
}
```

**What Gets Captured?**

- All TensorRT operations (layers, memory operations)
- CUDA kernels launched by TensorRT
- Memory copies within TRT execution
- Complete execution flow for the inference

### Graph Build Process

**Build Timing**: After first `configure_io()`, before first `execute_graph()`

**Build Flow** ([sample_pipeline.cpp:315-362](../src/sample_pipeline.cpp#L315-L362)):

```cpp
void SamplePipeline::build_graph() {
    // 1. Create graph manager (implicitly creates empty CUDA graph)
    //    Note: create_graph() is now part of GraphManager constructor
    graph_manager_ = std::make_unique<pipeline::GraphManager>();

    // 2. Add modules to graph with dependencies
    std::vector<CUgraphNode> prev_nodes;
    for (auto* module : modules_) {
        auto* graph_provider = module->as_graph_node_provider();

        // Add module's node(s) to graph with dependencies on previous nodes
        const auto nodes =
            graph_manager_->add_kernel_node(graph_provider, prev_nodes);

        // Current node(s) become dependencies for next module
        prev_nodes.assign(nodes.begin(), nodes.end());
    }

    // 3. Instantiate and upload graph (explicit expensive operations)
    graph_manager_->instantiate_graph();  // ~50-100ms one-time cost
    graph_manager_->upload_graph();        // ~10ms one-time cost

    graph_built_ = true;
}
```

**Resulting Graph Structure**:

```text
Pipeline CUDA Graph
┌────────────────────────────────────────┐
│ ┌─────────────────────────────────┐   │
│ │ TRT Child Graph (ModuleA)       │   │
│ │ ┌─────────────────────────────┐ │   │
│ │ │ TRT Layer 1                 │ │   │
│ │ │ TRT Layer 2                 │ │   │
│ │ │ TRT Layer N                 │ │   │
│ │ └─────────────────────────────┘ │   │
│ └─────────────────────────────────┘   │
│           │ (dependency)               │
│           ▼                            │
│ ┌─────────────────────────────────┐   │
│ │ Kernel Node (ModuleB)           │   │
│ │ sample_module_b_kernel          │   │
│ └─────────────────────────────────┘   │
└────────────────────────────────────────┘
```

### Graph Lifecycle Design Rationale

**Question**: Can we remove `instantiate_graph()`, `create_graph()`, etc., and make graph initialization part of the GraphManager constructor?

**Answer**: **Partially**. We can merge `create_graph()` into the constructor, but `instantiate_graph()` and `upload_graph()` **must** remain explicit due to temporal dependencies.

#### Critical Dependencies

The graph initialization cannot be fully consolidated into the constructor because:

1. **`create_graph()` - CAN move to constructor** ✅
   - No external dependencies
   - Just calls `cuGraphCreate(&graph_, 0)`
   - **Recommendation**: Merge into `GraphManager()` constructor

2. **`add_kernel_node()` - CANNOT be called until:** ❌
   - After `warmup()` completes (TRT graph captured in ModuleA)
   - After `configure_io()` establishes connections
   - Modules need captured TRT graphs and configured addresses

3. **`instantiate_graph()` - CANNOT be called until:** ❌
   - After **ALL** kernel nodes added to graph
   - Graph topology must be complete
   - Expensive operation: `cuGraphInstantiate()` optimizes entire graph

4. **`upload_graph()` - CANNOT be called until:** ❌
   - After `instantiate_graph()` creates `CUgraphExec` handle
   - Requires instantiated graph handle

#### Execution Order Dependencies

```text
Constructor Phase:
  GraphManager()
    └─→ cuGraphCreate()  ✅ Can be here (no dependencies)

Warmup Phase (external data arrives):
  configure_io()        // Establishes addresses, connections
  warmup()              // TRT captures child graphs
    └─→ ModuleA.warmup()
        └─→ trt_engine_->warmup()  // Captures TRT graph

Build Phase (can now build full graph):
  build_graph()
    ├─→ add_kernel_node(ModuleA)  ❌ Needs captured TRT graph
    ├─→ add_kernel_node(ModuleB)  ❌ Needs ModuleA to exist
    ├─→ instantiate_graph()       ❌ Needs all nodes added first
    └─→ upload_graph()            ❌ Needs instantiated exec handle

Execute Phase:
  execute_graph()
    ├─→ update_graph_node_params()
    └─→ launch_graph()
```

#### Industry Standards & Best Practices

**C++ Core Guidelines Compliance:**

- **C.41**: "A constructor should create a fully initialized object"
  - ✅ We can create an **empty graph** in constructor
  - ❌ We **cannot** create a **complete graph** until nodes are available
  - **Solution**: Constructor creates "as complete as possible" object (empty graph ready for nodes)

- **C.50**: "Use a factory function if you need 'virtual behavior' during initialization"
  - ❌ Not applicable - no virtual calls needed, problem is **temporal dependency**
  - ❌ Builder pattern doesn't help - data arrives over time, not configuration complexity

**Two-Phase Initialization Pattern:**

Widely accepted in real-time and GPU programming:

- **CUDA Driver API**: `cuGraphCreate()` → `cuGraphAddKernelNode()` → `cuGraphInstantiate()`
- **TensorRT**: `createBuilder()` → `buildEngine()` → `deserialize()`
- **Vulkan/DirectX**: `CreateDevice()` → `Initialize()` → `CreateSwapchain()`

**Rationale**: Initialization requires resources/data not available at construction time.

#### Why NOT Hide Expensive Operations

**Anti-Pattern: Lazy Initialization in Launch** ❌

```cpp
// DON'T DO THIS:
void GraphManager::launch_graph(cudaStream_t stream) const {
  if (!is_finalized_) {
    instantiate_graph_impl();  // HIDDEN EXPENSIVE OPERATION
    upload_graph_impl();        // HIDDEN EXPENSIVE OPERATION
    is_finalized_ = true;
  }
  main_graph_->launch(stream);
}
```

**Problems:**

- ❌ First `launch_graph()` has **dramatically different** latency (~100ms vs 1μs)
- ❌ Violates **principle of least surprise** - users expect launch to be fast
- ❌ Breaks **real-time predictability** - cannot predict when expensive operation occurs
- ❌ Harder to debug and profile - when did instantiation happen?

**Real-time systems require explicit control over expensive operations.**

#### Recommended Design

**Merge `create_graph()` into constructor, keep others explicit:**

```cpp
// GraphManager.hpp
class GraphManager {
public:
  GraphManager();  // Creates empty graph (cheap: ~1μs)
  // REMOVED: void create_graph();

  // Keep explicit (dependencies + expensive operations):
  void instantiate_graph() const;  // Expensive: ~50-100ms
  void upload_graph() const;        // Moderate: ~10ms

  // Node addition (requires module data):
  std::span<const CUgraphNode> add_kernel_node(...);  // Returns span of nodes
};

// GraphManager.cpp
GraphManager::GraphManager() : main_graph_(nullptr) {
  main_graph_ = std::make_unique<Graph>();
  main_graph_->create();  // Moved from create_graph()
}

// Pipeline usage:
void SamplePipeline::build_graph() {
  // Graph created implicitly in constructor
  graph_manager_ = std::make_unique<GraphManager>();

  // Add nodes (requires warmup to complete first)
  for (auto* module : modules_) {
    graph_manager_->add_kernel_node(module, deps);
  }

  // Explicit expensive operations (clear to reader)
  graph_manager_->instantiate_graph();  // User knows: expensive
  graph_manager_->upload_graph();        // User knows: one-time
}
```

**Benefits:**

- ✅ One fewer explicit call (`create_graph()` removed)
- ✅ GraphManager always has valid empty graph after construction (RAII)
- ✅ Clearer semantics: "creating a GraphManager creates a graph"
- ✅ **Explicit about expensive operations** (instantiate, upload remain visible)
- ✅ Matches CUDA patterns (`cuGraphCreate` is cheap, `cuGraphInstantiate` is expensive)
- ✅ Maintains real-time predictability
- ✅ No breaking changes to node addition logic

**What NOT to Do:**

- ❌ Don't move `instantiate_graph()` to constructor (needs nodes added first)
- ❌ Don't move `upload_graph()` to constructor (needs instantiation first)
- ❌ Don't hide expensive operations in getters/launch methods
- ❌ Don't use Builder pattern (wrong abstraction for temporal dependencies)
- ❌ Don't make virtual methods called from constructor (undefined behavior)

#### Summary

The current design with explicit initialization methods is **CORRECT** for this problem domain. The only safe improvement is merging the cheap, dependency-free `create_graph()` into the constructor. All other methods must remain explicit due to:

1. **Temporal dependencies** - data arrives from external sources (modules) after construction
2. **Performance transparency** - expensive operations must be visible to users
3. **Real-time predictability** - users need control over when expensive operations occur
4. **Industry standards** - matches patterns in CUDA, Vulkan, DirectX, TensorRT

This design prioritizes **correctness**, **predictability**, and **transparency** over API brevity.

### Parameter Updates for Graph Mode

**Problem**: CUDA graphs capture parameter **values** at creation time

**Solution**: Indirection pattern + `cuGraphExecKernelNodeSetParams()`

**Update Flow** ([sample_pipeline.cpp:202-229](../src/sample_pipeline.cpp#L202-L229)):

```cpp
void SamplePipeline::execute_graph(cudaStream_t stream) {
    // 1. Update graph node parameters before execution
    auto* const exec = graph_manager_->get_exec();
    const pipeline::DynamicParams dummy_tick{};

    for (auto* module : modules_) {
        auto* graph_node_provider = module->as_graph_node_provider();
        if (graph_node_provider != nullptr) {
            graph_node_provider->update_graph_node_params(exec, dummy_tick);
        }
    }

    // 2. Launch graph
    graph_manager_->launch_graph(stream);
}
```

**Module Updates**:

- **ModuleA**: No-op (TRT uses fixed addresses, no parameters to update)
- **ModuleB**: Calls `cuGraphExecKernelNodeSetParams()` to force descriptor re-read

**Why Update Every Iteration?**

- Even though descriptor **addresses** don't change
- Descriptor **data** changes via bulk copy in `configure_io()`
- CUDA needs to re-read the indirection pointers to see updated data

### Execution Mode Comparison

| Aspect | Stream Mode | Graph Mode |
|--------|-------------|------------|
| **Setup** | None (always ready) | One-time `build_graph()` |
| **Launch** | Multiple kernel launches | Single graph launch |
| **Latency** | ~5-10 μs per module | ~1-2 μs total |
| **Flexibility** | Can change execution flow | Fixed execution flow |
| **Parameter Updates** | Direct (via `execute()`) | Indirect (via `update_graph_node_params()`) |
| **Memory Allocation** | Conditional (based on zero-copy) | Always allocates input buffers |
| **Zero-Copy Support** | Both stable and dynamic upstream | Only stable upstream |
| **Use Case** | Development, debugging, dynamic inputs | Production, low-latency, fixed inputs |

---

## Factory Pattern Integration

### Overview of Factories

The Sample Pipeline implements the **factory pattern** for both module and pipeline creation, enabling:

- **Configuration-driven instantiation**: Create pipelines from `PipelineSpec` structures
- **Decoupling**: Separate pipeline construction from usage
- **Testability**: Standardized creation interface for testing
- **Extensibility**: Easy addition of new module types

### Factory Classes

#### Module Factories

**SampleModuleAFactory** ([sample_module_factories.hpp:28](../src/sample_module_factories.hpp#L28) / [sample_module_factories.cpp:24](../src/sample_module_factories.cpp#L24)):

```cpp
class SampleModuleAFactory final : public pipeline::IModuleFactory {
public:
  std::unique_ptr<pipeline::IModule> create_module(
      const std::string& module_type,
      const std::string& instance_id,
      const std::any& static_params) override;

  bool supports_module_type(const std::string& module_type) const override;
};
```

- **Supported Type**: `"sample_module_a"`
- **Parameters**: `SampleModuleA::StaticParams` (tensor_size, trt_engine_path)
- **Returns**: Configured `SampleModuleA` instance

**SampleModuleBFactory** ([sample_module_factories.hpp:73](../src/sample_module_factories.hpp#L73) / [sample_module_factories.cpp:51](../src/sample_module_factories.cpp#L51)):

```cpp
class SampleModuleBFactory final : public pipeline::IModuleFactory {
public:
  std::unique_ptr<pipeline::IModule> create_module(
      const std::string& module_type,
      const std::string& instance_id,
      const std::any& static_params) override;

  bool supports_module_type(const std::string& module_type) const override;
};
```

- **Supported Type**: `"sample_module_b"`
- **Parameters**: `SampleModuleB::StaticParams` (tensor_size)
- **Returns**: Configured `SampleModuleB` instance

**SampleModuleFactory** ([sample_module_factories.hpp:118](../src/sample_module_factories.hpp#L118) / [sample_module_factories.cpp:78](../src/sample_module_factories.cpp#L78)):

```cpp
class SampleModuleFactory final : public pipeline::IModuleFactory {
  // Aggregates both SampleModuleAFactory and SampleModuleBFactory
  // Delegates to appropriate sub-factory based on module_type
};
```

- **Supported Types**: `"sample_module_a"`, `"sample_module_b"`
- **Design**: Composite pattern - routes to appropriate sub-factory

#### Pipeline Factory

**SamplePipelineFactory** ([sample_pipeline_factory.hpp:40](../src/sample_pipeline_factory.hpp#L40) / [sample_pipeline_factory.cpp:28](../src/sample_pipeline_factory.cpp#L28)):

```cpp
class SamplePipelineFactory final : public pipeline::IPipelineFactory {
public:
  explicit SamplePipelineFactory(
      gsl_lite::not_null<pipeline::IModuleFactory*> module_factory);

  std::unique_ptr<pipeline::IPipeline> create_pipeline(
      const std::string& pipeline_type,
      const std::string& pipeline_id,
      const pipeline::PipelineSpec& spec) override;

  bool is_pipeline_type_supported(const std::string& pipeline_type) const override;
  std::vector<std::string> get_supported_pipeline_types() const override;
};
```

- **Supported Type**: `"sample"`
- **Dependencies**: Requires `IModuleFactory*` (dependency injection)
- **Configuration**: Parses `PipelineSpec` to extract module parameters
- **Returns**: Configured `SamplePipeline` instance

### PipelineSpec Structure

Example configuration for SamplePipeline:

```cpp
pipeline::PipelineSpec spec;
spec.pipeline_name = "SamplePipeline";

// Module A configuration
const SampleModuleA::StaticParams module_a_params{
    .tensor_size = 16384,
    .trt_engine_path = "path/to/engine.trtengine"
};

const pipeline::ModuleCreationInfo module_a_info{
    .module_type = "sample_module_a",
    .instance_id = "module_a",
    .init_params = std::any(module_a_params)
};

spec.modules.emplace_back(module_a_info);

// Module B configuration
const SampleModuleB::StaticParams module_b_params{
    .tensor_size = 16384
};

const pipeline::ModuleCreationInfo module_b_info{
    .module_type = "sample_module_b",
    .instance_id = "module_b",
    .init_params = std::any(module_b_params)
};

spec.modules.emplace_back(module_b_info);

// Connection configuration
const pipeline::PortConnection connection{
    .source_module = "module_a",
    .source_port = "output",
    .target_module = "module_b",
    .target_port = "input"
};

spec.connections.push_back(connection);

// External I/O
spec.external_inputs = {"input0", "input1"};
spec.external_outputs = {"output"};

// Execution mode (defaults to Graph if not set)
spec.execution_mode = pipeline::ExecutionMode::Graph;  // or ExecutionMode::Stream
```

### Factory Usage Example

The **main test suite** ([sample_pipeline_test.cpp](../tests/sample_pipeline_test.cpp)) demonstrates factory-based pipeline creation as the primary pattern. All three tests (`StreamExecution_ValidatesCorrectOutput`, `GraphExecution_ValidatesCorrectOutput`, and `GraphExecution_MultipleIterations`) use the factory pattern.

**Creating a Pipeline via Factories** ([sample_pipeline_test.cpp:111-120](../tests/sample_pipeline_test.cpp#L111-L120)):

```cpp
// 1. Create module factory
auto module_factory = std::make_unique<SampleModuleFactory>();

// 2. Create pipeline factory (with module factory dependency)
auto pipeline_factory = std::make_unique<SamplePipelineFactory>(
    gsl_lite::not_null<pipeline::IModuleFactory*>(module_factory.get()));

// 3. Create PipelineSpec
const auto spec = create_sample_pipeline_spec(tensor_size, engine_path);

// 4. Create pipeline from spec
auto pipeline = pipeline_factory->create_pipeline(
    "sample",          // pipeline type
    "my_pipeline",       // pipeline instance ID
    spec                 // configuration
);

// 5. Use pipeline normally
pipeline->setup();
pipeline->configure_io(params, external_inputs, external_outputs, stream);
pipeline->warmup(stream);
pipeline->execute_stream(stream);
```

### Pipeline Constructor Pattern

The `SamplePipeline` follows the reference architecture where **the pipeline receives `IModuleFactory*` and uses it to create modules**:

**Constructor Signature** ([sample_pipeline.hpp:65-67](../src/sample_pipeline.hpp#L65-L67)):

```cpp
SamplePipeline(std::string pipeline_id,
                 gsl_lite::not_null<pipeline::IModuleFactory*> module_factory,
                 const pipeline::PipelineSpec& spec);
```

**Module Creation Flow** ([sample_pipeline.cpp:62-103](../src/sample_pipeline.cpp#L62-L103)):

1. `SamplePipelineFactory` passes `module_factory_` to `SamplePipeline` constructor
2. `SamplePipeline` constructor calls `create_modules_from_spec()`
3. For each module in spec, pipeline calls `module_factory->create_module()`
4. Modules are created via factory, **not direct instantiation**

**Key Implementation Details**:

- Pipeline stores `IModuleFactory*` as a member (non-owning pointer)
- Module creation delegated to factory via `create_module()` calls
- Pipeline validates spec and extracts module configurations
- Type-erased parameters (`std::any`) passed to factory
- Factory returns `std::unique_ptr<IModule>` which pipeline stores

This ensures the factory pattern is fully implemented end-to-end, matching the `MultiModulePipeline` reference architecture.

### Factory Architecture Diagram

```text
┌──────────────────────────────────────────────────────────────┐
│                      User Code                               │
│  ┌───────────────────────────────────────────────────────┐   │
│  │ PipelineSpec (configuration)                          │   │
│  │  - modules[]                                          │   │
│  │  - connections[]                                      │   │
│  │  - external_inputs[]                                  │   │
│  │  - external_outputs[]                                 │   │
│  └────────────────────────┬──────────────────────────────┘   │
└─────────────────────────────┼────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│        SamplePipelineFactory (has IModuleFactory*)        │
│                                                             │
│  create_pipeline(type, id, spec)                           │
│    │                                                        │
│    ├─ Passes module_factory_ to SamplePipeline           │
│    └─ Passes PipelineSpec to SamplePipeline              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│        SamplePipeline (receives IModuleFactory*)          │
│                                                             │
│  Constructor:                                               │
│    ├─ Calls module_factory->create_module("sample_module_a")│
│    └─ Calls module_factory->create_module("sample_module_b")│
│                                                             │
│  ┌──────────────────────┐  ┌──────────────────────┐        │
│  │   SampleModuleA      │  │   SampleModuleB      │        │
│  │  (TensorRT Engine)   │  │   (CUDA ReLU Kernel) │        │
│  │  Created via factory │  │  Created via factory │        │
│  └──────────────────────┘  └──────────────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Type Erasure**: `std::any` enables type-safe parameter passing without template proliferation
2. **Dependency Injection**: Pipeline factory receives module factory as constructor parameter
3. **Interface Segregation**: Separate factories for modules and pipelines
4. **Error Handling**: Factories throw `std::invalid_argument` for unsupported types, `std::bad_any_cast` for parameter mismatches
5. **Backward Compatibility**: Direct instantiation (`new SamplePipeline()`) remains supported

### Testing

Factory-based tests are located in [sample_pipeline_factory_test.cpp](../tests/sample_pipeline_factory_test.cpp):

- `FactoryStreamExecution_ValidatesCorrectOutput`: Factory-created pipeline in stream mode
- `FactoryGraphExecution_ValidatesCorrectOutput`: Factory-created pipeline in graph mode
- `FactoryGraphExecution_MultipleIterations`: Factory-created pipeline with graph reuse

All factory tests verify that factory-created pipelines function identically to directly-instantiated pipelines.

---

## Testing Strategy

### Test Files

- [sample_pipeline_test.cpp](../tests/sample_pipeline_test.cpp) - **Main proof-of-concept tests** using factory pattern for pipeline creation
- [sample_pipeline_factory_test.cpp](../tests/sample_pipeline_factory_test.cpp) - Additional factory-specific tests
- [sample_module_b_standalone_test.cpp](../tests/sample_module_b_standalone_test.cpp) - Isolated ModuleB tests (no TensorRT dependency)

### Test Coverage

#### 1. Stream Execution Test

**File**: [sample_pipeline_test.cpp:259-343](../tests/sample_pipeline_test.cpp#L259-L343)

**Test**: `StreamExecution_ValidatesCorrectOutput`

**Flow**:

```cpp
// 1. Create pipeline via factory
auto module_factory = std::make_unique<SampleModuleFactory>();
auto pipeline_factory = std::make_unique<SamplePipelineFactory>(
    gsl_lite::not_null<pipeline::IModuleFactory*>(module_factory.get()));
const auto spec = create_pipeline_spec(tensor_size, engine_path);
auto pipeline = pipeline_factory->create_pipeline("sample", "test_pipeline", spec);

// 2. Setup pipeline
pipeline->setup();

// 3. Prepare input data
std::vector<float> h_input0(16384);  // [1, 2, 3, ...]
std::vector<float> h_input1(16384);  // [2, 3, 4, ...]

// 4. Compute expected output: ReLU(input0 + input1)
std::vector<float> expected = compute_expected_output(h_input0, h_input1);
// expected[i] = max(0.0f, h_input0[i] + h_input1[i])

// 5. Allocate device memory and copy inputs to device
void* d_input0 = allocate_device_memory(tensor_size * sizeof(float));
void* d_input1 = allocate_device_memory(tensor_size * sizeof(float));
copy_to_device(d_input0, h_input0);
copy_to_device(d_input1, h_input1);

// 6. Create external input/output port infos
std::vector<pipeline::PortInfo> external_inputs;
external_inputs.push_back(create_port_info("input0", d_input0, tensor_size));
external_inputs.push_back(create_port_info("input1", d_input1, tensor_size));
std::vector<pipeline::PortInfo> external_outputs(1);

// 7. Configure I/O with external inputs/outputs
pipeline->configure_io(params, external_inputs, external_outputs, stream);

// 8. Warmup (one-time initialization)
pipeline->warmup(stream);

// 9. Execute in stream mode
pipeline->execute_stream(stream);

// 10. Synchronize and copy output back
cudaStreamSynchronize(stream);
void* d_output = external_outputs[0].tensors[0].device_ptr;
copy_from_device(h_output, d_output, 16384);

// 11. Validate output
EXPECT_TRUE(validate_output(h_output, expected));
```

**Validates**:

- Pipeline construction and setup
- TensorRT engine loading and execution
- CUDA kernel execution
- Data routing between modules
- Output correctness (floating-point comparison with tolerance)

#### 2. Graph Execution Test

**File**: [sample_pipeline_test.cpp:353-437](../tests/sample_pipeline_test.cpp#L353-L437)

**Test**: `GraphExecution_ValidatesCorrectOutput`

**Flow**:

```cpp
// 1-8. Same as stream mode
// (create pipeline, setup, prepare data, compute expected, allocate/copy device memory,
//  create port infos, configure_io, warmup)

// 9. Build graph (retrieves captured TRT graph from warmup)
auto* sample_pipeline = dynamic_cast<SamplePipeline*>(pipeline.get());
sample_pipeline->build_graph();

// 10. Execute graph mode
pipeline.execute_graph(stream);

// 11. Synchronize and copy output back
cudaStreamSynchronize(stream);
void* d_output = external_outputs[0].tensors[0].device_ptr;
copy_from_device(h_output, d_output, tensor_size);

// 12. Validate output (same as stream mode)
EXPECT_TRUE(validate_output(h_output, expected));
```

**Validates**:

- CUDA graph construction
- TRT graph capture and retrieval
- Graph node dependencies
- Graph execution correctness
- Output matches stream mode

#### 3. Multi-Iteration Graph Test

**File**: [sample_pipeline_test.cpp:447-579](../tests/sample_pipeline_test.cpp#L447-L579)

**Test**: `GraphExecution_MultipleIterations`

**Flow**:

```cpp
// Iteration 1: Build graph
pipeline.configure_io(params_1, external_inputs, external_outputs, stream);
pipeline.warmup(stream);
pipeline.build_graph();
pipeline.execute_graph(stream);
EXPECT_TRUE(validate_output(h_output_1, expected_1));

// Iteration 2: Reuse graph with different data
std::vector<float> h_input0_iter2(16384);  // Different pattern
std::vector<float> h_input1_iter2(16384);  // Mix of negative/positive

// Some negative results to test ReLU clipping
for (std::size_t i = 0; i < 256; ++i) {
    h_input0_iter2[i] = -10.0f;  // Large negative
    h_input1_iter2[i] = 5.0f;    // Smaller positive
    // Result: -5.0 → ReLU → 0.0 (clipped)
}

pipeline.configure_io(params_2, external_inputs, external_outputs, stream);
pipeline.execute_graph(stream);  // No warmup/build_graph - reuses existing
EXPECT_TRUE(validate_output(h_output_2, expected_2));
```

**Validates**:

- Graph reusability across iterations
- Dynamic parameter updates
- ReLU clipping behavior (negative values → 0)
- No warmup/build overhead on subsequent iterations

#### 4. Standalone ModuleB Test

**File**: [sample_module_b_standalone_test.cpp:137-197](../tests/sample_module_b_standalone_test.cpp#L137-L197)

**Test**: `StreamExecution_ValidatesCorrectOutput` (standalone)

**Flow**:

```cpp
// Create ModuleB in isolation (no ModuleA, no TRT)
auto module_b = std::make_unique<SampleModuleB>("module_b", params);

// Setup memory
memory_mgr->allocate_all_module_slices({module_b.get()});
module_b->setup_memory(slice);

// Create stream for operations
cudaStream_t stream;
cudaStreamCreate(&stream);

// Warmup and copy static descriptors
module_b->warmup(stream);
memory_mgr->copy_all_static_descriptors_to_device(stream);

// Set input (external input, not from ModuleA)
module_b->set_inputs(input_port);

// Configure I/O and execute
module_b->configure_io(params);
memory_mgr->copy_all_dynamic_descriptors_to_device(stream);
module_b->as_stream_executor()->execute(stream);
cudaStreamDestroy(stream);

// Validate output
EXPECT_TRUE(validate_output(h_output, expected));
```

**Validates**:

- ModuleB works independently (no TRT dependency)
- Kernel descriptor management
- Static and dynamic descriptor copies
- Stream mode execution

#### 5. Standalone ModuleB Graph Test

**File**: [sample_module_b_standalone_test.cpp:202-293](../tests/sample_module_b_standalone_test.cpp#L202-L293)

**Test**: `GraphExecution_ValidatesCorrectOutput` (standalone)

**Validates**:

- ModuleB graph node creation
- Kernel parameter indirection
- `update_graph_node_params()` mechanism
- Graph execution without TRT

### Expected Output Computation

**Helper Function** ([sample_pipeline_test.cpp:235-243](../tests/sample_pipeline_test.cpp#L235-L243)):

```cpp
std::vector<float> compute_expected_output(
    const std::vector<float>& input0,
    const std::vector<float>& input1)
{
    std::vector<float> expected(input0.size());
    for (std::size_t i = 0; i < input0.size(); ++i) {
        const float add_result = input0[i] + input1[i];
        expected[i] = std::max(0.0F, add_result);  // ReLU
    }
    return expected;
}
```

**Validation** ([sample_pipeline_test.cpp:194-226](../tests/sample_pipeline_test.cpp#L194-L226)):

```cpp
bool validate_output(const std::vector<float>& actual,
                     const std::vector<float>& expected,
                     float tolerance = 1e-5F)
{
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const float diff = std::abs(actual[i] - expected[i]);
        if (diff > tolerance) {
            // Log mismatch
            return false;
        }
    }
    return true;
}
```

### Test Data Patterns

**Positive Values** (basic test):

- Input 0: `[1, 2, 3, ..., 16384]`
- Input 1: `[2, 3, 4, ..., 16385]`
- Expected: `[3, 5, 7, ..., 32769]` (all positive, ReLU no-op)

**Mixed Values** (ReLU clipping test):

- Input 0: `[-10, -10, ..., -10, ...]` (first 256 elements)
- Input 1: `[5, 5, ..., 5, ...]` (first 256 elements)
- Expected: `[0, 0, ..., 0, ...]` (clipped to 0 by ReLU)

**Large Range** (numerical stability):

- Input 0: `[-8192, -8191, ..., 8191]`
- Input 1: `[-8192, -8191, ..., 8191]`
- Expected: ReLU applied to sums (tests negative, zero, positive regions)

### Running Tests

**Build and run all tests**:

```bash
# Build tests
cmake --build out/build/clang-debug --target sample_pipeline_tests

# Run tests
./out/build/clang-debug/framework/pipeline/samples/tests/sample_pipeline_tests --gtest_output=detailed

# Run with compute-sanitizer (memory checking)
compute-sanitizer ./out/build/clang-debug/framework/pipeline/samples/tests/sample_pipeline_tests
```

**Test Output**:

```text
[==========] Running 4 tests from 2 test suites.
[----------] 3 tests from SamplePipelineTest
[ RUN      ] SamplePipelineTest.StreamExecution_ValidatesCorrectOutput
[       OK ] SamplePipelineTest.StreamExecution_ValidatesCorrectOutput (45 ms)
[ RUN      ] SamplePipelineTest.GraphExecution_ValidatesCorrectOutput
[       OK ] SamplePipelineTest.GraphExecution_ValidatesCorrectOutput (52 ms)
[ RUN      ] SamplePipelineTest.GraphExecution_MultipleIterations
[       OK ] SamplePipelineTest.GraphExecution_MultipleIterations (63 ms)
[----------] 2 tests from SampleModuleBStandaloneTest
[ RUN      ] SampleModuleBStandaloneTest.StreamExecution_ValidatesCorrectOutput
[       OK ] SampleModuleBStandaloneTest.StreamExecution_ValidatesCorrectOutput (12 ms)
[ RUN      ] SampleModuleBStandaloneTest.GraphExecution_ValidatesCorrectOutput
[       OK ] SampleModuleBStandaloneTest.GraphExecution_ValidatesCorrectOutput (18 ms)
[==========] 5 tests from 2 test suites ran. (190 ms total)
[  PASSED  ] 5 tests.
```

---

## Data Flow Diagrams

### Setup Phase Flow

```text
┌─────────────────────────────────────────────────────────────────┐
│ 1. Pipeline Construction                                        │
│    SamplePipeline constructor                                   │
│    ├─→ Create ModuleA (load TRT engine to memory)               │
│    ├─→ Create ModuleB                                           │
│    └─→ Setup ModuleRouter (define connections)                  │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Pipeline Setup                                               │
│    pipeline.setup()                                             │
│    ├─→ Create PipelineMemoryManager                             │
│    ├─→ Collect memory requirements from modules                 │
│    ├─→ Single cudaMalloc (all tensors + descriptors)            │
│    ├─→ Distribute memory slices to modules                      │
│    │   ├─→ ModuleA.setup_memory()                               │
│    │   │   ├─→ Assign d_input0_, d_input1_, d_output_           │
│    │   │   └─→ (TRT engine NOT loaded to GPU yet)               │
│    │   └─→ ModuleB.setup_memory()                               │
│    │       ├─→ Assign d_output_                                 │
│    │       ├─→ Create kernel descriptors (CPU + GPU)            │
│    │       └─→ Configure kernel launch parameters               │
│    └─→ (Static descriptors copied later in warmup)              │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. First I/O Configuration (Establish Connections)                     │
│    pipeline.configure_io(..., stream)                             │
│    ├─→ Step 1: ModuleA.set_inputs(external_inputs)              │
│    │   └─→ Store external_input0_data_, external_input1_data_   │
│    │       (lightweight - just pointer storage)                 │
│    ├─→ Step 2: ModuleA.configure_io(params)                    │
│    │   └─→ cudaMemcpy external inputs → fixed buffers           │
│    │       ModuleA can now determine its outputs                │
│    ├─→ Step 3: get_outputs(ModuleA)                             │
│    │   └─→ Retrieve outputs AFTER configure_io                    │
│    │       (may be dynamic based on what configure_io computed)   │
│    ├─→ Step 4: ModuleB.set_inputs(moduleA_outputs)              │
│    │   └─→ Store d_input_ (pointer to ModuleA's output)         │
│    ├─→ Step 5: ModuleB.configure_io(params)                    │
│    │   └─→ Update dynamic_params_cpu->input = d_input_          │
│    └─→ Copy dynamic descriptors to device (bulk) on stream      │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. Warmup (One-Time Initialization)                             │
│    pipeline.warmup(stream)  // Pipeline creates & passes stream │
│    ├─→ Copy static descriptors to device (bulk)                 │
│    ├─→ ModuleA.warmup(stream)                                   │
│    │   ├─→ Validate memory allocated and connections set        │
│    │   ├─→ trt_engine_->setup(fixed_buffers)                    │
│    │   │   └─→ Load TRT engine to GPU                           │
│    │   ├─→ trt_engine_->warmup(stream)  // Use passed stream    │
│    │   │   └─→ CaptureStreamPrePostTrtEngEnqueue captures       │
│    │   │       CUDA graph of TRT operations                     │
│    │   ├─→ cudaStreamSynchronize(stream)                        │
│    │   └─→ is_warmed_up_ = true                                 │
│    └─→ ModuleB.warmup(stream)                                   │
│        └─→ No-op (simple CUDA kernels don't need warmup)        │
│    // Note: No stream creation/destruction in modules           │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. Graph Build (Graph Mode Only)                                │
│    pipeline.build_graph()                                       │
│    ├─→ Create GraphManager (implicitly creates empty graph)     │
│    ├─→ Retrieve captured TRT graph from ModuleA                 │
│    ├─→ Add ModuleB kernel to pipeline graph                     │
│    ├─→ Set dependencies (ModuleB depends on ModuleA)            │
│    ├─→ Instantiate complete pipeline graph (expensive: ~50ms)   │
│    └─→ Upload graph to device (moderate: ~10ms)                 │
│    // Note: create_graph() is now implicit in constructor       │
└─────────────────────────────────────────────────────────────────┘
```

### Stream Mode Execution Flow

```text
┌─────────────────────────────────────────────────────────────────┐
│ Per Tick (Repeated)                                             │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. I/O Configuration                                                   │
│    pipeline.configure_io(params, external_inputs, ..., stream) │
│    ├─→ ModuleA.set_inputs(external_inputs)                      │
│    ├─→ ModuleA.configure_io(params)                            │
│    │   └─→ Copy external_input0/1 → d_input0_/d_input1_         │
│    ├─→ outputs = ModuleA.get_outputs()                          │
│    │   (may be dynamic - determined by configure_io)              │
│    ├─→ ModuleB.set_inputs(outputs)                              │
│    ├─→ ModuleB.configure_io(params)                            │
│    │   └─→ Update dynamic_params_cpu->input                     │
│    └─→ Bulk copy dynamic descriptors to device on stream        │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Stream Execution                                             │
│    pipeline.execute_stream(stream)                              │
│    ├─→ ModuleA.execute(stream)                                  │
│    │   └─→ trt_engine_->run(stream)                             │
│    │       ├─→ Reads: d_input0_, d_input1_                      │
│    │       └─→ Writes: d_output_ (ModuleA)                      │
│    └─→ ModuleB.execute(stream)                                  │
│        └─→ launch_relu_kernel(stream)                           │
│            ├─→ Reads: d_input_ (= ModuleA's d_output_)          │
│            └─→ Writes: d_output_ (ModuleB, final output)        │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Synchronize                                                  │
│    cudaStreamSynchronize(stream)                                │
│    └─→ Wait for all operations to complete                      │
└─────────────────────────────────────────────────────────────────┘
```

### Graph Mode Execution Flow

```text
┌─────────────────────────────────────────────────────────────────┐
│ One-Time Graph Build (After First Tick)                         │
│    pipeline.build_graph()                                       │
│    ├─→ Create GraphManager                                      │
│    ├─→ ModuleA.add_node_to_graph(deps=[])                       │
│    │   ├─→ Retrieve: trt_graph = graph_capturer_->get_graph()   │
│    │   │   (graph captured earlier during warmup)               │
│    │   └─→ Add as child graph node → returns trt_node_          │
│    ├─→ ModuleB.add_node_to_graph(deps=[trt_node_])              │
│    │   └─→ Add kernel node → returns kernel_node_               │
│    ├─→ Instantiate graph                                        │
│    └─→ Upload graph to GPU                                      │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Per Tick (Repeated)                                             │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. I/O Configuration -                                                 │
│    pipeline.configure_io(params, external_inputs, ..., stream) │
│    ├─→ ModuleA: set_inputs → configure_io → get_outputs           │
│    ├─→ ModuleB: set_inputs → configure_io                         │
│    └─→ Bulk copy dynamic descriptors on stream                  │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Graph Execution                                              │
│    pipeline.execute_graph(stream)                               │
│    ├─→ Update graph node parameters                             │
│    │   ├─→ ModuleA.update_graph_node_params() [no-op]           │
│    │   └─→ ModuleB.update_graph_node_params()                   │
│    │       └─→ cuGraphExecKernelNodeSetParams()                 │
│    │           (forces descriptor re-read)                      │
│    └─→ graph_manager_->launch_graph(stream)                     │
│        └─→ Single cuGraphLaunch()                               │
│            ├─→ TRT child graph executes                         │
│            │   ├─→ Reads: d_input0_, d_input1_                  │
│            │   └─→ Writes: d_output_ (ModuleA)                  │
│            └─→ ReLU kernel executes (dependency)                │
│                ├─→ Reads: d_input_ (= ModuleA's d_output_)      │
│                └─→ Writes: d_output_ (ModuleB)                  │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Synchronize                                                  │
│    cudaStreamSynchronize(stream)                                │
│    └─→ Wait for graph execution to complete                     │
└─────────────────────────────────────────────────────────────────┘
```

### Memory Layout Visualization

```text
Pipeline Memory Layout (tensor_size = 16384)

╔═════════════════════════════════════════════════════════════════╗
║ Device Memory (Single cudaMalloc allocation)                    ║
╠═════════════════════════════════════════════════════════════════╣
║                                                                 ║
║  ModuleA Memory Slice (196,608 bytes)                           ║
║  ┌─────────────────┬─────────────────┬─────────────────┐        ║
║  │   d_input0_     │   d_input1_     │   d_output_     │        ║
║  │   (65,536 B)    │   (65,536 B)    │   (65,536 B)    │        ║
║  │   Float[16384]  │   Float[16384]  │   Float[16384]  │        ║
║  └─────────────────┴─────────────────┴─────────────────┘        ║
║        ▲                  ▲                  │                  ║
║        │                  │                  │                  ║
║    External Input 0   External Input 1       │                  ║
║    copied here        copied here            │                  ║
║    (configure_io)       (configure_io)           │                  ║
║                                              │                  ║
║                                              ▼                  ║
║  ModuleB Memory Slice (65,536 bytes)                            ║
║  ┌─────────────────┐                                            ║
║  │   d_output_     │  ◀────── Final pipeline output            ║
║  │   (65,536 B)    │                                            ║
║  │   Float[16384]  │                                            ║
║  └─────────────────┘                                            ║
║        ▲                                                        ║
║        │                                                        ║
║     d_input_ (pointer only, no allocation)                      ║
║     = ModuleA's d_output_                                       ║
║                                                                 ║
╚═════════════════════════════════════════════════════════════════╝

╔═════════════════════════════════════════════════════════════════╗
║ ModuleB Kernel Descriptor Memory                                ║
╠═════════════════════════════════════════════════════════════════╣
║                                                                 ║
║  CPU (Pinned Host Memory)                                       ║
║  ┌──────────────────────────────┐                               ║
║  │ static_params_cpu            │                               ║
║  │ ├─ output: &d_output_        │ ──┐                           ║
║  │ └─ size: 16384               │   │                           ║
║  └──────────────────────────────┘   │  Bulk copy                ║
║  ┌──────────────────────────────┐   │  (setup)                  ║
║  │ dynamic_params_cpu           │   │                           ║
║  │ └─ input: &d_input_          │ ──┼──┐                        ║
║  │    (updated per iteration)        │   │  │  Bulk copy             ║
║  └──────────────────────────────┘   │  │  (configure_io)          ║
║                                     │  │                        ║
║  GPU (Device Memory)                │  │                        ║
║  ┌──────────────────────────────    │  │                        ║
║  │ static_params_gpu            │ ◀─┘  │                       ║
║  │ ├─ output: &d_output_        │      │                        ║
║  │ └─ size: 16384               │      │                        ║
║  └──────────────────────────────┘      │                        ║
║  ┌──────────────────────────────┐      │                        ║
║  │ dynamic_params_gpu           │ ◀────┘                       ║
║  │ └─ input: &d_input_          │                               ║
║  └──────────────────────────────┘                               ║
║              ▲                                                  ║
║              │                                                  ║
║  Kernel receives: (static_params_gpu*, dynamic_params_gpu*)     ║
║                                                                 ║
╚═════════════════════════════════════════════════════════════════╝
```

---

## Zero-Copy Optimization

### Overview Zero-Copy

The pipeline supports **zero-copy data flow** between modules when upstream modules provide stable device addresses. This eliminates unnecessary `cudaMemcpy` operations, reducing latency and improving throughput.

### How It Works

**Workflow Order (Updated):**

```text
set_inputs(ModuleA) → configure_io(ModuleA) → get_outputs(ModuleA) →
set_inputs(ModuleB) → configure_io(ModuleB)
```

This order enables:

1. **Dynamic output topology**: Modules can determine output tensor shapes/counts in `configure_io()`
2. **Zero-copy analysis**: Pipeline queries memory characteristics before allocation
3. **Conditional allocation**: Modules skip input buffer allocation when upstream is stable

### API

**Memory Characteristics Structs** (`core/types.hpp`):

```cpp
// Input port memory characteristics - declares what an input requires
struct InputPortMemoryCharacteristics {
  bool requires_fixed_address_for_zero_copy{false};
  // true  = Can ONLY zero-copy if upstream provides fixed addresses
  //         (e.g., TensorRT in Graph mode - address captured in CUDA graph)
  // false = Can zero-copy with ANY upstream (stable or dynamic)
  //         (e.g., CUDA kernel with dynamic descriptors)
};

// Output port memory characteristics - declares what an output provides
struct OutputPortMemoryCharacteristics {
  bool provides_fixed_address_for_zero_copy{true};
  // true  = Address allocated once in setup_memory(), never changes (typical)
  // false = Address may change per iteration (e.g., ping-pong buffers)
};
```

**IModule Methods** ([imodule.hpp:203-237](../../../../framework/core/lib/include/core/imodule.hpp#L203-L237)):

```cpp
// Declare input requirements (for zero-copy analysis)
virtual InputPortMemoryCharacteristics
get_input_memory_characteristics(std::string_view port_name) const;

// Declare output capabilities (for zero-copy analysis)
virtual OutputPortMemoryCharacteristics
get_output_memory_characteristics(std::string_view port_name) const;

// Configure connection copy mode (called by pipeline during setup)
virtual void set_connection_copy_mode(
    std::string_view port_name,
    ConnectionCopyMode mode);
```

**Helper Function** (`core/types.hpp`):

```cpp
// Determines if zero-copy is possible for a connection
[[nodiscard]] inline bool can_zero_copy(
    const OutputPortMemoryCharacteristics &upstream,
    const InputPortMemoryCharacteristics &downstream) {
  return !downstream.requires_fixed_address_for_zero_copy ||
         upstream.provides_fixed_address_for_zero_copy;
}
```

**Key Design Points**:

- `set_connection_copy_mode()` is part of **IModule interface** (not module-specific)
- Default implementation is no-op (modules that don't support zero-copy ignore it)
- Must be called **before** `get_requirements()` for memory optimization to work
- Pipelines can call directly on `IModule*` without `dynamic_cast`

**ConnectionCopyMode** (`core/types.hpp`):

```cpp
enum class ConnectionCopyMode : std::uint8_t {
  Copy,     // Allocate buffer and copy data (cudaMemcpy)
  ZeroCopy  // Use upstream address directly (no copy)
};
```

### Zero-Copy Use Cases

The key question for zero-copy optimization is: **Does the downstream module need to allocate its own buffer and cudaMemcpy?**

There are **three zero-copy scenarios** based on the combination of upstream/downstream characteristics:

#### Use Case 1: Graph Mode with Stable Upstream (requires_fixed = true, upstream stable)

**Scenario**: TensorRT engine in CUDA graph mode with stable upstream

**Characteristics**:

- Downstream MUST have fixed address BEFORE `warmup()` (for graph capture)
- Upstream provides fixed address
- **Result**: Downstream uses upstream's address directly (no allocation, no copy)

**Example**: SampleModuleA in Graph mode ← ModuleB (stable output)

```cpp
// SampleModuleA in Graph mode - requires stable for graph capture
InputPortMemoryCharacteristics get_input_memory_characteristics(...) const override {
  // Graph mode: requires fixed addresses for CUDA graph capture
  const bool requires_fixed = (execution_mode_ == ExecutionMode::Graph);
  return {.requires_fixed_address_for_zero_copy = requires_fixed};  // true in Graph mode
}

// Pipeline uses can_zero_copy() helper to decide
auto upstream_chars = module_b->get_output_memory_characteristics("output");
auto downstream_chars = module_a->get_input_memory_characteristics("input");
if (can_zero_copy(upstream_chars, downstream_chars)) {
  module_a->set_connection_copy_mode("input", ConnectionCopyMode::ZeroCopy);
  // SampleModuleA skips allocation, uses ModuleB's address for graph capture
}
```

#### Use Case 2: Stream Mode with Any Upstream (requires_fixed = false, any upstream)

**Scenario**: TensorRT in stream mode OR kernel with dynamic descriptors

**Characteristics**:

- Downstream CAN accept any address (flexible via `set_tensor_address()` or dynamic descriptors)
- Upstream can be stable OR dynamic
- **Result**: Downstream uses upstream address each iteration (no allocation, no copy)

**Example**: SampleModuleA in Stream mode ← Any upstream

```cpp
// SampleModuleA in Stream mode - flexible addressing
InputPortMemoryCharacteristics get_input_memory_characteristics(...) const override {
  // Stream mode: uses set_tensor_address() per iteration, doesn't require stable
  const bool requires_fixed = (execution_mode_ == ExecutionMode::Graph);
  return {.requires_fixed_address_for_zero_copy = requires_fixed};  // false in Stream mode
}

// Pipeline enables zero-copy (downstream is flexible)
if (can_zero_copy(upstream_chars, downstream_chars)) {
  module_a->set_connection_copy_mode("input", ConnectionCopyMode::ZeroCopy);
  // SampleModuleA uses ModuleB's fixed address, set via set_tensor_address() each iteration
  // Address happens to be same every iteration, but module doesn't care
}
```

#### Use Case 3: Stream Mode with Unstable Upstream (requires_fixed = false, upstream dynamic)

**Scenario**: TensorRT in stream mode with CHANGING upstream addresses

**Characteristics**:

- Downstream CAN accept any address (flexible)
- Upstream address CHANGES per iteration
- **Result**: Downstream uses different upstream address each iteration (no allocation, no copy!)

**Example**: Hypothetical SampleModuleA_StreamMode ← ExternalInput (changing addresses)

```cpp
// SampleModuleA in stream mode - flexible addressing
PortMemoryCharacteristics get_input_memory_characteristics(...) const override {
  return {.provides_fixed_address_for_zero_copy = false,
          .requires_fixed_address_for_zero_copy = false,  // ← Can use any address!
};
}

// Pipeline enables zero-copy (downstream flexible)
if (!downstream.requires_fixed_address_for_zero_copy) {
  module_a->set_connection_copy_mode("input0", ConnectionCopyMode::ZeroCopy);
  // SampleModuleA uses whatever address upstream provides each iteration
  // Address changes, but module just calls set_tensor_address() with new address
  // Still zero-copy! No allocation, no cudaMemcpy needed
}
```

#### Zero-Copy Decision Matrix

| Upstream `provides_fixed_address_for_zero_copy` | Downstream `requires_fixed_address_for_zero_copy` | Zero-copy? | Explanation |
|-------------------------------------|--------------------------------------|------------|-------------|
| ✅ true | ✅ true | **YES** | Use Case 1: Graph mode, downstream uses upstream's fixed address |
| ✅ true | ❌ false | **YES** | Use Case 2: Stream mode, downstream uses upstream's fixed address each iteration |
| ❌ false | ❌ false | **YES** | Use Case 3: Stream mode, downstream uses upstream's changing address each iteration |
| ❌ false | ✅ true | **NO** | Incompatible: Downstream needs fixed address for graph capture, but upstream changes → MUST allocate + cudaMemcpy |

**Critical Insight**: Zero-copy is possible in 3 out of 4 cases! The ONLY case requiring allocation + cudaMemcpy is when:

- Downstream needs fixed address (graph mode)
- Upstream provides changing addresses

**Key Insight**: `requires_fixed_address_for_zero_copy = false` means "I don't need my own buffer, I'll use yours directly" - this ENABLES zero-copy, not prevents it!

### Example: ModuleB→ModuleA Zero-Copy

**Without Zero-Copy (Copy Mode)**:

```text
ModuleB.d_output_ → [cudaMemcpy in configure_io] → ModuleA.d_input0_ → TRT
```

**With Zero-Copy**:

```text
ModuleB.d_output_ → [pointer only] → TRT uses ModuleB's address directly
```

**Conditions for Zero-Copy**:

- ✅ Upstream module provides fixed address (`provides_fixed_address_for_zero_copy = true`)
- ✅ Downstream module can accept external address (see use cases below)
- ✅ Upstream address available when needed (timing depends on use case)

### Implementation Details

**Module Annotations**:

- **SampleModuleB** (ReLU): Declares `provides_fixed_address_for_zero_copy = true` for output (always stable)
- **SampleModuleA** (TRT): Declares `requires_fixed_address_for_zero_copy = true/false` based on `execution_mode`:
  - **Graph mode**: `true` (needs fixed addresses for CUDA graph capture)
  - **Stream mode**: `false` (flexible, uses `set_tensor_address()` per iteration)

**Pipeline Analysis** (in `setup()`):

```cpp
// ModuleA → ModuleB connection negotiation
auto module_a_output_chars = module_a_->get_output_memory_characteristics("output");
auto module_b_input_chars = module_b_->get_input_memory_characteristics("input");

if (can_zero_copy(module_a_output_chars, module_b_input_chars)) {
  module_b_->set_connection_copy_mode("input", ConnectionCopyMode::ZeroCopy);
  RT_LOG_INFO("A→B zero-copy enabled (A provides_fixed={}, B requires_fixed={})",
              module_a_output_chars.provides_fixed_address_for_zero_copy,
              module_b_input_chars.requires_fixed_address_for_zero_copy);
} else {
  module_b_->set_connection_copy_mode("input", ConnectionCopyMode::Copy);
  RT_LOG_INFO("A→B requires copy (A provides_fixed={}, B requires_fixed={})",
              module_a_output_chars.provides_fixed_address_for_zero_copy,
              module_b_input_chars.requires_fixed_address_for_zero_copy);
}
```

**Conditional Allocation** (in `setup_memory()`):

```cpp
if (!input0_upstream_is_stable_) {
  d_input0_ = allocate(...);  // Copy mode: allocate buffer
} else {
  d_input0_ = nullptr;  // Zero-copy mode: will use upstream address
}
```

**Zero-Copy Assignment** (in `set_inputs()`):

```cpp
if (input0_upstream_is_stable_) {
  d_input0_ = upstream_address;  // Direct assignment, no allocation
}
```

**Skip Copy** (in `configure_io()`):

```cpp
if (!input0_upstream_is_stable_) {
  cudaMemcpy(d_input0_, external_input0_data_, ...);  // Copy mode
} else {
  // Zero-copy mode: skip copy, already using upstream address
}
```

### Limitations and Design Considerations

**Current Implementation** (as of October 2025):

- Pipeline enables zero-copy ONLY when: `upstream.provides_fixed_address_for_zero_copy && downstream.requires_fixed_address_for_zero_copy`
- This covers **Use Case 1** (Graph Mode with Stable Upstream) ONLY
- **Gaps**: Does NOT support Use Case 2 or Use Case 3 (stream mode scenarios)

**Future Enhancement**:
To support all three use cases, change pipeline logic to:

```cpp
// Enable zero-copy for stream mode (Use Cases 2 & 3)
const bool zero_copy_possible =
    (upstream_chars.provides_fixed_address_for_zero_copy && downstream_chars.requires_fixed_address_for_zero_copy) ||  // Use Case 1
    (!downstream_chars.requires_fixed_address_for_zero_copy);  // Use Cases 2 & 3 (stream mode)

// Simplified: zero-copy possible unless (upstream dynamic AND downstream needs stable)
const bool zero_copy_possible =
    upstream_chars.provides_fixed_address_for_zero_copy || !downstream_chars.requires_fixed_address_for_zero_copy;
```

**Zero-copy works when:**

- ✅ Use Case 1: Graph mode with stable upstream (current implementation)
- ✅ Use Case 2: Stream mode with stable upstream (future: needs pipeline + module changes)
- ✅ Use Case 3: Stream mode with dynamic upstream (future: needs pipeline + module changes)

**Zero-copy does NOT work when:**

- ❌ Graph mode with dynamic upstream (incompatible: needs fixed address but upstream changes)

### Performance Benefits

- **Reduced Latency**: Eliminates `cudaMemcpy` overhead (~10-20% configure_io reduction)
- **Lower Memory Usage**: No duplicate buffer allocation
- **Better Throughput**: Less memory bandwidth consumption

### Current Status

**Implementation Status**: ✅ Complete (infrastructure in place)
**Default Behavior**: Zero-copy configured based on ExecutionMode
**Tested Scenarios**: Graph mode with copy (external inputs require fixed addresses)
**Future Work**: Test stream mode with zero-copy for external inputs

### Relationship to ExecutionMode

ExecutionMode determines zero-copy eligibility for external inputs:

**Graph Mode**:

- External inputs → ModuleA: **Copy** (external addresses dynamic, TRT needs fixed addresses)
- ModuleA → ModuleB: **ZeroCopy** (ModuleA provides stable output, ModuleB flexible)
- Memory allocation: 3 tensors for ModuleA (2 inputs + 1 output)

**Stream Mode**:

- External inputs → ModuleA: **ZeroCopy** (TRT uses `set_tensor_address()`, can handle changing addresses)
- ModuleA → ModuleB: **ZeroCopy** (same as graph mode)
- Memory allocation: 1 tensor for ModuleA (1 output only)

**Key Insight**: ExecutionMode affects ONLY external input handling. Internal module connections (ModuleA→ModuleB) benefit from zero-copy in BOTH modes.

---

## References

### Source Files

**Core Pipeline and Modules:**

- [sample_module_a.hpp](../src/sample_module_a.hpp) / [sample_module_a.cpp](../src/sample_module_a.cpp)
- [sample_module_b.hpp](../src/sample_module_b.hpp) / [sample_module_b.cpp](../src/sample_module_b.cpp)
- [sample_module_b_kernel.cuh](../src/sample_module_b_kernel.cuh) / [sample_module_b_kernel.cu](../src/sample_module_b_kernel.cu)
- [sample_pipeline.hpp](../src/sample_pipeline.hpp) / [sample_pipeline.cpp](../src/sample_pipeline.cpp)

**Factory Pattern:**

- [sample_module_factories.hpp](../src/sample_module_factories.hpp) / [sample_module_factories.cpp](../src/sample_module_factories.cpp)
- [sample_pipeline_factory.hpp](../src/sample_pipeline_factory.hpp) / [sample_pipeline_factory.cpp](../src/sample_pipeline_factory.cpp)

**Tests:**

- [sample_pipeline_test.cpp](../tests/sample_pipeline_test.cpp)
- [sample_pipeline_factory_test.cpp](../tests/sample_pipeline_factory_test.cpp)
- [sample_module_b_standalone_test.cpp](../tests/sample_module_b_standalone_test.cpp)

### External References

- [CUDA Programming Guide - CUDA Graphs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)
- [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
- [CUDA Driver API Reference](https://docs.nvidia.com/cuda/cuda-driver-api/index.html)

---
