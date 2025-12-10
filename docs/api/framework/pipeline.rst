Pipeline
========

Modular data processing framework for GPU-accelerated computational
pipelines.

Overview
--------

The Pipeline library provides a high-performance framework for building
modular, GPU-accelerated processing pipelines. It enables composing complex
data transformations from reusable modules with efficient CUDA stream
management and optional CUDA graph execution for minimal latency.

Key Features
~~~~~~~~~~~~

-  **Modular Architecture**: Compose pipelines from reusable processing
   modules (``IModule``)
-  **Flexible Execution**: Support for both stream-based and CUDA graph
   execution modes
-  **Factory Pattern**: Configuration-driven pipeline and module creation
   via ``PipelineSpec``
-  **Memory Management**: Unified memory allocation with
   ``PipelineMemoryManager``
-  **Zero-Copy Optimization**: Direct data flow between modules without
   intermediate copies
-  **Module Routing**: Automatic data routing between modules based on
   port connections
-  **TensorRT Integration**: First-class support for TensorRT engines
-  **CUDA Graph Capture**: Automatic CUDA graph construction for low-latency
   execution

Quick Start
-----------

Creating a Module
~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../framework/pipeline/tests/pipeline_sample_tests.cpp
   :language: cpp
   :start-after: example-begin module-creation-1
   :end-before: example-end module-creation-1
   :dedent: 4

Querying Module Ports
~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../framework/pipeline/tests/pipeline_sample_tests.cpp
   :language: cpp
   :start-after: example-begin module-ports-1
   :end-before: example-end module-ports-1
   :dedent: 4

Using a Module Factory
~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../framework/pipeline/tests/pipeline_sample_tests.cpp
   :language: cpp
   :start-after: example-begin module-factory-1
   :end-before: example-end module-factory-1
   :dedent: 4

Core Concepts
-------------

Pipeline Architecture
~~~~~~~~~~~~~~~~~~~~~

A **Pipeline** orchestrates the execution of multiple **Modules** connected
through named **Ports**. Data flows through the pipeline according to defined
connections, with modules processing data on GPU streams.

.. code-block:: text

   External Input → Module A → Module B → Module C → External Output
                      ↓           ↓           ↓
                  Port Connections (defined in PipelineSpec)

Key abstractions:

- **IPipeline**: Coordinates module execution and manages data flow
- **IModule**: Individual processing unit with defined inputs/outputs
- **PortInfo**: Describes tensor data at module inputs/outputs
- **PipelineSpec**: Configuration structure for pipeline creation

Modules
~~~~~~~

A **Module** (``IModule``) is the fundamental processing unit in a pipeline.
Each module:

- Has named input and output ports
- Implements one of several execution interfaces:

  - ``IStreamExecutor``: Direct CUDA stream execution
  - ``IGraphNodeProvider``: CUDA graph node provider for graph mode
  - ``IAllocationInfoProvider``: Memory requirements for allocation

- Receives its memory allocation from ``PipelineMemoryManager``
- Processes data independently without knowledge of other modules

Modules are created via ``IModuleFactory`` and configured with static
parameters at construction time. Dynamic parameters (per-iteration data) are
provided via ``configure_io()``.

Pipeline Specification
~~~~~~~~~~~~~~~~~~~~~~

A ``PipelineSpec`` defines the complete pipeline configuration including
modules, connections, and execution mode.

Basic Pipeline Specification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../framework/pipeline/tests/pipeline_sample_tests.cpp
   :language: cpp
   :start-after: example-begin pipeline-spec-basic-1
   :end-before: example-end pipeline-spec-basic-1
   :dedent: 4

Pipeline with Module Connections
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../framework/pipeline/tests/pipeline_sample_tests.cpp
   :language: cpp
   :start-after: example-begin pipeline-spec-connections-1
   :end-before: example-end pipeline-spec-connections-1
   :dedent: 4

Execution Modes
~~~~~~~~~~~~~~~

The pipeline supports two execution modes configured via ``ExecutionMode``:

.. literalinclude:: ../../../framework/pipeline/tests/pipeline_sample_tests.cpp
   :language: cpp
   :start-after: example-begin execution-modes-1
   :end-before: example-end execution-modes-1
   :dedent: 4

**Stream Mode** (``ExecutionMode::Stream``):

- Sequential module execution on a CUDA stream
- Flexible addressing - modules accept different tensor addresses per
  iteration
- Supports dynamic topology changes
- Suitable for development and debugging

**Graph Mode** (``ExecutionMode::Graph``):

- Pre-built CUDA graph executed as a single unit
- Fixed addressing - tensor addresses captured during warmup
- Lower latency through single graph launch
- Requires stable tensor addresses before graph build
- Suitable for production deployments

Pipeline Lifecycle
~~~~~~~~~~~~~~~~~~

A typical pipeline follows this lifecycle:

1. **Construction**: Create pipeline and modules via factory
2. **Setup**: Allocate memory and initialize modules (``setup()``)
3. **I/O Configuration**: Establish connections and set inputs
   (``configure_io()``)
4. **Warmup**: One-time initialization - load models, capture graphs
   (``warmup()``)
5. **Graph Build** (graph mode only): Build CUDA graph (``build_graph()`` or
   automatic)
6. **Execution**: Process data (``execute_stream()`` or
   ``execute_graph()``)
7. **Iteration**: Repeat steps 3 and 6 with new data

**Warmup Phase**:

The ``warmup()`` method performs expensive one-time initialization:

- Loading models to device memory (TensorRT engines)
- Initializing module-specific resources

This is called once after the first ``configure_io()`` and before execution.
For TensorRT modules, graph capture requires a non-default CUDA stream.

Data Flow
~~~~~~~~~

Port Information
^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../framework/pipeline/tests/pipeline_sample_tests.cpp
   :language: cpp
   :start-after: example-begin port-info-1
   :end-before: example-end port-info-1
   :dedent: 4

``PortInfo`` describes tensor data at module ports:

- **name**: Port identifier (e.g., "input0", "output")
- **tensors**: Vector of device tensors with metadata

Data flow sequence per iteration:

1. External inputs provided to pipeline via ``PortInfo``
2. ``configure_io()`` calls ``set_inputs()`` on first module
3. Module processes and provides outputs via ``get_outputs()``
4. Router passes outputs to next module's inputs
5. Process repeats through all modules
6. Final outputs mapped to external outputs

``DynamicParams`` can be passed to ``configure_io()`` to provide per-iteration
configuration to modules. Modules receive these parameters and can use them to
update their per-iteration state.

Memory Management
~~~~~~~~~~~~~~~~~

Connection Copy Modes
^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../framework/pipeline/tests/pipeline_sample_tests.cpp
   :language: cpp
   :start-after: example-begin connection-copy-mode-1
   :end-before: example-end connection-copy-mode-1
   :dedent: 4

``ConnectionCopyMode`` controls data transfer between modules:

- **Copy**: Allocate buffer and copy data via ``cudaMemcpy``
- **ZeroCopy**: Use upstream pointer directly (no copy)

Memory Characteristics
^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../framework/pipeline/tests/pipeline_sample_tests.cpp
   :language: cpp
   :start-after: example-begin memory-characteristics-1
   :end-before: example-end memory-characteristics-1
   :dedent: 4

Modules declare their memory capabilities:

- ``provides_fixed_address_for_zero_copy``: Whether output addresses are
  stable
- ``requires_fixed_address_for_zero_copy``: Whether inputs need fixed
  addresses

The helper function ``can_zero_copy()`` determines if zero-copy is possible
for a connection based on upstream and downstream characteristics.

Zero-Copy Optimization
~~~~~~~~~~~~~~~~~~~~~~

The pipeline supports zero-copy data flow where downstream modules use
upstream pointers directly, eliminating intermediate ``cudaMemcpy`` operations.

**TensorRT Modules**:

Zero-copy requires upstream modules to provide fixed/stable device addresses. TensorRT
fuses I/O addresses during graph capture, so addresses must be known ahead of time.

**Non-TensorRT Modules**:

Zero-copy is supported in both Graph and Stream modes. Device pointers can be
updated per-iteration via ``configure_io()``, enabling flexible addressing.
However, when upstream module pointers change, data must be copied to the
downstream module's input buffer.

**General Rule**:

Zero-copy possible when: upstream modules provide fixed address OR downstream
modules accept dynamic addresses.

Benefits:

- Reduced latency (eliminates copy overhead)
- Lower memory usage (no duplicate buffers)
- Better throughput (less memory bandwidth consumption)

Factory Pattern
~~~~~~~~~~~~~~~

The pipeline library uses the factory pattern for flexible creation:

**Module Factory** (``IModuleFactory``):

- Creates modules by type identifier (string)
- Receives static configuration parameters as ``std::any``
- Returns ``std::unique_ptr<IModule>``

**Pipeline Factory** (``IPipelineFactory``):

- Creates pipelines by type identifier
- Receives module factory and ``PipelineSpec``
- Constructs complete pipeline with all modules and connections

Complete Example
----------------

The sample pipeline in ``framework/pipeline/samples/`` demonstrates a
complete two-module pipeline chaining TensorRT execution with a CUDA kernel:

.. code-block:: text

   External Input 0 ┐
                    ├─→ Module A (TensorRT Add) ─→ Module B (ReLU) ─→ Output
   External Input 1 ┘

Creating the Pipeline
~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../framework/pipeline/samples/tests/sample_pipeline_test.cpp
   :language: cpp
   :start-after: example-begin pipeline-creation-1
   :end-before: example-end pipeline-creation-1
   :dedent: 4

Configuring and Executing
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../framework/pipeline/samples/tests/sample_pipeline_test.cpp
   :language: cpp
   :start-after: example-begin pipeline-configure-execute-1
   :end-before: example-end pipeline-configure-execute-1
   :dedent: 4

Graph Mode Execution
~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../framework/pipeline/samples/tests/sample_pipeline_test.cpp
   :language: cpp
   :start-after: example-begin pipeline-graph-execute-1
   :end-before: example-end pipeline-graph-execute-1
   :dedent: 4

Additional Examples
-------------------

For complete working examples with full setup and validation, see the test
files:

-  **Documentation Examples**:
   ``framework/pipeline/tests/pipeline_sample_tests.cpp`` - Clean examples
   demonstrating core concepts with focused test cases
-  **Complete Pipeline**:
   ``framework/pipeline/samples/tests/sample_pipeline_test.cpp`` -
   Full-featured pipeline with TensorRT and CUDA kernels, including stream
   and graph execution modes

These test files demonstrate complete workflows including memory allocation,
module creation, pipeline setup, warmup, execution, and result validation.

API Reference
-------------

.. doxygennamespace:: framework::pipeline
   :content-only:
   :members:
   :undoc-members:
