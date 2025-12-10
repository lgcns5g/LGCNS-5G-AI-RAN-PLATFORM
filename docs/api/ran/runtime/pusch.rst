PUSCH
=====

Physical Uplink Shared Channel processing pipeline for 5G NR.

Overview
--------

The PUSCH runtime module provides a complete processing pipeline for
Physical Uplink Shared Channel (PUSCH) in 5G NR systems. It consists of
a TensorRT-accelerated inner receiver stage for signal processing and an
outer receiver stage with LDPC channel decoding to transform received
uplink signals into decoded transport blocks.

The pipeline consists of the following modules organized into two stages:

-  **Inner Receiver**: TensorRT-based signal processing that performs
   channel estimation, noise estimation, equalization, and soft
   demapping
-  **Outer Receiver**: Channel decoding chain that performs derate
   matching, LDPC decode, and CRC check

The inner receiver module uses TensorRT for GPU-accelerated neural
network inference, while the outer receiver uses the :doc:`LDPC library
</api/ran/runtime/ldpc>` modules for channel decoding. All modules
implement the ``pipeline::IModule`` interface from the
:doc:`pipeline library </api/framework/pipeline>`, enabling standardized
configuration, memory management, and execution patterns.

Key Features
~~~~~~~~~~~~

-  **TensorRT Inner Receiver**: GPU-accelerated signal processing with
   TensorRT
-  **Channel Decoding**: LDPC decoder with derate matching and CRC
   validation
-  **Pipeline Architecture**: Modular design with factory-based module
   creation
-  **Execution Modes**: Support for both stream and CUDA graph
   execution
-  **Memory Management**: Efficient memory allocation and reuse across
   iterations

Core Concepts
-------------

Pipeline Architecture
~~~~~~~~~~~~~~~~~~~~~

The PUSCH pipeline processes uplink signals through two stages:

**Inner Receiver Stage**

The inner receiver module (``InnerRxModule``) performs signal
processing on received time-frequency samples:

1. **Channel Estimation**: Estimates channel response using DMRS
   (Demodulation Reference Signals)
2. **Noise Estimation**: Estimates noise covariance matrix for equalization
3. **Equalization**: Compensates for channel effects using estimated
   channel
4. **Soft Demapping**: Generates Log-Likelihood Ratios (LLRs) from
   equalized symbols

This stage is implemented using TensorRT for optimized GPU execution.

**Outer Receiver Stage**

The outer receiver consists of three :doc:`LDPC library
</api/ran/runtime/ldpc>` modules:

1. **LDPC Derate Match Module**: Reverses rate matching to prepare LLRs
   for LDPC decoding
2. **LDPC Decoder Module**: Performs iterative LDPC decoding on LLRs to
   produce hard-decision bits
3. **CRC Decoder Module**: Validates CRC checksums and concatenates
   code blocks into transport blocks

Each module implements the ``IModule`` interface from the :doc:`pipeline
library </api/framework/pipeline>` and can be created using the factory
pattern.

Module Factory
~~~~~~~~~~~~~~

The ``PuschModuleFactory`` creates all module types used in the
pipeline:

.. literalinclude:: ../../../../ran/runtime/pusch/tests/pusch_sample_tests.cpp
   :language: cpp
   :start-after: example-begin create-factory-1
   :end-before: example-end create-factory-1
   :dedent: 4

Module Configuration
~~~~~~~~~~~~~~~~~~~~

Modules are configured using static parameters at creation time:

.. literalinclude:: ../../../../ran/runtime/pusch/tests/pusch_sample_tests.cpp
   :language: cpp
   :start-after: example-begin create-inner-rx-1
   :end-before: example-end create-inner-rx-1
   :dedent: 4

Physical layer parameters include antenna configuration, PRB count, and
bandwidth configuration.

Pipeline Specification
~~~~~~~~~~~~~~~~~~~~~~

Pipeline behavior is defined using ``PipelineSpec``:

.. literalinclude:: ../../../../ran/runtime/pusch/tests/pusch_sample_tests.cpp
   :language: cpp
   :start-after: example-begin create-pipeline-spec-1
   :end-before: example-end create-pipeline-spec-1
   :dedent: 4

Module Introspection
~~~~~~~~~~~~~~~~~~~~

Modules provide introspection APIs for discovering ports and tensor
information:

.. literalinclude:: ../../../../ran/runtime/pusch/tests/pusch_sample_tests.cpp
   :language: cpp
   :start-after: example-begin inspect-ports-1
   :end-before: example-end inspect-ports-1
   :dedent: 4

Execution Flow
~~~~~~~~~~~~~~

A typical PUSCH processing iteration follows these steps:

Pipeline Setup
^^^^^^^^^^^^^^

Setup allocates memory and initializes all modules:

.. literalinclude:: ../../../../ran/runtime/pusch/tests/pusch_pipeline_runner.cpp
   :language: cpp
   :start-after: example-begin pipeline-setup-1
   :end-before: example-end pipeline-setup-1
   :dedent: 4

This is called once during initialization before any processing begins.

Configure and Execute
^^^^^^^^^^^^^^^^^^^^^

For each slot, configure the pipeline with dynamic parameters:

.. literalinclude:: ../../../../ran/runtime/pusch/tests/pusch_pipeline_runner.cpp
   :language: cpp
   :start-after: example-begin configure-execute-1
   :end-before: example-end configure-execute-1
   :dedent: 4

The ``configure_io()`` method sets up input/output ports and prepares
internal state based on transport block parameters. The ``warmup()``
method prepares execution paths (called once after setup).

Then execute the pipeline:

.. literalinclude:: ../../../../ran/runtime/pusch/tests/pusch_pipeline_runner.cpp
   :language: cpp
   :start-after: example-begin execute-pipeline-1
   :end-before: example-end execute-pipeline-1
   :dedent: 4

The ``execute_stream()`` or ``execute_graph()`` method runs the
complete pipeline processing on the provided CUDA stream.

Dynamic parameters change per slot and include UE-specific
configuration like MCS, PRB allocation, and HARQ parameters.

Additional Examples
-------------------

For complete working examples with full setup and validation, see:

-  ``ran/runtime/pusch/tests/pusch_pipeline_runner.cpp`` - Pipeline
   runner implementation for benchmarks and tests
-  ``ran/runtime/pusch/tests/pusch_pipeline_test.cpp`` - Complete
   pipeline test with TensorRT integration
-  ``ran/runtime/pusch/tests/pusch_sample_tests.cpp`` -
   Documentation examples

API Reference
-------------

.. doxygennamespace:: ran::pusch
   :content-only:
   :members:
   :undoc-members:
