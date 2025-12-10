LDPC
====

Low-Density Parity-Check (LDPC) coding modules for 5G NR channel decoding.

Overview
--------

The LDPC library provides GPU-accelerated modules for 5G NR LDPC channel
decoding, implementing the 3GPP TS 38.212 specification. Built on highly
optimized cuPHY LDPC CUDA kernels, these modules deliver efficient runtime
performance on NVIDIA GPUs. The library consists of three pipeline modules
that work together to decode received data:

-  **LDPC Derate Matching**: Reverses rate matching to produce LLRs suitable
   for LDPC decoding
-  **LDPC Decoder**: Performs LDPC decoding on Log-Likelihood Ratios (LLRs)
-  **CRC Decoder**: Validates CRC checksums and concatenates code blocks into
   transport blocks

Each module implements the ``pipeline::IModule`` interface from
the :doc:`pipeline library </api/framework/pipeline>`, enabling integration
into larger processing pipelines with standardized configuration, memory
management, and execution patterns.

Core Concepts
-------------

Module Configuration
~~~~~~~~~~~~~~~~~~~~

All LDPC modules are configured with static parameters at construction time.
These parameters define maximum capacities and processing modes.

Creating an LDPC Decoder
^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../ran/runtime/ldpc/tests/ldpc_sample_tests.cpp
   :language: cpp
   :start-after: example-begin decoder-setup-1
   :end-before: example-end decoder-setup-1
   :dedent: 4

Creating a Derate Match Module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../ran/runtime/ldpc/tests/ldpc_sample_tests.cpp
   :language: cpp
   :start-after: example-begin derate-match-setup-1
   :end-before: example-end derate-match-setup-1
   :dedent: 4

Creating a CRC Decoder
^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../ran/runtime/ldpc/tests/ldpc_sample_tests.cpp
   :language: cpp
   :start-after: example-begin crc-decoder-setup-1
   :end-before: example-end crc-decoder-setup-1
   :dedent: 4

LDPC Decoder Parameters
^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../ran/runtime/ldpc/tests/ldpc_sample_tests.cpp
   :language: cpp
   :start-after: example-begin decoder-params-1
   :end-before: example-end decoder-params-1
   :dedent: 4

Key parameters include:

-  ``clamp_value``: Maximum absolute value for input LLRs
-  ``early_termination``: Stop decoding when convergence is detected
-  ``max_num_iterations``: Maximum decoder iterations per code block
-  ``normalization_factor``: LLR normalization applied during decoding

Module Ports
~~~~~~~~~~~~

Each module exposes input and output ports for data flow:

.. literalinclude:: ../../../../ran/runtime/ldpc/tests/ldpc_sample_tests.cpp
   :language: cpp
   :start-after: example-begin module-ports-1
   :end-before: example-end module-ports-1
   :dedent: 4

CRC Decoder Ports
^^^^^^^^^^^^^^^^^

The CRC decoder processes decoded bits and outputs CRC results for both code
blocks and transport blocks:

.. literalinclude:: ../../../../ran/runtime/ldpc/tests/ldpc_sample_tests.cpp
   :language: cpp
   :start-after: example-begin crc-ports-1
   :end-before: example-end crc-ports-1
   :dedent: 4

The CRC decoder takes decoded bits as input and produces three outputs: code
block CRCs, transport block CRCs, and transport block payloads after CRC
validation and concatenation.

Memory Management
~~~~~~~~~~~~~~~~~

Modules report memory requirements that must be allocated before use:

.. literalinclude:: ../../../../ran/runtime/ldpc/tests/ldpc_sample_tests.cpp
   :language: cpp
   :start-after: example-begin memory-requirements-1
   :end-before: example-end memory-requirements-1
   :dedent: 4

Processing Flow
~~~~~~~~~~~~~~~

A typical LDPC decoding flow involves three stages:

1. **Derate Matching**: Converts received symbols to LLRs for LDPC decoding
2. **LDPC Decoding**: Decodes LLRs to produce hard-decision bits
3. **CRC Validation**: Checks CRC and assembles transport blocks

Each stage is configured with dynamic parameters (transport block
configuration) via the ``configure_io()`` method and executed via
``execute()`` or integrated into a CUDA graph.

Execution Modes
~~~~~~~~~~~~~~~

Modules support two execution modes:

-  **Stream Execution**: Direct kernel launch via
   ``IStreamExecutor::execute()``
-  **CUDA Graph Execution**: Graph node creation via
   ``IGraphNodeProvider::add_node_to_graph()``

CUDA graph mode enables lower-latency operation via execution graphs.

Stream Execution
^^^^^^^^^^^^^^^^

Stream execution provides direct kernel launching on a CUDA stream. After
setting inputs, configure the module and execute:

.. literalinclude:: ../../../../ran/runtime/ldpc/tests/ldpc_decoder_module_test.cpp
   :language: cpp
   :start-after: example-begin configure-and-execute-1
   :end-before: example-end configure-and-execute-1
   :dedent: 4

The ``configure_io()`` method sets up internal state based on transport block
parameters, then ``execute()`` launches the kernel on the provided stream.

CUDA Graph Execution
^^^^^^^^^^^^^^^^^^^^

Modules implementing ``IGraphNodeProvider`` can be integrated into CUDA graphs
for lower-latency execution. First, create a graph manager and get the module's
graph interface:

.. literalinclude:: ../../../../ran/runtime/ldpc/tests/ldpc_derate_match_module_test.cpp
   :language: cpp
   :start-after: example-begin graph-execution-1
   :end-before: example-end graph-execution-1
   :dedent: 8

Next, add the module's kernel node to the graph:

.. literalinclude:: ../../../../ran/runtime/ldpc/tests/ldpc_derate_match_module_test.cpp
   :language: cpp
   :start-after: example-begin graph-execution-2
   :end-before: example-end graph-execution-2
   :dedent: 8

Finally, instantiate the graph, upload it to the GPU, update parameters, and
launch:

.. literalinclude:: ../../../../ran/runtime/ldpc/tests/ldpc_derate_match_module_test.cpp
   :language: cpp
   :start-after: example-begin graph-execution-3
   :end-before: example-end graph-execution-3
   :dedent: 8

The graph is instantiated once and can be launched repeatedly with updated
parameters. This approach minimizes kernel launch overhead for repeated
operations.

Additional Examples
-------------------

For complete working examples with full setup and validation, see the test
files:

-  **LDPC Decoder Tests**:
   ``ran/runtime/ldpc/tests/ldpc_decoder_module_test.cpp`` - Full
   decoder module tests with H5 test vectors
-  **Derate Match Tests**:
   ``ran/runtime/ldpc/tests/ldpc_derate_match_module_test.cpp`` -
   Stream and graph execution modes
-  **CRC Decoder Tests**:
   ``ran/runtime/ldpc/tests/crc_decoder_module_test.cpp`` - CRC
   validation and transport block assembly

These test files demonstrate complete workflows including memory allocation,
input/output setup, and result validation.

API Reference
-------------

.. doxygennamespace:: ran::ldpc
   :content-only:
   :members:
   :undoc-members:

