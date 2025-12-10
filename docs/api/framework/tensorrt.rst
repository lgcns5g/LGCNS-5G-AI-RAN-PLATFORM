TensorRT
========

Integration with `NVIDIA TensorRT <https://developer.nvidia.com/tensorrt>`_ for GPU-accelerated
execution.

Overview
--------

The TensorRT library provides a simplified interface for integrating
TensorRT engines into real-time applications.

Key Features
~~~~~~~~~~~~

-  **Flexible Tensor Metadata** - User-provided dimensions and strides
-  **Automatic Stride Computation** - Automatic layout computation for tensors
-  **CUDA Graph Support** - Pre/post enqueue hooks for graph capture
-  **Engine Abstraction** - Interface-based design

Core Concepts
-------------

MLIR Tensor Parameters
~~~~~~~~~~~~~~~~~~~~~~~

**MLIRTensorParams** defines the metadata for tensors used by the TensorRT
engine. Each tensor requires:

- **name**: Tensor identifier matching the TensorRT engine
- **data_type**: Element data type (e.g., ``TensorR32F`` for float32)
- **rank**: Number of dimensions (0 for scalar, 1-8 for tensors)
- **dims**: Size of each dimension
- **strides**: Optional memory layout (auto-computed if not provided)

When strides are not provided (last stride == 0), row-major strides are
automatically computed from dimensions.

Creating Tensor Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../framework/tensorrt/tests/tensorrt_sample_tests.cpp
   :language: cpp
   :start-after: example-begin tensor-params-1
   :end-before: example-end tensor-params-1
   :dedent: 4

Tensor Parameters with Strides
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../framework/tensorrt/tests/tensorrt_sample_tests.cpp
   :language: cpp
   :start-after: example-begin tensor-params-strides-1
   :end-before: example-end tensor-params-strides-1
   :dedent: 4

MLIR TensorRT Engine
~~~~~~~~~~~~~~~~~~~~~

**MLIRTrtEngine** provides a streamlined TensorRT interface that:

- Eliminates batch size management (users handle batching externally)
- Removes internal buffer allocation (users provide pre-allocated CUDA
  buffers)
- Uses constructor-based initialization (no separate init() phase)
- Accepts tensor dimensions and strides directly in MLIRTensorParams

The engine operates in three phases: construction, setup, and execution.

Engine Construction
^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../framework/tensorrt/tests/tensorrt_sample_tests.cpp
   :language: cpp
   :start-after: example-begin engine-construction-1
   :end-before: example-end engine-construction-1
   :dedent: 4

The engine is fully initialized in the constructor. All tensor shapes must be
provided during construction.

Engine Setup
^^^^^^^^^^^^

.. literalinclude:: ../../../framework/tensorrt/tests/tensorrt_sample_tests.cpp
   :language: cpp
   :start-after: example-begin engine-setup-1
   :end-before: example-end engine-setup-1
   :dedent: 4

Setup caches the provided buffer pointers for use during execution. Buffers
are direct pointers to CUDA device memory and must remain valid for the
lifetime of execution operations.

Engine Execution
^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../framework/tensorrt/tests/tensorrt_sample_tests.cpp
   :language: cpp
   :start-after: example-begin engine-execution-1
   :end-before: example-end engine-execution-1
   :dedent: 4

The ``run()`` method executes asynchronously on the provided CUDA stream.

TensorRT Engine Interface
~~~~~~~~~~~~~~~~~~~~~~~~~~

**ITrtEngine** is an abstract interface for TensorRT operations. The concrete
**TrtEngine** implementation wraps the NVIDIA TensorRT runtime, while
**NullTrtEngine** provides a null-object pattern for testing.

Multi-Rank Tensors
~~~~~~~~~~~~~~~~~~

The library supports tensors with ranks 0 through 8:

.. literalinclude:: ../../../framework/tensorrt/tests/tensorrt_sample_tests.cpp
   :language: cpp
   :start-after: example-begin multi-rank-tensors-1
   :end-before: example-end multi-rank-tensors-1
   :dedent: 4

Complete Example
----------------

This example demonstrates the full workflow from tensor definition
through execution:

.. literalinclude:: ../../../framework/tensorrt/tests/tensorrt_sample_tests.cpp
   :language: cpp
   :start-after: example-begin complete-workflow-1
   :end-before: example-end complete-workflow-1
   :dedent: 4

Additional Examples
-------------------

For more examples, see:

-  ``framework/tensorrt/tests/tensorrt_sample_tests.cpp`` - Documentation
   examples and unit tests

External Resources
------------------

-  `NVIDIA TensorRT Documentation <https://docs.nvidia.com/deeplearning/tensorrt/>`_

API Reference
-------------

.. doxygennamespace:: framework::tensorrt
   :content-only:
   :members:
   :undoc-members:
