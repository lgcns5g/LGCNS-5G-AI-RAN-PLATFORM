Tensor
======

Multi-dimensional array descriptors and memory management for tensor data.

Overview
--------

The Tensor library provides lightweight data structures for describing and allocating
multi-dimensional arrays (tensors) with type-safe memory management. It supports various
numeric types including integers, floating-point numbers, and complex values.

Core Concepts
-------------

Tensor Info
~~~~~~~~~~~

**TensorInfo** describes tensor properties including data type and dimensions. It provides
compatibility validation and element count calculation.

Creating a Tensor Descriptor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../framework/tensor/tests/tensor_sample_tests.cpp
   :language: cpp
   :start-after: example-begin basic-tensor-info-1
   :end-before: example-end basic-tensor-info-1
   :dedent: 4

Compatibility Checking
^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../framework/tensor/tests/tensor_sample_tests.cpp
   :language: cpp
   :start-after: example-begin tensor-compatibility-1
   :end-before: example-end tensor-compatibility-1
   :dedent: 4

Data Types
~~~~~~~~~~

**NvDataType** defines supported tensor element types for integers, floating-point values,
and complex numbers with various precisions.

Type Names and Traits
^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../framework/tensor/tests/tensor_sample_tests.cpp
   :language: cpp
   :start-after: example-begin data-type-strings-1
   :end-before: example-end data-type-strings-1
   :dedent: 4

The library provides compile-time type traits for mapping between ``NvDataType``
enumeration values and C++ types:

.. literalinclude:: ../../../framework/tensor/tests/tensor_sample_tests.cpp
   :language: cpp
   :start-after: example-begin data-type-traits-1
   :end-before: example-end data-type-traits-1
   :dedent: 4

Storage Element Size
^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../framework/tensor/tests/tensor_sample_tests.cpp
   :language: cpp
   :start-after: example-begin storage-element-size-1
   :end-before: example-end storage-element-size-1
   :dedent: 4

Tensor Arena
~~~~~~~~~~~~

**TensorArena** provides RAII-based memory allocation for tensor data with support for
device and host-pinned memory.

Device Memory
^^^^^^^^^^^^^

.. literalinclude:: ../../../framework/tensor/tests/tensor_sample_tests.cpp
   :language: cpp
   :start-after: example-begin device-arena-1
   :end-before: example-end device-arena-1
   :dedent: 4

Host Pinned Memory
^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../framework/tensor/tests/tensor_sample_tests.cpp
   :language: cpp
   :start-after: example-begin host-pinned-arena-1
   :end-before: example-end host-pinned-arena-1
   :dedent: 4

Complete Example
----------------

.. literalinclude:: ../../../framework/tensor/tests/tensor_sample_tests.cpp
   :language: cpp
   :start-after: example-begin complete-example-1
   :end-before: example-end complete-example-1
   :dedent: 4

Additional Examples
-------------------

For more examples, see:

-  ``framework/tensor/tests/tensor_sample_tests.cpp`` - Documentation examples and
   unit tests

API Reference
-------------

.. doxygennamespace:: framework::tensor
   :content-only:
   :members:
   :undoc-members:
