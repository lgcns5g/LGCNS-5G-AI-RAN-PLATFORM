Memory
======

Memory management and buffer allocation utilities for CPU and GPU.

Overview
--------

The Memory library provides efficient memory management abstractions for
CUDA-enabled applications. It simplifies memory allocation, buffer management,
and data transfers between CPU and GPU with RAII semantics.

Key Features
~~~~~~~~~~~~

-  **Type-Safe Buffers**: Template-based buffer management with automatic
   cleanup
-  **Multiple Allocators**: Device (GPU), pinned host, and monotonic allocators
-  **Smart Pointers**: CUDA-aware unique_ptr utilities with custom deleters
-  **Zero-Copy Support**: Efficient pinned memory for CPU-GPU transfers
-  **Fast Sequential Allocation**: Monotonic allocators for deterministic performance

Core Concepts
-------------

Buffers
~~~~~~~

**Buffer** is a RAII wrapper for memory allocations that handles cleanup
automatically. Buffers can use different allocators to control where memory is
allocated (device, pinned host, etc.).

Buffers support:

- Automatic memory deallocation on destruction
- Copy construction between different memory spaces
- Move semantics for zero-cost ownership transfer

Device Memory Buffer
^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../framework/memory/tests/memory_sample_tests.cpp
   :language: cpp
   :start-after: example-begin device-buffer-1
   :end-before: example-end device-buffer-1
   :dedent: 4

Pinned Host Memory Buffer
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../framework/memory/tests/memory_sample_tests.cpp
   :language: cpp
   :start-after: example-begin pinned-buffer-1
   :end-before: example-end pinned-buffer-1
   :dedent: 4

Copying Between Memory Spaces
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../framework/memory/tests/memory_sample_tests.cpp
   :language: cpp
   :start-after: example-begin buffer-copy-1
   :end-before: example-end buffer-copy-1
   :dedent: 4

Move Semantics
^^^^^^^^^^^^^^

.. literalinclude:: ../../../framework/memory/tests/memory_sample_tests.cpp
   :language: cpp
   :start-after: example-begin buffer-move-1
   :end-before: example-end buffer-move-1
   :dedent: 4

Allocators
~~~~~~~~~~

**DeviceAlloc** and **PinnedAlloc** are allocator types that provide static
methods for memory allocation and deallocation. They can be used directly or
with Buffers.

- **DeviceAlloc**: Allocates memory on the GPU using ``cudaMalloc``/``cudaFree``
- **PinnedAlloc**: Allocates pinned host memory using
  ``cudaHostAlloc``/``cudaFreeHost``

Direct Allocator Usage
^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../framework/memory/tests/memory_sample_tests.cpp
   :language: cpp
   :start-after: example-begin allocator-basics-1
   :end-before: example-end allocator-basics-1
   :dedent: 4

Smart Pointers
~~~~~~~~~~~~~~

The library provides CUDA-aware smart pointer utilities that integrate with
std::unique_ptr for automatic memory management.

Device Memory Smart Pointer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../framework/memory/tests/memory_sample_tests.cpp
   :language: cpp
   :start-after: example-begin unique-device-ptr-1
   :end-before: example-end unique-device-ptr-1
   :dedent: 4

Pinned Memory Smart Pointer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../framework/memory/tests/memory_sample_tests.cpp
   :language: cpp
   :start-after: example-begin unique-pinned-ptr-1
   :end-before: example-end unique-pinned-ptr-1
   :dedent: 4

Monotonic Allocator
~~~~~~~~~~~~~~~~~~~

**MonotonicAlloc** provides fast, sequential memory allocation from a
pre-allocated buffer. This allocator is ideal for temporary allocations with
known lifetimes, as it provides:

- Very fast allocation (just incrementing an offset)
- Guaranteed alignment for all allocations
- Bulk deallocation through reset()
- No per-allocation overhead

Basic Usage
^^^^^^^^^^^

.. literalinclude:: ../../../framework/memory/tests/memory_sample_tests.cpp
   :language: cpp
   :start-after: example-begin monotonic-alloc-1
   :end-before: example-end monotonic-alloc-1
   :dedent: 4

Resetting for Reuse
^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../framework/memory/tests/memory_sample_tests.cpp
   :language: cpp
   :start-after: example-begin monotonic-alloc-reset-1
   :end-before: example-end monotonic-alloc-reset-1
   :dedent: 4

Additional Examples
-------------------

For complete working examples with full setup and validation, see:

-  ``framework/memory/tests/memory_sample_tests.cpp`` - Documentation
   examples and basic usage patterns

API Reference
-------------

.. doxygennamespace:: framework::memory
   :content-only:
   :members:
   :undoc-members:
