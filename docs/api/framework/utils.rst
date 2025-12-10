Utils
=====

Common utilities for CUDA operations, error handling, and type-safe containers.

Overview
--------

The Utils library provides utilities for building robust CUDA applications
with automatic resource management, type-safe error handling, and efficient data
structures. It simplifies common patterns and reduces boilerplate code.

Key Features
~~~~~~~~~~~~

-  **CUDA Stream Management**: RAII wrapper for automatic stream lifecycle management
-  **Error Handling**: Standard C++ error codes compatible with std::error_code
-  **Exception Classes**: Type-safe exceptions for CUDA runtime and driver API errors
-  **Error Macros**: Convenient macros for checking and throwing on CUDA errors
-  **Fixed-Size Arrays**: STL-compatible array container for host and device code

Core Concepts
-------------

CUDA Stream Management
~~~~~~~~~~~~~~~~~~~~~~~

**CudaStream** provides RAII-based automatic lifetime management for CUDA streams.
Streams are created as non-blocking and automatically synchronized and destroyed when
the object goes out of scope.

Basic Stream Usage
^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../framework/utils/tests/utils_sample_tests.cpp
   :language: cpp
   :start-after: example-begin basic-stream-1
   :end-before: example-end basic-stream-1
   :dedent: 4

Moving Streams
^^^^^^^^^^^^^^

CudaStream supports move semantics for transferring ownership:

.. literalinclude:: ../../../framework/utils/tests/utils_sample_tests.cpp
   :language: cpp
   :start-after: example-begin move-stream-1
   :end-before: example-end move-stream-1
   :dedent: 4

Error Handling
~~~~~~~~~~~~~~

The library provides standard C++ error codes through **NvErrc** enum and integration
with ``std::error_code``. This enables idiomatic C++ error handling without exceptions
when desired.

Error Code Usage
^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../framework/utils/tests/utils_sample_tests.cpp
   :language: cpp
   :start-after: example-begin error-code-1
   :end-before: example-end error-code-1
   :dedent: 4

Error Code Conversion
^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../framework/utils/tests/utils_sample_tests.cpp
   :language: cpp
   :start-after: example-begin error-conversion-1
   :end-before: example-end error-conversion-1
   :dedent: 4

Exception Classes
~~~~~~~~~~~~~~~~~

Type-safe exception classes wrap CUDA errors and provide human-readable error messages.

CUDA Runtime Exceptions
^^^^^^^^^^^^^^^^^^^^^^^^

**CudaRuntimeException** wraps CUDA runtime API errors:

.. literalinclude:: ../../../framework/utils/tests/utils_sample_tests.cpp
   :language: cpp
   :start-after: example-begin cuda-runtime-exception-1
   :end-before: example-end cuda-runtime-exception-1
   :dedent: 4

CUDA Driver Exceptions
^^^^^^^^^^^^^^^^^^^^^^^

**CudaDriverException** wraps CUDA driver API errors:

.. literalinclude:: ../../../framework/utils/tests/utils_sample_tests.cpp
   :language: cpp
   :start-after: example-begin cuda-driver-exception-1
   :end-before: example-end cuda-driver-exception-1
   :dedent: 4

Error Checking Macros
~~~~~~~~~~~~~~~~~~~~~~

Convenience macros simplify error checking and exception throwing. These macros
automatically log error information with file and line number context.

The examples above demonstrate ``AERIAL_DSP_CUDA_RUNTIME_CHECK_THROW`` and
``AERIAL_DSP_CUDA_DRIVER_CHECK_THROW`` for automatic error checking. Additional
macros provide conditional throwing and non-throwing variants:

.. literalinclude:: ../../../framework/utils/tests/utils_sample_tests.cpp
   :language: cpp
   :start-after: example-begin throw-if-1
   :end-before: example-end throw-if-1
   :dedent: 4

Array Utilities
~~~~~~~~~~~~~~~

**Arr** provides a fixed-size array container compatible with both host and device
code. It offers STL-compatible iterators and bounds-checked access.

Basic Array Usage
^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../framework/utils/tests/utils_sample_tests.cpp
   :language: cpp
   :start-after: example-begin arr-basic-1
   :end-before: example-end arr-basic-1
   :dedent: 4

Array Iteration
^^^^^^^^^^^^^^^

.. literalinclude:: ../../../framework/utils/tests/utils_sample_tests.cpp
   :language: cpp
   :start-after: example-begin arr-iterators-1
   :end-before: example-end arr-iterators-1
   :dedent: 4

Accessing Data
^^^^^^^^^^^^^^

.. literalinclude:: ../../../framework/utils/tests/utils_sample_tests.cpp
   :language: cpp
   :start-after: example-begin arr-access-1
   :end-before: example-end arr-access-1
   :dedent: 4

String Hashing
~~~~~~~~~~~~~~

**TransparentStringHash** enables heterogeneous lookup in unordered containers,
eliminating temporary string allocations when using string literals or string_view
as keys.

Basic Hash Usage
^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../framework/utils/tests/utils_sample_tests.cpp
   :language: cpp
   :start-after: example-begin transparent-hash-1
   :end-before: example-end transparent-hash-1
   :dedent: 4

Efficient Lookups
^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../framework/utils/tests/utils_sample_tests.cpp
   :language: cpp
   :start-after: example-begin transparent-hash-lookup-1
   :end-before: example-end transparent-hash-lookup-1
   :dedent: 4

TransparentStringHash requires C++20 and must be used with a transparent comparator
like ``std::equal_to<>`` to enable heterogeneous lookup.

Additional Examples
-------------------

For more examples, see ``framework/utils/tests/utils_sample_tests.cpp`` for
documentation examples and sample usage patterns.

API Reference
-------------

.. doxygennamespace:: framework::utils
   :content-only:
   :members:
   :undoc-members:

