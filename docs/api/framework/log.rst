Log
===

Overview
--------

The Real-Time (RT) Logging library provides a high-performance, structured
logging system built on top of the `Quill logging library <https://github.com/odygrd/quill>`_.
It supports component-based and event-based logging with efficient runtime filtering,
multiple log levels, and custom type formatting.

Key Features
~~~~~~~~~~~~

-  **Component-based logging**: Organize logs by functional components
   with individual log levels
-  **Event-based logging**: Track specific events throughout your application
-  **High performance**: Built on Quill's asynchronous logging for minimal
   runtime overhead
-  **Type safety**: Compile-time enum validation and formatting
-  **Custom types**: Support for logging user-defined types with flexible
   formatting
-  **Multiple outputs**: Console, file, and rotating file logging
-  **Thread safety**: Safe for use in multi-threaded applications

Quick Start
-----------

.. _1-include-required-headers:

1. Include Required Headers
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../framework/log/samples/rt_log_sample.cpp
   :language: cpp
   :start-after: example-begin include-required-headers-1
   :end-before: example-end include-required-headers-1
   :dedent: 0

.. _2-define-your-components-and-events:

2. Define Your Components and Events
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../framework/log/tests/rt_log_sample_tests.cpp
   :language: cpp
   :start-after: example-begin simple-component-event-1
   :end-before: example-end simple-component-event-1
   :dedent: 0

.. _3-configure-the-logger:

3. Configure the Logger
~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../framework/log/tests/rt_log_sample_tests.cpp
   :language: cpp
   :start-after: example-begin simple-logger-configuration-1
   :end-before: example-end simple-logger-configuration-1
   :dedent: 4

.. _4-register-components:

4. Register Components
~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../framework/log/tests/rt_log_sample_tests.cpp
   :language: cpp
   :start-after: example-begin simple-component-registration-1
   :end-before: example-end simple-component-registration-1
   :dedent: 4

.. _5-start-logging:

5. Start Logging
~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../framework/log/tests/rt_log_sample_tests.cpp
   :language: cpp
   :start-after: example-begin simple-start-logging-1
   :end-before: example-end simple-start-logging-1
   :dedent: 4

Log Levels
----------

The framework supports multiple log levels in order of increasing severity:

============ ======================
Level        Use Case
============ ======================
``TraceL3``  Most verbose debugging
``TraceL2``  Detailed debugging
``TraceL1``  General debugging
``Debug``    Development debugging
``Info``     General information
``Notice``   Significant events
``Warn``     Warning conditions
``Error``    Error conditions
``Critical`` Critical failures
============ ======================

Log Level Filtering
~~~~~~~~~~~~~~~~~~~

Messages are only logged if their level is greater than or equal to the
configured level:

.. literalinclude:: ../../../framework/log/tests/rt_log_sample_tests.cpp
   :language: cpp
   :start-after: example-begin simple-log-level-filtering-1
   :end-before: example-end simple-log-level-filtering-1
   :dedent: 4

Basic Logging Macros
--------------------

Standard Logging
~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../framework/log/tests/rt_log_sample_tests.cpp
   :language: cpp
   :start-after: example-begin simple-standard-logging-1
   :end-before: example-end simple-standard-logging-1
   :dedent: 4

Format String Support
~~~~~~~~~~~~~~~~~~~~~

The logging macros support fmt-style format strings:

.. literalinclude:: ../../../framework/log/tests/rt_log_sample_tests.cpp
   :language: cpp
   :start-after: example-begin simple-format-strings-1
   :end-before: example-end simple-format-strings-1
   :dedent: 4

Component-Based Logging
-----------------------

Component-based logging allows you to organize log messages by functional
areas of your application and control log levels independently for each
component.

Declaring Components
~~~~~~~~~~~~~~~~~~~~

Use the ``DECLARE_LOG_COMPONENT`` macro to define your components:

.. literalinclude:: ../../../framework/log/tests/rt_log_sample_tests.cpp
   :language: cpp
   :start-after: example-begin detailed-component-declaration-1
   :end-before: example-end detailed-component-declaration-1
   :dedent: 0

Registering Components
~~~~~~~~~~~~~~~~~~~~~~

Before using component logging, register your components with desired log
levels:

.. literalinclude:: ../../../framework/log/tests/rt_log_sample_tests.cpp
   :language: cpp
   :start-after: example-begin detailed-component-registration-1
   :end-before: example-end detailed-component-registration-1
   :dedent: 4

Component Logging Macros
~~~~~~~~~~~~~~~~~~~~~~~~

Use ``RT_LOGC_*`` macros for component-based logging:

.. literalinclude:: ../../../framework/log/tests/rt_log_sample_tests.cpp
   :language: cpp
   :start-after: example-begin simple-component-logging-1
   :end-before: example-end simple-component-logging-1
   :dedent: 4

Runtime Component Level Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../framework/log/tests/rt_log_sample_tests.cpp
   :language: cpp
   :start-after: example-begin simple-runtime-component-level-1
   :end-before: example-end simple-runtime-component-level-1
   :dedent: 4

Event-Based Logging
-------------------

Event-based logging tracks specific occurrences throughout your application
lifecycle.

Declaring Events
~~~~~~~~~~~~~~~~

Define different types of events for your application:

.. literalinclude:: ../../../framework/log/tests/rt_log_sample_tests.cpp
   :language: cpp
   :start-after: example-begin event-examples-declarations-1
   :end-before: example-end event-examples-declarations-1
   :dedent: 0

Event Logging Macros
~~~~~~~~~~~~~~~~~~~~

Use ``RT_LOGE_*`` macros for event-based logging:

.. literalinclude:: ../../../framework/log/tests/rt_log_sample_tests.cpp
   :language: cpp
   :start-after: example-begin simple-event-logging-1
   :end-before: example-end simple-event-logging-1
   :dedent: 4

Combined Component and Event Logging
------------------------------------

For maximum context, combine both component and event information in your logs:

.. literalinclude:: ../../../framework/log/tests/rt_log_sample_tests.cpp
   :language: cpp
   :start-after: example-begin simple-combined-logging-1
   :end-before: example-end simple-combined-logging-1
   :dedent: 4

This produces logs with both component and event context, providing maximum traceability
for debugging and monitoring.

JSON Logging
------------

For structured logging, use JSON format macros:

.. literalinclude:: ../../../framework/log/tests/rt_log_sample_tests.cpp
   :language: cpp
   :start-after: example-begin simple-json-logging-1
   :end-before: example-end simple-json-logging-1
   :dedent: 4

Custom Type Logging
-------------------

Making Types Loggable
~~~~~~~~~~~~~~~~~~~~~

To log custom types, use the ``RT_LOGGABLE_*`` macros.

.. literalinclude:: ../../../framework/log/samples/rt_log_sample.cpp
   :language: cpp
   :start-after: example-begin product-struct-1
   :end-before: example-end product-struct-1
   :dedent: 0

Register the struct with the logging framework using ``RT_LOGGABLE_DEFERRED_FORMAT``:

.. literalinclude:: ../../../framework/log/samples/rt_log_sample.cpp
   :language: cpp
   :start-after: example-begin product-struct-2
   :end-before: example-end product-struct-2
   :dedent: 0

The above example shows a struct with value types using ``RT_LOGGABLE_DEFERRED_FORMAT``.
For types containing pointers or references, use ``RT_LOGGABLE_DIRECT_FORMAT``:

.. literalinclude:: ../../../framework/log/tests/rt_log_sample_tests.cpp
   :language: cpp
   :start-after: example-begin custom-type-user-struct-1
   :end-before: example-end custom-type-user-struct-1
   :dedent: 0

Advanced Formatting Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The formatting expressions can include complex C++ code:

.. literalinclude:: ../../../framework/log/tests/rt_log_sample_tests.cpp
   :language: cpp
   :start-after: example-begin custom-type-network-buffer-1
   :end-before: example-end custom-type-network-buffer-1
   :dedent: 0

What Can You Use in Formatting Expressions?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``RT_LOGGABLE_*`` macros accept any valid C++ expressions that can be
evaluated with the object instance (``obj``):

+-----------------------------------+--------------------------------------+
| Expression Type                   | Example                              |
+===================================+======================================+
| **Simple member access**          | ``obj.name``                         |
+-----------------------------------+--------------------------------------+
| **Conditional expressions**       | ``obj.ptr ? *obj.ptr : 0``           |
+-----------------------------------+--------------------------------------+
| **Method calls**                  | ``obj.size()``, ``obj.empty()``      |
+-----------------------------------+--------------------------------------+
| **Free function calls**           | ``std::to_string(obj.id)``           |
+-----------------------------------+--------------------------------------+
| **Mathematical operations**       | ``obj.width * obj.height``           |
+-----------------------------------+--------------------------------------+
| **Standard library algorithms**   | ``std::max(obj.a, obj.b)``           |
+-----------------------------------+--------------------------------------+
| **Type conversions**              | ``static_cast<int>(obj.value)``      |
+-----------------------------------+--------------------------------------+
| **Nested member access**          | ``obj.config.timeout.count()``       |
+-----------------------------------+--------------------------------------+
| **Container operations**          | ``obj.items.size()``,                |
|                                   | ``obj.map.empty()``                  |
+-----------------------------------+--------------------------------------+

Full Example
~~~~~~~~~~~~

Here's an example with multiple expression types:

.. literalinclude:: ../../../framework/log/tests/rt_log_sample_tests.cpp
   :language: cpp
   :start-after: example-begin custom-type-http-request-1
   :end-before: example-end custom-type-http-request-1
   :dedent: 0

Using Custom Types in Logs
~~~~~~~~~~~~~~~~~~~~~~~~~~

Once a type is registered with ``RT_LOGGABLE_*`` macros, you can log it directly:

.. literalinclude:: ../../../framework/log/samples/rt_log_sample.cpp
   :language: cpp
   :start-after: example-begin custom-type-usage-1
   :end-before: example-end custom-type-usage-1
   :dedent: 4

Choosing Format Types
~~~~~~~~~~~~~~~~~~~~~

-  ``RT_LOGGABLE_DEFERRED_FORMAT``: Use for types containing only value
   types (primitives, strings, containers of value types). The object is
   copied and formatted asynchronously in a background thread for better performance.

-  ``RT_LOGGABLE_DIRECT_FORMAT``: Use for types containing pointers or C++ references.
   The object is formatted immediately in the calling thread before the pointer could
   become invalid.

.. important::
   **Any type containing a pointer must use** ``RT_LOGGABLE_DIRECT_FORMAT`` **to avoid dangling pointer issues.**
   If you defer formatting, the pointer may point to invalid memory by the time formatting occurs.

Container Logging
-----------------

Standard C++ containers (``std::vector``, ``std::array``, etc.) are directly loggable
without any special setup:

.. literalinclude:: ../../../framework/log/tests/rt_log_sample_tests.cpp
   :language: cpp
   :start-after: example-begin container-logging-1
   :end-before: example-end container-logging-1
   :dedent: 4

Containers work seamlessly with all logging macros (``RT_LOG_*``, ``RT_LOGC_*``,
``RT_LOGE_*``, ``RT_LOGEC_*``).

Logger Configuration
--------------------

Console Logging
~~~~~~~~~~~~~~~

.. literalinclude:: ../../../framework/log/tests/rt_log_sample_tests.cpp
   :language: cpp
   :start-after: example-begin console-logging-config-1
   :end-before: example-end console-logging-config-1
   :dedent: 4

File Logging
~~~~~~~~~~~~

.. literalinclude:: ../../../framework/log/tests/rt_log_sample_tests.cpp
   :language: cpp
   :start-after: example-begin file-logging-config-1
   :end-before: example-end file-logging-config-1
   :dedent: 4

Rotating File Logging
~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../framework/log/tests/rt_log_sample_tests.cpp
   :language: cpp
   :start-after: example-begin rotating-file-logging-1
   :end-before: example-end rotating-file-logging-1
   :dedent: 4

Flushing Logs
~~~~~~~~~~~~~

Ensure all logs are written before application exit:

.. literalinclude:: ../../../framework/log/tests/rt_log_sample_tests.cpp
   :language: cpp
   :start-after: example-begin log-flushing-1
   :end-before: example-end log-flushing-1
   :dedent: 4

Best Practices
--------------

.. _component-level-management:

1. Component Level Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../framework/log/tests/rt_log_sample_tests.cpp
   :language: cpp
   :start-after: example-begin best-practice-component-level-1
   :end-before: example-end best-practice-component-level-1
   :dedent: 4

.. _meaningful-log-messages:

2. Meaningful Log Messages
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../framework/log/tests/rt_log_sample_tests.cpp
   :language: cpp
   :start-after: example-begin best-practice-meaningful-messages-1
   :end-before: example-end best-practice-meaningful-messages-1
   :dedent: 4

.. _error-context:

3. Error Context
~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../framework/log/tests/rt_log_sample_tests.cpp
   :language: cpp
   :start-after: example-begin best-practice-error-context-1
   :end-before: example-end best-practice-error-context-1
   :dedent: 4

Performance Considerations
--------------------------

.. _1-asynchronous-logging:

1. Asynchronous Logging
~~~~~~~~~~~~~~~~~~~~~~~

The framework uses Quill's asynchronous logging by default, which provides:

-  Minimal impact on application performance
-  Non-blocking log calls in most cases
-  Background thread handles actual I/O
-  Component level filtering is built into the logging macros (no manual checks needed)

.. _2-deferred-vs-direct-formatting:

2. Deferred vs Direct Formatting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../framework/log/tests/rt_log_sample_tests.cpp
   :language: cpp
   :start-after: example-begin performance-deferred-direct-formatting-1
   :end-before: example-end performance-deferred-direct-formatting-1
   :dedent: 0

.. _3-avoiding-common-pitfalls:

3. Avoiding Common Pitfalls
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../framework/log/tests/rt_log_sample_tests.cpp
   :language: cpp
   :start-after: example-begin avoiding-common-pitfalls-1
   :end-before: example-end avoiding-common-pitfalls-1
   :dedent: 4

--------------

For more examples, see:

-  ``framework/log/samples/rt_log_sample.cpp`` - Complete usage examples
-  ``framework/log/tests/rt_log_tests.cpp`` - Unit tests with various scenarios

API Reference
-------------

Complete C++ API documentation for the Real-Time Logging framework.

.. doxygennamespace:: framework::log
   :content-only:
   :members:
   :undoc-members:
