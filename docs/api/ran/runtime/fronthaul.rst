Fronthaul
=========

ORAN fronthaul interface for C-Plane and U-Plane packet processing.

Overview
--------

The Fronthaul library provides ORAN (Open Radio Access Network) fronthaul
processing for 5G Radio Units (RUs). It handles both Control-Plane (C-Plane)
and User-Plane (U-Plane) packet flows with precise timing and low latency.

Key Features
~~~~~~~~~~~~

-  **C-Plane Processing**: Convert FAPI messages to ORAN C-Plane packets
   and transmit via DPDK
-  **U-Plane Processing**: Receive and reorder U-Plane packets using GPU-
   accelerated Order Kernel pipeline
-  **YAML Configuration**: Parse RU emulator configuration files
-  **Timing Management**: Calculate packet send times with nanosecond
   precision
-  **Statistics Tracking**: Monitor packets sent, errors, and throughput

Core Concepts
-------------

C-Plane Configuration
~~~~~~~~~~~~~~~~~~~~~

Configure the Fronthaul library with network settings, cell parameters,
timing windows (T1a min/max), and antenna port configuration for ORAN
C-Plane operation.

.. literalinclude:: ../../../../ran/runtime/fronthaul/tests/fronthaul_sample_tests.cpp
   :language: cpp
   :start-after: example-begin fronthaul-config-1
   :end-before: example-end fronthaul-config-1
   :dedent: 4

U-Plane Configuration
~~~~~~~~~~~~~~~~~~~~~

U-Plane configuration controls the Order Kernel pipeline for receiving and
processing U-Plane packets. The configuration includes timing windows (Ta4
early/late), packet buffer sizes, and timeout parameters.

.. literalinclude:: ../../../../ran/runtime/fronthaul/tests/fronthaul_sample_tests.cpp
   :language: cpp
   :start-after: example-begin uplane-config-1
   :end-before: example-end uplane-config-1
   :dedent: 4

Packet Timing
~~~~~~~~~~~~~

Accurate packet timing is critical for ORAN fronthaul. The library
calculates packet send times based on slot timing, T1a windows (C-Plane
timing advance window relative to data slot boundary), and TAI offset
(International Atomic Time for GPS synchronization).

.. literalinclude:: ../../../../ran/runtime/fronthaul/tests/fronthaul_sample_tests.cpp
   :language: cpp
   :start-after: example-begin packet-timing-1
   :end-before: example-end packet-timing-1
   :dedent: 4

Packet Headers
~~~~~~~~~~~~~~

ORAN C-Plane packets require proper Ethernet, VLAN, and eCPRI headers. The
library provides a helper to create packet header templates.

.. literalinclude:: ../../../../ran/runtime/fronthaul/tests/fronthaul_sample_tests.cpp
   :language: cpp
   :start-after: example-begin packet-header-1
   :end-before: example-end packet-header-1
   :dedent: 4

Statistics
~~~~~~~~~~

The Fronthaul class tracks operational statistics including packets sent,
requests sent, errors, and average packets per request.

.. literalinclude:: ../../../../ran/runtime/fronthaul/tests/fronthaul_sample_tests.cpp
   :language: cpp
   :start-after: example-begin fronthaul-stats-1
   :end-before: example-end fronthaul-stats-1
   :dedent: 4

Usage Example
-------------

The Fronthaul library requires hardware resources (DPDK, NIC, GPU) for
actual operation. The following examples are extracted from the complete
sample application.

Fronthaul Configuration
~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../../ran/runtime/fronthaul/samples/fronthaul_app_utils.cpp
   :language: cpp
   :start-after: example-begin create-config-1
   :end-before: example-end create-config-1
   :dedent: 4

Fronthaul Construction
~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../../ran/runtime/fronthaul/samples/fronthaul_app.cpp
   :language: cpp
   :start-after: example-begin fronthaul-construction-1
   :end-before: example-end fronthaul-construction-1
   :dedent: 8

Sending C-Plane Messages
~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../../ran/runtime/fronthaul/samples/fronthaul_app_utils.cpp
   :language: cpp
   :start-after: example-begin send-cplane-1
   :end-before: example-end send-cplane-1
   :dedent: 12

Processing U-Plane
~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../../ran/runtime/fronthaul/samples/fronthaul_app_utils.cpp
   :language: cpp
   :start-after: example-begin process-uplane-1
   :end-before: example-end process-uplane-1
   :dedent: 8

Accessing Statistics
~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../../ran/runtime/fronthaul/samples/fronthaul_app_utils.cpp
   :language: cpp
   :start-after: example-begin get-stats-1
   :end-before: example-end get-stats-1
   :dedent: 4

For the complete working example with task scheduling, timing
synchronization, and FAPI replay, see
``ran/runtime/fronthaul/samples/fronthaul_app.cpp``.

Additional Examples
-------------------

For more examples with executable code, see:

-  ``ran/runtime/fronthaul/tests/fronthaul_sample_tests.cpp`` -
   Configuration and utility examples
-  ``ran/runtime/fronthaul/samples/fronthaul_app.cpp`` - Complete
   application with Fronthaul construction
-  ``ran/runtime/fronthaul/samples/fronthaul_app_utils.cpp`` -
   C-Plane/U-Plane processing and statistics

API Reference
-------------

.. doxygennamespace:: ran::fronthaul
   :content-only:
   :members:
   :undoc-members:
