ORAN
====

O-RAN fronthaul control plane message processing and numerology calculations
for 5G NR radio access networks.

Overview
--------

The ORAN library provides C-plane (control plane) message creation and
numerology calculations for O-RAN fronthaul interfaces. It handles 3GPP TS
38.211 compliant timing parameter calculations, C-plane message construction
with section fragmentation, and packet buffer management.

Key Features
~~~~~~~~~~~~

-  **Numerology Calculations**: Timing parameter calculations from subcarrier
   spacing (slots per subframe, slot period, symbol duration)
-  **C-Plane Messages**: Control plane message construction with automatic
   section fragmentation across MTU boundaries
-  **Buffer Abstraction**: Interface enabling both production DPDK mbufs and
   lightweight vector-based buffers for testing without DPDK initialization
-  **Packet Generation**: eCPRI packet creation with sequence ID and
   timestamp management

Core Concepts
-------------

Numerology
~~~~~~~~~~

**OranNumerology** encapsulates 3GPP timing parameters derived from
subcarrier spacing (SCS). The numerology determines slots per subframe, slot
period, and symbol duration based on the configured SCS.

Basic Numerology and Timing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../ran/runtime/oran/tests/oran_sample_tests.cpp
   :language: cpp
   :start-after: example-begin basic-numerology-1
   :end-before: example-end basic-numerology-1
   :dedent: 4

Slot Timing Calculations
^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../ran/runtime/oran/tests/oran_sample_tests.cpp
   :language: cpp
   :start-after: example-begin slot-timing-1
   :end-before: example-end slot-timing-1
   :dedent: 4

Subcarrier Spacing Conversion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../ran/runtime/oran/tests/oran_sample_tests.cpp
   :language: cpp
   :start-after: example-begin scs-conversion-1
   :end-before: example-end scs-conversion-1
   :dedent: 4

Slot Timestamps
^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../ran/runtime/oran/tests/oran_sample_tests.cpp
   :language: cpp
   :start-after: example-begin slot-timestamp-1
   :end-before: example-end slot-timestamp-1
   :dedent: 4

Timestamps are calculated as nanoseconds since System Frame Number (SFN) 0,
slot 0, enabling precise timing alignment across the radio access network.

C-Plane Messages
~~~~~~~~~~~~~~~~

**OranCPlaneMsgInfo** structures define control plane messages with sections
describing resource block allocations, beam IDs, and symbol timing. Messages
support multiple section types (0, 1, 3, 5) and optional section extensions.

Basic C-Plane Message
^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../ran/runtime/oran/tests/oran_sample_tests.cpp
   :language: cpp
   :start-after: example-begin basic-cplane-1
   :end-before: example-end basic-cplane-1
   :dedent: 4

Packet Creation
^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../ran/runtime/oran/tests/oran_sample_tests.cpp
   :language: cpp
   :start-after: example-begin cplane-packets-1
   :end-before: example-end cplane-packets-1
   :dedent: 4

The ``prepare_cplane_message`` function automatically handles MTU-based
fragmentation, splitting large messages across multiple packets when sections
exceed the configured maximum transmission unit.

Packet Counting
^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../ran/runtime/oran/tests/oran_sample_tests.cpp
   :language: cpp
   :start-after: example-begin count-packets-1
   :end-before: example-end count-packets-1
   :dedent: 4

Use ``count_cplane_packets`` to predict buffer requirements before message
preparation.

Buffer Abstraction
~~~~~~~~~~~~~~~~~~

**OranBuf** provides an abstract interface enabling packet preparation code to
work with different buffer implementations without modification. This design
allows testing with simple vector-based buffers while using high-performance
DPDK mbufs in production.

The abstraction serves two key purposes:

1. **Testing Without DPDK**: Unit tests can use ``VecBuf`` (std::vector-based)
   to verify packet construction logic without requiring DPDK initialization,
   which needs network hardware and elevated privileges
2. **Production Performance**: ``MBuf`` wraps DPDK mbufs for zero-copy packet
   transmission in production deployments

VecBuf Usage
^^^^^^^^^^^^

.. literalinclude:: ../../../../ran/runtime/oran/tests/oran_sample_tests.cpp
   :language: cpp
   :start-after: example-begin vec-buf-1
   :end-before: example-end vec-buf-1
   :dedent: 4

Accurate Send Scheduling
^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../ran/runtime/oran/tests/dpdk_buf_tests.cpp
   :language: cpp
   :start-after: example-begin mbuf-timestamp-1
   :end-before: example-end mbuf-timestamp-1
   :dedent: 4

The ``MBuf`` implementation supports timestamp metadata for accurate send
scheduling using NIC hardware timestamping. This ensures packets are
sent at precise timings required for O-RAN control plane messages to the Radio
Unit (RU), which demand nanosecond precision. The ``set_timestamp`` method
automatically configures the DPDK mbuf dynfield timestamp and sets the
TX_TIMESTAMP offload flag, enabling hardware-assisted packet transmission at
the specified time.

Additional Examples
-------------------

For more examples, see:

-  ``ran/runtime/oran/tests/oran_sample_tests.cpp`` - Documentation
   examples and basic usage patterns

API Reference
-------------

.. doxygennamespace:: ran::oran
   :content-only:
   :members:
   :undoc-members:

