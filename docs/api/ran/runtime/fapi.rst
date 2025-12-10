FAPI
====

5G Small Cell Forum (SCF) FAPI interface implementation for PHY-MAC communication
with message capture and replay capabilities.

Overview
--------

The FAPI module provides a complete implementation of the 5G NR FAPI
interface for communication between PHY and MAC layers. It includes state
machine management, message routing over NVIPC transport, and file-based
capture/replay for testing.

Key Features
~~~~~~~~~~~~

-  **State Machine Management**: Per-cell state tracking (Idle →
   Configured → Running → Stopped)
-  **NVIPC Transport**: High-performance shared memory message passing with
   RAII lifecycle management
-  **Message Routing**: Automatic dispatch of CONFIG, START, STOP requests
   with response generation
-  **File Capture/Replay**: Record and replay FAPI message sequences for
   deterministic testing
-  **Callback System**: User-configurable handlers for UL_TTI_REQUEST,
   DL_TTI_REQUEST, and other messages
-  **Thread Safety**: Lock-free slot management and thread-safe message
   operations

Core Concepts
-------------

FAPI State Machine
~~~~~~~~~~~~~~~~~~

**FapiState** manages the complete FAPI message lifecycle over NVIPC
transport. Each cell maintains its own state that transitions through the
FAPI state machine based on received messages.

The NVIPC configuration can be provided either as a file path or inline
YAML string. The state machine initializes transport endpoints and allocates
per-cell state structures.

State Initialization
^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../ran/runtime/fapi/tests/fapi_sample_tests.cpp
   :language: cpp
   :start-after: example-begin state-init-1
   :end-before: example-end state-init-1
   :dedent: 4

Slot Management
^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../ran/runtime/fapi/tests/fapi_sample_tests.cpp
   :language: cpp
   :start-after: example-begin slot-management-1
   :end-before: example-end slot-management-1
   :dedent: 4

Slot management uses lock-free atomics for thread-safe operation. The slot
counter automatically wraps around at configured SFN and slot limits.

Message Callbacks
~~~~~~~~~~~~~~~~~

**FapiState** provides callbacks for handling different message types.
Callbacks are invoked during message processing and can be used to forward
messages to application logic.

Basic Message Callbacks
^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../ran/runtime/fapi/tests/fapi_sample_tests.cpp
   :language: cpp
   :start-after: example-begin message-callbacks-1
   :end-before: example-end message-callbacks-1
   :dedent: 4

The ``on_message`` callback captures all messages before routing to
specific handlers, useful for logging or recording.

Configuration Callbacks
^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../ran/runtime/fapi/tests/fapi_sample_tests.cpp
   :language: cpp
   :start-after: example-begin config-callbacks-1
   :end-before: example-end config-callbacks-1
   :dedent: 4

Configuration callbacks return error codes to indicate validation results.
The state machine automatically sends appropriate responses or error
indications based on callback return values.

File Capture and Replay
~~~~~~~~~~~~~~~~~~~~~~~~

FAPI messages can be captured to binary files during execution and replayed
later for deterministic testing without NVIPC dependencies. This simplifies
unit testing of downstream modules that depend on FAPI messages, such as PHY
processing pipelines and ORAN C-plane message preparation.

Capturing Messages
^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../ran/runtime/fapi/tests/fapi_sample_tests.cpp
   :language: cpp
   :start-after: example-begin file-writer-1
   :end-before: example-end file-writer-1
   :dedent: 4

**FapiFileWriter** buffers messages in memory during capture and writes
them in a single operation to a binary file. The file format includes a
header with message count and per-message metadata.

Reading Messages
^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../ran/runtime/fapi/tests/fapi_sample_tests.cpp
   :language: cpp
   :start-after: example-begin file-reader-1
   :end-before: example-end file-reader-1
   :dedent: 4

**FapiFileReader** provides sequential or bulk access to captured messages:

.. literalinclude:: ../../../../ran/runtime/fapi/tests/fapi_sample_tests.cpp
   :language: cpp
   :start-after: example-begin file-reader-read-all-1
   :end-before: example-end file-reader-read-all-1
   :dedent: 4

Replaying Messages
^^^^^^^^^^^^^^^^^^

**FapiFileReplay** provides timed replay of UL_TTI_REQUEST messages with
automatic SFN/slot field updates:

.. literalinclude:: ../../../../ran/runtime/fapi/tests/fapi_sample_tests.cpp
   :language: cpp
   :start-after: example-begin file-replay-1
   :end-before: example-end file-replay-1
   :dedent: 4

The replay system maintains per-cell request indices and automatically
updates timing fields in FAPI structures to match the current replay
position. This enables deterministic testing with real capture data.

Additional Examples
-------------------

For complete working examples, see:

-  ``ran/runtime/fapi/tests/fapi_sample_tests.cpp`` - Documentation
   examples and basic usage patterns
-  ``ran/runtime/fapi/tests/fapi_state_tests.cpp`` - State machine
   tests with message processing
-  ``ran/runtime/fapi/tests/fapi_file_io_tests.cpp`` - File capture
   and reader tests
-  ``ran/runtime/fapi/tests/fapi_file_replay_tests.cpp`` - Replay
   system tests with timing validation

API Reference
-------------

.. doxygennamespace:: ran::fapi
   :content-only:
   :members:
   :undoc-members:
