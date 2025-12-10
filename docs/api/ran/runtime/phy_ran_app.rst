PHY RAN App
===========

Integration application bridging FAPI (MAC-PHY) and Fronthaul (RU) interfaces.

Overview
--------

The PHY RAN App integrates FAPI message processing with Fronthaul communication and
executes the PUSCH uplink receiver pipeline:

* **FAPI Integration** - Receives messages via NVIPC from MAC layer
* **Message Adapter** - Converts FAPI messages to PUSCH processing requests
* **Fronthaul Integration** - C-Plane (DPDK) and U-Plane (DOCA) processing
* **PUSCH RX Pipeline Execution** - Runs PUSCH uplink receiver processing
* **Slot Indication** - Periodic trigger coordinating slot-based processing

System Architecture
-------------------

The integration test orchestrates three independent processes that communicate to form a complete
5G RAN stack:

.. figure:: ../../../figures/generated/comms_phy_ran_app.drawio.svg
   :align: center
   :width: 100%
   :alt: PHY RAN App System Architecture (Light)
   :class: only-light

.. figure:: ../../../figures/generated/comms_phy_ran_app_dark.drawio.svg
   :align: center
   :width: 100%
   :alt: PHY RAN App System Architecture (Dark)
   :class: only-dark

Process Interactions
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Direction
     - Communication Details
   * - **testMAC** → **phy_ran_app**
     - Sends FAPI control messages (``UL_TTI_REQUEST``, ``SLOT_RESPONSE``).
   * - **phy_ran_app** → **testMAC**
     - Sends FAPI indications (``SLOT``, ``CRC``, measurements).
   * - **phy_ran_app** → **RU Emulator**
     - Sends O-RAN C-Plane control messages via DPDK (Ethernet).
   * - **RU Emulator** → **phy_ran_app**
     - Transmits O-RAN U-Plane I/Q samples via DOCA (NIC).


Key Components
--------------

1. FapiRxHandler
~~~~~~~~~~~~~~~~
**Location:** ``ran/runtime/phy_ran_app/lib/include/phy_ran_app/fapi_rx_handler.hpp``

The central component managing FAPI message processing:

*   Owns the **nvIPC endpoint** (shared memory transport)
*   Owns a **Sample5GPipeline** instance (Message Adapter)
*   Receives FAPI messages from testMAC
*   Processes messages through Sample5GPipeline's state machine
*   Provides interfaces for slot indication, slot info, and pipeline execution

2. TX Thread: Slot Indication Timed Trigger
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**Purpose:** Periodically sends ``SLOT.indication`` FAPI messages to testMAC

**Key Details:**

*   Runs on dedicated RT core with 500μs period
*   Created using ``TimedTrigger`` with ``make_slot_indication_func()``
*   Waits for cells to be configured/started before beginning
*   Triggers at SFN boundary with GPS-based timing calculation

3. RX Thread: FAPI Message Reception
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**Purpose:** Continuously polls nvIPC and processes incoming FAPI messages from testMAC

**Key Details:**

*   Runs on dedicated RT core as a single continuous task
*   Non-blocking polling with 100μs sleep when idle
*   Forwards messages to ``Sample5GPipeline`` for state machine processing
*   Important messages: ``UL_TTI_REQUEST``, ``SLOT_RESPONSE``

4. Uplink Processing Graph
~~~~~~~~~~~~~~~~~~~~~~~~~~
**Purpose:** Three-task serial pipeline for processing uplink data

The uplink graph is triggered by the FAPI state machine when ``SLOT_RESPONSE`` is received.
Then, it executes C-Plane, U-Plane and PUSCH RX pipeline tasks, in sequence.

C-Plane Task
^^^^^^^^^^^^
*   **Core:** Network category (dedicated RT worker)
*   **Function:** ``make_process_cplane_func()``
*   **Responsibility:** Convert FAPI messages to O-RAN C-Plane messages and transmit to RU via DPDK
*   **Output:** RU is prepared to send UL data for the slot

U-Plane Task
^^^^^^^^^^^^
*   **Core:** Network category (dedicated RT worker)
*   **Dependency:** Runs after C-Plane completes
*   **Function:** ``make_process_uplane_func()``
*   **Responsibility:** Receive UL I/Q samples from NIC via DOCA, invoke CUDA kernel for reordering
*   **Output:** Reordered I/Q samples in device buffer

PUSCH RX Task
^^^^^^^^^^^^^
*   **Core:** Compute category (dedicated RT worker)
*   **Dependency:** Runs after U-Plane completes
*   **Function:** ``make_process_pusch_func()``
*   **Responsibility:** Execute PUSCH pipeline (TensorRT + CUDA) using I/Q from U-Plane
*   **Output:** CRC results and measurements sent back to testMAC via FAPI

.. figure:: ../../../figures/generated/simple_pusch_rx.drawio.svg
   :align: center
   :width: 100%
   :alt: Simple PUSCH Receiver Pipeline
   :class: only-light

.. figure:: ../../../figures/generated/simple_pusch_rx_dark.drawio.svg
   :align: center
   :width: 100%
   :alt: Simple PUSCH Receiver Pipeline (Dark)
   :class: only-dark

FAPI State Machine
------------------
**Owner:** ``Sample5GPipeline`` (inside FapiRxHandler)

Tracks the FAPI protocol state for each cell and slot combination:

*   **States:** IDLE, UL_TTI_RECEIVED, SLOT_RESPONSE_RECEIVED, GRAPH_SCHEDULED
*   **Transitions:**
    *   ``UL_TTI_REQUEST`` -> IDLE to UL_TTI_RECEIVED
    *   ``SLOT_RESPONSE`` -> UL_TTI_RECEIVED to SLOT_RESPONSE_RECEIVED
    *   Schedule uplink graph -> SLOT_RESPONSE_RECEIVED to GRAPH_SCHEDULED
    *   Graph completes -> GRAPH_SCHEDULED to IDLE

**Critical Logic:**
When the state machine transitions to ``SLOT_RESPONSE_RECEIVED``,
it invokes the ``GraphScheduleCallback``, which triggers the uplink processing graph.

Initialization Phase
--------------------

The application initialization follows these steps:

1.  **Argument Parsing & Logging Setup:** Parse CLI args and setup logging.
2.  **Signal Handlers Setup:** Register SIGINT/SIGTERM handlers.
3.  **Fronthaul Initialization:** Load YAML config, init DPDK/DOCA, calc timing params.
4.  **Create Task Schedulers:** Setup uplink (3 RT workers) and RX (1 RT worker) schedulers.
5.  **Create nvIPC Endpoint & FapiRxHandler:** Setup shared memory transport and message adapter.
6.  **Build Uplink Processing Graph:** Create and connect C-Plane, U-Plane, and PUSCH tasks.
7.  **Schedule RX Task:** Start FAPI RX polling on dedicated core.
8.  **Create Slot Indication Trigger:** Setup 500us periodic trigger.
9.  **Wait for Cells to Start:** Poll until testMAC sends START.request.
10. **Start Slot Indication Trigger:** Begin slot processing at SFN boundary.

Runtime Execution Flow
----------------------

Overview
~~~~~~~~

Once initialized, the application operates with three concurrent threads and one periodic trigger:

*   **TX Thread (Trigger):** Sends ``SLOT.indication`` every 500us.
*   **RX Thread (Polling):** Polls nvIPC, processes FAPI messages, updates state machine.
*   **Uplink Workers (3 RT Cores):** Executes C-Plane -> U-Plane -> PUSCH RX graph on-demand.

Detailed Sequence Diagram
~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: ../../../figures/generated/phy_ran_app_flow.svg
   :align: center
   :width: 100%
   :alt: PHY RAN App Sequence Diagram

Flow for Single Slot Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   Time
     │
     │  (Slot N begins)
     │
     ├──> [TX Thread] Send SLOT.indication for Slot N
     │
     │  (testMAC processes scheduling)
     │
     ├──> [RX Thread] Receive UL_TTI_REQUEST for Slot N+4
     │       └─> FAPI State: IDLE -> UL_TTI_RECEIVED
     │
     ├──> [RX Thread] Receive SLOT_RESPONSE for Slot N+4
     │       └─> FAPI State: UL_TTI_RECEIVED -> SLOT_RESPONSE_RECEIVED
     │       └─> Trigger Uplink Graph for Slot N+4
     │
     │  (Uplink graph executes on dedicated workers)
     │
     ├──> [Network Core 1] C-Plane Processing
     │       └─> Convert FAPI messages to O-RAN C-Plane
     │       └─> DPDK TX to RU Emulator
     │       └─> RU prepares to send UL data
     │
     ├──> [Network Core 2] U-Plane Processing
     │       └─> DOCA NIC receives I/Q samples from RU
     │       └─> CUDA kernel reorders samples to device buffer
     │
     ├──> [Compute Core] PUSCH RX Processing
     │       └─> TensorRT pipeline executes with I/Q buffer
     │       └─> Generate CRC and measurements
     │       └─> Send results to testMAC via nvIPC
     │
     │  (Graph completes)
     │
     ├──> [FAPI State] GRAPH_SCHEDULED -> IDLE
     │
     │  (Ready for next slot)
     v
   Time

Threading Model
---------------

The application uses dedicated RT-priority cores for deterministic execution:

.. list-table::
   :header-rows: 1
   :widths: 20 20 10 10 40

   * - Thread/Task
     - Core
     - Priority
     - Category
     - Purpose
   * - Slot Indication Trigger
     - ``--slot-indication-core``
     - 95
     - N/A
     - Send ``SLOT.indication`` every 500μs
   * - FAPI RX Task
     - ``--rx-core``
     - 95
     - Default
     - Poll nvIPC, process FAPI messages
   * - C-Plane Task
     - ``--cplane-core``
     - 95
     - Network
     - Convert FAPI to O-RAN, DPDK TX
   * - U-Plane Task
     - ``--uplane-core``
     - 95
     - Network
     - DOCA RX, CUDA reordering
   * - PUSCH RX Task
     - ``--pusch-core``
     - 95
     - Compute
     - TensorRT pipeline execution


All RT threads run at priority 95 with ``SCHED_FIFO`` scheduling policy.

API Reference
-------------

.. doxygennamespace:: ran::phy_ran_app
   :content-only:
   :members:
   :undoc-members:
