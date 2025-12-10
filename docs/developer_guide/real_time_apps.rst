Real-Time Applications
======================

Overview
--------

The Aerial Framework provides real-time applications for 5G RAN processing with deterministic
performance requirements. These applications use the Framework's real-time task system, networking
libraries, and GPU acceleration.

This section covers building, running, and testing these applications, including system
requirements and integration test orchestration.

Fronthaul App
-------------

The ``fronthaul_app`` implements the DU (Distributed Unit) side of the O-RAN fronthaul interface,
processing both C-Plane (Control Plane) and U-Plane (User Plane) traffic with GPU acceleration.

**Key features:**

*   **C-Plane transmission**: Uses DPDK for accurate packet scheduling to the RU
*   **U-Plane reception**: Uses DOCA GPUNetIO for GPU-accelerated packet processing
*   **Real-time task scheduling**: Uses timed triggers for deterministic slot-based processing

The ``fronthaul_app`` replays FAPI messages from testMAC captures, converting them to O-RAN C-Plane
packets with precise timing, while receiving and processing U-Plane uplink IQ data from the RU
emulator.

.. seealso::
   For a hands-on, interactive walkthrough of the Fronthaul App, building and running
   the integration test, refer to the :doc:`../tutorials/generated/fronthaul_tutorial` tutorial.

PHY RAN App
-----------

.. figure:: ../figures/generated/phy_app_container.drawio.svg
   :align: center
   :width: 100%
   :alt: PHY App Container Architecture
   :class: only-light

.. figure:: ../figures/generated/phy_app_container_dark.drawio.svg
   :align: center
   :width: 100%
   :alt: PHY App Container Architecture (Dark)
   :class: only-dark

The ``phy_ran_app`` is a PHY layer application that integrates the following interfaces:

*   **FAPI (MAC-PHY Interface)**: Communicates with testMAC via nvIPC shared memory
*   **O-RAN Fronthaul (DU-RU Interface)**: Connects PHY-High (DU) to PHY-Low (RU Emulator)
    via DPDK/DOCA

The purpose of the ``phy_ran_app`` is to test the functionality of the PUSCH receiver pipeline.
End-to-end component testing is managed by the test-driver, which coordinates and specifies
testing for different test/launch patterns. This component is currently co-located with testMAC.

.. seealso::
   For a hands-on, interactive walkthrough of the PHY RAN App, building and running
   the integration test, refer to the :doc:`../tutorials/generated/phy_ran_app_tutorial` tutorial.

For architectural details, process interactions, and component descriptions, please refer to the
:doc:`../../api/ran/runtime/phy_ran_app` API reference.

Prerequisites
-------------

Both applications require:

*   NVIDIA GPU with drivers installed (``nvidia-smi`` should work)
*   NICs in loopback configuration (for fronthaul/networking tests)
*   CUBB_HOME environment variable set
*   Test vector file (HDF5 format)

Building
--------

Build applications and dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To build the applications and all their dependencies:

.. code-block:: bash

   # Configure with release preset for full performance
   cmake --preset clang-release

   # Build Fronthaul App and all dependencies (ru_emulator, setcap permissions)
   cmake --build out/build/clang-release --target fronthaul_all

   # Build PHY RAN App and all dependencies (ru_emulator, testMAC, setcap permissions)
   cmake --build out/build/clang-release --target phy_ran_app

The ``_all`` targets ensure all dependencies are built including setcap permissions:

*   **ru_emulator** - RU Emulator executable (from ACAR)
*   **testMAC** - testMAC executable (from ACAR)
*   **setcap targets** - Linux capabilities for RT priority and DPDK

.. note::
   The ``ru_emulator`` and ``testMAC`` executables are part of the ACAR (Aerial CUDA Accelerated RAN)
   dependency, which is automatically managed by the build system.

Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~

**IMPORTANT:** For full performance and deterministic real-time behavior, you must:

1.  **Use a release build preset** (e.g., ``clang-release``)

    *   Debug builds have significant overhead and will cause RT deadline misses
    *   Release builds enable compiler optimizations critical for meeting timing constraints

2.  **Use isolated CPU cores and real-time system configuration**

    *   Configure CPU isolation, low-latency kernel, and PTP synchronization
    *   See the `NVIDIA Aerial CUDA-Accelerated RAN Installation Guide for Grace Hopper
        <https://docs.nvidia.com/aerial/cuda-accelerated-ran/latest/install_guide/installing_tools_gh.html>`_
        for complete real-time system setup instructions

Running the Integration Tests
------------------------------

The recommended way to run these applications is via integration tests with a **release preset**.

Fronthaul App integration test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Basic run with defaults (1 cell, 100 slots)
   ctest --preset clang-release -R fronthaul_app.integration_test

   # Indefinite run (runs until manually stopped)
   TEST_SLOTS=0 ctest --preset clang-release -R fronthaul_app.integration_test

.. note::
   ``TEST_VECTOR`` affects FAPI capture generation (testMAC), not fronthaul_app directly.
   The fronthaul test uses the generated FAPI capture files.

PHY RAN App integration test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Basic run with defaults (1 cell, 100 slots, default test vector)
   ctest --preset clang-release -R phy_ran_app.integration_test

   # Custom test vector
   TEST_VECTOR=TVnr_7204_gNB_FAPI_s0.h5 ctest --preset clang-release -R phy_ran_app.integration_test

   # Indefinite run (runs until manually stopped)
   TEST_SLOTS=0 ctest --preset clang-release -R phy_ran_app.integration_test

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

The integration tests support these environment variables:

.. list-table::
   :header-rows: 1
   :widths: 30 50 20

   * - Variable
     - Description
     - Default
   * - ``TEST_CELLS``
     - Number of cells to configure
     - 1
   * - ``TEST_SLOTS``
     - Number of slots to run (0 = indefinite)
     - 100
   * - ``TEST_VECTOR``
     - Test vector HDF5 filename from ``ran/test_data/``
     - TVnr_7201_gNB_FAPI_s0.h5

Timeout Behavior
~~~~~~~~~~~~~~~~

The integration test Python script automatically calculates and adjusts the process timeout
based on ``TEST_SLOTS``. For ``TEST_SLOTS=0`` (indefinite runs), no timeout is applied.

If the test hangs or takes too long, the Python script will kill all processes to prevent
indefinite hangs.

You can also override the timeout using ctest's ``--timeout`` flag:

.. code-block:: bash

   ctest --preset clang-release --timeout 300 -R fronthaul_app.integration_test

Expected Output and Validation
-------------------------------

When running the integration tests on a properly tuned system, you should observe:

RU Emulator Output
~~~~~~~~~~~~~~~~~~

The RU Emulator will display statistics about packet transmission/reception:

*   **TX/RX packet counts** - Total packets transmitted and received
*   **On-time packets** - Packets that arrived within timing window
*   **Late packets** - Packets that missed their timing deadline

In a properly configured system (CPU isolation + clock sync), you should see:

*   Zero or minimal late packets
*   Consistent on-time packet delivery
*   Stable TX/RX rates

Fronthaul App Output
~~~~~~~~~~~~~~~~~~~~

The Fronthaul App displays U-Plane kernel statistics at the end of execution:

.. code-block:: text

   === U-Plane Kernel Statistics ===
   Slots processed: 1000, Total PUSCH PRBs (antennas x symbols x slots): 15288000 (expected: 15288000)
   [Stats] Validation PASSED

*   **Slots processed** - Number of slots completed
*   **Total PUSCH PRBs** - PRBs received vs expected (must match for validation to pass)
*   **Validation PASSED/FAILED** - Final test result

Clock Synchronization Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For deterministic timing and zero late packets, the system requires:

1.  **NIC clocks synchronized to system clock** (which should be GPS-synchronized in production)
2.  **PTP Hardware Clock (PHC) synchronization** via ``phc2sys``

Verify clock synchronization is running:

.. code-block:: bash

   systemctl status 'phc2sys*'

Example output from a properly configured system:

.. code-block:: shell

   ● phc2sys1.service - Synchronize system clock or PTP hardware clock (PHC)
        Loaded: loaded (/lib/systemd/system/phc2sys1.service; enabled; vendor preset: enabled)
        Active: active (running) since Mon 2025-11-24 21:45:09 UTC; 5min ago
          Docs: man:phc2sys
      Main PID: 2094869 (sh)
         Tasks: 2 (limit: 146810)
        Memory: 2.1M
           CPU: 1.285s
        CGroup: /system.slice/phc2sys1.service
                ├─2094869 /bin/sh -c "/usr/sbin/phc2sys -c /dev/ptp\$(ethtool -T aerial02 | grep PTP | awk '{print \$4}') -s CLOCK_REALTIME -n 24 -O 0 -R 256 -u 256"
                └─2094874 /usr/sbin/phc2sys -c /dev/ptp2 -s CLOCK_REALTIME -n 24 -O 0 -R 256 -u 256

**Key Points:**

*   **Both phc2sys services must be active** - one for each NIC in the loopback configuration
*   One service syncs NIC PTP clock to system clock (``-s CLOCK_REALTIME``)
*   Other service syncs system clock to NIC PTP clock (``-c CLOCK_REALTIME``)
*   **Critical for loopback**: Both NICs must be synchronized to avoid timing skew between DU and RU
*   Common issue: Forgetting to configure the second NIC causes late packets and timing violations

**Without proper clock synchronization:**

*   RU Emulator will report late packets
*   Timing deadlines will be missed
*   System performance will be degraded

Test Orchestration
------------------

Fronthaul App orchestration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The integration test script ``runtime/fronthaul/tools/src/run_fronthaul_integration_test.py``
manages the lifecycle of the ``fronthaul_app``:

1.  **Detects loopback NIC pair** (DU <-> RU)
2.  **Generates runtime configs** (cells, slots)
3.  **Uses FAPI capture files** from testMAC (generated by FAPI integration test)
4.  **Launches ru_emulator** (background)
5.  **Launches fronthaul_app** (foreground, waits for exit)
6.  **Cleanup** on timeout/completion

**Process Launch Order:**

1.  **RU Emulator** → Launch first, wait for initialization
2.  **fronthaul_app** → Launch second (foreground), connects to RU and processes FAPI captures

.. note::
   The fronthaul test requires FAPI capture files. Run the FAPI integration test first to
   generate captures:

   .. code-block:: bash

      # Generate FAPI captures
      ctest --preset clang-release -R fapi_sample.integration_test

      # Then run fronthaul test
      ctest --preset clang-release -R fronthaul_app.integration_test

PHY RAN App orchestration
~~~~~~~~~~~~~~~~~~~~~~~~~

The integration test script ``runtime/phy_ran_app/samples/src/run_phy_ran_app_integration_test.py``
manages the lifecycle of the ``phy_ran_app``:

1.  **Detects loopback NIC pair** (DU <-> RU)
2.  **Generates runtime configs** (cells, slots, test vector)
3.  **Launches ru_emulator** (background)
4.  **Launches testMAC** (background)
5.  **Launches phy_ran_app** (foreground, waits for exit)
6.  **Cleanup** on timeout/completion

**Process Launch Order (all on same machine):**

The script launches processes in this specific order with initialization delays:

1.  **RU Emulator** → Launch first, wait 8 seconds for initialization
2.  **testMAC** → Launch second, wait 8 seconds for nvIPC endpoint creation
3.  **phy_ran_app** → Launch last (foreground), connects to both processes

