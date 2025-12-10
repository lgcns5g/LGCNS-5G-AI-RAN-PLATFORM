Test
====

Running tests
-------------

.. code-block:: bash

   ctest --preset <preset-name>

Labels
------

.. code-block:: bash

   # Run all tests
   ctest --preset clang-debug

   # Run only integration tests
   ctest --preset clang-debug -L integration

   # Run only tests that require NIC hardware
   ctest --preset clang-debug -L requires_nic

   # Exclude tests requiring hardware
   ctest --preset clang-debug -LE requires_nic

   # Run parallel tests
   ctest --preset clang-debug -L parallel

   # Run benchmark tests
   ctest --preset clang-debug -L benchmark

   # Combine labels
   ctest --preset clang-debug -L "integration.*requires_nic"

Available labels:

- ``integration``: Orchestrates multiple processes and uses generated artifacts:

  - FAPI integration tests: run ``test_mac`` + ``fapi_sample`` to generate FAPI capture files
  - Fronthaul integration tests: run ``ru_emulator`` + ``fronthaul_app`` and consume captures
  - C++ tests: consume FAPI capture files for validation

- ``requires_nic``: Requires NIC hardware (e.g., NICs for fronthaul/DPDK tests)

- ``parallel``: Tests that can run in parallel without conflicts

- ``real_time``: Tests requiring real-time system configuration (CPU isolation, PTP sync)

- ``benchmark``: Benchmark tests that measure performance

- ``nsys``: NVIDIA Nsight Systems profiling tests

- ``phy``: PHY-related tests (channel estimation, PUSCH processing, TensorRT plugins)

- ``notebook``: Jupyter notebook tests from tutorials

Python testing
--------------

Python tests can be run via CTest, which automatically sets up virtual environments and
configures the test environment:

.. code-block:: bash

   # RAN Python tests
   ctest --preset clang-debug -R py_ran_test

   # Individual pytest targets
   ctest --preset clang-debug -R py_ran_test_dmrs_plugin
   ctest --preset clang-debug -R py_ran_test_fft_plugin
   ctest --preset clang-debug -R py_ran_test_phy_numpy

For faster iteration during development, tests can also be run directly with ``uv`` and
``pytest``. See :doc:`python` for detailed workflows.

Runtime configuration
---------------------

.. code-block:: bash

   # Configure test vector for FAPI integration tests
   TEST_VECTOR=TVnr_7204_gNB_FAPI_s0.h5 ctest --preset clang-debug -R fapi_sample

   # Configure fronthaul integration test with custom slot count
   TEST_SLOTS=500 ctest --preset clang-debug -R fronthaul_app

Environment variables:

- ``TEST_CELLS``: number of cells (default: 1)
- ``TEST_VECTOR``: test vector HDF5 filename from ``ran/test_data/``
    (default: ``TVnr_7201_gNB_FAPI_s0.h5``)
- ``TEST_SLOTS``: number of slots to run (default: 100), used by fronthaul tests to control
  duration

Durations and timeouts
----------------------

- FAPI tests: slot count derived automatically from the launch pattern cycle length
- Fronthaul/PHY RAN App tests: timeout calculated automatically based on ``TEST_SLOTS``
- Indefinite runs: set ``TEST_SLOTS=0``

.. warning::

   **CTest Default Timeout:** CTest has a default timeout of **1500 seconds**. For long-running
   integration tests with large ``TEST_SLOTS`` values, you may need to increase the timeout
   using the ``--timeout`` flag.

**Example:**

.. code-block:: bash

   # Increase timeout for long-running tests
   TEST_SLOTS=200000 ctest --preset gcc-release --timeout 3000 -R fronthaul_app.integration_test

Troubleshooting
---------------

See :doc:`troubleshooting` for common test issues and solutions.

