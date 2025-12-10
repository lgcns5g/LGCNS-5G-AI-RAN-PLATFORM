Troubleshooting
===============

Container Issues
----------------

1. Verifying container configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use a login shell (``bash -l``) to see the banner displaying your container configuration:

.. code-block:: bash

   docker exec -it aerial-framework-base-$USER bash -l

The banner shows your environment setup:

.. code-block:: text

   Entered Aerial Framework Container

   Container Configuration:
     User ID:          100963
     Group ID:         100963
     Dev workspace:    /path/to/aerial-framework
     Build workspace:  /path/to/aerial-framework/out

   GPU Configuration:
     Visible devices:  1

   Networking Configuration:
     VFIO:             /dev/vfio/vfio
     InfiniBand:       /dev/infiniband
     GDRCopy:          /dev/gdrdrv
     Hugepages:        /dev/hugepages

   SSH Configuration:
     SSH agent:        /ssh-agent

**Note:** Real-time and networking device checks in ``container/setup_container.sh`` are optional.
If devices show ``/dev/null``, those features are unavailable but the container will still function
for basic development and testing. DPDK/DOCA and fronthaul tests require proper networking hardware.

2. Container setup issues
^^^^^^^^^^^^^^^^^^^^^^^^^

If ``container/setup_container.sh`` reports warnings or errors:

.. code-block:: bash

   # Rerun setup with checks
   bash container/setup_container.sh

   # Regenerate .env file
   bash container/setup_container.sh --overwrite-env

   # Skip checks (for automation/CI)
   bash container/setup_container.sh --env-only

See :doc:`container` for detailed container setup and configuration.

3. "UID: readonly variable" or missing GROUP_ID
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set the user id and group id variables in your ``.env`` file:

.. code-block:: bash

   echo -e "\nUSER_ID=$(id -u)\nGROUP_ID=$(id -g)" >> container/.env

4. "service refers to undefined volume" or "empty section between colons"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set the ``DEV_WORKSPACE`` variable in your ``.env`` file:

.. code-block:: bash

   echo -e "\nDEV_WORKSPACE=/path/to/your/dev/workspace" >> container/.env

5. Container won't start
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   docker compose build --no-cache

6. Permission errors with files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Fix file ownership in local workspace
   sudo chown -R $USER:$USER workspace/

   # Ensure USER_ID and GROUP_ID exist in .env
   echo -e "\nUSER_ID=$(id -u)\nGROUP_ID=$(id -g)" >> container/.env

7. GPU not working
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Test GPU access
   docker compose -f container/compose.yaml run --rm aerial-framework-base nvidia-smi

   # Check your .env file settings
   grep NVIDIA_VISIBLE_DEVICES container/.env

8. Development workspace not accessible
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Check if directory exists on host
   ls -la /path/to/your/dev/workspace

   # Check your .env file settings
   grep DEV_WORKSPACE container/.env

Build Issues
------------

1. Build directory conflicts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   rm -rf out/build/<preset-name>
   cmake --preset <preset-name>

2. Clean all Python virtual environments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   cmake --build out/build/<preset-name> --target py_all_clean_venv

Runtime Issues
--------------

1. Real-time priority permission denied
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If an application fails with "Permission denied" when attempting to set real-time priority:

.. code-block:: bash

   # The build system automatically sets CAP_SYS_NICE capability
   # If you see permission errors, manually set capabilities:
   sudo setcap cap_sys_nice=eip out/build/<preset-name>/path/to/executable

**Note:** The development container is not run as privileged for security reasons. Applications
requiring real-time priority or DPDK operations use Linux capabilities (via ``cmake/CapPermissions.cmake``)
instead of requiring root or privileged containers. The build system automatically applies these
capabilities during the build process.

2. PTP clock synchronization for loopback
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptom:** Late packets, timing violations, or timing window errors in fronthaul/RU emulator
tests.

**Cause:** One or both NICs not synchronized to system clock.

**Solution:** Verify both ``phc2sys`` services are running (one for each NIC):

.. code-block:: bash

   systemctl status 'phc2sys*'

You should see two active services like this:

.. code-block:: text

   ● phc2sys1.service - running
   ● phc2sys.service - running

If only one service is running, configure the missing NIC following the
`NVIDIA Aerial CUDA-Accelerated RAN Installation Guide
<https://docs.nvidia.com/aerial/cuda-accelerated-ran/latest/install_guide/installing_tools_gh.html>`_.

**Critical:** For loopback configurations, **both NICs must be synchronized** to avoid timing
skew between the DU and RU sides. This is a common oversight that causes test failures.

3. Fronthaul integration test failure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptom:** ``fronthaul_app.integration_test`` fails with validation errors.

Common error in logs:

.. code-block:: text

   [ERROR] [fronthaul_app] [fronthaul_app_utils.cpp:380] [Stats] Validation FAILED:
   PRB mismatch - expected 76440, got 0

**Root causes:**

1. **Clock synchronization missing** - Second NIC not synchronized to system clock (most common)
2. **RU emulator not running** - Check that ru_emulator started successfully
3. **Network configuration issue** - NIC addresses or loopback connection problems
4. **FAPI capture file issue** - Missing or corrupt FAPI capture files

**Diagnostic steps:**

.. code-block:: bash

   # 1. CRITICAL: Verify BOTH phc2sys services are running (one per NIC)
   systemctl status 'phc2sys*'
   # Should show TWO active services for loopback configuration

   # 2. Check if ru_emulator is running
   ps aux | grep ru_emulator

   # 3. Verify both network interfaces are up
   ip link show aerial00
   ip link show aerial02

   # 4. Regenerate FAPI capture files
   ctest --preset <preset> -R fapi_sample.integration_test

   # 5. Run with verbose output for detailed error messages
   ctest --preset <preset> -R fronthaul_app.integration_test --verbose

See :doc:`real_time_apps` for detailed clock synchronization requirements.

4. CTest timeout for long-running integration tests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptom:** Integration tests terminate prematurely at exactly 1500 seconds when running
with large ``TEST_SLOTS`` values.

**Cause:** CTest has a default timeout of 1500 seconds.

**Solution:** Increase the timeout using the ``--timeout`` flag:

.. code-block:: bash

   # Increase timeout for long-running tests
   TEST_SLOTS=200000 ctest --preset gcc-release --timeout 3000 -R fronthaul_app.integration_test

