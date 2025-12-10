Build
=====

Basic build steps
^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # 1. Configure
   cmake --preset <preset-name>

   # 2. Build
   cmake --build out/build/<preset-name>

Building individual targets
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # List all available targets
   cmake --build out/build/<preset-name> --target help

   # Build a specific target
   cmake --build out/build/<preset-name> --target <target-name>

   # Examples
   cmake --build out/build/clang-debug --target fapi_sample
   cmake --build out/build/clang-debug --target fronthaul_app
   cmake --build out/build/clang-debug --target phy_ran_app

Available presets
^^^^^^^^^^^^^^^^^

The main build presets are ``debug`` and ``release`` for the ``clang`` compiler:

- ``clang-debug`` : Debug build with debug symbols and assertions
- ``clang-release`` : Release build with optimizations suitable for deployment

Additional presets are available for static analysis and sanitizers, and the ``gcc`` compiler.

See the top-level ``CMakePresets.json`` for preset definitions. List all presets:

.. code-block:: bash

  cmake --list-presets

Configuration options
^^^^^^^^^^^^^^^^^^^^^

Common CMake options can be passed during configuration:

.. code-block:: bash

   # Disable maintainer mode (relaxes strict checks)
   cmake --preset clang-debug -DMAINTAINER_MODE=OFF

   # Enable coverage reporting
   cmake --preset clang-debug -DENABLE_COVERAGE=ON

   # Disable static analysis (faster builds)
   cmake --preset clang-debug -DENABLE_CLANG_TIDY=OFF

Common options:

- ``MAINTAINER_MODE`` : Enable strict checks and warnings as errors
- ``ENABLE_COVERAGE`` : Enable code coverage reporting
- ``ENABLE_CLANG_TIDY`` : Enable clang-tidy static analysis
- ``ENABLE_CPPCHECK`` : Enable cppcheck static analysis
- ``ENABLE_IWYU`` : Enable include-what-you-use analysis
- ``ENABLE_MLIR_TRT`` : Enable MLIR-TensorRT for PHY JAX lowering

See ``cmake/ProjectOptions.cmake`` for the complete list of configuration options.

CMake helper modules
^^^^^^^^^^^^^^^^^^^^

The ``cmake/`` directory contains helper modules for managing dependencies and build
configuration. See :doc:`dependencies/index` for dependency management modules.

**Build Utilities:**

- ``RanTrtPlugin.cmake`` : Custom TensorRT plugins for RAN processing
- ``CapPermissions.cmake`` : Linux capabilities for non-privileged execution (RT priority, DPDK)
- ``Tests.cmake`` : Test configuration and CTest integration
- ``Python.cmake`` : Python environment and virtual environment management

See the individual CMake files in ``cmake/`` for detailed documentation.

Troubleshooting
^^^^^^^^^^^^^^^

See :doc:`troubleshooting` for common build issues and solutions.

