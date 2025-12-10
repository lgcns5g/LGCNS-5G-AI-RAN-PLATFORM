Python Development
==================

Detailed workflows for Python development in the Aerial Framework.

RAN Python Package
------------------

Python implementation of RAN PHY including lowering toolchain based on MLIR-TensorRT.

Project Structure
^^^^^^^^^^^^^^^^^

.. code-block:: text

   ran/py/src/ran/
   ├── constants/        # 5G NR physical layer constants
   ├── datasets/         # Channel simulation datasets (Sionna CDL)
   ├── mlir_trt_wrapper/ # MLIR-TensorRT compiler wrapper
   ├── phy/
   │   ├── jax/          # JAX PHY implementation (differentiable, TRT lowering)
   │   │   └── pusch/    # PUSCH inner receiver (channel est., EQ, soft demapper)
   │   └── numpy/        # NumPy PHY reference implementation
   │       └── pusch/    # PUSCH receiver (DMRS, channel est., EQ, LDPC, CRC)
   ├── trt_plugins/      # Custom TensorRT plugins (DMRS, FFT, Cholesky)
   ├── types/            # Type definitions
   └── utils/            # Shared helpers (HDF5 I/O, config, timing)

Package Management
^^^^^^^^^^^^^^^^^^

Python packages are managed using ``uv`` and configured via ``pyproject.toml``:

- **Package manager**: ``uv`` handles dependency resolution, virtual environment creation, and
  package installation
- **Configuration**: ``pyproject.toml`` defines dependencies, extras (``dev``, ``mlir_trt_wheels``,
  ``datasets``), and project metadata
- **Layout**: Uses ``src`` layout with editable installs (``pip install -e .``)
- **Virtual environments**: Created automatically in package directories (e.g., ``ran/py/.venv``)
- **CMake integration**: ``cmake/Python.cmake`` provides CMake functions that invoke
  ``scripts/setup_python_env.py``, which uses ``uv`` under the hood

**Extras defined in pyproject.toml:**

- ``dev``: Development tools (ruff, mypy, pytest, etc.)
- ``mlir_trt_wheels``: MLIR-TensorRT Python wheels
- ``datasets``: Dataset generation tools
- ``phy_jax``: JAX and related dependencies for PHY processing

Development Workflows
^^^^^^^^^^^^^^^^^^^^^

This project uses ``ruff`` for code formatting and linting, ``mypy`` for static type
checking, and ``doc8`` for documentation linting.

Option 1: Using CMake Targets (Recommended)
""""""""""""""""""""""""""""""""""""""""""""

Replace ``out/build/clang-debug`` with your actual CMake build directory. From the project root:

.. code-block:: bash

   # Setup environment
   cmake --build out/build/clang-debug --target py_ran_setup

   # Run tests
   cmake --build out/build/clang-debug --target py_ran_test

   # === CODE FORMATTING ===
   cmake --build out/build/clang-debug --target py_ran_check_format        # Check formatting
   cmake --build out/build/clang-debug --target py_ran_fix_format          # Auto-format code

   # === LINTING ===
   cmake --build out/build/clang-debug --target py_ran_lint                # Comprehensive linting
   cmake --build out/build/clang-debug --target py_ran_fix_lint            # Auto-fix linting issues

   # === INDIVIDUAL RUFF OPERATIONS ===
   cmake --build out/build/clang-debug --target py_ran_ruff_check          # Check linting (ruff)
   cmake --build out/build/clang-debug --target py_ran_ruff_fix            # Auto-fix linting (ruff)

   # === TYPE CHECKING ===
   cmake --build out/build/clang-debug --target py_ran_mypy                # Static type checking

   # === WHEEL OPERATIONS ===
   cmake --build out/build/clang-debug --target py_ran_wheel_build         # Build wheels
   cmake --build out/build/clang-debug --target py_ran_wheel_install       # Install wheels
   cmake --build out/build/clang-debug --target py_ran_wheel_test          # Test wheels

   # === COMPLETE PIPELINE ===
   cmake --build out/build/clang-debug --target py_ran_all                 # Run everything

   # === AGGREGATE TARGETS (ALL PACKAGES) ===
   cmake --build out/build/clang-debug --target py_all_setup               # Setup all packages
   cmake --build out/build/clang-debug --target py_all_clean_venv          # Remove all .venv
   cmake --build out/build/clang-debug --target py_all_check_format        # Check formatting (all)
   cmake --build out/build/clang-debug --target py_all_fix_format          # Fix formatting (all)
   cmake --build out/build/clang-debug --target py_all_lint                # Lint (all)
   cmake --build out/build/clang-debug --target py_all_fix_lint            # Fix lint (all)
   cmake --build out/build/clang-debug --target py_all_mypy                # Type check (all)

Option 2: Using Setup Script Directly
""""""""""""""""""""""""""""""""""""""

.. code-block:: bash

   # Setup environment
   uv run scripts/setup_python_env.py setup ran/py

   # Run tests
   uv run scripts/setup_python_env.py test ran/py

   # === CODE FORMATTING ===
   uv run scripts/setup_python_env.py check_format ran/py   # Check formatting
   uv run scripts/setup_python_env.py fix_format ran/py     # Fix formatting

   # === LINTING ===
   uv run scripts/setup_python_env.py lint ran/py          # Comprehensive linting
   uv run scripts/setup_python_env.py fix_lint ran/py      # Auto-fix linting

   # === INDIVIDUAL RUFF OPERATIONS ===
   uv run scripts/setup_python_env.py ruff_check ran/py    # Check linting
   uv run scripts/setup_python_env.py ruff_fix ran/py      # Auto-fix linting

   # === TYPE CHECKING ===
   uv run scripts/setup_python_env.py mypy ran/py          # Static type checking

   # === WHEEL OPERATIONS ===
   uv run scripts/setup_python_env.py wheel_build ran/py   # Build wheels
   uv run scripts/setup_python_env.py wheel_install ran/py # Install wheels
   uv run scripts/setup_python_env.py wheel_test ran/py    # Test wheels

   # === COMPLETE PIPELINE ===
   uv run scripts/setup_python_env.py all ran/py          # Run everything

   # === VERBOSE OUTPUT ===
   uv run scripts/setup_python_env.py test ran/py -v      # Enable verbose output

Development Tools
^^^^^^^^^^^^^^^^^

**Static analysis and formatting:**

- ``ruff`` — Fast Python linter and formatter

  - ``ruff check``: Linting (finds issues)
  - ``ruff check --fix``: Auto-fix linting issues
  - ``ruff format``: Code formatting
  - Configured in ``pyproject.toml`` under ``[tool.ruff]``

- ``mypy`` — Static type checker

  - Verifies type hints and finds type errors
  - Configured in ``pyproject.toml`` under ``[tool.mypy]``

**Testing and documentation:**

- ``pytest`` — Test framework with coverage support
- ``doc8`` — Documentation linting for RST files

Cleaning Virtual Environments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Python packages keep a local virtual environment under their source tree
(e.g., ``ran/py/.venv``). To reset the environment:

.. code-block:: bash

   rm -rf ran/py/.venv
   cmake --build out/build/clang-debug --target py_ran_setup

If you need to reset multiple packages, remove each package's ``.venv`` and
rerun the corresponding setup target.

MLIR-TensorRT
-------------

The RAN Python package includes TensorRT engine compilation and custom plugins for PHY
processing. The CMake option ``ENABLE_MLIR_TRT`` is enabled by default.

The MLIR-TensorRT compiler and Python wheels are downloaded from public GitHub releases
at build time. Downloads are cached locally in ``ran/py/mlir-trt-downloads/``.

Setup
^^^^^

When ``ENABLE_MLIR_TRT=ON``:

- **py_ran_mlir_trt_setup**: Downloads MLIR-TensorRT compiler tarball and Python wheels from
  public GitHub releases to ``ran/py/mlir-trt-downloads/``
- **py_ran_setup**: Creates virtual environment in ``ran/py/.venv`` and installs all dependencies
  including MLIR-TensorRT wheels from ``pyproject.toml`` (includes both ``dev`` and
  ``mlir_trt_wheels`` extras)

Build order: ``py_ran_mlir_trt_setup`` runs first to download artifacts, then
``py_ran_setup`` creates the venv and installs everything.

You don't need to manually create or activate virtual environments - CMake
handles everything.

Running Tests with CTest
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Run all C++ tests (automatically runs prerequisite Python tests)
   ctest --preset <preset> -R ran.phy_test

   # Run all C++ benchmarks
   ctest --preset <preset> -R ran.phy_bench

   # Run all nsys profiling
   ctest --preset <preset> -R ran.phy_nsys

   # Run all PHY tests, benchmarks, and nsys
   ctest --preset <preset> -R ran.phy

TRT Engine Generation and Test Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

C++ tests and benchmarks depend on TensorRT engines generated by Python tests. The build system
(``ran/py/CMakeLists.txt``) uses CTest fixtures to enforce proper execution order:

**Dependency chain:**

1. **Python tests** (``py_ran_test_*``) generate ``.engine`` files via MLIR-TensorRT compilation
2. **CTest fixtures** ensure Python tests run before C++ tests
3. **C++ tests** (``*_plugin_tests``) load and validate the generated engines
4. **C++ benchmarks** (``*_plugin_bench``) profile the engines with Google Benchmark
5. **Nsys profiling** (``phy_nsys.*``) generates performance traces

**Example from ran/py/CMakeLists.txt:**

.. code-block:: cmake

   # Python test generates engines and sets up fixture
   add_trt_engine_python_test(py_ran_test_dmrs_plugin ${PLUGIN_TEST_DIR}/dmrs dmrs_trt_engines)

   # C++ test depends on fixture (runs after Python test)
   set_tests_properties(ran.phy_test.dmrs_plugin_tests
                        PROPERTIES FIXTURES_REQUIRED "dmrs_trt_engines")

**Environment configuration (.env.python):**

CMake generates ``.env.python`` in each build directory (preset-specific) containing:

- ``ENABLE_MLIR_TRT=ON`` - Enables MLIR_TRT features in Python code
- ``MLIR_TRT_COMPILER_PATH`` - Path to MLIR-TensorRT compiler executable
- ``RAN_TRT_ENGINE_PATH`` - Directory for generated TensorRT engine files
- ``RAN_TRT_PLUGIN_DSO_PATH`` - Path to custom TensorRT plugin library

CTest passes the path via ``RAN_ENV_PYTHON_FILE`` (priority 1) and syncs to source directory as
fallback (priority 2). For manual pytest runs, use ``RAN_ENV_PYTHON_FILE`` or run
``sync_env_python`` target.

Each preset has isolated ``.env.python`` in its build directory.

Running Individual Tests/Benchmarks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Run specific C++ test
   ctest --preset <preset> -R ran.phy_test.dmrs_plugin_tests

   # Run specific benchmark
   ctest --preset <preset> -R ran.phy_bench.fft_plugin_bench

   # Run specific nsys profiling
   ctest --preset <preset> -R ran.phy_nsys.sample_plugin_bench

Running Python Tests Directly
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Setup venv (only needed once)
   cmake --build <build_dir> -t py_ran_setup

   # Sync .env.python to source directory
   cmake --build <build_dir> --target sync_env_python

   # Run tests with uv
   cd ran/py
   uv run pytest                           # Run all tests
   uv run pytest tests/trt_plugins/fft/    # Run specific directory
   uv run pytest tests/phy/numpy/          # Run NumPy PHY tests
   uv run pytest -k test_dmrs              # Run tests matching pattern

**Note:** The ``.env.python`` file is required for TRT-related tests as it provides paths to
MLIR-TensorRT compiler, engine directory, and plugin libraries.

