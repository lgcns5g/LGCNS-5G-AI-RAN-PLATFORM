C++ Development
===============

Guides for C++ development: code conventions, formatting, static analysis, and sanitizers.

Building
--------

.. code-block:: bash

   # Build all targets
   cmake --build out/build/clang-debug

   # Build a specific target
   cmake --build out/build/clang-debug --target <target-name>

   # Build and run tests
   ctest --preset clang-debug

   # Run a specific test
   ctest --preset clang-debug -R <test-name>

See :doc:`build` for configuration options and presets, and :doc:`test` for test labels
and runtime configuration.

Naming conventions
------------------

.. code-block:: cpp

   namespace my_project {          // namespace: snake_case
   class MyClass {                 // class: PascalCase
   public:
     void do_something();          // method: snake_case
     int calculate_result(int input_value);  // method: snake_case, parameter: snake_case
     int public_member;            // public member: snake_case
     static const int MAX_SIZE = 100;  // constant: SCREAMING_SNAKE_CASE
   private:
     int member_variable_;         // private member: snake_case with trailing underscore
     bool is_initialized_;         // private member: snake_case with trailing underscore
   };
   enum Status {                   // enum: PascalCase
     Success, Failure, Pending     // enum values: PascalCase
   };
   int free_function(int parameter) noexcept; // function: snake_case
   } // namespace my_project

Formatting
----------

.. code-block:: bash

   # Format C++ code
   cmake --build out/build/clang-debug --target format

   # Format CMake files
   cmake --build out/build/clang-debug --target cmake-format

   # Format C++ and CMake files
   cmake --build out/build/clang-debug --target fix-format

   # Check C++ formatting (friendly output)
   cmake --build out/build/clang-debug --target check-clang-format-friendly

   # Check all formatting (C++ + CMake)
   cmake --build out/build/clang-debug --target check-format-friendly

Include header management
-------------------------

Use Include What You Use (IWYU) to optimize includes:

.. code-block:: bash

   cmake --build out/build/clang-debug --target fix-includes

Recommended workflow:

1. Run ``fix-includes`` to trim headers to only what is used
2. Run ``fix-format`` to sort includes
3. Use ``clang-format on/off`` and ``IWYU pragma: keep`` as needed to resolve conflicts

Copyright headers
-----------------

All source files must include SPDX-compliant copyright headers.

.. code-block:: bash

   # Check copyright headers
   cmake --build out/build/clang-debug --target check-copyright

   # Auto-fix copyright headers
   cmake --build out/build/clang-debug --target fix-copyright

Include guards
--------------

Header files must use standard include guards (no ``#pragma once``).

.. code-block:: bash

   # Check include guards
   cmake --build out/build/clang-debug --target check-include-guards

   # Auto-fix include guards
   cmake --build out/build/clang-debug --target fix-include-guards

Static analysis
---------------

Tools:

- ``clang-tidy``: C++ linting and static analysis
- ``cppcheck``: Additional static analysis
- ``include-what-you-use (IWYU)``: Include header optimization

Disable selectively:

.. code-block:: bash

   # Disable individual tools
   cmake --preset clang-debug -DENABLE_CLANG_TIDY=OFF

   # Or disable all
   cmake --preset clang-debug \
     -DENABLE_CLANG_TIDY=OFF \
     -DENABLE_CPPCHECK=OFF \
     -DENABLE_IWYU=OFF

Sanitizer builds
----------------

Available sanitizer presets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**AddressSanitizer + UndefinedBehaviorSanitizer + LeakSanitizer:**

.. code-block:: bash

   # Clang sanitizer builds
   cmake --preset clang-asan-debug    # or clang-asan-release
   cmake --build out/build/clang-asan-debug
   ctest --preset clang-asan-debug

   # GCC sanitizer builds
   cmake --preset gcc-asan-debug      # or gcc-asan-release
   cmake --build out/build/gcc-asan-debug
   ctest --preset gcc-asan-debug

**ThreadSanitizer:**

.. code-block:: bash

   # Clang sanitizer builds
   cmake --preset clang-tsan-debug    # or clang-tsan-release
   cmake --build out/build/clang-tsan-debug
   ctest --preset clang-tsan-debug

   # GCC sanitizer builds
   cmake --preset gcc-tsan-debug      # or gcc-tsan-release
   cmake --build out/build/gcc-tsan-debug
   ctest --preset gcc-tsan-debug

**Note:** AddressSanitizer and ThreadSanitizer cannot be used together. Use separate builds.

Manual sanitizer configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can enable individual sanitizers on any preset:

.. code-block:: bash

   cmake --preset clang-debug \
     -DENABLE_SANITIZER_ADDRESS=ON \
     -DENABLE_SANITIZER_LEAK=ON \
     -DENABLE_SANITIZER_UNDEFINED=ON

Available sanitizer options:

- ``ENABLE_SANITIZER_ADDRESS`` : Detects memory errors (use-after-free, buffer overflows)
- ``ENABLE_SANITIZER_LEAK`` : Detects memory leaks
- ``ENABLE_SANITIZER_UNDEFINED`` : Detects undefined behavior
- ``ENABLE_SANITIZER_THREAD`` : Detects data races and threading issues
- ``ENABLE_SANITIZER_MEMORY`` : Detects uninitialized memory reads (Clang only, experimental)

Coverage
--------

Available for debug builds:

.. code-block:: bash

   # Configure with coverage enabled
   cmake --preset clang-debug -DENABLE_COVERAGE=ON

   # Build and run tests
   cmake --build out/build/clang-debug
   ctest --preset clang-debug

   # Generate coverage report
   cmake --build out/build/clang-debug --target coverage

Notes:
- ``ctest`` must run before coverage generation
- HTML report: ``out/build/clang-debug/coverage/index.html``

