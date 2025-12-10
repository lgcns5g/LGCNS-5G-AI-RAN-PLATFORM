CI Pipeline
===========

The CI (Continuous Integration) pipeline is orchestrated by ``scripts/ci_pipeline.sh`` and runs
configuration, format checks, documentation build, C++ build/install, Python package build/test,
and tests.

.. code-block:: bash

   # Help
   ./scripts/ci_pipeline.sh --help

   # Run full CI pipeline
   ./scripts/ci_pipeline.sh

   # Multiple presets in parallel (build), tests sequential
   PRESETS=gcc-debug,clang-release ./scripts/ci_pipeline.sh

   # Format checks only
   CHECK_FORMAT_ONLY=1 ./scripts/ci_pipeline.sh

   # Documentation build only
   BUILD_DOCS_ONLY=1 ./scripts/ci_pipeline.sh

   # Skip cleaning build/install directories
   SKIP_CLEAN=1 ./scripts/ci_pipeline.sh
