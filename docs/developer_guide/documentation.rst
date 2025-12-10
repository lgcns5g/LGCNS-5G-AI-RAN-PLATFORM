Documentation
=============

Overview
--------

The project uses Sphinx for documentation generation with integrated Doxygen C++ API docs
(via Breathe).

Building docs
-------------

CMake documentation targets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The build system provides several targets for documentation building and validation.
Many of these targets use ``scripts/setup_python_env.py`` internally.

**Main documentation targets:**

.. code-block:: bash

   # Build all documentation (Sphinx + Doxygen + linkcheck)
   cmake --build out/build/clang-debug --target docs

   # Build Sphinx HTML documentation only
   cmake --build out/build/clang-debug --target sphinx-docs

   # Build Doxygen C++ API documentation only
   cmake --build out/build/clang-debug --target doxygen-docs

   # Check documentation links (detects broken URLs)
   cmake --build out/build/clang-debug --target sphinx-linkcheck

**Documentation validation:**

.. code-block:: bash

   # Check all C++ docstrings (Doxygen compliance)
   cmake --build out/build/clang-debug --target check_all_docstrings

   # Check individual components
   cmake --build out/build/clang-debug --target framework_docstring_check
   cmake --build out/build/clang-debug --target ran_runtime_docstring_check
   cmake --build out/build/clang-debug --target ran_py_docstring_check

   # Lint RST documentation with doc8
   cmake --build out/build/clang-debug --target py_docs_doc8

   # Verify API documentation samples compile
   cmake --build out/build/clang-debug --target py_docs_samples

**Python environment setup:**

.. code-block:: bash

   # Setup docs Python environment
   cmake --build out/build/clang-debug --target py_docs_setup

   # Run all docs checks and builds
   cmake --build out/build/clang-debug --target py_docs_all

Outputs:

- Sphinx site: ``out/build/clang-debug/docs/sphinx/index.html``
- C++ API: ``out/build/clang-debug/docs/doxygen/html/index.html``

**Note:** Enable documentation building with ``-DBUILD_DOCS=ON``.
Enable docstring enforcement with ``-DENFORCE_DOCSTRINGS=ON``.

Serving documentation
---------------------

After building, you can serve the Sphinx site locally:

.. code-block:: bash

   cd out/build/clang-debug/docs/sphinx && python3 -m http.server 8000

Then open ``http://<your-host>:8000`` in a browser.

Documentation linting with setup_python_env.py
----------------------------------------------

The ``scripts/setup_python_env.py`` script provides direct access to documentation tools:

.. code-block:: bash

   # Lint RST documentation with doc8
   uv run scripts/setup_python_env.py doc8 docs

   # Build Sphinx docs (called by cmake target)
   uv run scripts/setup_python_env.py sphinx_docs docs --build-dir out/build/clang-debug

   # Run Sphinx linkcheck (called by cmake target)
   uv run scripts/setup_python_env.py sphinx_linkcheck docs --build-dir out/build/clang-debug

   # Convert notebooks (called by cmake target)
   uv run scripts/setup_python_env.py jupytext_convert docs

**Note:** The CMake targets automatically use these commands.

Tutorial notebooks
------------------

Tutorial conversion targets are part of the top-level ``docs`` target, but can be
run independently:

.. code-block:: bash

   # Convert Python tutorials to Jupyter notebooks
   cmake --build out/build/clang-debug --target py_notebook_convert

   # Test notebooks execution
   ctest --preset clang-debug -R py_notebook

Generated notebooks are placed in ``docs/tutorials/generated/``.

The conversion uses ``scripts/setup_python_env.py jupytext_convert`` to convert Python
files with special comment markers into ``.ipynb`` files.

Figure management
-----------------

Figure targets
^^^^^^^^^^^^^^

The build system provides targets for figure management and validation:

.. code-block:: bash

   # Clear metadata from XML/SVG files (reduces git diff noise)
   cmake --build out/build/clang-debug --target clear-figure-metadata

   # Generate SVG diagrams from Mermaid source files
   cmake --build out/build/clang-debug --target generate-mermaid-svgs

These targets use:

- ``cmake/helpers/clear_metadata.py`` - Strips host-specific metadata from figures
- ``scripts/generate_mermaid_svgs.py`` - Converts ``.mermaid`` files to ``.svg``

Draw.io figures
^^^^^^^^^^^^^^^

Source figures are under ``docs/figures/src/*.drawio.xml`` and should be exported to SVG in
``docs/figures/generated/*.drawio.svg``.

Steps (Draw.io):

1. Open the ``.drawio.xml`` file
2. File → Export as → SVG
3. Save to ``docs/figures/generated/`` using the same base name with ``.drawio.svg``

Why SVG:

- Vector format for crisp rendering at any scale
- Smaller size vs raster formats for technical docs

Validation:

``scripts/setup_python_env.py`` validates that each ``.drawio.xml`` has a matching exported
``.drawio.svg``.

Metadata cleanup
""""""""""""""""

Draw.io exports often include host-specific metadata (e.g. ``host="..."`` or ``agent="..."``)
that creates unnecessary noise in git diffs. The ``clear-figure-metadata`` target (documented above)
strips this metadata from all source XML and generated SVG files.

Mermaid diagrams
""""""""""""""""

While Mermaid diagrams can be embedded directly into RST files for simple use cases,
saving them as ``.mermaid`` files in ``docs/figures/src/`` is recommended for those referenced
in multiple places. This approach also makes it easier to control color schemes
(e.g. font and background colors).

The ``generate-mermaid-svgs`` target (documented above) generates ``.svg`` files in
``docs/figures/generated/`` which can be included using the ``.. figure::`` directive.

CMake integration
-----------------

The documentation build system integrates with several CMake modules:

- ``cmake/Sphinx.cmake`` - Sphinx HTML documentation and linkcheck targets
- ``cmake/Doxygen.cmake`` - C++ API documentation and docstring validation
- ``cmake/Mermaid.cmake`` - Mermaid diagram generation
- ``cmake/ImageMetadata.cmake`` - Figure metadata cleanup
- ``cmake/Python.cmake`` - Python environment setup and notebook conversion

See ``docs/CMakeLists.txt`` and ``docs/tutorials/CMakeLists.txt`` for the full build pipeline.

