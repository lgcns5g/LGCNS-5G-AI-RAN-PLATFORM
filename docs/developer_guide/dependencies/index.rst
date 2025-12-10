Dependencies
============

The ``cmake/`` directory contains helper modules for managing external dependencies.

``Dependencies.cmake`` is the main entry point that uses
`CPM.cmake <https://github.com/cpm-cmake/CPM.cmake>`_ (CMake Package Manager) to download
and configure dependencies. CPM caches downloads in ``~/.cache/CPM`` by default.

To use a custom cache location:

.. code-block:: bash

   cmake --preset clang-debug -DCPM_SOURCE_CACHE=/custom/path

Dependency Modules
------------------

- ``Acar.cmake`` : ACAR (Aerial CUDA Accelerated RAN) dependency
- ``Cuda.cmake`` : CUDA toolkit and compute architecture detection
- ``MathDx.cmake`` : NVIDIA Math libraries (cuFFTDx, cuSolverDx)
- ``Doca.cmake`` : NVIDIA DOCA (Data Center Infrastructure On a Chip Architecture)
- ``Dpdk.cmake`` : Data Plane Development Kit (DPDK)
- ``GDRCopy.cmake`` : GPUDirect RDMA copy library
- ``Nvinfer.cmake`` : TensorRT inference engine

ACAR
----

ACAR (Aerial CUDA Accelerated RAN) is managed via ``cmake/Acar.cmake`` using CPM.

**Download behavior**

By default, ACAR is automatically downloaded during CMake configuration to
``_deps/aerial_sdk-src`` in the build directory.

**Custom repository URL (optional)**

Use ``ACAR_REPO`` to point to a custom git repository:

.. code-block:: bash

   cmake --preset clang-debug -DACAR_REPO="https://github.com/NVIDIA/aerial-cuda-accelerated-ran.git"

Default repository: ``https://github.com/NVIDIA/aerial-cuda-accelerated-ran.git``

**Custom download location (optional)**

Use ``ACAR_SOURCE_DIR`` to specify a custom directory for downloading:

.. code-block:: bash

   cmake --preset clang-debug -DACAR_SOURCE_DIR=/path/to/acar

Use cases:

- Share a single source between multiple build trees
- Place the source in a specific location
- Use a pre-existing checkout

**Skip download and use existing source (optional)**

Use ``SKIP_ACAR_DOWNLOAD`` to use an existing source:

.. code-block:: bash

   cmake --preset clang-debug \
     -DSKIP_ACAR_DOWNLOAD=ON \
     -DACAR_SOURCE_DIR=/path/to/existing/acar

Note: When ``SKIP_ACAR_DOWNLOAD=ON``, ``ACAR_SOURCE_DIR`` must point to a valid source
directory containing ``CMakeLists.txt``.

