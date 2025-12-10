NVIDIA Aerial™ Framework
========================

A real-time signal processing framework
---------------------------------------

The Aerial Framework has been designed from the ground up to meet the needs of 3GPP Radio Access
Networks — signal processing workloads with microsecond latency requirements.
It is a single platform that unites research, testbeds, and production deployments to solve
development challenges for real-time applications.

**Use cases:** Signal processing applications with strict latency requirements
|br|
**Audience:** RAN system engineers, signal processing specialists, AI researchers
|br|
**Built with:** DOCA, DPDK, TensorRT, Python, JAX, PyTorch, C++, CUDA, and more

.. |br| raw:: html

   <br>

Features
--------

* ⚡ **Python → Real-time** - Prototype in Python and lower to high-performance GPU code.
* 🍱 **Clean separation** - Decouple signal-processing algorithm development from runtime execution.
* 🧩 **Modular pipelines** - Compose end-to-end pipelines from compiled, executable modules.
* 🔭 **Observability built-in** - Hooks for profiling and monitoring throughout development.
* 🔁 **One codebase** - Reuse components for prototyping, simulation, testing, and deployment.
* 🚀 **Modern toolchain** - Python 3.12+, C++20, CUDA 12.9, CMake, JAX, PyTorch, uv, ruff.
* 💻 **Developer-friendly** - Prototype on local machines and scale to live, production deployments.
* 📚 **Guided tutorials** - Jupyter notebooks ready to run in a Docker container.
* 🤖 **Targets 5GAdv & 6G** - Ships with an example AI-native PUSCH Pipeline. More to come.

How It Works
------------

.. image:: /figures/generated/how_works.drawio.svg
   :width: 90%
   :align: center
   :class: only-light

.. image:: /figures/generated/how_works_dark.drawio.svg
   :width: 90%
   :align: center
   :class: only-dark

.. raw:: html

   <div style="margin-bottom: 1.0rem;"></div>

The Aerial Framework combines two components:

* **Developer Tools**: Convert Python/JAX/PyTorch and C++/CUDA into pipelines of GPU-native code
* **Runtime engine**: Coordinates the execution of GPU pipelines with network interfaces

Developer Tools
~~~~~~~~~~~~~~~~

* **JAX → TensorRT** - Export JAX programs to StableHLO and lower via MLIR-TensorRT to TRT engines
* **Multi-language** - Author algorithms in JAX, PyTorch, or C++/CUDA and deploy to a uniform
  runtime
* **Modern Profiling** - Leverage NVIDIA Nsight Systems to optimize pipelines and kernels to
  μs-level
* **AI native** - Seamlessly integrate with AI Frameworks allowing end-to-end differentiability

Runtime Engine
~~~~~~~~~~~~~~

* **CUDA graphs** - GPU ops run as CUDA graphs with TensorRT integration for deterministic execution
* **Task scheduler** - Pinned, high-priority threads on isolated CPU cores enforce strict slot
  timing
* **Inline GPU networking** - DOCA GPUNetIO & GPUDirect RDMA enable zero-copy transfers NIC↔GPU
* **Production driver** - Orchestrates pipelines, memory pools & multi-cell coordination

Development → Deployment Workflow
---------------------------------

.. image:: /figures/generated/workflow.drawio.svg
   :width: 85%
   :align: center
   :class: only-light

.. image:: /figures/generated/workflow_dark.drawio.svg
   :width: 85%
   :align: center
   :class: only-dark

.. raw:: html

   <div style="margin-bottom: 1.0rem;"></div>

Aerial Framework supports two different environments depending on your use case.

**Development** - Developers prototype and optimize their algorithms in Python and then compile to
GPU native code using MLIR-TensorRT. Accessible to developers with a recent
GPU (`compute capability <https://developer.nvidia.com/cuda-gpus>`_ ≥ 8).

**Runtime** - Deployments run compiled TensorRT engines with deterministic scheduling and
high-performance networking. Testing requires a GPU, NIC, and real-time kernel to validate that
pipelines meet latency constraints using Medium Access Control (MAC) and Radio Unit (RU) emulation.

.. list-table::
   :widths: 13 78 9
   :header-rows: 1

   * - Stage
     - Description
     - Environment
   * - Prototype
     - Write and validate algorithms (NumPy/JAX/PyTorch)
     - Development
   * - Lower
     - Compile Python code into GPU executables using NVIDIA MLIR-TensorRT
     - Development
   * - Profile
     - Optimize performance using profiling tools like NVIDIA Nsight Systems
     - Development
   * - Compose
     - Assemble TensorRT engines and CUDA kernels into modular pipelines
     - Runtime
   * - Execute
     - Run with real-time task scheduling and networking
     - Runtime
   * - Validate
     - Test PHY applications using standards-compliant MAC and RU emulators
     - Runtime

This approach bridges:

* **Development Productivity** - Write in high-level languages with rich ecosystems
* **Runtime Performance** - Execute with the speed and determinism of optimized C++
* **Low Latency Requirements** - Meet strict timing and latency constraints

Quickstart
----------

`Install <https://docs.nvidia.com/aerial/framework/latest/installation/index.html>`_ the Docker container, then explore and build from source:

.. code-block:: bash

   # 1) Configure (release preset)
   cmake --preset clang-release

   # 2) Build
   cmake --build out/build/clang-release

   # 3) Install Example Python Package - 5G RAN
   cd ran/py && uv sync

Tutorials
---------

Get started & explore step-by-step `Tutorials <https://docs.nvidia.com/aerial/framework/latest/tutorials/index.html>`_.

.. list-table::
   :widths: 23 77
   :header-rows: 1

   * - Tutorial
     - Summary
   * - `Getting Started <https://docs.nvidia.com/aerial/framework/latest/tutorials/generated/getting_started.html>`_
     - Set up Docker, verify GPU access, build the project, and run tests.
   * - `PUSCH Receiver <https://docs.nvidia.com/aerial/framework/latest/tutorials/generated/pusch_receiver_tutorial.html>`_
     - Build a reference PUSCH receiver in NumPy with inner/outer receiver blocks.
   * - `MLIR-TensorRT <https://docs.nvidia.com/aerial/framework/latest/tutorials/generated/mlir_trt_tutorial.html>`_
     - Compile JAX functions (FIR filter example) to TensorRT engine(s).
   * - `Lowering PUSCH <https://docs.nvidia.com/aerial/framework/latest/tutorials/generated/pusch_receiver_lowering_tutorial.html>`_
     - Compile a PUSCH inner receiver and benchmark with NVIDIA Nsight Systems.
   * - `AI Channel Filter <https://docs.nvidia.com/aerial/framework/latest/tutorials/generated/ai_tukey_filter_training_tutorial.html>`_
     - Train a neural network to dynamically estimate channel filter parameters.
   * - `Channel Filter Design <https://docs.nvidia.com/aerial/framework/latest/tutorials/generated/pusch_channel_estimation_lowering_tutorial.html>`_
     - Design custom JAX channel estimators, lower to TensorRT & profile w/ Nsight.
   * - `Full PUSCH Pipeline <https://docs.nvidia.com/aerial/framework/latest/tutorials/generated/pipeline_tutorial.html>`_
     - Run complete pipeline mixing TensorRT engines and CUDA C++ kernels.
   * - `Fronthaul Testing <https://docs.nvidia.com/aerial/framework/latest/tutorials/generated/fronthaul_tutorial.html>`_
     - O-RAN fronthaul with DOCA GPUNetIO, task scheduling, and RU emulator.
   * - `PHY Integration <https://docs.nvidia.com/aerial/framework/latest/tutorials/generated/phy_ran_app_tutorial.html>`_
     - Full PHY application with MAC and RU emulators for integration testing.

NVIDIA AI Aerial™
-----------------

NVIDIA Aerial™ Framework is a part of `NVIDIA AI Aerial™ <https://developer.nvidia.com/industries/telecommunications/ai-aerial>`_,
a portfolio of accelerated computing platforms, software and tools to build, train, simulate, and
deploy AI-native wireless networks.
Learn more in `AI Aerial™ Documentation <https://docs.nvidia.com/aerial/index.html>`_.

The following AI Aerial™ software is available as open source:

* `NVIDIA Aerial™ Framework <https://github.com/NVIDIA/aerial-framework>`_
* `NVIDIA Aerial™ CUDA-Accelerated RAN <https://github.com/NVIDIA/aerial-cuda-accelerated-ran>`_

Visit `NVIDIA 6G Developer Program <https://developer.nvidia.com/6g-program>`_ for software releases,
events and technical training for AI Aerial™.

License
-------

Aerial Framework is licensed under the **Apache 2.0** license. See `LICENSE <https://github.com/NVIDIA/aerial-framework/blob/main/LICENSE>`_ for details.
Some dependencies may have different licenses. See `ATTRIBUTION <https://github.com/NVIDIA/aerial-framework/blob/main/ATTRIBUTION.md>`_ for
third-party attributions in the source repository.
