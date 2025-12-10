API Reference
=============

.. toctree::
   :maxdepth: 2

   framework/index
   ran/index

Overview
--------

The Aerial Framework provides two primary API layers:

**Framework Core** (:doc:`framework/index`)
  Foundational C++ libraries providing logging, memory management, networking,
  tensor operations, pipeline and module management, and concurrent task execution.
  These components form the common framework used by higher-level RAN applications.

**RAN API** (:doc:`ran/index`)
  Radio Access Network implementations further split into two parts:

  * **Python API** (:doc:`ran/py/index`) - Python implementations for
    rapid prototyping, algorithm validation, and PHY toolchain
    for lowering high-level Python code to TensorRT engines for real-time
    execution and performance profiling
  * **Runtime API** (:doc:`ran/runtime/index`) - High-performance C++
    components for real-time PHY processing, including SCF FAPI, O-RAN
    fronthaul, PUSCH/LDPC encoding, and GPU-accelerated signal processing chains

The framework follows a layered architecture where the Framework Core provides
reusable infrastructure components, and RAN API builds on this foundation to
deliver specialized RAN processing capabilities. The RAN Runtime API leverages
Framework components like pipelines and tensors for signal processing chains,
tasks for real-time execution, and networking for fronthaul communication.
