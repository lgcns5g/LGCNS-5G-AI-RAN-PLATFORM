Framework API
==================

Core infrastructure libraries providing foundational capabilities for
high-performance, real-time applications. Built with modern C++20.

Components
----------

* **Logging** (:doc:`log`) - Structured, thread-safe logging with minimal
  runtime overhead
* **Memory** (:doc:`memory`) - Buffer allocation with device-aware memory pools
  for zero-copy GPU/CPU transfers
* **Net** (:doc:`net`) - Low-latency network I/O using DPDK and DOCA
* **Pipeline** (:doc:`pipeline`) - Library for composing GPU-accelerated
  processing pipelines from reusable modules
* **Task** (:doc:`task`) - Lightweight concurrent task execution with priority
  scheduling and worker thread pool management
* **Tensor** (:doc:`tensor`) - Multi-dimensional array structures with GPU/CPU
  memory views
* **TensorRT** (:doc:`tensorrt`) - Integration layer for NVIDIA TensorRT
  engines
* **Utils** (:doc:`utils`) - Common utilities and helper functions

.. toctree::
   :maxdepth: 2
   :caption: Framework Components

   log
   memory
   net
   pipeline
   task
   tensor
   tensorrt
   utils
