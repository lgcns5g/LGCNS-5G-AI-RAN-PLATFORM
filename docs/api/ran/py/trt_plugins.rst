TensorRT Plugins
================

Custom TensorRT plugins for specialized operations.

Overview
--------

The TensorRT Plugins module provides custom TensorRT plugins for operations
not natively supported by TensorRT:

* **DMRS** - DMRS (Demodulation Reference Signal) generation and extraction
* **FFT** - Fast Fourier Transform operations
* **Cholesky** - Cholesky factorization and inversion
* **Plugin Management** - Global plugin registry and management

Usage
-----

The TensorRT plugins are automatically registered and available for use in
MLIR-TRT lowering. See the PUSCH channel estimation lowering tutorial for
examples of using these plugins in practice.

Python API
----------

.. automodule:: ran.trt_plugins.manager
   :members:
   :undoc-members:
   :show-inheritance:

C++ API Reference
-----------------

.. doxygennamespace:: ran::trt_plugin
   :content-only:
   :members:
   :undoc-members:

