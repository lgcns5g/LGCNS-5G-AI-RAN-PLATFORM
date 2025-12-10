Python API
==========

Python implementations of PHY algorithms and AI models for rapid prototyping,
algorithm validation, and integration testing. These components can then be
lowered into TensorRT engines for real-time execution in the
:doc:`../runtime/index`.

.. note::
   The Python API may be subject to change as more PHY channels are added
   in future releases.

Components
----------

* **Constants** (:doc:`constants`) - 5G NR physical layer constants and
  parameters
* **Datasets** (:doc:`datasets`) - Channel simulation dataset utilities using
  Sionna
* **MLIR-TensorRT Wrapper** (:doc:`mlir_trt_wrapper`) - Python wrapper for
  MLIR-TensorRT compiler and execution
* **PHY JAX** (:doc:`phy/jax/index`) - JAX-based PHY implementations including
  AI integration and TensorRT lowering
* **PHY NumPy** (:doc:`phy/numpy/index`) - Pure NumPy reference PHY
  implementations
* **TensorRT Plugins** (:doc:`trt_plugins`) - Custom TensorRT plugins for
  specialized operations (DMRS, FFT, Cholesky)
* **Types** (:doc:`types`) - Type definitions and data structure aliases
* **Utils** (:doc:`ran_utils`) - Helper utilities for configuration, I/O, and
  timing

.. toctree::
   :maxdepth: 2
   :caption: Python Components

   constants
   datasets
   mlir_trt_wrapper
   phy/jax/index
   phy/numpy/index
   trt_plugins
   types
   ran_utils
