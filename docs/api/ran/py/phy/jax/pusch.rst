PUSCH
=====

JAX implementations of Physical Uplink Shared Channel algorithms.

Overview
--------

The PUSCH JAX module provides differentiable implementations of the inner
receiver signal processing chain. These implementations can be lowered to
TensorRT for high-performance execution and are organized into signal
processing stages:

**Inner Receiver (Signal Processing)**

* **Channel Estimation** - Traditional and neural network-based channel
  estimators that can be trained end-to-end
* **AI Tukey Filter** - ML-based channel estimation filter with pretrained
  models for improved performance in challenging channel conditions
* **Noise Estimation** - Noise covariance estimation for MMSE equalization
* **Delay Compensation** - Time-domain delay correction
* **Equalization** - MMSE channel equalization
* **Free Energy Filter** - Advanced filtering for channel estimation refinement
* **Soft Demapping** - LLR generation from equalized symbols
* **Signal Quality Metrics** - Noise variance, RSRP, and SINR computation

**End-to-End Processing**

* **Complete Inner Receiver** - Full signal processing pipeline that can be
  lowered to TensorRT for real-time execution

API Reference
-------------

.. automodule:: ran.phy.jax.pusch
   :members:
   :undoc-members:
   :show-inheritance:

