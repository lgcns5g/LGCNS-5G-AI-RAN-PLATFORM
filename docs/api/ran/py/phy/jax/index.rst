JAX PHY
=======

JAX implementations of PHY algorithms including AI integration and TensorRT
lowering.

Overview
--------

The JAX PHY module provides JAX implementations that can be:

* **Used with ML models** - Integrate PHY algorithms with neural networks
* **Lowered to TensorRT** - Compile to optimized TensorRT engines
* **Trained end-to-end** - Differentiable implementations for training

Components
----------

* **PUSCH** (:doc:`pusch`) - JAX implementations of PUSCH receiver algorithms
* **Utils** (:doc:`utils`) - JAX signal processing utilities

.. toctree::
   :maxdepth: 1

   pusch
   utils

