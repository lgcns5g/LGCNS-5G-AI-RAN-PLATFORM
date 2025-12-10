Driver
======

PUSCH pipeline manager with device memory handling and multi-cell slot coordination.

Overview
--------

The Driver manages PUSCH pipeline lifecycle and coordinates multi-cell processing:

* **Pipeline Creation** - Creates PUSCH pipelines with TensorRT engines
  (Stream or Graph execution modes)
* **Device Memory Management** - Allocates CUDA device memory for pipeline inputs/outputs
* **Slot Response Tracking** - Tracks which cells have responded per slot using atomic operations
* **Pipeline Launch** - Launches PUSCH pipeline execution when all cells are ready
* **UL Indication Callbacks** - Notifies when pipeline execution completes

API Reference
-------------

.. doxygennamespace:: ran::driver
   :content-only:
   :members:
   :undoc-members:

