Message Adapter
===============

FAPI message receiver and processor converting FAPI PDUs to PUSCH pipeline inputs.

Overview
--------

The Message Adapter receives FAPI messages via NVIPC and processes them through pipelines:

* **FAPI Reception** - Receives FAPI messages via NVIPC in dedicated thread
* **Message Processing** - Routes messages through pipeline implementations
* **FAPI to PUSCH Conversion** - Converts FAPI PDUs (UL_TTI_REQUEST) to PUSCH input parameters
* **Slot Accumulation** - Accumulates messages per slot and cell before processing
* **Pipeline Interfaces** - Provides slot indication, slot info, and pipeline executor interfaces

API Reference
-------------

.. doxygennamespace:: ran::message_adapter
   :content-only:
   :members:
   :undoc-members:

.. doxygennamespace:: ran::fapi_5g
   :content-only:
   :members:
   :undoc-members:

