PUSCH
=====

Physical Uplink Shared Channel algorithms in NumPy.

Overview
--------

The PUSCH NumPy module provides reference implementations of the complete
Physical Uplink Shared Channel receiver chain. The module is organized into
signal processing stages:

**Inner Receiver (Signal Processing)**

* **DMRS Generation and Extraction** - Generate transmitted DMRS sequences and
  extract raw DMRS from resource grids
* **Channel Estimation** - Least-squares and delay domain estimators
* **Noise Estimation** - Noise covariance estimation, R-tilde estimation, and
  covariance shrinkage
* **Equalization** - MMSE equalizer derivation, application, and combined
  operation
* **Soft Demapping** - LLR generation from equalized symbols
* **Signal Quality Metrics** - RSSI measurement, noise variance/RSRP/SINR
  computation, and post-equalization metrics

**Outer Receiver (Channel Decoding)**

* **Derate Matching** - Reverse rate matching for LDPC decoder input
* **LDPC Decoding** - Iterative belief propagation decoder
* **Descrambling** - Bit descrambling
* **CRC Decoding** - CRC validation and removal
* **Code Block Concatenation** - Assemble transport blocks from code blocks

**End-to-End Processing**

* **Complete Receiver** - Full PUSCH receiver chain
* **Inner Receiver Only** - Signal processing without channel decoding
* **Outer Receiver Only** - Channel decoding chain

API Reference
-------------

.. automodule:: ran.phy.numpy.pusch
   :members:
   :undoc-members:
   :show-inheritance:

