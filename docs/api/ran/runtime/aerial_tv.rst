Aerial TV
=========

Test vector utilities for RAN runtime validation and verification.

Overview
--------

The Aerial Test Vector (TV) library provides utilities for loading and managing CUPHY test vectors
from HDF5 files. It enables validation of physical layer processing by providing reference
data, parameters, and helper functions for comparing processing results against known-good
outputs.

Key Features
~~~~~~~~~~~~

-  **HDF5 Test Vector Loading**: Read gNB, UE group, and transport block parameters from test files
-  **Parameter Conversion**: Convert test vectors to PhyParams and other configuration structures
-  **Validation Utilities**: Check processing results against reference data (SINR, noise
   variance, payloads)
-  **Lazy Loading**: Efficient memory usage with on-demand parameter loading
-  **Type-Safe Access**: Template-based HDF5 dataset reading with compile-time type checking

Quick Start
-----------

Loading a Test Vector
~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../../ran/runtime/aerial_tv/tests/cuphy_pusch_tv_sample_test.cpp
   :language: cpp
   :start-after: example-begin load-test-vector-1
   :end-before: example-end load-test-vector-1
   :dedent: 4

Reading gNB Parameters
~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../../ran/runtime/aerial_tv/tests/cuphy_pusch_tv_sample_test.cpp
   :language: cpp
   :start-after: example-begin read-gnb-params-1
   :end-before: example-end read-gnb-params-1
   :dedent: 4

Reading Transport Block Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../../ran/runtime/aerial_tv/tests/cuphy_pusch_tv_sample_test.cpp
   :language: cpp
   :start-after: example-begin read-tb-params-1
   :end-before: example-end read-tb-params-1
   :dedent: 4

Core Concepts
-------------

Test Vector Structure
~~~~~~~~~~~~~~~~~~~~~

A **CuphyPuschTestVector** represents a complete PUSCH test scenario loaded from an HDF5
file. Each test vector contains three main parameter groups:

-  **gNB Parameters** (``CuphyPuschTvGnbParams``): Cell configuration including number of
   PRBs, receive antennas, subcarrier spacing, and algorithm selections for channel
   estimation, LDPC decoding, and CSI processing
-  **UE Group Parameters** (``CuphyPuschTvUeGrpParams``): Resource allocation per UE group
   including PRB allocation, symbol ranges, and DMRS configuration
-  **Transport Block Parameters** (``CuphyPuschTvTbParams``): Per-TB configuration including
   MCS, redundancy version, layer mapping, and UCI (HARQ-ACK, CSI) parameters

Parameters are loaded lazily on first access, improving performance when only specific
parameter groups are needed.

Reading HDF5 Datasets
~~~~~~~~~~~~~~~~~~~~~

The library provides flexible dataset reading capabilities:

Scalar Values
^^^^^^^^^^^^^

.. literalinclude:: ../../../../ran/runtime/aerial_tv/tests/cuphy_pusch_tv_sample_test.cpp
   :language: cpp
   :start-after: example-begin read-scalar-1
   :end-before: example-end read-scalar-1
   :dedent: 4

Array Data
^^^^^^^^^^

.. literalinclude:: ../../../../ran/runtime/aerial_tv/tests/cuphy_pusch_tv_sample_test.cpp
   :language: cpp
   :start-after: example-begin read-array-1
   :end-before: example-end read-array-1
   :dedent: 4

Arrays are returned with their data flattened in row-major order and dimension information
preserved.

Parameter Conversion
~~~~~~~~~~~~~~~~~~~~

Convert test vector parameters to framework types:

.. literalinclude:: ../../../../ran/runtime/aerial_tv/tests/cuphy_pusch_tv_sample_test.cpp
   :language: cpp
   :start-after: example-begin convert-to-phy-params-1
   :end-before: example-end convert-to-phy-params-1
   :dedent: 4

Other Configuration Conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../ran/runtime/aerial_tv/tests/cuphy_pusch_tv_sample_test.cpp
   :language: cpp
   :start-after: example-begin convert-to-outer-rx-params-1
   :end-before: example-end convert-to-outer-rx-params-1
   :dedent: 4

Static Reading Methods
~~~~~~~~~~~~~~~~~~~~~~

For direct parameter access without creating a test vector object:

.. literalinclude:: ../../../../ran/runtime/aerial_tv/tests/cuphy_pusch_tv_sample_test.cpp
   :language: cpp
   :start-after: example-begin static-read-methods-1
   :end-before: example-end static-read-methods-1
   :dedent: 4

These static methods are useful for quick parameter inspection or when only a specific
parameter group is needed.

UE Group Parameters
~~~~~~~~~~~~~~~~~~~

Access UE group-specific resource allocation:

.. literalinclude:: ../../../../ran/runtime/aerial_tv/tests/cuphy_pusch_tv_sample_test.cpp
   :language: cpp
   :start-after: example-begin read-ue-grp-params-1
   :end-before: example-end read-ue-grp-params-1
   :dedent: 4

API Reference
-------------

.. doxygennamespace:: ran::aerial_tv
   :content-only:
   :members:
   :undoc-members:
