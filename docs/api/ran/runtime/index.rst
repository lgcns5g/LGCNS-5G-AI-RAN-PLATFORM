Runtime API
===========

C++ components for GPU-accelerated 5G NR PHY processing, designed for real-time
operation with deterministic latency and standards compliance (3GPP 5G NR,
FAPI, O-RAN).

Components
----------

* **Aerial TV** (:doc:`aerial_tv`) - Test vector generation and validation
  tools
* **Common** (:doc:`common`) - Shared types, constants, and utilities
* **Driver** (:doc:`driver`) - Pipeline execution driver managing device memory
  and kernel launches
* **FAPI** (:doc:`fapi`) - 5G FAPI (PHY-MAC) interface
* **Fronthaul** (:doc:`fronthaul`) - Packet-based fronthaul protocol handling
  with timing and synchronization
* **LDPC** (:doc:`ldpc`) - Low-density parity-check decoder
* **Message Adapter** (:doc:`message_adapter`) - Bridges external FAPI messages
  to internal PHY processing inputs and outputs
* **O-RAN** (:doc:`oran`) - O-RAN fronthaul implementation with C-Plane/U-Plane
  message processing
* **PHY RAN App** (:doc:`phy_ran_app`) - Top-level PHY application including
  all RAN components and integration with MAC and RU emulators
* **PUSCH** (:doc:`pusch`) - GPU-accelerated Physical Uplink Shared Channel
  receiver chain

.. toctree::
   :maxdepth: 2
   :caption: Runtime Components

   aerial_tv
   common
   driver
   fapi
   fronthaul
   ldpc
   message_adapter
   oran
   phy_ran_app
   pusch
