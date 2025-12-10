# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CRC decode (NumPy translation of CRC_decode.m). Minimal checks, fast path."""

import numpy as np

from ran.types import FloatArrayNP, IntArrayNP, IntNP

# (nbits, poly, init, xor_out) from 3GPP 38.212 Annex B
_CRC_PARAMS: dict[str, tuple[int, int, int, int]] = {
    "24A": (24, 0x1864CFB, 0x000000, 0x000000),
    "24B": (24, 0x1800063, 0x000000, 0x000000),
    "24C": (24, 0x1B2B117, 0x000000, 0x000000),
    "16": (16, 0x1021, 0x0000, 0x0000),
    "11": (11, 0x0307, 0x000, 0x000),
}


def _compute_crc_bits(x_llr: FloatArrayNP, crc_name: str) -> IntArrayNP:
    """Compute CRC bits for payload LLRs using specified CRC polynomial.

    Computes the CRC checksum for the given payload LLRs using the CRC parameters
    defined in 3GPP 38.212 Annex B. The computation follows the standard LFSR
    (Linear Feedback Shift Register) approach with MSB-first bit ordering.

    Args:
        x_llr: Payload LLRs as float64 array. Negative values represent bit 1,
               non-negative values represent bit 0.
        crc_name: CRC type identifier. Must be one of: "24A", "24B", "24C", "16", "11".
                  Corresponds to CRC polynomials defined in 3GPP 38.212.

    Returns
    -------
        CRC bits as int64 array with MSB-first ordering. Each element is 0 or 1.
        Array length matches the CRC size (e.g., 24 bits for CRC-24A).

    Note:
        Assumes inputs are valid (e.g., crc_name exists in _CRC_PARAMS).
    """
    nbits, poly, init, xor_out = _CRC_PARAMS[crc_name]
    mask = (1 << nbits) - 1
    shift = nbits - 1

    # Hard bits as uint8 (0/1). copy=False avoids extra alloc if possible.
    bits_u8 = (x_llr < 0).astype(np.uint8, copy=False)

    # Iterate over raw bytes (fast Python ints 0/1).
    reg = init
    for b in bits_u8.tobytes():
        top = (reg >> shift) & 1
        reg = ((reg << 1) & mask) ^ (poly if (top ^ b) else 0)

    reg ^= xor_out

    # Export register to MSB-first vector
    out: IntArrayNP = np.empty(nbits, dtype=IntNP)
    for i in range(nbits):
        out[i] = (reg >> (nbits - 1 - i)) & 1
    return out


def crc_decode(tb_crc_est: FloatArrayNP, crc_name: str = "24A") -> tuple[FloatArrayNP, int]:
    """Decode CRC-protected transport block and detect errors.

    Separates the payload from the trailing CRC bits, computes the expected CRC
    for the payload, and compares it with the received CRC to detect errors.
    This implements the CRC checking procedure for 5G NR transport blocks
    as specified in 3GPP 38.212.

    Args:
        tb_crc_est: 1-D float64 array of LLRs with shape ``(K + ncrc,)`` containing
            the payload followed by its CRC bits. Negative values represent bit 1,
            non-negative values represent bit 0.
        crc_name: CRC type identifier. Must be one of: "24A", "24B", "24C", "16", "11".
            Determines the CRC length ``ncrc`` stripped from the end:
            - "24A"/"24B"/"24C" -> ``ncrc = 24``
            - "16" -> ``ncrc = 16``
            - "11" -> ``ncrc = 11``

    Returns
    -------
        tuple containing:
        - x_wo_crc: 1-D float64 array with shape ``(K,)`` (CRC bits removed)
        - err: Error flag (int scalar in {0, 1}). 0 if CRC check passes, 1 if it fails

    Notes
    -----
    - ``K`` is the number of payload LLRs (transport block without CRC).
    - ``ncrc`` is the number of CRC bits appended to the payload, determined by
      ``crc_name`` as listed above.
    - ``K + ncrc`` equals ``tb_crc_est.size``; equivalently
      ``K = tb_crc_est.size - ncrc``.
    - Expects ``tb_crc_est.ndim == 1`` (no batching).
    - Requires ``tb_crc_est.size >= ncrc`` so that payload and CRC can be separated.
    """
    if crc_name not in _CRC_PARAMS:
        raise ValueError(f"Unsupported CRC name: {crc_name!r}")
    ncrc = _CRC_PARAMS[crc_name][0]
    x_wo_crc = tb_crc_est[:-ncrc]
    crc_llr = tb_crc_est[-ncrc:]
    crc_expect = _compute_crc_bits(x_wo_crc, crc_name)
    err = int(np.any((crc_llr < 0).astype(IntNP, copy=False) != crc_expect))
    return x_wo_crc, err


__all__ = ["crc_decode"]
