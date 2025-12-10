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

"""Unit tests for `ran.phy.numpy.pusch.crc_decode`."""

import numpy as np
import pytest

from ran.phy.numpy.pusch.crc_decoder import _compute_crc_bits, crc_decode
from ran.types import FloatNP


def test_roundtrip_pass_for_all_variants() -> None:
    """Build LLRs, append correct CRC for 24A/24B/24C/16/11 and expect pass (err=0)."""
    data_len = 64
    x_llr: np.ndarray = np.linspace(1.0, -1.0, data_len, dtype=FloatNP)
    for crcstr in ("24A", "24B", "24C", "16", "11"):
        crc_bits = _compute_crc_bits(x_llr, crcstr)
        crc_llr = np.where(crc_bits == 0, 1.0, -1.0).astype(FloatNP)
        y = np.concatenate([x_llr, crc_llr])
        x_out, err = crc_decode(y, crcstr)
        np.testing.assert_allclose(x_out, x_llr, rtol=0, atol=0)
        assert err == 0


def test_crc_failure_detected() -> None:
    """CRC failure detection for 24A variant."""
    data_len = 40
    x_llr: np.ndarray = np.ones((data_len,), dtype=FloatNP)
    crc_bits = _compute_crc_bits(x_llr, "24A")
    # flip first bit to force mismatch
    crc_bits[0] ^= 1
    crc_llr = np.where(crc_bits == 0, 1.0, -1.0).astype(FloatNP)
    y = np.concatenate([x_llr, crc_llr])
    _, err = crc_decode(y, "24A")
    assert err == 1


def test_unsupported_crc_raises() -> None:
    """Unsupported CRC name raises ValueError."""
    with pytest.raises(ValueError):
        _ = crc_decode(np.zeros((10,), dtype=FloatNP), "7")
