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

"""Unit tests for `ran.phy.numpy.pusch.codeblock_concatenation`."""

import numpy as np
import pytest

from ran.phy.numpy.pusch import codeblock_concatenation
from ran.phy.numpy.pusch.crc_decoder import _compute_crc_bits
from ran.types import FloatNP, IntNP


def test_single_cb_pass_through() -> None:
    """C==1 passes through without CRC removal."""
    c = 1
    k_prime = 32
    # Build tb_cbs_est with arbitrary values
    tb_cbs_est: np.ndarray = np.arange(k_prime, dtype=FloatNP)[:, None]

    out_vec, cb_err = codeblock_concatenation(
        tb_cbs_est=tb_cbs_est,
        c=c,
        k_prime=k_prime,
    )

    # Expect flattened (column-major) equality
    np.testing.assert_allclose(out_vec, tb_cbs_est.ravel(order="F"), rtol=0, atol=0)
    np.testing.assert_array_equal(cb_err, np.zeros((1,), dtype=IntNP))


def test_multi_cb_crc_removal_success() -> None:
    """C>1 removes CRC24B per CB and concatenates; errors should be zero."""
    c = 2
    # Choose K' so that data length is K'-24
    k_prime = 64
    data_len = k_prime - 24

    # Build LLRs for data: positive => bit 0; negative => bit 1
    # CB0: all zeros (all +1.0)
    data0: np.ndarray = np.ones((data_len,), dtype=FloatNP)
    crc0_bits = _compute_crc_bits(data0, "24B")
    crc0_llr = np.where(crc0_bits == 0, 1.0, -1.0).astype(FloatNP)
    col0 = np.concatenate([data0, crc0_llr])

    # CB1: alternating bits (10 pattern)
    bits1: np.ndarray = np.zeros((data_len,), dtype=IntNP)
    bits1[::2] = 1
    data1 = np.where(bits1 == 0, 1.0, -1.0).astype(FloatNP)
    crc1_bits = _compute_crc_bits(data1, "24B")
    crc1_llr = np.where(crc1_bits == 0, 1.0, -1.0).astype(FloatNP)
    col1 = np.concatenate([data1, crc1_llr])

    tb_cbs_est = np.stack([col0, col1], axis=1)

    out_vec, cb_err = codeblock_concatenation(
        tb_cbs_est=tb_cbs_est,
        c=c,
        k_prime=k_prime,
    )

    # Expect cb_err all zeros and output equals concatenated data (without CRC)
    np.testing.assert_array_equal(cb_err, np.zeros((c,), dtype=IntNP))
    expected = tb_cbs_est[:data_len, :].ravel(order="F")
    np.testing.assert_allclose(out_vec, expected, rtol=0, atol=0)


def test_multi_cb_crc_failure_detected() -> None:
    """Flip one CRC bit; expect cb_err marks failure for that CB."""
    c = 2
    k_prime = 64
    data_len = k_prime - 24
    # Build passing CB0
    data0: np.ndarray = np.ones((data_len,), dtype=FloatNP)
    crc0_bits = _compute_crc_bits(data0, "24B")
    crc0_llr = np.where(crc0_bits == 0, 1.0, -1.0).astype(FloatNP)
    col0 = np.concatenate([data0, crc0_llr])
    # Build CB1 then flip one CRC bit
    data1: np.ndarray = -np.ones((data_len,), dtype=FloatNP)  # all ones bits
    crc1_bits = _compute_crc_bits(data1, "24B")
    crc1_bits[0] ^= 1  # flip one bit
    crc1_llr = np.where(crc1_bits == 0, 1.0, -1.0).astype(FloatNP)
    col1 = np.concatenate([data1, crc1_llr])
    tb_cbs_est = np.stack([col0, col1], axis=1)
    _, cb_err = codeblock_concatenation(
        tb_cbs_est=tb_cbs_est,
        c=c,
        k_prime=k_prime,
    )
    np.testing.assert_array_equal(cb_err, np.array([0, 1], dtype=IntNP))


def test_kprime_less_than_crc_bits_raises() -> None:
    """When c>1 and K' < 24, there aren't enough CRC bits; expect error."""
    c = 2
    k_prime = 16
    tb_cbs_est: np.ndarray = np.zeros((k_prime, c), dtype=FloatNP)
    with pytest.raises(ValueError):
        _ = codeblock_concatenation(tb_cbs_est, c=c, k_prime=k_prime)
