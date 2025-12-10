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

"""Unit tests for `ran.phy.numpy.pusch.soft_demapper` (NumPy implementation)."""

from typing import TYPE_CHECKING

import numpy as np
import pytest

from ran.phy.numpy.pusch import soft_demapper
from ran.types import ComplexArrayNP, ComplexNP, FloatNP

if TYPE_CHECKING:
    from ran.types import ComplexArrayNP, FloatArrayNP


def test_bpsk_rotation_pattern_tdimode0() -> None:
    """For qam_bits=1, verify rotation yields alternating signs across tones."""
    nl, n_f, n_sym = 1, 12, 1
    # x constant ones, shape (n_f, n_sym, n_layers)
    x: ComplexArrayNP = np.ones((n_f, n_sym, nl), dtype=ComplexNP)
    # Set ree so that final scaling yields 1/sqrt(2) after rotation: use ree=4.0
    ree: FloatArrayNP = 4.0 * np.ones((nl, n_f), dtype=FloatNP)

    llr = soft_demapper(x=x, ree=ree, qam_bits=1)

    # Expect shape (1, nl, 12, 1)
    assert llr.shape == (1, nl, 12, n_sym)
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    vals = llr[0, 0, :, 0]
    np.testing.assert_allclose(vals[0::2], inv_sqrt2, rtol=0, atol=1e-12)
    np.testing.assert_allclose(vals[1::2], -inv_sqrt2, rtol=0, atol=1e-12)


def test_qam16_llr_scales_with_n0() -> None:
    """LLR magnitudes should scale ~1/N0 for QAM>1 (qam_bits=4)."""
    nl, n_f, n_sym = 1, 12, 1
    x: ComplexArrayNP = np.full((n_f, n_sym, nl), 3.0 + 3.0j, dtype=ComplexNP)
    # N0=1 run
    ree1: FloatArrayNP = np.ones((nl, n_f), dtype=FloatNP)
    llr1 = soft_demapper(x=x, ree=ree1, qam_bits=4)
    # N0=2 run
    ree2: FloatArrayNP = 2.0 * np.ones((nl, n_f), dtype=FloatNP)
    llr2 = soft_demapper(x=x, ree=ree2, qam_bits=4)
    # Same shapes
    assert llr1.shape == llr2.shape == (4, nl, 12, n_sym)
    # Magnitude roughly halves when N0 doubles
    ratio = float(np.linalg.norm(llr1)) / max(float(np.linalg.norm(llr2)), 1e-12)
    np.testing.assert_allclose(ratio, 2.0, rtol=1e-2, atol=1e-2)


def test_qam64_and_qam256_shapes_and_lengths() -> None:
    """Test QAM64 and QAM256 output shapes over one PRB (12 tones)."""
    nl, n_f, n_sym = 1, 12, 2
    x: ComplexArrayNP = np.ones((n_f, n_sym, nl), dtype=ComplexNP)
    ree: FloatArrayNP = np.ones((nl, n_f), dtype=FloatNP)
    # QAM64
    qam64_bits = 6
    llr64 = soft_demapper(x=x, ree=ree, qam_bits=qam64_bits)
    assert llr64.shape == (qam64_bits, nl, 12, n_sym)
    # QAM256
    qam256_bits = 8
    llr256 = soft_demapper(x=x, ree=ree, qam_bits=qam256_bits)
    assert llr256.shape == (qam256_bits, nl, 12, n_sym)


def test_invalid_qam_bits_raises() -> None:
    """Unsupported qam_bits should raise ValueError."""
    nl, n_f, n_sym = 1, 12, 1
    x: ComplexArrayNP = np.ones((n_f, n_sym, nl), dtype=ComplexNP)
    ree: FloatArrayNP = np.ones((nl, n_f), dtype=FloatNP)
    with pytest.raises(ValueError):
        _ = soft_demapper(x=x, ree=ree, qam_bits=3)


def test_zero_prb_allocation_yields_zero_length_freq_axis() -> None:
    """With n_prb=0, frequency axis of output should be length 0."""
    nl, n_sym = 1, 1
    x: ComplexArrayNP = np.ones((0, n_sym, nl), dtype=ComplexNP)
    ree: FloatArrayNP = np.ones((nl, 0), dtype=FloatNP)
    llr = soft_demapper(x=x, ree=ree, qam_bits=4)
    assert llr.shape == (4, nl, 0, n_sym)
