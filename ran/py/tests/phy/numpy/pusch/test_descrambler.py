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

"""Tests for `ran.phy.numpy.pusch.descramble_bits` (NumPy implementation)."""

from typing import TYPE_CHECKING

import numpy as np

from ran.phy.numpy.pusch import descramble_bits
from ran.phy.numpy.utils import gold_sequence
from ran.types import FloatNP

if TYPE_CHECKING:
    from ran.types import FloatArrayNP


def test_shapes_and_types() -> None:
    """Output has the correct shape and dtype."""
    n = 257
    rng = np.random.default_rng()
    x: FloatArrayNP = rng.standard_normal(n, dtype=FloatNP)
    y = descramble_bits(x, n_id=100, n_rnti=0x1234)
    assert isinstance(y, np.ndarray)
    assert y.dtype == FloatNP
    assert y.shape == (n,)


def test_involution_property() -> None:
    """Applying descramble twice returns the original sequence."""
    n = 513
    rng = np.random.default_rng()
    x: FloatArrayNP = rng.standard_normal(n, dtype=FloatNP)
    n_id = 321
    n_rnti = 0xACE1
    y = descramble_bits(x, n_id=n_id, n_rnti=n_rnti)
    z = descramble_bits(y, n_id=n_id, n_rnti=n_rnti)
    np.testing.assert_allclose(z, x, rtol=0.0, atol=0.0)


def test_matches_gold_mask() -> None:
    """Output matches multiplying by the Gold mask built externally."""
    n = 1024
    n_id = 512
    n_rnti = 0xA5A5
    x: FloatArrayNP = np.arange(1, n + 1, dtype=FloatNP)
    y = descramble_bits(x, n_id=n_id, n_rnti=n_rnti)

    c_init = ((int(n_rnti) << 15) + int(n_id)) % (1 << 31)
    c = gold_sequence(c_init, n)
    mask = 1 - 2 * c.astype(FloatNP)
    expected = x * mask
    np.testing.assert_allclose(y, expected, rtol=0.0, atol=0.0)


def test_zero_input() -> None:
    """Zero input remains zero after descrambling."""
    n = 64
    x: FloatArrayNP = np.zeros((n,), dtype=FloatNP)
    y = descramble_bits(x, n_id=0, n_rnti=0)
    np.testing.assert_allclose(y, x, rtol=0.0, atol=0.0)


def test_extreme_values() -> None:
    """Applying mask preserves finiteness and involution on extreme values."""
    x: FloatArrayNP = np.array([1e-9, 1.0, 1e9, -1e-9, -1.0, -1e9], dtype=FloatNP)
    y = descramble_bits(x, n_id=7, n_rnti=3)
    z = descramble_bits(y, n_id=7, n_rnti=3)
    np.testing.assert_allclose(z, x, rtol=0.0, atol=0.0)
