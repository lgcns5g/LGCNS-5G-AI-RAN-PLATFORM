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

"""Unit tests for `ran.phy.numpy.pusch.equalizer` (NumPy implementation)."""

import numpy as np
import pytest

from ran.constants import SC_PER_PRB
from ran.phy.numpy.pusch.equalizer import (
    apply_equalizer,
    derive_equalizer,
    equalize,
)
from ran.types import ComplexNP, FloatNP


def _make_orthonormal_h(*, n_ant: int, nl: int, n_f: int, n_pos: int = 1) -> np.ndarray:
    """Build H with orthonormal columns, constant over tones and positions.

    Returns array with shape (n_f, nl, n_ant, n_pos).
    """
    # Base H (n_ant x nl)
    h0 = np.eye(n_ant, dtype=ComplexNP)[:, :nl]
    # Allocate and fill (n_f, nl, n_ant, n_pos) with H^H per tone/pos
    h: np.ndarray = np.zeros((n_f, nl, n_ant, n_pos), dtype=ComplexNP)
    h_conj_t = h0.conj().T  # (nl, n_ant)
    for f in range(n_f):
        for p in range(n_pos):
            h[f, :, :, p] = h_conj_t
    return h


class TestDeriveEqualizer:
    """Tests for derive_equalizer."""

    def test_shapes_and_dtypes_identity_noise(self) -> None:
        """Shapes and dtypes for orthonormal H and identity covariance."""
        n_ant, nl, n_f, n_pos = 3, 2, 5, 1
        h: np.ndarray = _make_orthonormal_h(n_ant=n_ant, nl=nl, n_f=n_f, n_pos=n_pos)
        # Identity covariance, 1 PRB, 1 position -> shape (n_ant, n_ant, 1, 1)
        r: np.ndarray = np.eye(n_ant, dtype=ComplexNP)[:, :, None, None]

        w, ree = derive_equalizer(h_est=h, noise_intf_cov=r)
        assert w.shape == (nl, n_ant, n_f)
        assert w.dtype == ComplexNP
        assert ree.shape == (nl, n_f, n_pos)
        assert ree.dtype == FloatNP

    def test_expected_solution_orthonormal_identity_noise(self) -> None:
        """For orthonormal H and identity covariance, expect W == H^H and Ree == 1."""
        n_ant, nl, n_f, n_pos = 3, 2, 4, 1
        h: np.ndarray = _make_orthonormal_h(n_ant=n_ant, nl=nl, n_f=n_f, n_pos=n_pos)
        r: np.ndarray = np.eye(n_ant, dtype=ComplexNP)[:, :, None, None]

        w, ree = derive_equalizer(h_est=h, noise_intf_cov=r)

        for f in range(n_f):
            h_f = np.transpose(h[f, :, :, 0], (1, 0))  # (n_ant, nl)
            expected_w = h_f.conj().T
            np.testing.assert_allclose(w[:, :, f], expected_w, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(ree[..., 0], 1.0, rtol=0, atol=1e-12)

    def test_covariance_selection_per_prb(self) -> None:
        """Scalar covariance per PRB yields W invariant (= H^H) across PRBs."""
        n_ant, nl, n_f, n_pos = 3, 2, 2 * SC_PER_PRB, 1  # 2 PRBs of 12 tones each
        h: np.ndarray = _make_orthonormal_h(n_ant=n_ant, nl=nl, n_f=n_f, n_pos=n_pos)
        r: np.ndarray = np.zeros((n_ant, n_ant, 2, n_pos), dtype=ComplexNP)
        r[:, :, 0, 0] = 4.0 * np.eye(n_ant)
        r[:, :, 1, 0] = 1.0 * np.eye(n_ant)

        w, _ = derive_equalizer(h_est=h, noise_intf_cov=r)
        for f in range(n_f):
            h_f = np.transpose(h[f, :, :, 0], (1, 0))
            expected = h_f.conj().T
            np.testing.assert_allclose(w[:, :, f], expected, rtol=1e-12, atol=1e-12)

    def test_mmse_form_non_orthonormal_identity_noise(self) -> None:
        """With identity noise, W = (G^{-1} H^H) scaled by lambda per layer."""
        n_ant, nl, n_f, n_pos = 4, 3, 3, 1
        rng = np.random.default_rng(0)
        h: np.ndarray = np.zeros((n_f, nl, n_ant, n_pos), dtype=ComplexNP)
        for f in range(n_f):
            real_part = rng.standard_normal((n_ant, nl))
            imag_part = rng.standard_normal((n_ant, nl))
            h_f = (real_part + 1j * imag_part) / np.sqrt(2)
            h[f, :, :, 0] = h_f.conj().T  # store as (nl, n_ant)
        r: np.ndarray = np.eye(n_ant, dtype=ComplexNP)[:, :, None, None]

        w, _ = derive_equalizer(h_est=h, noise_intf_cov=r)

        for f in range(n_f):
            h_f = np.transpose(h[f, :, :, 0], (1, 0))  # (n_ant, nl)
            g = h_f.conj().T @ h_f + np.eye(nl)
            ree = np.linalg.inv(g)
            lambda_vec = 1.0 / (1.0 - np.real(np.diag(ree)))
            expected_w = (lambda_vec[:, None]) * (ree @ h_f.conj().T)
            np.testing.assert_allclose(w[:, :, f], expected_w, rtol=1e-10, atol=1e-10)

    def test_per_prb_covariance_selection_single_pos(self) -> None:
        """Different covariance per PRB yields W invariant (= H^H) with orthonormal H."""
        n_ant, nl, n_f, n_pos = 3, 2, 24, 1
        h = _make_orthonormal_h(n_ant=n_ant, nl=nl, n_f=n_f, n_pos=n_pos)
        r: np.ndarray = np.zeros((n_ant, n_ant, 2, n_pos), dtype=ComplexNP)
        r[:, :, 0, 0] = 2.0 * np.eye(n_ant)
        r[:, :, 1, 0] = 5.0 * np.eye(n_ant)

        w, _ = derive_equalizer(h_est=h, noise_intf_cov=r)
        for f in range(n_f):
            h_f = np.transpose(h[f, :, :, 0], (1, 0))
            expected = h_f.conj().T
            np.testing.assert_allclose(w[:, :, f], expected, rtol=1e-12, atol=1e-12)

    def test_invalid_h_shape_raises(self) -> None:
        """h_est must be 4D (n_f, nl, n_ant, n_pos)."""
        n_ant, nl, n_f = 3, 2, 5
        h_bad: np.ndarray = np.zeros((n_f, nl, n_ant), dtype=ComplexNP)  # 3D, invalid
        r: np.ndarray = np.eye(n_ant, dtype=ComplexNP)[:, :, None, None]
        with pytest.raises(ValueError):
            _ = derive_equalizer(h_est=h_bad, noise_intf_cov=r)

    def test_non_psd_covariance_raises(self) -> None:
        """Non-PSD covariance should raise from Cholesky (per PRB)."""
        n_ant, nl, n_f, n_pos = 3, 2, 2, 1
        h = _make_orthonormal_h(n_ant=n_ant, nl=nl, n_f=n_f, n_pos=n_pos)
        # Indefinite matrix in (n_ant, n_ant, 1, 1)
        r: np.ndarray = np.array([[1, 2, 3], [2, 1, 4], [3, 4, -1]], dtype=ComplexNP)
        r = r[:, :, None, None]
        with pytest.raises(np.linalg.LinAlgError):
            _ = derive_equalizer(h_est=h, noise_intf_cov=r)


class TestApplyEqualizerTdi:
    """Tests for apply_equalizer and equalize."""

    def test_apply_identity_recovery(self) -> None:
        """With W = H^H and Y = H*S at one data symbol, output recovers S."""
        n_ant, nl, n_f, n_t, n_pos = 3, 2, 8, 3, 1
        h = _make_orthonormal_h(n_ant=n_ant, nl=nl, n_f=n_f, n_pos=n_pos)

        # Build W = H^H per tone
        w: np.ndarray = np.zeros((nl, n_ant, n_f), dtype=ComplexNP)
        for f in range(n_f):
            # h stores H^H per tone/pos: (nl, n_ant)
            w[:, :, f] = h[f, :, :, 0]

        # Build TF grid Y for one data symbol (symbol index 2 in 0-based)
        s_port = np.array([1.0 + 0.0j, -1.0 + 0.0j], dtype=ComplexNP)  # (nl,)
        y: np.ndarray = np.zeros((n_f, n_t, n_ant), dtype=ComplexNP)
        s_idx = 2  # 0-based
        for f in range(n_f):
            # h[f, :, :, 0] = H^H (nl, n_ant) ⇒ H = (H^H)^H = (n_ant, nl)
            h_f = h[f, :, :, 0].conj().T
            y[f, s_idx - 1, :] = (h_f @ s_port).ravel()

        xtf_data = y[:, [s_idx - 1], :]
        x_est = apply_equalizer(xtf_data=xtf_data, w=w)
        expected = np.tile(s_port[None, None, :], (n_f, 1, 1))
        np.testing.assert_allclose(x_est, expected, rtol=1e-12, atol=1e-12)

    def test_equalize_end_to_end(self) -> None:
        """Equalize derives W and applies it, returning expected shapes/results."""
        n_ant, nl, n_f, n_t, n_pos = 2, 1, 6, 2, 1
        h = _make_orthonormal_h(n_ant=n_ant, nl=nl, n_f=n_f, n_pos=n_pos)
        r: np.ndarray = np.eye(n_ant, dtype=ComplexNP)[:, :, None, None]
        y: np.ndarray = np.zeros((n_f, n_t, n_ant), dtype=ComplexNP)
        s_idx = 1  # 0-based
        s_port = np.array([1.0 + 0.0j], dtype=ComplexNP)
        for f in range(n_f):
            h_f = h[f, :, :, 0].conj().T
            y[f, s_idx - 1, :] = (h_f @ s_port).ravel()

        xtf_data = y[:, [s_idx - 1], :]
        x_est, ree = equalize(
            h_est=h,
            noise_intf_cov=r,
            xtf_data=xtf_data,
        )

        # Shapes
        assert x_est.shape == (n_f, 1, nl)
        assert ree.shape == (nl, n_f, n_pos)

        # Values
        expected = np.tile(s_port[None, None, :], (n_f, 1, 1))
        np.testing.assert_allclose(x_est, expected, rtol=1e-12, atol=1e-12)
