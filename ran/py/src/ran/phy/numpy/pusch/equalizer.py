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

"""
Equalizer helpers (NumPy translations of MATLAB functions).

- derive_equalizer: frequency-domain MMSE-IRC equalizer per tone.
- apply_equalizer: apply equalizer to TF grid (basic TDI=0 behavior).

Assumptions
- NumPy only
- Inputs follow MATLAB shapes: `h_est` as (n_ant, nl, n_f)
- Outputs follow MATLAB shapes: `W` as (nl, n_ant, n_f)
- CFO/TO correction ignored in this translation (flags accepted but unused)
"""

import numpy as np

from ran.types import (
    ComplexArrayNP,
    ComplexNP,
    FloatArrayNP,
    FloatNP,
    IntNP,
)


def derive_equalizer(
    h_est: ComplexArrayNP,
    noise_intf_cov: ComplexArrayNP,
) -> tuple[ComplexArrayNP, FloatArrayNP]:
    """
    Derive per-tone MMSE-IRC equalizer and error metric.

    Args:
        h_est: channel estimates, shape (n_f, nl, n_ant, n_pos)
        noise_intf_cov: noise+interference covariance, shape(n_ant, n_ant, n_prb, n_pos)

    Returns
    -------
        w: equalizer weights, shape (nl, n_ant, n_f)
        ree: per-layer error metric, shape (nl, n_f, n_pos)
    """
    n_f, nl, n_ant, _ = h_est.shape

    # covariance to shape (n_ant, n_ant, n_prb, n_pos)
    _, _, n_prb, _ = noise_intf_cov.shape

    # Select first DMRS position to slice covariance (additional DMRS positions not supported)
    pos_ix = 0

    # precompute identities and prb mapping
    eye_nl = np.eye(nl, dtype=ComplexNP)
    prb_idx_per_f = np.clip(np.arange(n_f) // 12, 0, n_prb - 1).astype(IntNP)

    # precompute one whitening inverse per prb (reuse across its 12 tones) (n_prb, n_ant, n_ant)
    n_cov = noise_intf_cov[:, :, :, pos_ix].copy().transpose(2, 0, 1)

    # ensure diagonal is real for numerical stability
    diag_indices = np.arange(n_ant)
    n_cov[:, diag_indices, diag_indices] = np.real(n_cov[:, diag_indices, diag_indices])

    l_chol = np.linalg.cholesky(n_cov)  # (n_prb, n_ant, n_ant)
    l_inv = np.linalg.inv(l_chol)  # (n_prb, n_ant, n_ant)

    # gather whitening inverse per tone: (n_f, n_ant, n_ant)
    l_inv_f = l_inv[prb_idx_per_f]

    # h (n_f, nl, n_ant, n_pos) -> (n_f, n_ant, nl)
    h_f = np.transpose(h_est[:, :, :, pos_ix], (0, 2, 1))

    # n_f = l_inv_f @ h_f  -> (n_f, n_ant, nl)
    n_mat = l_inv_f @ h_f

    # g_f = n_f^h n_f + i -> (n_f, nl, nl)
    g_mat = np.einsum("fmn,fmk->fnk", n_mat.conj(), n_mat) + eye_nl[None, :, :]

    # ree_f = inv(g_f) -> (n_f, nl, nl)
    ree_f = np.linalg.inv(g_mat)

    # t_f = n_f^h @ l_inv_f -> (n_f, nl, n_ant)
    t_mat = (n_mat.conj().transpose(0, 2, 1)) @ l_inv_f

    # w_f = ree_f @ t_f -> (n_f, nl, n_ant)
    w_f = ree_f @ t_mat

    # bias correction and clamping
    diag_ree = np.real(ree_f.diagonal(axis1=1, axis2=2))  # (n_f, nl)
    lambda_vec = 1.0 / (1.0 - diag_ree)
    min_ree = 1.0 / 10000.0
    ree_layer_tone = np.maximum(min_ree, lambda_vec * diag_ree)  # (n_f, nl)

    # scale w per layer
    w_f = w_f * lambda_vec[:, :, None]

    w = np.transpose(w_f, (1, 2, 0)).astype(ComplexNP, copy=False)  # (nl, n_ant, n_f)
    ree = ree_layer_tone.T.astype(FloatNP, copy=False)  # (nl, n_f)

    return w, ree[..., None]  # add n_pos dimension


def apply_equalizer(
    xtf_data: ComplexArrayNP,
    w: ComplexArrayNP,
) -> ComplexArrayNP:
    """
    Apply equalizer to TF grid for data symbols. Implements TDI=0 behavior.

    Args:
        xtf_data: TF grid restricted to data symbols, shape (n_f, n_sym_data, n_ant)
        w: equalizer weights, shape (nl, n_ant, n_f)

    Returns
    -------
        x_est: estimated symbols per tone and data symbol, shape (n_f, n_sym_data, nl)
    """
    y = np.transpose(xtf_data, (2, 0, 1))  # (n_ant, n_f, n_sym)

    # w: (nl, n_ant, n_f)  x  y: (n_ant, n_f, n_sym) -> (nl, n_f, n_sym)
    x = np.einsum("lan,ans->lns", w, y, optimize=True)

    return np.transpose(x, (1, 2, 0))  # (n_f, n_sym, nl)


def equalize(
    h_est: ComplexArrayNP,
    noise_intf_cov: ComplexArrayNP,
    xtf_data: ComplexArrayNP,
) -> tuple[ComplexArrayNP, FloatArrayNP]:
    """
    Derive and apply equalizer to TF grid for data symbols.

    Args:
        h_est: Channel estimate, shape (n_f, nl, n_ant, n_pos)
        noise_intf_cov: Noise/interference covariance, shape (n_f, n_ant, n_ant)
        xtf_data: TF grid restricted to data symbols, shape (n_f, n_sym_data, n_ant)

    Returns
    -------
        x_est: estimated symbols per tone and data symbol, shape (n_f, n_sym_data, nl)
        ree: per-layer error metric, shape (nl, n_f)
    """
    # Derive equalizer weights
    w, ree = derive_equalizer(h_est, noise_intf_cov)

    # Apply equalizer
    x_est = apply_equalizer(xtf_data, w)

    return x_est, ree


__all__ = [
    "apply_equalizer",
    "derive_equalizer",
    "equalize",
]
