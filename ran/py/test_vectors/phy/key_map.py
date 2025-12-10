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

"""Central MATLAB→Python key mappings and helpers for PUSCH pipeline tests."""

from collections.abc import Callable
from typing import Any

import numpy as np

from ran.types import (
    ComplexArrayNP,
    ComplexNP,
    FloatArrayNP,
    FloatNP,
    IntArrayNP,
    IntNP,
)


#   Casting Functions
# ---------- one utility to compose operations sequentially ----------
def apply_ops(x: Any, ops: list[Callable[[Any], Any]]) -> Any:  # noqa: ANN401
    """Apply a list of functions sequentially to x."""
    for fn in ops:
        x = fn(x)
    return x


# ---------- fundamental named functions ----------
def _as_c128(x: Any) -> ComplexArrayNP:  # noqa: ANN401
    return np.asarray(x, dtype=ComplexNP)


def _as_f64(x: Any) -> FloatArrayNP:  # noqa: ANN401
    return np.asarray(x, dtype=FloatNP)


def _as_i64(x: Any) -> IntArrayNP:  # noqa: ANN401
    return np.asarray(x, dtype=IntNP)


def _minus_1(x: Any) -> Any:  # noqa: ANN401
    return x - 1


def _to_scalar(x: Any) -> Any:  # noqa: ANN401
    """Convert size-1 arrays (or 0-d arrays) to a Python scalar; otherwise return x."""
    if isinstance(x, np.ndarray) and (x.shape == () or x.size == 1):
        return x.item()
    return x


def _first(x: Any) -> Any:  # noqa: ANN401
    """Take the first element after np.asarray & flattening."""
    return np.asarray(x).reshape(-1)[0]


def _ascii_from_bytes(x: Any) -> str:  # noqa: ANN401
    return np.asarray(x).tobytes().decode("ascii")


def _add_axis(x: Any) -> Any:  # noqa: ANN401
    return x[..., None]


def _swapaxes_02(x: Any) -> Any:  # noqa: ANN401
    return x.swapaxes(0, 2)


KEY_TO_MAP: dict[str, tuple[str, list[Callable[[Any], Any]]]] = {
    # Complex arrays
    "xtf": ("Xtf", [_as_c128]),
    "x_dmrs": ("XtfDmrs", [_as_c128]),
    "r_dmrs": ("r_dmrs", [_as_c128]),
    "h_est": ("H_est", [_as_c128, _add_axis, _swapaxes_02]),
    "noise_intf_cov": ("noiseIntfCov", [_as_c128, _add_axis]),
    "x_est": ("X_est", [_as_c128, _add_axis]),
    "w": ("W", [_as_c128]),
    "n_cov": ("nCov_before_shrinkage", [_as_c128, _add_axis]),
    "r_tilde": ("r_tilde", [_as_c128]),
    # Float arrays
    "ree": ("Ree", [_as_f64, _add_axis]),
    "llrseq": ("LLRseq", [_as_f64, np.ravel]),
    "llr_descr": ("LLR_descr", [_as_f64]),
    "derate_cbs": ("derateCbs", [_as_f64]),
    "tb_cbs_est": ("TbCbs_est", [_as_f64]),
    "mean_noise_var": ("tmp_noiseVar", [_as_f64]),
    "rsrp_db": ("rsrpdB", [_as_f64]),
    "noise_var_db": ("noiseVardB", [_as_f64]),
    # Int arrays / indices
    "layer2ue": ("layer2Ue", [_as_i64, np.ravel]),
    "vec_scid": ("vec_scid", [_as_i64, np.ravel]),
    "port_idx": ("portIdx", [_as_i64, np.ravel, _minus_1]),
    "sym_idx_dmrs": ("symIdx_dmrs", [_as_i64, np.ravel, _minus_1]),
    "sym_idx_data": ("symIdx_data", [_as_i64, np.ravel, _minus_1]),
    # Scalar ints
    "slot_number": ("slotNumber", [_to_scalar, int]),
    "n_f": ("Nf", [_to_scalar, int]),
    "n_t": ("Nt", [_to_scalar, int]),
    "n_dmrs_id": ("N_dmrs_id", [_to_scalar, int]),
    "n_prb": ("nPrb", [_to_scalar, int]),
    "n_ue": ("nUe", [_to_scalar, int]),
    "qam_bits": ("qam", [_to_scalar, int]),
    "n_id": ("N_id", [_to_scalar, int]),
    "n_rnti": ("n_rnti", [_to_scalar, int]),
    "bgn": ("BGN", [_to_scalar, int]),
    "c": ("C", [_to_scalar, int]),
    "k": ("K", [_to_scalar, int]),
    "f": ("F", [_to_scalar, int]),
    "k_prime": ("K_prime", [_to_scalar, int]),
    "zc": ("Zc", [_to_scalar, int]),
    "nref": ("Nref", [_to_scalar, int]),
    "g": ("G", [_to_scalar, int]),
    "nv_parity": ("nV_parity", [_to_scalar, int]),
    "nl": ("nl", [_to_scalar, int]),
    "rv_idx": ("rvIdx", [_to_scalar, int]),
    "i_ls": ("i_LS", [_to_scalar, int]),
    # 0-based scalar indices (compose from fundamentals)
    "start_prb": ("startPrb", [_to_scalar, int, _minus_1]),
    # Scalar floats
    "energy": ("energy", [_to_scalar, float]),
    "rww_regularizer_val": ("Rww_regularizer_val", [_to_scalar, float]),
    # Special derived
    "max_num_itr_cbs": ("maxNumItr_CBs", [_first, _to_scalar, int]),
    "tb_crc_est": ("TbCrc_est", [_as_f64]),
    "crc_name": ("CRC", [_ascii_from_bytes]),
}


def _cast_selected(tv: dict, keys: tuple[str, ...]) -> dict:
    """Return selected keys applying sequential ops defined in the global map.

    Each entry is (mat_key, [op1, op2, ...]) applied in order.
    """
    out: dict[str, Any] = {}
    for py_key in keys:
        mapper = KEY_TO_MAP.get(py_key)
        if mapper is None:
            raise KeyError(f"Unknown key: {py_key}")
        mat_key, ops = mapper
        out[py_key] = apply_ops(tv[mat_key], ops)
    return out


#   Builder Functions
def build_dmrs_kwargs(tv: dict) -> dict:
    """Explicit kwargs for ran.phy.numpy.pusch.gen_dmrs_sym from H5 dict."""
    return _cast_selected(tv, ("slot_number", "n_f", "n_dmrs_id", "sym_idx_dmrs"))


def build_ls_kwargs(tv: dict) -> dict:
    """Explicit kwargs for ran.phy.numpy.pusch.channel_estimation_ls from H5 dict."""
    keys = (
        "xtf",
        "nl",
        "port_idx",
        "vec_scid",
        "sym_idx_dmrs",
        "n_prb",
        "start_prb",
        "r_dmrs",
    )
    return _cast_selected(tv, keys)


def build_cov_kwargs(tv: dict) -> dict:
    """Explicit kwargs for ran.phy.numpy.pusch.estimate_covariance from H5 dict."""
    keys = (
        "xtf",
        "r_dmrs",
        "nl",
        "port_idx",
        "n_prb",
        "start_prb",
        "h_est",
        "rww_regularizer_val",
        "sym_idx_dmrs",
        "vec_scid",
        "energy",
    )
    return _cast_selected(tv, keys)


def build_embed_dmrs_kwargs(tv: dict) -> dict:
    """Explicit kwargs for ran.phy.numpy.pusch.embed_dmrs_ul from H5 dict."""
    keys = (
        "r_dmrs",
        "nl",
        "port_idx",
        "vec_scid",
        "sym_idx_dmrs",
        "energy",
        "n_prb",
        "start_prb",
    )
    return _cast_selected(tv, keys)


def build_r_tilde_kwargs(tv: dict) -> dict:
    """Explicit kwargs for ran.phy.numpy.pusch.estimate_r_tilde from H5 dict."""
    keys = ("xtf", "x_dmrs", "h_est", "start_prb", "n_prb", "sym_idx_dmrs")
    return _cast_selected(tv, keys)


def build_ncov_kwargs(tv: dict) -> dict:
    """Explicit kwargs for ran.phy.numpy.pusch.estimate_noise_covariance from H5 dict."""
    return _cast_selected(tv, ("r_tilde", "sym_idx_dmrs", "rww_regularizer_val"))


def build_shrink_kwargs(tv: dict) -> dict:
    """Explicit kwargs for ran.phy.numpy.pusch.ncov_shrinkage from H5 dict."""
    return _cast_selected(tv, ("n_cov",))


def build_noisevar_kwargs(tv: dict) -> dict:
    """Explicit kwargs for ran.phy_ref.pusch.noisevar from H5 dict."""
    return _cast_selected(tv, ("mean_noise_var",))


def build_rsrp_kwargs(tv: dict) -> dict:
    """Explicit kwargs for ran.phy_ref.pusch.rsrp from H5 dict."""
    return _cast_selected(tv, ("h_est", "layer2ue", "n_ue"))


def build_sinr_kwargs(tv: dict) -> dict:
    """Explicit kwargs for ran.phy_ref.pusch.sinr from H5 dict."""
    return _cast_selected(tv, ("rsrp_db", "noise_var_db"))


def build_noise_rsrp_sinr_kwargs(tv: dict) -> dict:
    """Explicit kwargs for ran.phy.numpy.pusch.noise_rsrp_sinr_db from H5 dict."""
    keys = ("mean_noise_var", "h_est", "layer2ue", "n_ue")
    return _cast_selected(tv, keys)


def build_eq_derive_kwargs(tv: dict) -> dict:
    """Explicit kwargs for ran.phy_ref.pusch.eq_derive from H5 dict."""
    return _cast_selected(tv, ("h_est", "noise_intf_cov"))


def build_eq_apply_kwargs(tv: dict) -> dict:
    """Explicit kwargs for ran.phy_ref.pusch.eq_apply from H5 dict."""
    return _cast_selected(tv, ("xtf", "w", "sym_idx_data"))


def build_eq_equalizer_kwargs(tv: dict) -> dict:
    """Explicit kwargs for ran.phy.numpy.pusch.equalize from H5 dict."""
    keys = ("h_est", "noise_intf_cov", "xtf", "sym_idx_data")
    return _cast_selected(tv, keys)


def build_post_eq_kwargs(tv: dict) -> dict:
    """Explicit kwargs for ran.phy_ref.pusch.post_eq from H5 dict."""
    return _cast_selected(tv, ("ree", "start_prb", "n_prb", "layer2ue", "n_ue"))


def build_softdemap_kwargs(tv: dict) -> dict:
    """Explicit kwargs for ran.phy_ref.pusch.softdemap from H5 dict."""
    keys = ("x_est", "ree", "n_prb", "start_prb", "qam_bits")
    return _cast_selected(tv, keys)


def build_descramble_kwargs(tv: dict) -> dict:
    """Explicit kwargs for ran.phy_ref.pusch.descramble from H5 dict."""
    return _cast_selected(tv, ("llrseq", "n_id", "n_rnti"))


def build_derate_kwargs(tv: dict) -> dict:
    """Explicit kwargs for ran.phy_ref.pusch.derate from H5 dict."""
    keys = (
        "llr_descr",
        "bgn",
        "c",
        "qam_bits",
        "k",
        "f",
        "k_prime",
        "zc",
        "nl",
        "rv_idx",
        "nref",
        "g",
    )
    return _cast_selected(tv, keys)


def build_ldpc_kwargs(tv: dict) -> dict:
    """Explicit kwargs for ran.phy_ref.pusch.ldpc from H5 dict."""
    keys = ("derate_cbs", "nv_parity", "zc", "c", "bgn", "i_ls", "max_num_itr_cbs")
    return _cast_selected(tv, keys)


def build_cb_concat_kwargs(tv: dict) -> dict:
    """Explicit kwargs for ran.phy_ref.pusch.concat from H5 dict."""
    return _cast_selected(tv, ("tb_cbs_est", "c", "k_prime"))


def build_crc_kwargs(tv: dict) -> dict:
    """Explicit kwargs for ran.phy_ref.pusch.crc from H5 dict."""
    return _cast_selected(tv, ("tb_crc_est", "crc_name"))


def build_dmrs_rssi_kwargs(tv: dict) -> dict:
    """Explicit kwargs for ran.phy_ref.pusch.dmrs_rssi from H5 dict."""
    keys = ("xtf", "start_prb", "n_prb", "sym_idx_dmrs")
    return _cast_selected(tv, keys)


def build_pusch_rx_kwargs(tv: dict) -> dict:
    """Aggregate all builder outputs into a single dict for pusch_rx.

    Keys match the respective function signatures used across the pipeline.
    """
    keys = (
        "slot_number",
        "n_f",
        "n_dmrs_id",  # Block 1: DMRS gen
        "xtf",
        "nl",
        "port_idx",
        "vec_scid",
        "sym_idx_dmrs",
        "n_prb",
        "start_prb",  # Block 2: LS
        "energy",  # Block 4.1: Embed DMRS
        "rww_regularizer_val",  # Block 4.3: Noise covariance
        "sym_idx_data",  # Block 6.2: Apply equalizer
        "layer2ue",
        "n_ue",  # Block 7: Post-EQ noise var and SINR
        "qam_bits",  # Block 8: Soft demapper
        "n_id",
        "n_rnti",  # Block 9: Descramble
        "bgn",
        "c",
        "k",
        "f",
        "k_prime",
        "zc",
        "rv_idx",
        "nref",
        "g",  # Block 10: Derate
        "nv_parity",
        "i_ls",
        "max_num_itr_cbs",  # Block 11: LDPC
        "crc_name",  # Block 13: CRC
    )
    return _cast_selected(tv, keys)


def build_pusch_inner_rx_kwargs(tv: dict) -> dict:
    """Aggregate all builder outputs into a single dict for pusch_inner_rx."""
    keys = (
        "slot_number",
        "n_dmrs_id",
        "xtf",
        "nl",
        "port_idx",
        "vec_scid",
        "sym_idx_dmrs",
        "n_prb",
        "start_prb",
        "energy",
        "rww_regularizer_val",
        "sym_idx_data",
        "layer2ue",
        "n_ue",
        "qam_bits",
    )
    return _cast_selected(tv, keys)


__all__ = (
    "build_cb_concat_kwargs",
    "build_cov_kwargs",
    "build_crc_kwargs",
    "build_dmrs_kwargs",
    "build_dmrs_rssi_kwargs",
    "build_embed_dmrs_kwargs",
    "build_eq_apply_kwargs",
    "build_eq_derive_kwargs",
    "build_eq_equalizer_kwargs",
    "build_ldpc_kwargs",
    "build_ls_kwargs",
    "build_ncov_kwargs",
    "build_noise_rsrp_sinr_kwargs",
    "build_noisevar_kwargs",
    "build_pusch_inner_rx_kwargs",
    "build_r_tilde_kwargs",
    "build_rsrp_kwargs",
    "build_shrink_kwargs",
    "build_sinr_kwargs",
    "build_softdemap_kwargs",
)
