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

"""Test cuSOLVER Cholesky inversion with custom TensorRT plugin."""

from __future__ import annotations

import logging
import os

import numpy as np
import pytest

# Limit JAX GPU memory pre-allocation to prevent OOM issues
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.05"

# Only import JAX and internal dependencies when MLIR_TRT is enabled
if os.getenv("ENABLE_MLIR_TRT", "OFF") == "ON":
    import jax  # noqa: E402
    import jax.numpy as jnp  # noqa: E402

    from ran.trt_plugins.cholesky_factor_inv.cholesky_factor_inv_trt_plugin import (  # noqa: E402
        cholesky_factor_inv,
        CholeskyFactorInvTrtPlugin,
    )

# Check if we should skip slow tests (same env var as engine generation)
_skip_slow_tests = os.getenv("SKIP_TRT_ENGINE_GENERATION", "0") == "1"

# All tests in this module require MLIR_TRT to be enabled
pytestmark = [
    pytest.mark.skipif(
        os.getenv("ENABLE_MLIR_TRT", "OFF") != "ON",
        reason="Requires MLIR-TensorRT compiler (ENABLE_MLIR_TRT=OFF)",
    ),
    pytest.mark.skipif(
        _skip_slow_tests,
        reason="Skipping slow JAX plugin tests (SKIP_TRT_ENGINE_GENERATION=1)",
    ),
]

logger = logging.getLogger(__name__)

rtol = 1e-3
atol = 1e-3


def generate_psd_matrix(n: int, seed: int = 0) -> np.ndarray:
    """Generate a positive semi-definite matrix for testing.

    Args:
        n: Matrix dimension (n x n)
        seed: Random seed for reproducibility

    Returns:
        Positive semi-definite matrix of shape (n, n)
    """
    key = jax.random.PRNGKey(seed)
    a = jax.random.normal(key, (n, n))
    # Make symmetric positive semi-definite: A = A^T A
    psd = jnp.dot(a.T, a)
    return np.array(psd, dtype=np.float32)


def generate_complex_psd_matrix(n: int, seed: int = 0) -> np.ndarray:
    """Generate a complex positive semi-definite Hermitian matrix.

    Args:
        n: Matrix dimension (n x n)
        seed: Random seed for reproducibility

    Returns:
        Complex Hermitian positive semi-definite matrix of shape (n, n)
    """
    key = jax.random.PRNGKey(seed)
    # Split key for real and imaginary parts
    key_real, key_imag = jax.random.split(key)
    a_real = jax.random.normal(key_real, (n, n))
    a_imag = jax.random.normal(key_imag, (n, n))
    a = a_real + 1j * a_imag

    # Make Hermitian positive semi-definite: A = A^H A
    psd = jnp.dot(jnp.conj(a.T), a)
    return np.array(psd, dtype=np.complex64)


@pytest.mark.parametrize("matrix_size", [2, 4, 8])
def test_cholesky_inv_real(matrix_size: int) -> None:
    """Test Cholesky inversion for real matrices of various sizes.

    Args:
        matrix_size: Size of the matrix (matrix_size x matrix_size)
    """
    # Generate test matrix
    cov = generate_psd_matrix(n=matrix_size)

    # JAX reference Cholesky inversion
    l_chol_ref = jax.numpy.linalg.cholesky(cov)
    l_inv_ref = np.linalg.inv(l_chol_ref)

    # cuSOLVERDx Cholesky inversion using TensorRT plugin
    l_inv = cholesky_factor_inv(jnp.asarray(cov))

    # Verify results match reference implementation
    np.testing.assert_allclose(l_inv, l_inv_ref, rtol=rtol, atol=atol)

    logger.info(f"Real {matrix_size}x{matrix_size} Cholesky inversion test passed")


@pytest.mark.parametrize("matrix_size", [2, 4, 8])
def test_cholesky_inv_complex(matrix_size: int) -> None:
    """Test Cholesky inversion for complex matrices of various sizes.

    Args:
        matrix_size: Size of the matrix (matrix_size x matrix_size)
    """
    # Generate test matrix
    cov_complex = generate_complex_psd_matrix(n=matrix_size)

    # JAX reference Cholesky inversion
    l_chol_complex_ref = jax.numpy.linalg.cholesky(cov_complex)
    l_inv_complex_ref = np.linalg.inv(l_chol_complex_ref)

    # Split complex input into real/imag for TensorRT plugin
    cov_real = jnp.real(cov_complex).astype(jnp.float32)
    cov_imag = jnp.imag(cov_complex).astype(jnp.float32)

    # cuSOLVERDx Cholesky inversion using TensorRT plugin
    l_inv_real, l_inv_imag = cholesky_factor_inv(cov_real, cov_imag)
    l_inv_complex = l_inv_real + 1j * l_inv_imag

    # Verify results match reference implementation
    try:
        np.testing.assert_allclose(l_inv_complex, l_inv_complex_ref, rtol=rtol, atol=atol)
    except AssertionError:
        logger.error(f"l_inv_complex:\n{l_inv_complex}")
        logger.error(f"l_inv_complex_ref:\n{l_inv_complex_ref}")
        logger.error(f"Difference:\n{l_inv_complex - l_inv_complex_ref}")
        raise

    logger.info(f"Complex {matrix_size}x{matrix_size} Cholesky inversion test passed")


def test_cholesky_inv_singleton_usage() -> None:
    """Test using the singleton plugin instances (complex)."""
    from ran.trt_plugins.cholesky_factor_inv.cholesky_factor_inv_trt_plugin import (
        cholesky_inv_2x2,
        cholesky_inv_4x4,
        cholesky_inv_8x8,
    )

    # Test 2x2 complex
    cov_2x2 = generate_complex_psd_matrix(n=2)
    l_chol_2x2_ref = jax.numpy.linalg.cholesky(cov_2x2)
    l_inv_2x2_ref = np.linalg.inv(l_chol_2x2_ref)
    cov_2x2_real = np.real(cov_2x2).astype(np.float32)
    cov_2x2_imag = np.imag(cov_2x2).astype(np.float32)
    l_inv_2x2_real, l_inv_2x2_imag = cholesky_inv_2x2(
        jnp.asarray(cov_2x2_real), jnp.asarray(cov_2x2_imag)
    )
    l_inv_2x2 = l_inv_2x2_real + 1j * l_inv_2x2_imag
    np.testing.assert_allclose(l_inv_2x2, l_inv_2x2_ref, rtol=rtol, atol=atol)

    # Test 4x4 complex
    cov_4x4 = generate_complex_psd_matrix(n=4)
    l_chol_4x4_ref = jax.numpy.linalg.cholesky(cov_4x4)
    l_inv_4x4_ref = np.linalg.inv(l_chol_4x4_ref)
    cov_4x4_real = np.real(cov_4x4).astype(np.float32)
    cov_4x4_imag = np.imag(cov_4x4).astype(np.float32)
    l_inv_4x4_real, l_inv_4x4_imag = cholesky_inv_4x4(
        jnp.asarray(cov_4x4_real), jnp.asarray(cov_4x4_imag)
    )
    l_inv_4x4 = l_inv_4x4_real + 1j * l_inv_4x4_imag
    np.testing.assert_allclose(l_inv_4x4, l_inv_4x4_ref, rtol=rtol, atol=atol)

    # Test 8x8 complex
    cov_8x8 = generate_complex_psd_matrix(n=8)
    l_chol_8x8_ref = jax.numpy.linalg.cholesky(cov_8x8)
    l_inv_8x8_ref = np.linalg.inv(l_chol_8x8_ref)
    cov_8x8_real = np.real(cov_8x8).astype(np.float32)
    cov_8x8_imag = np.imag(cov_8x8).astype(np.float32)
    l_inv_8x8_real, l_inv_8x8_imag = cholesky_inv_8x8(
        jnp.asarray(cov_8x8_real), jnp.asarray(cov_8x8_imag)
    )
    l_inv_8x8 = l_inv_8x8_real + 1j * l_inv_8x8_imag
    np.testing.assert_allclose(l_inv_8x8, l_inv_8x8_ref, rtol=rtol, atol=atol)

    logger.info("Singleton plugin usage test passed")


def test_cholesky_inv_dtype_validation() -> None:
    """Test that plugin validates input dtype correctly."""
    # Real plugin should work with float32
    plugin_real = CholeskyFactorInvTrtPlugin(matrix_size=2, is_complex=False, name="test_real")
    cov_real = generate_psd_matrix(n=2)
    result = plugin_real(jnp.asarray(cov_real))
    assert result.dtype == np.float32  # type: ignore[union-attr]

    # Complex plugin should work with separate real/imag float32 arrays
    plugin_complex = CholeskyFactorInvTrtPlugin(matrix_size=2, is_complex=True, name="test_complex")
    cov_complex = generate_complex_psd_matrix(n=2)

    # Split complex input into real/imag components
    cov_real = np.real(cov_complex).astype(np.float32)
    cov_imag = np.imag(cov_complex).astype(np.float32)

    # Call plugin with separate real/imag inputs
    result_real, result_imag = plugin_complex(jnp.asarray(cov_real), jnp.asarray(cov_imag))

    # Verify both outputs are float32
    assert result_real.dtype == np.float32
    assert result_imag.dtype == np.float32

    # Recombine to verify complex result
    result_complex = result_real + 1j * result_imag
    assert result_complex.dtype == np.complex64

    logger.info("Dtype validation test passed")


@pytest.mark.parametrize("matrix_size", [2, 4, 8])
def test_cholesky_inv_complex_realimag_interface(matrix_size: int) -> None:
    """Test complex Cholesky inversion using explicit real/imag interface.

    This test demonstrates the recommended pattern for TensorRT compilation:
    explicitly splitting complex inputs into real/imag components.

    Args:
        matrix_size: Size of the matrix (matrix_size x matrix_size)
    """
    # Generate test matrix
    cov_complex = generate_complex_psd_matrix(n=matrix_size)

    # Split complex input into real/imag components
    cov_real = jnp.real(cov_complex).astype(jnp.float32)
    cov_imag = jnp.imag(cov_complex).astype(jnp.float32)

    # Create complex plugin and call with separate real/imag inputs
    plugin = CholeskyFactorInvTrtPlugin(
        matrix_size=matrix_size, is_complex=True, name=f"test_cholesky_inv_{matrix_size}"
    )
    l_inv_real, l_inv_imag = plugin(cov_real, cov_imag)

    # Recombine result
    l_inv_complex = l_inv_real + 1j * l_inv_imag

    # JAX reference Cholesky inversion
    l_chol_complex_ref = jax.numpy.linalg.cholesky(cov_complex)
    l_inv_complex_ref = np.linalg.inv(l_chol_complex_ref)

    # Verify results match reference implementation
    np.testing.assert_allclose(l_inv_complex, l_inv_complex_ref, rtol=rtol, atol=atol)

    logger.info(
        f"Complex {matrix_size}x{matrix_size} Cholesky inversion test (real/imag interface) passed"
    )


def test_cholesky_inv_complex_8x8_from_main() -> None:
    """Test 8x8 complex Cholesky inversion.

    Verifies cholesky_factor_inv() against JAX reference implementation.
    """
    # Test parameters
    cov_complex = generate_complex_psd_matrix(n=8)

    # JAX reference Cholesky inversion
    l_chol_ref = jax.numpy.linalg.cholesky(cov_complex)
    l_inv_ref = np.linalg.inv(l_chol_ref)

    # Split complex input into real/imag for TensorRT plugin
    cov_real = jnp.real(cov_complex).astype(jnp.float32)
    cov_imag = jnp.imag(cov_complex).astype(jnp.float32)

    # cuSOLVERDx Cholesky inversion using TensorRT plugin
    l_inv_real, l_inv_imag = cholesky_factor_inv(cov_real, cov_imag)
    l_inv_complex = l_inv_real + 1j * l_inv_imag

    # Verify results match reference implementation
    np.testing.assert_allclose(l_inv_complex, l_inv_ref, rtol=1e-3, atol=1e-3)

    logger.info("8x8 complex Cholesky inversion test passed")
