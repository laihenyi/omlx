# SPDX-License-Identifier: Apache-2.0
"""Tests for TurboQuant+ KV cache."""

import importlib.util
import math
import pytest
import numpy as np
import mlx.core as mx

# Import turboquant_kv directly to avoid triggering full omlx package import
_spec = importlib.util.spec_from_file_location(
    "turboquant_kv",
    "/Users/laihenyi/Documents/GitHub/omlx/omlx/turboquant_kv.py"
)
tq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tq)


class TestWalshHadamard:
    """Tests for Walsh-Hadamard rotation."""

    def test_wht_orthogonality(self):
        """WHT matrix should be orthogonal: H @ H.T = n * I."""
        for n in [64, 128, 256]:
            H = tq._wht_matrix(n)
            product = H @ H.T
            expected = n * mx.eye(n)
            assert mx.allclose(product, expected, atol=1e-5), f"Failed for n={n}"

    def test_wht_deterministic(self):
        """WHT should be deterministic (same input -> same output)."""
        H1 = tq._wht_matrix(128)
        H2 = tq._wht_matrix(128)
        assert mx.allclose(H1, H2)

    def test_random_sign_flip(self):
        """Random sign flip should produce ±1 values."""
        signs = tq._random_sign_flip(128, seed=42)
        assert signs.shape == (128,)
        assert mx.all((signs == 1.0) | (signs == -1.0))

        # Different seeds produce different signs
        signs2 = tq._random_sign_flip(128, seed=43)
        assert not mx.allclose(signs, signs2)

    def test_wht_rotation_preserves_norm(self):
        """WHT rotation should preserve vector norm."""
        mx.random.seed(42)
        v = mx.random.normal(shape=(128,))
        original_norm = float(mx.linalg.norm(v))

        rotated = tq.apply_wht_rotation(v, seed=0)
        rotated_norm = float(mx.linalg.norm(rotated))

        assert abs(original_norm - rotated_norm) < 1e-4, f"Norm changed: {original_norm} -> {rotated_norm}"

    def test_wht_rotation_inverse(self):
        """WHT rotation should be invertible."""
        mx.random.seed(42)
        v = mx.random.normal(shape=(128,))

        rotated = tq.apply_wht_rotation(v, seed=0)
        restored = tq.apply_wht_inverse(rotated, seed=0)

        assert mx.allclose(v, restored, atol=1e-4), "Inverse did not restore original"
