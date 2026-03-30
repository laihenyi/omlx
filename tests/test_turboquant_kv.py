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
        """Random sign flip should produce +/-1 values."""
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


class TestPolarQuantCodec:
    """Tests for PolarQuant codec."""

    def test_quantize_dequantize_roundtrip(self):
        """Quantize then dequantize should approximate original."""
        from omlx.turboquant_kv import PolarQuantCodec

        mx.random.seed(42)
        vectors = mx.random.normal(shape=(2, 4, 100, 128))  # B, H, T, D

        for bits in [2, 3, 4]:
            codec = PolarQuantCodec(dim=128, bits=bits, group_size=32)
            scales, zeros, packed = codec.quantize(vectors)
            restored = codec.dequantize(scales, zeros, packed)

            mse = float(mx.mean((vectors - restored) ** 2))
            # Higher bits = lower error
            # Polar codec uses per-group min/max, which has higher error than codebook approach
            max_mse = {2: 1.0, 3: 0.2, 4: 0.05}[bits]
            assert mse < max_mse, f"bits={bits}, MSE={mse} > {max_mse}"

    def test_asymmetric_bits(self):
        """K and V can use different bit widths."""
        from omlx.turboquant_kv import PolarQuantCodec

        k_codec = PolarQuantCodec(dim=128, bits=4, group_size=32)
        v_codec = PolarQuantCodec(dim=128, bits=2, group_size=32)

        mx.random.seed(42)
        vectors = mx.random.normal(shape=(1, 2, 50, 128))

        k_scales, k_zeros, k_packed = k_codec.quantize(vectors)
        v_scales, v_zeros, v_packed = v_codec.quantize(vectors)

        # Packed width differs by bits
        assert k_packed.shape[-1] != v_packed.shape[-1]

    def test_group_size_adaptation(self):
        """2-bit should use smaller group_size by default."""
        from omlx.turboquant_kv import PolarQuantCodec

        codec_2bit = PolarQuantCodec(dim=128, bits=2)
        codec_4bit = PolarQuantCodec(dim=128, bits=4)

        assert codec_2bit.group_size == 32  # Smaller for 2-bit
        assert codec_4bit.group_size == 64  # Default

    def test_packed_width(self):
        """Verify packed width calculation."""
        from omlx.turboquant_kv import _packed_width

        for bits in [2, 3, 4]:
            for group_size in [32, 64]:
                pw = _packed_width(group_size, bits)
                # pw = ceil(group_size * bits / 32)
                expected = (group_size * bits + 31) // 32
                assert pw == expected, f"bits={bits}, group={group_size}"
