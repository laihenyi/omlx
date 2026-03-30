# SPDX-License-Identifier: Apache-2.0
"""Integration tests for TurboQuant+ KV cache."""

import pytest
import mlx.core as mx


class TestTurboQuantIntegration:
    """End-to-end tests for TurboQuant+ with various configurations."""

    def test_turbo4_symmetric(self):
        """Test turbo4 symmetric configuration."""
        from omlx.turboquant_kv import TurboQuantKVCache

        cache = TurboQuantKVCache(k_bits=4, v_bits=4, sparse_v=False)

        mx.random.seed(42)
        keys = mx.random.normal(shape=(1, 4, 500, 128))
        values = mx.random.normal(shape=(1, 4, 500, 128))

        k_out, v_out = cache.update_and_fetch(keys, values)
        assert k_out.shape == keys.shape

        # Decode
        k_dec = mx.random.normal(shape=(1, 4, 1, 128))
        v_dec = mx.random.normal(shape=(1, 4, 1, 128))
        cache.update_and_fetch(k_dec, v_dec)

        assert cache._quantized
        assert cache.offset == 501

    def test_turbo3_k_turbo4_v_asymmetric(self):
        """Test asymmetric K=3, V=4 configuration."""
        from omlx.turboquant_kv import TurboQuantKVCache

        cache = TurboQuantKVCache(k_bits=3, v_bits=4, sparse_v=False)

        mx.random.seed(42)
        keys = mx.random.normal(shape=(1, 4, 1000, 128))
        values = mx.random.normal(shape=(1, 4, 1000, 128))

        cache.update_and_fetch(keys, values)

        # Decode triggers quantization
        k_dec = mx.random.normal(shape=(1, 4, 1, 128))
        v_dec = mx.random.normal(shape=(1, 4, 1, 128))
        cache.update_and_fetch(k_dec, v_dec)

        assert cache._k_codec.bits == 3
        assert cache._v_codec.bits == 4

    def test_turbo2_memory_constrained(self):
        """Test turbo2 for extreme memory constraints."""
        from omlx.turboquant_kv import TurboQuantKVCache

        cache = TurboQuantKVCache(k_bits=2, v_bits=2, sparse_v=False)

        mx.random.seed(42)
        keys = mx.random.normal(shape=(1, 4, 2000, 128))
        values = mx.random.normal(shape=(1, 4, 2000, 128))

        cache.update_and_fetch(keys, values)

        # Decode
        k_dec = mx.random.normal(shape=(1, 4, 1, 128))
        v_dec = mx.random.normal(shape=(1, 4, 1, 128))
        cache.update_and_fetch(k_dec, v_dec)

        # Memory should be significantly reduced
        nbytes = cache.nbytes
        # fp16: 2 bytes * 4 heads * 2001 tokens * 128 dim * 2 (K+V) = ~4MB
        fp16_bytes = 2 * 4 * 2001 * 128 * 2
        compression_ratio = fp16_bytes / nbytes
        assert compression_ratio > 5.0  # At least 5x compression for 2-bit

    def test_decode_attention_output_shape(self):
        """Test that decode_attention produces correct output shape."""
        from omlx.turboquant_kv import TurboQuantKVCache

        cache = TurboQuantKVCache(k_bits=4, v_bits=3, sparse_v=False)

        mx.random.seed(42)
        keys = mx.random.normal(shape=(1, 8, 1000, 64))  # 8 kv heads
        values = mx.random.normal(shape=(1, 8, 1000, 64))
        cache.update_and_fetch(keys, values)

        # Decode
        k_dec = mx.random.normal(shape=(1, 8, 1, 64))
        v_dec = mx.random.normal(shape=(1, 8, 1, 64))
        cache.update_and_fetch(k_dec, v_dec)

        # Query
        q = mx.random.normal(shape=(1, 8, 1, 64))
        out = cache.decode_attention(q)
        assert out.shape == (1, 8, 1, 64)
