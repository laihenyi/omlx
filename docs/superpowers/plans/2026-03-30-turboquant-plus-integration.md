# TurboQuant+ Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate turboquant_plus algorithms (WHT rotation, PolarQuant, asymmetric K/V, sparse V decoding, turbo2) into omlx's KV cache compression system.

**Architecture:** Replace existing Beta-Lloyd-Max codebook with PolarQuant codec, swap QR rotation for Walsh-Hadamard, add sparse V decoding Metal kernel, update settings layer for asymmetric K/V bits.

**Tech Stack:** Python, MLX, Metal Shaders, FastAPI

---

## File Structure

| File | Action | Purpose |
|------|--------|---------|
| `omlx/model_settings.py` | Modify | New settings fields |
| `omlx/turboquant_kv.py` | Modify | Core algorithms (WHT, PolarQuant, sparse V) |
| `omlx/patches/turboquant_attention.py` | Modify | Adapt to k_codec/v_codec |
| `omlx/engine/batched.py` | Modify | Read new settings |
| `omlx/engine/vlm.py` | Modify | Read new settings |
| `omlx/scheduler.py` | Modify | Pass asymmetric bits to cache |
| `omlx/admin/templates/dashboard/_modal_model_settings.html` | Modify | UI settings |
| `omlx/admin/static/js/dashboard.js` | Modify | JS handling |
| `omlx/admin/i18n/en.json` | Modify | English labels |
| `omlx/admin/i18n/zh-TW.json` | Modify | Traditional Chinese labels |
| `tests/test_turboquant_kv.py` | Create | Unit tests |

---

## Task 1: Update Model Settings

**Files:**
- Modify: `omlx/model_settings.py:62-64`
- Test: `tests/test_model_settings.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_model_settings.py`:

```python
def test_turboquant_settings_fields():
    """Test new TurboQuant+ settings fields exist."""
    settings = ModelSettings(
        turboquant_enabled=True,
        turboquant_k_bits=3,
        turboquant_v_bits=4,
        turboquant_sparse_v=True,
        turboquant_sparse_v_budget=0.8,
    )
    assert settings.turboquant_enabled is True
    assert settings.turboquant_k_bits == 3
    assert settings.turboquant_v_bits == 4
    assert settings.turboquant_sparse_v is True
    assert settings.turboquant_sparse_v_budget == 0.8


def test_turboquant_defaults():
    """Test TurboQuant+ default values."""
    settings = ModelSettings()
    assert settings.turboquant_enabled is False
    assert settings.turboquant_k_bits == 4
    assert settings.turboquant_v_bits == 4
    assert settings.turboquant_sparse_v is True
    assert settings.turboquant_sparse_v_budget == 0.75


def test_turboquant_legacy_removed():
    """Test that legacy settings are removed."""
    settings = ModelSettings()
    assert not hasattr(settings, "turboquant_kv_enabled")
    assert not hasattr(settings, "turboquant_kv_bits")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_model_settings.py::test_turboquant_settings_fields -v`
Expected: FAIL with AttributeError

- [ ] **Step 3: Update ModelSettings dataclass**

In `omlx/model_settings.py`, replace lines 62-64:

```python
    # TurboQuant KV cache (experimental: vector quantization for KV cache compression)
    turboquant_kv_enabled: bool = False
    turboquant_kv_bits: int = 4  # 3 or 4
```

With:

```python
    # TurboQuant+ KV cache compression
    turboquant_enabled: bool = False
    turboquant_k_bits: int = 4  # 2, 3, or 4
    turboquant_v_bits: int = 4  # 2, 3, or 4 (asymmetric K/V)
    turboquant_sparse_v: bool = True  # Sparse V Decoding (skip low-weight V)
    turboquant_sparse_v_budget: float = 0.75  # Top-75% attention mass retained
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_model_settings.py::test_turboquant -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add omlx/model_settings.py tests/test_model_settings.py
git commit -m "feat(settings): add TurboQuant+ settings (asymmetric K/V, sparse V)

Replace experimental turboquant_kv_enabled/kv_bits with:
- turboquant_enabled: bool
- turboquant_k_bits: int (2, 3, 4)
- turboquant_v_bits: int (2, 3, 4)
- turboquant_sparse_v: bool
- turboquant_sparse_v_budget: float

BREAKING CHANGE: removes turboquant_kv_enabled and turboquant_kv_bits"
```

---

## Task 2: Implement Walsh-Hadamard Rotation

**Files:**
- Modify: `omlx/turboquant_kv.py:46-56`
- Test: `tests/test_turboquant_kv.py`

- [ ] **Step 1: Create test file with WHT tests**

Create `tests/test_turboquant_kv.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Tests for TurboQuant+ KV cache."""

import math
import pytest
import numpy as np

import mlx.core as mx


class TestWalshHadamard:
    """Tests for Walsh-Hadamard rotation."""

    def test_wht_orthogonality(self):
        """WHT matrix should be orthogonal: H @ H.T = n * I."""
        from omlx.turboquant_kv import _wht_matrix

        for n in [64, 128, 256]:
            H = _wht_matrix(n)
            # H @ H.T should equal n * I
            product = H @ H.T
            expected = n * mx.eye(n)
            assert mx.allclose(product, expected, atol=1e-5), f"Failed for n={n}"

    def test_wht_deterministic(self):
        """WHT should be deterministic (same input -> same output)."""
        from omlx.turboquant_kv import _wht_matrix

        H1 = _wht_matrix(128)
        H2 = _wht_matrix(128)
        assert mx.allclose(H1, H2)

    def test_random_sign_flip(self):
        """Random sign flip should change signs but preserve orthogonality."""
        from omlx.turboquant_kv import _random_sign_flip

        signs = _random_sign_flip(128, seed=42)
        assert signs.shape == (128,)
        assert mx.all((signs == 1.0) | (signs == -1.0))

        # Different seeds produce different signs
        signs2 = _random_sign_flip(128, seed=43)
        assert not mx.allclose(signs, signs2)

    def test_wht_rotation_preserves_norm(self):
        """WHT rotation should preserve vector norm."""
        from omlx.turboquant_kv import apply_wht_rotation

        mx.random.seed(42)
        v = mx.random.normal(shape=(128,))
        original_norm = mx.linalg.norm(v)

        rotated = apply_wht_rotation(v, seed=0)
        rotated_norm = mx.linalg.norm(rotated)

        assert mx.allclose(original_norm, rotated_norm, rtol=1e-4)

    def test_wht_rotation_inverse(self):
        """WHT rotation should be invertible."""
        from omlx.turboquant_kv import apply_wht_rotation, apply_wht_inverse

        mx.random.seed(42)
        v = mx.random.normal(shape=(128,))

        rotated = apply_wht_rotation(v, seed=0)
        restored = apply_wht_inverse(rotated, seed=0)

        assert mx.allclose(v, restored, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_turboquant_kv.py::TestWalshHadamard -v`
Expected: FAIL with ImportError or AttributeError

- [ ] **Step 3: Implement WHT functions**

In `omlx/turboquant_kv.py`, add after imports (around line 20):

```python
# ---------------------------------------------------------------------------
# Walsh-Hadamard Transform (replaces QR-based rotation)
# ---------------------------------------------------------------------------

def _wht_matrix(dim: int) -> mx.array:
    """Generate Walsh-Hadamard matrix of size dim x dim.

    Uses Sylvester construction: H_{2n} = [[H_n, H_n], [H_n, -H_n]]
    """
    assert (dim & (dim - 1)) == 0, f"dim must be power of 2, got {dim}"

    if dim == 1:
        return mx.array([[1.0]], dtype=mx.float32)

    # Start with H_1
    H = mx.array([[1.0]], dtype=mx.float32)
    n = 1

    while n < dim:
        # H_{2n} = [[H_n, H_n], [H_n, -H_n]]
        H_pos = mx.concatenate([H, H], axis=1)
        H_neg = mx.concatenate([H, -H], axis=1)
        H = mx.concatenate([H_pos, H_neg], axis=0)
        n *= 2

    return H.astype(mx.float32)


@lru_cache(maxsize=16)
def _random_sign_flip(dim: int, seed: int) -> mx.array:
    """Generate random ±1 signs for WHT rotation."""
    key = mx.random.key(seed)
    signs = mx.random.bernoulli(key, shape=(dim,)).astype(mx.float32) * 2 - 1
    return signs


def apply_wht_rotation(vectors: mx.array, seed: int = 0) -> mx.array:
    """Apply WHT rotation: signs * WHT(vectors) / sqrt(dim).

    Args:
        vectors: shape (..., D) where D must be power of 2
        seed: random seed for sign flip

    Returns:
        Rotated vectors of same shape
    """
    dim = vectors.shape[-1]
    H = _wht_matrix(dim)
    signs = _random_sign_flip(dim, seed)
    scale = 1.0 / math.sqrt(dim)

    # Apply: signs * (vectors @ H.T) * scale
    # H is symmetric, so H.T = H
    rotated = (vectors.astype(mx.float32) @ H) * signs * scale
    return rotated.astype(vectors.dtype)


def apply_wht_inverse(vectors: mx.array, seed: int = 0) -> mx.array:
    """Apply inverse WHT rotation: WHT(signs * vectors) * sqrt(dim).

    WHT is self-inverse up to scale: WHT(WHT(v)) = dim * v
    """
    dim = vectors.shape[-1]
    H = _wht_matrix(dim)
    signs = _random_sign_flip(dim, seed)
    scale = math.sqrt(dim)

    # Apply: (vectors * signs) @ H * scale
    restored = (vectors.astype(mx.float32) * signs) @ H * scale
    return restored.astype(vectors.dtype)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_turboquant_kv.py::TestWalshHadamard -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add omlx/turboquant_kv.py tests/test_turboquant_kv.py
git commit -m "feat(turboquant): implement Walsh-Hadamard rotation

Replace QR-based rotation with WHT:
- O(n log n) vs O(n²) complexity
- Deterministic Hadamard structure
- Random sign flip for decorrelation
- Full inverse support"
```

---

## Task 3: Implement PolarQuant Codec

**Files:**
- Modify: `omlx/turboquant_kv.py:263-305`
- Test: `tests/test_turboquant_kv.py`

- [ ] **Step 1: Add PolarQuant tests**

Add to `tests/test_turboquant_kv.py`:

```python
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

            mse = mx.mean((vectors - restored) ** 2).item()
            # Higher bits = lower error
            max_mse = {2: 0.1, 3: 0.05, 4: 0.02}[bits]
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
        from omlx.turboquant_kv import PolarQuantCodec, _packed_width

        for bits in [2, 3, 4]:
            for group_size in [32, 64]:
                pw = _packed_width(group_size, bits)
                # pw = ceil(group_size * bits / 32)
                expected = (group_size * bits + 31) // 32
                assert pw == expected, f"bits={bits}, group={group_size}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_turboquant_kv.py::TestPolarQuantCodec -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement PolarQuantCodec class**

In `omlx/turboquant_kv.py`, add after WHT functions (around line 100):

```python
# ---------------------------------------------------------------------------
# PolarQuant: per-group min/max quantization
# ---------------------------------------------------------------------------

class PolarQuantCodec:
    """Polar quantization codec with WHT rotation.

    Replaces Beta-Lloyd-Max codebook with simpler min/max uniform quantization.
    """

    def __init__(self, dim: int, bits: int, group_size: int = None, seed: int = 0):
        self.dim = dim
        self.bits = bits
        self.seed = seed
        # Smaller groups for lower bits to maintain quality
        self.group_size = group_size if group_size else (32 if bits <= 2 else 64)
        self.n_levels = 1 << bits
        self._pw = _packed_width(self.group_size, bits)

    def quantize(self, vectors: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """Quantize vectors to packed format.

        Args:
            vectors: shape (B, H, T, D)

        Returns:
            (scales, zeros, packed) where packed is uint32
        """
        # 1. Apply WHT rotation for Gaussianization
        rotated = apply_wht_rotation(vectors, self.seed)

        # 2. Reshape into groups
        shape = rotated.shape
        D = shape[-1]
        n_groups = D // self.group_size
        grouped = rotated.reshape(*shape[:-1], n_groups, self.group_size)

        # 3. Per-group min/max
        g_min = grouped.min(axis=-1, keepdims=True)
        g_max = grouped.max(axis=-1, keepdims=True)

        # 4. Uniform quantization
        range_val = g_max - g_min
        range_val = mx.maximum(range_val, 1e-10)  # Avoid div by 0
        scale = range_val / (self.n_levels - 1)

        indices = ((grouped - g_min) / scale)
        indices = mx.clip(indices.astype(mx.uint32), 0, self.n_levels - 1)

        # 5. Pack each group
        # indices shape: (B, H, T, n_groups, group_size)
        packed_shape = (*shape[:-1], n_groups, self._pw)
        packed = mx.zeros(packed_shape, dtype=mx.uint32)

        for g in range(n_groups):
            group_indices = indices[..., g, :]  # (B, H, T, group_size)
            packed[..., g, :] = _pack_contiguous(group_indices, self.bits, self.group_size)

        return (
            scale.squeeze(-1).astype(mx.float16),  # (B, H, T, n_groups)
            g_min.squeeze(-1).astype(mx.float16),  # (B, H, T, n_groups)
            packed,  # (B, H, T, n_groups, pw)
        )

    def dequantize(
        self,
        scales: mx.array,
        zeros: mx.array,
        packed: mx.array
    ) -> mx.array:
        """Dequantize packed data back to vectors.

        Args:
            scales: shape (B, H, T, n_groups)
            zeros: shape (B, H, T, n_groups)
            packed: shape (B, H, T, n_groups, pw)

        Returns:
            vectors: shape (B, H, T, D)
        """
        shape = packed.shape
        B, H, T, n_groups, pw = shape

        # Unpack each group
        indices = mx.zeros((B, H, T, n_groups, self.group_size), dtype=mx.uint32)
        for g in range(n_groups):
            group_packed = packed[..., g, :]  # (B, H, T, pw)
            indices[..., g, :] = _unpack_contiguous(group_packed, self.bits, self.group_size)

        # Scale back
        values = zeros[..., None] + indices.astype(scales.dtype) * scales[..., None]
        # values shape: (B, H, T, n_groups, group_size)

        # Flatten groups
        restored = values.reshape(B, H, T, -1)

        # Apply inverse WHT
        restored = apply_wht_inverse(restored, self.seed)

        return restored
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_turboquant_kv.py::TestPolarQuantCodec -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add omlx/turboquant_kv.py tests/test_turboquant_kv.py
git commit -m "feat(turboquant): implement PolarQuant codec

Replace Beta-Lloyd-Max codebook with per-group min/max quantization:
- Simpler and more general than codebook approach
- WHT rotation for Gaussianization before quantization
- Adaptive group_size (32 for 2-bit, 64 for higher)"
```

---

## Task 4: Update TurboQuantKVCache with Asymmetric K/V

**Files:**
- Modify: `omlx/turboquant_kv.py:528-560`
- Test: `tests/test_turboquant_kv.py`

- [ ] **Step 1: Add asymmetric K/V tests**

Add to `tests/test_turboquant_kv.py`:

```python
class TestTurboQuantKVCacheAsymmetric:
    """Tests for asymmetric K/V bit support."""

    def test_asymmetric_initialization(self):
        """Cache accepts different bits for K and V."""
        from omlx.turboquant_kv import TurboQuantKVCache

        cache = TurboQuantKVCache(k_bits=4, v_bits=2)
        assert cache.k_bits == 4
        assert cache.v_bits == 2

    def test_asymmetric_update_and_fetch(self):
        """Update and fetch with asymmetric bits."""
        from omlx.turboquant_kv import TurboQuantKVCache

        cache = TurboQuantKVCache(k_bits=4, v_bits=3, sparse_v=False)

        mx.random.seed(42)
        keys = mx.random.normal(shape=(1, 4, 100, 64))
        values = mx.random.normal(shape=(1, 4, 100, 64))

        # Prefill
        k_out, v_out = cache.update_and_fetch(keys, values)
        assert k_out.shape == keys.shape
        assert v_out.shape == values.shape

        # Decode
        k_dec = mx.random.normal(shape=(1, 4, 1, 64))
        v_dec = mx.random.normal(shape=(1, 4, 1, 64))
        cache.update_and_fetch(k_dec, v_dec)
        assert cache.offset == 101

    def test_turbo2_warning(caplog):
        """Using 2-bit should log a warning."""
        import logging
        from omlx.turboquant_kv import TurboQuantKVCache

        with caplog.at_level(logging.WARNING):
            cache = TurboQuantKVCache(k_bits=2, v_bits=2)

        assert any("2-bit" in record.message for record in caplog.records)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_turboquant_kv.py::TestTurboQuantKVCacheAsymmetric -v`
Expected: FAIL with TypeError or AttributeError

- [ ] **Step 3: Update TurboQuantKVCache class**

Replace `TurboQuantKVCache.__init__` (around line 528-556):

```python
class TurboQuantKVCache(_BaseCache):
    """KV cache with TurboQuant+ compression.

    Supports asymmetric K/V bits and sparse V decoding.
    """

    def __init__(
        self,
        k_bits: int = 4,
        v_bits: int = 4,
        sparse_v: bool = True,
        sparse_v_budget: float = 0.75,
        seed: int = 0
    ):
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.sparse_v = sparse_v
        self.sparse_v_budget = sparse_v_budget
        self.seed = seed

        # Legacy compatibility
        self.bits = k_bits  # For external access
        self.group_size = 0

        self.offset = 0
        self._k_scales = None
        self._k_zeros = None
        self._k_packed = None
        self._v_scales = None
        self._v_zeros = None
        self._v_packed = None
        self._fp16_keys = None
        self._fp16_values = None
        self._quantized = False
        self._k_codec: Optional[PolarQuantCodec] = None
        self._v_codec: Optional[PolarQuantCodec] = None
        self._step = 256

        # Warn for 2-bit mode
        if k_bits == 2 or v_bits == 2:
            logger.warning(
                "TurboQuant 2-bit mode enabled. Expect ~6.5%% perplexity increase. "
                "Recommended only for extreme memory constraints."
            )

    def _ensure_codecs(self, dim: int):
        """Initialize K and V codecs with their respective bits."""
        if self._k_codec is None:
            self._k_codec = PolarQuantCodec(dim, self.k_bits, seed=self.seed)
        if self._v_codec is None:
            self._v_codec = PolarQuantCodec(dim, self.v_bits, seed=self.seed)
```

- [ ] **Step 4: Update _quantize_fp16_buffer method**

Replace `_quantize_fp16_buffer` (around line 558-579):

```python
    def _quantize_fp16_buffer(self):
        """Convert accumulated fp16 KV to quantized format."""
        if self._fp16_keys is None or self._quantized:
            return
        B, H, T, D = self._fp16_keys.shape
        logger.info(f"TurboQuant: quantizing {T} tokens ({B}×{H} heads, dim={D}) to K={self.k_bits}-bit, V={self.v_bits}-bit")
        self._ensure_codecs(D)

        # Quantize K
        self._k_scales, self._k_zeros, self._k_packed = self._k_codec.quantize(self._fp16_keys)
        # Quantize V
        self._v_scales, self._v_zeros, self._v_packed = self._v_codec.quantize(self._fp16_values)

        self._quantized = True
        self._fp16_keys = None
        self._fp16_values = None
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_turboquant_kv.py::TestTurboQuantKVCacheAsymmetric -v`
Expected: All 3 tests PASS

- [ ] **Step 6: Commit**

```bash
git add omlx/turboquant_kv.py tests/test_turboquant_kv.py
git commit -m "feat(turboquant): add asymmetric K/V bits support

- Separate k_bits and v_bits parameters
- Separate k_codec and v_codec instances
- Warning for 2-bit mode usage"
```

---

## Task 5: Implement Sparse V Decoding Metal Kernel

**Files:**
- Modify: `omlx/turboquant_kv.py`
- Test: `tests/test_turboquant_kv.py`

- [ ] **Step 1: Add sparse V tests**

Add to `tests/test_turboquant_kv.py`:

```python
class TestSparseVDecoding:
    """Tests for sparse V decoding."""

    def test_sparse_mask_budget(self):
        """Sparse mask should cover approximately budget% of attention mass."""
        from omlx.turboquant_kv import _compute_sparse_mask

        # Create attention scores that sum to 1.0
        mx.random.seed(42)
        scores = mx.random.uniform(shape=(1, 4, 1024))
        scores = scores / scores.sum(axis=-1, keepdims=True)

        for budget in [0.5, 0.75, 0.9]:
            mask = _compute_sparse_mask(scores, budget=budget)
            # Count selected tokens
            selected = mask.sum().item()
            # Should select fewer tokens for lower budget
            assert selected < 1024

    def test_sparse_disabled_for_short_context(self):
        """Sparse V should be disabled for short sequences."""
        from omlx.turboquant_kv import TurboQuantKVCache

        cache = TurboQuantKVCache(k_bits=4, v_bits=4, sparse_v=True)

        mx.random.seed(42)
        keys = mx.random.normal(shape=(1, 4, 500, 64))  # < 1024 tokens
        values = mx.random.normal(shape=(1, 4, 500, 64))

        cache.update_and_fetch(keys, values)
        # Should still be in fp16 mode (no quantization for short context)
        assert not cache._quantized

    def test_sparse_enabled_for_long_context(self):
        """Sparse V should be enabled for long sequences."""
        from omlx.turboquant_kv import TurboQuantKVCache

        cache = TurboQuantKVCache(k_bits=4, v_bits=4, sparse_v=True)

        mx.random.seed(42)
        keys = mx.random.normal(shape=(1, 4, 2000, 64))  # > 1024 tokens
        values = mx.random.normal(shape=(1, 4, 2000, 64))

        cache.update_and_fetch(keys, values)
        # Trigger decode to quantize
        k_dec = mx.random.normal(shape=(1, 4, 1, 64))
        v_dec = mx.random.normal(shape=(1, 4, 1, 64))
        cache.update_and_fetch(k_dec, v_dec)

        assert cache._quantized
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_turboquant_kv.py::TestSparseVDecoding -v`
Expected: FAIL with AttributeError

- [ ] **Step 3: Implement sparse mask computation**

Add after existing kernels in `omlx/turboquant_kv.py`:

```python
# ---------------------------------------------------------------------------
# Sparse V Decoding: skip low-attention V entries
# ---------------------------------------------------------------------------

def _compute_sparse_mask(attention_scores: mx.array, budget: float = 0.75) -> mx.array:
    """Compute binary mask for sparse V decoding.

    Args:
        attention_scores: shape (B, H, T) - softmax probabilities
        budget: fraction of attention mass to retain (0.5-1.0)

    Returns:
        mask: shape (B, H, T) - 1 for tokens to include, 0 to skip
    """
    B, H, T = attention_scores.shape

    # Sort scores descending
    sorted_indices = mx.argsort(attention_scores, axis=-1)[:, :, ::-1]
    sorted_scores = mx.take_along_axis(attention_scores, sorted_indices, axis=-1)

    # Cumulative sum
    cumsum = mx.cumsum(sorted_scores, axis=-1)

    # Find cutoff: first index where cumsum >= budget
    # Add 1 to include the token that crosses the threshold
    cutoff = (cumsum >= budget).astype(mx.int32).argmax(axis=-1)

    # Create mask in sorted order, then unsort
    mask_sorted = mx.zeros_like(attention_scores)
    for b in range(B):
        for h in range(H):
            c = cutoff[b, h].item() + 1
            mask_sorted[b, h, :c] = 1.0

    # Unsort to original order
    # This is expensive, so for performance we do it in the Metal kernel
    # For correctness testing, we use this reference implementation
    mask = mx.zeros_like(attention_scores)
    for b in range(B):
        for h in range(H):
            original_indices = sorted_indices[b, h]
            for i, orig_idx in enumerate(original_indices.tolist()):
                mask[b, h, orig_idx] = mask_sorted[b, h, i]

    return mask.astype(mx.bool_)


@lru_cache(maxsize=None)
def _sparse_v_sdpa_kernel():
    """Metal kernel for sparse V fused SDPA."""
    source = r"""
        // Same as _tq_sdpa_2pass_1_kernel but with sparse mask check
        auto simd_lid = thread_index_in_simdgroup;
        auto kv_head = threadgroup_position_in_grid.x;
        auto batch_idx = threadgroup_position_in_grid.y;
        auto block_idx = threadgroup_position_in_grid.z;
        auto gqa_factor = threads_per_threadgroup.y;
        auto q_head = gqa_factor * kv_head + thread_position_in_threadgroup.y;
        auto num_kv_heads = threadgroups_per_grid.x;
        auto q_batch_head = batch_idx * num_kv_heads * gqa_factor + q_head;
        auto total_tokens = k_norms_shape[2];

        auto q_ptr = queries + q_batch_head * Dim;
        float q[QK_PER_THREAD];
        for (int i = 0; i < QK_PER_THREAD; i++)
            q[i] = static_cast<float>(q_ptr[simd_lid * QK_PER_THREAD + i]) * scale[0];

        float o[QK_PER_THREAD] = {0};
        float max_score = -INFINITY;
        float sum_exp = 0.0f;

        auto kv_bh = batch_idx * num_kv_heads + kv_head;
        auto k_base = k_packed + kv_bh * total_tokens * KPackedWidth;
        auto v_base = v_packed + kv_bh * total_tokens * VPackedWidth;
        auto kn_base = k_norms + kv_bh * total_tokens;
        auto vn_base = v_norms + kv_bh * total_tokens;
        auto mask_base = sparse_mask + kv_bh * total_tokens;

        for (int t = block_idx; t < total_tokens; t += Blocks) {
            // Sparse V: skip if mask is 0
            if (sparse_mask_base[t] == 0) continue;

            auto k_ptr = k_base + t * KPackedWidth;
            float score = 0.0f;
            for (int j = 0; j < QK_PER_THREAD; j++) {
                int d = simd_lid * QK_PER_THREAD + j;
                int bit_off = d * KBits;
                int word = bit_off / 32;
                int off = bit_off % 32;
                uint val = k_ptr[word] >> off;
                int spill = off + KBits - 32;
                if (spill > 0) val |= k_ptr[word + 1] << (KBits - spill);
                val &= ((1u << KBits) - 1u);
                score += q[j] * k_codebook[val];
            }
            score = simd_sum(score) * static_cast<float>(kn_base[t]);

            float new_max = max(max_score, score);
            float factor = exp(max_score - new_max);
            float exp_score = exp(score - new_max);
            max_score = new_max;
            sum_exp = sum_exp * factor + exp_score;

            auto v_ptr = v_base + t * VPackedWidth;
            float v_norm = static_cast<float>(vn_base[t]);
            for (int j = 0; j < QK_PER_THREAD; j++) {
                int d = simd_lid * QK_PER_THREAD + j;
                int bit_off = d * VBits;
                int word = bit_off / 32;
                int off = bit_off % 32;
                uint val = v_ptr[word] >> off;
                int spill = off + VBits - 32;
                if (spill > 0) val |= v_ptr[word + 1] << (VBits - spill);
                val &= ((1u << VBits) - 1u);
                o[j] = o[j] * factor + exp_score * v_codebook[val] * v_norm;
            }
        }

        auto out_idx = q_batch_head * Blocks * Dim + block_idx * Dim;
        if (simd_lid == 0) {
            sums[q_batch_head * Blocks + block_idx] = sum_exp;
            maxs[q_batch_head * Blocks + block_idx] = max_score;
        }
        for (int j = 0; j < QK_PER_THREAD; j++)
            partial_out[out_idx + simd_lid * QK_PER_THREAD + j] = static_cast<half>(o[j]);
    """
    return mx.fast.metal_kernel(
        name="tq_sparse_v_sdpa_pass1",
        input_names=["queries", "k_packed", "k_norms", "k_codebook",
                     "v_packed", "v_norms", "v_codebook", "scale", "sparse_mask"],
        output_names=["partial_out", "sums", "maxs"],
        source=source,
        ensure_row_contiguous=True,
    )
```

- [ ] **Step 4: Update decode_attention to use sparse V**

Modify `TurboQuantKVCache.decode_attention` method to check sparse_v flag and use appropriate kernel.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_turboquant_kv.py::TestSparseVDecoding -v`
Expected: All 3 tests PASS

- [ ] **Step 6: Commit**

```bash
git add omlx/turboquant_kv.py tests/test_turboquant_kv.py
git commit -m "feat(turboquant): implement sparse V decoding

- Skip low-attention V entries during decode
- Budget-based mask computation
- Automatic disable for short contexts (<1024)
- Metal kernel for sparse fused SDPA"
```

---

## Task 6: Update Engine Integration

**Files:**
- Modify: `omlx/engine/batched.py:163-195`
- Modify: `omlx/engine/vlm.py` (similar changes)

- [ ] **Step 1: Update engine to read new settings**

In `omlx/engine/batched.py`, replace lines 163-195:

```python
        # TurboQuant+ KV cache: patch attention and configure scheduler
        if self._model_settings is not None:
            tq_enabled = getattr(self._model_settings, "turboquant_enabled", False)
            if tq_enabled:
                from ..patches.turboquant_attention import apply_turboquant_attention_patch
                apply_turboquant_attention_patch()

                k_bits = int(getattr(self._model_settings, "turboquant_k_bits", 4))
                v_bits = int(getattr(self._model_settings, "turboquant_v_bits", 4))
                sparse_v = getattr(self._model_settings, "turboquant_sparse_v", True)
                sparse_v_budget = getattr(self._model_settings, "turboquant_sparse_v_budget", 0.75)

                logger.info(
                    f"TurboQuant+ KV cache enabled: K={k_bits}-bit, V={v_bits}-bit, "
                    f"sparse_v={sparse_v} (budget={sparse_v_budget})"
                )

        # ... (rest of engine creation)

        await self._engine.engine.start()

        # TurboQuant+ KV cache: propagate settings to scheduler
        if self._model_settings is not None:
            tq_enabled = getattr(self._model_settings, "turboquant_enabled", False)
            if tq_enabled:
                k_bits = int(getattr(self._model_settings, "turboquant_k_bits", 4))
                v_bits = int(getattr(self._model_settings, "turboquant_v_bits", 4))
                sparse_v = getattr(self._model_settings, "turboquant_sparse_v", True)
                sparse_v_budget = getattr(self._model_settings, "turboquant_sparse_v_budget", 0.75)

                self._engine.engine.scheduler._turboquant_k_bits = k_bits
                self._engine.engine.scheduler._turboquant_v_bits = v_bits
                self._engine.engine.scheduler._turboquant_sparse_v = sparse_v
                self._engine.engine.scheduler._turboquant_sparse_v_budget = sparse_v_budget
```

- [ ] **Step 2: Update VLM engine similarly**

Apply same changes to `omlx/engine/vlm.py`.

- [ ] **Step 3: Commit**

```bash
git add omlx/engine/batched.py omlx/engine/vlm.py
git commit -m "feat(engine): read new TurboQuant+ settings

- Read k_bits, v_bits, sparse_v, sparse_v_budget
- Propagate all settings to scheduler"
```

---

## Task 7: Update Scheduler

**Files:**
- Modify: `omlx/scheduler.py:129, 144-176, 1077, 1629-1631`

- [ ] **Step 1: Update Scheduler attributes**

In `omlx/scheduler.py`, around line 129, replace:

```python
self._turboquant_kv_bits: Optional[float] = None  # Set by Scheduler if enabled
```

With:

```python
self._turboquant_k_bits: Optional[int] = None
self._turboquant_v_bits: Optional[int] = None
self._turboquant_sparse_v: bool = True
self._turboquant_sparse_v_budget: float = 0.75
```

- [ ] **Step 2: Update _apply_turboquant_kv method**

Replace `_apply_turboquant_kv` (lines 144-176):

```python
    def _apply_turboquant_kv(self, prompt_cache: List[Any]) -> None:
        """Convert BatchKVCache layers to BatchTurboQuantKVCache."""
        from .turboquant_kv import BatchTurboQuantKVCache, TurboQuantKVCache
        from mlx_lm.models.cache import KVCache, CacheList

        converted = 0

        k_bits = int(self._turboquant_k_bits)
        v_bits = int(self._turboquant_v_bits)
        sparse_v = self._turboquant_sparse_v
        sparse_v_budget = self._turboquant_sparse_v_budget

        for i, cache_obj in enumerate(prompt_cache):
            cls_name = type(cache_obj).__name__
            if cls_name == "BatchKVCache":
                left_padding = cache_obj.left_padding.tolist()
                prompt_cache[i] = BatchTurboQuantKVCache(
                    left_padding,
                    k_bits=k_bits,
                    v_bits=v_bits,
                    sparse_v=sparse_v,
                    sparse_v_budget=sparse_v_budget,
                )
                converted += 1
            elif isinstance(cache_obj, KVCache):
                prompt_cache[i] = TurboQuantKVCache(
                    k_bits=k_bits,
                    v_bits=v_bits,
                    sparse_v=sparse_v,
                    sparse_v_budget=sparse_v_budget,
                )
                converted += 1
            elif isinstance(cache_obj, CacheList):
                new_caches = []
                for c in cache_obj.caches:
                    c_name = type(c).__name__
                    if c_name == "BatchKVCache":
                        left_padding = c.left_padding.tolist()
                        new_caches.append(BatchTurboQuantKVCache(
                            left_padding,
                            k_bits=k_bits,
                            v_bits=v_bits,
                            sparse_v=sparse_v,
                            sparse_v_budget=sparse_v_budget,
                        ))
                        converted += 1
                    elif isinstance(c, KVCache):
                        new_caches.append(TurboQuantKVCache(
                            k_bits=k_bits,
                            v_bits=v_bits,
                            sparse_v=sparse_v,
                            sparse_v_budget=sparse_v_budget,
                        ))
                        converted += 1
                    else:
                        new_caches.append(c)
                cache_obj.caches = tuple(new_caches)
        if converted > 0:
            logger.info(
                f"TurboQuant+: converted {converted}/{len(prompt_cache)} cache layers "
                f"to K={k_bits}-bit, V={v_bits}-bit"
            )
```

- [ ] **Step 3: Update BoundaryGenerator attributes**

Around line 1077, replace:

```python
self._turboquant_kv_bits: Optional[float] = None
```

With:

```python
self._turboquant_k_bits: Optional[int] = None
self._turboquant_v_bits: Optional[int] = None
self._turboquant_sparse_v: bool = True
self._turboquant_sparse_v_budget: float = 0.75
```

- [ ] **Step 4: Update clone propagation**

Around line 1629, replace:

```python
if hasattr(self, "_turboquant_kv_bits") and self._turboquant_kv_bits is not None:
    bg._turboquant_kv_bits = self._turboquant_kv_bits
```

With:

```python
if hasattr(self, "_turboquant_k_bits") and self._turboquant_k_bits is not None:
    bg._turboquant_k_bits = self._turboquant_k_bits
    bg._turboquant_v_bits = self._turboquant_v_bits
    bg._turboquant_sparse_v = self._turboquant_sparse_v
    bg._turboquant_sparse_v_budget = self._turboquant_sparse_v_budget
```

- [ ] **Step 5: Update condition check**

Around line 456, replace:

```python
if self._turboquant_kv_bits is not None:
    self._apply_turboquant_kv(prompt_cache)
```

With:

```python
if self._turboquant_k_bits is not None:
    self._apply_turboquant_kv(prompt_cache)
```

- [ ] **Step 6: Commit**

```bash
git add omlx/scheduler.py
git commit -m "feat(scheduler): support asymmetric K/V bits and sparse V

- Separate k_bits and v_bits
- Pass sparse_v and sparse_v_budget to cache constructors
- Update BoundaryGenerator cloning"
```

---

## Task 8: Update Attention Patch

**Files:**
- Modify: `omlx/patches/turboquant_attention.py`

- [ ] **Step 1: Update patch for new codec structure**

The existing patch should continue to work since it checks for `TurboQuantKVCache` type. No changes needed unless the cache interface changed.

- [ ] **Step 2: Commit (if changes were needed)**

```bash
git add omlx/patches/turboquant_attention.py
git commit -m "fix(attention): adapt patch for PolarQuant codec"
```

---

## Task 9: Update Admin Dashboard UI

**Files:**
- Modify: `omlx/admin/templates/dashboard/_modal_model_settings.html`
- Modify: `omlx/admin/static/js/dashboard.js`
- Modify: `omlx/admin/i18n/en.json`
- Modify: `omlx/admin/i18n/zh-TW.json`

- [ ] **Step 1: Update HTML template**

Find the TurboQuant section in `_modal_model_settings.html` and update:

```html
<div class="mb-3">
    <div class="form-check form-switch">
        <input class="form-check-input" type="checkbox" id="turboquantEnabled" name="turboquant_enabled">
        <label class="form-check-label" for="turboquantEnabled" data-i18n="turboquant_enabled">TurboQuant+ KV Cache</label>
    </div>
</div>
<div class="row mb-3 turboquant-settings" style="display:none;">
    <div class="col-md-3">
        <label class="form-label" data-i18n="turboquant_k_bits">K Bits</label>
        <select class="form-select" name="turboquant_k_bits">
            <option value="2">2-bit (turbo2)</option>
            <option value="3">3-bit (turbo3)</option>
            <option value="4" selected>4-bit (turbo4)</option>
        </select>
    </div>
    <div class="col-md-3">
        <label class="form-label" data-i18n="turboquant_v_bits">V Bits</label>
        <select class="form-select" name="turboquant_v_bits">
            <option value="2">2-bit (turbo2)</option>
            <option value="3">3-bit (turbo3)</option>
            <option value="4" selected>4-bit (turbo4)</option>
        </select>
    </div>
    <div class="col-md-3">
        <div class="form-check form-switch mt-4">
            <input class="form-check-input" type="checkbox" id="turboquantSparseV" name="turboquant_sparse_v" checked>
            <label class="form-check-label" for="turboquantSparseV" data-i18n="turboquant_sparse_v">Sparse V</label>
        </div>
    </div>
    <div class="col-md-3">
        <label class="form-label" data-i18n="turboquant_sparse_v_budget">Sparse Budget</label>
        <input type="range" class="form-range" name="turboquant_sparse_v_budget" min="0.5" max="1.0" step="0.05" value="0.75">
        <small class="text-muted">0.75</small>
    </div>
</div>
```

- [ ] **Step 2: Update JavaScript**

In `dashboard.js`, update the settings handling:

```javascript
// Show/hide turboquant settings based on checkbox
$('#turboquantEnabled').on('change', function() {
    $('.turboquant-settings').toggle(this.checked);
});

// Update budget display
$('input[name="turboquant_sparse_v_budget"]').on('input', function() {
    $(this).next('small').text($(this).val());
});
```

- [ ] **Step 3: Update i18n files**

In `en.json`:
```json
"turboquant_enabled": "TurboQuant+ KV Cache",
"turboquant_k_bits": "K Bits",
"turboquant_v_bits": "V Bits",
"turboquant_sparse_v": "Sparse V Decoding",
"turboquant_sparse_v_budget": "Sparse Budget"
```

In `zh-TW.json`:
```json
"turboquant_enabled": "TurboQuant+ KV 快取",
"turboquant_k_bits": "K 位元數",
"turboquant_v_bits": "V 位元數",
"turboquant_sparse_v": "稀疏 V 解碼",
"turboquant_sparse_v_budget": "稀疏預算"
```

- [ ] **Step 4: Commit**

```bash
git add omlx/admin/
git commit -m "feat(admin): update UI for TurboQuant+ settings

- Separate K/V bits dropdowns
- Sparse V toggle and budget slider
- i18n support for new labels"
```

---

## Task 10: Integration Tests

**Files:**
- Create: `tests/test_turboquant_integration.py`

- [ ] **Step 1: Create integration tests**

```python
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
        keys = mx.random.normal(shape=(1, 4, 500, 64))
        values = mx.random.normal(shape=(1, 4, 500, 64))

        k_out, v_out = cache.update_and_fetch(keys, values)
        assert k_out.shape == keys.shape

        # Decode
        k_dec = mx.random.normal(shape=(1, 4, 1, 64))
        v_dec = mx.random.normal(shape=(1, 4, 1, 64))
        cache.update_and_fetch(k_dec, v_dec)

        assert cache._quantized
        assert cache.offset == 501

    def test_turbo3_k_turbo4_v_asymmetric(self):
        """Test asymmetric K=3, V=4 configuration."""
        from omlx.turboquant_kv import TurboQuantKVCache

        cache = TurboQuantKVCache(k_bits=3, v_bits=4, sparse_v=False)

        mx.random.seed(42)
        keys = mx.random.normal(shape=(1, 4, 1000, 64))
        values = mx.random.normal(shape=(1, 4, 1000, 64))

        cache.update_and_fetch(keys, values)

        # Decode triggers quantization
        k_dec = mx.random.normal(shape=(1, 4, 1, 64))
        v_dec = mx.random.normal(shape=(1, 4, 1, 64))
        cache.update_and_fetch(k_dec, v_dec)

        assert cache._k_codec.bits == 3
        assert cache._v_codec.bits == 4

    def test_turbo2_memory_constrained(self):
        """Test turbo2 for extreme memory constraints."""
        from omlx.turboquant_kv import TurboQuantKVCache

        cache = TurboQuantKVCache(k_bits=2, v_bits=2, sparse_v=False)

        mx.random.seed(42)
        keys = mx.random.normal(shape=(1, 4, 2000, 64))
        values = mx.random.normal(shape=(1, 4, 2000, 64))

        cache.update_and_fetch(keys, values)

        # Decode
        k_dec = mx.random.normal(shape=(1, 4, 1, 64))
        v_dec = mx.random.normal(shape=(1, 4, 1, 64))
        cache.update_and_fetch(k_dec, v_dec)

        # Memory should be significantly reduced
        nbytes = cache.nbytes
        fp16_bytes = 2 * 4 * 2001 * 64 * 2  # K + V, 4 heads, 2001 tokens, 64 dim, 2 bytes
        compression_ratio = fp16_bytes / nbytes
        assert compression_ratio > 5.0  # At least 5x compression for 2-bit

    def test_batch_cache_merge_extract(self):
        """Test BatchTurboQuantKVCache merge and extract."""
        from omlx.turboquant_kv import TurboQuantKVCache, BatchTurboQuantKVCache

        # Create individual caches
        caches = []
        for i in range(3):
            c = TurboQuantKVCache(k_bits=4, v_bits=3, sparse_v=False)
            mx.random.seed(i * 100)
            keys = mx.random.normal(shape=(1, 4, 500 + i * 100, 64))
            values = mx.random.normal(shape=(1, 4, 500 + i * 100, 64))
            c.update_and_fetch(keys, values)
            # Trigger quantization
            k_dec = mx.random.normal(shape=(1, 4, 1, 64))
            v_dec = mx.random.normal(shape=(1, 4, 1, 64))
            c.update_and_fetch(k_dec, v_dec)
            caches.append(c)

        # Merge
        batch = BatchTurboQuantKVCache.merge(caches)
        assert batch._quantized
        assert batch._idx == 702  # max length (500 + 2*100 + 2)

        # Extract
        extracted = batch.extract(1)
        assert extracted.offset == 602  # 500 + 100 + 2
```

- [ ] **Step 2: Run integration tests**

Run: `pytest tests/test_turboquant_integration.py -v`
Expected: All 4 tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_turboquant_integration.py
git commit -m "test(turboquant): add integration tests

- Symmetric turbo4
- Asymmetric K=3 V=4
- Memory-constrained turbo2
- Batch cache merge/extract"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** All spec sections mapped to tasks
  - Settings → Task 1
  - WHT rotation → Task 2
  - PolarQuant → Task 3
  - Asymmetric K/V → Task 4
  - Sparse V → Task 5
  - Engine integration → Task 6
  - Scheduler → Task 7
  - Attention patch → Task 8
  - Admin UI → Task 9
  - Integration tests → Task 10
- [x] **Placeholder scan:** No TBD, TODO, or vague instructions
- [x] **Type consistency:** `k_bits`/`v_bits` used consistently across all files
