# TurboQuant+ Integration Design

**Date:** 2026-03-30
**Status:** Approved
**Breaking Changes:** Yes (replaces experimental `turboquant_kv_*` settings)

## Overview

Integrate key algorithms from [turboquant_plus](https://github.com/TheTom/turboquant_plus) into omlx's KV cache compression system to improve inference performance, especially for long-context scenarios.

### Key Improvements

| Feature | Current (omlx) | After Integration |
|---------|---------------|-------------------|
| Compression bits | 3-4 bit | 2-3-4 bit (turbo2/3/4) |
| Rotation algorithm | QR decomposition O(n²) | Walsh-Hadamard O(n log n) |
| K/V compression | Symmetric (same bits) | Asymmetric (K/V different bits) |
| Sparse V Decoding | None | Yes (22.8% decode speedup at 32K context) |
| Quantization | Beta-Lloyd-Max codebook | PolarQuant + WHT |

## 1. Configuration Layer (`model_settings.py`)

Replace existing `turboquant_kv_enabled` / `turboquant_kv_bits` with:

```python
# TurboQuant+ KV cache compression
turboquant_enabled: bool = False
turboquant_k_bits: int = 4        # 2, 3, or 4
turboquant_v_bits: int = 4        # 2, 3, or 4 (asymmetric K/V)
turboquant_sparse_v: bool = True  # Sparse V Decoding (skip low-weight V)
turboquant_sparse_v_budget: float = 0.75  # Top-75% attention mass retained
```

### Defaults

- `sparse_v` defaults to `True` (highest impact, no quality loss)
- `k_bits` and `v_bits` default to `4` (balanced quality/memory)
- `sparse_v_budget` defaults to `0.75` (top 75% attention mass)

### Legacy Migration

Old settings (`turboquant_kv_enabled`, `turboquant_kv_bits`) are removed. Users must update their `model_settings.json`.

## 2. Core Algorithm Layer (`turboquant_kv.py`)

### 2.1 Walsh-Hadamard Rotation

Replace `_rotation_matrix()` (QR decomposition) with Walsh-Hadamard Transform:

```python
def _walsh_hadamard_matrix(dim: int, seed: int = 0) -> tuple[mx.array, mx.array]:
    """Generate WHT rotation: deterministic Hadamard + random sign flip.

    Returns:
        (signs, scale) where rotated = signs * WHT(v) * scale
    """
    # Random ±1 signs
    key = mx.random.key(seed)
    signs = mx.random.bernoulli(key, shape=(dim,)).astype(mx.float32) * 2 - 1

    # WHT scale: 1/sqrt(dim) for orthonormality
    scale = 1.0 / math.sqrt(dim)
    return signs, mx.array(scale)
```

**Benefits:**
- O(n log n) vs O(n²) for QR
- No need to store full matrix (just signs + scale)
- Deterministic Hadamard structure enables fast kernel

### 2.2 PolarQuant

Replace Beta-Lloyd-Max codebook with per-group min/max quantization:

```python
class PolarQuantCodec:
    """Polar quantization: per-group min/max with uniform bins."""

    def __init__(self, dim: int, bits: int, group_size: int = 64):
        self.dim = dim
        self.bits = bits
        self.group_size = group_size if bits > 2 else 32  # smaller groups for 2-bit
        self.n_levels = 1 << bits

    def quantize(self, vectors: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """Quantize: (B, H, T, D) -> (scales, zeros, packed_indices)."""
        # 1. Apply WHT rotation
        rotated = self._apply_wht(vectors)

        # 2. Per-group min/max
        grouped = rotated.reshape(*shape, -1, self.group_size)
        g_min = grouped.min(axis=-1, keepdims=True)
        g_max = grouped.max(axis=-1, keepdims=True)

        # 3. Uniform quantization
        scale = (g_max - g_min) / (self.n_levels - 1)
        indices = ((grouped - g_min) / scale).astype(mx.uint32)
        indices = mx.clip(indices, 0, self.n_levels - 1)

        # 4. Pack
        packed = _pack_contiguous(indices, self.bits, self.group_size)
        return scale.squeeze(-1), g_min.squeeze(-1), packed

    def dequantize(self, scales, zeros, packed) -> mx.array:
        """Dequantize packed indices back to vectors."""
        indices = _unpack_contiguous(packed, self.bits, self.group_size)
        values = zeros[..., None] + indices.astype(mx.float32) * scales[..., None]
        return self._apply_wht_inverse(values.reshape(*shape))
```

### 2.3 Asymmetric K/V Codec

Split single codec into separate K and V codecs:

```python
class TurboQuantKVCache:
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
        self._k_codec: Optional[PolarQuantCodec] = None
        self._v_codec: Optional[PolarQuantCodec] = None
```

The existing fused SDPA kernel already has `KBits`/`VBits` template parameters - just pass different values.

## 3. Sparse V Decoding (New Metal Kernel)

### 3.1 Algorithm

**Current flow:** Q×K → softmax → weighted sum ALL V → output

**Sparse V flow:** Q×K → softmax → **select top-k tokens covering budget%** → weighted sum SELECTED V → output

### 3.2 Implementation

Add Pass 1.5 between existing Pass 1 and Pass 2:

```metal
// Pass 1.5: Compute sparse V mask
// Input: per-block softmax sums from Pass 1
// Output: binary mask indicating which tokens to include

kernel void sparse_v_mask(
    const device float* sums [[buffer(0)]],      // (B*H_q, num_blocks)
    const device float* maxs [[buffer(1)]],     // (B*H_q, num_blocks)
    device uint* mask [[buffer(2)]],            // (B*H_q, T) bitmap
    uint tid [[thread_position_in_grid]]
) {
    // 1. Compute global softmax (reduce across blocks)
    // 2. Sort tokens by attention score (descending)
    // 3. Cumulative sum until budget reached
    // 4. Mark tokens in budget as 1, others as 0
}
```

### 3.3 Modified Pass 2

```metal
// Pass 2: Reduce with sparse mask
// Only accumulate V entries where mask[t] == 1

for (int t = block_idx; t < total_tokens; t += Blocks) {
    if (sparse_mask[t] == 0) continue;  // Skip low-weight V
    // ... existing accumulation logic
}
```

### 3.4 Fallback

- When `T < 1024`: disable sparse (no benefit for short sequences)
- When `sparse_v_budget >= 1.0`: disable sparse (all tokens included)

## 4. turbo2 (2-bit) Support

### 4.1 Changes

- Metal kernel `Bits` template already supports arbitrary values
- Reduce `group_size` from 64 to 32 for 2-bit (smaller groups = better quality)
- WHT rotation more critical at 2-bit (Gaussianization improves low-bit quantization)

### 4.2 Quality Warning

Log warning when turbo2 is enabled:

```python
if k_bits == 2 or v_bits == 2:
    logger.warning(
        "TurboQuant 2-bit mode enabled. Expect ~6.5% perplexity increase. "
        "Recommended only for extreme memory constraints."
    )
```

## 5. Integration Points

### Files to Modify

| File | Changes |
|------|---------|
| `model_settings.py` | New settings fields, remove `turboquant_kv_*` |
| `turboquant_kv.py` | WHT rotation, PolarQuant, asymmetric codec, sparse V kernel |
| `patches/turboquant_attention.py` | Adapt to `k_codec`/`v_codec` structure |
| `engine/batched.py` | Read new settings, pass to cache constructor |
| `engine/vlm.py` | Same as above |
| `server.py` | Expose new settings in API responses |
| `scheduler.py` | Pass asymmetric bits during cache reconstruction |
| `admin/templates/dashboard/_modal_model_settings.html` | UI for new settings |
| `admin/static/js/dashboard.js` | JS to handle new settings |
| `admin/i18n/*.json` | Translations for new setting labels |

### Files NOT Modified

- `oq.py` (weight quantization - separate concern)
- `engine_core.py` (inference loop - no changes needed)
- `engine_pool.py` (model management - no changes needed)
- Cache tier logic (SSD cache, prefix cache)

## 6. Testing Strategy

### Unit Tests

1. **WHT Orthogonality**: Verify `WHT(WHT(v)) = n * v` and orthogonality
2. **PolarQuant Round-trip**: Encode/decode MSE < threshold for each bit level
3. **Sparse V Mask**: Verify budget coverage matches expected percentage
4. **Asymmetric K/V**: K at 4-bit, V at 2-bit produces correct shapes

### Integration Tests

1. **turbo2/3/4 Decode Attention**: Compare output MSE vs fp16 SDPA
2. **Long Context (32K)**: Verify sparse V speedup > 15%
3. **Continuous Batching**: Verify `BatchTurboQuantKVCache` merge/extract/filter with asymmetric bits

### Benchmark Targets

| Configuration | Memory Reduction | Decode Speed | Perplexity Δ |
|---------------|------------------|--------------|--------------|
| turbo4 (baseline) | ~70% | 1.0x | +0.2% |
| turbo3 | ~78% | 1.0x | +1.0% |
| turbo2 | ~84% | 1.0x | +6.5% |
| turbo4 + sparse_v | ~70% | 1.15x @ 8K, 1.23x @ 32K | +0.2% |
| turbo3 K + turbo4 V | ~74% | 1.0x | +0.5% |

## 7. Implementation Phases

### Phase 1: Core Algorithms (Priority: High)
- Implement WHT rotation
- Implement PolarQuant codec
- Update `TurboQuantMSECodec` → `PolarQuantCodec`
- Add asymmetric K/V support

### Phase 2: Sparse V Decoding (Priority: High)
- Add Pass 1.5 sparse mask kernel
- Modify Pass 2 to respect mask
- Add fallback logic for short sequences

### Phase 3: Configuration & Integration (Priority: Medium)
- Update `model_settings.py`
- Update engine files (`batched.py`, `vlm.py`)
- Update `scheduler.py` cache reconstruction

### Phase 4: UI & API (Priority: Low)
- Update admin dashboard modal
- Update API responses
- Add i18n translations

### Phase 5: Testing & Benchmarking (Priority: High)
- Unit tests
- Integration tests
- Performance benchmarks
