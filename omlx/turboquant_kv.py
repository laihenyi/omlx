# SPDX-License-Identifier: Apache-2.0
"""TurboQuant KV cache: codebook-quantized KV with fused Flash Attention.

Key design:
  - MSE codec: rotation → codebook quantization (per-coordinate optimal)
  - 2-pass fused SDPA kernel: score + softmax + weighted_sum in GPU
  - 8K+ context: faster than fp16 SDPA (bandwidth reduction)
  - Memory: ~70% reduction vs fp16 KV cache
"""

from __future__ import annotations

import logging
import math
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import _BaseCache


# ---------------------------------------------------------------------------
# Walsh-Hadamard Transform (replaces QR-based rotation)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=16)
def _wht_matrix(dim: int) -> mx.array:
    """Generate Walsh-Hadamard matrix of size dim x dim.

    Uses Sylvester construction: H_{2n} = [[H_n, H_n], [H_n, -H_n]]
    Only valid for dim = 2^k.
    """
    if dim == 1:
        return mx.array([[1.0]], dtype=mx.float32)

    # Verify dim is power of 2
    assert (dim & (dim - 1)) == 0, f"dim must be power of 2, got {dim}"

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
    signs = mx.random.bernoulli(p=0.5, shape=(dim,), key=key).astype(mx.float32) * 2 - 1
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

    # Apply: signs * (vectors @ H) * scale
    rotated = (vectors.astype(mx.float32) @ H) * signs * scale
    return rotated.astype(vectors.dtype)


def apply_wht_inverse(vectors: mx.array, seed: int = 0) -> mx.array:
    """Apply inverse WHT rotation: WHT(signs * vectors) / sqrt(dim).

    WHT is self-inverse up to scale: WHT(WHT(v)) = dim * v
    Since forward uses 1/sqrt(dim), inverse also uses 1/sqrt(dim).
    """
    dim = vectors.shape[-1]
    H = _wht_matrix(dim)
    signs = _random_sign_flip(dim, seed)
    scale = 1.0 / math.sqrt(dim)

    # Apply: (vectors * signs) @ H * scale
    restored = (vectors.astype(mx.float32) * signs) @ H * scale
    return restored.astype(vectors.dtype)


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

    def quantize(self, vectors: mx.array) -> tuple:
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
        packed_groups = []
        for g in range(n_groups):
            group_indices = indices[..., g, :]  # (B, H, T, group_size)
            packed_group = _pack_contiguous(group_indices, self.bits, self.group_size)
            packed_groups.append(packed_group[..., None, :])  # Add n_groups dim

        # Concatenate along n_groups axis: (B, H, T, n_groups, pw)
        packed = mx.concatenate(packed_groups, axis=-2)

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
        unpacked_groups = []
        for g in range(n_groups):
            group_packed = packed[..., g, :]  # (B, H, T, pw)
            unpacked = _unpack_contiguous(group_packed, self.bits, self.group_size)
            unpacked_groups.append(unpacked[..., None, :])  # Add n_groups dim

        # Concatenate along n_groups axis: (B, H, T, n_groups, group_size)
        indices = mx.concatenate(unpacked_groups, axis=-2)

        # Scale back
        values = zeros[..., None] + indices.astype(scales.dtype) * scales[..., None]
        # values shape: (B, H, T, n_groups, group_size)

        # Flatten groups
        restored = values.reshape(B, H, T, -1)

        # Apply inverse WHT
        restored = apply_wht_inverse(restored, self.seed)

        return restored


# ---------------------------------------------------------------------------
# Codebook generation (Beta distribution Lloyd-Max quantizer)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=32)
def _codebook(dim: int, bits: int) -> mx.array:
    """Optimal scalar codebook for Beta(dim/2, dim/2) via Lloyd's algorithm."""
    n_levels = 1 << bits
    alpha = dim / 2.0
    rng = np.random.default_rng(seed=0)
    samples = 2.0 * rng.beta(alpha, alpha, size=100_000) - 1.0
    centroids = np.linspace(samples.min(), samples.max(), n_levels)
    for _ in range(100):
        dists = np.abs(samples[:, None] - centroids[None, :])
        assignments = np.argmin(dists, axis=1)
        for j in range(n_levels):
            mask = assignments == j
            if mask.sum() > 0:
                centroids[j] = samples[mask].mean()
    return mx.array(sorted(centroids), dtype=mx.float32)


@lru_cache(maxsize=16)
def _rotation_matrix(dim: int, seed: int) -> mx.array:
    """Random orthogonal rotation via QR decomposition."""
    key = mx.random.key(seed)
    Q, R = mx.linalg.qr(mx.random.normal(shape=(dim, dim), key=key), stream=mx.cpu)
    signs = mx.sign(mx.diag(R))
    signs = mx.where(signs == 0, mx.ones_like(signs), signs)
    Q = (Q * signs[None, :]).astype(mx.float32)
    mx.eval(Q)
    return Q


# ---------------------------------------------------------------------------
# Contiguous bit packing (compatible with mlx-vlm / Flash Attention kernel)
# ---------------------------------------------------------------------------

def _packed_width(dim: int, bits: int) -> int:
    return (dim * bits + 31) // 32


@lru_cache(maxsize=None)
def _pack_lowbit_kernel():
    """Metal kernel for contiguous bit packing."""
    source = r"""
        auto word = thread_position_in_grid.x;
        auto row = thread_position_in_grid.y;

        if (row >= values_shape[0] || word >= PackedWidth) return;

        auto values_ptr = values + row * Length;
        uint packed_word = 0u;
        int start = max(0, (int(word) * 32 - (Bits - 1)) / Bits);
        int end = min(Length, ((int(word) + 1) * 32 + (Bits - 1)) / Bits);

        for (int idx = start; idx < end; ++idx) {
            int bit_offset = idx * Bits;
            int word_idx = bit_offset / 32;
            int offset = bit_offset % 32;
            uint value = values_ptr[idx] & ((1u << Bits) - 1u);
            if (word_idx == word) packed_word |= value << offset;
            if (word_idx + 1 == word) {
                int spill = offset + Bits - 32;
                if (spill > 0) packed_word |= value >> (Bits - spill);
            }
        }
        out[row * PackedWidth + word] = packed_word;
    """
    return mx.fast.metal_kernel(
        name="tq_pack_lowbit",
        input_names=["values"],
        output_names=["out"],
        source=source,
    )


def _pack_contiguous(indices: mx.array, bits: int, dim: int) -> mx.array:
    """Pack indices using contiguous bit packing via Metal kernel."""
    batch_shape = indices.shape[:-1]
    flat = indices.reshape(-1, dim).astype(mx.uint32)
    pw = _packed_width(dim, bits)
    rows = flat.shape[0]

    kernel = _pack_lowbit_kernel()
    packed = kernel(
        inputs=[flat],
        output_shapes=[(rows, pw)],
        output_dtypes=[mx.uint32],
        grid=(pw, rows, 1),
        threadgroup=(min(pw, 32), 1, 1),
        template=[("Bits", bits), ("Length", dim), ("PackedWidth", pw)],
        init_value=0,
    )[0]
    return packed.reshape(*batch_shape, pw)


@lru_cache(maxsize=None)
def _fused_quantize_kernel():
    """Fused: norm + normalize + rotate + boundary + pack in ONE kernel."""
    source = r"""
        // Grid: (PackedWidth, rows, 1)  Threadgroup: (min(PackedWidth, 32), 1, 1)
        // Each thread packs one uint32 word of output

        auto word = thread_position_in_grid.x;
        auto row = thread_position_in_grid.y;

        if (row >= vectors_shape[0] || word >= PackedWidth) return;

        auto vec = vectors + row * Dim;

        // Step 1: Compute norm (cooperative — but for 1 token, single thread is fine)
        float norm_sq = 0.0f;
        for (int d = 0; d < Dim; d++) {
            float v = static_cast<float>(vec[d]);
            norm_sq += v * v;
        }
        float norm = sqrt(norm_sq);
        float inv_norm = norm > 1e-10f ? 1.0f / norm : 1.0f;

        // Store norm (only first thread per row)
        if (word == 0) {
            out_norms[row] = norm;
        }

        // Step 2+3+4: For each value this word packs: rotate + boundary + pack
        uint packed_word = 0u;
        int start = max(0, (int(word) * 32 - (Bits - 1)) / Bits);
        int end = min(Dim, ((int(word) + 1) * 32 + (Bits - 1)) / Bits);

        for (int idx = start; idx < end; ++idx) {
            // Rotate: rotated[idx] = sum_j(normalized[j] * rotation[j * Dim + idx])
            float rotated_val = 0.0f;
            for (int j = 0; j < Dim; j++) {
                rotated_val += static_cast<float>(vec[j]) * inv_norm * rotation[j * Dim + idx];
            }

            // Boundary quantize: count how many boundaries this value exceeds
            uint quant_idx = 0u;
            for (int b = 0; b < NLevels - 1; b++) {
                if (rotated_val > boundaries[b]) quant_idx++;
            }

            // Pack into word
            int bit_offset = idx * Bits;
            int word_idx = bit_offset / 32;
            int offset = bit_offset % 32;
            if (word_idx == word) {
                packed_word |= (quant_idx & ((1u << Bits) - 1u)) << offset;
            }
            if (word_idx + 1 == word) {
                int spill = offset + Bits - 32;
                if (spill > 0) {
                    packed_word |= (quant_idx & ((1u << Bits) - 1u)) >> (Bits - spill);
                }
            }
        }

        out_packed[row * PackedWidth + word] = packed_word;
    """
    return mx.fast.metal_kernel(
        name="tq_fused_quantize",
        input_names=["vectors", "rotation", "boundaries"],
        output_names=["out_packed", "out_norms"],
        source=source,
    )


def _fused_quantize(vectors: mx.array, rotation: mx.array, boundaries: mx.array,
                     bits: int, dim: int) -> tuple:
    """Fused quantize: norm + rotate + boundary + pack in one Metal dispatch."""
    batch_shape = vectors.shape[:-1]
    flat = vectors.reshape(-1, dim)
    rows = flat.shape[0]
    pw = _packed_width(dim, bits)
    n_levels = 1 << bits

    kernel = _fused_quantize_kernel()
    packed, norms = kernel(
        inputs=[flat.astype(mx.float16), rotation.astype(mx.float32), boundaries.astype(mx.float32)],
        output_shapes=[(rows, pw), (rows,)],
        output_dtypes=[mx.uint32, mx.float32],
        grid=(pw, rows, 1),
        threadgroup=(min(pw, 32), 1, 1),
        template=[("Bits", bits), ("Dim", dim), ("PackedWidth", pw), ("NLevels", n_levels)],
        init_value=0,
    )
    return norms.reshape(*batch_shape), packed.reshape(*batch_shape, pw)


@lru_cache(maxsize=None)
def _unpack_lowbit_kernel():
    source = r"""
        auto idx = thread_position_in_grid.x;
        auto row = thread_position_in_grid.y;
        if (row >= packed_shape[0] || idx >= Length) return;

        auto packed_ptr = packed + row * PackedWidth;
        int bit_offset = idx * Bits;
        int word_idx = bit_offset / 32;
        int offset = bit_offset % 32;
        uint value = packed_ptr[word_idx] >> offset;
        int spill = offset + Bits - 32;
        if (spill > 0) value |= packed_ptr[word_idx + 1] << (Bits - spill);
        out[row * Length + idx] = value & ((1u << Bits) - 1u);
    """
    return mx.fast.metal_kernel(
        name="tq_unpack_lowbit",
        input_names=["packed"],
        output_names=["out"],
        source=source,
    )


def _unpack_contiguous(packed: mx.array, bits: int, dim: int) -> mx.array:
    """Unpack contiguous bit-packed indices via Metal kernel."""
    batch_shape = packed.shape[:-1]
    flat = packed.reshape(-1, packed.shape[-1])
    rows = flat.shape[0]
    pw = flat.shape[-1]

    kernel = _unpack_lowbit_kernel()
    indices = kernel(
        inputs=[flat.astype(mx.uint32)],
        output_shapes=[(rows, dim)],
        output_dtypes=[mx.uint32],
        grid=(dim, rows, 1),
        threadgroup=(min(dim, 32), 1, 1),
        template=[("Bits", bits), ("Length", dim), ("PackedWidth", pw)],
        init_value=0,
    )[0]
    return indices.reshape(*batch_shape, dim)


# ---------------------------------------------------------------------------
# MSE Codec: rotation → nearest codebook → pack
# ---------------------------------------------------------------------------

class TurboQuantMSECodec:
    """MSE-optimal vector quantization codec."""

    def __init__(self, dim: int, bits: int, seed: int = 0):
        self.dim = dim
        self.bits = bits
        self.seed = seed
        self.codebook = _codebook(dim, bits)
        self.rotation = _rotation_matrix(dim, seed)
        self._pw = _packed_width(dim, bits)
        # Pre-compute decision boundaries for fast quantization
        cb = self.codebook
        self._boundaries = (cb[:-1] + cb[1:]) / 2  # midpoints between sorted centroids

    def quantize(self, vectors: mx.array):
        """Quantize vectors: (B, H, T, D) → (norms, packed_indices)."""
        norms = mx.linalg.norm(vectors, axis=-1, keepdims=True)
        safe_norms = mx.maximum(norms, 1e-10)
        normalized = vectors / safe_norms

        # Rotate
        shape = normalized.shape
        grouped = normalized.reshape(*shape[:-1], shape[-1] // self.dim, self.dim)
        rotated = (grouped.astype(mx.float32) @ self.rotation).reshape(shape)

        # Boundary-based quantization (19x faster than argmin)
        indices = (rotated[..., None] > self._boundaries).sum(axis=-1).astype(mx.uint32)

        # Pack
        packed = _pack_contiguous(indices, self.bits, self.dim)
        return norms.squeeze(-1), packed

    def dequantize(self, norms: mx.array, packed: mx.array) -> mx.array:
        """Dequantize: (norms, packed) → vectors."""
        indices = _unpack_contiguous(packed, self.bits, self.dim)
        coords = self.codebook[indices]

        shape = coords.shape
        grouped = coords.reshape(*shape[:-1], shape[-1] // self.dim, self.dim)
        restored = (grouped @ self.rotation.T).reshape(shape)

        return restored * norms[..., None]


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

    # For simplicity, we select top-k tokens that cover the budget
    # Sort scores descending
    sorted_indices = mx.argsort(attention_scores, axis=-1)[:, :, ::-1]
    sorted_scores = mx.take_along_axis(attention_scores, sorted_indices, axis=-1)

    # Cumulative sum
    cumsum = mx.cumsum(sorted_scores, axis=-1)

    # Find cutoff: first index where cumsum >= budget
    cutoff = (cumsum >= budget).astype(mx.int32).argmax(axis=-1)

    # Create mask in sorted order using broadcasting
    # token_indices: (1, 1, T) broadcast to (B, H, T)
    token_indices = mx.arange(T)[None, None, :]
    # cutoff: (B, H, 1) broadcast to (B, H, T)
    cutoff_expanded = cutoff[:, :, None]
    # mask_sorted is 1 where token_idx <= cutoff
    mask_sorted = (token_indices <= cutoff_expanded).astype(attention_scores.dtype)

    # Unsort to original order
    mask = mx.zeros_like(attention_scores)
    mask = mx.put_along_axis(mask, sorted_indices, mask_sorted, axis=-1)

    return mask.astype(mx.bool_)


# ---------------------------------------------------------------------------
# 2-pass Fused Flash Attention kernels
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _tq_sdpa_2pass_1_kernel():
    source = r"""
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

        for (int t = block_idx; t < total_tokens; t += Blocks) {
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
        name="tq_fused_sdpa_pass1",
        input_names=["queries", "k_packed", "k_norms", "k_codebook",
                     "v_packed", "v_norms", "v_codebook", "scale"],
        output_names=["partial_out", "sums", "maxs"],
        source=source,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=None)
def _tq_sdpa_2pass_2_kernel():
    source = r"""
        auto simd_gid = simdgroup_index_in_threadgroup;
        auto simd_lid = thread_index_in_simdgroup;
        auto head_idx = threadgroup_position_in_grid.x;

        auto s_base = sums + head_idx * Blocks;
        auto m_base = maxs + head_idx * Blocks;
        auto o_ptr = out + head_idx * Dim + simd_gid * QK_PER_THREAD;

        threadgroup float tg_outputs[BN * BD];

        float max_score = -INFINITY;
        for (int b = 0; b < Blocks / BN; b++)
            max_score = max(max_score, m_base[simd_lid + BN * b]);
        if (Blocks % BN != 0) {
            int b_idx = (Blocks / BN) * BN + simd_lid;
            if (b_idx < Blocks) max_score = max(max_score, m_base[b_idx]);
        }
        max_score = simd_max(max_score);

        float sum_exp = 0.0f;
        for (int b = 0; b < Blocks / BN; b++) {
            float factor = exp(m_base[simd_lid + BN * b] - max_score);
            sum_exp += factor * s_base[simd_lid + BN * b];
        }
        if (Blocks % BN != 0) {
            int b_idx = (Blocks / BN) * BN + simd_lid;
            if (b_idx < Blocks) {
                sum_exp += exp(m_base[b_idx] - max_score) * s_base[b_idx];
            }
        }
        sum_exp = simd_sum(sum_exp);

        float o[QK_PER_THREAD] = {0};
        for (int b = 0; b < Blocks / BN; b++) {
            float factor = exp(m_base[simd_gid + BN * b] - max_score);
            auto p_ptr = partials + head_idx * Blocks * Dim
                         + (simd_gid + BN * b) * Dim + simd_lid * QK_PER_THREAD;
            for (int i = 0; i < QK_PER_THREAD; i++)
                o[i] += factor * static_cast<float>(p_ptr[i]);
        }

        for (int i = 0; i < QK_PER_THREAD; i++) {
            tg_outputs[simd_lid * BD + simd_gid] = o[i];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            o[i] = simd_sum(tg_outputs[simd_gid * BD + simd_lid]);
            o[i] = sum_exp > 0 ? o[i] / sum_exp : 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (simd_lid == 0) {
            for (int i = 0; i < QK_PER_THREAD; i++)
                o_ptr[i] = static_cast<half>(o[i]);
        }
    """
    return mx.fast.metal_kernel(
        name="tq_fused_sdpa_pass2",
        input_names=["partials", "sums", "maxs"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


def _fused_tq_sdpa(
    queries: mx.array,       # (B*H_q, D)
    k_packed: mx.array,      # (B, H_kv, T, packed_width)
    k_norms: mx.array,       # (B, H_kv, T)
    k_codebook: mx.array,    # (n_levels,)
    v_packed: mx.array,      # (B, H_kv, T, packed_width)
    v_norms: mx.array,       # (B, H_kv, T)
    v_codebook: mx.array,    # (n_levels,)
    scale: float,
    B: int, H_q: int, H_kv: int, D: int, bits: int,
) -> mx.array:
    """Fused 2-pass TurboQuant Flash Attention (supports B>=1)."""
    GQA = H_q // H_kv
    pw = _packed_width(D, bits)
    qpt = D // 32
    BN = BD = 32
    T = k_norms.shape[2]
    total_heads = B * H_q

    # Adaptive blocks: cap at 1024, each handles ~32 tokens
    num_blocks = min(1024, ((max(32, T // 32) + 31) // 32) * 32)

    scale_arr = mx.array([scale], dtype=mx.float32)

    # Pass 1: parallel block-wise attention
    # grid.y = B * GQA → threadgroup_position_in_grid.y = batch_idx (0..B-1)
    partials, sums, maxs = _tq_sdpa_2pass_1_kernel()(
        inputs=[queries, k_packed, k_norms, k_codebook,
                v_packed, v_norms, v_codebook, scale_arr],
        output_shapes=[
            (total_heads * num_blocks, D),
            (total_heads, num_blocks),
            (total_heads, num_blocks),
        ],
        output_dtypes=[mx.float16, mx.float32, mx.float32],
        grid=(H_kv * 32, B * GQA, num_blocks),
        threadgroup=(32, GQA, 1),
        template=[
            ("Dim", D), ("Blocks", num_blocks), ("QK_PER_THREAD", qpt),
            ("KBits", bits), ("VBits", bits),
            ("KPackedWidth", pw), ("VPackedWidth", pw),
        ],
        init_value=0.0,
    )

    # Fix: empty blocks have maxs=0 from init_value, should be -inf
    maxs = mx.where(sums == 0, mx.full(maxs.shape, float("-inf"), dtype=maxs.dtype), maxs)

    # Pass 2: SIMD-parallel reduction
    out = _tq_sdpa_2pass_2_kernel()(
        inputs=[partials, sums, maxs],
        output_shapes=[(total_heads, D)],
        output_dtypes=[mx.float16],
        grid=(total_heads * 32, BN, 1),
        threadgroup=(32, BN, 1),
        template=[
            ("Dim", D), ("Blocks", num_blocks), ("QK_PER_THREAD", qpt),
            ("BN", BN), ("BD", BD),
        ],
        init_value=0.0,
    )
    return out[0]


# ---------------------------------------------------------------------------
# TurboQuantKVCache
# ---------------------------------------------------------------------------

class TurboQuantKVCache(_BaseCache):
    """KV cache with TurboQuant codebook quantization.

    Stores keys and values as packed codebook indices + norms.
    Decode attention uses fused 2-pass Flash Attention kernel.
    Prefill uses dequantize + standard mx.fast.scaled_dot_product_attention.

    Supports asymmetric K/V compression with separate bit widths for keys and values.
    """

    def __init__(
        self,
        bits: int = None,
        k_bits: int = 4,
        v_bits: int = 4,
        sparse_v: bool = True,
        sparse_v_budget: float = 0.75,
        seed: int = 0
    ):
        # Support legacy 'bits' parameter for backward compatibility
        if bits is not None:
            k_bits = bits
            v_bits = bits
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.bits = k_bits  # Legacy compatibility attribute
        self.sparse_v = sparse_v
        self.sparse_v_budget = sparse_v_budget
        self.seed = seed
        # Safety: mlx-lm's base.py SDPA checks hasattr(cache, "bits") and then
        # accesses cache.group_size for affine quantized caches.  Prevents
        # AttributeError if our attention patch doesn't intercept.
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

    def _quantize_fp16_buffer(self):
        """Convert accumulated fp16 KV to quantized format."""
        if self._fp16_keys is None or self._quantized:
            return
        B, H, T, D = self._fp16_keys.shape
        logger.info(f"TurboQuant: quantizing {T} tokens ({B}x{H} heads, dim={D}) to K={self.k_bits}-bit, V={self.v_bits}-bit")
        self._ensure_codecs(D)

        # Quantize K
        k_scales, k_zeros, k_packed = self._k_codec.quantize(self._fp16_keys)
        # Quantize V
        v_scales, v_zeros, v_packed = self._v_codec.quantize(self._fp16_values)

        k_pw = _packed_width(self._k_codec.group_size, self.k_bits)
        v_pw = _packed_width(self._v_codec.group_size, self.v_bits)
        n_groups = D // self._k_codec.group_size

        alloc = ((T + self._step - 1) // self._step) * self._step
        self._k_scales = mx.zeros((B, H, alloc, n_groups), dtype=mx.float16)
        self._k_zeros = mx.zeros((B, H, alloc, n_groups), dtype=mx.float16)
        self._k_packed = mx.zeros((B, H, alloc, n_groups, k_pw), dtype=mx.uint32)
        self._v_scales = mx.zeros((B, H, alloc, n_groups), dtype=mx.float16)
        self._v_zeros = mx.zeros((B, H, alloc, n_groups), dtype=mx.float16)
        self._v_packed = mx.zeros((B, H, alloc, n_groups, v_pw), dtype=mx.uint32)

        self._k_scales[:, :, :T] = k_scales
        self._k_zeros[:, :, :T] = k_zeros
        self._k_packed[:, :, :T] = k_packed
        self._v_scales[:, :, :T] = v_scales
        self._v_zeros[:, :, :T] = v_zeros
        self._v_packed[:, :, :T] = v_packed

        self._quantized = True
        self._fp16_keys = None
        self._fp16_values = None

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Store new K,V. Prefill: fp16. Decode: quantize."""
        B, H, T_new, D = keys.shape
        self._ensure_codecs(D)

        if T_new > 1:
            # Prefill: accumulate fp16 (no quantize overhead, full quality)
            new_end = self.offset + T_new
            if self._fp16_keys is None:
                alloc = ((new_end + self._step - 1) // self._step) * self._step
                self._fp16_keys = mx.zeros((B, H, alloc, D), dtype=keys.dtype)
                self._fp16_values = mx.zeros((B, H, alloc, D), dtype=values.dtype)
            elif new_end > self._fp16_keys.shape[2]:
                alloc = ((new_end + self._step - 1) // self._step) * self._step
                pad = alloc - self._fp16_keys.shape[2]
                self._fp16_keys = mx.concatenate([self._fp16_keys, mx.zeros((B, H, pad, D), dtype=keys.dtype)], axis=2)
                self._fp16_values = mx.concatenate([self._fp16_values, mx.zeros((B, H, pad, D), dtype=values.dtype)], axis=2)
            self._fp16_keys[:, :, self.offset:new_end] = keys
            self._fp16_values[:, :, self.offset:new_end] = values
            self.offset = new_end
            return self._fp16_keys[:, :, :self.offset], self._fp16_values[:, :, :self.offset]
        else:
            # Decode: quantize prefill buffer on first decode token
            if not self._quantized:
                self._quantize_fp16_buffer()

            # Quantize decode tokens with separate codecs
            k_scales, k_zeros, k_packed = self._k_codec.quantize(keys)
            v_scales, v_zeros, v_packed = self._v_codec.quantize(values)

            new_end = self.offset + 1
            k_pw = _packed_width(self._k_codec.group_size, self.k_bits)
            v_pw = _packed_width(self._v_codec.group_size, self.v_bits)
            n_groups = D // self._k_codec.group_size

            if self._k_scales is None:
                alloc = self._step
                self._k_scales = mx.zeros((B, H, alloc, n_groups), dtype=mx.float16)
                self._k_zeros = mx.zeros((B, H, alloc, n_groups), dtype=mx.float16)
                self._k_packed = mx.zeros((B, H, alloc, n_groups, k_pw), dtype=mx.uint32)
                self._v_scales = mx.zeros((B, H, alloc, n_groups), dtype=mx.float16)
                self._v_zeros = mx.zeros((B, H, alloc, n_groups), dtype=mx.float16)
                self._v_packed = mx.zeros((B, H, alloc, n_groups, v_pw), dtype=mx.uint32)
            elif new_end > self._k_scales.shape[2]:
                alloc = ((new_end + self._step - 1) // self._step) * self._step
                pad = alloc - self._k_scales.shape[2]
                self._k_scales = mx.concatenate([self._k_scales, mx.zeros((B, H, pad, n_groups), dtype=mx.float16)], axis=2)
                self._k_zeros = mx.concatenate([self._k_zeros, mx.zeros((B, H, pad, n_groups), dtype=mx.float16)], axis=2)
                self._k_packed = mx.concatenate([self._k_packed, mx.zeros((B, H, pad, n_groups, k_pw), dtype=mx.uint32)], axis=2)
                self._v_scales = mx.concatenate([self._v_scales, mx.zeros((B, H, pad, n_groups), dtype=mx.float16)], axis=2)
                self._v_zeros = mx.concatenate([self._v_zeros, mx.zeros((B, H, pad, n_groups), dtype=mx.float16)], axis=2)
                self._v_packed = mx.concatenate([self._v_packed, mx.zeros((B, H, pad, n_groups, v_pw), dtype=mx.uint32)], axis=2)

            self._k_scales[:, :, self.offset:new_end] = k_scales
            self._k_zeros[:, :, self.offset:new_end] = k_zeros
            self._k_packed[:, :, self.offset:new_end] = k_packed
            self._v_scales[:, :, self.offset:new_end] = v_scales
            self._v_zeros[:, :, self.offset:new_end] = v_zeros
            self._v_packed[:, :, self.offset:new_end] = v_packed
            self.offset = new_end
            return self._quantized_state

    @property
    def _quantized_state(self):
        return (
            (self._k_scales[:, :, :self.offset], self._k_zeros[:, :, :self.offset], self._k_packed[:, :, :self.offset]),
            (self._v_scales[:, :, :self.offset], self._v_zeros[:, :, :self.offset], self._v_packed[:, :, :self.offset]),
        )

    @property
    def state(self):
        if self._fp16_keys is not None and not self._quantized:
            return self._fp16_keys[:, :, :self.offset], self._fp16_values[:, :, :self.offset]
        if self._k_scales is None:
            return None, None
        return self._quantized_state

    @state.setter
    def state(self, v):
        if v[0] is None:
            self._k_scales = self._k_zeros = self._k_packed = None
            self._v_scales = self._v_zeros = self._v_packed = None
            self.offset = 0
        else:
            (k_scales, k_zeros, k_packed), (v_scales, v_zeros, v_packed) = v
            self._k_scales, self._k_zeros, self._k_packed = k_scales, k_zeros, k_packed
            self._v_scales, self._v_zeros, self._v_packed = v_scales, v_zeros, v_packed
            self.offset = k_scales.shape[2]

    @property
    def meta_state(self):
        return (self.offset, self.bits, self.seed)

    @meta_state.setter
    def meta_state(self, v):
        if isinstance(v, (list, tuple)) and len(v) >= 3:
            self.offset = int(v[0])
            self.bits = int(v[1])
            self.seed = int(v[2])
        else:
            self.offset = int(v) if not isinstance(v, (list, tuple)) else int(v[0])

    def dequantize(self, keys_state=None, values_state=None):
        """Full dequantize for prefill fallback."""
        if keys_state is None:
            keys_state, values_state = self.state
        k_scales, k_zeros, k_packed = keys_state
        v_scales, v_zeros, v_packed = values_state
        # Lazy codec init from packed tensor shape
        if self._k_codec is None:
            pw = k_packed.shape[-1]
            dim = pw * 32 // self.k_bits
            self._ensure_codecs(dim)
        keys = self._k_codec.dequantize(k_scales, k_zeros, k_packed)
        values = self._v_codec.dequantize(v_scales, v_zeros, v_packed)
        return keys, values

    def decode_attention(
        self,
        queries: mx.array,      # (B, H_q, 1, D)
        keys_state=None,
        values_state=None,
        scale: float = 1.0,
        mask=None,
    ) -> mx.array:
        """Flash Attention from quantized KV using PolarQuantCodec.

        For PolarQuantCodec, we dequantize then apply standard SDPA since
        the codec uses WHT rotation (not MSE codebook-based rotation).

        When sparse_v is enabled and context is long (>1024 tokens), applies
        sparse V decoding to skip low-attention V entries for faster decode.
        """
        if keys_state is None:
            keys_state, values_state = self.state

        B, H_q, L, D = queries.shape

        # Dequantize K and V using PolarQuant codecs
        keys, values = self.dequantize(keys_state, values_state)

        # Apply inverse WHT rotation to dequantized K and V
        keys = apply_wht_inverse(keys, self._k_codec.seed)
        values = apply_wht_inverse(values, self._v_codec.seed)

        # keys/values are (B, H_kv, T, D), queries is (B, H_q, 1, D)
        # For GQA, we need to expand K/V to match Q heads
        if H_q != keys.shape[1]:
            # GQA: repeat K/V for each query head
            gqa_factor = H_q // keys.shape[1]
            keys = mx.concatenate([keys] * gqa_factor, axis=1)
            values = mx.concatenate([values] * gqa_factor, axis=1)

        # Sparse V decoding: skip low-attention V entries for long contexts
        # Only apply when sparse_v=True and context length > 1024
        T = keys.shape[2]
        use_sparse_v = (
            self.sparse_v and
            T > 1024 and
            self.sparse_v_budget < 1.0
        )

        if use_sparse_v:
            # Compute attention scores to determine which V entries to keep
            # Q @ K^T: (B, H_q, 1, D) @ (B, H_q, D, T) -> (B, H_q, 1, T)
            q = queries.astype(mx.float32)
            k = keys.astype(mx.float32)
            scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) * scale

            # Softmax over T dimension: (B, H_q, 1, T)
            scores_max = scores.max(axis=-1, keepdims=True)
            scores_exp = mx.exp(scores - scores_max)
            attn_weights = scores_exp / scores_exp.sum(axis=-1, keepdims=True)

            # Use same attention for all query heads (or first head group)
            # Shape: (B, H_kv, 1, T) after averaging over GQA factor
            gqa_factor = H_q // keys.shape[1]
            if gqa_factor > 1:
                attn_weights = attn_weights.reshape(B, gqa_factor, keys.shape[1], 1, T)
                attn_weights = attn_weights.mean(axis=1)  # (B, H_kv, 1, T)
            else:
                attn_weights = attn_weights.reshape(B, keys.shape[1], 1, T)

            # Compute sparse mask from attention weights
            sparse_mask = _compute_sparse_mask(
                attn_weights.squeeze(2),  # (B, H_kv, T)
                budget=self.sparse_v_budget
            )  # (B, H_kv, T)

            # Expand mask to V dimension: (B, H_kv, T, 1) for broadcasting
            sparse_mask = sparse_mask[..., None]  # (B, H_kv, T, 1)

            # Apply sparse mask to V (zero out low-attention entries)
            values = values * sparse_mask.astype(values.dtype)

        # Use mx.fast.scaled_dot_product_attention with 4D inputs
        out = mx.fast.scaled_dot_product_attention(
            queries.astype(mx.float32), keys.astype(mx.float32), values.astype(mx.float32),
            scale=scale, mask=mask
        )
        return out

    def size(self) -> int:
        return self.offset

    def empty(self) -> bool:
        return self._k_scales is None or self.offset == 0

    @property
    def nbytes(self) -> int:
        if self._k_scales is None:
            return 0
        T = self.offset
        return (
            self._k_scales[:, :, :T].nbytes + self._k_zeros[:, :, :T].nbytes + self._k_packed[:, :, :T].nbytes +
            self._v_scales[:, :, :T].nbytes + self._v_zeros[:, :, :T].nbytes + self._v_packed[:, :, :T].nbytes
        )

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        n = min(self.offset, n)
        self.offset -= n
        return n

    @classmethod
    def from_cache(cls, cache, bits: int = 4, seed: int = 0) -> "TurboQuantKVCache":
        """Convert an existing KVCache to TurboQuantKVCache."""
        tq = cls(bits=bits, seed=seed)
        keys, values = cache.state
        if keys is not None:
            tq.update_and_fetch(keys, values)
        return tq


class BatchTurboQuantKVCache(_BaseCache):
    """Batched TurboQuant KV cache for continuous batching.

    Prefill phase: stores fp16 (like BatchKVCache) for full-quality attention.
    Decode phase: quantizes to TurboQuant for memory-efficient generation.
    Implements all BatchKVCache methods for BatchGenerator compatibility.

    Supports asymmetric K/V compression with separate k_bits and v_bits.
    """
    step = 256

    def __init__(
        self,
        left_padding,
        k_bits: int = 4,
        v_bits: int = 4,
        sparse_v: bool = True,
        sparse_v_budget: float = 0.75,
        seed: int = 0,
        # Legacy compatibility
        bits: int = None,
    ):
        # Support legacy bits parameter for backward compatibility
        if bits is not None:
            self.k_bits = bits
            self.v_bits = bits
        else:
            self.k_bits = k_bits
            self.v_bits = v_bits
        self.bits = self.k_bits  # For external access
        self.sparse_v = sparse_v
        self.sparse_v_budget = sparse_v_budget
        self.seed = seed
        # Safety: mlx-lm's base.py SDPA checks hasattr(cache, "bits") and then
        # accesses cache.group_size for affine quantized caches.  If our attention
        # patch doesn't intercept (e.g. VLM proxy), this prevents AttributeError.
        self.group_size = 0
        # fp16 storage (prefill phase)
        self.keys = None
        self.values = None
        self.left_padding = mx.array(left_padding)
        self.offset = mx.array([-l for l in left_padding])
        self._idx = 0
        self._right_padding = None
        # Quantized storage (decode phase)
        self._k_scales = None
        self._k_zeros = None
        self._k_packed = None
        self._v_scales = None
        self._v_zeros = None
        self._v_packed = None
        self._quantized = False
        self._k_codec = None
        self._v_codec = None

        # Warn for 2-bit mode
        if self.k_bits == 2 or self.v_bits == 2:
            logger.warning(
                "TurboQuant 2-bit mode enabled. Expect ~6.5%% perplexity increase. "
                "Recommended only for extreme memory constraints."
            )

    def _ensure_codecs(self, dim):
        if self._k_codec is None:
            self._k_codec = PolarQuantCodec(dim, self.k_bits, seed=self.seed)
        if self._v_codec is None:
            self._v_codec = PolarQuantCodec(dim, self.v_bits, seed=self.seed)

    def _quantize_buffer(self):
        """Convert fp16 KV to quantized. Called at decode start."""
        if self._quantized or self.keys is None:
            return
        B, H, T, D = self.keys.shape
        logger.info(f"TurboQuant batch: quantizing {self._idx} tokens ({B}×{H} heads, dim={D}) to K={self.k_bits}-bit, V={self.v_bits}-bit")
        self._ensure_codecs(D)
        # Quantize full buffer
        k = self.keys[..., :self._idx, :]
        v = self.values[..., :self._idx, :]
        k_scales, k_zeros, k_packed = self._k_codec.quantize(k)
        v_scales, v_zeros, v_packed = self._v_codec.quantize(v)
        self._k_scales = k_scales
        self._k_zeros = k_zeros
        self._k_packed = k_packed
        self._v_scales = v_scales
        self._v_zeros = v_zeros
        self._v_packed = v_packed
        self._quantized = True
        # Free fp16
        self.keys = None
        self.values = None

    def update_and_fetch(self, keys, values):
        B, H, T_new, D = keys.shape

        if T_new > 1:
            # Prefill: fp16 (same logic as BatchKVCache)
            prev = self._idx
            if self.keys is None or (prev + T_new) > self.keys.shape[2]:
                n_steps = (self.step + T_new - 1) // self.step
                k_shape = (B, H, n_steps * self.step, D)
                v_shape = (B, H, n_steps * self.step, values.shape[3])
                new_k = mx.zeros(k_shape, keys.dtype)
                new_v = mx.zeros(v_shape, values.dtype)
                if self.keys is not None:
                    if prev % self.step != 0:
                        self.keys = self.keys[..., :prev, :]
                        self.values = self.values[..., :prev, :]
                    self.keys = mx.concatenate([self.keys, new_k], axis=2)
                    self.values = mx.concatenate([self.values, new_v], axis=2)
                else:
                    self.keys, self.values = new_k, new_v
            self.offset += T_new
            self._idx += T_new
            self.keys[..., prev:self._idx, :] = keys
            self.values[..., prev:self._idx, :] = values
            return self.keys[..., :self._idx, :], self.values[..., :self._idx, :]
        else:
            # Decode: quantize on first token
            if not self._quantized:
                self._quantize_buffer()

            self._ensure_codec(D)
            k_norms, k_packed = self._codec.quantize(keys)
            v_norms, v_packed = self._codec.quantize(values)
            pw = _packed_width(D, self.bits)

            # Grow quantized storage
            new_idx = self._idx + 1
            if new_idx > self._k_norms.shape[2]:
                alloc = ((new_idx + self.step - 1) // self.step) * self.step
                pad = alloc - self._k_norms.shape[2]
                self._k_norms = mx.concatenate([self._k_norms, mx.zeros((B, H, pad), dtype=mx.float32)], axis=2)
                self._k_packed = mx.concatenate([self._k_packed, mx.zeros((B, H, pad, pw), dtype=mx.uint32)], axis=2)
                self._v_norms = mx.concatenate([self._v_norms, mx.zeros((B, H, pad), dtype=mx.float32)], axis=2)
                self._v_packed = mx.concatenate([self._v_packed, mx.zeros((B, H, pad, pw), dtype=mx.uint32)], axis=2)

            self._k_norms[:, :, self._idx:new_idx] = k_norms
            self._k_packed[:, :, self._idx:new_idx] = k_packed
            self._v_norms[:, :, self._idx:new_idx] = v_norms
            self._v_packed[:, :, self._idx:new_idx] = v_packed
            self.offset += 1
            self._idx = new_idx
            return self._quantized_state

    @property
    def _quantized_state(self):
        return (
            (self._k_norms[:, :, :self._idx], self._k_packed[:, :, :self._idx]),
            (self._v_norms[:, :, :self._idx], self._v_packed[:, :, :self._idx]),
        )

    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        if left_padding is not None:
            if self.keys is not None:
                raise ValueError("Left padding can only be added to empty cache")
            left_padding = mx.array(left_padding)
            self.left_padding += left_padding
            self.offset -= left_padding
        if right_padding is not None and max(right_padding) > 0:
            self._right_padding = mx.array(right_padding)

    def finalize(self):
        if self._right_padding is not None and not self._quantized:
            from mlx_lm.models.cache import dynamic_roll
            padding = self._right_padding
            self.keys = dynamic_roll(self.keys, padding[:, None], axis=2)
            self.values = dynamic_roll(self.values, padding[:, None], axis=2)
            self.offset -= padding
            self.left_padding += padding
            self._right_padding = None

    @property
    def state(self):
        if self._quantized:
            return self._quantized_state
        if self.keys is None:
            return None, None, self.offset, self.left_padding
        k, v = self.keys, self.values
        if self._idx < k.shape[2]:
            k = k[..., :self._idx, :]
            v = v[..., :self._idx, :]
        return k, v, self.offset, self.left_padding

    @state.setter
    def state(self, v):
        if len(v) == 4:
            self.keys, self.values, self.offset, self.left_padding = v
            self._idx = self.keys.shape[2] if self.keys is not None else 0
        else:
            # Quantized state
            (self._k_norms, self._k_packed), (self._v_norms, self._v_packed) = v[0], v[1]
            self._idx = self._k_norms.shape[2]
            self._quantized = True

    def filter(self, batch_indices):
        if self._quantized:
            self._k_norms = self._k_norms[batch_indices]
            self._k_packed = self._k_packed[batch_indices]
            self._v_norms = self._v_norms[batch_indices]
            self._v_packed = self._v_packed[batch_indices]
        else:
            self.keys = self.keys[batch_indices]
            self.values = self.values[batch_indices]
        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]
        min_left_pad = self.left_padding.min().item()
        if min_left_pad > 0 and not self._quantized:
            self.keys = self.keys[..., min_left_pad:, :]
            self.values = self.values[..., min_left_pad:, :]
            self._idx -= min_left_pad
            self.left_padding -= min_left_pad

    def extend(self, other):
        if self._quantized != other._quantized:
            # Force both to quantized
            if not self._quantized:
                self._quantize_buffer()
            if not other._quantized:
                other._quantize_buffer()

        if self._quantized:
            max_idx = max(self._idx, other._idx)
            # Pad quantized tensors (trim to _idx first, then left-pad)
            def pad_q(c):
                # Trim to actual used length
                kn = c._k_norms[:, :, :c._idx]
                kp = c._k_packed[:, :, :c._idx]
                vn = c._v_norms[:, :, :c._idx]
                vp = c._v_packed[:, :, :c._idx]
                left = max_idx - c._idx
                if left > 0:
                    B, H = kn.shape[:2]
                    pw = kp.shape[-1]
                    kn = mx.concatenate([mx.zeros((B, H, left), dtype=mx.float32), kn], axis=2)
                    kp = mx.concatenate([mx.zeros((B, H, left, pw), dtype=mx.uint32), kp], axis=2)
                    vn = mx.concatenate([mx.zeros((B, H, left), dtype=mx.float32), vn], axis=2)
                    vp = mx.concatenate([mx.zeros((B, H, left, pw), dtype=mx.uint32), vp], axis=2)
                return kn, kp, vn, vp, c.offset, c.left_padding + left
            s_kn, s_kp, s_vn, s_vp, s_off, s_lp = pad_q(self)
            o_kn, o_kp, o_vn, o_vp, o_off, o_lp = pad_q(other)
            self._k_norms = mx.concatenate([s_kn, o_kn], axis=0)
            self._k_packed = mx.concatenate([s_kp, o_kp], axis=0)
            self._v_norms = mx.concatenate([s_vn, o_vn], axis=0)
            self._v_packed = mx.concatenate([s_vp, o_vp], axis=0)
            self.offset = mx.concatenate([s_off, o_off])
            self.left_padding = mx.concatenate([s_lp, o_lp])
            self._idx = max_idx
        else:
            # fp16 extend (same as BatchKVCache)
            max_idx = max(self._idx, other._idx)
            max_size = max(self.keys.shape[2], other.keys.shape[2])
            def pad(c):
                left = max_idx - c._idx
                right = max_size - c.keys.shape[2] - left
                k, v = c.keys, c.values
                if right < 0:
                    k = k[..., :right, :]; v = v[..., :right, :]
                    right = 0
                if left != 0 or right != 0:
                    p = [(0,0),(0,0),(left,right),(0,0)]
                    k = mx.pad(k, p); v = mx.pad(v, p)
                return k, v, c.offset, c.left_padding + left
            self.keys, self.values, self.offset, self.left_padding = map(
                mx.concatenate, zip(*(pad(self), pad(other)))
            )
            self._idx = max_idx

    def extract(self, idx):
        """Extract single request as TurboQuantKVCache."""
        if not self._quantized:
            self._quantize_buffer()
        padding = self.left_padding[idx].item()
        tq = TurboQuantKVCache(bits=self.bits, seed=self.seed)
        tq._ensure_codec(self._k_packed.shape[-1] * 32 // self.bits)  # infer dim from packed_width
        # Copy quantized data for this request
        end = self._idx
        tq._k_norms = mx.contiguous(self._k_norms[idx:idx+1, :, padding:end])
        tq._k_packed = mx.contiguous(self._k_packed[idx:idx+1, :, padding:end])
        tq._v_norms = mx.contiguous(self._v_norms[idx:idx+1, :, padding:end])
        tq._v_packed = mx.contiguous(self._v_packed[idx:idx+1, :, padding:end])
        tq.offset = end - padding
        tq._quantized = True
        tq._codec = self._codec
        return tq

    @classmethod
    def merge(cls, caches):
        """Merge TurboQuantKVCache instances into BatchTurboQuantKVCache."""
        bits = caches[0].bits
        seed = caches[0].seed
        lengths = [c.offset for c in caches]
        max_length = max(lengths)
        padding = [max_length - l for l in lengths]
        B = len(caches)

        batch = cls(padding, bits=bits, seed=seed)
        batch._codec = caches[0]._codec

        # All caches should be quantized
        if not all(c._quantized for c in caches):
            # Force quantize
            for c in caches:
                if not c._quantized and c._fp16_keys is not None:
                    c._quantize_fp16_buffer()

        H = caches[0]._k_norms.shape[1]
        pw = caches[0]._k_packed.shape[-1]

        k_norms = mx.zeros((B, H, max_length), dtype=mx.float32)
        k_packed = mx.zeros((B, H, max_length, pw), dtype=mx.uint32)
        v_norms = mx.zeros((B, H, max_length), dtype=mx.float32)
        v_packed = mx.zeros((B, H, max_length, pw), dtype=mx.uint32)

        for i, (p, c) in enumerate(zip(padding, caches)):
            T = c.offset
            if c._k_norms is not None:
                k_norms[i:i+1, :, p:p+T] = c._k_norms[:, :, :T]
                k_packed[i:i+1, :, p:p+T] = c._k_packed[:, :, :T]
                v_norms[i:i+1, :, p:p+T] = c._v_norms[:, :, :T]
                v_packed[i:i+1, :, p:p+T] = c._v_packed[:, :, :T]

        batch._k_norms = k_norms
        batch._k_packed = k_packed
        batch._v_norms = v_norms
        batch._v_packed = v_packed
        batch.offset += max_length
        batch._idx = max_length
        batch._quantized = True
        return batch

    def decode_attention(
        self,
        queries: mx.array,      # (B, H_q, 1, D)
        keys_state=None,
        values_state=None,
        scale: float = 1.0,
        mask=None,
    ) -> mx.array:
        """Fused 2-pass Flash Attention for batch decode. No dequantize."""
        if keys_state is None:
            keys_state, values_state = self._quantized_state
        k_norms, k_packed = keys_state
        v_norms, v_packed = values_state

        B, H_q, L, D = queries.shape
        H_kv = k_norms.shape[1]

        # Prepare queries: scale and flatten to (B*H_q, D)
        q_flat = (queries.squeeze(2) * scale).reshape(B * H_q, D).astype(mx.float16)

        # Rotate queries (same rotation as codec)
        R = self._codec.rotation
        q_grouped = q_flat.reshape(B * H_q, D // self._codec.dim, self._codec.dim)
        q_rot = (q_grouped.astype(mx.float32) @ R).reshape(B * H_q, D).astype(mx.float16)

        # Fused 2-pass SDPA
        out = _fused_tq_sdpa(
            q_rot, k_packed, k_norms, self._codec.codebook,
            v_packed, v_norms, self._codec.codebook,
            scale=1.0,  # already applied to queries
            B=B, H_q=H_q, H_kv=H_kv, D=D, bits=self.bits,
        )

        # Inverse rotate output
        out_grouped = out.reshape(B * H_q, D // self._codec.dim, self._codec.dim).astype(mx.float32)
        out_restored = (out_grouped @ R.T).reshape(B * H_q, D).astype(queries.dtype)

        return out_restored.reshape(B, H_q, 1, D)

    def dequantize(self, keys_state=None, values_state=None):
        """Dequantize batch quantized state for SDPA fallback."""
        if keys_state is None:
            keys_state, values_state = self._quantized_state
        k_norms, k_packed = keys_state
        v_norms, v_packed = values_state
        keys = self._codec.dequantize(k_norms, k_packed)
        values = self._codec.dequantize(v_norms, v_packed)
        return keys, values

    def make_mask(self, N, return_array=False, **kwargs):
        from mlx_lm.models.cache import create_causal_mask
        return create_causal_mask(
            N, offset=self._idx, left_padding=self.left_padding, **kwargs
        )

    def empty(self):
        return self.keys is None and self._k_norms is None

    @property
    def nbytes(self):
        if self._quantized:
            return (self._k_norms.nbytes + self._k_packed.nbytes +
                    self._v_norms.nbytes + self._v_packed.nbytes)
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self._idx, n)
        self._idx -= n
        self.offset -= n
        return n

    def size(self):
        return max(0, max(self.offset.tolist())) if isinstance(self.offset, mx.array) else self.offset

    @property
    def meta_state(self):
        return ""

    @meta_state.setter
    def meta_state(self, v):
        pass


def turboquant_enabled(bits, scheme=None):
    """Check if TurboQuant should be used for given bits/scheme."""
    if scheme == "turboquant":
        return True
    if bits is not None and not float(bits).is_integer():
        return True
    return False
