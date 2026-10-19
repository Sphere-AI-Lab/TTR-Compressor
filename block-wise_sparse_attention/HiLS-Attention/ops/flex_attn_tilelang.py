"""
Flex Attention
=====================
TileLang implementation of the HiLS-Attention flex attention kernel.

Author: Xinyu Wei
"""

import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T


# ---------------------------------------------------------------------------
# Forward kernel (fused train + inference via use_cache flag)
# ---------------------------------------------------------------------------
@tilelang.jit(
    out_idx=[3, 4],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def flex_attn_fwd(
    batch,
    heads,
    q_len,
    kv_len,
    dim_qk,
    dim_v,
    window_size,
    chunk_size,
    mask_lmk,
    expand_to_chunk,
    block_M,
    block_N,
    groups=1,
    use_cache=False,
    sm_scale=None,
):
    """Forward kernel.

    When use_cache=False, q_len/kv_len are compile-time constants (must equal).
    When use_cache=True, q_len/kv_len are runtime dynamic (q_len <= kv_len).

    Tail handling: every global load / store is guarded by `tq < q_len_var`
    or `ts < kv_len_var`, so non-multiple-of-block seq lengths are handled
    directly by the kernel without requiring the caller to pad. OOB slots
    in Q/K/V are filled with zeros in shared memory and then masked to
    -inf before the softmax, matching the tail strategy used by
    `ops/topk_head_softmax.py`.

    `sm_scale` is the natural-log-domain softmax scale applied to the raw
    Q @ K^T scores (i.e. `score = sm_scale * <q, k>`). Default (None) is
    `1 / sqrt(dim_qk)`, matching the canonical attention scale.
    """
    _base = (1.0 / dim_qk) ** 0.5 if sm_scale is None else float(sm_scale)
    scale = _base * 1.44269504  # log2(e)
    head_kv = heads // groups
    dtype = "bfloat16"
    accum_dtype = "float"

    if use_cache:
        q_len_var = T.dynamic("q_len")
        kv_len_var = T.dynamic("kv_len")
    else:
        q_len_var = q_len
        kv_len_var = kv_len

    q_shape = [batch, q_len_var, heads, dim_qk]
    k_shape = [batch, kv_len_var, head_kv, dim_qk]
    v_shape = [batch, kv_len_var, head_kv, dim_v]
    o_shape = [batch, q_len_var, heads, dim_v]
    lse_shape = [batch, q_len_var, heads]

    @T.prim_func
    def fwd(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        K: T.Tensor(k_shape, dtype),  # type: ignore
        V: T.Tensor(v_shape, dtype),  # type: ignore
        Output: T.Tensor(o_shape, dtype),  # type: ignore
        Lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
        KVStart: T.Tensor([1], "int32"),  # type: ignore
    ):
        with T.Kernel(T.ceildiv(q_len_var, block_M), heads, batch, threads=128) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim_qk], dtype)
            K_shared = T.alloc_shared([block_N, dim_qk], dtype)
            V_shared = T.alloc_shared([block_N, dim_v], dtype)
            O_shared = T.alloc_shared([block_M, dim_v], dtype)
            lse_out_shared = T.alloc_shared([block_M], accum_dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, dim_v], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            # Guarded load Q: OOB q rows are filled with 0 so the GEMM output
            # is 0 for those rows, then masked to -inf below.
            for i, j in T.Parallel(block_M, dim_qk):
                tq = bx * block_M + i
                if tq < q_len_var:
                    Q_shared[i, j] = Q[bz, tq, by, j]
                else:
                    Q_shared[i, j] = T.Cast(dtype, 0)

            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            # In inference, absolute q_idx = kv_start + (kv_len - q_len) + (bx * block_M + i).
            # KV tensor indices are raw indices into the possibly-truncated cache;
            # mask/window/lmk semantics use absolute indices.
            q_offset = kv_len_var - q_len_var
            kv_start = KVStart[0]

            q_block_start_abs = kv_start + bx * block_M + q_offset
            q_block_end_abs = kv_start + (bx + 1) * block_M + q_offset

            if expand_to_chunk:
                tmp_left = q_block_start_abs - window_size + 1
                left_kv_abs_raw = T.floordiv(tmp_left, chunk_size) * chunk_size
            else:
                left_kv_abs_raw = q_block_start_abs - window_size + 1
            left_kv = T.max(left_kv_abs_raw - kv_start, 0)
            right_kv = T.min(q_block_end_abs - kv_start, kv_len_var)

            loop_st = T.floordiv(left_kv, block_N)
            loop_ed = T.ceildiv(right_kv, block_N)

            for k in T.Pipelined(loop_st, loop_ed, num_stages=1):
                # Guarded load K / V: OOB kv rows are filled with 0 and then
                # get masked to -inf (K) / multiplied by the -inf mask (V) so
                # they contribute nothing to the softmax.
                for i, j in T.Parallel(block_N, dim_qk):
                    ts = k * block_N + i
                    if ts < kv_len_var:
                        K_shared[i, j] = K[bz, ts, by // groups, j]
                    else:
                        K_shared[i, j] = T.Cast(dtype, 0)
                for i, j in T.Parallel(block_N, dim_v):
                    ts = k * block_N + i
                    if ts < kv_len_var:
                        V_shared[i, j] = V[bz, ts, by // groups, j]
                    else:
                        V_shared[i, j] = T.Cast(dtype, 0)

                T.clear(acc_s)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                if expand_to_chunk and mask_lmk:
                    for i, j in T.Parallel(block_M, block_N):
                        q_raw_idx = bx * block_M + i
                        kv_raw_idx = k * block_N + j
                        q_abs_idx = kv_start + q_raw_idx + q_offset
                        kv_abs_idx = kv_start + kv_raw_idx
                        chunk_start = T.floordiv(q_abs_idx - window_size + 1, chunk_size) * chunk_size
                        acc_s[i, j] = T.if_then_else(
                            (kv_abs_idx >= chunk_start)
                            & (kv_abs_idx <= q_abs_idx)
                            & (kv_raw_idx < kv_len_var)
                            & (q_raw_idx < q_len_var)
                            & (T.floormod(kv_abs_idx + 1, chunk_size) != 0),
                            acc_s[i, j],
                            -T.infinity(accum_dtype),
                        )
                elif expand_to_chunk and (not mask_lmk):
                    for i, j in T.Parallel(block_M, block_N):
                        q_raw_idx = bx * block_M + i
                        kv_raw_idx = k * block_N + j
                        q_abs_idx = kv_start + q_raw_idx + q_offset
                        kv_abs_idx = kv_start + kv_raw_idx
                        chunk_start = T.floordiv(q_abs_idx - window_size + 1, chunk_size) * chunk_size
                        acc_s[i, j] = T.if_then_else(
                            (kv_abs_idx >= chunk_start)
                            & (kv_abs_idx <= q_abs_idx)
                            & (kv_raw_idx < kv_len_var)
                            & (q_raw_idx < q_len_var),
                            acc_s[i, j],
                            -T.infinity(accum_dtype),
                        )
                elif (not expand_to_chunk) and mask_lmk:
                    for i, j in T.Parallel(block_M, block_N):
                        q_raw_idx = bx * block_M + i
                        kv_raw_idx = k * block_N + j
                        q_abs_idx = kv_start + q_raw_idx + q_offset
                        kv_abs_idx = kv_start + kv_raw_idx
                        chunk_start = q_abs_idx - window_size + 1
                        acc_s[i, j] = T.if_then_else(
                            (kv_abs_idx >= chunk_start)
                            & (kv_abs_idx <= q_abs_idx)
                            & (kv_raw_idx < kv_len_var)
                            & (q_raw_idx < q_len_var)
                            & (T.floormod(kv_abs_idx + 1, chunk_size) != 0),
                            acc_s[i, j],
                            -T.infinity(accum_dtype),
                        )
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        q_raw_idx = bx * block_M + i
                        kv_raw_idx = k * block_N + j
                        q_abs_idx = kv_start + q_raw_idx + q_offset
                        kv_abs_idx = kv_start + kv_raw_idx
                        chunk_start = q_abs_idx - window_size + 1
                        acc_s[i, j] = T.if_then_else(
                            (kv_abs_idx >= chunk_start)
                            & (kv_abs_idx <= q_abs_idx)
                            & (kv_raw_idx < kv_len_var)
                            & (q_raw_idx < q_len_var),
                            acc_s[i, j],
                            -T.infinity(accum_dtype),
                        )

                T.copy(scores_max, scores_max_prev)
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                # Guard: when the row has no visible KV yet (scores_max == -inf),
                # naive exp2(-inf - (-inf)) is NaN. Use scale=1.0 in that case
                # (acc_o is still 0 anyway).
                for i in T.Parallel(block_M):
                    scores_scale[i] = T.if_then_else(
                        scores_max[i] > -1e30,
                        T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale),
                        1.0,
                    )
                for i, j in T.Parallel(block_M, dim_v):
                    acc_o[i, j] *= scores_scale[i]
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.if_then_else(
                        scores_max[i] > -1e30,
                        T.exp2(acc_s[i, j] * scale - scores_max[i] * scale),
                        0.0,
                    )
                T.copy(acc_s, acc_s_cast)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

            # finalize: O = acc_o / logsum, lse in log2 domain (scale * x).
            for i, j in T.Parallel(block_M, dim_v):
                acc_o[i, j] = T.if_then_else(logsum[i] > 0, acc_o[i, j] / logsum[i], 0.0)
            for i in T.Parallel(block_M):
                logsum[i] = T.if_then_else(
                    logsum[i] > 0,
                    T.log2(logsum[i]) + scores_max[i] * scale,
                    -T.infinity(accum_dtype),
                )
            # Stage fragment -> shared, then guarded store to global so tail
            # rows (tq >= q_len_var) never write OOB.
            T.copy(acc_o, O_shared)
            T.copy(logsum, lse_out_shared)
            for i, j in T.Parallel(block_M, dim_v):
                tq = bx * block_M + i
                if tq < q_len_var:
                    Output[bz, tq, by, j] = O_shared[i, j]
            for i in T.Parallel(block_M):
                tq = bx * block_M + i
                if tq < q_len_var:
                    Lse[bz, tq, by] = lse_out_shared[i]

    return fwd


# ---------------------------------------------------------------------------
# Backward preprocess (Delta = sum(O * dO))
# ---------------------------------------------------------------------------
@tilelang.jit(
    out_idx=[2],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def flex_attn_bwd_preprocess(batch, heads, seq_len, dim_v):
    """Compute Delta = sum(O * dO, dim=-1).

    Tail handling: per-element guarded load / store -- when seq_len is not a
    multiple of `blk`, OOB rows skip both the read (so we never read garbage
    bytes) and the write (so we never corrupt neighbouring tensors).
    """
    dtype = "bfloat16"
    accum_dtype = "float"
    shape = [batch, seq_len, heads, dim_v]
    blk = 32

    @T.prim_func
    def prep(
        O: T.Tensor(shape, dtype),  # type: ignore
        dO: T.Tensor(shape, dtype),  # type: ignore
        Delta: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
    ):
        with T.Kernel(heads, T.ceildiv(seq_len, blk), batch) as (bx, by, bz):
            o = T.alloc_fragment([blk, blk], dtype)
            do = T.alloc_fragment([blk, blk], dtype)
            acc = T.alloc_fragment([blk, blk], accum_dtype)
            delta = T.alloc_fragment([blk], accum_dtype)
            delta_shared = T.alloc_shared([blk], accum_dtype)
            T.clear(acc)
            for k in range(T.ceildiv(dim_v, blk)):
                # Guarded load: tail rows are filled with 0 so the multiply
                # below contributes 0 to the accumulator.
                for i, j in T.Parallel(blk, blk):
                    tq = by * blk + i
                    if tq < seq_len:
                        o[i, j] = O[bz, tq, bx, k * blk + j]
                        do[i, j] = dO[bz, tq, bx, k * blk + j]
                    else:
                        o[i, j] = T.Cast(dtype, 0)
                        do[i, j] = T.Cast(dtype, 0)
                for i, j in T.Parallel(blk, blk):
                    acc[i, j] += o[i, j] * do[i, j]
            T.reduce_sum(acc, delta, 1)
            # Stage fragment -> shared, then guarded store.
            T.copy(delta, delta_shared)
            for i in T.Parallel(blk):
                tq = by * blk + i
                if tq < seq_len:
                    Delta[bz, bx, tq] = delta_shared[i]

    return prep


def make_dq_layout(dQ):
    return T.Layout(
        dQ.shape,
        lambda b, l, h, d: [b, l // 8, h, d // 8, (d % 2), 4 * (l % 8) + (d % 8) // 2],
    )


@tilelang.jit(
    out_idx=[1],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def flex_attn_bwd_postprocess(batch, heads, seq_len, seq_len_padded, dim_qk):
    """Cast swizzled accumulator-precision dQ to bf16 with guarded tail copy."""
    dtype = "bfloat16"
    accum_dtype = "float"
    swizzled_shape = [batch, seq_len_padded, heads, dim_qk]
    output_shape = [batch, seq_len, heads, dim_qk]
    blk = 64

    @T.prim_func
    def post(
        dQ: T.Tensor(swizzled_shape, accum_dtype),  # type: ignore
        dQ_out: T.Tensor(output_shape, dtype),  # type: ignore
    ):
        with T.Kernel(T.ceildiv(seq_len, blk), heads, batch, threads=128) as (bx, by, bz):
            T.annotate_layout({dQ: make_dq_layout(dQ)})
            for i, d in T.Parallel(blk, dim_qk):
                tq = bx * blk + i
                if tq < seq_len:
                    dQ_out[bz, tq, by, d] = T.Cast(dtype, dQ[bz, tq, by, d])

    return post


# ---------------------------------------------------------------------------
# Backward kernel (swizzled dQ + split dK/dV, training only)
# ---------------------------------------------------------------------------
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def flex_attn_bwd(
    batch,
    heads,
    seq_len,
    dim_qk,
    dim_v,
    window_size,
    chunk_size,
    mask_lmk,
    expand_to_chunk,
    block_M,
    block_N,
    threads=256,
    num_stages=2,
    groups=1,
    sm_scale=None,
):
    """Backward kernel.

    Tail handling: every global load / store is guarded by `tq < seq_len` or
    `ts < seq_len`, mirroring the strategy used by `ops/topk_head_softmax.py`.
    OOB rows in K / V / Q / dO / lse / Delta are filled with zero in shared
    memory; the same row/col bound terms in the mask zero out their
    contributions to qkT before any GEMM that produces dq / dk / dv.
    dQ uses the swizzled atomic-add layout; dK / dV are written to split group
    buffers with T.copy and reduced on the Python side.

    `sm_scale` must match the forward kernel's `sm_scale` (default
    `1 / sqrt(dim_qk)` when None).
    """
    sm_scale = (1.0 / dim_qk) ** 0.5 if sm_scale is None else float(sm_scale)
    scale = sm_scale * 1.44269504
    head_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim_qk]
    k_shape = [batch, seq_len, head_kv, dim_qk]
    v_shape = [batch, seq_len, head_kv, dim_v]
    dq_len_padded = tilelang.cdiv(seq_len, 8) * 8
    dq_shape = [batch, dq_len_padded, heads, dim_qk]
    dk_shape = [groups, batch, seq_len, head_kv, dim_qk]
    dv_shape = [groups, batch, seq_len, head_kv, dim_v]
    dtype = "bfloat16"
    accum_dtype = "float"

    @T.prim_func
    def bwd(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        K: T.Tensor(k_shape, dtype),  # type: ignore
        V: T.Tensor(v_shape, dtype),  # type: ignore
        dO: T.Tensor([batch, seq_len, heads, dim_v], dtype),  # type: ignore
        lse: T.Tensor([batch, seq_len, heads], accum_dtype),  # type: ignore
        Delta: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
        dQ: T.Tensor(dq_shape, accum_dtype),  # type: ignore
        dK: T.Tensor(dk_shape, accum_dtype),  # type: ignore
        dV: T.Tensor(dv_shape, accum_dtype),  # type: ignore
    ):
        with T.Kernel(heads, T.ceildiv(seq_len, block_M), batch, threads=threads) as (bx, by, bz):
            K_shared = T.alloc_shared([block_M, dim_qk], dtype)
            dsT_shared = T.alloc_shared([block_M, block_N], dtype)
            q = T.alloc_shared([block_N, dim_qk], dtype)
            V_shared = T.alloc_shared([block_M, dim_v], dtype)
            qkT = T.alloc_fragment([block_M, block_N], accum_dtype)
            dsT = T.alloc_fragment([block_M, block_N], accum_dtype)
            qkT_cast = T.alloc_fragment([block_M, block_N], dtype)
            dsT_cast = T.alloc_fragment([block_M, block_N], dtype)
            lse_shared = T.alloc_shared([block_N], accum_dtype)
            delta = T.alloc_shared([block_N], accum_dtype)
            do = T.alloc_shared([block_N, dim_v], dtype)
            dv = T.alloc_fragment([block_M, dim_v], accum_dtype)
            dk = T.alloc_fragment([block_M, dim_qk], accum_dtype)
            dq = T.alloc_fragment([block_N, dim_qk], accum_dtype)
            dv_shared = T.alloc_shared([block_M, dim_v], accum_dtype)
            dk_shared = T.alloc_shared([block_M, dim_qk], accum_dtype)

            T.annotate_layout({dQ: make_dq_layout(dQ)})

            # Guarded load K / V (per-element). Tail kv rows are filled with 0.
            # Their contributions are then zeroed by the (kv_real < seq_len)
            # mask term before being used by qkT_cast @ do = dv and dsT @ q = dk.
            for i, j in T.Parallel(block_M, dim_qk):
                ts = by * block_M + i
                if ts < seq_len:
                    K_shared[i, j] = K[bz, ts, bx // groups, j]
                else:
                    K_shared[i, j] = T.Cast(dtype, 0)
            for i, j in T.Parallel(block_M, dim_v):
                ts = by * block_M + i
                if ts < seq_len:
                    V_shared[i, j] = V[bz, ts, bx // groups, j]
                else:
                    V_shared[i, j] = T.Cast(dtype, 0)

            T.clear(dv)
            T.clear(dk)

            # Tighter loop range: only iterate over q-blocks that can have a
            # non-empty mask intersection with this kv-block.
            kv_lo = by * block_M
            kv_hi = (by + 1) * block_M
            q_lo = kv_lo
            # Preserve symbolic loop bounds.
            q_hi = T.min(kv_hi + window_size + chunk_size, seq_len)

            loop_st = T.floordiv(q_lo, block_N)
            loop_ed = T.ceildiv(q_hi, block_N)

            for k in T.Pipelined(loop_st, loop_ed, num_stages=num_stages):
                # Guarded load Q / dO / lse / Delta.
                for i, j in T.Parallel(block_N, dim_qk):
                    tq = k * block_N + i
                    if tq < seq_len:
                        q[i, j] = Q[bz, tq, bx, j]
                    else:
                        q[i, j] = T.Cast(dtype, 0)
                for i, j in T.Parallel(block_N, dim_v):
                    tq = k * block_N + i
                    if tq < seq_len:
                        do[i, j] = dO[bz, tq, bx, j]
                    else:
                        do[i, j] = T.Cast(dtype, 0)
                for i in T.Parallel(block_N):
                    tq = k * block_N + i
                    if tq < seq_len:
                        lse_shared[i] = lse[bz, tq, bx]
                    else:
                        # Use 0, not -inf: the (q_real < seq_len) mask below
                        # will zero out qkT for these rows anyway, and 0
                        # avoids any -inf - (-inf) NaN risk in exp2.
                        lse_shared[i] = T.Cast(accum_dtype, 0)
                for i in T.Parallel(block_N):
                    tq = k * block_N + i
                    if tq < seq_len:
                        delta[i] = Delta[bz, bx, tq]
                    else:
                        delta[i] = T.Cast(accum_dtype, 0)

                T.clear(qkT)
                T.gemm(K_shared, q, qkT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                # Guard: when lse == -inf (fully-masked query row, e.g. landmark
                # tokens with mask_lmk=True), exp2(x - (-inf)) = +inf → NaN.
                # Mirror the forward kernel guard: use 0.0 for those rows.
                for i, j in T.Parallel(block_M, block_N):
                    qkT[i, j] = T.if_then_else(
                        lse_shared[j] > -1e30,
                        T.exp2(qkT[i, j] * scale - lse_shared[j]),
                        0.0,
                    )

                # Mask: window + (optional) lmk + causal + bounds.
                # The (q_real < seq_len) & (kv_real < seq_len) terms also zero
                # out OOB tail rows / cols.
                for i, j in T.Parallel(block_M, block_N):
                    kv_real = by * block_M + i
                    q_real = k * block_N + j
                    if expand_to_chunk:
                        chunk_start = T.floordiv(q_real - window_size + 1, chunk_size) * chunk_size
                    else:
                        chunk_start = q_real - window_size + 1
                    if mask_lmk:
                        qkT[i, j] = T.if_then_else(
                            (kv_real >= chunk_start)
                            & (kv_real <= q_real)
                            & (kv_real < seq_len)
                            & (q_real < seq_len)
                            & (T.floormod(kv_real + 1, chunk_size) != 0),
                            qkT[i, j],
                            0.0,
                        )
                    else:
                        qkT[i, j] = T.if_then_else(
                            (kv_real >= chunk_start)
                            & (kv_real <= q_real)
                            & (kv_real < seq_len)
                            & (q_real < seq_len),
                            qkT[i, j],
                            0.0,
                        )

                T.clear(dsT)
                T.gemm(V_shared, do, dsT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(qkT, qkT_cast)
                T.gemm(qkT_cast, do, dv, policy=T.GemmWarpPolicy.FullRow)

                for i, j in T.Parallel(block_M, block_N):
                    dsT_cast[i, j] = qkT[i, j] * (dsT[i, j] - delta[j]) * sm_scale

                T.gemm(dsT_cast, q, dk, policy=T.GemmWarpPolicy.FullRow)

                T.copy(dsT_cast, dsT_shared)
                T.clear(dq)
                T.gemm(dsT_shared, K_shared, dq, transpose_A=True)

                # Per-row guarded atomic_add to dQ.
                # Pattern matches topk kernel (T.Parallel + if + fragment atomic):
                # `for g_row, k in T.Parallel(M_G, BK):
                #      if tq < q_len: T.atomic_add(DQ[...], dQ_local[g_row, k])`
                for i, j in T.Parallel(block_N, dim_qk):
                    if (k * block_N + i) < seq_len:
                        T.atomic_add(dQ[bz, k * block_N + i, bx, j], dq[i, j])

            # Final dK / dV direct write to group-split buffers.
            g_idx = bx % groups
            h_kv = bx // groups
            kv_start = by * block_M
            T.copy(dk, dk_shared)
            T.copy(dv, dv_shared)
            for i, d in T.Parallel(block_M, dim_qk):
                ts = kv_start + i
                if ts < seq_len:
                    dK[g_idx, bz, ts, h_kv, d] = dk_shared[i, d]
            for i, d in T.Parallel(block_M, dim_v):
                ts = kv_start + i
                if ts < seq_len:
                    dV[g_idx, bz, ts, h_kv, d] = dv_shared[i, d]

    return bwd


# ---------------------------------------------------------------------------
# Autograd Function
# ---------------------------------------------------------------------------
def _as_standard_blhd(t: torch.Tensor) -> torch.Tensor:
    """Ensure (B, L, H, D) layout with strides expected by TileLang kernels.

    During autoregressive decode (Lq=1), transpose+``.contiguous()`` can be a
    no-op because size-1 dims are ignored in PyTorch's contiguity check, leaving
    stride[1]==D instead of H*D and tripping the kernel stride assertion.
    """
    _, _, h, d = t.shape
    if t.stride(1) == h * d and t.stride(2) == d and t.stride(3) == 1:
        return t
    dst = torch.empty(t.shape, device=t.device, dtype=t.dtype)
    dst.copy_(t)
    return dst


class _FlexAttnTL(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q_blhd, k_blhd, v_blhd, window_size, chunk_size, mask_lmk, expand_to_chunk, use_cache, sm_scale=None, kv_start=0):
        """Inputs are already in (B, L, H, D) layout (transposed by wrapper).

        The kernels themselves handle non-multiple-of-block seq lengths via
        per-element guarded load / store (see `flex_attn_fwd` / `flex_attn_bwd`),
        so no caller-side padding is needed.

        """
        BATCH, Q_LEN, H, D_QK = q_blhd.shape
        KV_LEN = k_blhd.shape[1]
        H_KV = k_blhd.shape[2]
        D_V = v_blhd.shape[-1]
        groups = H // H_KV

        # block_M must be 128 (not 64): the forward attention output is only a
        # bf16 reduction-order variation between the two, but the smaller tile
        # perturbs every layer by ~1e-6 which compounds through 32 layers into a
        # ~3e-3 logit shift (e.g. GPQA 34.34 -> 32.83). 128 matches the original
        # kernel bit-for-bit.
        block_M = 128
        block_N = 64

        if use_cache:
            q_blhd = _as_standard_blhd(q_blhd)
            k_blhd = _as_standard_blhd(k_blhd)
            v_blhd = _as_standard_blhd(v_blhd)

        mod = flex_attn_fwd(
            BATCH, H, Q_LEN, KV_LEN, D_QK, D_V,
            window_size, chunk_size,
            mask_lmk, expand_to_chunk,
            block_M, block_N, groups,
            use_cache=use_cache,
            sm_scale=sm_scale,
        )
        kv_start_t = torch.tensor([int(kv_start)], device=q_blhd.device, dtype=torch.int32)
        o, lse = mod(q_blhd, k_blhd, v_blhd, kv_start_t)

        ctx.save_for_backward(q_blhd, k_blhd, v_blhd, o, lse)
        ctx.window_size = window_size
        ctx.chunk_size = chunk_size
        ctx.mask_lmk = mask_lmk
        ctx.expand_to_chunk = expand_to_chunk
        ctx.use_cache = use_cache
        ctx.sm_scale = sm_scale
        return o, lse

    @staticmethod
    def backward(ctx, do, dlse):
        q, k, v, o, lse = ctx.saved_tensors
        if ctx.use_cache:
            raise RuntimeError("flex_attn_tl backward is not supported in inference (use_cache=True) mode")

        BATCH, N_CTX, H, D_QK = q.shape
        H_KV = v.shape[-2]
        D_V = v.shape[-1]
        groups = H // H_KV

        # Force contiguous before passing tensors to TileLang kernels.
        do = do.contiguous()
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        o = o.contiguous()

        block_M = 64
        block_N = 64
        dq_len_padded = ((N_CTX + 7) // 8) * 8

        mod_prep = flex_attn_bwd_preprocess(BATCH, H, N_CTX, D_V)
        mod_post = flex_attn_bwd_postprocess(BATCH, H, N_CTX, dq_len_padded, D_QK)
        delta = mod_prep(o, do)

        # Fold dlse into Delta so the kernel's standard
        #   d(qk)_ij = P_ij * (dP_ij - Delta_i) * sm_scale_nat
        # also accounts for the gradient flowing through `lse`.
        # Derivation (natural-log domain):
        #   s_ij    = qk_ij * sm_scale_nat
        #   d lse_nat / d s_ij = P_ij
        #   => extra contribution to ds_ij is dlse_nat_i * P_ij,
        #      i.e. extra contribution to d(qk)_ij is
        #           P_ij * dlse_nat_i * sm_scale_nat
        #   => fold as Delta'_i = Delta_i - dlse_nat_i  (sm_scale cancels).
        # The autograd `dlse` argument is in log2 domain because the wrapper
        # returns lse = lse_log2 * ln2; chain rule gives
        #   dlse_arg = dlse_nat * ln2  =>  dlse_nat = dlse_arg / ln2.
        if dlse is not None:
            ln2 = 0.6931471805599453
            # dlse: (B, L, H) -> (B, H, L), match Delta layout/dtype.
            dlse_bhl = dlse.permute(0, 2, 1).contiguous().to(delta.dtype)
            delta = delta - dlse_bhl / ln2

        kernel = flex_attn_bwd(
            BATCH, H, N_CTX, D_QK, D_V,
            ctx.window_size, ctx.chunk_size,
            ctx.mask_lmk, ctx.expand_to_chunk,
            block_M, block_N,
            threads=128, num_stages=1, groups=groups,
            sm_scale=ctx.sm_scale,
        )
        dq = torch.zeros([BATCH, dq_len_padded, H, D_QK], dtype=torch.float32, device=q.device)
        dk = torch.zeros([groups, BATCH, N_CTX, H_KV, D_QK], dtype=torch.float32, device=q.device)
        dv = torch.zeros([groups, BATCH, N_CTX, H_KV, D_V], dtype=torch.float32, device=q.device)
        kernel(q, k, v, do, lse, delta, dq, dk, dv)
        dq = mod_post(dq)
        dk = dk.sum(dim=0).to(torch.bfloat16)
        dv = dv.sum(dim=0).to(torch.bfloat16)
        return dq, dk, dv, None, None, None, None, None, None, None


# ===========================================================================
# Two-phase backward kernels (atomic-free).
#
# Phase 1 (dQ):   grid parallelizes over Q-blocks; each thread block "owns" a
#                 Q-block and iterates the KV-blocks it can attend to. dQ is
#                 written directly to bf16 global memory (no atomics, no
#                 fp32 staging buffer, no post-cast kernel).
#
# Phase 2 (dKdV): grid parallelizes over KV-blocks (per KV head); each thread
#                 block "owns" a KV-block and iterates the Q-blocks that
#                 contribute to it. For GQA the kernel sums contributions from
#                 the `groups` Q-heads that share this KV head internally so
#                 dK/dV are also written directly to bf16 with no atomics.
#
# Mask semantics are identical to the atomic `flex_attn_bwd` kernel:
#   keep = (kv_real >= chunk_start) & (kv_real <= q_real) & bounds
#          [ & ((kv_real + 1) % chunk_size != 0) if mask_lmk ]
#   chunk_start = (q_real - W + 1) // C * C   if expand_to_chunk
#               = q_real - W + 1              otherwise
# ===========================================================================




# ---------------------------------------------------------------------------
# Two-phase autograd Function (atomic-free backward).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def flex_attn_tl(
    q,
    k,
    v,
    window_size: int,
    chunk_size: int,
    training: bool = True,
    mask_lmk: bool = True,
    expand_to_chunk: bool = True,
    sm_scale: float = None,
    kv_start: int = 0,
):
    """Tilelang flex attention.

    Args:
        q: (B, Lq, H_q, D_qk)
        k: (B, Lk, H_kv, D_qk)
        v: (B, Lk, H_kv, D_v)
        window_size:    sliding window size (in tokens)
        chunk_size:     chunk size for landmark / window-expansion semantics
        training:       True for training (Lq == Lk); False for inference (Lq <= Lk).
                        Backward only supported when training=True.
        mask_lmk:       if True, exclude landmark token columns where (kv_idx + 1) % chunk_size == 0
        expand_to_chunk:if True, sliding window left edge expands to start of containing chunk
        sm_scale:       optional softmax scale (natural-log domain), applied as
                        `score = sm_scale * <q, k>` before the softmax. When
                        None (default), uses `1 / sqrt(D_qk)` to match the
                        canonical attention scale.
        kv_start:       absolute position of k[:, 0]. Keep the default 0 for
                        full-prefix caches; pass a non-zero value when the KV
                        cache has been truncated by sliding-window generation.

    Returns:
        o:   (B, Lq, H_q, D_v)
        lse: (B, Lq, H_q) natural log
    """
    if training and int(kv_start) != 0:
        raise ValueError("flex_attn_tl only supports non-zero kv_start in inference (training=False)")
    use_cache = not training

    o_blhd, lse_log2 = _FlexAttnTL.apply(
        q, k, v,
        window_size, chunk_size,
        mask_lmk, expand_to_chunk, use_cache,
        sm_scale, kv_start,
    )
    # convert log2-domain lse to natural log
    ln2 = 0.6931471805599453  # ln(2)
    lse = lse_log2 * ln2
    return o_blhd, lse




# ---------------------------------------------------------------------------
# Reference (torch) implementation for testing
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Consistency test
# ---------------------------------------------------------------------------
import pytest


# Build a comprehensive list of (name, cfg_dict) test cases covering:
# - flag combinations (mask_lmk x expand_to_chunk)
# - GQA group counts (1, 2, 4, 8)
# - short / long sequences
# - window vs chunk size relations (W == S, W > S, W < S, W not multiple of S)
# - different head_dim
# - inference: q_len < kv_len, single-token decode, prefill boundary cases
# - edge: seq_len exactly fits block_M / not; window_size bigger than seq_len
_TEST_CASES = [
    # -------------------- training cases --------------------
    # flag combinations (baseline config)
    ("train_lmk_expand",     dict(training=True,  mask_lmk=True,  expand_to_chunk=True,  seq_len=512, window_size=128, chunk_size=64)),
    ("train_no_flags",       dict(training=True,  mask_lmk=False, expand_to_chunk=False, seq_len=512, window_size=128, chunk_size=64)),
    ("train_lmk_only",       dict(training=True,  mask_lmk=True,  expand_to_chunk=False, seq_len=512, window_size=128, chunk_size=64)),
    ("train_expand_only",    dict(training=True,  mask_lmk=False, expand_to_chunk=True,  seq_len=512, window_size=128, chunk_size=64)),

    # # GQA group variations
    # ("train_gqa_g1",         dict(training=True,  heads_q=4, heads_kv=4, seq_len=512)),
    # ("train_gqa_g2",         dict(training=True,  heads_q=8, heads_kv=4, seq_len=512)),
    # ("train_gqa_g4",         dict(training=True,  heads_q=8, heads_kv=2, seq_len=512)),
    # ("train_gqa_g8",         dict(training=True,  heads_q=8, heads_kv=1, seq_len=512)),

    # # batch variations
    # ("train_batch2",         dict(training=True,  batch=2, seq_len=512)),
    # ("train_batch4",         dict(training=True,  batch=4, seq_len=256)),

    # # short sequence
    # ("train_short_seq",      dict(training=True,  seq_len=256, window_size=64,  chunk_size=32)),
    # ("train_seq128",         dict(training=True,  seq_len=128, window_size=64,  chunk_size=32)),

    # # long sequence
    # ("train_long_seq",       dict(training=True,  seq_len=1024, window_size=256, chunk_size=64)),
    # ("train_long_seq_2048",  dict(training=True,  seq_len=2048, window_size=512, chunk_size=128)),

    # # window == chunk_size
    # ("train_w_eq_cs",        dict(training=True,  seq_len=512, window_size=64,  chunk_size=64)),
    # # window > chunk_size (multi-chunk window)
    # ("train_w_gt_cs",        dict(training=True,  seq_len=512, window_size=256, chunk_size=64)),
    # # window < chunk_size (window inside a chunk, only meaningful with expand_to_chunk)
    # ("train_w_lt_cs_expand", dict(training=True,  seq_len=512, window_size=32,  chunk_size=64, expand_to_chunk=True,  mask_lmk=True)),
    # ("train_w_lt_cs_noexp",  dict(training=True,  seq_len=512, window_size=32,  chunk_size=64, expand_to_chunk=False, mask_lmk=True)),
    # # window not a multiple of chunk_size
    # ("train_w_not_mult",     dict(training=True,  seq_len=512, window_size=100, chunk_size=64)),
    # # window >= seq_len (degenerate: full causal)
    # ("train_w_ge_seq",       dict(training=True,  seq_len=256, window_size=512, chunk_size=64)),

    # # head_dim variations
    # ("train_hd64",           dict(training=True,  seq_len=512, dim_qk=64,  dim_v=64)),
    # ("train_hd128",          dict(training=True,  seq_len=512, dim_qk=128, dim_v=128)),
    # # dim_qk != dim_v
    # ("train_hd_qk_ne_v",     dict(training=True,  seq_len=512, dim_qk=192, dim_v=128)),

    # # seq_len not a multiple of block_M (block_M=128)
    # ("train_seq384",         dict(training=True,  seq_len=384, window_size=128, chunk_size=64)),
    # ("train_seq640",         dict(training=True,  seq_len=640, window_size=128, chunk_size=64)),

    # -------------------- inference cases --------------------
    # q_len < kv_len, various prefill chunk sizes
    ("infer_prefill_half",   dict(training=False, seq_len=512,  q_len_inf=256,  window_size=128, chunk_size=64)),
    ("infer_prefill_small",  dict(training=False, seq_len=512,  q_len_inf=64,   window_size=128, chunk_size=64)),
    ("infer_prefill_big",    dict(training=False, seq_len=1024, q_len_inf=512,  window_size=256, chunk_size=64)),
    # single-token decode
    ("infer_decode_q1",      dict(training=False, seq_len=512,  q_len_inf=1,    window_size=128, chunk_size=64)),
    ("infer_decode_q1_long", dict(training=False, seq_len=2048, q_len_inf=1,    window_size=512, chunk_size=128)),
    # q_len == kv_len via inference path
    ("infer_full",           dict(training=False, seq_len=256,  q_len_inf=256,  window_size=128, chunk_size=64)),
    # inference with GQA
    ("infer_gqa_g4_q1",      dict(training=False, seq_len=512,  q_len_inf=1,    heads_q=8, heads_kv=2)),
    ("infer_gqa_g8_q1",      dict(training=False, seq_len=512,  q_len_inf=1,    heads_q=8, heads_kv=1)),
    # inference flag combinations
    ("infer_no_flags",       dict(training=False, seq_len=512,  q_len_inf=128,  mask_lmk=False, expand_to_chunk=False)),
    ("infer_lmk_only",       dict(training=False, seq_len=512,  q_len_inf=128,  mask_lmk=True,  expand_to_chunk=False)),
    ("infer_expand_only",    dict(training=False, seq_len=512,  q_len_inf=128,  mask_lmk=False, expand_to_chunk=True)),
    # inference q_len on block boundary / not
    ("infer_q_block_edge",   dict(training=False, seq_len=1024, q_len_inf=128,  window_size=256, chunk_size=64)),
    ("infer_q_non_aligned",  dict(training=False, seq_len=1024, q_len_inf=100,  window_size=256, chunk_size=64)),

]


# ---------------------------------------------------------------------------
# Chunk-size 65 cases: physical chunks are 64 payload tokens + 1 LMK token.
# Keep these before all legacy cases so this layout is validated first.
_CHUNK65_CASES = [
    ("train_chunk65_short", dict(
        training=True, seq_len=64, window_size=512, chunk_size=65,
        mask_lmk=True, expand_to_chunk=True,
    )),
    ("train_chunk65_long", dict(
        training=True, seq_len=2080, window_size=512, chunk_size=65,
        mask_lmk=True, expand_to_chunk=True,
    )),
    ("train_chunk65_nonmultiple", dict(
        training=True, seq_len=1000, window_size=512, chunk_size=65,
        mask_lmk=True, expand_to_chunk=True,
    )),
    ("infer_chunk65_short", dict(
        training=False, seq_len=64, q_len_inf=32, window_size=512, chunk_size=65,
        mask_lmk=True, expand_to_chunk=True,
    )),
    ("infer_chunk65_long", dict(
        training=False, seq_len=2080, q_len_inf=512, window_size=512, chunk_size=65,
        mask_lmk=True, expand_to_chunk=True,
    )),
    ("infer_chunk65_nonmultiple", dict(
        training=False, seq_len=1000, q_len_inf=333, window_size=512, chunk_size=65,
        mask_lmk=True, expand_to_chunk=True,
    )),
]


# ---------------------------------------------------------------------------
# Regression cases: seq_len / q_len not aligned to block_N (=32 for bwd, =64 for fwd).
# The dynamic-batching production path produces non-multiple-of-block_N lengths;
# previously the bwd kernel read out-of-bounds Q/lse/dO/Delta on the tail block,
# which combined with IEEE 754 (0 * NaN = NaN) silently corrupted dq/dk/dv.
# The training-loop NaN was reproducible only when seq_len % block_N != 0.
# Keep these listed first so we test the most fragile cases up front.
_REGRESSION_CASES = [
("infer_decode_q1_noexp_lmk",      dict(training=False, seq_len=1024, q_len_inf=1, window_size=512, chunk_size=64, mask_lmk=True, expand_to_chunk=False)),
("infer_decode_q1_noexp_lmk_k333", dict(training=False, seq_len=333,  q_len_inf=1, window_size=512, chunk_size=64, mask_lmk=True, expand_to_chunk=False)),
("infer_decode_q2_noexp_lmk",      dict(training=False, seq_len=1024, q_len_inf=2, window_size=512, chunk_size=64, mask_lmk=True, expand_to_chunk=False)),


    ("train_L32_lt_chunk_lmk_expand", dict(
        training=True, seq_len=16, window_size=512, chunk_size=64,
        heads_q=16, heads_kv=2, dim_qk=64, dim_v=64,
        mask_lmk=True, expand_to_chunk=True,
    )),

    # ---- training: seq_len not a multiple of block_N (bwd block_N=32) ----
    # ("train_seq130_misaligned",   dict(training=True, seq_len=130, window_size=64,  chunk_size=32)),
    # ("train_seq200_misaligned",   dict(training=True, seq_len=200, window_size=64,  chunk_size=32)),
    # ("train_seq250_misaligned",   dict(training=True, seq_len=250, window_size=128, chunk_size=64)),
    # ("train_seq333_misaligned",   dict(training=True, seq_len=333, window_size=128, chunk_size=64)),
    # ("train_seq513_misaligned",   dict(training=True, seq_len=513, window_size=128, chunk_size=64)),
    # ("train_seq1000_misaligned",  dict(training=True, seq_len=1000, window_size=256, chunk_size=64)),
    # # repro the production setting (mask_lmk=True, expand_to_chunk=False) at
    # # an awkward length close to the failing run (max_seq_len=8192, dyn_bsz tail).
    # ("train_seq2049_lmk_no_exp",  dict(training=True, seq_len=2049, window_size=512, chunk_size=64,
    #                                     mask_lmk=True, expand_to_chunk=False)),
    # ("train_seq4097_lmk_no_exp",  dict(training=True, seq_len=4097, window_size=512, chunk_size=64,
    #                                     mask_lmk=True, expand_to_chunk=False)),
    # # GQA + misaligned (matches production: heads_q=16, heads_kv=4 → groups=4)
    # ("train_misaligned_gqa",      dict(training=True, seq_len=333, heads_q=16, heads_kv=4,
    #                                     window_size=128, chunk_size=64,
    #                                     mask_lmk=True, expand_to_chunk=False)),
    # # head_dim=64
    # ("train_misaligned_hd64",     dict(training=True, seq_len=333, dim_qk=64, dim_v=64,
    #                                     window_size=128, chunk_size=64,
    #                                     mask_lmk=True, expand_to_chunk=False)),

    # ---- inference: q_len / kv_len not aligned to block_N (fwd block_N=64) ----
    ("infer_q_misaligned_70",     dict(training=False, seq_len=512, q_len_inf=70,
                                        window_size=128, chunk_size=64)),
    ("infer_q_misaligned_130",    dict(training=False, seq_len=512, q_len_inf=130,
                                        window_size=128, chunk_size=64)),
    ("infer_q_misaligned_333",    dict(training=False, seq_len=1024, q_len_inf=333,
                                        window_size=256, chunk_size=64)),
    ("infer_kv_misaligned",       dict(training=False, seq_len=1023, q_len_inf=100,
                                        window_size=256, chunk_size=64)),
    ("infer_both_misaligned",     dict(training=False, seq_len=777, q_len_inf=55,
                                        window_size=128, chunk_size=64,
                                        mask_lmk=True, expand_to_chunk=False)),
    # decode (q_len=1) at a non-aligned kv_len
    ("infer_decode_q1_kv1023",    dict(training=False, seq_len=1023, q_len_inf=1,
                                        window_size=256, chunk_size=64)),
    ("infer_decode_q1_kv333",     dict(training=False, seq_len=333, q_len_inf=1,
                                        window_size=128, chunk_size=64)),

]


# Run chunk-size-65 and regression cases first so newly-introduced bugs surface early.
_REGRESSION_CASES = _CHUNK65_CASES + _REGRESSION_CASES
_TEST_CASES = _REGRESSION_CASES + _TEST_CASES






# ---------------------------------------------------------------------------
# Decode(q=1) vs prefill(q=full) self-consistency test.
#
# This mirrors the production "generate (decode, KV-cache, q=1) vs forward
# (prefill teacher-forcing, q=full)" comparison ENTIRELY at the kernel level.
#
# Key property: both paths use the SAME q/k/v tensors -- the decode path only
# *slices* the shared cache (k[:, :p+1]) instead of recomputing it. Therefore
# any mismatch isolates the flex_attn_tl kernel's own decode-vs-prefill
# behaviour, ruling out bf16-cache rounding, rope, landmark bookkeeping and all
# other modeling-glue differences.
#
# Note: the existing flex-vs-torch_ref tests CANNOT catch this, because the
# torch reference (`_torch_ref_flex_attn`) infers the absolute query index with
# the same `q_offset = L_k - L_q` convention as the kernel, so a q=1 vs q=full
# divergence would be invisible (both "agree" with each other at q=1).
# ---------------------------------------------------------------------------


# Decode-vs-prefill self-consistency cases. The OLMo sliding-layer production
# config is (window=512, chunk=64, mask_lmk=True, expand_to_chunk=False) -- the
# exact combination that ONLY Olmo3Attention.forward hits and that Qwen never
# uses. The mask_lmk=False variant is a control: if it PASSES while mask_lmk
# FAILS, the landmark masking is conclusively the differentiator.
_DECODE_VS_PREFILL_CASES = [
    # chunk_size=65 self-consistency cases, kept first for priority coverage
    ("decvspre_chunk65_short",      dict(seq_len=64, window_size=512, chunk_size=65,
                                         heads_q=8, heads_kv=2, mask_lmk=True, expand_to_chunk=True)),
    ("decvspre_chunk65_long",       dict(seq_len=2080, window_size=512, chunk_size=65,
                                         heads_q=8, heads_kv=2, mask_lmk=True, expand_to_chunk=True)),
    ("decvspre_chunk65_nonmultiple", dict(seq_len=1000, window_size=512, chunk_size=65,
                                          heads_q=8, heads_kv=2, mask_lmk=True, expand_to_chunk=True)),
    # production OLMo sliding-layer config (GQA), aligned kv
    ("decvspre_olmo_lmk",        dict(seq_len=1024, window_size=512, chunk_size=64,
                                      heads_q=8, heads_kv=2, mask_lmk=True,  expand_to_chunk=False)),
    # MHA variant (OLMo uses h_kv == h_q in some configs)
    ("decvspre_olmo_lmk_mha",    dict(seq_len=1024, window_size=512, chunk_size=64,
                                      heads_q=8, heads_kv=8, mask_lmk=True,  expand_to_chunk=False)),
    # CONTROL: same config but no landmark masking -> expected to PASS
    ("decvspre_olmo_nolmk",      dict(seq_len=1024, window_size=512, chunk_size=64,
                                      heads_q=8, heads_kv=2, mask_lmk=False, expand_to_chunk=False)),
    # misaligned kv length (not a multiple of chunk / block)
    ("decvspre_olmo_lmk_k1000",  dict(seq_len=1000, window_size=512, chunk_size=64,
                                      heads_q=8, heads_kv=2, mask_lmk=True,  expand_to_chunk=False)),
    # production generate path: KV cache is truncated, so k[:, 0] has kv_start > 0
    ("decvspre_olmo_lmk_trunc",  dict(seq_len=1024, window_size=512, chunk_size=64,
                                      heads_q=8, heads_kv=2, mask_lmk=True,  expand_to_chunk=False,
                                      truncate_kv=True)),
]


# ---------------------------------------------------------------------------
# Two-phase backward consistency test (training only).
#
# Strategy: run both `flex_attn_tl` (atomic bwd) and `flex_attn_tl_two_phase`
# (two-phase bwd) on identical inputs, then compare:
#   1. fwd outputs (o, lse): MUST be bitwise identical (same fwd kernel).
#   2. dq / dk / dv vs the torch reference: same tolerance as the atomic path.
#   3. dq / dk / dv vs the atomic path: tighter tolerance (both bf16, same fwd
#      lse / Delta inputs, only the bwd kernel differs).
# ---------------------------------------------------------------------------

# Only training cases are valid for two-phase (bwd not supported in inference).
_TEST_CASES_TWO_PHASE = [(name, cfg) for (name, cfg) in _TEST_CASES if cfg.get("training", True)]






# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


_BENCH_CASES = [
    # ---- training: typical training shapes ----
    # ("train_2k_w512",   dict(training=True,  seq_len=2048, window_size=512,  chunk_size=64)),
    ("train_8k_w512",   dict(training=True,  seq_len=8192, window_size=512,  chunk_size=64)),
    # ("train_8k_w1024",  dict(training=True,  seq_len=8192, window_size=1024, chunk_size=128)),
    # # ---- inference: prefill ----
    # ("infer_prefill_4k_q1k", dict(training=False, seq_len=4096, q_len_inf=1024, window_size=512, chunk_size=64)),
    # # ---- inference: single token decode ----
    # ("infer_decode_4k_q1",   dict(training=False, seq_len=4096, q_len_inf=1,    window_size=512, chunk_size=64)),
    # ("infer_decode_8k_q1",   dict(training=False, seq_len=8192, q_len_inf=1,    window_size=1024, chunk_size=128)),
]


