"""
Rotary Position Embedding
================================================
TileLang implementation of RoPE.

Author: Xinyu Wei
"""

import torch
import tilelang
from tilelang import language as T






@tilelang.jit(
    out_idx=[3],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    })
def single_tensor_rope_kernel(batch, seq_len, heads, dim, cos_sin_dtype="bfloat16", threads=64):
    x_dtype = "bfloat16"
    compute_dtype = "float32"

    assert dim % 2 == 0
    dim_half = dim // 2

    @T.prim_func
    def main(
        X: T.Buffer((batch, seq_len, heads, dim), x_dtype),
        X_cos: T.Buffer((seq_len, dim), cos_sin_dtype),
        X_sin: T.Buffer((seq_len, dim), cos_sin_dtype),
        X_out: T.Buffer((batch, seq_len, heads, dim), cos_sin_dtype),
    ):
        total_tokens = batch * seq_len
        with T.Kernel(total_tokens, threads=threads) as bx:
            b_idx = bx // seq_len
            l_idx = bx % seq_len

            cos_lo = T.alloc_fragment((dim_half,), compute_dtype)
            sin_lo = T.alloc_fragment((dim_half,), compute_dtype)
            cos_hi = T.alloc_fragment((dim_half,), compute_dtype)
            sin_hi = T.alloc_fragment((dim_half,), compute_dtype)

            for d in T.Parallel(dim_half):
                cos_lo[d] = T.cast(X_cos[l_idx, d], compute_dtype)
                sin_lo[d] = T.cast(X_sin[l_idx, d], compute_dtype)
                cos_hi[d] = T.cast(X_cos[l_idx, d + dim_half], compute_dtype)
                sin_hi[d] = T.cast(X_sin[l_idx, d + dim_half], compute_dtype)

            for h in T.unroll(heads):
                for d in T.Parallel(dim_half):
                    x1 = T.cast(X[b_idx, l_idx, h, d], compute_dtype)
                    x2 = T.cast(X[b_idx, l_idx, h, d + dim_half], compute_dtype)

                    o1 = x1 * cos_lo[d] - x2 * sin_lo[d]
                    o2 = x2 * cos_hi[d] + x1 * sin_hi[d]

                    X_out[b_idx, l_idx, h, d] = T.cast(o1, cos_sin_dtype)
                    X_out[b_idx, l_idx, h, d + dim_half] = T.cast(o2, cos_sin_dtype)

    return main


def _single_tensor_rope(x, x_cos, x_sin):
    B, L, H, D = x.shape
    if x_cos.ndim == 3:
        assert x_cos.shape[0] == 1 and x_sin.shape[0] == 1, \
            f"Expected batch dim of x_cos/x_sin to be 1 when ndim==3 (shared across batch), but got x_cos.shape={tuple(x_cos.shape)}, x_sin.shape={tuple(x_sin.shape)}"
        x_cos = x_cos[0]
        x_sin = x_sin[0]
    x_cos = x_cos.contiguous()
    x_sin = x_sin.contiguous()
    if x_cos.dtype == torch.float32:
        cos_sin_dtype = "float32"
    else:
        cos_sin_dtype = "bfloat16"
    kernel_func = single_tensor_rope_kernel(B, L, H, D, cos_sin_dtype)
    return kernel_func(x, x_cos, x_sin)


class SingleTensorRoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, x_cos, x_sin):
        x_dtype = x.dtype
        x_out = _single_tensor_rope(x, x_cos, x_sin)
        ctx.save_for_backward(x_cos, x_sin)
        ctx.x_dtype = x_dtype
        return x_out.to(x_dtype)

    @staticmethod
    def backward(ctx, dx):
        x_cos, x_sin = ctx.saved_tensors
        dx_out = _single_tensor_rope(dx.contiguous(), x_cos, -x_sin)
        return dx_out.to(ctx.x_dtype), None, None


def single_tensor_rope_autograd(x, x_cos, x_sin):
    """
    Args:
        x:     (B, L, H, D)  - BF16
        x_cos: (B, L, D) - BF16 or FP32
        x_sin: (B, L, D) - BF16 or FP32
    Returns:
        x_out: (B, L, H, D) - same dtype as x (BF16)
    """
    return SingleTensorRoPEFunction.apply(x, x_cos, x_sin)








import torch.nn.functional as F
from liger_kernel.transformers.rope import liger_rotary_pos_emb







