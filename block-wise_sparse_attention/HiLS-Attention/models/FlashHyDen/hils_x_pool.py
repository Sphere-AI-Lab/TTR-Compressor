"""HiLS prior-query pooling with a shared, pre-Wk x summary."""

import torch
from torch import nn


class HiLSXPool(nn.Module):
    def forward(self, x, query, key):
        """x: [B,N,S,Dm], query/key: [B,N,S,H,D]."""
        if x.shape[2] < 2:
            raise ValueError("HiLS pooling requires at least two tokens per block")
        groups = query.shape[-2] // key.shape[-2]
        if query.shape[-2] % key.shape[-2]:
            raise ValueError("query heads must be divisible by KV heads")
        keys = key.float().repeat_interleave(groups, dim=-2)
        logits = torch.einsum("bnhd,bnshd->bnhs", query[:, :, -1].float(), keys)
        logits = logits * (key.shape[-1] ** -0.5)
        # Match HiLS: the last token supplies the prior query, not a pooled value.
        log_prob = logits[..., :-1].log_softmax(dim=-1)
        prob = log_prob.exp()
        entropy = -(prob * log_prob).sum(dim=-1)
        weights = prob.mean(dim=2)
        summary = torch.einsum("bns,bnsd->bnd", weights, x[:, :, :-1].float())
        return summary.to(x.dtype), entropy
