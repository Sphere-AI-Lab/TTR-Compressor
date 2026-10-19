"""Block-wise sparse attention with HiLS pooling or TTR compression."""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from ops.flex_attn_tilelang import flex_attn_tl
from ops.rope_tilelang_fp32 import single_tensor_rope_autograd
from .compressor import build_compressor
from .hils_x_pool import HiLSXPool

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim: int=1):
    (q_dtype, k_dtype) = (q.dtype, k.dtype)
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return (q_embed.to(q_dtype), k_embed.to(k_dtype))

class HyDenAttention(nn.Module):
    """Compress each block of real tokens before Wk; retrieve with calibrated Q."""

    def __init__(self, config, layer_idx: int, norm_cls=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.d_model = config.hidden_size
        self.head_dim = self.d_model // config.num_attention_heads
        self.h_kv = config.num_key_value_heads
        self.h_q = config.num_attention_heads
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.h_kv * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.h_kv * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.enable_lmk_q_proj = getattr(config, 'enable_lmk_q_proj', False)
        if self.enable_lmk_q_proj:
            self.lmk_q_lora_dim = getattr(config, 'lmk_q_lora_dim', -1)
            if self.lmk_q_lora_dim <= 0:
                self.lmk_q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
            else:
                self.lmk_q_proj = nn.Sequential(nn.Linear(self.d_model, self.lmk_q_lora_dim, bias=False), nn.Linear(self.lmk_q_lora_dim, self.d_model, bias=False))
            self.layerwise_lmkq_norm = getattr(config, 'layerwise_lmkq_norm', False)
            self.lmk_q_norm = norm_cls(self.d_model if self.layerwise_lmkq_norm else self.head_dim)
        self.layerwise_qk_norm = getattr(config, 'layerwise_qk_norm', False)
        self.q_norm = norm_cls(self.d_model if self.layerwise_qk_norm else self.head_dim)
        self.k_norm = norm_cls(self.h_kv * self.head_dim if self.layerwise_qk_norm else self.head_dim)
        self.compressor_x_apply_rope = getattr(config, 'compressor_x_apply_rope', True)
        self.apply_hyden_rope = getattr(config, 'apply_hyden_rope', getattr(config, 'apply_hils_rope', False))
        self.chunk_size = config.chunk_size
        self.topk = config.hils_topk
        self.sliding_window = config.sliding_window
        self.hyden_sliding_window = getattr(config, 'hyden_sliding_window', getattr(config, 'hils_sliding_window', self.sliding_window))
        self.enable_softmax1 = config.enable_softmax1
        self.scaling = self.head_dim ** (-0.5)
        self.is_causal = True
        (compressor_heads, compressor_dim) = (1, self.d_model)
        self.hils_x = config.compressor_type == 'hils'
        self.compressor = HiLSXPool() if self.hils_x else build_compressor(config, num_heads=compressor_heads, head_dim=compressor_dim)
        from ops.hils_fwd_bwd_head import HiLS_block_M_head
        from ops.topk_head_softmax import online_softmax_topk_head
        self.sparse_attention_func = HiLS_block_M_head
        self.topk_func = online_softmax_topk_head

    def _project_qkv(self, hidden_states):
        q_projected = self.q_proj(hidden_states)
        routing_q = None
        if self.enable_lmk_q_proj:
            routing_q = self.lmk_q_proj(hidden_states)
            if self.lmk_q_lora_dim > 0:
                routing_q = routing_q + q_projected
            if self.layerwise_lmkq_norm:
                routing_q = self.lmk_q_norm(routing_q)
                routing_q = rearrange(routing_q, 'B L (h d) -> B L h d', h=self.h_q)
            else:
                routing_q = rearrange(routing_q, 'B L (h d) -> B L h d', h=self.h_q)
                routing_q = self.lmk_q_norm(routing_q)
        if self.layerwise_qk_norm:
            q_projected = self.q_norm(q_projected)
        q_projected = rearrange(q_projected, 'B L (h d) -> B L h d', d=self.head_dim)
        q_nope = self.q_norm(q_projected) if not self.layerwise_qk_norm else q_projected
        if routing_q is None:
            routing_q = q_nope
        k_projected = self.k_proj(hidden_states)
        if self.layerwise_qk_norm:
            k_projected = self.k_norm(k_projected)
        k_projected = rearrange(k_projected, 'B L (h d) -> B L h d', d=self.head_dim)
        k_nope = self.k_norm(k_projected) if not self.layerwise_qk_norm else k_projected
        value = rearrange(self.v_proj(hidden_states), 'B L (h d) -> B L h d', d=self.head_dim)
        return (q_nope, k_nope, value, routing_q)

    def _apply_rope(self, q_nope, k_nope, routing_q, cos, sin):
        if not self.apply_hyden_rope:
            return (q_nope, k_nope, routing_q)
        (q_rope, k_rope) = apply_rotary_pos_emb(q_nope.transpose(1, 2), k_nope.transpose(1, 2), cos, sin)
        q_rope = q_rope.transpose(1, 2).contiguous()
        k_rope = k_rope.transpose(1, 2).contiguous()
        if self.enable_lmk_q_proj:
            routing_q = single_tensor_rope_autograd(routing_q, cos, sin)
        else:
            routing_q = q_rope
        return (q_rope, k_rope, routing_q)

    def _compress_full_x_chunks(self, hidden_states: torch.Tensor, chunk_count: int) -> torch.Tensor:
        """Compress raw attention inputs before their shared K projection."""
        if chunk_count <= 0:
            return hidden_states.new_empty(hidden_states.shape[0], 0, self.d_model)
        chunked = rearrange(hidden_states[:, :chunk_count * self.chunk_size], 'B (N S) D -> B N S 1 D', S=self.chunk_size)
        return self.compressor(chunked).squeeze(-2)

    def _project_x_summaries(self, x_summaries: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply the shared Wk, K RMSNorm, and optional chunk-end RoPE."""
        if x_summaries.shape[1] == 0:
            return x_summaries.new_empty(x_summaries.shape[0], 0, self.h_kv, self.head_dim)
        summary_keys = self.k_proj(x_summaries)
        if self.layerwise_qk_norm:
            summary_keys = self.k_norm(summary_keys)
        summary_keys = rearrange(summary_keys, 'B N (h d) -> B N h d', d=self.head_dim)
        if not self.layerwise_qk_norm:
            summary_keys = self.k_norm(summary_keys)
        if self.apply_hyden_rope and self.compressor_x_apply_rope:
            summary_keys = single_tensor_rope_autograd(summary_keys.contiguous(), cos.contiguous(), sin.contiguous())
        return summary_keys

    def _compute_x_chunk_summaries(self, current_hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, is_continuation: bool, use_cache: bool, past_key_values):
        """Compress x, then map x_c through Wk to obtain routing keys k_c.

        Cached decoding retains at most ``chunk_size - 1`` uncompressed x
        vectors. Completed summaries are cached only after Wk/RMSNorm/RoPE.
        """
        batch_size = current_hidden_states.shape[0]
        empty_summaries = current_hidden_states.new_empty(batch_size, 0, self.h_kv, self.head_dim)
        if use_cache and past_key_values is not None:
            if not hasattr(past_key_values, '_hyden_summary_cache'):
                past_key_values._hyden_summary_cache = {}
            if not hasattr(past_key_values, '_hyden_x_summary_pending_cache'):
                past_key_values._hyden_x_summary_pending_cache = {}
            summary_cache = past_key_values._hyden_summary_cache
            pending_cache = past_key_values._hyden_x_summary_pending_cache
            if is_continuation:
                if self.layer_idx not in summary_cache or self.layer_idx not in pending_cache:
                    raise ValueError('HyDen x-compression summary metadata is missing during cached decoding. Use the original in-memory past_key_values returned by the model.')
                cached_summaries = summary_cache[self.layer_idx]
                pending = pending_cache[self.layer_idx]
                pending_length = pending.shape[1]
                inputs = torch.cat((pending, current_hidden_states), dim=1)
            else:
                cached_summaries = empty_summaries
                pending_length = 0
                inputs = current_hidden_states
        else:
            cached_summaries = empty_summaries
            pending_length = 0
            inputs = current_hidden_states
        new_chunk_count = inputs.shape[1] // self.chunk_size
        x_summaries = self._compress_full_x_chunks(inputs, new_chunk_count)
        if new_chunk_count > 0:
            if self.apply_hyden_rope and self.compressor_x_apply_rope:
                first_endpoint = self.chunk_size - pending_length - 1
                endpoint_indices = torch.arange(first_endpoint, first_endpoint + new_chunk_count * self.chunk_size, self.chunk_size, device=cos.device)
                last_endpoint = first_endpoint + (new_chunk_count - 1) * self.chunk_size
                if last_endpoint >= current_hidden_states.shape[1]:
                    raise RuntimeError('x-compression chunk endpoint exceeds current position embeddings')
                summary_cos = cos.index_select(-2, endpoint_indices)
                summary_sin = sin.index_select(-2, endpoint_indices)
            else:
                (summary_cos, summary_sin) = (cos, sin)
            new_summaries = self._project_x_summaries(x_summaries, summary_cos, summary_sin)
        else:
            new_summaries = empty_summaries
        if cached_summaries.shape[1] == 0:
            summaries = new_summaries
        elif new_summaries.shape[1] == 0:
            summaries = cached_summaries
        else:
            summaries = torch.cat((cached_summaries, new_summaries), dim=1)
        if use_cache and past_key_values is not None:
            consumed = new_chunk_count * self.chunk_size
            summary_cache[self.layer_idx] = summaries
            pending_cache[self.layer_idx] = inputs[:, consumed:].contiguous()
        return summaries

    def _compute_hils_x_summaries(self, x, query, key, cos, sin, is_continuation, use_cache, past_key_values):
        """Cache completed keys/entropy and only the unfinished block's inputs."""
        cached = None
        cache = None
        if use_cache and past_key_values is not None:
            if not hasattr(past_key_values, '_hyden_hils_x_cache'):
                past_key_values._hyden_hils_x_cache = {}
            cache = past_key_values._hyden_hils_x_cache
            if is_continuation:
                cached = cache.get(self.layer_idx)
                if cached is None:
                    raise ValueError('Missing HiLS-x cache during continuation')
        pending_length = 0
        if cached is not None:
            (old_keys, old_bias, px, pq, pk) = cached
            pending_length = px.shape[1]
            x = torch.cat((px, x), dim=1)
            query = torch.cat((pq, query), dim=1)
            key = torch.cat((pk, key), dim=1)
        count = x.shape[1] // self.chunk_size
        consumed = count * self.chunk_size
        if count:
            b = x.shape[0]
            (xc, bias) = self.compressor(x[:, :consumed].reshape(b, count, self.chunk_size, self.d_model), query[:, :consumed].reshape(b, count, self.chunk_size, self.h_q, self.head_dim), key[:, :consumed].reshape(b, count, self.chunk_size, self.h_kv, self.head_dim))
            if self.apply_hyden_rope and self.compressor_x_apply_rope:
                endpoints = torch.arange(self.chunk_size - pending_length - 1, consumed - pending_length, self.chunk_size, device=cos.device)
                (cos, sin) = (cos.index_select(-2, endpoints), sin.index_select(-2, endpoints))
            summaries = self._project_x_summaries(xc, cos, sin)
        else:
            summaries = x.new_empty(x.shape[0], 0, self.h_kv, self.head_dim)
            bias = x.new_empty(x.shape[0], 0, self.h_q, dtype=torch.float32)
        if cached is not None:
            summaries = torch.cat((old_keys, summaries), dim=1)
            bias = torch.cat((old_bias, bias), dim=1)
        if cache is not None:
            cache[self.layer_idx] = (summaries, bias, x[:, consumed:].contiguous(), query[:, consumed:].contiguous(), key[:, consumed:].contiguous())
        return (summaries, bias)

    def forward(self, hidden_states: torch.Tensor, attention_mask=None, position_ids=None, past_key_values=None, output_attentions=None, use_cache: bool=False, cache_position=None, position_embeddings=None, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        del attention_mask, output_attentions
        (batch_size, query_length, _) = hidden_states.shape
        (cos, sin) = position_embeddings
        q_nope, k_nope, value, routing_q = self._project_qkv(hidden_states)
        cos = cos.to(dtype=q_nope.dtype, device=q_nope.device)
        sin = sin.to(dtype=q_nope.dtype, device=q_nope.device)
        q_rope, k_rope, routing_q = self._apply_rope(q_nope, k_nope, routing_q, cos, sin)
        key_for_cache = k_rope if self.apply_hyden_rope else k_nope
        current_keys_for_hils_x = key_for_cache
        if use_cache and past_key_values is not None:
            cache_idx = self.layer_idx + self.config.num_hidden_layers
            (key_for_cache, value) = past_key_values.update(key_for_cache, value, cache_idx, {'cache_position': cache_position})
            if self.apply_hyden_rope:
                k_rope = key_for_cache
            else:
                k_nope = key_for_cache
                k_rope = k_nope
        if kwargs.get('cu_seq_lens_q') is not None:
            raise AssertionError('HyDen headwise Top-K does not support cu_seq_lens_q')
        (local_output, local_lse) = flex_attn_tl(q_rope, k_rope, value, window_size=self.hyden_sliding_window, chunk_size=self.chunk_size, training=past_key_values is None, mask_lmk=False, expand_to_chunk=True)
        local_lse = local_lse.contiguous().to(hidden_states.dtype)
        full_seq_len = key_for_cache.shape[1]
        is_continuation = use_cache and past_key_values is not None and (query_length < full_seq_len)
        summaries = None
        prior_bias = None
        if self.hils_x:
            (summaries, prior_bias) = self._compute_hils_x_summaries(hidden_states, routing_q, current_keys_for_hils_x, cos, sin, is_continuation, use_cache, past_key_values)
        else:
            x_summary_result = self._compute_x_chunk_summaries(hidden_states, cos, sin, is_continuation=is_continuation, use_cache=use_cache, past_key_values=past_key_values)
            summaries = x_summary_result
        if summaries is not None and summaries.shape[1] > 0:
            query_offset = full_seq_len - query_length if use_cache and past_key_values is not None else 0
            (indices, scores) = self.topk_func(routing_q, summaries, local_lse, self.topk, block_size=self.chunk_size, window_size=self.hyden_sliding_window, is_training=self.training, is_causal=True, q_offset=query_offset, G=1, bias=prior_bias)
            if prior_bias is not None:
                head_indices = indices.repeat_interleave(self.h_q // indices.shape[2], dim=2)
                source = prior_bias.unsqueeze(-1).expand(-1, -1, -1, indices.shape[-1])
                selected_bias = torch.gather(source, 1, head_indices.clamp_min(0).long())
                scores = scores + selected_bias.to(scores.dtype)
            global_score = None
            expanded_global_value = None
            if self.enable_softmax1:
                extra_scores = (global_score.unsqueeze(-1),) if global_score is not None else ()
                mixture_scores = torch.cat((scores, *extra_scores, local_lse.unsqueeze(-1), scores.new_zeros(batch_size, query_length, scores.shape[2], 1)), dim=-1)
                local_weight_idx = -2
                global_weight_idx = -3
            else:
                extra_scores = (global_score.unsqueeze(-1),) if global_score is not None else ()
                mixture_scores = torch.cat((scores, *extra_scores, local_lse.unsqueeze(-1)), dim=-1)
                local_weight_idx = -1
                global_weight_idx = -2
            chunk_weights = F.softmax(mixture_scores, dim=-1).to(hidden_states.dtype)
            sparse_q = q_rope if self.apply_hyden_rope else q_nope
            sparse_k = k_rope if self.apply_hyden_rope else k_nope
            sparse_output = self.sparse_attention_func(sparse_q, sparse_k, value, weights=chunk_weights, indices=indices, block_size=self.chunk_size, mask_last_token=False, is_training=self.training)
            local_weight = chunk_weights[:, :, :, local_weight_idx]
            output = torch.addcmul(sparse_output, local_output, local_weight.unsqueeze(-1))
            if expanded_global_value is not None:
                global_weight = chunk_weights[:, :, :, global_weight_idx]
                output = torch.addcmul(output, expanded_global_value, global_weight.unsqueeze(-1))
        elif self.enable_softmax1:
            local_weight = torch.sigmoid(local_lse).to(local_output.dtype)
            output = local_output * local_weight.unsqueeze(-1)
        else:
            output = local_output
        output = rearrange(output, 'B L H D -> B L (H D)')
        return (self.o_proj(output), None)
__all__ = ['HyDenAttention']
