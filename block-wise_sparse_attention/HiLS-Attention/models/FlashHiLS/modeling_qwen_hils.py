from typing import Any, Callable, Optional, Tuple, Union
from dataclasses import dataclass, field

from .HoPE import HoPERotaryEmbedding
import torch
import math
from torch import nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.processing_utils import Unpack
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    can_return_tuple,
    is_torch_flex_attn_available,
    replace_return_docstrings,
)

from functools import partial
from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import slice_position_embedding
from veomni.utils import logging
from veomni.utils.import_utils import (
    is_liger_kernel_available,
    is_torch_npu_available,
    is_transformers_version_greater_or_equal_to,
)
from veomni.models.module_utils import GradientCheckpointingLayer
from ops.flex_attn_tilelang import flex_attn_tl


if is_torch_flex_attn_available():
    pass


if is_liger_kernel_available():
    from liger_kernel.transformers.rope import liger_rotary_pos_emb
    from liger_kernel.transformers.swiglu import LigerSwiGLUMLP


rms_norm_fn = None

try:
    from flash_attn.ops.triton.layer_norm import rms_norm_fn
    USE_FLASH_ATTN_RMSNORM = True
except ImportError:
    USE_FLASH_ATTN_RMSNORM = False


logger = logging.get_logger(__name__)




def rms_norm(hidden_states, weight, variance_epsilon):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)

    return weight * hidden_states.to(input_dtype)




def next_of_y(x, y):
    return (x + y - 1) // y * y


class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return rms_norm(hidden_states, self.weight, self.variance_epsilon)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
    
class FlashAttnRMSNorm(Qwen3RMSNorm):
    def forward(self, hidden_states):
        global rms_norm_fn
        if rms_norm_fn is None:
            try:
                from flash_attn.ops.triton.layer_norm import rms_norm_fn as flash_attn_rms_norm_fn
            except ImportError as exc:
                raise ImportError(
                    "flash_attn.ops.triton.layer_norm.rms_norm_fn is required for Olmo3FlashAttnRMSNorm."
                ) from exc
            rms_norm_fn = flash_attn_rms_norm_fn

        hidden_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_shape[-1])
        hidden_states = rms_norm_fn(
            hidden_states,
            self.weight,
            None,
            eps=self.variance_epsilon,
        )
        return hidden_states.reshape(hidden_shape)

class Qwen3FlashAttnRMSNorm(Qwen3RMSNorm):
    def forward(self, hidden_states):
        global rms_norm_fn
        if rms_norm_fn is None:
            try:
                from flash_attn.ops.triton.layer_norm import rms_norm_fn as flash_attn_rms_norm_fn
            except ImportError as exc:
                raise ImportError(
                    "flash_attn.ops.triton.layer_norm.rms_norm_fn is required for Qwen3FlashAttnRMSNorm."
                ) from exc
            rms_norm_fn = flash_attn_rms_norm_fn

        hidden_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_shape[-1])
        hidden_states = rms_norm_fn(
            hidden_states,
            self.weight,
            None,
            eps=self.variance_epsilon,
        )
        return hidden_states.reshape(hidden_shape)


class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x): 
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class Qwen3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.mask_lmk_token = getattr(config, "mask_lmk_token", False)
        if self.mask_lmk_token:
            self.chunk_size = config.chunk_size

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape

        self.apply_rope = True
        self.sliding_window = config.sliding_window

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)  # (B, h, L, d)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)  # (B, h, L, d)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        # assert not torch.allclose(q_rot, query_states, atol=1e-5)
        # assert not torch.allclose(k_rot, key_states, atol=1e-5)
        # query_states = q_rot
        # key_states = k_rot
        

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
            if self.sliding_window is not None and not self.mask_lmk_token:
                if self.layer_idx < len(past_key_values.layers):
                    kv_item = past_key_values.layers[self.layer_idx]
                    if kv_item.keys is not None and kv_item.keys.shape[-2] > self.sliding_window:
                        kv_item.keys = kv_item.keys[:, :, -self.sliding_window :, :]
                        kv_item.values = kv_item.values[:, :, -self.sliding_window :, :]



        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        kwargs.pop("position_ids", None)
        if not self.mask_lmk_token:
            attn_output, attn_weights = attention_interface(
                self,
                query_states, # (B, h, L, d)
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,  # diff with Llama
                **kwargs,
            )
            # attn_output: (B, L, h, d) — attention_interface already transposes
        else:
            attn_weights = None
            q_blhd = query_states.transpose(1, 2).contiguous()
            k_blhd = key_states.transpose(1, 2).contiguous()
            v_blhd = value_states.transpose(1, 2).contiguous()
            kv_start = 0
            if past_key_values is not None:
                kv_end = None
                if cache_position is not None and cache_position.numel() > 0:
                    kv_end = int(cache_position[-1].item()) + 1
                elif self.layer_idx < len(past_key_values.layers):
                    kv_layer = past_key_values.layers[self.layer_idx]
                    if hasattr(kv_layer, "get_seq_length"):
                        kv_end = int(kv_layer.get_seq_length())
                if kv_end is not None:
                    kv_start = max(kv_end - k_blhd.shape[1], 0)
            attn_output, _ = flex_attn_tl(
                q_blhd,
                k_blhd,
                v_blhd,
                window_size = self.sliding_window,
                chunk_size = self.chunk_size,
                training = self.training,
                mask_lmk= True,
                expand_to_chunk = False,
                kv_start=kv_start,
            )
            # flex_attn_tl returns (B, L, h, d) directly

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int, attn_cls):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = attn_cls(config, layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3Config, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        # ----------------------------------------------------------------
        # Partial-RoPE / HoPE: wavelength-based masking (opt-in).
        #   ``enable_inrange_rope`` keeps only freq-pairs whose period
        #   ``lambda_j = 2*pi / inv_freq[j]`` fits inside the training
        #   context window ``L`` -- so every kept freq completes at least
        #   ``m`` full cycles within ``L`` tokens and the model never sees
        #   rotary angles it was not trained on.  Cutoff:
        #       ``inv_freq[j] >= 2*pi*m / L``
        #   where
        #       ``L = rope_context_length``   (default max_position_embeddings)
        #       ``m = rope_period_multiplier`` (default 1.0; 2.0 = "at least
        #                                       two full cycles in L").
        #   The HoPE paper corresponds to ``m = 1.0``.
        #
        # Because HF's rotary pairs dim ``j`` with dim ``j + head_dim/2`` and
        # both share ``inv_freq[j]``, zeroing ``inv_freq[j]`` makes BOTH of
        # those head-dim components position-invariant.
        # ----------------------------------------------------------------
        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        self._rope_head_dim_half = head_dim // 2
        self.enable_inrange_rope = bool(getattr(config, "enable_inrange_rope", False))
        self.rope_context_length = int(
            getattr(config, "rope_context_length", config.max_position_embeddings)
        )
        self.rope_period_multiplier = float(
            getattr(config, "rope_period_multiplier", 1.0)
        )

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        if self.enable_inrange_rope:
            inv_freq = self._mask_long_period_(inv_freq)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq
        # print(f'self.inv_freq: {self.inv_freq}, {self.inv_freq[None, :, None]}')
        self.reinit = False

    def _mask_long_period_(self, inv_freq: torch.Tensor) -> torch.Tensor:
        """Zero out freq-pairs whose period exceeds the training window.

        Keeps ``inv_freq[j]`` iff ``2*pi / inv_freq[j] <= L / m``, i.e.
        ``inv_freq[j] >= 2*pi*m / L`` where ``L = self.rope_context_length``
        and ``m = self.rope_period_multiplier``.  HoPE is the ``m=1`` case.
        At least one freq-pair (the very highest) is always kept so the
        rotary path does not silently degenerate to full NoPE.  Caller is
        responsible for gating this on ``self.enable_inrange_rope``.
        """
        L = self.rope_context_length
        m = self.rope_period_multiplier
        # omega >= 2*pi*m / L  <=>  period <= L/m
        threshold = (2.0 * math.pi * m) / max(L, 1)
        keep = inv_freq >= threshold                                     # (head_dim//2,)
        keep[0] = True  # guarantee at least the top freq survives
        n_keep = int(keep.sum().item())
        if not hasattr(self, "_inrange_rope_logged"):
            self._inrange_rope_logged = True
            longest_kept_period = (
                2 * math.pi / inv_freq[keep][-1].item() if n_keep > 0 else 0.0
            )
            logger.warning_once(
                f"[inrange-RoPE] head_dim_half={self._rope_head_dim_half}, "
                f"L={L}, m={m}, threshold(inv_freq)={threshold:.4e} rad/token "
                f"-> keep {n_keep} / {self._rope_head_dim_half} freq-pairs; "
                f"longest kept period = {longest_kept_period:.1f} tokens."
            )
        inv_freq[~keep] = 0.0
        return inv_freq


    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        if not self.reinit:
            self.reinit = True
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, x.device)
            if self.enable_inrange_rope:
                inv_freq = self._mask_long_period_(inv_freq)
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        # print(f'self.inv_freq: {self.inv_freq[None, :, None]}')
        # print(f'fwd: pos_ids: {position_ids}')
        # print(f'inv_freq: {inv_freq_expanded}')

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos, sin


QWEN3_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Qwen3Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Qwen3 Model outputting raw hidden-states without any specific head on top.",
    QWEN3_START_DOCSTRING,
)
class Qwen3PreTrainedModel(PreTrainedModel):
    config_class = Qwen3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, Qwen3RMSNorm):
            module.weight.data.fill_(1.0)
