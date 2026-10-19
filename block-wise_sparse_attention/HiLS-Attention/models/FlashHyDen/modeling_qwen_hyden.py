"""Qwen decoder using landmark-free HyDen sparse attention."""

from __future__ import annotations

from functools import partial
from typing import Union

import torch
from torch import nn
from transformers.generation import GenerationMixin
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from models.FlashHiLS.modeling_qwen_hils import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    next_of_y,
)

from .configuration_hyden import HyDenConfig
from .hyden_attention import HyDenAttention
from .compressor import Compressor
from .hyden_forward import (
    get_hyden_cache_seq_length,
    hyden_causal_lm_forward,
    hyden_model_forward,
)


class HyDenPreTrainedModel(Qwen3PreTrainedModel):
    config_class = HyDenConfig
    _no_split_modules = ["Qwen3DecoderLayer"]

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Compressor):
            module.reset_parameters()


class HyDenModel(HyDenPreTrainedModel):
    def __init__(self, config: HyDenConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = next_of_y(config.vocab_size, 32)
        self.embed_tokens = nn.Embedding(
            self.vocab_size, config.hidden_size, self.padding_idx
        )

        self.full_attn_interleave = config.full_attn_interleave
        self.num_swa_layers = getattr(config, "num_swa_layers", 0)

        def attention_class(layer_idx: int):
            if layer_idx < self.num_swa_layers:
                return Qwen3Attention
            if self.full_attn_interleave > 0 and (
                (layer_idx - self.num_swa_layers) % self.full_attn_interleave
                == self.full_attn_interleave - 1
            ):
                return partial(HyDenAttention, norm_cls=Qwen3RMSNorm)
            return Qwen3Attention

        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(
                    config, layer_idx, attn_cls=attention_class(layer_idx)
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        cache_position=None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        return hyden_model_forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )


class KwargsForCausalLM(FlashAttentionKwargs):
    pass


class HyDenForCausalLM(HyDenPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: HyDenConfig, **kwargs):
        super().__init__(config)
        self.model = HyDenModel(config)
        self.vocab_size = next_of_y(config.vocab_size, 32)
        self.chunk_size = config.chunk_size
        self.lm_head = nn.Linear(config.hidden_size, self.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def get_decoder(self):
        return self.model

    def set_decoder(self, decoder):
        self.model = decoder

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        logits_to_keep=0,
        num_logits_to_keep=None,
        **kwargs,
    ):
        if num_logits_to_keep is not None:
            logits_to_keep = num_logits_to_keep
        if isinstance(logits_to_keep, int) and logits_to_keep == 0:
            logits_to_keep = 1

        past_length = get_hyden_cache_seq_length(
            past_key_values, self.config.num_hidden_layers
        )
        model_input_ids = input_ids[:, -1:] if past_length > 0 else input_ids
        seq_length = model_input_ids.shape[1]
        if cache_position is None or cache_position.numel() != seq_length:
            cache_position = torch.arange(
                past_length,
                past_length + seq_length,
                device=input_ids.device,
            )

        if position_ids is None:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                if past_length > 0:
                    position_ids = position_ids[:, -seq_length:]
            else:
                position_ids = cache_position.unsqueeze(0).expand(
                    model_input_ids.shape[0], -1
                )

        return {
            "input_ids": model_input_ids,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "logits_to_keep": logits_to_keep,
        }

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        cache_position=None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        return hyden_causal_lm_forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )


__all__ = ["HyDenConfig", "HyDenForCausalLM", "HyDenModel", "HyDenPreTrainedModel"]
