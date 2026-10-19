"""Forward functions for the landmark-free HyDen model."""

from __future__ import annotations

import inspect

import torch
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from utils.hils_cache_utils import HiLSDynamicLayer
from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import slice_position_embedding
from veomni.utils import logging
from veomni.utils.seqlen_pos_transform_utils import prepare_fa_kwargs_from_position_ids


logger = logging.get_logger(__name__)


def _cu_seqlens_to_doc_ids(cu_seq_lens, total_len, device):
    if not isinstance(cu_seq_lens, torch.Tensor):
        cu_seq_lens = torch.tensor(cu_seq_lens, dtype=torch.int32, device=device)
    else:
        cu_seq_lens = cu_seq_lens.to(device)
    doc_ids = torch.zeros(total_len, dtype=torch.int32, device=device)
    boundaries = cu_seq_lens[1:-1]
    if boundaries.numel() > 0:
        doc_ids[boundaries.long()] = 1
        doc_ids = doc_ids.cumsum(0)
    return doc_ids


def _build_mask_kwargs(
    mask_fn,
    self,
    attention_mask,
    inputs_embeds,
    cache_position,
    past_key_values,
    position_ids,
):
    params = inspect.signature(mask_fn).parameters
    kwargs = {
        "config": self.config,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
    }
    if "inputs_embeds" in params:
        kwargs["inputs_embeds"] = inputs_embeds
    elif "input_embeds" in params:
        kwargs["input_embeds"] = inputs_embeds
    if "cache_position" in params:
        kwargs["cache_position"] = cache_position
    return kwargs


def _prepare_causal_mask_mapping(
    self,
    attention_mask,
    inputs_embeds,
    cache_position,
    past_key_values,
    position_ids,
):
    if isinstance(attention_mask, dict):
        return attention_mask

    mapping = {
        "full_attention": create_causal_mask(
            **_build_mask_kwargs(
                create_causal_mask,
                self,
                attention_mask,
                inputs_embeds,
                cache_position,
                past_key_values,
                position_ids,
            )
        )
    }
    layer_types = getattr(self.config, "layer_types", None) or ()
    if (
        "sliding_attention" in layer_types
        or getattr(self.config, "sliding_window", None) is not None
    ):
        mapping["sliding_attention"] = create_sliding_window_causal_mask(
            **_build_mask_kwargs(
                create_sliding_window_causal_mask,
                self,
                attention_mask,
                inputs_embeds,
                cache_position,
                past_key_values,
                position_ids,
            )
        )
    return mapping


def get_hyden_cache_seq_length(past_key_values, num_hidden_layers: int) -> int:
    if past_key_values is None:
        return 0
    layers = getattr(past_key_values, "layers", None)
    if layers is not None and len(layers) > num_hidden_layers:
        lengths = [
            int(layer.get_seq_length())
            for layer in layers[num_hidden_layers:]
            if hasattr(layer, "get_seq_length")
        ]
        if lengths:
            return max(lengths)
    return int(past_key_values.get_seq_length())


def hyden_model_forward(
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
):
    """Independent HyDen decoder loop with no landmark preprocessing."""
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError("Specify exactly one of input_ids or inputs_embeds")
    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once(
            "`use_cache=True` is incompatible with gradient checkpointing; disabling cache."
        )
        use_cache = False
    if not isinstance(past_key_values, (type(None), Cache)):
        raise ValueError("past_key_values must be a Cache object or None")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache()
    if use_cache and not self.training and isinstance(past_key_values, DynamicCache):
        required_layers = 2 * self.config.num_hidden_layers
        while len(past_key_values.layers) < required_layers:
            idx = len(past_key_values.layers)
            layer = (
                HiLSDynamicLayer()
                if idx >= self.config.num_hidden_layers
                else DynamicLayer()
            )
            past_key_values.layers.append(layer)

    seq_length = inputs_embeds.shape[1]
    if cache_position is None:
        past_length = get_hyden_cache_seq_length(
            past_key_values, self.config.num_hidden_layers
        )
        cache_position = torch.arange(
            past_length,
            past_length + seq_length,
            device=inputs_embeds.device,
        )
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    if (
        inputs_embeds.shape[0] == 1
        and position_ids is not None
        and past_key_values is None
        and "cu_seq_lens_q" in kwargs
    ):
        (cu_q, cu_k), (max_q, max_k) = prepare_fa_kwargs_from_position_ids(
            position_ids
        )
        kwargs = dict(kwargs)
        kwargs.update(
            {
                "cu_seq_lens_q": cu_q,
                "cu_seq_lens_k": cu_k,
                "max_length_q": max_q,
                "max_length_k": max_k,
                "doc_ids": _cu_seqlens_to_doc_ids(
                    cu_q, seq_length, inputs_embeds.device
                ),
            }
        )

    causal_masks = _prepare_causal_mask_mapping(
        self,
        attention_mask,
        inputs_embeds,
        cache_position,
        past_key_values,
        position_ids,
    )
    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)
    sp_group = (
        get_parallel_state().sp_group if get_parallel_state().sp_enabled else None
    )
    position_embeddings = slice_position_embedding(
        position_embeddings, dim=1, sp_group=sp_group
    )



    all_hidden_states = () if output_hidden_states else None
    all_self_attentions = () if output_attentions else None
    for layer_idx, decoder_layer in enumerate(
        self.layers[: self.config.num_hidden_layers]
    ):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        attention_type = getattr(
            decoder_layer,
            "attention_type",
            self.config.layer_types[layer_idx],
        )
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_masks.get(
                attention_type, causal_masks["full_attention"]
            ),
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = layer_outputs[0]
        if output_attentions:
            all_self_attentions += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
    )


def hyden_causal_lm_forward(
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
    logits_to_keep=0,
    **kwargs,
) -> CausalLMOutputWithPast:
    """Causal LM forward without landmark insertion or label rewriting."""
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )

    outputs = self.model(
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
    hidden_states = outputs.last_hidden_state
    slice_indices = (
        slice(-logits_to_keep, None)
        if isinstance(logits_to_keep, int)
        else logits_to_keep
    )
    hidden_states = hidden_states[:, slice_indices, :]

    loss = None
    logits = None
    if labels is not None:
        loss, logits = self.loss_function(
            logits=None,
            labels=labels,
            vocab_size=self.vocab_size,
            hidden_states=hidden_states,
            weights=self.lm_head.weight,
            **kwargs,
        )
    else:
        logits = self.lm_head(hidden_states)

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


__all__ = [
    "get_hyden_cache_seq_length",
    "hyden_model_forward",
    "hyden_causal_lm_forward",
]
