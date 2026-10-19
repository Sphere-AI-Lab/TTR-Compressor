#!/usr/bin/env python3

# All local paths are relative to the top-level clone directory.
import os as _os
import sys as _sys
from pathlib import Path as _Path
_code_root = _Path(__file__).resolve().parents[1]
_project_root = _code_root.parent.parent
_os.chdir(_project_root)
_os.environ.setdefault("HF_HOME", ".cache/huggingface")
_os.environ.setdefault("RULER_DATA_DIR", _os.environ["HF_HOME"] + "/ruler")
_sys.path[:0] = [str(_code_root), str(_code_root.parent / "lm-evaluation-harness")]


from functools import wraps
import json
from pathlib import Path
import sys

import torch
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM
from transformers import AutoConfig, AutoModelForCausalLM


class _EvalParallelState:
    """Minimal VeOmni state for accelerate data-parallel evaluation.

    lm-evaluation-harness uses one complete model replica per accelerate process.
    Its distributed world therefore must not be interpreted as a VeOmni model-
    parallel mesh.  In particular, sequence parallelism must remain disabled.
    """

    sp_enabled = False
    sp_group = None
    ulysses_enabled = False
    ulysses_group = None


_EVAL_PARALLEL_STATE = _EvalParallelState()


def _eval_parallel_state():
    return _EVAL_PARALLEL_STATE


def _install_veomni_eval_parallel_state():
    """Disable VeOmni model parallelism inside this lm-eval process only.

    This is installed before importing any project model, so both model code
    and VeOmni attention adapters bind to the evaluation-safe state getter.
    Accelerate's torch.distributed data parallelism remains untouched.
    """
    from veomni.distributed import parallel_state

    parallel_state.get_parallel_state = _eval_parallel_state


def _patch_loaded_veomni_state_consumers():
    """Update direct imports held by VeOmni modules already in sys.modules."""
    for module_name, module in tuple(sys.modules.items()):
        if not module_name.startswith("veomni.") or module is None:
            continue
        if hasattr(module, "get_parallel_state"):
            module.get_parallel_state = _eval_parallel_state


def _canonical_kernel_layout(tensor: torch.Tensor) -> torch.Tensor:
    """Return a tensor with canonical dense strides, including size-1 dims.

    PyTorch may consider a tensor contiguous even when a singleton dimension
    has a non-canonical stride. TileLang kernels validate every stride exactly,
    which matters for the [B, 1, H, D] tensors used during cached generation.
    """
    expected_strides = []
    running_stride = 1
    for size in reversed(tensor.shape):
        expected_strides.append(running_stride)
        running_stride *= size
    expected_strides.reverse()

    if tuple(tensor.stride()) == tuple(expected_strides):
        return tensor

    canonical = torch.empty(
        tensor.shape,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    canonical.copy_(tensor)
    return canonical


def _make_tilelang_eval_layout_adapter(original):
    @wraps(original)
    def eval_layout_adapter(q, k, v, *args, **kwargs):
        return original(
            _canonical_kernel_layout(q),
            _canonical_kernel_layout(k),
            _canonical_kernel_layout(v),
            *args,
            **kwargs,
        )

    eval_layout_adapter._hils_eval_layout_adapter = True
    return eval_layout_adapter


def _install_tilelang_eval_layout_adapters(model):
    """Patch only the loaded eval model, without changing training modules."""
    adapted = 0
    for module in model.modules():
        for attribute in ("sparse_attention_func", "hils_func"):
            original = getattr(module, attribute, None)
            if original is None or getattr(
                original, "_hils_eval_layout_adapter", False
            ):
                continue
            setattr(
                module,
                attribute,
                _make_tilelang_eval_layout_adapter(original),
            )
            adapted += 1
    return adapted


def load_local_config(pretrained: str) -> dict:
    config_path = Path(pretrained) / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"config.json not found in checkpoint: {pretrained}"
        )
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def register_hyden():
    from models.FlashHyDen import hyden_forward
    from models.FlashHyDen.configuration_hyden import HyDenConfig
    from models.FlashHyDen.modeling_qwen_hyden import HyDenForCausalLM

    _patch_loaded_veomni_state_consumers()

    hyden_forward.get_parallel_state = _eval_parallel_state

    HyDenForCausalLM.config_class = HyDenConfig

    AutoConfig.register(
        "qwen_hyden",
        HyDenConfig,
        exist_ok=True,
    )
    AutoModelForCausalLM.register(
        HyDenConfig,
        HyDenForCausalLM,
        exist_ok=True,
    )


@register_model("hils_attn_hf")
class HiLSAttentionHFLM(HFLM):
    def __init__(self, pretrained: str, **kwargs):
        _install_veomni_eval_parallel_state()

        config_dict = load_local_config(pretrained)
        model_type = config_dict.get("model_type")
        architectures = config_dict.get("architectures") or []

        if model_type != "qwen_hyden":
            raise ValueError(f"Unsupported model_type={model_type!r}")
        register_hyden()
        kwargs.setdefault("attn_implementation", config_dict.get("_attn_implementation") or "flash_attention_2")

        kwargs.setdefault("backend", "causal")
        super().__init__(pretrained=pretrained, **kwargs)

        _install_tilelang_eval_layout_adapters(self.model)


if __name__ == "__main__":
    cli_evaluate()
