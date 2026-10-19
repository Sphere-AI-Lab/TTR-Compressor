from veomni.models.loader import MODELING_REGISTRY, MODEL_CONFIG_REGISTRY


def _select_model_class(architecture: str, model_cls, causal_lm_cls):
    if "ForCausalLM" in architecture:
        return causal_lm_cls
    if "Model" in architecture:
        return model_cls
    return causal_lm_cls


@MODELING_REGISTRY.register("qwen_hyden")
def register_qwen_hyden_modeling(architecture: str):
    from .modeling_qwen_hyden import HyDenForCausalLM, HyDenModel

    return _select_model_class(architecture, HyDenModel, HyDenForCausalLM)


@MODEL_CONFIG_REGISTRY.register("qwen_hyden")
def register_qwen_hyden_config():
    from .configuration_hyden import HyDenConfig

    return HyDenConfig


__all__ = ["register_qwen_hyden_modeling", "register_qwen_hyden_config"]
