"""Model implementations for lm_eval.

Models are lazily loaded via the registry system to improve startup performance.

Usage
-----
For programmatic access, use the registry:

    from lm_eval.api.registry import get_model
    model_cls = get_model("hf")
    model = model_cls(pretrained="gpt2")

For direct imports (e.g., subclassing), use explicit module paths:

    from lm_eval.models.huggingface import HFLM
    from lm_eval.models.vllm_causallms import VLLM

Adding New Models
-----------------
1. Create your model class in a new file under lm_eval/models/
2. Use the @register_model decorator on your class
3. Add an entry to MODEL_MAPPING below for lazy discovery
"""

MODEL_MAPPING = {
    "hf": "lm_eval.models.huggingface:HFLM",
    "hf-auto": "lm_eval.models.huggingface:HFLM",
    "huggingface": "lm_eval.models.huggingface:HFLM",
}


def _register_all_models():
    """Register all known models lazily in the registry."""
    from lm_eval.api.registry import model_registry

    for name, path in MODEL_MAPPING.items():
        # Only register if not already present (avoids conflicts when modules are imported)
        if name not in model_registry:
            # Register the lazy placeholder
            model_registry.register(name, target=path)


# Call registration on module import
_register_all_models()

__all__ = ["MODEL_MAPPING"]