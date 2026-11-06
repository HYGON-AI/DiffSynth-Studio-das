"""FA replay cache state and model configuration."""
from .fa_cache_impl import FACheckpointCache

DEFAULT_FA_CACHE_LAYERS = 20


def fa_cache_layers(value=None, *, strict=True):
    if value is None:
        return DEFAULT_FA_CACHE_LAYERS
    try:
        value = int(value)
    except (TypeError, ValueError):
        if strict:
            raise ValueError(f"fa_cache_layers must be a non-negative integer, got {value!r}") from None
        return 0
    if value < 0:
        if strict:
            raise ValueError("fa_cache_layers must be non-negative")
        return 0
    return value


def configure_fa_cache(dit, layers=None, *, mode="gpu", checkpointing=True, checkpoint_offload=False):
    layers = fa_cache_layers(layers)
    import torch
    clear_fa_cache_config(dit)
    if layers == 0 or torch.version.hip is None or not checkpointing or checkpoint_offload:
        return 0
    if mode != "gpu":
        raise ValueError("Only GPU FA replay cache is supported")
    from diffsynth.core.attention.attention import ATTENTION_IMPLEMENTATION
    if ATTENTION_IMPLEMENTATION != "flash_attention_2":
        return 0
    blocks = list(getattr(dit, "blocks", ()))
    if layers > len(blocks):
        raise ValueError(f"fa_cache_layers={layers} exceeds model depth {len(blocks)}")
    for block in blocks[:layers]:
        block._hcu_fa_cache_mode = mode
    return layers


def clear_fa_cache_config(dit):
    for block in getattr(dit, "blocks", ()):
        if hasattr(block, "_hcu_fa_cache_mode"):
            delattr(block, "_hcu_fa_cache_mode")
