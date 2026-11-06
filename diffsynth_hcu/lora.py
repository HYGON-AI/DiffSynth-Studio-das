"""PEFT LoRA unit-scale and optional AITER helpers."""
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from types import MethodType

import torch

_SPLIT_FC1 = ContextVar("hcu_split_fc1", default=None)


def _load_aiter_add_swiglu():
    try:
        from aiter.ops.triton.add_swiglu import add_swiglu
    except Exception as exc:
        print(f"AITER Add+SwiGLU unavailable; using the PEFT path: {exc}", flush=True)
        return None
    return add_swiglu


@contextmanager
def split_fc1_scope(fc1):
    token = _SPLIT_FC1.set(fc1)
    try:
        yield
    finally:
        _SPLIT_FC1.reset(token)


def _unit_scale_forward(self, x, *args, **kwargs):
    original = self._hcu_original_lora_forward
    counters = self._hcu_dispatch_counts
    counters["lora_forward"] += 1
    adapters = self.active_adapters
    if args or kwargs or self.disable_adapters or self.merged or len(adapters) != 1:
        counters["original_forward"] += 1
        return original(x, *args, **kwargs)
    adapter = adapters[0]
    scale = self.scaling.get(adapter)
    if (adapter not in self.lora_A or adapter not in self.lora_B
            or type(scale) not in (int, float) or scale != 1.0
            or getattr(self, "use_dora", {}).get(adapter, False)
            or getattr(self, "lora_variant", {}).get(adapter) is not None):
        counters["original_forward"] += 1
        return original(x, *args, **kwargs)
    counters["unit_scale_fast_path"] += 1
    self._check_forward_args(x)
    result = self.base_layer(x)
    result_dtype = result.dtype
    lora_a, lora_b = self.lora_A[adapter], self.lora_B[adapter]
    x = self._cast_input_dtype(x, lora_a.weight.dtype) if hasattr(self, "_cast_input_dtype") else x.to(lora_a.weight.dtype)
    delta = lora_b(lora_a(self.lora_dropout[adapter](x)))
    if (_SPLIT_FC1.get() is self and result.is_cuda and result.device == delta.device
            and result.dtype in (torch.float16, torch.bfloat16) and result.dtype == delta.dtype
            and result.shape == delta.shape and result.ndim >= 2 and result.shape[-1] % 2 == 0
            and result.is_contiguous() and delta.is_contiguous()):
        return result, delta
    return (result + delta).to(result_dtype)


class LoRAAdapter:
    def __init__(self, model, *, aiter="auto"):
        from peft.tuners.lora.layer import Linear
        if aiter not in ("off", "auto", "on"):
            raise ValueError("aiter must be off, auto, or on")
        self.calls = Counter()
        self.originals = []
        add_swiglu = None if aiter == "off" else _load_aiter_add_swiglu()
        if aiter == "on" and add_swiglu is None:
            raise RuntimeError("AITER Add+SwiGLU was requested but is unavailable")
        fc1_modules = {
            m for m in model.modules()
            if getattr(m, "_hcu_fc1_swiglu", False)
            or getattr(getattr(m, "base_layer", None), "_hcu_fc1_swiglu", False)
        }
        for name, module in model.named_modules():
            if type(module) is not Linear or type(module.base_layer) is not torch.nn.Linear:
                continue
            if getattr(module, "_hcu_lora_adapter", None) is not None:
                continue
            if getattr(module.forward, "__func__", None) is not Linear.forward:
                continue
            original = module.forward
            self.originals.append((module, original, getattr(module, "_hcu_original_lora_forward", None),
                                   getattr(module, "_hcu_aiter_add_swiglu", None), "forward" in vars(module)))
            module._hcu_original_lora_forward = original
            module._hcu_aiter_add_swiglu = add_swiglu if module in fc1_modules else None
            module._hcu_dispatch_counts = self.calls
            module.forward = MethodType(_unit_scale_forward, module)
            module._hcu_lora_adapter = self
        self.count = len(self.originals)

    def close(self):
        for module, original, old_forward, old_aiter, had_forward in reversed(self.originals):
            if getattr(module, "_hcu_lora_adapter", None) is not self:
                continue
            if had_forward:
                module.forward = original
            else:
                delattr(module, "forward")
            for attr in ("_hcu_lora_adapter", "_hcu_dispatch_counts"):
                if hasattr(module, attr):
                    delattr(module, attr)
            if old_forward is None:
                if hasattr(module, "_hcu_original_lora_forward"):
                    delattr(module, "_hcu_original_lora_forward")
            else:
                module._hcu_original_lora_forward = old_forward
            if old_aiter is None:
                if hasattr(module, "_hcu_aiter_add_swiglu"):
                    delattr(module, "_hcu_aiter_add_swiglu")
            else:
                module._hcu_aiter_add_swiglu = old_aiter
        self.originals.clear()


def install_lora_fast_paths(model, *, aiter="auto"):
    if torch.version.hip is None:
        return None
    if not any(hasattr(module, "lora_A") and hasattr(module, "lora_B") for module in model.modules()):
        return None
    return LoRAAdapter(model, aiter=aiter)
