# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""H3 fast path for ordinary PEFT Linear adapters with unit scaling.

No parameter replacement or global PEFT patching. All linear/dropout modules
are called normally, including their DeepSpeed hooks. Other PEFT modes retain
the installed version's forward implementation.
"""

from types import MethodType
from contextvars import ContextVar

import torch


_SPLIT_FC1 = ContextVar("h3_split_fc1", default=None)


def _load_aiter_add_swiglu():
    try:
        from aiter.ops.triton.add_swiglu import add_swiglu
    except Exception as exc:
        # Optional backend imports may also fail while loading native libraries.
        print(f"H3 AITER Add+SwiGLU unavailable; using original path: {exc}", flush=True)
        return None
    return add_swiglu


def minimax_h3_fc1_swiglu(fc1, x):
    """Keep module hooks active, including during checkpoint recomputation.

    Only an explicitly enabled fc1 may return a tuple inside this scope.
    Outside it, the PEFT Linear retains its normal Tensor output contract.
    """
    from . import fuse_bias_swiglu

    if not getattr(fc1, "_h3_aiter_add_swiglu", False):
        return fuse_bias_swiglu(fc1(x))
    token = _SPLIT_FC1.set(fc1)
    try:
        hidden = fc1(x)
    finally:
        _SPLIT_FC1.reset(token)
    if not isinstance(hidden, tuple):
        return fuse_bias_swiglu(hidden)
    return fc1._h3_aiter_add_swiglu(*hidden)


def _unit_scale_forward(self, x, *args, **kwargs):
    original = self._h3_original_lora_forward
    adapters = self.active_adapters
    if args or kwargs or self.disable_adapters or self.merged or len(adapters) != 1:
        return original(x, *args, **kwargs)
    adapter = adapters[0]
    scale = self.scaling.get(adapter)
    if (
        adapter not in self.lora_A or adapter not in self.lora_B
        or type(scale) not in (int, float) or scale != 1.0
        or getattr(self, "use_dora", {}).get(adapter, False)
        or getattr(self, "lora_variant", {}).get(adapter) is not None
    ):
        return original(x, *args, **kwargs)

    self._check_forward_args(x)
    result = self.base_layer(x)
    result_dtype = result.dtype
    lora_a, lora_b = self.lora_A[adapter], self.lora_B[adapter]
    if hasattr(self, "_cast_input_dtype"):
        x = self._cast_input_dtype(x, lora_a.weight.dtype)
    else:
        x = x.to(lora_a.weight.dtype)
    delta = lora_b(lora_a(self.lora_dropout[adapter](x)))
    # Do not cast a mixed-dtype LoRA branch to force eligibility: doing so
    # would move the original PEFT rounding boundary before the addition.
    if (
        _SPLIT_FC1.get() is self
        and result.is_cuda and result.device == delta.device
        and result.dtype in (torch.float16, torch.bfloat16)
        and result.dtype == delta.dtype and result.shape == delta.shape
        and result.ndim >= 2 and result.shape[-1] > 0
        and result.shape[-1] % 2 == 0
        and result.is_contiguous() and delta.is_contiguous()
    ):
        return result, delta
    # Out-of-place addition preserves PEFT/ZeRO view and saved-tensor semantics.
    return (result + delta).to(result_dtype)


def enable_minimax_h3_unit_lora_scale(model):
    """Install before DeepSpeed wrapping; dispatch rechecks mutable adapter state."""
    from peft.tuners.lora.layer import Linear

    add_swiglu = _load_aiter_add_swiglu()

    installed = 0
    for module in model.modules():
        # Quantized/custom linear variants and already-patched forwards are
        # deliberately excluded, rather than guessing their forward semantics.
        if (
            type(module) is not Linear or type(module.base_layer) is not torch.nn.Linear
            or getattr(module.forward, "__func__", None) is not Linear.forward
        ):
            continue
        module._h3_original_lora_forward = module.forward
        module.forward = MethodType(_unit_scale_forward, module)
        installed += 1
    for module in model.modules():
        if type(module).__name__ != "MiniMaxH3MLP":
            continue
        fc1 = module.fc1
        if getattr(fc1.forward, "__func__", None) is _unit_scale_forward:
            fc1._h3_aiter_add_swiglu = add_swiglu
    return installed
