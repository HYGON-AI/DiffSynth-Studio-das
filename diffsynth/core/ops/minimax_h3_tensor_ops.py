# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""H3 tensor expressions and indexed modulation gradients.

ZeRO module hooks, parameter gathers, linear layers and lightop stay outside
these graphs. Only parameter-free tensor expressions are compiled.
Configuration is process-wide and must precede training.
Trainable modulation uses a first-order custom backward outside those graphs.
"""

from functools import lru_cache
import warnings

import torch
from torch.autograd.function import once_differentiable


_COMPILED = {}
_CALLS = {}
_OPTIONS = None


def _scale_shift(x, shift, scale, indices):
    return (x * (1.0 + scale.index_select(0, indices)) + shift.index_select(0, indices)).to(x.dtype)


def _gate(x, gate, other, indices):
    return (x + gate.index_select(0, indices) * other).to(x.dtype)


def _update(gate, other, indices):
    return gate.index_select(0, indices) * other


def _scale_shift_backward(grad, x, scale, indices, need_x, need_scale):
    dx = grad * (1.0 + scale.index_select(0, indices)) if need_x else None
    dscale = grad * x if need_scale else None
    return dx, dscale


def _gate_backward(grad, gate, other, indices, need_gate, need_other):
    dgate = grad * other if need_gate else None
    dother = grad * gate.index_select(0, indices) if need_other else None
    return dgate, dother


@lru_cache(None)
def _load_modulation_reduce():
    try:
        from .minimax_h3_segment_reduce import indexed_row_sum
    except (ImportError, OSError) as error:
        warnings.warn(f"H3 modulation reduction unavailable; using native autograd: {error}",
                      RuntimeWarning, stacklevel=2)
        return None
    return indexed_row_sum


def _can_reduce_modulation(values, indices, *tables):
    # Frozen LoRA modulation and inference retain their existing compiled path.
    if not torch.is_grad_enabled() or not any(t.requires_grad for t in tables):
        return False
    if not values.is_cuda or values.ndim != 2 or values.dtype not in (
        torch.float16, torch.bfloat16, torch.float32,
    ):
        return False
    if indices.ndim != 1 or indices.dtype != torch.long or indices.device != values.device:
        return False
    if indices.shape[0] != values.shape[0] or not 0 < values.shape[0] <= 65536:
        return False
    if values.shape[1] == 0:
        return False
    if any(t.ndim != 2 or t.dtype != values.dtype or t.device != values.device
           or t.shape[1] != values.shape[1] or not 0 < t.shape[0] <= 16 for t in tables):
        return False
    return _load_modulation_reduce() is not None


def _sum_modulation(grad, indices, rows):
    if not grad.is_cuda:
        # CPU reference for autograd/gradcheck tests; production CPU uses native.
        return grad.new_zeros((rows, grad.shape[1])).index_add(0, indices, grad)
    return _load_modulation_reduce()(grad, indices, rows)


class _ScaleShift(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, shift, scale, indices):
        ctx.save_for_backward(x, scale, indices)
        ctx.shift_rows = shift.shape[0]
        return _run("scale_shift", _scale_shift, x, shift, scale, indices)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        x, scale, indices = ctx.saved_tensors
        need_x, need_shift, need_scale, _ = ctx.needs_input_grad
        dx, dscale = _run("scale_shift_backward", _scale_shift_backward,
                         grad, x, scale, indices, need_x, need_scale)
        dshift = _sum_modulation(grad, indices, ctx.shift_rows) if need_shift else None
        dscale = _sum_modulation(dscale, indices, scale.shape[0]) if need_scale else None
        return dx, dshift, dscale, None


class _Gate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate, other, indices):
        ctx.save_for_backward(gate, other, indices)
        return _run("gate", _gate, x, gate, other, indices)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        gate, other, indices = ctx.saved_tensors
        need_x, need_gate, need_other, _ = ctx.needs_input_grad
        dgate, dother = _run("gate_backward", _gate_backward,
                            grad, gate, other, indices, need_gate, need_other)
        dgate = _sum_modulation(dgate, indices, gate.shape[0]) if need_gate else None
        return grad if need_x else None, dgate, dother, None


class _Update(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, other, indices):
        ctx.save_for_backward(gate, other, indices)
        return _run("update", _update, gate, other, indices)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        gate, other, indices = ctx.saved_tensors
        need_gate, need_other, _ = ctx.needs_input_grad
        dgate, dother = _run("gate_backward", _gate_backward,
                            grad, gate, other, indices, need_gate, need_other)
        dgate = _sum_modulation(dgate, indices, gate.shape[0]) if need_gate else None
        return dgate, dother, None


def _rope(x, freqs):
    rot_dim = freqs.shape[-1]
    x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    cos = torch.cos(freqs).to(x.dtype).unsqueeze(1)
    sin = torch.sin(freqs).to(x.dtype).unsqueeze(1)
    x1, x2 = torch.chunk(x_rot, 2, dim=-1)
    rotated = torch.cat((-x2, x1), dim=-1)
    return torch.cat((x_rot * cos + rotated * sin, x_pass), dim=-1)


_FALLBACK_MODE_OPTIONS = {
    # Stable subset of torch._inductor.list_mode_options outputs for the
    # elementwise regions compiled here. CUDA Graphs are always disabled by
    # the caller, so reduce-overhead degenerates to the default mode.
    "default": {},
    "reduce-overhead": {"triton.cudagraphs": True},
    "max-autotune": {"max_autotune": True, "triton.cudagraphs": True},
    "max-autotune-no-cudagraphs": {"max_autotune": True},
}


def _mode_options(mode):
    """Resolve a compile mode to Inductor options.

    torch._inductor.list_mode_options is a private API and may change or
    disappear across torch versions; use it when present for exact parity
    with torch.compile's mode handling and fall back otherwise.
    """
    if mode not in _FALLBACK_MODE_OPTIONS:
        raise ValueError(f"Unknown compile mode: {mode}")
    try:
        from torch._inductor import list_mode_options
    except ImportError:
        return dict(_FALLBACK_MODE_OPTIONS[mode])
    return dict(list_mode_options(mode))


def configure_minimax_h3_tensor_compile(mode="default"):
    global _OPTIONS
    # Use mode options without CUDA Graphs: ZeRO-3/checkpoint recompute can
    # change parameter storage lifetime. Do not capture module/parameter hooks.
    options = _mode_options(mode)
    options["triton.cudagraphs"] = False
    if _OPTIONS == options:
        return
    functions = {
        "scale_shift": _scale_shift, "gate": _gate, "update": _update,
        "rope": _rope,
        "scale_shift_backward": _scale_shift_backward, "gate_backward": _gate_backward,
    }
    compiled = {
        name: torch.compile(fn, backend="inductor", fullgraph=True, dynamic=True, options=options)
        for name, fn in functions.items()
    }
    _COMPILED.clear()
    _COMPILED.update(compiled)
    _CALLS.clear()
    _OPTIONS = options
    print(f"H3 tensor compile configured: {tuple(compiled)}, fullgraph=True, dynamic=True, options={options}. "
          "Compilation occurs on first invocation; module hooks and lightop remain eager.")


def get_minimax_h3_tensor_compile_info():
    """Calls prove dispatch only; inspect TORCH_LOGS=output_code for fusion."""
    return {
        "configured": list(_COMPILED), "calls": dict(_CALLS),
        "options": _OPTIONS,
    }


def _run(name, native, *args):
    if name in _COMPILED:
        # No catch-and-fallback: graph breaks/compiler failures must be visible.
        result = _COMPILED[name](*args)
        _CALLS[name] = _CALLS.get(name, 0) + 1
        return result
    return native(*args)


def modulate_scale_shift(x, shift, scale, indices):
    if _can_reduce_modulation(x, indices, shift, scale):
        return _ScaleShift.apply(x, shift, scale, indices)
    return _run("scale_shift", _scale_shift, x, shift, scale, indices)


def modulate_gate(x, gate, other, indices):
    if (x.shape == other.shape and x.dtype == other.dtype and x.device == other.device
            and _can_reduce_modulation(other, indices, gate)):
        return _Gate.apply(x, gate, other, indices)
    return _run("gate", _gate, x, gate, other, indices)


def modulate_update(gate, other, indices):
    if _can_reduce_modulation(other, indices, gate):
        return _Update.apply(gate, other, indices)
    return _run("update", _update, gate, other, indices)


def compiled_rope(x, freqs):
    if "rope" in _COMPILED:
        return _run("rope", _rope, x, freqs)
    return None
