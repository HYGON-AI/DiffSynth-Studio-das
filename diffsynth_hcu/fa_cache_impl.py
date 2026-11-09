# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""FA2 O/LSE cache scoped to one reentrant checkpoint.

Q/K/V are recomputed. No detached output is substituted for an autograd node:
replay constructs a new node using the original FA backward implementation.
Only BF16/FP16, non-causal, dropout=0 packed self attention is supported.
"""
from contextvars import ContextVar
from contextlib import contextmanager
from functools import lru_cache
import importlib

import torch
from torch.autograd.function import once_differentiable


_SCOPE = ContextVar("hcu_fa_cache", default=None)
_CALLS = {"record": 0, "replay": 0}


@lru_cache(None)
def _backend():
    interface = importlib.import_module("flash_attn.flash_attn_interface")
    for name in ("_wrapped_flash_attn_varlen_forward", "_wrapped_flash_attn_varlen_backward"):
        if not callable(getattr(interface, name, None)):
            raise RuntimeError(f"FA cache needs flash-attention-cutlass interface {name}")
    return interface


def _signature(q, k, v, cu, max_seqlen, scale):
    return (tuple((tuple(t.shape), tuple(t.stride()), t.dtype, t.device) for t in (q, k, v)),
            cu.data_ptr(), cu._version, tuple(cu.shape), int(max_seqlen), scale)


class FACheckpointCache:
    def __init__(self, mode="gpu"):
        if mode != "gpu":
            raise ValueError("Only GPU FA replay cache is supported")
        self.mode = mode
        self.calls = 0
        self.signature = None
        self.gpu = None
        self.prefetched = None
        self.gpu_bytes = 0

    @contextmanager
    def scope(self):
        replay = self.calls > 0
        self.calls += 1
        if replay:
            if self.gpu is None:
                raise RuntimeError("FA checkpoint has no cached result; repeated backward/retain_graph is unsupported")
            self.prefetched = self.gpu
        token = _SCOPE.set((self, replay))
        self.seen = 0
        try:
            yield
            if self.seen != 1:
                raise RuntimeError("FA cache expects exactly one FA2 call per selected checkpoint")
        finally:
            _SCOPE.reset(token)
            self.prefetched = None
            if replay:
                self.gpu = None

    def wrap(self, function):
        def run(*args, **kwargs):
            with self.scope():
                return function(*args, **kwargs)
        return run

    def save(self, signature, tensors, device):
        if self.gpu is not None:
            raise RuntimeError("Cannot record the same checkpoint twice")
        self.signature = signature
        self.gpu = tuple(t.detach() if t is not None else None for t in tensors)
        self.gpu_bytes = sum(t.numel() * t.element_size() for t in self.gpu
                             if t is not None and t.device.type == "cuda")

    def take(self, signature):
        if signature != self.signature:
            raise RuntimeError("FA replay metadata differs from original forward")
        return self.prefetched


class _CachedFA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cu, max_seqlen, scale, cache, replay):
        signature = _signature(q, k, v, cu, max_seqlen, scale)
        if replay:
            _CALLS["replay"] += 1
            out, lse, rng = cache.take(signature)
        else:
            out, lse, _, rng = _backend()._wrapped_flash_attn_varlen_forward(
                q, k, v, None, cu, cu, None, None, None, None,
                max_seqlen, max_seqlen, 0.0, scale, False, False, -1, -1,
                softcap=0.0, return_softmax=False,
            )
            cache.save(signature, (out, lse, rng), q.device)
            _CALLS["record"] += 1
        ctx.save_for_backward(q, k, v, out, lse, cu, rng)
        ctx.max_seqlen, ctx.scale = max_seqlen, scale
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        q, k, v, out, lse, cu, rng = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _backend()._wrapped_flash_attn_varlen_backward(
            dout, q, k, v, out, lse, dq, dk, dv, cu, cu,
            ctx.max_seqlen, ctx.max_seqlen, 0.0, ctx.scale, False, -1, -1,
            0.0, None, False, rng_state=rng,
        )
        return dq, dk, dv, None, None, None, None, None


def cached_attention(q, k, v, cu_seqlens, max_seqlen, scale):
    scope = _SCOPE.get()
    if scope is None:
        return None
    cache, replay = scope
    cache.seen += 1
    if cache.seen != 1:
        raise RuntimeError("Multiple attention calls in one FA cache scope")
    if (q.device.type != "cuda" or q.dtype not in (torch.bfloat16, torch.float16)
            or q.ndim != 3 or q.shape[-1] % 8 or q.shape != k.shape or q.shape != v.shape
            or any(t.dtype != q.dtype or t.device != q.device or t.stride(-1) != 1 for t in (k, v))
            or q.stride(-1) != 1 or cu_seqlens.dtype != torch.int32
            or cu_seqlens.device != q.device or not cu_seqlens.is_contiguous()):
        raise RuntimeError("FA cache requires packed FP16/BF16 self attention with aligned head dim")
    scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    return _CachedFA.apply(q, k, v, cu_seqlens, max_seqlen, scale, cache, replay)


# Layer selection is owned by the model-level configuration. This module only
# owns cache state for one reentrant invocation.
