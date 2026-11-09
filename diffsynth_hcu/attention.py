"""Capability based packed varlen attention for HCU and CUDA backends."""
from collections import Counter
from functools import lru_cache
import importlib

import torch


_CALLS = Counter()


@lru_cache(None)
def _find_flash_varlen(implementation):
    modules = {
        "flash_attention_2": "flash_attn",
        "flash_attention_3": "flash_attn_3.flash_attn_interface",
        "flash_attention_4": "flash_attn.cute",
    }
    module_name = modules.get(implementation)
    if module_name is not None:
        try:
            func = getattr(importlib.import_module(module_name), "flash_attn_varlen_func", None)
        except (ImportError, OSError):
            return None, None
        if callable(func):
            return func, implementation
    return None, None


def _max_seqlen(cu_seqlens):
    values = cu_seqlens.detach().to(device="cpu", dtype=torch.int64).tolist()
    return max((stop - start for start, stop in zip(values[:-1], values[1:])), default=0)


def packed_varlen_attention(q, k, v, cu_seqlens, *, softmax_scale=None, max_seqlen=None, causal=False,
                            fallback=None, cache=None):
    """Run packed self attention, falling back per segment when FA is absent."""
    if max_seqlen is None:
        max_seqlen = _max_seqlen(cu_seqlens)
    flash, name = None, None
    if torch.version.hip is not None and q.is_cuda:
        from diffsynth.core.attention.attention import ATTENTION_IMPLEMENTATION
        if cache is None and not causal and ATTENTION_IMPLEMENTATION == "flash_attention_2":
            from .fa_cache_impl import cached_attention as cache
        if cache is not None:
            cached = cache(q, k, v, cu_seqlens, max_seqlen, softmax_scale)
            if cached is not None:
                _CALLS["fa_cache"] += 1
                return cached
        flash, name = _find_flash_varlen(ATTENTION_IMPLEMENTATION)
    if flash is not None:
        _CALLS[name] += 1
        out = flash(
            q, k, v,
            cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
            softmax_scale=softmax_scale, causal=causal,
        )
        return out[0] if isinstance(out, tuple) else out
    if fallback is None:
        from diffsynth.core.attention import attention_forward
        fallback = attention_forward
    _CALLS["segmented"] += 1
    out = torch.empty_like(q)
    bounds = cu_seqlens.detach().to(device="cpu", dtype=torch.int64).tolist()
    for start, stop in zip(bounds[:-1], bounds[1:]):
        if start == stop:
            continue
        seg = fallback(
            q[start:stop].transpose(0, 1).unsqueeze(0),
            k[start:stop].transpose(0, 1).unsqueeze(0),
            v[start:stop].transpose(0, 1).unsqueeze(0),
            scale=softmax_scale,
            is_causal=causal,
        )
        out[start:stop] = seg.squeeze(0).transpose(0, 1)
    return out


def get_attention_info():
    return {"calls": dict(_CALLS)}
