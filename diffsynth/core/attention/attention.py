# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# Modified by Hygon Information Technology Co., Ltd., 2026.
import os
import torch
from .h3_fa_cache import fa_cache_layers
from .registry import (
    get_attention_implementation,
    list_attention_implementations,
    register_attention_implementation,
)
from .flash_attention import CUSTOMIZED_FA_KERNEL_AVAILABLE, FLASH_ATTN_4_AVAILABLE, FLASH_ATTN_3_AVAILABLE, FLASH_ATTN_2_AVAILABLE
from .sage_attention import SAGE_ATTN_AVAILABLE
from .xformers_attention import XFORMERS_AVAILABLE
from .torch_attention import torch_sdpa, torch_sdpa_sliding_window
from .flex_attention import flex_attention


def initialize_attention_priority():
    if os.environ.get('DIFFSYNTH_ATTENTION_IMPLEMENTATION') is not None:
        return os.environ.get('DIFFSYNTH_ATTENTION_IMPLEMENTATION').strip().lower()
    else:
        for name in ["customized_fa_kernel", "flash_attention_4", "flash_attention_3", "flash_attention_2", "xformers"]:
            if name in list_attention_implementations():
                return name
        return "torch"


ATTENTION_IMPLEMENTATION = initialize_attention_priority()
_H3_FA_CACHE_ENABLED = any(layers > 0 for layers in fa_cache_layers())


def attention_forward(q, k, v, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, attn_mask=None, scale=None, is_causal=False, compatibility_mode=False, window_size=None, use_flex=False, score_mod=None):
    if use_flex or score_mod is not None:
        return flex_attention(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, attn_mask=attn_mask, scale=scale, score_mod=score_mod)

    if compatibility_mode:
        return torch_sdpa(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, attn_mask=attn_mask, scale=scale, is_causal=is_causal, window_size=window_size)

    backend = get_attention_implementation(ATTENTION_IMPLEMENTATION)

    if attn_mask is not None and not backend.supports_mask:
        return torch_sdpa(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, attn_mask=attn_mask, scale=scale, is_causal=is_causal)
    if window_size is not None and not backend.supports_window:
        assert is_causal == False and attn_mask is None
        return torch_sdpa_sliding_window(q, k, v, window_size, q_pattern, k_pattern, v_pattern, out_pattern, dims, scale=scale)
    if is_causal and not backend.supports_causal:
        return torch_sdpa(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, attn_mask=attn_mask, scale=scale, is_causal=is_causal)

    kwargs = {"scale": scale}
    if backend.supports_causal:
        kwargs["is_causal"] = is_causal
    if backend.supports_mask:
        kwargs["attn_mask"] = attn_mask
    if backend.supports_window:
        kwargs["window_size"] = window_size
    return backend.func(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, **kwargs)


def attention_varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    scale=None,
):
    """Compute independent packed attention segments without padding between them."""
    if ATTENTION_IMPLEMENTATION == "flash_attention_2":
        from .flash_attention import flash_attn
        if _H3_FA_CACHE_ENABLED:
            from .h3_fa_cache import cached_attention
            cached = cached_attention(q, k, v, cu_seqlens, max_seqlen, scale)
            if cached is not None:
                return cached
        out = flash_attn.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=scale,
            causal=False,
        )
        return out[0] if isinstance(out, tuple) else out

    out = torch.empty_like(q)
    bounds = cu_seqlens.tolist()
    for start, stop in zip(bounds[:-1], bounds[1:]):
        if stop == start:
            continue
        segment = attention_forward(
            q[start:stop].transpose(0, 1).unsqueeze(0),
            k[start:stop].transpose(0, 1).unsqueeze(0),
            v[start:stop].transpose(0, 1).unsqueeze(0),
            scale=scale,
        )
        out[start:stop] = segment.squeeze(0).transpose(0, 1)
    return out
