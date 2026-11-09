# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Two-stage indexed row sum for H3 modulation gradients (no atomics).

Each first-stage program owns one target row and token/channel tile. The
second stage sums partials in FP32, then rounds once to the modulation dtype.
Indices need not be sorted and target rows may be empty.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _partial_sum(Values, Indices, Partial, N: tl.constexpr, C: tl.constexpr,
                 SPLITS: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr):
    split = tl.program_id(0)
    channel = tl.program_id(1) * BC + tl.arange(0, BC)
    row = tl.program_id(2)
    token = split * BT + tl.arange(0, BT)
    index = tl.load(Indices + token, token < N, other=-1)
    values = tl.load(Values + token[:, None] * C + channel[None, :],
                     (token[:, None] < N) & (channel[None, :] < C)
                     & (index[:, None] == row), other=0).to(tl.float32)
    total = tl.sum(values, axis=0)
    tl.store(Partial + (row * SPLITS + split) * C + channel, total, channel < C)


@triton.jit
def _finish_sum(Partial, Output, C: tl.constexpr, SPLITS: tl.constexpr,
                BS: tl.constexpr, BC: tl.constexpr):
    channel = tl.program_id(0) * BC + tl.arange(0, BC)
    row = tl.program_id(1)
    split = tl.arange(0, BS)
    values = tl.load(Partial + (row * SPLITS + split[:, None]) * C + channel[None, :],
                     (split[:, None] < SPLITS) & (channel[None, :] < C), other=0)
    tl.store(Output + row * C + channel, tl.sum(values, axis=0), channel < C)


def indexed_row_sum(values, indices, rows):
    """Sum [tokens, channels] into [rows, channels] on the current stream."""
    n, channels = values.shape
    if n == 0:
        return values.new_zeros((rows, channels))
    values = values.contiguous()
    indices = indices.contiguous()
    block_tokens, block_channels = 256, 32
    splits = triton.cdiv(n, block_tokens)
    # Storage is private to this invocation; no cross-stream/global scratch.
    with torch.cuda.device(values.device):
        partial = torch.empty((rows, splits, channels), device=values.device, dtype=torch.float32)
        output = values.new_empty((rows, channels))
        _partial_sum[(splits, triton.cdiv(channels, block_channels), rows)](
            values, indices, partial, n, channels, splits, block_tokens, block_channels,
            num_warps=4)
        _finish_sum[(triton.cdiv(channels, block_channels), rows)](
            partial, output, channels, splits, triton.next_power_of_2(splits), block_channels,
            num_warps=4)
    return output
