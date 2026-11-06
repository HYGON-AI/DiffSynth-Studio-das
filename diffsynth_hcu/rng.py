# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# Extracted from the reviewed legacy baseline db63d0476ad1ee4bf57a1a44d830a33f2c14a4f7.
from __future__ import annotations
import torch

_STEP_SEED_MODULUS = 2**63 - 1


def set_training_rng_context(accelerator, model, seed, epoch_id, step_id):
    pipe = getattr(accelerator.unwrap_model(model), "pipe", None)
    if pipe is None:
        return
    if seed is None:
        if hasattr(pipe, "_diffsynth_training_rng_context"):
            delattr(pipe, "_diffsynth_training_rng_context")
        return
    pipe._diffsynth_training_rng_context = {
        "seed": int(seed), "epoch": int(epoch_id), "step": int(step_id),
        "rank": int(accelerator.process_index),
    }


def get_training_rng_context(pipe):
    """Return the runner-provided deterministic loss RNG context, if enabled."""
    context = getattr(pipe, "_diffsynth_training_rng_context", None)
    if context is None:
        return None
    return (
        int(context["seed"]),
        int(context["epoch"]),
        int(context["step"]),
        int(context["rank"]),
    )


def _derive_step_seed(context, stream):
    """Derive independent, stable seeds for timestep/video/audio streams."""
    seed, epoch, step, rank = context
    return (
        seed
        + 1_000_003 * epoch
        + 1_000_033 * step
        + 1_000_037 * rank
        + 1_000_081 * stream
    ) % _STEP_SEED_MODULUS


def sample_timestep_id(min_timestep_boundary, max_timestep_boundary, context=None):
    if context is None:
        return torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    generator = torch.Generator(device="cpu").manual_seed(_derive_step_seed(context, stream=0))
    return torch.randint(
        min_timestep_boundary,
        max_timestep_boundary,
        (1,),
        generator=generator,
        device="cpu",
    )


def sample_noise_like(tensor, context=None, stream=1):
    if context is None:
        return torch.randn_like(tensor)
    # Generate on CPU so the seeded stream does not depend on CUDA/HCU PRNG
    # implementation details. The tensor is small relative to the DiT forward,
    # and the cast/transfer happens before scheduler arithmetic.
    generator = torch.Generator(device="cpu").manual_seed(_derive_step_seed(context, stream))
    noise = torch.randn(tensor.shape, generator=generator, device="cpu", dtype=torch.float32)
    return noise.to(device=tensor.device, dtype=tensor.dtype)
