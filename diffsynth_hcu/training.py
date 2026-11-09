# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
from contextlib import contextmanager, ExitStack

from .cpu_adam import create_cpu_adam_pipeline
from .deepspeed_hip import configure_deepspeed_hip, install_zero3_lora_fetch
from .lora import install_lora_fast_paths


@contextmanager
def training_lifetime(accelerator, model):
    """Install pre-prepare adapters and keep them active through final save."""
    with ExitStack() as lifetime:
        configure_deepspeed_hip(accelerator)
        lora_adapter = install_lora_fast_paths(model)
        if lora_adapter is not None:
            lifetime.callback(lora_adapter.close)
        yield lifetime


def install_prepared_model_adapters(lifetime, accelerator, model):
    """Bind adapters that need the prepared DeepSpeed model."""
    fetch_handle = install_zero3_lora_fetch(accelerator, model)
    if fetch_handle is not None:
        lifetime.callback(fetch_handle.close)
    cpu_pipeline = create_cpu_adam_pipeline(model)
    if cpu_pipeline is not None:
        lifetime.callback(cpu_pipeline.close)
