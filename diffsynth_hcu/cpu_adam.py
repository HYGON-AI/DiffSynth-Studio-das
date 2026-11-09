# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Fused CPUAdam with the validated ZeRO-3 asynchronous writeback path."""
import inspect
import os
import sys
from pathlib import Path

import torch


def load_fused_cpu_adam():
    import deepspeed
    if deepspeed.__version__.split("+")[0] != "0.18.4":
        raise RuntimeError("Vendored fused CPUAdam arithmetic requires the validated DeepSpeed 0.18.4 source contract")
    from deepspeed.ops.op_builder import CPUAdamBuilder
    from torch.utils.cpp_extension import load
    root = Path(__file__).resolve().parent / "ops/csrc/cpu_adam"
    builder = CPUAdamBuilder()
    if not all(callable(getattr(builder, name, None)) for name in ("cxx_args", "extra_ldflags")):
        raise RuntimeError("Installed DeepSpeed CPUAdamBuilder lacks the required build interface")
    builder.build_for_cpu = True
    return load(
        name="diffsynth_cpu_adam_fused_v1",
        sources=[str(root / n) for n in ("cpu_adam.cpp", "cpu_adam_impl.cpp")],
        extra_include_paths=[str(root)],
        extra_cflags=["-O3", "-std=c++17"] + [arg for arg in builder.cxx_args() if arg],
        extra_ldflags=builder.extra_ldflags(), with_cuda=False,
        verbose=False,
    )


class _AdamProxy:
    def __init__(self, original, owner):
        self.original, self.owner = original, owner

    def __getattr__(self, name):
        return getattr(self.original, name)

    def adam_update(self, *args):
        active = self.owner.active
        if active is None:
            return self.original.adam_update(*args)
        output, scale = active
        return self.owner.native.adam_update_fused(*args, output, scale)


class CPUAdamPipeline:
    """Bounded pinned-host ring; no extra GPU parameter staging allocation.

    Vendored DeepSpeed SIMD Adam arithmetic, with scaling/conversion in-loop.
    """
    def __init__(self, zero):
        import deepspeed
        from deepspeed.ops.adam import DeepSpeedCPUAdam
        from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
        if any(getattr(getattr(zero, name, None), "_hcu_cpu_pipeline_wrapper", False)
               for name in ("step", "_optimizer_step")):
            raise RuntimeError("A CPUAdam pipeline already owns this optimizer")
        if type(zero) is not DeepSpeedZeroOptimizer_Stage3 or type(zero.optimizer) is not DeepSpeedCPUAdam:
            raise RuntimeError("CPU pipeline requires ZeRO-3 with DeepSpeedCPUAdam")
        if (not zero.offload_optimizer or zero.swap_optimizer or zero.offload_param
                or getattr(zero, "torch_autocast_gradscaler", None) or getattr(zero, "zenflow", False)):
            raise RuntimeError("CPU pipeline requires CPU optimizer offload, GPU params, no NVMe/GradScaler/ZenFlow")
        if not zero.optimizer.fp32_optimizer_states:
            raise RuntimeError("CPU pipeline requires FP32 optimizer states")
        self.zero = zero
        self.originals = {}
        self.pending = {}
        self.active = None
        self.native = None
        self.old_adam = None
        self._native_created = False
        self.stats = dict(cpu_subgroups=0, h2d_bytes=0, steps=0)
        self.cpu_ids = []
        devices = set()
        for i, p in enumerate(zero.fp32_partitioned_groups_flat):
            if p.device.type != "cpu":
                continue
            dst = zero.fp16_partitioned_groups_flat[i]
            if (p.dtype != torch.float32 or not p.is_contiguous() or dst is None
                    or dst.device.type != "cuda" or dst.dtype != torch.bfloat16
                    or not dst.is_contiguous() or dst.numel() != p.numel()):
                raise RuntimeError(f"Unsupported CPU subgroup {i}: requires flat FP32 CPU -> BF16 GPU")
            self.cpu_ids.append(i)
            devices.add(dst.device)
        if not self.cpu_ids or len(devices) != 1:
            raise RuntimeError("Expected CPU subgroups targeting one GPU per rank")
        required = {"unscale_and_clip_grads": ("sub_group_id", "total_norm"),
                    "_optimizer_step": ("sub_group_id",),
                    "_reassign_or_swap_out_partitioned_parameters": ("sub_group_id",),
                    "_post_step": ("timer_names",), "step": ()}
        for name, params in required.items():
            sig = inspect.signature(getattr(zero, name))
            # DeepSpeed's NVTX decorator exposes (*args, **kwargs) instead of
            # preserving the wrapped method's named parameters.
            generic_wrapper = (any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())
                               and any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()))
            if not generic_wrapper and any(p not in sig.parameters for p in params):
                raise RuntimeError(f"Unsupported DeepSpeed interface: {name}{sig}")
        self.device = next(iter(devices))
        size = max(zero.fp32_partitioned_groups_flat[i].numel() for i in self.cpu_ids)
        self.buffers = [torch.empty(size, dtype=torch.bfloat16, pin_memory=True) for _ in range(2)]
        self.events = [None, None]
        self.cursor = 0
        self.stream = torch.cuda.Stream(device=self.device)
        try:
            self.native = load_fused_cpu_adam()
            opt = zero.optimizer
            g = opt.param_groups[0]
            self.native.create_adam(opt.opt_id, g["lr"], *g["betas"], g["eps"],
                                    g["weight_decay"], opt.adam_w_mode, False)
            self._native_created = True
            self.old_adam = opt.ds_opt_adam
            opt.ds_opt_adam = _AdamProxy(self.old_adam, self)
            self._install()
        except BaseException:
            self.close()
            raise
        print(f"CPUAdam pipeline enabled: fused, CPU subgroups={len(self.cpu_ids)}, "
              f"pinned ring bytes={size * 4}, DeepSpeed={deepspeed.__version__}", flush=True)

    def _patch(self, name, wrapper):
        wrapper._hcu_cpu_pipeline_wrapper = True
        wrapper._hcu_cpu_pipeline_owner = self
        if not hasattr(self, "_prior_bindings"):
            self._prior_bindings = {}
        self._prior_bindings[name] = getattr(self.zero, name)
        self.originals[name] = (name in vars(self.zero), vars(self.zero).get(name))
        setattr(self.zero, name, wrapper)

    def _install(self):
        z = self.zero
        unscale, update = z.unscale_and_clip_grads, z._optimizer_step
        reassign, post, step = z._reassign_or_swap_out_partitioned_parameters, z._post_step, z.step

        def scaled(sub_group_id, total_norm):
            if sub_group_id not in self.cpu_ids:
                return unscale(sub_group_id, total_norm)
            p = z.fp32_partitioned_groups_flat[sub_group_id]
            if p.grad is None or p.grad.device.type != "cpu" or p.grad.dtype != torch.float32 or not p.grad.is_contiguous():
                raise RuntimeError("Fused CPUAdam requires a contiguous CPU FP32 gradient")
            combined_scale = z.loss_scale
            if z.clip_grad > 0.:
                clip = ((total_norm / z.loss_scale) + 1e-6) / z.clip_grad
                combined_scale = torch.clamp(clip, min=1.0) * z.loss_scale
            self.pending[sub_group_id] = float(1. / combined_scale)

        def updated(sub_group_id):
            if sub_group_id not in self.cpu_ids:
                return update(sub_group_id)
            slot = self.cursor % 2
            self.cursor += 1
            if self.events[slot] is not None:
                self.events[slot].synchronize()  # Do not overwrite a DMA source.
            p = z.fp32_partitioned_groups_flat[sub_group_id]
            output = self.buffers[slot][:p.numel()]
            self.active = (output, self.pending.pop(sub_group_id))
            try:
                result = update(sub_group_id)  # Keep Python state/step/checkpoint semantics.
            finally:
                self.active = None
            self.pending[sub_group_id] = (slot, output)
            return result

        def reassigned(sub_group_id):
            if sub_group_id not in self.cpu_ids:
                return reassign(sub_group_id)
            slot, output = self.pending.pop(sub_group_id)
            dst = z.fp16_partitioned_groups_flat[sub_group_id].detach()
            self.stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(self.stream):
                dst.copy_(output, non_blocking=True)
                event = torch.cuda.Event()
                event.record(self.stream)
                dst.record_stream(self.stream)
            self.events[slot] = event
            z._unflatten_partitioned_parameters(sub_group_id)
            self.stats["cpu_subgroups"] += 1
            self.stats["h2d_bytes"] += dst.numel() * dst.element_size()

        def posted(*args, **kwargs):
            # ProcessGroupNCCL may use an internal stream: establish dependency
            # on its submitting stream before persistent parameter all-gather.
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
            return post(*args, **kwargs)

        def stepped(*args, **kwargs):
            try:
                result = step(*args, **kwargs)
                self.stats["steps"] += 1
                if self.pending:
                    raise RuntimeError("CPU pipeline left an unfinished subgroup")
                if self.stats["steps"] == 1:
                    print(f"CPUAdam pipeline rank={os.environ.get('RANK', '0')} {self.stats}", flush=True)
                return result
            except BaseException:
                self.stream.synchronize()
                self.pending.clear()
                raise

        for name, fn in (("unscale_and_clip_grads", scaled), ("_optimizer_step", updated),
                         ("_reassign_or_swap_out_partitioned_parameters", reassigned),
                         ("_post_step", posted), ("step", stepped)):
            self._patch(name, fn)

    def close(self):
        self.stream.synchronize()
        if self.native is not None and self.old_adam is not None:
            proxy = self.zero.optimizer.ds_opt_adam
            if not isinstance(proxy, _AdamProxy) or proxy.owner is not self or proxy.original is not self.old_adam:
                raise RuntimeError("CPUAdam native backend binding was overwritten")
        # Check ownership before restoring anything; do not overwrite another
        # adapter installed after this one. Failed setattr can leave originals.
        for name, (present, previous) in self.originals.items():
            value = getattr(self.zero, name)
            original = self._prior_bindings[name]
            same = value is original or (inspect.ismethod(value) and inspect.ismethod(original)
                and value.__func__ is original.__func__ and value.__self__ is original.__self__)
            if not (getattr(value, "_hcu_cpu_pipeline_owner", None) is self or same):
                raise RuntimeError(f"CPUAdam pipeline binding was overwritten: {name}")
        for name, (present, previous) in reversed(list(self.originals.items())):
            if present:
                setattr(self.zero, name, previous)
            elif name in vars(self.zero):
                delattr(self.zero, name)
        self.originals.clear()
        if self.native is not None:
            if self.old_adam is not None:
                self.zero.optimizer.ds_opt_adam = self.old_adam
            if self._native_created:
                self.native.destroy_adam(self.zero.optimizer.opt_id)
            self.native = None


def create_cpu_adam_pipeline(model):
    """Create the validated fused path when the prepared model is eligible."""
    if torch.version.hip is None or "deepspeed" not in sys.modules:
        return None
    from deepspeed.ops.adam import DeepSpeedCPUAdam
    from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
    zero = getattr(model, "optimizer", None)
    if type(zero) is not DeepSpeedZeroOptimizer_Stage3 or type(zero.optimizer) is not DeepSpeedCPUAdam:
        return None
    if (not zero.offload_optimizer or zero.swap_optimizer or zero.offload_param
            or getattr(zero, "torch_autocast_gradscaler", None)
            or getattr(zero, "zenflow", False) or not zero.optimizer.fp32_optimizer_states):
        return None
    cpu_groups = [(master, zero.fp16_partitioned_groups_flat[i])
                  for i, master in enumerate(zero.fp32_partitioned_groups_flat)
                  if master.device.type == "cpu"]
    if not cpu_groups or any(
        master.dtype != torch.float32 or not master.is_contiguous()
        or shard is None or shard.device.type != "cuda" or shard.dtype != torch.bfloat16
        or not shard.is_contiguous() or shard.numel() != master.numel()
        for master, shard in cpu_groups
    ):
        return None
    if len({shard.device for _, shard in cpu_groups}) != 1:
        return None
    return CPUAdamPipeline(zero)
