# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# Extracted from the reviewed legacy baseline db63d0476ad1ee4bf57a1a44d830a33f2c14a4f7.
from __future__ import annotations
import contextlib
import csv
import os
import time
import torch


class TrainingStepRecorder:
    """Rank-0 numerical/timing evidence when ordinary CSV logging is requested."""

    def __init__(self, accelerator, args):
        self.file = None
        self.device = accelerator.device
        self.optimizer_steps = 0
        if not accelerator.is_main_process or not getattr(args, "enable_csv_log", False):
            return
        path = os.path.join(args.output_path, "training_steps.csv")
        os.makedirs(args.output_path, exist_ok=True)
        exists = os.path.exists(path) and os.path.getsize(path) > 0
        self.file = open(path, "a", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=[
            "epoch", "micro_step", "optimizer_step", "did_update", "loss",
            "global_grad_norm", "elapsed_s", "peak_allocated_bytes", "peak_reserved_bytes",
        ])
        if not exists:
            self.writer.writeheader()

    def _synchronize(self):
        if torch.device(self.device).type == "cuda":
            torch.cuda.synchronize(self.device)

    def start(self):
        if self.file is not None:
            self._synchronize()
            self.started = time.perf_counter()

    def end(self, model, epoch, micro_step, loss, did_update):
        if self.file is None:
            return
        self._synchronize()
        elapsed = time.perf_counter() - self.started
        self.optimizer_steps += int(did_update)
        get_norm = getattr(model, "get_global_grad_norm", None)
        norm = get_norm() if did_update and callable(get_norm) else None
        gpu = torch.device(self.device).type == "cuda"
        self.writer.writerow(dict(
            epoch=epoch, micro_step=micro_step, optimizer_step=self.optimizer_steps,
            did_update=int(did_update), loss=loss.detach().float().mean().item(),
            global_grad_norm="" if norm is None else float(norm), elapsed_s=elapsed,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device) if gpu else 0,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device) if gpu else 0,
        ))
        self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()


@contextlib.contextmanager
def maybe_enable_torch_profiler(accelerator, args):
    if args is None or not getattr(args, "enable_torch_profiler", False):
        yield None
        return

    profile_freq = getattr(args, "profiler_freq", None)
    warmup = getattr(args, "profiler_warmup", 1)
    active = getattr(args, "profiler_active", 1)
    repeat = getattr(args, "profiler_repeat", 1)
    if profile_freq is None:
        wait = getattr(args, "profiler_wait", 0)
    else:
        wait = profile_freq - warmup - active
        if profile_freq < 1 or wait < 0:
            raise ValueError("Profiler frequency must be >= warmup + active and at least 1.")
    if wait < 0 or warmup < 0 or active < 1 or repeat < 1:
        raise ValueError("Profiler schedule requires wait/warmup >= 0 and active/repeat >= 1.")

    ranks_arg = getattr(args, "profiler_target_ranks", "0").strip().lower()
    if ranks_arg == "all":
        target_ranks = set(range(accelerator.num_processes))
    else:
        try:
            target_ranks = {int(rank.strip()) for rank in ranks_arg.split(",") if rank.strip()}
        except ValueError as e:
            raise ValueError("--profiler_target_ranks must be comma-separated integers or `all`.") from e
        invalid_ranks = {rank for rank in target_ranks if rank < 0 or rank >= accelerator.num_processes}
        if not target_ranks or invalid_ranks:
            raise ValueError(f"Invalid profiler target ranks: {sorted(invalid_ranks or target_ranks)}")

    trace_root = getattr(args, "profiler_output_path", None) or os.path.join(args.output_path, "torch_trace")
    os.makedirs(trace_root, exist_ok=True)
    rank = accelerator.process_index
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    def trace_handler(prof):
        if rank not in target_ranks:
            return
        trace_dir = os.path.join(trace_root, f"iteration_{prof.step_num}")
        os.makedirs(trace_dir, exist_ok=True)
        trace_path = os.path.join(trace_dir, f"rank{rank}_trace.json.gz")
        begin = time.monotonic()
        prof.export_chrome_trace(trace_path)
        print(f"Rank {rank} exported profiler trace to `{trace_path}` in {time.monotonic() - begin:.2f}s.")

    if accelerator.is_main_process:
        schedule_summary = (
            f"freq={profile_freq}, wait={wait}" if profile_freq is not None else f"wait={wait}"
        )
        print(
            f"PyTorch profiler enabled: {schedule_summary}, warmup={warmup}, "
            f"active={active}, repeat={repeat}, traces=`{trace_root}`."
        )

    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
        on_trace_ready=trace_handler,
        record_shapes=getattr(args, "profiler_record_shapes", False),
        profile_memory=getattr(args, "profiler_profile_memory", False),
        with_stack=getattr(args, "profiler_with_stack", False),
        with_modules=getattr(args, "profiler_with_modules", False),
    ) as profiler:
        yield profiler


class PerformanceMeter:
    """Windowed training throughput and model-FLOPs estimates."""

    def __init__(self, accelerator, model, interval=10, peak_tflops=None):
        self.accelerator = accelerator
        self.interval = max(1, int(interval))
        self.peak_tflops = peak_tflops
        # Only rank 0 measures. Every rank running a device synchronize twice
        # per micro-step stalls CPU run-ahead and serializes the ZeRO-3
        # communication overlap, which costs far more than the telemetry is
        # worth. Ranks are already lock-stepped by the optimizer collectives,
        # so rank 0's window is representative.
        self.enabled = bool(accelerator.is_main_process)
        self.optimizer_steps = 0
        self.micro_steps = 0
        self.logical_tokens = 0
        self.tokens_valid = True
        self._window_start = None
        if not self.enabled:
            self.num_params = 0
            self.recomputed_params = 0
            self.flops_per_token = 0
            return
        unwrapped = accelerator.unwrap_model(model)
        dit = getattr(getattr(unwrapped, "pipe", None), "dit", None)
        self.num_params = self._parameter_count(dit)
        self.recomputed_params = self._recomputed_parameter_count(unwrapped, dit)
        self.flops_per_token = 6 * self.num_params + 2 * self.recomputed_params

    @staticmethod
    def _parameter_count(module):
        if module is None:
            return 0
        total = 0
        for name, parameter in module.named_parameters():
            if "lora_" in name.lower():
                continue
            total += int(parameter.ds_numel if hasattr(parameter, "ds_numel") else parameter.numel())
        return total

    def _recomputed_parameter_count(self, model, dit):
        if not getattr(model, "use_gradient_checkpointing", False) or dit is None:
            return 0
        blocks = getattr(dit, "blocks", None)
        if blocks is None:
            return self.num_params
        return sum(self._parameter_count(block) for block in blocks)

    def start(self):
        # No device synchronize: the window spans many steps, and rank 0 is
        # bracketed each step by the `loss.item()` device-to-host copy, so
        # per-step queuing noise averages out over the window.
        if self.enabled and self._window_start is None:
            self._window_start = time.perf_counter()

    def end(self, model, optimizer_step=True):
        if not self.enabled or self._window_start is None:
            return None
        logical_tokens = int(getattr(self.accelerator.unwrap_model(model), "last_perf_tokens", 0))
        self.micro_steps += 1
        if logical_tokens > 0:
            self.logical_tokens += logical_tokens
        else:
            self.tokens_valid = False
        if not optimizer_step:
            return None
        self.optimizer_steps += 1
        if self.optimizer_steps < self.interval:
            return None

        window_elapsed = time.perf_counter() - self._window_start
        if window_elapsed <= 0:
            self._reset_window()
            return None

        # Rank 0's own device, extrapolated to the cluster. No collectives.
        world_size = self.accelerator.num_processes
        result = {
            "perf/step_time_s": window_elapsed / self.optimizer_steps,
            "perf/samples_per_sec": world_size * self.micro_steps / window_elapsed,
        }
        if self.tokens_valid and self.flops_per_token > 0:
            tokens_per_sec_per_gpu = self.logical_tokens / window_elapsed
            result["perf/logical_tokens_per_sec"] = tokens_per_sec_per_gpu * world_size
            result["perf/tokens_per_sec_per_gpu"] = tokens_per_sec_per_gpu
            estimated_tflops = self.flops_per_token * tokens_per_sec_per_gpu / 1e12
            result["perf/estimated_tflops_per_gpu"] = estimated_tflops
            if self.peak_tflops and self.peak_tflops > 0:
                result["perf/mfu"] = estimated_tflops / self.peak_tflops

        self._reset_window()
        return result

    def _reset_window(self):
        self.optimizer_steps = 0
        self.micro_steps = 0
        self.logical_tokens = 0
        self.tokens_valid = True
        self._window_start = time.perf_counter()


def _format_performance_metrics(metrics):
    fields = [
        f"{metrics['perf/step_time_s']:.3f}s/step",
        f"{metrics['perf/samples_per_sec']:.3f} samples/s",
    ]
    tflops = metrics.get("perf/estimated_tflops_per_gpu")
    if tflops is not None:
        fields.append(f"{tflops:.1f} TFLOPS/GPU")
    mfu = metrics.get("perf/mfu")
    if mfu is not None:
        fields.append(f"MFU {mfu:.1%}")
    return "Performance: " + ", ".join(fields)
