# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""HIP ZeRO-3 setup and validated single-parameter LoRA fetch arithmetic."""
from collections import Counter
import torch


def configure_deepspeed_hip(accelerator):
    plugin = getattr(accelerator.state, "deepspeed_plugin", None)
    if plugin is None or torch.version.hip is None:
        return
    zero = plugin.deepspeed_config.get("zero_optimization", {})
    if zero.get("stage", getattr(plugin, "zero_stage", None)) == 3:
        zero["overlap_comm"] = False


class FetchHandle:
    def __init__(self, originals, device, calls):
        self.originals, self.device, self.calls = originals, device, calls

    def close(self):
        if not self.originals:
            return
        self.device.synchronize()
        for param, present, previous in reversed(self.originals):
            if present:
                param.all_gather_coalesced = previous
            elif "all_gather_coalesced" in vars(param):
                # Installation records the parameter before assigning the
                # wrapper.  If a later assignment fails, cleanup must remain
                # exception-safe and must not delete an inherited method.
                delattr(param, "all_gather_coalesced")
        self.originals.clear()


def install_zero3_lora_fetch(accelerator, model):
    plugin = getattr(accelerator.state, "deepspeed_plugin", None)
    if plugin is None or torch.version.hip is None:
        return None
    zero = plugin.deepspeed_config.get("zero_optimization", {})
    if zero.get("stage", getattr(plugin, "zero_stage", None)) != 3:
        return None
    candidates = [(name, param) for name, param in model.named_parameters()
                  if param.requires_grad and (".lora_A." in name or ".lora_B." in name)
                  and hasattr(param, "all_gather_coalesced")]
    if not candidates:
        return None
    if zero.get("overlap_comm") is not False:
        raise RuntimeError("HIP ZeRO-3 LoRA fetch requires overlap_comm=false before prepare")
    from deepspeed import comm as dist
    from deepspeed.accelerator import get_accelerator
    from deepspeed.runtime.zero.partition_parameters import AllReduceCoalescedHandle, ZeroParamStatus
    from deepspeed.runtime.swap_tensor.partitioned_param_swapper import PartitionedParamStatus

    calls = Counter()
    def wrap_all_gather_coalesced(original, param_name):
        def all_gather_coalesced(params, safe_mode=False, quantize=False):
            params = list(params)
            if (
                len(params) != 1
                or safe_mode
                or quantize
                or params[0].ds_secondary_tensor is not None
            ):
                calls["original_fetch"] += 1
                return original(params, safe_mode=safe_mode, quantize=quantize)

            param = params[0]
            if param.ds_tensor.status != PartitionedParamStatus.AVAILABLE:
                calls["original_fetch"] += 1
                return original(params, safe_mode=safe_mode, quantize=quantize)
            if param.ds_status != ZeroParamStatus.NOT_AVAILABLE:
                raise RuntimeError(
                    f"Expected LoRA parameter {param_name} to be NOT_AVAILABLE before fetch, "
                    f"got {param.ds_summary()}"
                )

            process_group = param.ds_process_group
            partition_rank = dist.get_rank(group=process_group)
            world_size = dist.get_world_size(group=process_group)
            expected_aligned_numel = param.ds_tensor.ds_numel * world_size
            if param.ds_numel_aligned != expected_aligned_numel:
                raise RuntimeError(
                    f"Unexpected ZeRO partition layout for LoRA parameter {param_name}: "
                    f"ds_numel_aligned={param.ds_numel_aligned}, "
                    f"partition_numel={param.ds_tensor.ds_numel}, world_size={world_size}"
                )

            param.ds_status = ZeroParamStatus.INFLIGHT
            flat_tensor = torch.zeros(
                param.ds_numel_aligned,
                dtype=param.ds_tensor.dtype,
                device=get_accelerator().current_device_name(),
                requires_grad=False,
            )
            param.data = flat_tensor.narrow(0, 0, param.ds_numel).view(param.ds_shape)
            partition_start = param.ds_tensor.ds_numel * partition_rank
            flat_tensor.narrow(0, partition_start, param.ds_tensor.ds_numel).copy_(param.ds_tensor)
            work = dist.all_reduce(flat_tensor, group=process_group, async_op=True)
            calls["all_reduce_fast_path"] += 1
            return AllReduceCoalescedHandle(handle=work, params=params)

        all_gather_coalesced._hcu_lora_fetch_wrapper = True
        return all_gather_coalesced

    originals = []
    try:
        for name, param in candidates:
            original = param.all_gather_coalesced
            if getattr(original, "_hcu_lora_fetch_wrapper", False):
                raise RuntimeError(f"HIP ZeRO-3 LoRA fetch already installed on {name}")
            originals.append((param, "all_gather_coalesced" in vars(param), vars(param).get("all_gather_coalesced")))
            param.all_gather_coalesced = wrap_all_gather_coalesced(original, name)
    except BaseException:
        FetchHandle(originals, get_accelerator(), calls).close()
        raise
    if accelerator.is_main_process:
        print(f"HIP ZeRO-3 all-reduce fetch installed on {len(originals)} LoRA parameters", flush=True)
    return FetchHandle(originals, get_accelerator(), calls)
