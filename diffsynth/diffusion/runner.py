# Modified by Hygon Information Technology Co., Ltd., 2026.
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
import importlib
import json
import os
import torch
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger
from diffsynth.core import OffloadTrainingManager
from diffsynth_hcu.rng import set_training_rng_context
from diffsynth_hcu.telemetry import PerformanceMeter, TrainingStepRecorder, maybe_enable_torch_profiler, _format_performance_metrics
from diffsynth_hcu.training import install_prepared_model_adapters, training_lifetime


def get_optimizer_class(customized_optimizer=None):
    if customized_optimizer is None:
        return torch.optim.AdamW
    else:
        module_name, class_name = customized_optimizer.rsplit(".", 1)
        module = importlib.import_module(module_name)
        print(f"Customized opimizer `{customized_optimizer}` imported.")
        return getattr(module, class_name)


def save_training_args(args):
    output_path = getattr(args, "output_path", None) if args is not None else None
    if output_path is None:
        return
    try:
        os.makedirs(args.output_path, exist_ok=True)
        save_path = os.path.join(args.output_path, "training_args.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=4, ensure_ascii=False, default=str)
        print(f"Training arguments saved to `{save_path}`.")
    except Exception as e:
        print(f"Warning: failed to save training arguments: {e}")


def exclude_quantized_params_from_ddp_sync(accelerator: Accelerator, model: DiffusionTrainingModule):
    """Skip frozen quantized tensor subclasses that DDP cannot broadcast."""
    try:
        from torch.utils._python_dispatch import is_traceable_wrapper_subclass
        quant_configs = [module.quantize_config for module in model.modules()
                         if getattr(module, "quantize_config", None) is not None]
        ignored = [
            f"{name}.weight" for name, module in model.named_modules()
            if hasattr(module, "weight")
            and any(quantize.is_quantized_linear(module) for quantize in quant_configs)
            and not module.weight.requires_grad
            and is_traceable_wrapper_subclass(module.weight)
        ]
        if ignored:
            model._ddp_params_and_buffers_to_ignore = ignored
            if accelerator.is_main_process:
                print(f"{len(ignored)} quantized weights are excluded from DDP state synchronization.")
    except Exception as error:
        print(f"Warning: failed to exclude quantized weights from DDP state synchronization: {error}")


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    enable_model_cpu_offload: bool = False,
    enable_optimizer_cpu_offload: bool = False,
    cpu_offload_split_threshold: int = None,
    customized_optimizer: str = None,
    warm_up_steps: int = 5,
    args = None,
    **kwargs,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        enable_model_cpu_offload = args.enable_model_cpu_offload
        enable_optimizer_cpu_offload = args.enable_optimizer_cpu_offload
        cpu_offload_split_threshold = args.cpu_offload_split_threshold
        customized_optimizer = args.customized_optimizer
        warm_up_steps = getattr(args, "warm_up_steps", warm_up_steps)

    if accelerator.is_main_process:
        save_training_args(args)
    limit = getattr(args, "max_train_steps", None)
    if limit is not None and limit < 1:
        raise ValueError("max_train_steps must be positive")

    with training_lifetime(accelerator, model) as lifetime:
        optimizer_class = get_optimizer_class(customized_optimizer)
        optimizer = optimizer_class(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / max(1, warm_up_steps)))
        dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)

        if enable_model_cpu_offload:
            optimizer, dataloader = accelerator.prepare(optimizer, dataloader)
            model.pipe.device = accelerator.device
            offload_manager = OffloadTrainingManager(model, accelerator.device, enable_optimizer_cpu_offload, cpu_offload_split_threshold)
        else:
            model.to(device=accelerator.device)
            exclude_quantized_params_from_ddp_sync(accelerator, model)
            model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

        initialize_deepspeed_gradient_checkpointing(accelerator)
        install_prepared_model_adapters(lifetime, accelerator, model)
        enable_metrics = bool(getattr(model_logger, "enable_wandb_log", False) or getattr(args, "enable_wandb_log", False))
        perf_meter = PerformanceMeter(accelerator, model,
            interval=getattr(args, "performance_log_interval", 10),
            peak_tflops=getattr(args, "hardware_peak_tflops", None)) if enable_metrics else None
        step_recorder = TrainingStepRecorder(accelerator, args)
        lifetime.callback(step_recorder.close)

        total_microsteps = 0
        with maybe_enable_torch_profiler(accelerator, args) as profiler:
            for epoch_id in range(num_epochs):
                progress_bar = tqdm(dataloader) if accelerator.is_main_process else dataloader
                for step_id, data in enumerate(progress_bar):
                    with accelerator.accumulate(model):
                        set_training_rng_context(accelerator, model, getattr(args, "seed", None), epoch_id, step_id)
                        step_recorder.start()
                        if perf_meter is not None:
                            perf_meter.start()
                        loss = model({}, inputs=data) if dataset.load_from_cache else model(data)
                        accelerator.backward(loss)
                        if enable_model_cpu_offload:
                            offload_manager.after_backward()
                        optimizer.step()
                        did_update = accelerator.sync_gradients and not getattr(optimizer, "step_was_skipped", False)
                        if did_update:
                            scheduler.step()
                        optimizer.zero_grad()
                        step_recorder.end(model, epoch_id, total_microsteps, loss, did_update)
                        if accelerator.is_main_process and hasattr(progress_bar, "set_postfix"):
                            progress_bar.set_postfix({"loss": f"{loss.detach().item():.4f}"})
                        metrics = perf_meter.end(model, optimizer_step=did_update) if perf_meter is not None else None
                        model_logger.on_step_end(accelerator, model, save_steps, loss=loss)
                        if metrics is not None and accelerator.is_main_process:
                            print(_format_performance_metrics(metrics))
                            model_logger.log_metrics(accelerator, metrics, model_logger.num_steps)
                    if profiler is not None:
                        profiler.step()
                    total_microsteps += 1
                    if limit is not None and total_microsteps >= limit:
                        break
                if save_steps is None:
                    model_logger.on_epoch_end(accelerator, model, epoch_id)
                if limit is not None and total_microsteps >= limit:
                    break
        # Keep optimizer and fetch bindings alive until final checkpoint save.
        model_logger.on_training_end(accelerator, model, save_steps)


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
    **kwargs,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        enable_model_cpu_offload = args.enable_model_cpu_offload
        enable_optimizer_cpu_offload = args.enable_optimizer_cpu_offload
        cpu_offload_split_threshold = args.cpu_offload_split_threshold
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    if enable_model_cpu_offload:
        dataloader = accelerator.prepare(dataloader)
        offload_manager = OffloadTrainingManager(model, accelerator.device, enable_optimizer_cpu_offload, cpu_offload_split_threshold)
        model.pipe.device = accelerator.device
    else:
        model.to(device=accelerator.device)
        exclude_quantized_params_from_ddp_sync(accelerator, model)
        model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)
                if enable_model_cpu_offload:
                    offload_manager.after_backward()

def initialize_deepspeed_gradient_checkpointing(accelerator: Accelerator):
    if getattr(accelerator.state, "deepspeed_plugin", None) is not None:
        ds_config = accelerator.state.deepspeed_plugin.deepspeed_config
        if "activation_checkpointing" in ds_config:
            import deepspeed
            act_config = ds_config["activation_checkpointing"]
            deepspeed.checkpointing.configure(
                mpu_=None,
                partition_activations=act_config.get("partition_activations", False),
                checkpoint_in_cpu=act_config.get("cpu_checkpointing", False),
                contiguous_checkpointing=act_config.get("contiguous_memory_optimization", False)
            )
        else:
            deepspeed_plugin = accelerator.state.deepspeed_plugin
            zero_stage = ds_config.get("zero_optimization", {}).get(
                "stage",
                getattr(deepspeed_plugin, "zero_stage", None),
            )
            if torch.version.hip is None or zero_stage != 3:
                print("Do not find activation_checkpointing config in deepspeed config, skip initializing deepspeed gradient checkpointing.")
                return
            # HCU only: Accelerate's shorthand DeepSpeed config drops unknown
            # nested keys, so the HCU launch scripts carry no
            # activation_checkpointing section. Without configure(),
            # gradient_checkpoint_forward falls back to torch's non-reentrant
            # checkpoint, whose recompute runs while ZeRO-3 parameters are
            # partitioned (numel 0) and fails with a metadata mismatch. Route
            # recompute through DeepSpeed's reentrant implementation instead.
            import deepspeed
            deepspeed.checkpointing.configure(mpu_=None)
