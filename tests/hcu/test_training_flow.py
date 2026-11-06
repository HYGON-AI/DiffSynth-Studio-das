"""CPU/mock checks of the actual launcher function without target dependencies."""
import ast
import contextlib
from pathlib import Path
import unittest

import torch

from diffsynth_hcu.rng import set_training_rng_context
from diffsynth_hcu.telemetry import TrainingStepRecorder


def launcher(events, *, prepare_error=False, skipped_second_update=False):
    source = (Path(__file__).resolve().parents[2] / "diffsynth/diffusion/runner.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "launch_training_task")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function], type_ignores=[])
    ast.fix_missing_locations(module)

    class Handle:
        def __init__(self, name): self.name = name
        def close(self): events.append("close:" + self.name)

    @contextlib.contextmanager
    def profiler(accelerator, args):
        yield None

    namespace = dict(
        torch=torch, contextlib=contextlib, tqdm=lambda iterator: iterator,
        save_training_args=lambda args: None,
        get_optimizer_class=lambda _: torch.optim.SGD,
        exclude_quantized_params_from_ddp_sync=lambda *args: None,
        set_training_rng_context=set_training_rng_context,
        configure_deepspeed_hip=lambda *args: events.append("configure"),
        initialize_deepspeed_gradient_checkpointing=lambda *args: events.append("checkpoint"),
        install_lora_fast_paths=lambda *args: Handle("lora"),
        install_zero3_lora_fetch=lambda *args: Handle("fetch"),
        create_cpu_adam_pipeline=lambda *args: Handle("cpuadam"),
        maybe_enable_torch_profiler=profiler, TrainingStepRecorder=TrainingStepRecorder,
    )
    exec(compile(module, "<actual launch_training_task>", "exec"), namespace)

    class Accelerator:
        is_main_process = True
        process_index = 0
        device = "cpu"
        sync_gradients = False

        def __init__(self):
            self.microsteps = 0
            self.rates = []

        def unwrap_model(self, model): return model

        def prepare(self, model, optimizer, dataloader):
            events.append("prepare")
            if prepare_error:
                raise RuntimeError("prepare sentinel")
            accelerator = self
            class WrappedOptimizer:
                step_was_skipped = False
                def step(self):
                    self.step_was_skipped = skipped_second_update and accelerator.microsteps == 4
                    if accelerator.sync_gradients and not self.step_was_skipped:
                        accelerator.rates.append(optimizer.param_groups[0]["lr"])
                        return optimizer.step()
                def zero_grad(self):
                    if accelerator.sync_gradients:
                        optimizer.zero_grad()
            return model, WrappedOptimizer(), dataloader

        @contextlib.contextmanager
        def accumulate(self, model):
            self.microsteps += 1
            self.sync_gradients = self.microsteps % 2 == 0
            yield

        def backward(self, loss): (loss / 2).backward()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
        def trainable_modules(self): return self.parameters()
        def forward(self, data): return self.weight.square() * data

    class Dataset(list): load_from_cache = False

    class Logger:
        enable_wandb_log = False
        num_steps = 0
        def on_step_end(self, *args, **kwargs): self.num_steps += 1
        def on_epoch_end(self, *args): events.append("epoch_save")
        def on_training_end(self, *args): events.append("final_save")

    accelerator = Accelerator()
    return namespace["launch_training_task"], accelerator, Dataset([torch.tensor(1.0)] * 4), Model(), Logger()


class TrainingFlowTests(unittest.TestCase):
    def test_accumulation_two_preserves_upstream_warmup_and_save_lifetime(self):
        events = []
        run, accelerator, dataset, model, logger = launcher(events)
        run(accelerator, dataset, model, logger, learning_rate=0.1, warm_up_steps=4, num_workers=0)
        self.assertEqual(accelerator.rates, [0.025, 0.05])
        self.assertEqual(logger.num_steps, 4)
        self.assertLess(events.index("configure"), events.index("prepare"))
        self.assertLess(events.index("final_save"), events.index("close:cpuadam"))
        self.assertEqual(events[-3:], ["close:cpuadam", "close:fetch", "close:lora"])

    def test_prepare_failure_restores_installed_lora(self):
        events = []
        run, accelerator, dataset, model, logger = launcher(events, prepare_error=True)
        with self.assertRaisesRegex(RuntimeError, "prepare sentinel"):
            run(accelerator, dataset, model, logger, num_workers=0)
        self.assertEqual(events[-1], "close:lora")

    def test_skipped_optimizer_update_does_not_advance_warmup(self):
        events = []
        run, accelerator, dataset, model, logger = launcher(events, skipped_second_update=True)
        run(accelerator, dataset, model, logger, learning_rate=0.1, warm_up_steps=4, num_workers=0)
        self.assertEqual(accelerator.rates, [0.025])


if __name__ == "__main__":
    unittest.main()
