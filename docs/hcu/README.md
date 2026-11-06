# HCU thin fork

This branch keeps HCU-specific behavior in `diffsynth_hcu/` and model-specific
MiniMax-H3 tensor paths in `diffsynth/models/`. The upstream runner, loader,
checkpoint helper, loss functions, and H3 model call these helpers directly.

## Current optimizations

| Area | Implementation | Default condition |
|---|---|---|
| Packed varlen attention | `diffsynth_hcu.attention.packed_varlen_attention` honors the selected FA2/FA3/FA4 backend and retains segmented fallback | Eligible HIP tensors; other devices retain upstream segmented dispatch |
| LightOp | RMSNorm+residual, SwiGLU, and conservative concat dispatch | HIP device, supported dtype/layout, kernel available |
| H3 tensor path | RoPE, modulation, residual norm fusion, QKV unbind, fused MLP activation, indexed modulation reduction | H3 model path |
| Tensor compile | H3 RoPE/modulation expressions in `diffsynth/models/minimax_h3_tensor_ops.py` | HIP training automatically; disabled for `:data_process` |
| LoRA | Unit-scale PEFT forward and optional AITER Add+SwiGLU | HIP LoRA training; only eligible PEFT Linear modules, installed before prepare |
| ZeRO-3 fetch | Single-parameter LoRA all-reduce and HIP default-stream ordering | HIP + ZeRO-3 + LoRA |
| CPUAdam | Vendored fused update with asynchronous BF16 writeback | HIP ZeRO-3 + CPU FP32 DeepSpeedCPUAdam states and GPU BF16 shards; no NVMe, parameter offload, or GradScaler |
| FA replay cache | GPU cache around configured reentrant checkpoints | `--fa_cache_layers` (default 20; 0 disables); automatically bypassed without HIP/FA2/reentrant checkpoints or with data processing, CFG training, independent checkpoint offload |
| Training tools | Deterministic per-step RNG, scheduled profiler, throughput/MFU telemetry | Seed/profiler/W&B options |

The code preserves parameter names and state-dict keys. Unsupported devices,
dtypes, layouts, or dependency APIs use the upstream path or raise a direct
compatibility error where silently falling back would change correctness.

## Upstream files changed

- `diffsynth/diffusion/runner.py`: HIP checkpoint setup, RNG context, LoRA
  fetch, CPUAdam lifecycle, profiler and telemetry while retaining warm-up.
- `diffsynth/core/gradient/gradient_checkpoint.py`: FA replay wrapping for
  configured blocks without capturing activation tuples.
- `diffsynth/core/loader/model.py`: CPU DiskMap reads for ordinary ZeRO-3
  loading; quantized, VRAM-managed, and caller-supplied state-dict paths stay
  separate.
- `diffsynth/models/minimax_h3_dit.py` and `_comfy.py`: H3 tensor layout and
  model-level fusion paths.
- `diffsynth/models/minimax_h3_audio_vae.py`: explicit CPU filter construction.
- `diffsynth/diffusion/loss.py`, `logger.py`, and `parsers.py`: RNG, metrics,
  and HCU arguments.
- `examples/minimax_h3/model_training/train.py`: seed, compile, FA cache, and
  performance token wiring.

## Validation and upstream maintenance

The project owner accepted the preceding 115-step HCU run as the baseline:
steps 2-100 averaged 48.50 s (median 48.49 s), and steps 101-115 averaged
48.68 s. Strict per-step numerical equivalence remains unproven. This is the
accepted pre-upgrade result, not a new benchmark of this revision.

This fork incorporates upstream main at `08fdc0da`. See
[UPSTREAM_UPGRADES.md](UPSTREAM_UPGRADES.md) for source provenance, the two-step
upgrade workflow, lifecycle invariants, test layers and rollback guidance.
Detailed operational logs and interrupted-run handoffs are retained separately
from the distributable source tree.

`--enable_csv_log` retains `loss.csv` and adds `training_steps.csv`. Compare the
same logging configuration and exclude warm-up when reporting throughput.
Weights-only reload does not restore optimizer, scheduler or RNG state.

## Launch and setup

Run the H3 scripts from the repository root. They call the upstream
`examples/minimax_h3/model_training/train.py` entry directly and use ordinary
DiffSynth arguments. See [`Setup.md`](Setup.md) for the pinned dependency tuple
and CPUAdam build preflight.

```bash
DIFFSYNTH_ATTENTION_IMPLEMENTATION=torch PYTHONPATH=. python -m unittest discover -s tests/hcu -p 'test_*.py' -v
MAX_JOBS=1 python tests/hcu/validate_kernels.py --native --fa --model
bash tests/hcu/smoke.sh /absolute/path/to/cached-inputs /absolute/path/to/new-smoke-output
```

The kernel checks compare sixty CPUAdam updates against installed DeepSpeed,
FA2 GPU replay forward/backward against uncached FA2, and standard/Comfy H3
blocks against the pinned upstream source at fixed tolerances. The smoke
script then launches three microsteps on eight cards and saves weights. It
requires the pinned target environment, available cards, local model weights,
and existing cached inputs; it does not install dependencies or process data.
