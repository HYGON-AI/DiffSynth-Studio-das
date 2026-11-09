# Maintaining the HCU fork

## Source baseline

This revision incorporates ModelScope DiffSynth-Studio `main` at
`08fdc0da8f41426885d9432e9007e6e460a4c30d` (2026-11-06 fetch).
The HCU implementation comes from the thin fork built on `1e8ae1d5`;
the optimized math was migrated from the HCU adapter and legacy HCU source.
There is one training loop: the upstream `diffsynth/diffusion/runner.py`.
The HCU examples call the upstream MiniMax-H3 training entry.

## Ownership of changes

| Boundary | Owner | Upgrade invariant |
|---|---|---|
| Pre/post-prepare and teardown | `diffsynth_hcu/training.py` | Install LoRA before prepare; fetch/CPUAdam after prepare; keep all adapters alive through final save; unwind in reverse order on failure. |
| Training loop | Upstream runner with explicit helper calls | Preserve accumulation, optimizer-update detection, scheduler and save cadence. A save step and the optional limit count microbatches. |
| Packed attention / replay | `diffsynth_hcu/attention.py`, `fa_cache*.py`, checkpoint call site | One cache per reentrant checkpoint, one attention call per cached block; keep kwargs tensors in checkpoint inputs without retaining activation tuples in a closure. |
| CPUAdam / HIP fetch | `diffsynth_hcu/cpu_adam.py`, `deepspeed_hip.py` | Preserve gradient scaling, optimizer state, pinned-ring reuse waits, H2D ordering, and private DeepSpeed interface contracts. |
| H3 expressions and layouts | `diffsynth/models/minimax_h3_*` | Standard and Comfy QKV layouts differ. Preserve dtype/rounding, RoPE, modulation and reduction semantics. |
| Other upstream models and DMD2 | Upstream | Do not copy their loops into an HCU launcher or claim the ordinary training hooks cover DMD2's separate loop. |

No class swapping, imported-alias replacement, source-fingerprint lock, or
parallel training framework is required. The necessary third-party PEFT and
DeepSpeed bindings remain local to model/optimizer instances.

## Update in two steps

1. Fetch upstream and record its exact SHA. Merge it into a clean candidate
   branch while keeping HCU math and the deployment dependency tuple fixed.
   Review both textual conflicts and semantic changes in shared functions.
2. In a separate commit, adjust HCU call sites or compatibility logic only
   where the new upstream contract requires it. Keep mature kernels unchanged
   unless a specific failure demonstrates a need to change them.

Review the runner, training entry, loader, loss, checkpoint helper, H3 model,
VRAM maps and parser diff against upstream. Preserve public runner helper
signatures consumed by other training entry points. Refresh the reference H3
test source only after reviewing upstream model changes, not to make tests pass.

## Verification and acceptance

Focused tests cover checkpoint activation lifetime, interleaved replay
gradients, non-HIP fallback, training accumulation/warm-up, prepare failure,
partial adapter installation and final-save failure cleanup. Run them in the
target test environment:

```bash
DIFFSYNTH_ATTENTION_IMPLEMENTATION=torch PYTHONPATH=. python -m unittest discover -s tests/hcu -p 'test_*.py' -v
```

Native FA/CPUAdam checks and hardware training are additional layers; CPU/mock
tests cannot establish their numerical or performance equivalence. Before a
new hardware comparison, pin the source revisions, dependency tuple, inputs,
seed, optimization set, scheduler and logging configuration. Keep original
CSV/log/checkpoints and compare per-step values as well as steady throughput.

The project owner accepted the preceding 115-step run as the upgrade baseline:
steps 2-100 averaged 48.50 s (median 48.49 s), steps 101-115 averaged 48.68 s,
against approximately 48.6 s in the reference full log. First loss matched at
0.5495060086. Later loss values accumulated differences; strict per-step
numerical equivalence was not established. This is owner-reported baseline
acceptance, not a benchmark of this upstream-update revision.

The thin fork retains upstream LambdaLR warm-up. The older adapter/release
runner used ConstantLR: do not mix those trajectories without aligning the
effective schedule. Weights-only reload is not full optimizer/RNG resume.

For rollback, retain the previous deployed revision and its dependency tuple.
Deploy the new revision to a separate directory before replacing a running
installation. Do not reset a shared deployment or overwrite its results.
