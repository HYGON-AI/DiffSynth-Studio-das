# Third-party notices

## DeepSpeed CPUAdam

- Project: DeepSpeed
- Upstream: https://github.com/deepspeedai/DeepSpeed
- Version: v0.18.4
- License: Apache-2.0 (see `ops/csrc/cpu_adam/LICENSE` relative to this document)
- Local path: `diffsynth_hcu/ops/csrc/cpu_adam/`
- Changes: Adapted the CPUAdam sources to add a fused FP32 gradient scale, Adam update, and BF16 staging-output entry point for the optional ZeRO-3 CPUAdam pipeline. Original Microsoft copyright and SPDX notices are retained in the source files.
