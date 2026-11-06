# MiniMax-H3 HCU launcher

Run the scripts from the repository root. They call the upstream
`examples/minimax_h3/model_training/train.py` entry directly.

The full and LoRA scripts use the pinned HCU/DTK environment and the ordinary
DiffSynth arguments. `--fa_cache_layers 0` disables replay; stage 2 defaults
to the configured GPU cache depth. CPUAdam is selected automatically for an
eligible DeepSpeed ZeRO-3 run and requires DeepSpeed 0.18.4.

The launcher keeps model download and local model-root settings as normal
DiffSynth configuration. No environment lock, patch manager, or HCU-specific
runtime switch is required.
