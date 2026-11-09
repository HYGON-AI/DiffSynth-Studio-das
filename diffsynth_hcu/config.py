"""Small argparse additions for HCU training."""
from __future__ import annotations
import argparse


def add_hcu_config(parser: argparse.ArgumentParser):
    parser.add_argument("--enable_torch_profiler", action="store_true")
    parser.add_argument("--profiler_output_path", default=None)
    parser.add_argument("--profiler_freq", type=int, default=None)
    parser.add_argument("--profiler_wait", type=int, default=0)
    parser.add_argument("--profiler_warmup", type=int, default=1)
    parser.add_argument("--profiler_active", type=int, default=1)
    parser.add_argument("--profiler_repeat", type=int, default=1)
    parser.add_argument("--profiler_target_ranks", default="0")
    parser.add_argument("--profiler_record_shapes", action="store_true")
    parser.add_argument("--profiler_profile_memory", action="store_true")
    parser.add_argument("--profiler_with_stack", action="store_true")
    parser.add_argument("--profiler_with_modules", action="store_true")
    parser.add_argument("--performance_log_interval", type=int, default=10)
    parser.add_argument("--hardware_peak_tflops", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_train_steps", type=int, default=None, help="Optional microbatch limit for bounded validation.")
    parser.add_argument("--fa_cache_layers", type=int, default=20, help="GPU FA replay cache layers; 0 disables.")
    return parser
