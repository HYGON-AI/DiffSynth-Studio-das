#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if [[ $# -ne 2 ]]; then
  echo "Usage: bash tests/hcu/smoke.sh CACHED_INPUT_DIR NEW_OUTPUT_DIR" >&2
  exit 2
fi
cache_dir="$1"
output_dir="$2"
test -d "$cache_dir"
if [[ -e "$output_dir" ]]; then
  echo "Output already exists; choose a fresh smoke output directory." >&2
  exit 2
fi
export MAX_JOBS=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m unittest discover -s tests/hcu -p 'test_*.py' -v
python tests/hcu/validate_kernels.py --native --fa --model
accelerate launch --config_file examples-hcu/MiniMax-H3/configs/accelerate_zero3_hcu_offload.yaml \
  examples/minimax_h3/model_training/train.py \
  --dataset_base_path "$cache_dir" \
  --data_file_keys "video,input_audio,references" \
  --extra_inputs "input_audio,references" \
  --height 480 --width 832 --num_frames 124 \
  --dataset_repeat 100 --seed 42 --learning_rate 1e-5 --num_epochs 1 \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:Ref2VA/transformer/model*.safetensors" \
  --processor_path "MiniMax/MiniMax-H3:Ref2VA/processor/" \
  --output_path "$output_dir" --remove_prefix_in_ckpt "pipe.dit." \
  --trainable_models dit --use_gradient_checkpointing --find_unused_parameters \
  --max_train_steps 3 --enable_csv_log --task sft:train
