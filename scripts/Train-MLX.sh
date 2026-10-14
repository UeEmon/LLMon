#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -x .venv-mlx/bin/mlx_lm.lora ]] || { echo '先にbash scripts/Setup-MLX.shを実行してください。'; exit 1; }
.venv-mlx/bin/python app/data_utils.py data
mkdir -p outputs/mlx
run="outputs/mlx/$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$run"
model="${MLX_MODEL:-mlx-community/Qwen3-4B-4bit}"
echo "モデル: $model / 出力先: $run"
printf '%s\n' "$model" > "$run/base-model.txt"
cp config/mlx-installed.txt "$run/installed.txt"
.venv-mlx/bin/mlx_lm.lora --model "$model" --train --data data --adapter-path "$run/adapter" \
  --batch-size 1 --num-layers 4 --iters 20 --learning-rate 0.0001 \
  --max-seq-length 512 --mask-prompt --grad-checkpoint "$@" 2>&1 | tee "$run/train.log"
.venv-mlx/bin/mlx_lm.lora --model "$model" --adapter-path "$run/adapter" --data data --test \
  --batch-size 1 2>&1 | tee "$run/test.log"
