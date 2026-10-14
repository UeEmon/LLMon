#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/cpu
run="outputs/cpu/$(date +%Y%m%d-%H%M%S)-$$"
echo "出力先: $run"
docker compose run --rm -T --no-deps lab python /opt/app/train_cpu.py --output "$run" "$@" 2>&1 | tee "${run}.log"
