#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || { echo 'Apple Silicon上のネイティブターミナルで実行してください。'; exit 1; }
py="${PYTHON_BIN:-python3.11}"
command -v "$py" >/dev/null || { echo 'Python 3.11をインストールするか、PYTHON_BINを指定してください。'; exit 1; }
"$py" -m venv .venv-mlx
.venv-mlx/bin/python -m pip install --upgrade pip
.venv-mlx/bin/python -m pip install 'mlx-lm[train]==0.26.4' 'transformers==4.53.3' 'datasets==3.6.0' 'numpy==2.2.6'
.venv-mlx/bin/python -m pip freeze > config/mlx-installed.txt
.venv-mlx/bin/python -c 'import mlx.core as mx; print("Metal available:", mx.metal.is_available()); assert mx.metal.is_available()'
echo 'MLXの準備が完了しました。'
