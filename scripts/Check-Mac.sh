#!/bin/bash
set -euo pipefail
echo "OS: $(uname -s) / Architecture: $(uname -m)"
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || { echo 'Apple Silicon Macのターミナルから実行してください。'; exit 1; }
sysctl -n hw.memsize
sysctl -n hw.ncpu
df -h .
if command -v docker >/dev/null; then
  docker info --format 'Docker: {{.Architecture}} / CPUs: {{.NCPU}} / Memory(bytes): {{.MemTotal}}'
  docker compose version
fi
