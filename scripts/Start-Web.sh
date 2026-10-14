#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v docker >/dev/null || { echo 'Docker Desktopをインストールしてください。'; exit 1; }
docker info >/dev/null
docker compose up -d --build rag
echo 'LLMon Web学習管理: http://localhost:18083'
docker compose logs --tail=25 rag
