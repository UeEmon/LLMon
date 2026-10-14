#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v docker >/dev/null || { echo 'Docker Desktopをインストールしてください。'; exit 1; }
docker info >/dev/null
docker compose up -d --build
echo 'JupyterLab: http://localhost:18082 (トークンは次のログ内のURLから取得)'
echo 'LLMon Web管理（学習・専門文書RAG）: http://localhost:18083'
docker compose logs --tail=30 lab
