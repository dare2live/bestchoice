#!/usr/bin/env bash
# MACD 金叉选股 — 启动脚本
# 用法: bash run.sh [port]
set -e
PORT=${1:-8765}
cd "$(dirname "$0")"
echo "启动 MACD 金叉选股服务 → http://localhost:${PORT}"
python -m uvicorn main:app --host 0.0.0.0 --port "${PORT}"
