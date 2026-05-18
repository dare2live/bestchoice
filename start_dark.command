#!/bin/bash
# MACD 选股台 — 深色第一版启动脚本
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT=8765
export PATH="/opt/homebrew/opt/python@3.13/libexec/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
PYTHON_BIN="${PYTHON_BIN:-python}"
export BESTCHOICE_START_PROFILE="first_dark_legacy"
export BESTCHOICE_SKIP_WARMUP="1"

find_port_pids() {
  lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null | sort -u
}

pid_belongs_to_project() {
  local pid="$1"
  local cwd
  cwd="$(lsof -a -p "$pid" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p' | head -n 1)"
  [[ "$cwd" == "$ROOT_DIR" ]] && return 0
  local cmd
  cmd="$(ps -ww -p "$pid" -o command= 2>/dev/null)"
  [[ "$cmd" == *"$ROOT_DIR"* ]]
}

stop_old_server() {
  local matched=()
  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    pid_belongs_to_project "$pid" && matched+=("$pid")
  done < <(find_port_pids)
  [[ ${#matched[@]} -eq 0 ]] && return 0
  echo "停止旧实例（PID: ${matched[*]}）..."
  kill "${matched[@]}" 2>/dev/null || true
  for _ in {1..20}; do
    sleep 0.3
    local still=0
    for p in "${matched[@]}"; do kill -0 "$p" 2>/dev/null && still=1 && break; done
    [[ $still -eq 0 ]] && return 0
  done
  kill -9 "${matched[@]}" 2>/dev/null || true
}

check_port_conflict() {
  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    if ! pid_belongs_to_project "$pid"; then
      echo "端口 $PORT 被其他程序占用（PID: $pid），请先关闭它。"
      exit 1
    fi
  done < <(find_port_pids)
}

cd "$ROOT_DIR"
stop_old_server
check_port_conflict

echo "========================================"
echo "  MACD 选股台 深色第一版启动中..."
echo "  地址: http://localhost:$PORT/dark"
echo "  Python: $($PYTHON_BIN --version 2>&1)"
echo "  按 Ctrl+C 停止"
echo "========================================"

(sleep 1.5 && open "http://localhost:${PORT}/dark") &

exec "$PYTHON_BIN" -m uvicorn main:app --host 127.0.0.1 --port "$PORT"
