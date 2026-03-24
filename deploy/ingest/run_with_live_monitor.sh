#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

PY_BIN="${PYTHON_BIN:-python3}"

"$PY_BIN" 8_ingest_pipeline.py "$@" &
PIPE_PID=$!

cleanup() {
  if kill -0 "$PIPE_PID" >/dev/null 2>&1; then
    kill "$PIPE_PID" >/dev/null 2>&1 || true
  fi
}

trap cleanup INT TERM

"$PY_BIN" 8_ingest_monitor.py --watch --stop_when_done
wait "$PIPE_PID"
