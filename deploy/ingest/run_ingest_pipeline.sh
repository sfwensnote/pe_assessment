#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

PY_BIN="${PYTHON_BIN:-python3}"
"$PY_BIN" 8_ingest_pipeline.py "$@"
