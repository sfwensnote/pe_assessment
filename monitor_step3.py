#!/usr/bin/env python3
"""Monitor Step3 auto-annotation progress."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).parent
STATE_PATH = PROJECT_ROOT / "data/processed/annotate/pipeline_state.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor Step3 auto-annotation")
    parser.add_argument("--watch", action="store_true", help="Refresh continuously")
    parser.add_argument(
        "--interval", type=float, default=2.0, help="Refresh interval seconds"
    )
    parser.add_argument(
        "--stop_when_done",
        action="store_true",
        help="Exit watch mode when status is completed/failed/partial/cancelled",
    )
    return parser.parse_args()


def load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {}
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def fmt_time(value: Any) -> str:
    if not value:
        return "-"
    try:
        return datetime.fromisoformat(str(value)).strftime("%m-%d %H:%M:%S")
    except Exception:
        return str(value)


def bar(current: int, total: int, width: int = 18) -> str:
    if total <= 0:
        total = 1
    ratio = max(0.0, min(1.0, current / total))
    filled = int(ratio * width)
    return f"[{'#' * filled}{'-' * (width - filled)}]"


def render(state: dict[str, Any]) -> str:
    if not state:
        return "No Step3 state file yet. Run: python run_step3_prepare.py"

    summary = state.get("summary", {})
    actions = state.get("actions", {})
    lines: list[str] = []
    lines.append("=" * 86)
    lines.append("Step3 Auto-Annotation Monitor")
    lines.append("=" * 86)
    lines.append(
        f"run_id={state.get('run_id', '-')} | status={state.get('status', '-')} | "
        f"workers={state.get('workers', '-')} | chunk_size={state.get('chunk_size', '-')}"
    )
    lines.append(
        f"started={fmt_time(state.get('started_at'))} | updated={fmt_time(state.get('updated_at'))}"
    )
    lines.append(f"message={state.get('message', '')}")
    lines.append(
        "summary: "
        f"skeleton={summary.get('skeleton_total', 0)} | "
        f"existing={summary.get('annotation_existing', 0)} | "
        f"added={summary.get('annotation_added_this_run', 0)} | "
        f"remaining={summary.get('remaining', 0)} | "
        f"chunks={summary.get('chunks_done', 0)}/{summary.get('chunks_total', 0)} | "
        f"rate={summary.get('rate_files_per_min', 0.0)} files/min"
    )
    lines.append("-" * 86)
    lines.append(
        f"{'Action':<12} {'Progress':<28} {'Done':>6} {'Remain':>7} {'Chunks':>10} {'Status':>10}"
    )
    lines.append("-" * 86)

    for action in sorted(actions.keys()):
        item = actions[action]
        current = int(
            item.get("annotation_current", item.get("annotation_existing", 0))
        )
        total = int(item.get("skeleton_total", 0))
        remaining = int(item.get("remaining", 0))
        progress = bar(current, total)
        pct = (current / total * 100.0) if total else 0.0
        chunk_text = f"{item.get('chunks_done', 0)}/{item.get('chunks_total', 0)}"
        lines.append(
            f"{action:<12} {progress} {pct:5.1f}% {current:>6} {remaining:>7} {chunk_text:>10} {str(item.get('status', '-')):>10}"
        )

    events = state.get("events", [])[-8:]
    lines.append("-" * 86)
    lines.append("recent events:")
    if not events:
        lines.append("(none)")
    else:
        for event in events:
            lines.append(
                f"[{fmt_time(event.get('time'))}] [{event.get('level', 'info')}] "
                f"[{event.get('action') or 'global'}] {event.get('message', '')}"
            )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if not args.watch:
        print(render(load_state()))
        return 0

    while True:
        print("\033[2J\033[H", end="")
        state = load_state()
        print(render(state))
        if args.stop_when_done and str(state.get("status", "")) in {
            "completed",
            "failed",
            "partial",
            "cancelled",
        }:
            break
        time.sleep(max(0.5, args.interval))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
