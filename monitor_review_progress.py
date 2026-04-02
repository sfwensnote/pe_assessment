#!/usr/bin/env python3
"""Monitor manual review progress for sampled review queues."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from utils.annotation_io import load_json_any


PROJECT_ROOT = Path(__file__).parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

ACTION_LABELS = {
    "pushup": "Push-up",
    "squat": "Squat",
    "situp": "Sit-up",
    "jump_rope": "Jump Rope",
    "long_jump": "Long Jump",
    "pullup": "Pull-up",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor manual review progress")
    parser.add_argument(
        "--queue_file",
        type=str,
        default="data/processed/review/review_queue.jsonl",
        help="Review queue JSONL file",
    )
    parser.add_argument("--watch", action="store_true", help="Refresh continuously")
    parser.add_argument(
        "--interval", type=float, default=5.0, help="Refresh interval seconds"
    )
    return parser.parse_args()


def load_queue(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        if isinstance(item, dict):
            records.append(item)
    return records


def reviewed_flag(path: Path) -> bool:
    data, _ = load_json_any(path)
    return bool(data.get("reviewed", False))


def corpus_counts() -> dict[str, tuple[int, int]]:
    root = PROJECT_ROOT / CONFIG["paths"]["annotations"]
    stats: dict[str, tuple[int, int]] = {}
    for action in CONFIG["actions"].keys():
        total = 0
        reviewed = 0
        action_dir = root / action
        for path in sorted(action_dir.glob("*.json")) if action_dir.exists() else []:
            total += 1
            if reviewed_flag(path):
                reviewed += 1
        stats[action] = (total, reviewed)
    return stats


def render(queue_records: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("=" * 82)
    lines.append(
        f"Manual Review Progress - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    lines.append("=" * 82)

    reviewed_cache: dict[Path, bool] = {}

    def cached_reviewed_flag(path: Path) -> bool:
        if path not in reviewed_cache:
            reviewed_cache[path] = reviewed_flag(path)
        return reviewed_cache[path]

    queue_by_action: dict[str, list[dict[str, Any]]] = {}
    queue_reviewed = 0
    for item in queue_records:
        action = item["action_type"]
        queue_by_action.setdefault(action, []).append(item)
        annotation_path = PROJECT_ROOT / item["annotation_path"]
        if cached_reviewed_flag(annotation_path):
            queue_reviewed += 1

    lines.append(
        f"queue selected={len(queue_records)} | queue reviewed={queue_reviewed} | queue pending={len(queue_records) - queue_reviewed}"
    )
    lines.append("-" * 82)
    lines.append(
        f"{'Action':<12} {'Corpus':<16} {'Queue':<16} {'Queue Done':<12} {'Label':<16}"
    )
    lines.append("-" * 82)

    corpus = corpus_counts()
    for action in CONFIG["actions"].keys():
        total, reviewed = corpus.get(action, (0, 0))
        action_queue = queue_by_action.get(action, [])
        action_queue_done = 0
        for item in action_queue:
            if cached_reviewed_flag(PROJECT_ROOT / item["annotation_path"]):
                action_queue_done += 1
        queue_text = f"{action_queue_done}/{len(action_queue)}"
        queue_pct = (
            f"{(action_queue_done / len(action_queue) * 100):.1f}%"
            if action_queue
            else "-"
        )
        lines.append(
            f"{action:<12} {f'{reviewed}/{total}':<16} {queue_text:<16} {queue_pct:<12} {ACTION_LABELS.get(action, action):<16}"
        )

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    queue_path = Path(args.queue_file)
    if not queue_path.is_absolute():
        queue_path = PROJECT_ROOT / queue_path

    if not args.watch:
        print(render(load_queue(queue_path)))
        return 0

    while True:
        print("\033[2J\033[H", end="")
        print(render(load_queue(queue_path)))
        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
