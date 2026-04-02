#!/usr/bin/env python3
"""Build a prioritized manual-review queue from auto-annotated samples."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from utils.annotation_io import load_json_any, save_json_utf8


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
    parser = argparse.ArgumentParser(
        description="Prepare a sampled manual-review queue"
    )
    parser.add_argument(
        "--actions", type=str, default="all", help="Comma-separated actions"
    )
    parser.add_argument(
        "--annotation_dir",
        type=str,
        default=CONFIG["paths"]["annotations"],
        help="Annotation directory",
    )
    parser.add_argument(
        "--skeleton_dir",
        type=str,
        default=CONFIG["paths"]["skeletons"],
        help="Skeleton directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/review",
        help="Output directory for review queue files",
    )
    parser.add_argument(
        "--low_per_action", type=int, default=20, help="Low-score samples per action"
    )
    parser.add_argument(
        "--error_per_action",
        type=int,
        default=15,
        help="Error-heavy samples per action",
    )
    parser.add_argument(
        "--clean_per_action",
        type=int,
        default=10,
        help="Clean random spot-check samples",
    )
    parser.add_argument(
        "--boundary_per_action",
        type=int,
        default=5,
        help="Boundary-score samples per action",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for spot checks"
    )
    parser.add_argument(
        "--include_reviewed",
        action="store_true",
        help="Include already reviewed samples in candidate pool",
    )
    return parser.parse_args()


def resolve_actions(raw: str) -> list[str]:
    actions = list(CONFIG["actions"].keys())
    if raw == "all":
        return actions
    values = [item.strip() for item in raw.split(",") if item.strip()]
    invalid = [item for item in values if item not in actions]
    if invalid:
        raise ValueError(f"Unknown actions: {', '.join(invalid)}")
    return values


def compute_risk_score(record: dict[str, Any]) -> float:
    score = float(record["score"])
    error_count = len(record["errors"])
    phase_count = int(record["phase_count"])
    expected_phase_count = int(record["expected_phase_count"])

    risk = 0.0
    risk += max(0.0, 60.0 - score) * 1.2
    if score < 40:
        risk += 15.0
    risk += error_count * 12.0
    if not record["is_standard"]:
        risk += 10.0
    if expected_phase_count > 0:
        if phase_count <= 1:
            risk += 25.0
        elif phase_count <= max(1, expected_phase_count // 2):
            risk += 12.0
        elif phase_count < expected_phase_count:
            risk += 4.0
    return round(risk, 2)


def load_record(
    annotation_path: Path, skeleton_dir: Path
) -> tuple[dict[str, Any], str]:
    data, encoding = load_json_any(annotation_path)
    action = data["action_type"]
    quality = data.get("quality", {})
    phases = data.get("phases", [])
    record = {
        "video_id": data.get("video_id", annotation_path.stem),
        "action_type": action,
        "action_label": ACTION_LABELS.get(action, action),
        "action_name": CONFIG["actions"][action]["name"],
        "annotation_path": str(annotation_path.relative_to(PROJECT_ROOT)),
        "skeleton_path": str(
            (
                skeleton_dir
                / action
                / f"{data.get('video_id', annotation_path.stem)}.json"
            ).relative_to(PROJECT_ROOT)
        ),
        "reviewed": bool(data.get("reviewed", False)),
        "score": float(quality.get("overall_score", 0.0) or 0.0),
        "errors": list(quality.get("errors", []) or []),
        "is_standard": bool(quality.get("is_standard", False)),
        "phase_count": len(set(phases)) if phases else 0,
        "expected_phase_count": len(data.get("phase_names", [])),
    }
    record["risk_score"] = compute_risk_score(record)
    return record, encoding


def select_records(
    action: str,
    records: list[dict[str, Any]],
    low_per_action: int,
    error_per_action: int,
    clean_per_action: int,
    boundary_per_action: int,
    seed: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_paths: set[str] = set()

    def take(
        candidates: list[dict[str, Any]], limit: int, bucket: str, reason_builder
    ) -> None:
        count = 0
        for item in candidates:
            key = item["annotation_path"]
            if key in selected_paths:
                continue
            row = dict(item)
            row["bucket"] = bucket
            row["review_reason"] = reason_builder(item)
            selected.append(row)
            selected_paths.add(key)
            count += 1
            if count >= limit:
                break

    low_candidates = sorted(
        records, key=lambda item: (item["score"], -item["risk_score"])
    )
    take(
        low_candidates,
        low_per_action,
        "low_score",
        lambda item: f"Low score {item['score']:.1f}",
    )

    error_candidates = sorted(
        [item for item in records if item["errors"]],
        key=lambda item: (-len(item["errors"]), -item["risk_score"], item["score"]),
    )
    take(
        error_candidates,
        error_per_action,
        "error_focus",
        lambda item: f"Error-heavy sample ({len(item['errors'])} errors)",
    )

    boundary_pool = [item for item in records if 45.0 <= item["score"] <= 85.0]
    if not boundary_pool:
        boundary_pool = records
    boundary_candidates = sorted(
        boundary_pool,
        key=lambda item: (
            min(abs(item["score"] - 60.0), abs(item["score"] - 80.0)),
            -item["risk_score"],
        ),
    )
    take(
        boundary_candidates,
        boundary_per_action,
        "boundary",
        lambda item: f"Boundary score {item['score']:.1f}",
    )

    clean_candidates = [item for item in records if not item["errors"]]
    rng = random.Random(seed + sum(ord(ch) for ch in action))
    rng.shuffle(clean_candidates)
    take(
        clean_candidates,
        clean_per_action,
        "clean_random",
        lambda item: f"Spot check clean sample {item['score']:.1f}",
    )
    return selected


def main() -> int:
    args = parse_args()
    actions = resolve_actions(args.actions)
    annotation_dir = PROJECT_ROOT / args.annotation_dir
    skeleton_dir = PROJECT_ROOT / args.skeleton_dir
    output_dir = PROJECT_ROOT / args.output_dir
    by_action_dir = output_dir / "by_action"
    output_dir.mkdir(parents=True, exist_ok=True)
    by_action_dir.mkdir(parents=True, exist_ok=True)

    queue: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "actions": {},
        "total_selected": 0,
        "normalized_to_utf8": 0,
        "seed": args.seed,
        "policy": {
            "low_per_action": args.low_per_action,
            "error_per_action": args.error_per_action,
            "boundary_per_action": args.boundary_per_action,
            "clean_per_action": args.clean_per_action,
        },
    }
    rank = 1

    for action in actions:
        action_dir = annotation_dir / action
        records: list[dict[str, Any]] = []
        normalized_count = 0
        reviewed_count = 0

        for annotation_path in (
            sorted(action_dir.glob("*.json")) if action_dir.exists() else []
        ):
            record, encoding = load_record(annotation_path, skeleton_dir)
            if encoding not in {"utf-8", "utf-8-sig"}:
                data, _ = load_json_any(annotation_path)
                save_json_utf8(annotation_path, data)
                normalized_count += 1
            if record["reviewed"]:
                reviewed_count += 1
            if not args.include_reviewed and record["reviewed"]:
                continue
            records.append(record)

        picked = select_records(
            action,
            records,
            args.low_per_action,
            args.error_per_action,
            args.clean_per_action,
            args.boundary_per_action,
            args.seed,
        )
        for item in picked:
            item["rank"] = rank
            rank += 1
            queue.append(item)

        bucket_counter = Counter(item["bucket"] for item in picked)
        summary["actions"][action] = {
            "action_name": CONFIG["actions"][action]["name"],
            "candidate_count": len(records),
            "reviewed_existing": reviewed_count,
            "selected": len(picked),
            "normalized_to_utf8": normalized_count,
            "bucket_counts": dict(bucket_counter),
        }
        summary["total_selected"] += len(picked)
        summary["normalized_to_utf8"] += normalized_count

        action_queue_path = by_action_dir / f"{action}.jsonl"
        with open(action_queue_path, "w", encoding="utf-8") as f:
            for item in picked:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    queue_path = output_dir / "review_queue.jsonl"
    with open(queue_path, "w", encoding="utf-8") as f:
        for item in queue:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary_path = output_dir / "review_queue_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 72)
    print("Prepared sampled review queue")
    print("=" * 72)
    print(f"actions           : {', '.join(actions)}")
    print(f"total_selected    : {summary['total_selected']}")
    print(f"normalized_utf8   : {summary['normalized_to_utf8']}")
    print(f"queue_file        : {queue_path}")
    print(f"summary_file      : {summary_path}")
    print(f"by_action_dir     : {by_action_dir}")
    print(
        "start review      : python 2_review_annotations.py --queue_file data/processed/review/review_queue.jsonl --only_unreviewed"
    )
    print(
        "monitor progress  : python monitor_review_progress.py --queue_file data/processed/review/review_queue.jsonl"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
