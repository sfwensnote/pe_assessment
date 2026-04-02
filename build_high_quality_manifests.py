#!/usr/bin/env python3
"""Build high-quality manifests for a cleaner retraining round."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from utils.annotation_io import load_json_any
from utils.training_manifest import project_relative, write_manifest


PROJECT_ROOT = Path(__file__).parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build high-quality retraining manifests"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/review/high_quality_manifests",
        help="Output directory",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=50.0,
        help="Minimum score for non-reviewed high-quality candidates",
    )
    parser.add_argument(
        "--max_errors",
        type=int,
        default=1,
        help="Maximum allowed detected errors for non-reviewed candidates",
    )
    parser.add_argument(
        "--action_cap",
        type=int,
        default=50,
        help="Per-action cap for balanced action/quality manifests",
    )
    parser.add_argument(
        "--phase_cap",
        type=int,
        default=120,
        help="Per-action cap for phase manifest",
    )
    return parser.parse_args()


def collect_records(
    score_threshold: float, max_errors: int
) -> dict[str, list[dict[str, Any]]]:
    annotation_root = PROJECT_ROOT / CONFIG["paths"]["annotations"]
    skeleton_root = PROJECT_ROOT / CONFIG["paths"]["skeletons"]
    grouped: dict[str, list[dict[str, Any]]] = {
        action: [] for action in CONFIG["actions"].keys()
    }

    for action in CONFIG["actions"].keys():
        for annotation_path in sorted((annotation_root / action).glob("*.json")):
            annotation, _ = load_json_any(annotation_path)
            quality = annotation.get("quality", {})
            score = float(quality.get("overall_score", 0.0) or 0.0)
            errors = list(quality.get("errors", []) or [])
            decision = annotation.get("review_decision")
            review_source = annotation.get("review_source")
            is_pass = decision in {"confirmed_pass", "provisional_pass"}
            is_fail = decision in {"confirmed_fail", "provisional_fail"}
            meets_clean_rule = (
                (not is_fail) and score >= score_threshold and len(errors) <= max_errors
            )

            if not (is_pass or meets_clean_rule):
                continue

            video_id = annotation.get("video_id", annotation_path.stem)
            skeleton_path = skeleton_root / action / f"{video_id}.json"
            if not skeleton_path.exists():
                continue

            grouped[action].append(
                {
                    "video_id": video_id,
                    "action_type": action,
                    "annotation_path": project_relative(annotation_path),
                    "skeleton_path": project_relative(skeleton_path),
                    "overall_score": score,
                    "error_count": len(errors),
                    "errors": errors,
                    "reviewed": bool(annotation.get("reviewed", False)),
                    "review_source": review_source,
                    "review_decision": decision,
                    "is_standard": bool(quality.get("is_standard", False)),
                    "selection_reason": (
                        "review_pass"
                        if is_pass
                        else f"score>={score_threshold:.0f}_errors<={max_errors}"
                    ),
                }
            )

    for action, records in grouped.items():
        grouped[action] = sorted(
            records,
            key=lambda item: (
                0
                if item["review_decision"] in {"confirmed_pass", "provisional_pass"}
                else 1,
                item["error_count"],
                -item["overall_score"],
                0 if item["reviewed"] else 1,
                item["video_id"],
            ),
        )
    return grouped


def flatten(
    grouped: dict[str, list[dict[str, Any]]], cap: int | None = None
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for action in CONFIG["actions"].keys():
        values = grouped[action][:cap] if cap else grouped[action]
        records.extend(values)
    return records


def write_manifest_group(
    output_dir: Path, name: str, records: list[dict[str, Any]]
) -> Path:
    path = output_dir / f"{name}.jsonl"
    write_manifest(path, records)
    by_action_dir = output_dir / "by_action" / name
    for action in CONFIG["actions"].keys():
        write_manifest(
            by_action_dir / f"{action}.jsonl",
            [r for r in records if r["action_type"] == action],
        )
    return path


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_action = Counter(record["action_type"] for record in records)
    by_reason = Counter(record["selection_reason"] for record in records)
    by_decision = Counter(
        record.get("review_decision")
        for record in records
        if record.get("review_decision")
    )
    return {
        "total": len(records),
        "by_action": dict(by_action),
        "selection_reasons": dict(by_reason),
        "review_decisions": dict(by_decision),
    }


def main() -> int:
    args = parse_args()
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = collect_records(args.score_threshold, args.max_errors)
    action_records = flatten(grouped, cap=args.action_cap)
    phase_records = flatten(grouped, cap=args.phase_cap)
    quality_records = flatten(grouped, cap=args.action_cap)

    summary = {
        "policy": {
            "score_threshold": args.score_threshold,
            "max_errors": args.max_errors,
            "action_cap": args.action_cap,
            "phase_cap": args.phase_cap,
        },
        "manifests": {},
    }

    manifest_map = {
        "action_high_quality_balanced": action_records,
        "phase_high_quality": phase_records,
        "quality_high_quality_balanced": quality_records,
        "high_quality_all_candidates": flatten(grouped, cap=None),
    }

    for name, records in manifest_map.items():
        path = write_manifest_group(output_dir, name, records)
        summary["manifests"][name] = {
            "path": project_relative(path),
            **summarize(records),
        }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 72)
    print("High-quality manifests generated")
    print("=" * 72)
    for name, payload in summary["manifests"].items():
        print(f"{name:<28}: {payload['total']}")
    print(f"summary_file                 : {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
