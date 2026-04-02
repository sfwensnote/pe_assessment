#!/usr/bin/env python3
"""Generate manifest files for reviewed training subsets."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import yaml

from utils.annotation_io import load_json_any
from utils.training_manifest import (
    project_relative,
    resolve_project_path,
    review_tier,
    write_manifest,
)


PROJECT_ROOT = Path(__file__).parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
OUTPUT_DIR = PROJECT_ROOT / "data/processed/review/training_manifests"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)


def collect_records() -> list[dict[str, Any]]:
    annotation_root = resolve_project_path(CONFIG["paths"]["annotations"])
    skeleton_root = resolve_project_path(CONFIG["paths"]["skeletons"])
    records: list[dict[str, Any]] = []

    for action in CONFIG["actions"].keys():
        action_dir = annotation_root / action
        if not action_dir.exists():
            continue
        for annotation_path in sorted(action_dir.glob("*.json")):
            annotation, _ = load_json_any(annotation_path)
            video_id = annotation.get("video_id", annotation_path.stem)
            skeleton_path = skeleton_root / action / f"{video_id}.json"
            if not skeleton_path.exists():
                continue

            quality = annotation.get("quality", {})
            record = {
                "video_id": video_id,
                "action_type": action,
                "annotation_path": project_relative(annotation_path),
                "skeleton_path": project_relative(skeleton_path),
                "reviewed": bool(annotation.get("reviewed", False)),
                "review_source": annotation.get("review_source"),
                "review_mode": annotation.get("review_mode"),
                "review_decision": annotation.get("review_decision"),
                "review_tier": review_tier(annotation),
                "overall_score": float(quality.get("overall_score", 0.0) or 0.0),
                "is_standard": bool(quality.get("is_standard", False)),
                "errors": list(quality.get("errors", []) or []),
            }
            records.append(record)
    return records


def write_manifest_group(
    records: list[dict[str, Any]],
    name: str,
    predicate: Callable[[dict[str, Any]], bool],
) -> tuple[list[dict[str, Any]], Path]:
    selected = [record for record in records if predicate(record)]
    manifest_path = OUTPUT_DIR / f"{name}.jsonl"
    write_manifest(manifest_path, selected)

    by_action_dir = OUTPUT_DIR / "by_action" / name
    for action in CONFIG["actions"].keys():
        action_records = [
            record for record in selected if record["action_type"] == action
        ]
        write_manifest(by_action_dir / f"{action}.jsonl", action_records)
    return selected, manifest_path


def summarize_manifest(records: list[dict[str, Any]]) -> dict[str, Any]:
    action_counts = Counter(record["action_type"] for record in records)
    tier_counts = Counter(record["review_tier"] for record in records)
    decision_counts = Counter(
        record.get("review_decision")
        for record in records
        if record.get("review_decision")
    )
    return {
        "total": len(records),
        "by_action": dict(action_counts),
        "review_tiers": dict(tier_counts),
        "review_decisions": dict(decision_counts),
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = collect_records()

    manifest_specs = {
        "all_annotations": lambda record: True,
        "reviewed_all": lambda record: record["reviewed"],
        "action_excluding_reviewed_fail": lambda record: record.get("review_decision")
        not in {"confirmed_fail", "provisional_fail"},
        "high_confidence": lambda record: record["review_tier"]
        in {"manual", "confirmed"},
        "main_reviewed": lambda record: record["review_tier"]
        in {"manual", "confirmed", "provisional_pass"},
        "provisional_fail": lambda record: record["review_tier"] == "provisional_fail",
        "unreviewed_pool": lambda record: record["review_tier"] == "unreviewed",
        "action_all": lambda record: True,
        "phase_reviewed_all": lambda record: record["reviewed"],
        "quality_high_confidence": lambda record: record["review_tier"]
        in {"manual", "confirmed"},
        "quality_main_reviewed": lambda record: record["review_tier"]
        in {"manual", "confirmed", "provisional_pass"},
    }

    summary: dict[str, Any] = {
        "all_records": len(records),
        "manifests": {},
        "recommended_usage": {
            "action_model": "data/processed/review/training_manifests/action_excluding_reviewed_fail.jsonl",
            "phase_model": "data/processed/review/training_manifests/phase_reviewed_all.jsonl",
            "quality_model_strict": "data/processed/review/training_manifests/quality_high_confidence.jsonl",
            "quality_model_main": "data/processed/review/training_manifests/quality_main_reviewed.jsonl",
        },
    }

    for name, predicate in manifest_specs.items():
        selected, manifest_path = write_manifest_group(records, name, predicate)
        summary["manifests"][name] = {
            "path": project_relative(manifest_path),
            **summarize_manifest(selected),
        }

    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 72)
    print("Training manifests generated")
    print("=" * 72)
    print(f"all_records      : {len(records)}")
    for name, payload in summary["manifests"].items():
        print(f"{name:<16}: {payload['total']}")
    print(f"summary_file     : {summary_path}")
    print("recommended usage:")
    for key, value in summary["recommended_usage"].items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
