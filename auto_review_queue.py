#!/usr/bin/env python3
"""Conservative agent-assisted review for sampled annotation queues."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from utils.annotation_io import load_json_any, save_json_utf8
from utils.skeleton import SkeletonProcessor


PROJECT_ROOT = Path(__file__).parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agent-assisted review for sampled queues"
    )
    parser.add_argument(
        "--queue_file",
        type=str,
        default="data/processed/review/review_queue.jsonl",
        help="Review queue JSONL file",
    )
    parser.add_argument(
        "--report_file",
        type=str,
        default="data/processed/review/auto_review_report.json",
        help="Output report JSON",
    )
    parser.add_argument(
        "--remaining_queue_file",
        type=str,
        default="data/processed/review/review_queue_remaining.jsonl",
        help="Output remaining queue JSONL",
    )
    parser.add_argument(
        "--min_confidence",
        type=float,
        default=80.0,
        help="Minimum confidence required for auto-review",
    )
    parser.add_argument(
        "--score_low",
        type=float,
        default=25.0,
        help="Low-score threshold for auto-confirmed fail",
    )
    parser.add_argument(
        "--score_high",
        type=float,
        default=85.0,
        help="High-score threshold for auto-confirmed pass",
    )
    parser.add_argument(
        "--force_all_remaining",
        action="store_true",
        help="Force a decision for every remaining queue item using a broader heuristic",
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="Do not modify annotation files"
    )
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


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


def extract_coords_and_scores(
    skeleton_payload: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    coords = []
    scores = []
    for frame in skeleton_payload.get("skeleton_sequence", []):
        keypoints = frame.get("keypoints", [])
        coords.append([[kp.get("x", 0.0), kp.get("y", 0.0)] for kp in keypoints])
        scores.append([kp.get("score", 0.0) for kp in keypoints])
    return np.asarray(coords, dtype=float), np.asarray(scores, dtype=float)


def compare_error_sets(
    annotation_errors: set[str], processor_errors: set[str]
) -> tuple[float, str]:
    if not annotation_errors and not processor_errors:
        return 1.0, "No errors in either annotation or processor check"
    union = annotation_errors | processor_errors
    overlap = annotation_errors & processor_errors
    ratio = len(overlap) / max(len(union), 1)
    return ratio, f"error overlap {len(overlap)}/{len(union)}"


def score_phase_structure(
    phases: list[int], expected_phase_count: int
) -> tuple[float, str]:
    if not phases:
        return -25.0, "No phases recorded"
    unique_phase_count = len(set(phases))
    transitions = (
        int(np.count_nonzero(np.diff(np.asarray(phases, dtype=int))))
        if len(phases) > 1
        else 0
    )

    score = 0.0
    reason = [f"phase_count={unique_phase_count}", f"transitions={transitions}"]
    if unique_phase_count <= 1:
        score -= 25.0
    elif expected_phase_count and unique_phase_count >= max(
        2, expected_phase_count - 1
    ):
        score += 10.0
    elif expected_phase_count and unique_phase_count >= max(
        2, expected_phase_count - 2
    ):
        score += 4.0
    else:
        score -= 10.0

    if 1 <= transitions <= max(2, expected_phase_count + 1):
        score += 10.0
    elif transitions == 0:
        score -= 15.0
    else:
        score -= 8.0
    return score, ", ".join(reason)


def analyze_sample(
    record: dict[str, Any],
    min_confidence: float,
    score_low: float,
    score_high: float,
) -> dict[str, Any]:
    annotation_path = resolve_path(record["annotation_path"])
    skeleton_path = resolve_path(record["skeleton_path"])
    annotation, _ = load_json_any(annotation_path)
    skeleton_payload, _ = load_json_any(skeleton_path)

    quality = annotation.get("quality", {})
    action = annotation["action_type"]
    coords, raw_scores = extract_coords_and_scores(skeleton_payload)
    if len(coords) == 0:
        return {
            "status": "needs_human",
            "decision": None,
            "confidence": 0.0,
            "notes": ["Empty skeleton sequence"],
            "annotation": annotation,
            "annotation_path": annotation_path,
            "record": record,
        }

    processor = SkeletonProcessor(target_frames=CONFIG["skeleton"]["target_frames"])
    features = processor.process(coords)
    processor_metrics = processor.compute_quality_metrics(features, action)
    processor_score, _ = processor.compute_score(
        processor_metrics, CONFIG["actions"][action]
    )

    annotation_score = float(quality.get("overall_score", 0.0) or 0.0)
    annotation_errors = set(quality.get("errors", []) or [])
    processor_errors = set(processor_metrics.get("errors", []) or [])
    error_overlap, error_reason = compare_error_sets(
        annotation_errors, processor_errors
    )

    visible = (raw_scores >= 0.25) & np.any(coords != 0.0, axis=2)
    mean_score = float(raw_scores.mean()) if raw_scores.size else 0.0
    avg_visible_joints = float(visible.sum(axis=1).mean()) if len(visible) else 0.0
    confident_frame_ratio = (
        float(np.mean(visible.sum(axis=1) >= 8)) if len(visible) else 0.0
    )

    confidence = 40.0
    notes = []

    if mean_score >= 0.65:
        confidence += 20.0
        notes.append(f"mean keypoint score {mean_score:.2f} strong")
    elif mean_score >= 0.55:
        confidence += 12.0
        notes.append(f"mean keypoint score {mean_score:.2f} good")
    elif mean_score >= 0.45:
        confidence += 6.0
        notes.append(f"mean keypoint score {mean_score:.2f} usable")
    else:
        confidence -= 20.0
        notes.append(f"mean keypoint score {mean_score:.2f} weak")

    if avg_visible_joints >= 10:
        confidence += 15.0
        notes.append(f"avg visible joints {avg_visible_joints:.1f} high")
    elif avg_visible_joints >= 8:
        confidence += 8.0
        notes.append(f"avg visible joints {avg_visible_joints:.1f} acceptable")
    else:
        confidence -= 15.0
        notes.append(f"avg visible joints {avg_visible_joints:.1f} low")

    if confident_frame_ratio >= 0.8:
        confidence += 10.0
        notes.append(f"confident frame ratio {confident_frame_ratio:.2f} strong")
    elif confident_frame_ratio >= 0.6:
        confidence += 5.0
        notes.append(f"confident frame ratio {confident_frame_ratio:.2f} acceptable")
    else:
        confidence -= 10.0
        notes.append(f"confident frame ratio {confident_frame_ratio:.2f} weak")

    phase_delta, phase_reason = score_phase_structure(
        list(annotation.get("phases", []) or []),
        len(annotation.get("phase_names", []) or []),
    )
    confidence += phase_delta
    notes.append(phase_reason)

    if error_overlap >= 0.8:
        confidence += 15.0
        notes.append(f"{error_reason}, consistent")
    elif error_overlap >= 0.5:
        confidence += 8.0
        notes.append(f"{error_reason}, mostly consistent")
    elif error_overlap > 0:
        confidence += 2.0
        notes.append(f"{error_reason}, partial consistency")
    else:
        confidence -= 15.0
        notes.append(f"{error_reason}, mismatch")

    score_gap = abs(annotation_score - float(processor_score))
    if score_gap <= 8:
        confidence += 15.0
        notes.append(f"score gap {score_gap:.1f}, very consistent")
    elif score_gap <= 15:
        confidence += 8.0
        notes.append(f"score gap {score_gap:.1f}, acceptable")
    elif score_gap <= 25:
        confidence += 2.0
        notes.append(f"score gap {score_gap:.1f}, moderate")
    else:
        confidence -= 12.0
        notes.append(f"score gap {score_gap:.1f}, large")

    decision = None
    bucket = record.get("bucket", "")
    if (
        annotation_score <= score_low
        and not quality.get("is_standard", False)
        and (annotation_errors or processor_errors)
    ):
        confidence += 10.0
        notes.append("extreme low-score fail sample")
        if confidence >= min_confidence:
            decision = "confirmed_fail"
    elif (
        annotation_score >= score_high
        and quality.get("is_standard", False)
        and not annotation_errors
        and error_overlap >= 0.8
    ):
        confidence += 10.0
        notes.append("extreme high-score pass sample")
        if confidence >= min_confidence:
            decision = "confirmed_pass"
    elif (
        action == "long_jump"
        and annotation_score >= 75
        and not annotation_errors
        and error_overlap >= 0.8
        and bucket == "clean_random"
    ):
        confidence += 8.0
        notes.append("stable long-jump clean sample")
        if confidence >= min_confidence:
            decision = "confirmed_pass"

    confidence = round(max(0.0, min(100.0, confidence)), 2)
    status = "auto_reviewed" if decision else "needs_human"
    return {
        "status": status,
        "decision": decision,
        "confidence": confidence,
        "notes": notes,
        "annotation_score": round(annotation_score, 2),
        "annotation_errors": sorted(annotation_errors),
        "annotation_is_standard": bool(quality.get("is_standard", False)),
        "mean_score": round(mean_score, 4),
        "avg_visible_joints": round(avg_visible_joints, 2),
        "confident_frame_ratio": round(confident_frame_ratio, 4),
        "error_overlap": round(error_overlap, 4),
        "processor_score": round(float(processor_score), 2),
        "processor_errors": sorted(processor_errors),
        "annotation": annotation,
        "annotation_path": annotation_path,
        "record": record,
    }


def fallback_force_decision(result: dict[str, Any]) -> tuple[str, list[str]]:
    """Force a provisional pass/fail when the full queue must be completed."""

    record = result["record"]
    action = record["action_type"]
    annotation_score = float(result.get("annotation_score") or 0.0)
    processor_score = float(result.get("processor_score") or 0.0)
    confidence = float(result.get("confidence") or 0.0)
    annotation_errors = set(result.get("annotation_errors") or [])
    processor_errors = set(result.get("processor_errors") or [])
    union_errors = annotation_errors | processor_errors
    is_standard = bool(result.get("annotation_is_standard"))
    blended_score = 0.6 * annotation_score + 0.4 * processor_score

    notes = [
        "forced full-queue auto review requested by user",
        f"annotation_score={annotation_score:.1f}",
        f"processor_score={processor_score:.1f}",
        f"blended_score={blended_score:.1f}",
        f"confidence={confidence:.1f}",
    ]

    decision = "provisional_fail"
    if action == "long_jump":
        if annotation_score >= 60.0 or processor_score >= 30.0 or blended_score >= 45.0:
            notes.append("long_jump fallback favors current clean annotation")
            return "provisional_pass", notes
        notes.append("long_jump fallback rejects low blended quality")
        return decision, notes

    if (
        not union_errors
        and is_standard
        and (annotation_score >= 70.0 or blended_score >= 65.0)
    ):
        notes.append("no detected errors and strong score profile")
        return "provisional_pass", notes

    if (
        is_standard
        and blended_score >= 60.0
        and confidence >= 55.0
        and len(processor_errors) <= 1
    ):
        notes.append("standard sample with acceptable blended score")
        return "provisional_pass", notes

    if annotation_score >= 75.0 and processor_score >= 55.0 and len(union_errors) <= 1:
        notes.append("high annotation and processor score alignment")
        return "provisional_pass", notes

    notes.append("fallback marks sample as failed for training conservatism")
    return decision, notes


def save_remaining_queue(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    queue_path = resolve_path(args.queue_file)
    report_path = resolve_path(args.report_file)
    remaining_queue_path = resolve_path(args.remaining_queue_file)
    queue_records = load_queue(queue_path)

    if not queue_records:
        print(f"No queue records found: {queue_path}")
        return 1

    decision_counter: Counter[str] = Counter()
    auto_by_action: dict[str, int] = defaultdict(int)
    remaining_records: list[dict[str, Any]] = []
    report_items: list[dict[str, Any]] = []

    for record in queue_records:
        annotation_path = resolve_path(record["annotation_path"])
        annotation, _ = load_json_any(annotation_path)
        if (
            annotation.get("reviewed")
            and annotation.get("review_source") != "agent_auto_review"
        ):
            report_items.append(
                {
                    "video_id": annotation.get("video_id", annotation_path.stem),
                    "action_type": annotation.get(
                        "action_type", record.get("action_type")
                    ),
                    "status": "already_reviewed_manual",
                    "decision": None,
                    "confidence": None,
                    "annotation_path": record["annotation_path"],
                }
            )
            continue

        result = analyze_sample(
            record, args.min_confidence, args.score_low, args.score_high
        )
        action = record["action_type"]
        item_report = {
            "video_id": result["annotation"].get(
                "video_id", Path(record["annotation_path"]).stem
            ),
            "action_type": action,
            "bucket": record.get("bucket"),
            "status": result["status"],
            "decision": result["decision"],
            "confidence": result["confidence"],
            "annotation_path": record["annotation_path"],
            "notes": result["notes"],
            "annotation_score": result.get("annotation_score"),
            "mean_score": result.get("mean_score"),
            "avg_visible_joints": result.get("avg_visible_joints"),
            "confident_frame_ratio": result.get("confident_frame_ratio"),
            "error_overlap": result.get("error_overlap"),
            "processor_score": result.get("processor_score"),
        }

        if args.force_all_remaining and result["status"] != "auto_reviewed":
            forced_decision, forced_notes = fallback_force_decision(result)
            result["status"] = "auto_reviewed"
            result["decision"] = forced_decision
            result["notes"] = list(result["notes"]) + forced_notes
            item_report["status"] = "auto_reviewed"
            item_report["decision"] = forced_decision
            item_report["notes"] = result["notes"]

        report_items.append(item_report)

        if result["status"] == "auto_reviewed" and result["decision"]:
            decision_counter[result["decision"]] += 1
            auto_by_action[action] += 1
            annotation_payload = result["annotation"]
            annotation_payload["reviewed"] = True
            annotation_payload["reviewed_at"] = now_iso()
            annotation_payload["review_source"] = "agent_auto_review"
            annotation_payload["review_mode"] = (
                "auto_force" if args.force_all_remaining else "auto"
            )
            annotation_payload["review_confidence"] = result["confidence"]
            annotation_payload["review_decision"] = result["decision"]
            annotation_payload["review_notes"] = result["notes"]
            if not args.dry_run:
                save_json_utf8(result["annotation_path"], annotation_payload)
        else:
            remaining_records.append(record)

    save_remaining_queue(remaining_queue_path, remaining_records)
    report = {
        "time": now_iso(),
        "queue_file": str(queue_path),
        "dry_run": args.dry_run,
        "policy": {
            "min_confidence": args.min_confidence,
            "score_low": args.score_low,
            "score_high": args.score_high,
            "force_all_remaining": args.force_all_remaining,
        },
        "summary": {
            "queue_total": len(queue_records),
            "auto_reviewed_total": sum(auto_by_action.values()),
            "remaining_total": len(remaining_records),
            "decision_counts": dict(decision_counter),
            "auto_reviewed_by_action": dict(auto_by_action),
        },
        "items": report_items,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("=" * 72)
    print("Agent-assisted queue review")
    print("=" * 72)
    print(f"queue_total        : {len(queue_records)}")
    print(f"auto_reviewed      : {sum(auto_by_action.values())}")
    print(f"remaining_for_human: {len(remaining_records)}")
    print(f"decision_counts    : {dict(decision_counter)}")
    print(f"report_file        : {report_path}")
    print(f"remaining_queue    : {remaining_queue_path}")
    if args.dry_run:
        print("dry_run            : true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
