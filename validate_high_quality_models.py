#!/usr/bin/env python3
"""Run end-to-end validation using the high-quality retrained checkpoints."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from utils.annotation_io import load_json_any


PROJECT_ROOT = Path(__file__).parent
ACTIONS = ["pushup", "squat", "situp", "jump_rope", "long_jump", "pullup"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate high-quality retrained models"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/high_quality_round",
        help="Checkpoint directory for the retrained models",
    )
    parser.add_argument(
        "--manifest_dir",
        type=str,
        default="data/processed/review/high_quality_manifests/by_action/action_high_quality_balanced",
        help="Per-action manifest directory used to pick validation videos",
    )
    parser.add_argument(
        "--raw_video_dir",
        type=str,
        default="data/raw_videos",
        help="Raw video root directory",
    )
    parser.add_argument(
        "--pose_model",
        type=str,
        default="yolov8n-pose.pt",
        help="YOLO pose model used during validation",
    )
    parser.add_argument(
        "--candidate_limit",
        type=int,
        default=10,
        help="How many per-action manifest entries to scan when choosing a short video",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference device",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/processed/validation/high_quality_validation.json",
        help="Validation report JSON path",
    )
    return parser.parse_args()


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path_str


def load_inference_module():
    module_path = PROJECT_ROOT / "6_inference.py"
    spec = importlib.util.spec_from_file_location("inference_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_manifest_entries(path: Path, limit: int) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines()[:limit]:
        line = line.strip()
        if not line:
            continue
        entries.append(json.loads(line))
    return entries


def video_meta(path: Path) -> tuple[bool, int, float]:
    cap = cv2.VideoCapture(str(path))
    opened = cap.isOpened()
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    return opened, frames, fps


def select_video_for_action(
    manifest_path: Path, raw_video_root: Path, action: str, candidate_limit: int
) -> dict[str, Any] | None:
    entries = load_manifest_entries(manifest_path, candidate_limit)
    best = None
    for item in entries:
        video_path = raw_video_root / action / f"{item['video_id']}.mp4"
        if not video_path.exists():
            continue
        opened, frames, fps = video_meta(video_path)
        if not opened or frames <= 0:
            continue
        candidate = {
            "video_id": item["video_id"],
            "action_type": action,
            "video_path": video_path,
            "annotation_path": resolve(item["annotation_path"]),
            "frames": frames,
            "fps": fps,
        }
        if best is None or candidate["frames"] < best["frames"]:
            best = candidate
    return best


def score_delta(predicted: float, expected: float) -> float:
    return round(abs(float(predicted) - float(expected)), 2)


def error_overlap(predicted: list[str], expected: list[str]) -> float:
    predicted_set = set(predicted)
    expected_set = set(expected)
    union = predicted_set | expected_set
    if not union:
        return 1.0
    return round(len(predicted_set & expected_set) / len(union), 4)


def assess_with_optional_hint(
    assessor,
    skeleton_sequence: np.ndarray,
    features: np.ndarray,
    action_hint: str | None,
):
    if action_hint is None:
        action_type, confidence, action_source = assessor.recognize_action(
            features, skeleton_sequence
        )
    else:
        action_type = action_hint
        confidence = 1.0
        action_source = "user_hint"
    phases = assessor.segment_phases(features, action_type)
    quality = assessor.assess_quality(features, action_type)
    return {
        "action_type": action_type,
        "confidence": float(confidence),
        "action_source": action_source,
        "phase_count": len(
            set(phases.tolist() if hasattr(phases, "tolist") else phases)
        ),
        "quality": quality,
    }


def main() -> int:
    args = parse_args()
    checkpoint_dir = resolve(args.checkpoint_dir)
    manifest_dir = resolve(args.manifest_dir)
    raw_video_root = resolve(args.raw_video_dir)
    output_path = resolve(args.output_file)

    infer_mod = load_inference_module()
    yolo_cls = infer_mod.YOLO
    infer_mod.YOLO = lambda _path: yolo_cls(args.pose_model)
    assessor = infer_mod.ActionAssessor(
        checkpoint_dir=str(checkpoint_dir), device=args.device
    )

    records: list[dict[str, Any]] = []
    started = time.time()

    for action in ACTIONS:
        manifest_path = manifest_dir / f"{action}.jsonl"
        selected = select_video_for_action(
            manifest_path, raw_video_root, action, args.candidate_limit
        )
        if selected is None:
            records.append(
                {"action_type": action, "ok": False, "error": "No valid video found"}
            )
            continue

        annotation, _ = load_json_any(selected["annotation_path"])
        expected_quality = annotation.get("quality", {})
        skeleton_sequence = assessor.extract_skeleton(str(selected["video_path"]))
        if skeleton_sequence is None:
            records.append(
                {
                    "action_type": action,
                    "video_id": selected["video_id"],
                    "video_path": str(selected["video_path"]),
                    "ok": False,
                    "error": "No skeleton detected during validation",
                }
            )
            continue

        features = assessor.processor.process(skeleton_sequence)
        full_result = assess_with_optional_hint(
            assessor, skeleton_sequence, features, action_hint=None
        )
        hinted_result = assess_with_optional_hint(
            assessor, skeleton_sequence, features, action_hint=action
        )

        record = {
            "action_type": action,
            "video_id": selected["video_id"],
            "video_path": str(selected["video_path"]),
            "frames_in_video": selected["frames"],
            "fps": round(selected["fps"], 2),
            "detected_frames": int(len(skeleton_sequence)),
            "expected": {
                "action_type": action,
                "score": float(expected_quality.get("overall_score", 0.0) or 0.0),
                "errors": list(expected_quality.get("errors", []) or []),
                "is_standard": bool(expected_quality.get("is_standard", False)),
            },
            "full_inference": {
                "predicted_action": full_result["action_type"],
                "action_match": full_result["action_type"] == action,
                "confidence": round(full_result["confidence"], 4),
                "action_source": full_result["action_source"],
                "phase_count": full_result["phase_count"],
                "score": round(float(full_result["quality"]["overall_score"]), 2),
                "errors": list(full_result["quality"].get("errors", []) or []),
                "score_delta_vs_expected": score_delta(
                    full_result["quality"]["overall_score"],
                    expected_quality.get("overall_score", 0.0),
                ),
                "error_overlap": error_overlap(
                    full_result["quality"].get("errors", []),
                    expected_quality.get("errors", []),
                ),
            },
            "hinted_inference": {
                "phase_count": hinted_result["phase_count"],
                "score": round(float(hinted_result["quality"]["overall_score"]), 2),
                "errors": list(hinted_result["quality"].get("errors", []) or []),
                "score_delta_vs_expected": score_delta(
                    hinted_result["quality"]["overall_score"],
                    expected_quality.get("overall_score", 0.0),
                ),
                "error_overlap": error_overlap(
                    hinted_result["quality"].get("errors", []),
                    expected_quality.get("errors", []),
                ),
            },
            "ok": True,
        }
        records.append(record)

    ok_records = [record for record in records if record.get("ok")]
    action_match_count = sum(
        1 for record in ok_records if record["full_inference"]["action_match"]
    )
    report = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint_dir": str(checkpoint_dir),
        "pose_model": args.pose_model,
        "device": args.device,
        "elapsed_sec": round(time.time() - started, 2),
        "summary": {
            "videos_attempted": len(records),
            "videos_ok": len(ok_records),
            "action_match_count": action_match_count,
            "action_match_rate": round(action_match_count / len(ok_records), 4)
            if ok_records
            else 0.0,
            "avg_full_score_delta": round(
                sum(
                    record["full_inference"]["score_delta_vs_expected"]
                    for record in ok_records
                )
                / len(ok_records),
                2,
            )
            if ok_records
            else None,
            "avg_hinted_score_delta": round(
                sum(
                    record["hinted_inference"]["score_delta_vs_expected"]
                    for record in ok_records
                )
                / len(ok_records),
                2,
            )
            if ok_records
            else None,
        },
        "records": records,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("=" * 72)
    print("High-quality model validation")
    print("=" * 72)
    print(f"checkpoint_dir      : {checkpoint_dir}")
    print(f"pose_model          : {args.pose_model}")
    print(f"videos_ok           : {len(ok_records)}/{len(records)}")
    print(f"action_match_rate   : {report['summary']['action_match_rate']:.2%}")
    print(f"avg_full_score_delta: {report['summary']['avg_full_score_delta']}")
    print(f"avg_hint_score_delta: {report['summary']['avg_hinted_score_delta']}")
    print(f"output_file         : {output_path}")
    for record in ok_records:
        print(
            f"- {record['action_type']}: predicted={record['full_inference']['predicted_action']} "
            f"match={record['full_inference']['action_match']} "
            f"full_score={record['full_inference']['score']} "
            f"hint_score={record['hinted_inference']['score']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
