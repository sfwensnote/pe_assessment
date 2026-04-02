#!/usr/bin/env python3
"""Build a mixed checkpoint bundle from baseline and high-quality retraining outputs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).parent
ACTIONS = ["pushup", "squat", "situp", "jump_rope", "long_jump", "pullup"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build mixed-best checkpoint bundle")
    parser.add_argument(
        "--baseline_dir",
        type=str,
        default="checkpoints",
        help="Baseline checkpoint directory",
    )
    parser.add_argument(
        "--high_quality_dir",
        type=str,
        default="checkpoints/high_quality_round",
        help="High-quality retraining checkpoint directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for the mixed bundle",
    )
    parser.add_argument(
        "--action_source",
        choices=["baseline", "high_quality"],
        default="high_quality",
        help="Which action checkpoint to keep in the mixed bundle",
    )
    parser.add_argument(
        "--rf_dir",
        type=str,
        default="checkpoints/action_rf_round",
        help="Optional directory containing action_model_rf.joblib",
    )
    return parser.parse_args()


def resolve(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / raw


def load_checkpoint_meta(path: Path) -> dict[str, Any]:
    data = torch.load(path, map_location="cpu")
    return data if isinstance(data, dict) else {}


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> int:
    args = parse_args()
    baseline_dir = resolve(args.baseline_dir)
    high_quality_dir = resolve(args.high_quality_dir)
    output_dir = resolve(args.output_dir)
    rf_dir = resolve(args.rf_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "action_source": args.action_source,
        "baseline_dir": str(baseline_dir),
        "high_quality_dir": str(high_quality_dir),
        "rf_dir": str(rf_dir),
        "output_dir": str(output_dir),
        "files": {},
    }

    action_src = (
        baseline_dir / "action_model_best.pth"
        if args.action_source == "baseline"
        else high_quality_dir / "action_model_best.pth"
    )
    copy_file(action_src, output_dir / "action_model_best.pth")
    action_meta = load_checkpoint_meta(action_src)
    summary["files"]["action_model_best.pth"] = {
        "source": str(action_src),
        "best_acc": float(action_meta.get("best_acc", 0.0)),
    }

    rf_model = rf_dir / "action_model_rf.joblib"
    rf_summary = rf_dir / "action_model_rf_summary.json"
    if rf_model.exists():
        copy_file(rf_model, output_dir / "action_model_rf.joblib")
        summary["files"]["action_model_rf.joblib"] = {"source": str(rf_model)}
    if rf_summary.exists():
        copy_file(rf_summary, output_dir / "action_model_rf_summary.json")

    baseline_quality = baseline_dir / "quality_model_best.pth"
    high_quality_quality = high_quality_dir / "quality_model_best.pth"
    baseline_quality_meta = load_checkpoint_meta(baseline_quality)
    high_quality_quality_meta = load_checkpoint_meta(high_quality_quality)
    chosen_quality = (
        high_quality_quality
        if float(high_quality_quality_meta.get("best_val_loss", 1e9))
        <= float(baseline_quality_meta.get("best_val_loss", 1e9))
        else baseline_quality
    )
    copy_file(chosen_quality, output_dir / "quality_model_best.pth")
    summary["files"]["quality_model_best.pth"] = {
        "source": str(chosen_quality),
        "baseline_best_val_loss": float(
            baseline_quality_meta.get("best_val_loss", 0.0)
        ),
        "high_quality_best_val_loss": float(
            high_quality_quality_meta.get("best_val_loss", 0.0)
        ),
    }

    for action in ACTIONS:
        base_path = baseline_dir / f"phase_model_{action}.pth"
        high_path = high_quality_dir / f"phase_model_{action}.pth"
        base_meta = load_checkpoint_meta(base_path)
        high_meta = load_checkpoint_meta(high_path)
        base_f1 = float(base_meta.get("best_f1", 0.0))
        high_f1 = float(high_meta.get("best_f1", 0.0))
        chosen = high_path if high_f1 >= base_f1 else base_path
        copy_file(chosen, output_dir / f"phase_model_{action}.pth")
        summary["files"][f"phase_model_{action}.pth"] = {
            "source": str(chosen),
            "baseline_best_f1": base_f1,
            "high_quality_best_f1": high_f1,
        }

    summary_path = output_dir / "bundle_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 72)
    print("Mixed bundle created")
    print("=" * 72)
    print(f"output_dir    : {output_dir}")
    print(f"action_source : {args.action_source}")
    print(f"summary_file  : {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
