#!/usr/bin/env python3
"""Train a classical action recognition model on compact summary features."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import yaml

from utils.action_features import extract_action_summary_features
from utils.annotation_io import load_json_any
from utils.skeleton import SkeletonProcessor
from utils.training_manifest import read_manifest, resolve_project_path


PROJECT_ROOT = Path(__file__).parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train random-forest action recognizer"
    )
    parser.add_argument(
        "--manifest_file",
        type=str,
        default="data/processed/review/training_manifests/action_excluding_reviewed_fail.jsonl",
        help="Training manifest JSONL",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/action_rf_round",
        help="Output directory",
    )
    parser.add_argument("--trees", type=int, default=400, help="Number of trees")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_dataset(manifest_file: str) -> tuple[np.ndarray, list[str]]:
    manifest_path = resolve_project_path(manifest_file)
    records = read_manifest(manifest_path)
    processor = SkeletonProcessor(target_frames=CONFIG["skeleton"]["target_frames"])

    X = []
    y = []
    for record in records:
        action = record["action_type"]
        skeleton_path = resolve_project_path(record["skeleton_path"])
        if not skeleton_path.exists():
            continue
        skeleton_data, _ = load_json_any(skeleton_path)
        sequence = []
        for frame in skeleton_data["skeleton_sequence"]:
            coords = [[kp["x"], kp["y"]] for kp in frame["keypoints"]]
            sequence.append(coords)
        sequence = np.asarray(sequence, dtype=float)
        features = processor.process(sequence)
        X.append(extract_action_summary_features(sequence, features))
        y.append(action)

    return np.asarray(X, dtype=float), y


def main() -> int:
    args = parse_args()
    checkpoint_dir = resolve_project_path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_dataset(args.manifest_file)
    if len(X) < 20:
        print("Not enough samples for RF training")
        return 1

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=args.seed,
        stratify=y,
    )

    clf = RandomForestClassifier(
        n_estimators=args.trees,
        random_state=args.seed,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    pred = clf.predict(X_val)
    probs = clf.predict_proba(X_val)
    acc = float(accuracy_score(y_val, pred))

    joblib.dump(clf, checkpoint_dir / "action_model_rf.joblib")
    summary = {
        "manifest_file": str(resolve_project_path(args.manifest_file)),
        "samples_total": int(len(X)),
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "class_counts": dict(Counter(y)),
        "val_accuracy": acc,
        "classes": list(clf.classes_),
        "feature_dim": int(X.shape[1]),
        "trees": args.trees,
        "seed": args.seed,
        "classification_report": classification_report(y_val, pred, output_dict=True),
    }
    with open(
        checkpoint_dir / "action_model_rf_summary.json", "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 72)
    print("Random-forest action model trained")
    print("=" * 72)
    print(f"samples_total : {len(X)}")
    print(f"val_accuracy  : {acc:.4f}")
    print(f"checkpoint    : {checkpoint_dir / 'action_model_rf.joblib'}")
    print(f"summary       : {checkpoint_dir / 'action_model_rf_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
