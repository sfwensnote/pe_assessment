"""Helpers for building and consuming training manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from utils.annotation_io import load_json_any


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except Exception:
        return str(path)


def resolve_project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def review_tier(annotation: dict[str, Any]) -> str:
    if not annotation.get("reviewed"):
        return "unreviewed"

    review_source = annotation.get("review_source")
    decision = annotation.get("review_decision")

    if review_source != "agent_auto_review":
        return "manual"
    if decision in {"confirmed_fail", "confirmed_pass"}:
        return "confirmed"
    if decision == "provisional_pass":
        return "provisional_pass"
    if decision == "provisional_fail":
        return "provisional_fail"
    return "reviewed_other"


def read_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    try:
        payload, _ = load_json_any(path)
    except Exception:
        payload = None
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]

    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        if isinstance(item, dict):
            records.append(item)
    return records


def write_manifest(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
