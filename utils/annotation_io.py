"""Utilities for reading and writing annotation JSON files safely."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Tuple


JSON_ENCODINGS = ("utf-8", "utf-8-sig", "gb18030", "gbk")


def load_json_any(path: Path) -> Tuple[Any, str]:
    """Load JSON with common Windows/UTF encodings.

    Returns the parsed payload and the encoding that succeeded.
    """

    raw = path.read_bytes()
    errors = []
    for encoding in JSON_ENCODINGS:
        try:
            return json.loads(raw.decode(encoding)), encoding
        except Exception as exc:  # pragma: no cover - diagnostic path
            errors.append(f"{encoding}: {exc}")
    raise UnicodeError(f"Unable to decode JSON file {path}: {' | '.join(errors)}")


def save_json_utf8(path: Path, payload: Any) -> None:
    """Write JSON in UTF-8 for stable cross-platform handling."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def normalize_json_to_utf8(path: Path) -> str:
    """Normalize a JSON file to UTF-8 and return the source encoding."""

    payload, encoding = load_json_any(path)
    save_json_utf8(path, payload)
    return encoding
