#!/usr/bin/env python3
"""Download action videos from Pexels to Lenovo drive."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import requests
import urllib3


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

API_KEY = "Y8yt8M2F0j3J55hC0uT1pOKjEQQ8A62s8tEAymscAkd1Z5MzaHFqkaGf"
BASE_URL = "https://api.pexels.com/videos/search"
TARGET_PER_ACTION = 500
PER_PAGE = 80
MAX_PAGES = 60

ROOT = Path("data/raw_videos")
ACTIONS: Dict[str, str] = {
    "pushup": "俯卧撑",
    "squat": "深蹲",
    "situp": "仰卧起坐",
    "jump_rope": "跳绳",
    "long_jump": "跳远",
    "pullup": "引体向上",
}


def pick_best_mp4(video: dict) -> Optional[str]:
    files = video.get("video_files") or []
    best = None
    best_area = -1
    for item in files:
        link = str(item.get("link") or "")
        if not link or ".mp4" not in link.lower():
            continue
        width = int(item.get("width") or 0)
        height = int(item.get("height") or 0)
        area = width * height
        if area > best_area:
            best_area = area
            best = link
    return best


def count_action_videos(action_dir: Path) -> int:
    exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    return sum(
        1
        for p in action_dir.iterdir()
        if p.is_file() and p.suffix.lower() in exts and not p.name.startswith(".")
    )


def main() -> None:
    if not API_KEY:
        raise RuntimeError("Please set PEXELS_API_KEY before running this script.")

    headers = {"Authorization": API_KEY, "Accept": "application/json"}
    session = requests.Session()
    session.headers.update(headers)

    ROOT.mkdir(parents=True, exist_ok=True)

    for action_id, query in ACTIONS.items():
        action_dir = ROOT / action_id
        action_dir.mkdir(parents=True, exist_ok=True)

        existing_ids = {
            p.stem.replace("pexels_", "")
            for p in action_dir.glob("pexels_*.mp4")
            if p.is_file()
        }

        print(f"\n==> {action_id} ({query})")
        page = 1
        while count_action_videos(action_dir) < TARGET_PER_ACTION and page <= MAX_PAGES:
            params = {
                "query": query,
                "page": page,
                "per_page": PER_PAGE,
                "locale": "zh-CN",
                "orientation": "landscape",
            }
            try:
                resp = session.get(BASE_URL, params=params, timeout=30, verify=False)
                resp.raise_for_status()
                payload = resp.json()
            except Exception as exc:
                print(f"[warn] search failed page={page}: {exc}")
                page += 1
                continue

            videos = payload.get("videos") or []
            if not videos:
                print("[info] no more videos")
                break

            for video in videos:
                if count_action_videos(action_dir) >= TARGET_PER_ACTION:
                    break

                video_id = str(video.get("id") or "")
                if not video_id or video_id in existing_ids:
                    continue

                link = pick_best_mp4(video)
                if not link:
                    continue

                output_path = action_dir / f"pexels_{video_id}.mp4"
                try:
                    with session.get(link, stream=True, timeout=60, verify=False) as dl:
                        dl.raise_for_status()
                        with open(output_path, "wb") as f:
                            for chunk in dl.iter_content(chunk_size=1024 * 512):
                                if chunk:
                                    f.write(chunk)
                    if output_path.stat().st_size < 500_000:
                        output_path.unlink(missing_ok=True)
                        continue
                    existing_ids.add(video_id)
                    print(
                        f"[ok] {action_id}: {count_action_videos(action_dir)}/{TARGET_PER_ACTION}"
                    )
                except Exception as exc:
                    output_path.unlink(missing_ok=True)
                    print(f"[warn] download failed {video_id}: {exc}")

            page += 1

        final_count = count_action_videos(action_dir)
        print(f"[done] {action_id}: {final_count}/{TARGET_PER_ACTION}")


if __name__ == "__main__":
    main()
