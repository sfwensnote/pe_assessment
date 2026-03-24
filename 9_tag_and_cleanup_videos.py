#!/usr/bin/env python3
"""
9_tag_and_cleanup_videos.py
为采集视频打标签，并清理无用垃圾视频。

用法:
    python 9_tag_and_cleanup_videos.py
    python 9_tag_and_cleanup_videos.py --cleanup
    python 9_tag_and_cleanup_videos.py --cleanup --dry_run
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="视频打标签与垃圾清理")
    parser.add_argument("--actions", type=str, default="all", help="动作列表，默认 all")
    parser.add_argument("--cleanup", action="store_true", help="启用垃圾视频清理")
    parser.add_argument("--dry_run", action="store_true", help="仅预览清理结果")
    parser.add_argument(
        "--min_bytes",
        type=int,
        default=None,
        help="最小视频体积阈值（默认读取 config ingest.quality_gate.min_filesize_bytes）",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def extract_video_id(filename: str) -> Optional[str]:
    match = re.search(r"pexels_(\d+)", filename)
    if not match:
        return None
    return match.group(1)


def resolve_actions(all_actions: Dict[str, Any], actions_arg: str) -> List[str]:
    if actions_arg.strip().lower() == "all":
        return list(all_actions.keys())
    values = [x.strip() for x in actions_arg.split(",") if x.strip()]
    unknown = [x for x in values if x not in all_actions]
    if unknown:
        raise ValueError(f"未知动作: {', '.join(unknown)}")
    return values


def validate_directory_path(path: Path, label: str) -> tuple[bool, str]:
    if path.is_symlink() and not path.exists():
        return False, f"{label} 路径不可用（符号链接目标不存在）: {path}"
    if path.exists() and not path.is_dir():
        return False, f"{label} 不是目录: {path}"
    return True, ""


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).parent
    config_path = project_root / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    actions = resolve_actions(config.get("actions", {}), args.actions)
    raw_root = project_root / config["paths"]["raw_videos"]
    skeleton_root = project_root / config["paths"]["skeletons"]
    annotation_root = project_root / config["paths"]["annotations"]
    ingest_root = project_root / config["paths"]["processed"] / "ingest"

    ok_raw, msg_raw = validate_directory_path(raw_root, "原始视频目录")
    if not ok_raw:
        print(f"错误: {msg_raw}")
        print("请先确认外接硬盘已挂载，再运行此脚本。")
        return

    min_filesize = int(
        args.min_bytes
        or config.get("ingest", {})
        .get("quality_gate", {})
        .get("min_filesize_bytes", 800000)
    )

    manifest_rows = read_jsonl(ingest_root / "pexels_manifest.jsonl")
    rejected_rows = read_jsonl(ingest_root / "quality_rejected.jsonl")
    rejected_ids: Set[str] = {
        str(row.get("pexels_video_id"))
        for row in rejected_rows
        if row.get("pexels_video_id") is not None
    }

    manifest_by_id: Dict[str, Dict[str, Any]] = {}
    for row in manifest_rows:
        video_id = row.get("pexels_video_id")
        if video_id is None:
            continue
        manifest_by_id[str(video_id)] = row

    tags_path = ingest_root / "video_tags.jsonl"
    cleanup_log_path = ingest_root / "cleanup_log.jsonl"
    if tags_path.exists():
        tags_path.unlink()

    total_files = 0
    removed_files = 0
    tagged_valid = 0
    tagged_garbage = 0

    for action_id in actions:
        action_dir = raw_root / action_id
        if not action_dir.exists():
            continue

        skeleton_ids = {
            p.stem.replace("pexels_", "")
            for p in (skeleton_root / action_id).glob("*.json")
        }
        annotation_ids = {
            p.stem.replace("pexels_", "")
            for p in (annotation_root / action_id).glob("*.json")
        }

        for path in sorted(action_dir.iterdir()):
            if not path.is_file():
                continue

            total_files += 1
            filename = path.name
            suffix = path.suffix.lower()
            size = path.stat().st_size if path.exists() else 0
            video_id = extract_video_id(filename)

            tags: List[str] = [f"action:{action_id}"]
            garbage_reasons: List[str] = []

            if (
                filename.startswith("._")
                or filename.startswith(".")
                or filename == ".DS_Store"
            ):
                garbage_reasons.append("sidecar_or_hidden")

            if suffix not in VIDEO_EXTENSIONS:
                garbage_reasons.append("invalid_extension")

            if size < max(1, int(min_filesize * 0.35)):
                tags.append("small_file")

            if video_id:
                tags.append("from_pexels")
                if video_id in skeleton_ids:
                    tags.append("has_skeleton")
                if video_id in annotation_ids:
                    tags.append("has_annotation")
                if video_id in rejected_ids:
                    tags.append("quality_rejected")
                if video_id in manifest_by_id:
                    tags.append("manifested")
            else:
                tags.append("manual_or_unknown_source")

            if "quality_rejected" in tags and "has_skeleton" not in tags:
                garbage_reasons.append("rejected_without_skeleton")

            if (
                "small_file" in tags
                and "has_skeleton" not in tags
                and "has_annotation" not in tags
            ):
                garbage_reasons.append("too_small_and_unused")

            is_garbage = len(garbage_reasons) > 0
            if is_garbage:
                tagged_garbage += 1
                tags.append("garbage")
            else:
                tagged_valid += 1
                tags.append("usable")

            row = {
                "time": time.time(),
                "action": action_id,
                "filename": filename,
                "path": str(path),
                "video_id": video_id,
                "size": size,
                "tags": tags,
                "garbage_reasons": garbage_reasons,
            }
            append_jsonl(tags_path, row)

            if args.cleanup and is_garbage:
                removed = False
                if not args.dry_run:
                    try:
                        path.unlink(missing_ok=True)
                        removed = True
                    except Exception:
                        removed = False

                append_jsonl(
                    cleanup_log_path,
                    {
                        "time": time.time(),
                        "action": action_id,
                        "filename": filename,
                        "path": str(path),
                        "removed": removed,
                        "dry_run": bool(args.dry_run),
                        "reasons": garbage_reasons,
                    },
                )
                if removed or args.dry_run:
                    removed_files += 1

    print("\n视频标签与清理完成")
    print(f"- 扫描文件数: {total_files}")
    print(f"- 可用文件: {tagged_valid}")
    print(f"- 垃圾候选: {tagged_garbage}")
    if args.cleanup:
        mode = "预览" if args.dry_run else "已删除"
        print(f"- 清理结果: {mode} {removed_files} 个")
    print(f"- 标签文件: {tags_path}")
    if args.cleanup:
        print(f"- 清理日志: {cleanup_log_path}")


if __name__ == "__main__":
    main()
