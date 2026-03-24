#!/usr/bin/env python3
"""
8_ingest_monitor.py
实时监控 8_ingest_pipeline.py 的执行进展。

用法:
    python 8_ingest_monitor.py --watch
    python 8_ingest_monitor.py
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="监控自动采集与自动载入流水线")
    parser.add_argument(
        "--state",
        type=str,
        default="data/processed/ingest/pipeline_state.json",
        help="状态文件路径",
    )
    parser.add_argument("--watch", action="store_true", help="持续刷新监控")
    parser.add_argument("--interval", type=float, default=2.0, help="刷新间隔（秒）")
    parser.add_argument(
        "--stop_when_done",
        action="store_true",
        help="watch 模式下，检测到 completed/failed/partial/cancelled 后退出",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def load_action_names(config_path: Path) -> Dict[str, str]:
    if not config_path.exists():
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception:
        return {}

    actions = config.get("actions", {}) if isinstance(config, dict) else {}
    mapping: Dict[str, str] = {}
    for action_id, info in actions.items():
        if isinstance(info, dict):
            mapping[str(action_id)] = str(info.get("name", action_id))
    return mapping


def format_time(ts: Any) -> str:
    if ts is None:
        return "-"
    try:
        value = float(ts)
    except Exception:
        return "-"
    return datetime.fromtimestamp(value).strftime("%m-%d %H:%M:%S")


def short_status(value: str) -> str:
    mapping = {
        "pending": "待处理",
        "running": "进行中",
        "done": "完成",
        "failed": "失败",
        "partial": "部分完成",
        "skipped": "跳过",
        "completed": "已完成",
        "cancelled": "已取消",
    }
    return mapping.get(value, value)


def bar(current: int, total: int, width: int = 20) -> str:
    if total <= 0:
        total = 1
    ratio = max(0.0, min(1.0, current / total))
    filled = int(ratio * width)
    return f"[{'#' * filled}{'-' * (width - filled)}] {current}/{total}"


def render(state: Dict[str, Any], action_names: Dict[str, str]) -> str:
    if not state:
        return "尚未检测到状态文件，请先运行: python 8_ingest_pipeline.py"

    run_id = state.get("run_id", "-")
    status = short_status(str(state.get("status", "-")))
    stage = str(state.get("stage", "-"))
    message = str(state.get("message", ""))
    updated_at = format_time(state.get("updated_at"))

    summary = state.get("summary", {})
    scanned_total = int(summary.get("scanned_total", 0))
    added_total = int(summary.get("added_total", 0))
    skeleton_added_total = int(summary.get("skeleton_added_total", 0))
    annotation_added_total = int(summary.get("annotation_added_total", 0))
    duplicate_total = int(summary.get("duplicate_total", 0))
    quality_skipped_total = int(summary.get("quality_skipped_total", 0))
    metadata_skipped_total = int(summary.get("metadata_skipped_total", 0))
    pose_skipped_total = int(summary.get("pose_skipped_total", 0))
    failed_total = int(summary.get("failed_total", 0))
    api_failure_total = int(summary.get("api_failure_total", 0))

    lines = []
    lines.append("=" * 78)
    lines.append("自动采集与自动载入流水线监控")
    lines.append("=" * 78)
    lines.append(f"运行ID: {run_id}")
    lines.append(f"状态: {status} | 阶段: {stage} | 更新时间: {updated_at}")
    lines.append(f"消息: {message}")
    lines.append(
        "汇总: "
        f"扫描 {scanned_total} | 下载 {added_total} | 骨骼 {skeleton_added_total} | "
        f"标注 {annotation_added_total} | 去重 {duplicate_total} | "
        f"质检过滤 {quality_skipped_total}"
        f"(元数据 {metadata_skipped_total}/姿态 {pose_skipped_total}) | "
        f"失败 {failed_total} | API失败 {api_failure_total}"
    )

    pipeline = state.get("pipeline", {})
    lines.append(
        "编排: "
        f"deploy={short_status(str(pipeline.get('deploy', '-')))} | "
        f"preprocess={short_status(str(pipeline.get('preprocess', '-')))} | "
        f"annotate={short_status(str(pipeline.get('annotate', '-')))}"
    )
    lines.append("-" * 78)
    lines.append(
        f"{'动作':<10} {'采集进度':<30} "
        f"{'扫描':>6} {'下载':>6} {'筛掉':>6} {'骨骼':>6} {'标注':>6} "
        f"{'失败':>6} {'预处理':>8} {'标注态':>8}"
    )
    lines.append("-" * 78)

    actions = state.get("actions", {})
    for action_id in sorted(actions.keys()):
        row = actions[action_id]
        name = action_names.get(action_id, row.get("name", action_id))
        target = int(row.get("run_quota", 0))
        added = int(row.get("added_this_run", 0))
        scanned = int(row.get("scanned", 0))
        downloaded = int(row.get("added_this_run", 0))
        filtered = int(row.get("quality_skipped", 0))
        skeleton_added = int(row.get("skeleton_added_this_run", 0))
        annotation_added = int(row.get("annotation_added_this_run", 0))
        failed = int(row.get("failed", 0))
        preprocess_status = short_status(str(row.get("preprocess_status", "-")))
        annotate_status = short_status(str(row.get("annotate_status", "-")))

        progress_text = bar(added, target if target > 0 else 1, width=16)
        lines.append(
            f"{name:<10} {progress_text:<30} {scanned:>6} {downloaded:>6} "
            f"{filtered:>6} {skeleton_added:>6} {annotation_added:>6} {failed:>6} "
            f"{preprocess_status:>8} {annotate_status:>8}"
        )

    events = state.get("events", [])
    lines.append("-" * 78)
    lines.append("最近事件:")
    if not events:
        lines.append("(暂无事件)")
    else:
        for item in events[-8:]:
            ts = format_time(item.get("time"))
            level = item.get("level", "info")
            action_id = item.get("action_id")
            action_name = (
                action_names.get(str(action_id), str(action_id)) if action_id else "全局"
            )
            msg = item.get("message", "")
            lines.append(f"[{ts}] [{level}] [{action_name}] {msg}")

    error_message = str(state.get("error") or "").strip()
    if error_message:
        lines.append("-" * 78)
        lines.append(f"错误: {error_message}")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).parent
    state_path = project_root / args.state
    action_names = load_action_names(project_root / "config.yaml")

    if not args.watch:
        state = load_json(state_path)
        print(render(state, action_names))
        return

    while True:
        state = load_json(state_path)
        print("\033[2J\033[H", end="")
        print(render(state, action_names))

        status = str(state.get("status", ""))
        if args.stop_when_done and status in {
            "completed",
            "failed",
            "partial",
            "cancelled",
        }:
            break
        time.sleep(max(0.5, float(args.interval)))


if __name__ == "__main__":
    main()
