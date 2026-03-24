#!/usr/bin/env python3
"""实时监控 0_preprocess_videos.py 进度。"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="监控骨骼预处理进展")
    parser.add_argument(
        "--state",
        type=str,
        default="data/processed/preprocess/pipeline_state.json",
        help="状态文件路径",
    )
    parser.add_argument("--watch", action="store_true", help="持续刷新")
    parser.add_argument("--interval", type=float, default=2.0, help="刷新间隔秒")
    parser.add_argument(
        "--stop_when_done",
        action="store_true",
        help="遇到 completed/failed/cancelled 自动退出",
    )
    return parser.parse_args()


def load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def fmt_time(ts: Any) -> str:
    try:
        if ts is None:
            return "-"
        return datetime.fromtimestamp(float(ts)).strftime("%m-%d %H:%M:%S")
    except Exception:
        return "-"


def zh_status(status: str) -> str:
    return {
        "pending": "待处理",
        "running": "进行中",
        "done": "完成",
        "completed": "已完成",
        "failed": "失败",
        "skipped": "跳过",
        "cancelled": "取消",
    }.get(status, status)


def bar(current: int, total: int, width: int = 22) -> str:
    if total <= 0:
        total = 1
    ratio = max(0.0, min(1.0, current / total))
    done = int(ratio * width)
    return f"[{'#' * done}{'-' * (width - done)}] {current}/{total}"


def render(state: Dict[str, Any]) -> str:
    if not state:
        return "尚未检测到预处理状态文件，请先运行: python 0_preprocess_videos.py"

    summary = state.get("summary", {})
    lines = []
    lines.append("=" * 84)
    lines.append("骨骼预处理实时监控")
    lines.append("=" * 84)
    lines.append(
        f"运行ID: {state.get('run_id', '-')} | 状态: {zh_status(str(state.get('status', '-')))}"
    )
    lines.append(
        f"阶段: {state.get('stage', '-')} | 更新时间: {fmt_time(state.get('updated_at'))}"
    )
    lines.append(f"消息: {state.get('message', '')}")
    lines.append(
        "汇总: "
        f"总视频 {int(summary.get('total_videos', 0))} | "
        f"成功 {int(summary.get('processed', 0))} | "
        f"已存在 {int(summary.get('skipped_existing', 0))} | "
        f"过短 {int(summary.get('skipped_short', 0))} | "
        f"失败 {int(summary.get('errors', 0))}"
    )
    lines.append("-" * 84)
    lines.append(
        f"{'动作':<12} {'进度':<34} {'成功':>6} {'已存在':>8} {'过短':>6} {'失败':>6} {'状态':>8}"
    )
    lines.append("-" * 84)

    actions = state.get("actions", {})
    for action_id in sorted(actions.keys()):
        row = actions[action_id]
        name = str(row.get("name") or action_id)
        total = int(row.get("total_videos", 0))
        current = int(row.get("current", 0))
        processed = int(row.get("processed", 0))
        existing = int(row.get("skipped_existing", 0))
        short = int(row.get("skipped_short", 0))
        errors = int(row.get("errors", 0))
        status = zh_status(str(row.get("status", "pending")))
        lines.append(
            f"{name:<12} {bar(current, total):<34} {processed:>6} {existing:>8} {short:>6} {errors:>6} {status:>8}"
        )

    lines.append("-" * 84)
    lines.append("最近事件:")
    events = state.get("events", [])
    if not events:
        lines.append("(暂无)")
    else:
        for item in events[-10:]:
            ts = fmt_time(item.get("time"))
            lv = item.get("level", "info")
            aid = item.get("action_id", "全局")
            msg = item.get("message", "")
            lines.append(f"[{ts}] [{lv}] [{aid}] {msg}")

    if state.get("error"):
        lines.append("-" * 84)
        lines.append(f"错误: {state.get('error')}")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    state_path = Path(__file__).parent / args.state

    if not args.watch:
        print(render(load_state(state_path)))
        return

    while True:
        state = load_state(state_path)
        print("\033[2J\033[H", end="")
        print(render(state))

        if args.stop_when_done and str(state.get("status", "")) in {
            "completed",
            "failed",
            "cancelled",
        }:
            break
        time.sleep(max(0.5, float(args.interval)))


if __name__ == "__main__":
    main()
