#!/usr/bin/env python3
"""
实时监控 Step2 处理进度
使用: python monitor_progress.py [--watch]
"""

import json
import time
import argparse
from pathlib import Path
from datetime import datetime

ACTIONS = ["pushup", "squat", "situp", "jump_rope", "long_jump", "pullup"]
PROGRESS_FILE = Path("data/processed/preprocess/progress.json")


def format_time(iso_time):
    """格式化ISO时间"""
    if not iso_time:
        return "-"
    try:
        dt = datetime.fromisoformat(iso_time)
        return dt.strftime("%H:%M:%S")
    except:
        return iso_time[:19] if len(iso_time) > 19 else iso_time


def display_progress():
    """显示当前进度"""
    # 获取进度数据
    progress = {}
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            progress = json.load(f)

    # 获取实际文件数量
    skeleton_root = Path("data/skeletons")
    raw_root = Path("data/raw_videos")

    print("\n" + "=" * 70)
    print(f"Step2 Progress - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print(
        f"{'Action':<12} {'Status':<10} {'Progress':<15} {'Started':<10} {'Completed':<10}"
    )
    print("-" * 70)

    total_skeletons = 0
    total_videos = 0
    completed = 0
    running = 0

    for action in ACTIONS:
        action_dir = skeleton_root / action
        raw_dir = raw_root / action

        json_count = len(list(action_dir.glob("*.json"))) if action_dir.exists() else 0
        video_count = len(list(raw_dir.glob("*.mp4"))) if raw_dir.exists() else 0

        total_skeletons += json_count
        total_videos += video_count

        # 状态
        act_progress = progress.get(action, {})
        status = act_progress.get("status", "unknown")

        if status == "completed":
            status_str = "[DONE]"
            completed += 1
        elif status == "running":
            status_str = "[RUNNING]"
            running += 1
        elif status == "failed":
            status_str = "[FAIL]"
        elif status == "queued":
            status_str = "[QUEUE]"
        else:
            status_str = "[WAIT]"

        # 进度百分比
        pct = json_count / video_count * 100 if video_count > 0 else 0
        progress_str = f"{json_count}/{video_count} ({pct:.1f}%)"

        # 时间
        started = format_time(act_progress.get("started"))
        completed_time = format_time(act_progress.get("completed"))

        print(
            f"{action:<12} {status_str:<10} {progress_str:<15} {started:<10} {completed_time:<10}"
        )

    print("-" * 70)
    total_pct = total_skeletons / total_videos * 100 if total_videos > 0 else 0
    print(
        f"{'Total':<12} {'':<10} {total_skeletons}/{total_videos} ({total_pct:.1f}%){'':<6} [{completed}/{len(ACTIONS)} done, {running} running]"
    )
    print("=" * 70)

    # Show hints
    if running > 0:
        print(f"\n{running} action(s) running...")
        print("Check logs: logs/preprocess_*.log")
    elif completed == len(ACTIONS):
        print("\n[OK] All actions completed!")
    else:
        print(f"\nWaiting: {len(ACTIONS) - completed - running} action(s)")

    return completed == len(ACTIONS)


def watch_mode():
    """Continuous monitoring mode"""
    print("Monitoring (Press Ctrl+C to exit)...")
    try:
        while True:
            display_progress()
            time.sleep(30)  # Refresh every 30 seconds
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped")


def main():
    parser = argparse.ArgumentParser(description="Monitor Step2 preprocessing progress")
    parser.add_argument(
        "--watch", "-w", action="store_true", help="Continuous monitoring mode"
    )
    args = parser.parse_args()

    if args.watch:
        watch_mode()
    else:
        display_progress()
        print("\nTip: Use --watch for continuous monitoring")


if __name__ == "__main__":
    main()
