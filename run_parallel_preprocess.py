#!/usr/bin/env python3
"""
并行运行 Step2 骨骼提取 - 同时处理所有 6 个动作
"""

import subprocess
import sys
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# 动作列表
ACTIONS = ["pushup", "squat", "situp", "jump_rope", "long_jump", "pullup"]


def process_action(action):
    """处理单个动作的视频"""
    cmd = [
        sys.executable,
        "0_preprocess_videos.py",
        "--action",
        action,
        "--model",
        "yolov8n-pose.pt",
    ]

    log_file = Path(f"logs/preprocess_{action}.log")
    log_file.parent.mkdir(exist_ok=True)

    with open(log_file, "w", encoding="utf-8") as f:
        process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
        return action, process


def monitor_progress():
    """监控所有动作的处理进度"""
    skeleton_root = Path("data/skeletons")

    print("\n" + "=" * 60)
    print("Step2 并行处理进度")
    print("=" * 60)

    for action in ACTIONS:
        action_dir = skeleton_root / action
        if action_dir.exists():
            json_count = len(list(action_dir.glob("*.json")))
            # 获取原始视频数量
            raw_dir = Path("data/raw_videos") / action
            video_count = len(list(raw_dir.glob("*.mp4"))) if raw_dir.exists() else 0
            progress = (
                f"{json_count}/{video_count}" if video_count > 0 else f"{json_count}/?"
            )
            print(f"{action:15s}: {progress:15s} skeleton files")
        else:
            print(f"{action:15s}: 0/?             skeleton files (not started)")

    print("=" * 60)


def main():
    print("启动并行处理 - 同时处理 6 个动作...")
    print("每个动作独立运行，速度提升约 6 倍\n")

    # 启动所有进程
    processes = []
    for action in ACTIONS:
        action, process = process_action(action)
        processes.append((action, process))
        print(f"[启动] {action} (PID: {process.pid})")

    print(f"\n所有 6 个进程已启动！")
    print(f"日志文件保存在: logs/preprocess_*.log")
    print(f"按 Ctrl+C 可以查看当前进度（不会停止处理）\n")

    try:
        # 等待所有进程完成，每 30 秒显示一次进度
        completed = set()
        while len(completed) < len(ACTIONS):
            for action, process in processes:
                if action not in completed:
                    ret = process.poll()
                    if ret is not None:
                        completed.add(action)
                        status = "✓ 完成" if ret == 0 else f"✗ 失败 (code {ret})"
                        print(f"[{status}] {action}")

            if len(completed) < len(ACTIONS):
                time.sleep(30)
                monitor_progress()

        print("\n" + "=" * 60)
        print("所有动作处理完成！")
        print("=" * 60)
        monitor_progress()

    except KeyboardInterrupt:
        print("\n\n用户中断，正在显示当前进度...")
        monitor_progress()
        print("\n注意：后台进程仍在运行！")
        print("查看日志: tail -f logs/preprocess_*.log")


if __name__ == "__main__":
    main()
