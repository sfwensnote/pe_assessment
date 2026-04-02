#!/usr/bin/env python3
"""
后台双进程并行运行 Step2 骨骼提取
使用: python run_background_preprocess.py
监控: python monitor_progress.py
"""

import subprocess
import sys
from pathlib import Path
import json
import time
from datetime import datetime

ACTIONS = ["pushup", "squat", "situp", "jump_rope", "long_jump", "pullup"]
PROGRESS_FILE = Path("data/processed/preprocess/progress.json")


def save_progress(status_dict):
    """保存进度到文件"""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(status_dict, f, indent=2, ensure_ascii=False)


def load_progress():
    """加载进度"""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def get_current_counts():
    """获取当前处理数量"""
    skeleton_root = Path("data/skeletons")
    raw_root = Path("data/raw_videos")

    counts = {}
    for action in ACTIONS:
        action_dir = skeleton_root / action
        raw_dir = raw_root / action

        json_count = len(list(action_dir.glob("*.json"))) if action_dir.exists() else 0
        video_count = len(list(raw_dir.glob("*.mp4"))) if raw_dir.exists() else 0

        counts[action] = {
            "skeletons": json_count,
            "videos": video_count,
            "percent": round(json_count / video_count * 100, 1)
            if video_count > 0
            else 0,
        }
    return counts


def process_single_action(action):
    """处理单个动作"""
    cmd = [
        sys.executable,
        "0_preprocess_videos.py",
        "--action",
        action,
        "--model",
        "yolov8n-pose.pt",
    ]

    log_file = Path(f"logs/preprocess_{action}.log")
    log_file.parent.mkdir(exist_ok=True, parents=True)

    # 使用subprocess.Popen启动进程
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
    )

    # 读取输出并保存到日志
    with open(log_file, "w", encoding="utf-8") as f:
        for line in process.stdout:
            f.write(line)
            f.flush()

    process.wait()
    return process.returncode == 0


def main():
    print("=" * 60)
    print("Step2 后台双进程并行处理")
    print("=" * 60)
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("监控进度: python monitor_progress.py")
    print("查看日志: logs/preprocess_*.log")
    print("=" * 60)

    # 初始化进度
    progress = load_progress()
    if not progress:
        progress = {
            action: {"status": "pending", "started": None, "completed": None}
            for action in ACTIONS
        }

    save_progress(progress)

    # 双进程并行处理
    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(max_workers=2) as executor:
        # 提交所有未完成的任务
        futures = {}
        for action in ACTIONS:
            if progress[action]["status"] != "completed":
                progress[action]["status"] = "queued"
                future = executor.submit(process_single_action, action)
                futures[future] = action
                progress[action]["status"] = "running"
                progress[action]["started"] = datetime.now().isoformat()
                save_progress(progress)
                print(f"[START] {action}")

        # 等待完成
        for future in as_completed(futures):
            action = futures[future]
            try:
                success = future.result()
                progress[action]["status"] = "completed" if success else "failed"
                progress[action]["completed"] = datetime.now().isoformat()
                print(f"[{'OK' if success else 'FAIL'}] {action}")
            except Exception as e:
                progress[action]["status"] = "error"
                progress[action]["error"] = str(e)
                print(f"[ERROR] {action}: {e}")

            save_progress(progress)

    print("=" * 60)
    print("所有任务处理完成！")
    print("=" * 60)


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
