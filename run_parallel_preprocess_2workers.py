#!/usr/bin/env python3
"""
双进程并行运行 Step2 骨骼提取 - 同时处理 2 个动作
适合配置较低的电脑
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
    print("Step2 双进程并行处理进度")
    print("=" * 60)

    total_skeletons = 0
    total_videos = 0

    for action in ACTIONS:
        action_dir = skeleton_root / action
        raw_dir = Path("data/raw_videos") / action

        json_count = len(list(action_dir.glob("*.json"))) if action_dir.exists() else 0
        video_count = len(list(raw_dir.glob("*.mp4"))) if raw_dir.exists() else 0

        total_skeletons += json_count
        total_videos += video_count

        progress_pct = (json_count / video_count * 100) if video_count > 0 else 0
        if json_count >= video_count and video_count > 0:
            status = "[DONE]"
        elif json_count > 0:
            status = "[RUNNING]"
        else:
            status = "[PENDING]"
        print(
            f"{action:15s}: {json_count:4d}/{video_count:4d} ({progress_pct:5.1f}%) {status}"
        )

    overall_pct = (total_skeletons / total_videos * 100) if total_videos > 0 else 0
    print("-" * 60)
    print(
        f"总计          : {total_skeletons:4d}/{total_videos:4d} ({overall_pct:5.1f}%)"
    )
    print("=" * 60)


def main():
    print("启动双进程并行处理 - 同时处理 2 个动作")
    print("适合配置较低的电脑，减少 CPU/内存压力\n")

    # 检查当前进度
    print("当前进度：")
    monitor_progress()
    print()

    # 使用双进程处理（max_workers=2）
    print("开始双进程并行处理...")
    print("=" * 60)

    completed = set()
    failed = set()

    with ProcessPoolExecutor(max_workers=2) as executor:
        # 提交所有任务
        future_to_action = {
            executor.submit(process_action_worker, action): action
            for action in ACTIONS
            if action not in completed
        }

        # 处理完成的任务
        for future in as_completed(future_to_action):
            action = future_to_action[future]
            try:
                result = future.result()
                if result["success"]:
                    completed.add(action)
                    print(f"[OK] {action}")
                else:
                    failed.add(action)
                    print(f"[FAIL] {action}: {result.get('error', 'Unknown error')}")
            except Exception as e:
                failed.add(action)
                print(f"[ERROR] {action}: {e}")

            # 显示当前进度
            monitor_progress()

    print("\n" + "=" * 60)
    print("双进程并行处理完成！")
    print("=" * 60)
    monitor_progress()

    if failed:
        print(f"\n失败的动作: {', '.join(failed)}")
        print("查看日志: logs/preprocess_*.log")


def process_action_worker(action):
    """工作进程：处理单个动作"""
    import subprocess
    import sys
    from pathlib import Path

    try:
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

        with open(log_file, "w", encoding="utf-8") as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=7200,  # 2小时超时
            )

        return {
            "action": action,
            "success": result.returncode == 0,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"action": action, "success": False, "error": "Timeout (2 hours)"}
    except Exception as e:
        return {"action": action, "success": False, "error": str(e)}


if __name__ == "__main__":
    # Windows 需要这行来支持多进程
    import multiprocessing

    multiprocessing.freeze_support()
    main()
