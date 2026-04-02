#!/usr/bin/env python3
"""
Step2 完成报告生成器
生成处理统计和可视化报告
"""

import json
from pathlib import Path
from datetime import datetime

ACTIONS = ["pushup", "squat", "situp", "jump_rope", "long_jump", "pullup"]
ACTION_NAMES = {
    "pushup": "Push-up",
    "squat": "Squat",
    "situp": "Sit-up",
    "jump_rope": "Jump Rope",
    "long_jump": "Long Jump",
    "pullup": "Pull-up",
}


def generate_report():
    """生成Step2完成报告"""
    skeleton_root = Path("data/skeletons")
    raw_root = Path("data/raw_videos")

    results = []
    total_skeletons = 0
    total_videos = 0

    print("\n" + "=" * 70)
    print("STEP 2: SKELETON EXTRACTION - FINAL REPORT")
    print("=" * 70)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(
        f"{'Action':<15} {'Videos':>8} {'Skeletons':>10} {'Remaining':>10} {'Progress':>12}"
    )
    print("-" * 70)

    for action in ACTIONS:
        action_dir = skeleton_root / action
        raw_dir = raw_root / action

        json_count = len(list(action_dir.glob("*.json"))) if action_dir.exists() else 0
        video_count = len(list(raw_dir.glob("*.mp4"))) if raw_dir.exists() else 0
        remaining = video_count - json_count
        percent = json_count / video_count * 100 if video_count > 0 else 0

        total_skeletons += json_count
        total_videos += video_count

        status = "[DONE]" if remaining == 0 else "[PARTIAL]"

        print(
            f"{ACTION_NAMES[action]:<15} {video_count:>8} {json_count:>10} {remaining:>10} {percent:>11.1f}% {status}"
        )

        results.append(
            {
                "action": action,
                "videos": video_count,
                "skeletons": json_count,
                "remaining": remaining,
                "percent": round(percent, 1),
            }
        )

    print("-" * 70)
    total_percent = total_skeletons / total_videos * 100 if total_videos > 0 else 0
    total_remaining = total_videos - total_skeletons
    print(
        f"{'TOTAL':<15} {total_videos:>8} {total_skeletons:>10} {total_remaining:>10} {total_percent:>11.1f}%"
    )
    print("=" * 70)

    # Save detailed report
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_videos": total_videos,
            "total_skeletons": total_skeletons,
            "total_remaining": total_remaining,
            "overall_percent": round(total_percent, 1),
            "completed": total_remaining == 0,
        },
        "actions": results,
    }

    report_file = Path("data/processed/preprocess/step2_final_report.json")
    report_file.parent.mkdir(parents=True, exist_ok=True)

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {report_file}")

    # ASCII progress bar
    print("\n" + "=" * 70)
    print("VISUAL PROGRESS")
    print("=" * 70)
    bar_length = 50
    filled = int(total_percent / 100 * bar_length)
    bar = "[" + "=" * filled + ">" + " " * (bar_length - filled - 1) + "]"
    print(f"{bar} {total_percent:.1f}%")
    print("=" * 70)

    if total_remaining == 0:
        print("\n[OK] Step 2 COMPLETED! All videos processed.")
    else:
        print(f"\n[!] Step 2 PARTIAL: {total_remaining} videos remaining")
        print("Run: run_step2_batch.bat to process remaining videos")

    return report


if __name__ == "__main__":
    generate_report()
