#!/usr/bin/env python3
# @author Coder建设｜javpower
"""
0_preprocess_videos.py
视频预处理：提取骨骼关键点 + 进度状态输出

用法:
    python 0_preprocess_videos.py [--action pushup]
    [--input_dir path] [--output_dir path]

支持的动作:
    pushup - 俯卧撑
    squat - 深蹲
    situp - 仰卧起坐
    jump_rope - 跳绳
    long_jump - 跳远
    pullup - 引体向上
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml
from tqdm import tqdm

from ultralytics import YOLO


CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH, encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class PreprocessProgressTracker:
    """Write preprocess state for realtime monitor."""

    def __init__(self, state_path: Path) -> None:
        self.state_path = state_path
        self.state: Dict[str, Any] = {}

    def start(self, run_id: str, actions_state: Dict[str, Dict[str, Any]]) -> None:
        now = time.time()
        self.state = {
            "run_id": run_id,
            "status": "running",
            "stage": "preprocess",
            "message": "开始提取骨骼关键点",
            "started_at": now,
            "updated_at": now,
            "finished_at": None,
            "actions": actions_state,
            "summary": {
                "total_videos": 0,
                "processed": 0,
                "skipped_existing": 0,
                "skipped_short": 0,
                "errors": 0,
            },
            "events": [],
            "error": "",
        }
        self._recompute_summary()
        self._flush()

    def update_action(self, action_id: str, **kwargs: Any) -> None:
        action = self.state.get("actions", {}).get(action_id)
        if not action:
            return
        action.update(kwargs)
        self._recompute_summary()
        self._flush()

    def add_event(
        self,
        level: str,
        message: str,
        action_id: Optional[str] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "time": time.time(),
            "level": level,
            "message": message,
        }
        if action_id:
            payload["action_id"] = action_id

        events = self.state.setdefault("events", [])
        events.append(payload)
        if len(events) > 300:
            self.state["events"] = events[-300:]
        self._flush()

    def set_message(self, message: str) -> None:
        self.state["message"] = message
        self._flush()

    def finish(self, status: str, message: str, error: str = "") -> None:
        self.state["status"] = status
        self.state["message"] = message
        self.state["error"] = error
        self.state["finished_at"] = time.time()
        self._recompute_summary()
        self._flush()

    def _recompute_summary(self) -> None:
        actions = self.state.get("actions", {})
        summary = self.state.setdefault("summary", {})
        summary["total_videos"] = int(
            sum(int(v.get("total_videos", 0)) for v in actions.values())
        )
        summary["processed"] = int(
            sum(int(v.get("processed", 0)) for v in actions.values())
        )
        summary["skipped_existing"] = int(
            sum(int(v.get("skipped_existing", 0)) for v in actions.values())
        )
        summary["skipped_short"] = int(
            sum(int(v.get("skipped_short", 0)) for v in actions.values())
        )
        summary["errors"] = int(sum(int(v.get("errors", 0)) for v in actions.values()))

    def _flush(self) -> None:
        self.state["updated_at"] = time.time()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)


def list_video_files(
    action_input_dir: Path, only_files: Optional[set[str]]
) -> list[Path]:
    try:
        candidates = list(action_input_dir.iterdir())
    except OSError as exc:
        print(f"警告: 无法读取目录 {action_input_dir}: {exc}")
        return []

    values: list[Path] = []
    for p in candidates:
        try:
            if not p.is_file():
                continue
            if p.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            if p.name.startswith("._") or p.name.startswith("."):
                continue
            values.append(p)
        except OSError as exc:
            print(f"警告: 跳过不可读取文件 {p}: {exc}")

    if only_files:
        values = [p for p in values if p.name in only_files or p.stem in only_files]
    return sorted(values)


def extract_skeleton_from_video(
    video_path: Path, model: YOLO, conf_threshold: float = 0.3
) -> dict:
    """从视频中提取骨骼关键点。"""
    results = model(str(video_path), verbose=False, stream=True)

    skeleton_sequence = []
    for frame_idx, result in enumerate(results):
        if result.keypoints is None or len(result.keypoints) == 0:
            continue

        kpts = result.keypoints.xy[0].cpu().numpy()
        if kpts is None or len(kpts) == 0:
            continue

        conf = result.keypoints.conf
        if conf is None or len(conf) == 0:
            scores = [1.0] * len(kpts)
        else:
            scores = conf[0].cpu().numpy().tolist()

        if len(scores) < len(kpts):
            scores.extend([1.0] * (len(kpts) - len(scores)))
        scores = np.asarray(scores, dtype=np.float32)

        kpts[scores < conf_threshold] = [0, 0]

        frame_data = {
            "frame_id": frame_idx,
            "timestamp": frame_idx / CONFIG["camera"]["fps"],
            "keypoints": [
                {
                    "id": i,
                    "name": name,
                    "x": float(x),
                    "y": float(y),
                    "score": float(s),
                }
                for i, (name, (x, y), s) in enumerate(
                    zip(CONFIG["skeleton"]["joint_names"], kpts, scores)
                )
            ],
        }
        skeleton_sequence.append(frame_data)

    return {
        "video_id": video_path.stem,
        "source_path": str(video_path),
        "total_frames": len(skeleton_sequence),
        "fps": CONFIG["camera"]["fps"],
        "skeleton_sequence": skeleton_sequence,
    }


def process_action(
    action_type: str,
    input_dir: Path,
    output_dir: Path,
    model: YOLO,
    conf_threshold: float,
    tracker: Optional[PreprocessProgressTracker] = None,
    only_files: Optional[set[str]] = None,
) -> int:
    """处理单个动作类型的所有视频。"""
    if action_type not in CONFIG["actions"]:
        print(f"警告: 未知的动作类型 '{action_type}'，跳过")
        if tracker:
            tracker.update_action(action_type, status="skipped")
        return 0

    action_input_dir = input_dir / action_type
    if not action_input_dir.exists():
        print(f"警告: 输入目录不存在 {action_input_dir}")
        if tracker:
            tracker.update_action(action_type, status="skipped", total_videos=0)
        return 0

    action_output_dir = output_dir / action_type
    action_output_dir.mkdir(parents=True, exist_ok=True)

    video_files = list_video_files(action_input_dir, only_files)
    if not video_files:
        print(f"警告: 在 {action_input_dir} 中未找到视频文件")
        if tracker:
            tracker.update_action(action_type, status="skipped", total_videos=0)
        return 0

    print(f"\n处理动作: {action_type} ({CONFIG['actions'][action_type]['name']})")
    print(f"找到 {len(video_files)} 个视频文件")

    if tracker:
        tracker.update_action(
            action_type,
            status="running",
            total_videos=len(video_files),
            current=0,
            current_video="",
        )
        tracker.add_event("info", "开始处理动作", action_id=action_type)

    processed_count = 0
    skipped_existing = 0
    skipped_short = 0
    error_count = 0

    for idx, video_path in enumerate(
        tqdm(video_files, desc=f"extract {action_type}", ascii=True),
        start=1,
    ):
        out_path = action_output_dir / f"{video_path.stem}.json"

        if out_path.exists():
            skipped_existing += 1
            if tracker:
                tracker.update_action(
                    action_type,
                    current=idx,
                    current_video=video_path.name,
                    skipped_existing=skipped_existing,
                    processed=processed_count,
                    skipped_short=skipped_short,
                    errors=error_count,
                )
            continue

        try:
            skeleton_data = extract_skeleton_from_video(
                video_path,
                model,
                conf_threshold=conf_threshold,
            )

            if skeleton_data["total_frames"] < 10:
                print(
                    f"  跳过 {video_path.name}: 帧数太少 ({skeleton_data['total_frames']})"
                )
                skipped_short += 1
            else:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(skeleton_data, f, indent=2)
                processed_count += 1
        except Exception as e:
            print(f"  错误处理 {video_path.name}: {e}")
            error_count += 1

        if tracker:
            tracker.update_action(
                action_type,
                current=idx,
                current_video=video_path.name,
                skipped_existing=skipped_existing,
                processed=processed_count,
                skipped_short=skipped_short,
                errors=error_count,
            )

    print(
        f"完成: 成功 {processed_count} 个, 已存在 {skipped_existing} 个, 失败 {error_count} 个"
    )

    if tracker:
        tracker.update_action(
            action_type,
            status="done",
            current=len(video_files),
            processed=processed_count,
            skipped_existing=skipped_existing,
            skipped_short=skipped_short,
            errors=error_count,
        )
        tracker.add_event(
            "info",
            f"动作处理完成: 成功 {processed_count}, 已存在 {skipped_existing}, 失败 {error_count}",
            action_id=action_type,
        )

    return processed_count


def build_actions_state(
    action_ids: list[str], input_dir: Path, only_files: Optional[set[str]]
) -> Dict[str, Dict[str, Any]]:
    values: Dict[str, Dict[str, Any]] = {}
    for action_id in action_ids:
        action_input_dir = input_dir / action_id
        total = 0
        if action_input_dir.exists() and action_input_dir.is_dir():
            total = len(list_video_files(action_input_dir, only_files))

        action_name = (
            CONFIG.get("actions", {}).get(action_id, {}).get("name", action_id)
        )
        values[action_id] = {
            "name": action_name,
            "status": "pending",
            "total_videos": total,
            "current": 0,
            "current_video": "",
            "processed": 0,
            "skipped_existing": 0,
            "skipped_short": 0,
            "errors": 0,
        }
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="提取视频骨骼关键点")
    parser.add_argument(
        "--action", type=str, default=None, help="指定处理的动作类型，默认处理所有"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=CONFIG["paths"]["raw_videos"],
        help="输入视频目录",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=CONFIG["paths"]["skeletons"],
        help="输出骨骼目录",
    )
    parser.add_argument(
        "--model", type=str, default="yolov8x-pose.pt", help="YOLO模型路径"
    )
    parser.add_argument("--conf", type=float, default=0.3, help="关键点置信度阈值")
    parser.add_argument(
        "--only_files",
        type=str,
        default="",
        help="仅处理指定文件（文件名或stem，逗号分隔）",
    )
    parser.add_argument(
        "--state",
        type=str,
        default="data/processed/preprocess/pipeline_state.json",
        help="预处理实时状态文件输出路径",
    )
    args = parser.parse_args()

    only_files = {x.strip() for x in (args.only_files or "").split(",") if x.strip()}
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    action_ids = [args.action] if args.action else list(CONFIG["actions"].keys())
    actions_state = build_actions_state(action_ids, input_dir, only_files or None)
    tracker = PreprocessProgressTracker(Path(args.state))
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    tracker.start(run_id, actions_state)

    print("加载YOLO姿态检测模型...")
    try:
        model = YOLO(args.model)
        print(f"模型加载成功: {args.model}")
        tracker.add_event("info", f"模型加载成功: {args.model}")
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("尝试下载模型...")
        tracker.add_event("warn", f"模型加载失败，尝试默认模型: {e}")
        model = YOLO("yolov8x-pose.pt")

    total_processed = 0
    try:
        for action_type in action_ids:
            count = process_action(
                action_type,
                input_dir,
                output_dir,
                model,
                conf_threshold=float(args.conf),
                tracker=tracker,
                only_files=only_files or None,
            )
            total_processed += count

        print(f"\n{'=' * 50}")
        print(f"全部处理完成！共处理 {total_processed} 个视频")
        print(f"骨骼数据保存在: {output_dir}")
        tracker.finish("completed", f"预处理完成，共成功处理 {total_processed} 个视频")
    except Exception as exc:
        tracker.finish("failed", "预处理失败", error=str(exc))
        raise


if __name__ == "__main__":
    main()
