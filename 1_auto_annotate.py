#!/usr/bin/env python3
# @author Coder建设｜javpower
"""
1_auto_annotate.py
自动标注工具：基于规则自动标注动作阶段和质量

用法:
    python 1_auto_annotate.py [--action pushup] [--review]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

# 添加项目路径
sys.path.append(str(Path(__file__).parent))
from utils.skeleton import SkeletonProcessor

# 加载配置
CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)


class AutoAnnotator:
    """自动标注器"""

    def __init__(self):
        self.processor = SkeletonProcessor(
            target_frames=CONFIG["skeleton"]["target_frames"]
        )

    def annotate_pushup(self, features: np.ndarray) -> tuple:
        """
        俯卧撑自动标注

        Returns:
            phases: 阶段标签列表
            quality: 质量评估字典
        """
        T = len(features)

        # 提取关键指标
        left_elbow = features[:, 7, 3]
        right_elbow = features[:, 8, 3]
        elbow_angles = (left_elbow + right_elbow) / 2

        hip_y = features[:, 11, 1]

        # 找到最低点
        bottom_idx = np.argmin(elbow_angles)

        # 阶段分割（基于肘部角度）
        phases = np.zeros(T, dtype=int)

        # 使用自适应阈值
        ready_threshold = 160
        down_threshold = 90

        phases[elbow_angles > ready_threshold] = 0  # ready
        phases[
            (elbow_angles > down_threshold) & (elbow_angles <= ready_threshold)
        ] = 1  # down
        phases[elbow_angles <= down_threshold] = 2  # bottom
        phases[(elbow_angles > down_threshold) & (np.arange(T) > bottom_idx)] = 3  # up
        phases[
            (elbow_angles > ready_threshold) & (np.arange(T) > bottom_idx)
        ] = 4  # finish

        # 质量评估
        quality = {
            "bottom_elbow_angle": float(elbow_angles[bottom_idx]),
            "body_straightness": float(180 - abs(180 - np.mean(features[:, 11, 5]))),
            "speed_down": float(bottom_idx / 30.0),
            "speed_up": float((T - bottom_idx) / 30.0),
            "is_standard": elbow_angles[bottom_idx] <= 90,
            "errors": [],
        }

        # 错误检测
        if elbow_angles[bottom_idx] > 90:
            quality["errors"].append("未达深度")

        hip_range = np.max(hip_y) - np.min(hip_y)
        if hip_range > 0.15:
            quality["errors"].append(
                "塌腰" if hip_y[bottom_idx] > np.mean(hip_y) else "撅臀"
            )

        # 计算得分
        from utils.metrics import compute_scores

        overall_score, details = compute_scores(quality, CONFIG["actions"]["pushup"])
        quality["overall_score"] = overall_score
        quality["details"] = details

        return phases.tolist(), quality

    def annotate_squat(self, features: np.ndarray) -> tuple:
        """深蹲自动标注"""
        T = len(features)

        left_knee = features[:, 13, 4]
        right_knee = features[:, 14, 4]
        knee_angles = (left_knee + right_knee) / 2

        bottom_idx = np.argmin(knee_angles)

        # 阶段分割
        phases = np.zeros(T, dtype=int)
        phases[: bottom_idx // 3] = 0  # standing
        phases[bottom_idx // 3 : bottom_idx] = 1  # descent
        phases[max(0, bottom_idx - 2) : min(T, bottom_idx + 3)] = 2  # bottom
        phases[bottom_idx : bottom_idx + 2 * (T - bottom_idx) // 3] = 3  # ascent
        phases[bottom_idx + 2 * (T - bottom_idx) // 3 :] = 4  # lockout

        # 膝盖内扣检测
        left_knee_x = features[:, 11, 0]
        left_ankle_x = features[:, 15, 0]
        knee_collapse = np.max(np.abs(left_knee_x - left_ankle_x))

        quality = {
            "depth_knee_angle": float(knee_angles[bottom_idx]),
            "knee_collapse": float(knee_collapse),
            "is_standard": knee_angles[bottom_idx] <= 90,
            "errors": [],
        }

        if knee_angles[bottom_idx] > 100:
            quality["errors"].append("未达深度")
        if knee_collapse > 0.1:
            quality["errors"].append("膝盖内扣")

        from utils.metrics import compute_scores

        overall_score, details = compute_scores(quality, CONFIG["actions"]["squat"])
        quality["overall_score"] = overall_score
        quality["details"] = details

        return phases.tolist(), quality

    def annotate_situp(self, features: np.ndarray) -> tuple:
        """仰卧起坐自动标注"""
        T = len(features)

        shoulder_y = features[:, 5, 1]
        hip_y = features[:, 11, 1]
        wrist_y = features[:, 9, 1]
        knee_y = features[:, 13, 1]

        # 检测最高点（起坐完成）
        top_idx = np.argmin(shoulder_y)

        # 阶段分割
        phases = np.zeros(T, dtype=int)
        phases[: T // 4] = 0  # lying
        phases[T // 4 : top_idx] = 1  # up
        phases[top_idx : top_idx + 3] = 2  # touch
        phases[top_idx + 3 : 3 * T // 4] = 3  # down
        phases[3 * T // 4 :] = 4  # back

        # 质量评估
        touch_distance = abs(wrist_y[top_idx] - knee_y[top_idx])
        hip_variance = np.var(hip_y)

        quality = {
            "up_angle": float(
                np.degrees(np.arctan2(hip_y[0] - shoulder_y[top_idx], 0.5))
            ),
            "touch_distance": float(touch_distance),
            "hip_stability": float(hip_variance),
            "is_standard": touch_distance < 0.15 and hip_variance < 0.05,
            "errors": [],
        }

        if touch_distance > 0.15:
            quality["errors"].append("未触膝")
        if hip_variance > 0.05:
            quality["errors"].append("臀部离地")

        from utils.metrics import compute_scores

        overall_score, details = compute_scores(quality, CONFIG["actions"]["situp"])
        quality["overall_score"] = overall_score
        quality["details"] = details

        return phases.tolist(), quality

    def annotate_jump_rope(self, features: np.ndarray) -> tuple:
        """跳绳自动标注"""
        T = len(features)

        ankle_y = features[:, 15, 1]
        knee_angles = features[:, 13, 4]

        # 检测跳跃周期
        from scipy.signal import find_peaks

        peaks, _ = find_peaks(-ankle_y, distance=10)

        # 阶段分割
        phases = np.zeros(T, dtype=int)
        if len(peaks) >= 2:
            for i, peak in enumerate(peaks):
                if i < len(peaks) - 1:
                    start = peak
                    end = peaks[i + 1]
                    mid = (start + end) // 2
                    phases[start:mid] = 1  # jump
                    phases[mid:end] = 2  # air
                    phases[end : min(end + 5, T)] = 3  # land

        # 质量评估
        jump_height = np.max(ankle_y) - np.min(ankle_y)
        landing_flexion = np.max(knee_angles) - np.min(knee_angles)

        if len(peaks) > 1:
            intervals = np.diff(peaks)
            rhythm_regularity = np.std(intervals) / (np.mean(intervals) + 1e-6)
        else:
            rhythm_regularity = 1.0

        quality = {
            "jump_height": float(jump_height),
            "landing_softness": float(landing_flexion),
            "rhythm_regularity": float(rhythm_regularity),
            "is_standard": rhythm_regularity < 0.3 and landing_flexion > 15,
            "errors": [],
        }

        if landing_flexion < 10:
            quality["errors"].append("膝盖过直")
        if rhythm_regularity > 0.3:
            quality["errors"].append("节奏不稳")

        from utils.metrics import compute_scores

        overall_score, details = compute_scores(quality, CONFIG["actions"]["jump_rope"])
        quality["overall_score"] = overall_score
        quality["details"] = details

        return phases.tolist(), quality

    def annotate_long_jump(self, features: np.ndarray) -> tuple:
        """跳远自动标注"""
        T = len(features)

        hip_y = features[:, 11, 1]

        # 检测起跳点（髋部最低点）
        takeoff_idx = np.argmin(hip_y)

        # 阶段分割
        phases = np.zeros(T, dtype=int)
        phases[: takeoff_idx // 2] = 0  # runup
        phases[takeoff_idx // 2 : takeoff_idx] = 1  # takeoff
        phases[takeoff_idx : 3 * T // 4] = 2  # flight
        phases[3 * T // 4 :] = 3  # landing

        # 质量评估
        quality = {
            "takeoff_angle": 20.0,  # 简化计算
            "takeoff_knee_angle": float(features[takeoff_idx, 13, 4]),
            "is_standard": True,
            "errors": [],
        }

        from utils.metrics import compute_scores

        overall_score, details = compute_scores(quality, CONFIG["actions"]["long_jump"])
        quality["overall_score"] = overall_score
        quality["details"] = details

        return phases.tolist(), quality

    def annotate_pullup(self, features: np.ndarray) -> tuple:
        """引体向上自动标注"""
        T = len(features)

        nose_y = features[:, 0, 1]
        elbow_angles = features[:, 7, 3]

        # 检测最高点
        top_idx = np.argmin(nose_y)

        # 阶段分割
        phases = np.zeros(T, dtype=int)
        phases[: T // 5] = 0  # hang
        phases[T // 5 : top_idx] = 1  # pull
        phases[top_idx : top_idx + 3] = 2  # chin_over
        phases[top_idx + 3 : 4 * T // 5] = 3  # lower
        phases[4 * T // 5 :] = 4  # finish

        # 质量评估
        max_height = features[0, 0, 1] - nose_y[top_idx]
        min_elbow = np.min(elbow_angles)
        max_elbow = np.max(elbow_angles)

        quality = {
            "chin_height": float(max_height),
            "min_elbow_angle": float(min_elbow),
            "max_elbow_angle": float(max_elbow),
            "is_standard": max_height > 0.05 and max_elbow > 150,
            "errors": [],
        }

        if max_height < 0.05:
            quality["errors"].append("未过杆")
        if max_elbow < 150:
            quality["errors"].append("未充分下放")

        from utils.metrics import compute_scores

        overall_score, details = compute_scores(quality, CONFIG["actions"]["pullup"])
        quality["overall_score"] = overall_score
        quality["details"] = details

        return phases.tolist(), quality

    def annotate(self, features: np.ndarray, action_type: str) -> tuple:
        """根据动作类型调用对应的标注方法"""
        annotators = {
            "pushup": self.annotate_pushup,
            "squat": self.annotate_squat,
            "situp": self.annotate_situp,
            "jump_rope": self.annotate_jump_rope,
            "long_jump": self.annotate_long_jump,
            "pullup": self.annotate_pullup,
        }

        if action_type not in annotators:
            # 通用标注
            T = len(features)
            phases = [0] * (T // 2) + [1] * (T - T // 2)
            quality = {
                "is_standard": True,
                "overall_score": 75.0,
                "errors": [],
                "note": "使用通用标注",
            }
            return phases, quality

        return annotators[action_type](features)

    def process_action(
        self,
        action_type: str,
        skeleton_dir: Path,
        anno_dir: Path,
        only_files: set[str] | None = None,
    ):
        """处理单个动作的标注"""
        action_skeleton_dir = skeleton_dir / action_type
        action_anno_dir = anno_dir / action_type

        if not action_skeleton_dir.exists():
            print(f"警告: 骨骼目录不存在 {action_skeleton_dir}")
            return 0

        action_anno_dir.mkdir(parents=True, exist_ok=True)

        json_files = list(action_skeleton_dir.glob("*.json"))
        if only_files:
            json_files = [
                p for p in json_files if p.name in only_files or p.stem in only_files
            ]

        if not json_files:
            print(f"警告: 在 {action_skeleton_dir} 中没有找到骨骼文件")
            return 0

        print(f"\n标注动作: {action_type} ({CONFIG['actions'][action_type]['name']})")
        print(f"找到 {len(json_files)} 个骨骼文件")

        processed = 0

        for json_path in tqdm(json_files, desc=f"标注 {action_type}"):
            out_path = action_anno_dir / json_path.name

            # 跳过已标注
            if out_path.exists():
                continue

            try:
                # 加载骨骼
                with open(json_path) as f:
                    skeleton_data = json.load(f)

                # 转换为numpy
                sequence = []
                for frame in skeleton_data["skeleton_sequence"]:
                    coords = [[kp["x"], kp["y"]] for kp in frame["keypoints"]]
                    sequence.append(coords)
                sequence = np.array(sequence)

                # 预处理
                features = self.processor.process(sequence)

                # 自动标注
                phases, quality = self.annotate(features, action_type)

                # 保存标注
                annotation = {
                    "video_id": skeleton_data["video_id"],
                    "action_type": action_type,
                    "phases": phases,
                    "phase_names": CONFIG["actions"][action_type]["phases"],
                    "quality": quality,
                    "auto_annotated": True,
                    "reviewed": False,
                }

                with open(out_path, "w") as f:
                    json.dump(
                        to_builtin_types(annotation), f, indent=2, ensure_ascii=False
                    )

                processed += 1

            except Exception as e:
                print(f"  错误处理 {json_path.name}: {e}")

        print(f"完成: 成功标注 {processed} 个样本")
        return processed


def to_builtin_types(value):
    """Convert numpy scalar/list/dict values into Python builtin types."""
    if isinstance(value, dict):
        return {k: to_builtin_types(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_builtin_types(v) for v in value]
    if isinstance(value, tuple):
        return [to_builtin_types(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main():
    parser = argparse.ArgumentParser(description="自动标注骨骼数据")
    parser.add_argument("--action", type=str, default=None, help="指定标注的动作类型")
    parser.add_argument(
        "--skeleton_dir", type=str, default=CONFIG["paths"]["skeletons"], help="骨骼数据目录"
    )
    parser.add_argument(
        "--anno_dir", type=str, default=CONFIG["paths"]["annotations"], help="标注输出目录"
    )
    parser.add_argument(
        "--only_files",
        type=str,
        default="",
        help="仅处理指定文件（文件名或stem，逗号分隔）",
    )
    args = parser.parse_args()
    only_files = {x.strip() for x in (args.only_files or "").split(",") if x.strip()}

    skeleton_dir = Path(args.skeleton_dir)
    anno_dir = Path(args.anno_dir)

    # 创建标注目录
    anno_dir.mkdir(parents=True, exist_ok=True)

    # 创建标注器
    annotator = AutoAnnotator()

    # 处理动作
    if args.action:
        annotator.process_action(
            args.action,
            skeleton_dir,
            anno_dir,
            only_files=only_files or None,
        )
    else:
        total = 0
        for action_type in CONFIG["actions"].keys():
            count = annotator.process_action(
                action_type,
                skeleton_dir,
                anno_dir,
                only_files=only_files or None,
            )
            total += count

        print(f"\n{'='*50}")
        print(f"全部标注完成！共标注 {total} 个样本")

    print(f"\n标注数据保存在: {anno_dir}")
    print("提示: 请运行 2_review_annotations.py 进行人工复核")


if __name__ == "__main__":
    main()
