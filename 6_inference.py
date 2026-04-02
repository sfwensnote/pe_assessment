#!/usr/bin/env python3
# @author Coder建设｜javpower
"""
6_inference.py
推理测试脚本 - 评估视频中的体育动作

用法:
    python 6_inference.py --video path/to/video.mp4 [--action pushup]

输出:
    - 动作类型识别结果
    - 动作阶段分割
    - 质量评分 (0-100)
    - 错误检测
    - 详细评估报告
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml

try:
    import joblib
except ImportError:
    joblib = None

# 添加项目路径
sys.path.append(str(Path(__file__).parent))
from utils.action_features import extract_action_summary_features
from utils.skeleton import SkeletonProcessor
from utils.models import STGCNAction, TemporalUNet, QualityNet
from utils.metrics import compute_scores, get_grade, format_assessment_report

# 加载配置
CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH, encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

DEFAULT_CHECKPOINT_DIR = Path(__file__).parent / "checkpoints" / "mixed_best_bundle"

# 检查YOLO
try:
    from ultralytics import YOLO
except ImportError:
    print("错误: 请先安装依赖: pip install ultralytics")
    sys.exit(1)


class ActionAssessor:
    """体育动作评估器"""

    MODEL_CONF_THRESHOLD = 0.65
    RULE_OVERRIDE_THRESHOLD = 0.7

    def __init__(self, checkpoint_dir: str = None, device: str = None):
        """
        初始化评估器

        Args:
            checkpoint_dir: 模型检查点目录
            device: 计算设备
        """
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"使用设备: {self.device}")

        if checkpoint_dir is None:
            checkpoint_dir = (
                str(DEFAULT_CHECKPOINT_DIR)
                if DEFAULT_CHECKPOINT_DIR.exists()
                else CONFIG["paths"]["checkpoints"]
            )
        self.checkpoint_dir = Path(checkpoint_dir)

        # 加载YOLO姿态检测模型
        print("加载YOLO姿态检测模型...")
        pose_model = CONFIG.get("inference", {}).get("pose_model", "yolov8n-pose.pt")
        self.yolo = YOLO(pose_model)

        # 加载动作识别模型
        print("加载动作识别模型...")
        self.action_model = self._load_action_model()
        self.action_rf_model = self._load_action_rf_model()

        # 加载阶段分割模型
        print("加载阶段分割模型...")
        self.phase_models = self._load_phase_models()

        # 加载质量评估模型
        print("加载质量评估模型...")
        self.quality_model = self._load_quality_model()

        # 初始化骨骼处理器
        self.processor = SkeletonProcessor(
            target_frames=CONFIG["skeleton"]["target_frames"]
        )

        print("评估器初始化完成！")

    def _load_action_model(self) -> nn.Module:
        """加载动作识别模型"""
        model_path = self.checkpoint_dir / "action_model_best.pth"

        if not model_path.exists():
            print(f"警告: 动作识别模型不存在 {model_path}")
            return None

        num_classes = len(CONFIG["actions"])
        model = STGCNAction(num_classes=num_classes).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        self.action_to_id = checkpoint["action_to_id"]
        self.id_to_action = {v: k for k, v in self.action_to_id.items()}

        return model

    def _load_action_rf_model(self):
        """加载随机森林动作识别模型（若存在）。"""
        if joblib is None:
            return None

        model_path = self.checkpoint_dir / "action_model_rf.joblib"
        if not model_path.exists():
            return None

        try:
            return joblib.load(model_path)
        except Exception:
            return None

    def _load_phase_models(self) -> Dict[str, nn.Module]:
        """加载所有阶段分割模型"""
        models = {}

        for action_type in CONFIG["actions"].keys():
            model_path = self.checkpoint_dir / f"phase_model_{action_type}.pth"

            if not model_path.exists():
                continue

            num_phases = len(CONFIG["actions"][action_type]["phases"])
            model = TemporalUNet(num_phases=num_phases).to(self.device)

            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            models[action_type] = model

        return models

    def _load_quality_model(self) -> nn.Module:
        """加载质量评估模型"""
        model_path = self.checkpoint_dir / "quality_model_best.pth"

        if not model_path.exists():
            print(f"警告: 质量评估模型不存在 {model_path}")
            return None

        num_errors = len(CONFIG["error_types"])
        model = QualityNet(num_errors=num_errors).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        self.error_types = checkpoint.get("error_types", CONFIG["error_types"])

        return model

    def extract_skeleton(self, video_path: str) -> Optional[np.ndarray]:
        """
        从视频中提取骨骼序列

        Args:
            video_path: 视频路径

        Returns:
            skeleton_sequence: [T, 17, 2] 骨骼坐标序列
        """
        try:
            results = self.yolo(video_path, verbose=False, stream=True)
        except Exception:
            return None

        skeleton_sequence = []

        for result in results:
            if result.keypoints is None or len(result.keypoints) == 0:
                continue

            # 取最前面的人
            kpts = result.keypoints.xy[0].cpu().numpy()
            if kpts.shape != (17, 2):
                continue
            skeleton_sequence.append(kpts)

        if len(skeleton_sequence) == 0:
            return None

        return np.array(skeleton_sequence)

    def recognize_action(
        self, features: np.ndarray, skeleton_window: Optional[np.ndarray] = None
    ) -> tuple:
        """
        识别动作类型

        Returns:
            action_type: 动作类型
            confidence: 置信度
        """
        rule_action, rule_conf = self._recognize_action_by_rules(
            skeleton_window, features
        )

        rf_action = None
        rf_conf = 0.0
        if self.action_rf_model is not None and skeleton_window is not None:
            rf_action, rf_conf = self._recognize_action_rf(skeleton_window, features)

        if rf_action is not None:
            if rule_conf >= 0.85 and rf_conf < 0.25:
                return rule_action, max(rule_conf, rf_conf), "rule_override"
            return rf_action, rf_conf, "rf"

        if self.action_model is None and rf_action is None:
            return rule_action, rule_conf, "rule_fallback"

        model_action = None
        model_conf = 0.0
        if self.action_model is not None:
            with torch.no_grad():
                input_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                output = self.action_model(input_tensor)
                probs = torch.softmax(output, dim=1)
                pred_id = output.argmax(1).item()
                model_conf = probs[0][pred_id].item()
            model_action = self.id_to_action[pred_id]

        candidates = []
        if model_action is not None:
            candidates.append(("model", model_action, float(model_conf)))
        if rf_action is not None:
            candidates.append(("rf", rf_action, float(rf_conf)))

        if not candidates:
            return rule_action, rule_conf, "rule_fallback"

        best_source, best_action, best_conf = max(candidates, key=lambda item: item[2])

        if best_conf < self.MODEL_CONF_THRESHOLD:
            if rule_conf >= self.RULE_OVERRIDE_THRESHOLD or (
                best_source == "model" and best_conf < 0.35
            ):
                return rule_action, max(rule_conf, best_conf), "rule_override"

        if (
            best_action != rule_action
            and best_conf < 0.8
            and rule_conf >= self.RULE_OVERRIDE_THRESHOLD
        ):
            return rule_action, max(rule_conf, best_conf), "rule_override"

        return best_action, best_conf, best_source

    def _recognize_action_rf(
        self, skeleton_window: np.ndarray, features: np.ndarray
    ) -> tuple[str, float]:
        """使用随机森林动作分类器做动作识别。"""
        if self.action_rf_model is None:
            return "pushup", 0.0

        vector = extract_action_summary_features(skeleton_window, features).reshape(
            1, -1
        )
        pred = self.action_rf_model.predict(vector)[0]
        probs = self.action_rf_model.predict_proba(vector)[0]
        conf = float(np.max(probs)) if len(probs) else 0.0
        return str(pred), conf

    def _recognize_action_by_rules(
        self, skeleton_window: Optional[np.ndarray], features: np.ndarray
    ) -> tuple[str, float]:
        """规则兜底的动作识别逻辑。"""
        action_names = list(CONFIG["actions"].keys())
        if not action_names:
            return "pushup", 0.2

        scores = {name: 0.0 for name in action_names}

        def between(value: float, low: float, high: float) -> float:
            if high <= low:
                return 0.0
            if value <= low:
                return 0.0
            if value >= high:
                return 1.0
            return float((value - low) / (high - low))

        def close_to(value: float, center: float, tolerance: float) -> float:
            if tolerance <= 0:
                return 0.0
            return float(max(0.0, 1.0 - abs(value - center) / tolerance))

        if skeleton_window is None:
            default_action = action_names[0]
            return default_action, 0.2

        seq = np.array(skeleton_window, dtype=float)
        if seq.ndim != 3 or seq.shape[1] < 17:
            return action_names[0], 0.2

        seq = seq[:, :, :2]
        left_hip_raw = seq[:, 11]
        right_hip_raw = seq[:, 12]
        hip_center0 = (left_hip_raw[0] + right_hip_raw[0]) / 2
        seq_norm = seq - hip_center0[np.newaxis, np.newaxis, :]

        left_shoulder_raw = seq_norm[:, 5]
        right_shoulder_raw = seq_norm[:, 6]
        shoulder_widths = np.linalg.norm(left_shoulder_raw - right_shoulder_raw, axis=1)
        scale = (
            float(np.median(shoulder_widths[shoulder_widths > 1e-6]))
            if np.any(shoulder_widths > 1e-6)
            else 1.0
        )
        scale = max(scale, 1e-3)
        seq_norm = seq_norm / scale

        shoulders = (seq_norm[:, 5] + seq_norm[:, 6]) / 2
        hips = (seq_norm[:, 11] + seq_norm[:, 12]) / 2
        ankles = (seq_norm[:, 15] + seq_norm[:, 16]) / 2
        wrists = (seq_norm[:, 9] + seq_norm[:, 10]) / 2

        torso_vec = shoulders - hips
        torso_norm = np.linalg.norm(torso_vec, axis=1) + 1e-6
        torso_horizontal = float(np.mean(np.abs(torso_vec[:, 0]) / torso_norm))
        torso_vertical = float(np.mean(np.abs(torso_vec[:, 1]) / torso_norm))

        elbow = (features[:, 7, 3] + features[:, 8, 3]) / 2
        knee = (features[:, 13, 4] + features[:, 14, 4]) / 2
        elbow_range = float(np.percentile(elbow, 95) - np.percentile(elbow, 5))
        knee_range = float(np.percentile(knee, 95) - np.percentile(knee, 5))

        shoulder_y_move = float(np.ptp(shoulders[:, 1]))
        hip_y_move = float(np.ptp(hips[:, 1]))
        hip_x_move = float(np.ptp(hips[:, 0]))
        ankle_y_move = float(np.ptp(ankles[:, 1]))

        wrist_above_shoulder = float(
            np.mean((wrists[:, 1] < shoulders[:, 1]).astype(float))
        )

        ankle_signal = ankles[:, 1] - float(np.mean(ankles[:, 1]))
        dy = np.diff(ankle_signal)
        minima_count = int(np.sum((dy[:-1] < 0) & (dy[1:] >= 0))) if len(dy) > 2 else 0
        repetition_density = float(minima_count / max(len(ankles), 1))

        if "pushup" in scores:
            scores["pushup"] += 0.30 * between(torso_horizontal, 0.55, 0.85)
            scores["pushup"] += 0.22 * between(elbow_range, 20, 85)
            scores["pushup"] += 0.18 * between(
                max(0.0, elbow_range - knee_range), 5, 55
            )
            scores["pushup"] += 0.10 * close_to(wrist_above_shoulder, 0.0, 0.2)
            scores["pushup"] += 0.08 * close_to(hip_x_move, 0.0, 1.1)
            scores["pushup"] -= 0.12 * between(wrist_above_shoulder, 0.35, 0.9)
            scores["pushup"] -= 0.08 * between(hip_x_move, 1.0, 3.5)

        if "squat" in scores:
            scores["squat"] += 0.28 * between(torso_vertical, 0.5, 0.9)
            scores["squat"] += 0.25 * between(knee_range, 18, 90)
            scores["squat"] += 0.15 * between(hip_y_move, 0.25, 3.0)
            scores["squat"] += 0.10 * between(shoulder_y_move, 0.2, 3.0)
            scores["squat"] += 0.08 * close_to(wrist_above_shoulder, 0.0, 0.2)
            scores["squat"] += 0.08 * close_to(hip_x_move, 0.0, 1.5)
            scores["squat"] -= 0.10 * between(torso_horizontal, 0.7, 0.95)

        if "situp" in scores:
            scores["situp"] += 0.26 * between(torso_horizontal, 0.5, 0.9)
            scores["situp"] += 0.22 * between(shoulder_y_move, 0.8, 3.5)
            scores["situp"] += 0.14 * close_to(knee_range, 25, 30)
            scores["situp"] += 0.10 * close_to(hip_x_move, 0.0, 1.2)
            scores["situp"] -= 0.10 * between(wrist_above_shoulder, 0.35, 0.9)

        if "pullup" in scores:
            scores["pullup"] += 0.32 * between(wrist_above_shoulder, 0.2, 0.9)
            scores["pullup"] += 0.24 * between(elbow_range, 25, 95)
            scores["pullup"] += 0.14 * between(torso_vertical, 0.5, 0.9)
            scores["pullup"] += 0.10 * between(shoulder_y_move, 0.5, 3.0)
            scores["pullup"] += 0.06 * close_to(hip_x_move, 0.0, 1.2)
            scores["pullup"] -= 0.12 * between(torso_horizontal, 0.65, 0.95)

        if "jump_rope" in scores:
            scores["jump_rope"] += 0.24 * between(ankle_y_move, 0.4, 2.8)
            scores["jump_rope"] += 0.18 * close_to(knee_range, 20, 35)
            scores["jump_rope"] += 0.16 * close_to(shoulder_y_move, 0.0, 1.4)
            scores["jump_rope"] += 0.12 * between(repetition_density, 0.015, 0.09)
            scores["jump_rope"] += 0.08 * between(torso_vertical, 0.5, 0.9)

        if "long_jump" in scores:
            scores["long_jump"] += 0.30 * between(hip_x_move, 0.8, 4.0)
            scores["long_jump"] += 0.20 * between(ankle_y_move, 0.5, 3.5)
            scores["long_jump"] += 0.15 * between(knee_range, 20, 95)
            scores["long_jump"] += 0.08 * between(shoulder_y_move, 0.4, 3.0)
            scores["long_jump"] += 0.06 * close_to(repetition_density, 0.01, 0.03)

        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        action_type = sorted_scores[0][0]
        top_score = float(max(0.0, sorted_scores[0][1]))
        second_score = (
            float(max(0.0, sorted_scores[1][1])) if len(sorted_scores) > 1 else 0.0
        )
        margin = max(0.0, top_score - second_score)
        confidence = min(0.95, max(0.2, 0.45 + 0.45 * top_score + 0.2 * margin))
        return action_type, confidence

    def segment_phases(self, features: np.ndarray, action_type: str) -> np.ndarray:
        """
        分割动作阶段

        Returns:
            phases: [T] 阶段标签
        """
        if action_type in self.phase_models:
            model = self.phase_models[action_type]

            with torch.no_grad():
                input_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                output = model(input_tensor)
                phases = output.argmax(2)[0].cpu().numpy()

            return phases
        else:
            # 使用规则-based阶段检测
            return self._rule_based_phases(features, action_type)

    def _rule_based_phases(self, features: np.ndarray, action_type: str) -> np.ndarray:
        """基于规则的阶段检测（备用）"""
        T = len(features)
        phases = np.zeros(T, dtype=int)

        if action_type == "pushup":
            elbow_angles = features[:, 7, 3]
            bottom_idx = np.argmin(elbow_angles)

            phases[elbow_angles > 160] = 0
            phases[(elbow_angles > 90) & (elbow_angles <= 160)] = 1
            phases[elbow_angles <= 90] = 2
            phases[(elbow_angles > 90) & (np.arange(T) > bottom_idx)] = 3
            phases[(elbow_angles > 160) & (np.arange(T) > bottom_idx)] = 4

        elif action_type == "squat":
            knee_angles = features[:, 13, 4]
            bottom_idx = np.argmin(knee_angles)

            phases[: bottom_idx // 3] = 0
            phases[bottom_idx // 3 : bottom_idx] = 1
            phases[max(0, bottom_idx - 2) : min(T, bottom_idx + 3)] = 2
            phases[bottom_idx : bottom_idx + 2 * (T - bottom_idx) // 3] = 3
            phases[bottom_idx + 2 * (T - bottom_idx) // 3 :] = 4

        return phases

    def assess_quality(self, features: np.ndarray, action_type: str) -> Dict:
        """
        评估动作质量

        Returns:
            quality: 质量评估结果字典
        """
        # 1. 使用模型评估
        if self.quality_model is not None:
            with torch.no_grad():
                input_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                output = self.quality_model(input_tensor)

                model_score = float(output["overall"][0, 0].item() * 100.0)
                model_errors = [
                    self.error_types[i]
                    for i, p in enumerate(output["errors"][0])
                    if p > 0.5
                ]
        else:
            model_score = None
            model_errors = []

        # 2. 使用规则评估
        metrics = self.processor.compute_quality_metrics(features, action_type)
        rule_score, details = compute_scores(metrics, CONFIG["actions"][action_type])
        rule_errors = metrics.get("errors", [])

        # 3. 融合结果
        if model_score is not None:
            final_score = (model_score + rule_score) / 2
        else:
            final_score = rule_score

        # 合并错误（去重）
        all_errors = list(set(model_errors + rule_errors))

        return {
            "overall_score": float(final_score),
            "model_score": float(model_score) if model_score is not None else None,
            "rule_score": float(rule_score),
            "is_standard": final_score >= 60,
            "errors": all_errors,
            "details": details,
            "metrics": metrics,
        }

    def assess_video(self, video_path: str, action_type: str = None) -> Dict:
        """
        评估视频

        Args:
            video_path: 视频路径
            action_type: 指定动作类型（可选）

        Returns:
            assessment: 评估结果字典
        """
        print(f"\n评估视频: {video_path}")
        print("-" * 60)

        if not Path(video_path).exists():
            return {"error": f"视频不存在: {video_path}"}

        # 1. 提取骨骼
        print("1. 提取骨骼关键点...")
        skeleton_sequence = self.extract_skeleton(video_path)

        if skeleton_sequence is None:
            return {"error": "未检测到人体"}

        print(f"   检测到 {len(skeleton_sequence)} 帧")

        # 2. 预处理
        print("2. 预处理骨骼数据...")
        features = self.processor.process(skeleton_sequence)

        # 3. 动作识别
        print("3. 识别动作类型...")
        if action_type is None:
            recognized_action, confidence, action_source = self.recognize_action(
                features, skeleton_sequence
            )
            action_type = recognized_action
        else:
            confidence = 1.0
            action_source = "user_hint"

        action_name = CONFIG["actions"][action_type]["name"]
        print(
            f"   识别动作: {action_name} (置信度: {confidence:.2%}, 来源: {action_source})"
        )

        # 4. 阶段分割
        print("4. 分割动作阶段...")
        phases = self.segment_phases(features, action_type)
        unique_phases = sorted(set(phases))
        print(f"   检测到 {len(unique_phases)} 个阶段: {unique_phases}")

        # 5. 质量评估
        print("5. 评估动作质量...")
        quality = self.assess_quality(features, action_type)
        print(f"   质量评分: {quality['overall_score']:.1f}/100")
        print(f"   等级评定: {get_grade(quality['overall_score'])}")

        if quality["errors"]:
            print(
                f"   检测到 {len(quality['errors'])} 个错误: {', '.join(quality['errors'])}"
            )
        else:
            print("   未检测到明显错误")

        # 6. 生成完整报告
        assessment = {
            "video_path": video_path,
            "action_type": action_type,
            "action_name": action_name,
            "recognition_confidence": float(confidence),
            "action_source": action_source,
            "total_frames": len(skeleton_sequence),
            "phases": phases.tolist(),
            "phase_names": CONFIG["actions"][action_type]["phases"],
            "quality": quality,
        }

        return assessment

    def visualize_result(
        self, video_path: str, assessment: Dict, save_path: str = None
    ):
        """
        可视化评估结果
        """
        cap = cv2.VideoCapture(video_path)

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()

        if not frames:
            return

        # 在每一帧上绘制信息
        phases = assessment["phases"]
        action_name = assessment["action_name"]
        score = assessment["quality"]["overall_score"]

        # 计算阶段颜色
        phase_colors = [
            (0, 255, 0),  # 绿色 - 准备
            (255, 255, 0),  # 青色 - 下降/进行
            (0, 0, 255),  # 红色 - 最低点
            (255, 0, 255),  # 紫色 - 上升/恢复
            (255, 255, 255),  # 白色 - 完成
        ]

        for i, frame in enumerate(frames):
            # 获取当前阶段
            phase_idx = min(i * len(phases) // len(frames), len(phases) - 1)
            current_phase = phases[phase_idx]

            # 绘制信息面板
            h, w = frame.shape[:2]

            # 背景
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (350, 120), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            # 文字
            cv2.putText(
                frame,
                f"Action: {action_name}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Score: {score:.1f}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if score >= 60 else (0, 0, 255),
                2,
            )

            phase_name = assessment["phase_names"][current_phase]
            color = phase_colors[current_phase % len(phase_colors)]
            cv2.putText(
                frame,
                f"Phase: {phase_name}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

            # 绘制阶段指示条
            bar_y = h - 30
            for j, phase in enumerate(phases):
                x = int(w * j / len(phases))
                color = phase_colors[phase % len(phase_colors)]
                cv2.line(frame, (x, bar_y), (x, bar_y + 20), color, 2)

            frames[i] = frame

        # 保存或显示
        if save_path:
            # 保存为视频
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                save_path, fourcc, 30.0, (frames[0].shape[1], frames[0].shape[0])
            )

            for frame in frames:
                out.write(frame)

            out.release()
            print(f"可视化结果已保存: {save_path}")
        else:
            # 显示
            for frame in frames:
                cv2.imshow("Assessment", frame)
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="体育动作评估")
    parser.add_argument("--video", type=str, required=True, help="输入视频路径")
    parser.add_argument("--action", type=str, default=None, help="指定动作类型（可选）")
    parser.add_argument(
        "--checkpoint_dir", type=str, default=None, help="模型检查点目录"
    )
    parser.add_argument("--device", type=str, default=None, help="计算设备")
    parser.add_argument("--output", type=str, default=None, help="输出JSON文件路径")
    parser.add_argument(
        "--visualize", type=str, default=None, help="可视化输出视频路径"
    )
    parser.add_argument(
        "--format", type=str, default="text", choices=["text", "json"], help="输出格式"
    )
    args = parser.parse_args()

    # 创建评估器
    assessor = ActionAssessor(checkpoint_dir=args.checkpoint_dir, device=args.device)

    # 评估视频
    assessment = assessor.assess_video(args.video, args.action)

    if "error" in assessment:
        if args.format == "json":
            print(json.dumps(assessment, indent=2, ensure_ascii=False))
        else:
            print(f"评估失败: {assessment['error']}")

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(assessment, f, indent=2, ensure_ascii=False)
            print(f"\n评估结果已保存: {args.output}")
        return

    # 输出结果
    if args.format == "json":
        print(json.dumps(assessment, indent=2, ensure_ascii=False))
    else:
        print("\n" + "=" * 60)
        print(
            format_assessment_report(
                action_type=assessment["action_name"],
                overall_score=assessment["quality"]["overall_score"],
                details=assessment["quality"]["details"],
                errors=assessment["quality"]["errors"],
                phases=assessment["phases"],
            )
        )

    # 保存结果
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(assessment, f, indent=2, ensure_ascii=False)
        print(f"\n评估结果已保存: {args.output}")

    # 可视化
    if args.visualize:
        assessor.visualize_result(args.video, assessment, args.visualize)


if __name__ == "__main__":
    main()
