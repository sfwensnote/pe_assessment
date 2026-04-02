"""Model runtime wrapper for realtime single-person inference."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml

try:
    import joblib
except ImportError:  # pragma: no cover - optional runtime dependency
    joblib = None

from app.services.coach_feedback import build_tips
from utils.action_features import extract_action_summary_features
from utils.metrics import compute_scores
from utils.models import QualityNet, STGCNAction, TemporalUNet
from utils.skeleton import SkeletonProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".ultralytics"))

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - runtime dependency check
    YOLO = None


@dataclass
class RuntimeStatus:
    """Tracks model/component availability for health endpoint."""

    yolo_ready: bool
    action_model_ready: bool
    phase_models_ready: int
    quality_model_ready: bool
    device: str


class ModelRuntime:
    """Loads models and performs per-window inference."""

    MODEL_CONF_THRESHOLD = 0.65
    RULE_OVERRIDE_THRESHOLD = 0.7
    DEFAULT_CHECKPOINT_SUBDIR = Path("checkpoints") / "mixed_best_bundle"
    RF_CONF_THRESHOLD = 0.3

    def __init__(self) -> None:
        root = PROJECT_ROOT
        self.project_root = root
        self.config_path = root / "config.yaml"

        with open(self.config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device(
            self.config["training"].get("device", "cuda")
            if torch.cuda.is_available()
            else "cpu"
        )

        self.processor = SkeletonProcessor(
            target_frames=self.config["skeleton"]["target_frames"]
        )

        self.error_types = self.config.get("error_types", [])
        self._init_yolo()
        self._init_models()

    def _init_yolo(self) -> None:
        self.yolo = None
        if YOLO is None:
            return

        try:
            pose_model = self.config.get("inference", {}).get(
                "pose_model", "yolov8n-pose.pt"
            )
            self.yolo = YOLO(pose_model)
        except Exception:
            self.yolo = None

    def _init_models(self) -> None:
        checkpoint_dir = self.project_root / self.DEFAULT_CHECKPOINT_SUBDIR
        if not checkpoint_dir.exists():
            checkpoint_dir = self.project_root / self.config["paths"]["checkpoints"]
        self.action_model = self._load_action_model(checkpoint_dir)
        self.action_rf_model = self._load_action_rf_model(checkpoint_dir)
        self.phase_models = self._load_phase_models(checkpoint_dir)
        self.quality_model = self._load_quality_model(checkpoint_dir)

    def _load_action_model(self, checkpoint_dir: Path) -> Optional[nn.Module]:
        model_path = checkpoint_dir / "action_model_best.pth"
        self.action_to_id = {
            name: idx for idx, name in enumerate(self.config["actions"].keys())
        }
        self.id_to_action = {v: k for k, v in self.action_to_id.items()}

        if not model_path.exists():
            return None

        try:
            model = STGCNAction(num_classes=len(self.config["actions"]))
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(self.device).eval()

            if "action_to_id" in checkpoint:
                self.action_to_id = checkpoint["action_to_id"]
                self.id_to_action = {v: k for k, v in self.action_to_id.items()}

            return model
        except Exception:
            return None

    def _load_action_rf_model(self, checkpoint_dir: Path):
        if joblib is None:
            return None

        model_path = checkpoint_dir / "action_model_rf.joblib"
        if not model_path.exists():
            return None

        try:
            model = joblib.load(model_path)
            if hasattr(model, "n_jobs"):
                try:
                    model.set_params(n_jobs=1)
                except Exception:
                    try:
                        model.n_jobs = 1
                    except Exception:
                        pass
            return model
        except Exception:
            return None

    def _load_phase_models(self, checkpoint_dir: Path) -> Dict[str, nn.Module]:
        models: Dict[str, nn.Module] = {}
        for action_type in self.config["actions"].keys():
            model_path = checkpoint_dir / f"phase_model_{action_type}.pth"
            if not model_path.exists():
                continue

            try:
                num_phases = len(self.config["actions"][action_type]["phases"])
                model = TemporalUNet(num_phases=num_phases)
                checkpoint = torch.load(model_path, map_location=self.device)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.to(self.device).eval()
                models[action_type] = model
            except Exception:
                continue

        return models

    def _load_quality_model(self, checkpoint_dir: Path) -> Optional[nn.Module]:
        model_path = checkpoint_dir / "quality_model_best.pth"
        if not model_path.exists():
            return None

        try:
            model = QualityNet(num_errors=len(self.error_types))
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(self.device).eval()

            if "error_types" in checkpoint:
                self.error_types = checkpoint["error_types"]
            return model
        except Exception:
            return None

    def status(self) -> RuntimeStatus:
        """Return runtime readiness summary."""
        return RuntimeStatus(
            yolo_ready=self.yolo is not None,
            action_model_ready=self.action_model is not None,
            phase_models_ready=len(self.phase_models),
            quality_model_ready=self.quality_model is not None,
            device=str(self.device),
        )

    def extract_skeleton_from_frame(
        self, frame_bgr: np.ndarray
    ) -> Optional[np.ndarray]:
        """Extract 17-keypoint skeleton from one frame."""
        if self.yolo is None:
            return None

        try:
            results = self.yolo(frame_bgr, verbose=False)
        except Exception:
            return None

        if not results:
            return None

        result = results[0]
        if result.keypoints is None or len(result.keypoints) == 0:
            return None

        kpts = result.keypoints.xy[0].cpu().numpy()
        if kpts.shape != (17, 2):
            return None

        return kpts

    def extract_skeleton_sequence_from_video(
        self,
        video_path: Path,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Optional[np.ndarray]:
        """Extract per-frame skeleton sequence from video file."""
        if self.yolo is None:
            return None

        total_frames = 0
        try:
            import importlib

            cv2 = importlib.import_module("cv2")

            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
        except Exception:
            total_frames = 0

        try:
            results = self.yolo(str(video_path), verbose=False, stream=True)
        except Exception:
            return None

        skeleton_sequence = []
        for idx, result in enumerate(results, start=1):
            if result.keypoints is None or len(result.keypoints) == 0:
                if progress_callback and total_frames > 0:
                    progress = min(80.0, (idx / max(total_frames, 1)) * 80.0)
                    progress_callback(progress, "提取人体关键点中")
                continue
            kpts = result.keypoints.xy[0].cpu().numpy()
            if kpts.shape != (17, 2):
                if progress_callback and total_frames > 0:
                    progress = min(80.0, (idx / max(total_frames, 1)) * 80.0)
                    progress_callback(progress, "关键点异常，跳过当前帧")
                continue
            skeleton_sequence.append(kpts)
            if progress_callback and total_frames > 0:
                progress = min(80.0, (idx / max(total_frames, 1)) * 80.0)
                progress_callback(progress, "提取人体关键点中")

        if progress_callback:
            progress_callback(85.0, "关键点提取完成，计算动作特征")

        if not skeleton_sequence:
            return None

        return np.array(skeleton_sequence)

    def assess_video_file(
        self,
        video_path: Path,
        action_hint: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, Any]:
        """Run full assessment for one uploaded video."""
        if progress_callback:
            progress_callback(5.0, "开始分析视频")

        skeleton_sequence = self.extract_skeleton_sequence_from_video(
            video_path, progress_callback=progress_callback
        )
        if skeleton_sequence is None:
            return {"ok": False, "error": "未检测到人体关键点"}

        if progress_callback:
            progress_callback(92.0, "执行动作识别与质量评估")

        result = self.infer_window(skeleton_sequence, action_hint)
        action_type = result["action_type"]
        errors = result.get("errors", [])
        estimated_reps = self._estimate_reps(skeleton_sequence, action_type)

        if progress_callback:
            progress_callback(100.0, "评估完成")

        return {
            "ok": True,
            "video_name": video_path.name,
            "total_frames": int(len(skeleton_sequence)),
            "action_type": action_type,
            "action_name": self.config["actions"][action_type]["name"],
            "confidence": float(result["confidence"]),
            "action_source": result.get("action_source", "unknown"),
            "phase": int(result["phase"]),
            "phase_name": result["phase_name"],
            "overall_score": float(result["overall_score"]),
            "is_standard": bool(result["is_standard"]),
            "estimated_reps": estimated_reps,
            "errors": errors,
            "tips": build_tips(errors),
            "quality": result.get("quality", {}),
        }

    def infer_window(
        self,
        skeleton_window: np.ndarray,
        action_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run action, phase and quality inference on a skeleton window."""
        features = self.processor.process(skeleton_window)

        action_type, confidence, action_source = self._recognize_action(
            skeleton_window, features, action_hint
        )
        phase_id = self._predict_phase(features, action_type)
        quality = self._assess_quality(features, action_type)

        return {
            "action_type": action_type,
            "confidence": float(confidence),
            "action_source": action_source,
            "phase": int(phase_id),
            "phase_name": self.config["actions"][action_type]["phases"][int(phase_id)],
            "overall_score": float(quality["overall_score"]),
            "is_standard": bool(quality["is_standard"]),
            "errors": quality["errors"],
            "quality": quality,
        }

    def _recognize_action(
        self,
        skeleton_window: np.ndarray,
        features: np.ndarray,
        action_hint: Optional[str],
    ) -> tuple[str, float, str]:
        if action_hint:
            return action_hint, 1.0, "user_hint"

        rule_action, rule_conf = self._recognize_action_by_rules(
            skeleton_window, features
        )

        rf_action = None
        rf_conf = 0.0
        if self.action_rf_model is not None:
            rf_action, rf_conf = self._recognize_action_rf(skeleton_window, features)

        if self.action_model is None and rf_action is None:
            return rule_action, rule_conf, "rule_fallback"

        model_action = None
        model_conf = 0.0
        if self.action_model is not None:
            with torch.no_grad():
                x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                logits = self.action_model(x)
                probs = torch.softmax(logits, dim=1)
                pred_id = int(logits.argmax(1).item())

            model_action = self.id_to_action.get(pred_id)
            if not isinstance(model_action, str):
                model_action = next(iter(self.config["actions"].keys()))
            model_conf = float(probs[0, pred_id].item())

        candidates = []
        if model_action is not None:
            candidates.append(("model", model_action, model_conf))
        if rf_action is not None:
            candidates.append(("rf", rf_action, rf_conf))

        if not candidates:
            return rule_action, rule_conf, "rule_fallback"

        best_source, best_action, best_conf = max(candidates, key=lambda item: item[2])
        if rule_conf >= self.RULE_OVERRIDE_THRESHOLD:
            if best_conf < self.MODEL_CONF_THRESHOLD:
                return rule_action, max(rule_conf, best_conf), "rule_override"
            if best_action != rule_action and rule_conf >= best_conf + 0.08:
                return rule_action, rule_conf, "rule_override"
            if best_source == "rf" and best_conf < 0.4:
                return rule_action, max(rule_conf, best_conf), "rule_override"

        # 低置信度时启用规则兜底，优先减少明显误判
        if best_conf < self.MODEL_CONF_THRESHOLD:
            if rule_conf >= self.RULE_OVERRIDE_THRESHOLD or (
                best_source == "model" and best_conf < 0.35
            ):
                return rule_action, max(rule_conf, best_conf), "rule_override"

        # 模型与规则冲突且模型不够稳定时，选择规则结果
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
        if self.action_rf_model is None:
            default_action = next(iter(self.config["actions"].keys()))
            return default_action, 0.0

        vector = extract_action_summary_features(skeleton_window, features).reshape(
            1, -1
        )
        pred = self.action_rf_model.predict(vector)[0]
        probs = self.action_rf_model.predict_proba(vector)[0]
        conf = float(np.max(probs)) if len(probs) else 0.0
        return str(pred), conf

    def _recognize_action_by_rules(
        self, skeleton_window: np.ndarray, features: np.ndarray
    ) -> tuple[str, float]:
        """Heuristic action recognition fallback for missing/low-confidence models."""
        action_names = list(self.config["actions"].keys())
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

        seq = np.array(skeleton_window, dtype=float)
        if seq.ndim != 3 or seq.shape[1] < 17:
            default_action = next(iter(action_names))
            return default_action, 0.2

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

        # pushup: body more horizontal, upper-limb dominant flexion, low overhead wrist ratio
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

        # squat: lower-limb dominant flexion with vertical torso and center-of-mass motion
        if "squat" in scores:
            scores["squat"] += 0.28 * between(torso_vertical, 0.5, 0.9)
            scores["squat"] += 0.25 * between(knee_range, 18, 90)
            scores["squat"] += 0.15 * between(hip_y_move, 0.25, 3.0)
            scores["squat"] += 0.10 * between(shoulder_y_move, 0.2, 3.0)
            scores["squat"] += 0.08 * close_to(wrist_above_shoulder, 0.0, 0.2)
            scores["squat"] += 0.08 * close_to(hip_x_move, 0.0, 1.5)
            scores["squat"] -= 0.10 * between(torso_horizontal, 0.7, 0.95)

        # situp: horizontal trunk with large shoulder rise, relatively small knee motion
        if "situp" in scores:
            scores["situp"] += 0.26 * between(torso_horizontal, 0.5, 0.9)
            scores["situp"] += 0.22 * between(shoulder_y_move, 0.8, 3.5)
            scores["situp"] += 0.14 * close_to(knee_range, 25, 30)
            scores["situp"] += 0.10 * close_to(hip_x_move, 0.0, 1.2)
            scores["situp"] -= 0.10 * between(wrist_above_shoulder, 0.35, 0.9)

        # pullup: overhead wrist ratio high + strong elbow flexion + vertical torso
        if "pullup" in scores:
            scores["pullup"] += 0.32 * between(wrist_above_shoulder, 0.2, 0.9)
            scores["pullup"] += 0.24 * between(elbow_range, 25, 95)
            scores["pullup"] += 0.14 * between(torso_vertical, 0.5, 0.9)
            scores["pullup"] += 0.10 * between(shoulder_y_move, 0.5, 3.0)
            scores["pullup"] += 0.06 * close_to(hip_x_move, 0.0, 1.2)
            scores["pullup"] -= 0.12 * between(torso_horizontal, 0.65, 0.95)

        # jump rope: repeated ankle oscillations, small trunk motion, relatively small knee flexion
        if "jump_rope" in scores:
            scores["jump_rope"] += 0.24 * between(ankle_y_move, 0.4, 2.8)
            scores["jump_rope"] += 0.18 * close_to(knee_range, 20, 35)
            scores["jump_rope"] += 0.16 * close_to(shoulder_y_move, 0.0, 1.4)
            scores["jump_rope"] += 0.12 * between(repetition_density, 0.015, 0.09)
            scores["jump_rope"] += 0.08 * between(torso_vertical, 0.5, 0.9)

        # long jump: strong forward displacement and takeoff/landing amplitude, low periodicity
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

    def _predict_phase(self, features: np.ndarray, action_type: str) -> int:
        model = self.phase_models.get(action_type)
        if model is None:
            return self._rule_phase(features, action_type)

        with torch.no_grad():
            x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            output = model(x)
            phases = output.argmax(-1)[0].cpu().numpy()

        phase_id = int(phases[-1]) if len(phases) else 0
        phase_count = len(self.config["actions"][action_type]["phases"])
        return max(0, min(phase_id, phase_count - 1))

    @staticmethod
    def _rule_phase(features: np.ndarray, action_type: str) -> int:
        if action_type == "pushup":
            elbow = features[:, 7, 3]
            cur = elbow[-1]
            if cur > 160:
                return 0
            if cur > 90:
                return 1
            return 2

        if action_type == "squat":
            knee = features[:, 13, 4]
            cur = knee[-1]
            if cur > 150:
                return 0
            if cur > 100:
                return 1
            return 2

        if action_type == "situp":
            shoulder_y = features[:, 5, 1]
            return 2 if shoulder_y[-1] <= np.percentile(shoulder_y, 20) else 1

        return 0

    def _estimate_reps(self, skeleton_sequence: np.ndarray, action_type: str) -> int:
        """Estimate repetition count from raw skeleton trajectory."""
        if len(skeleton_sequence) < 8:
            return 0

        seq = np.array(skeleton_sequence, dtype=float)
        features = self.processor.process(seq)

        if action_type in {"pushup", "pullup"}:
            elbow_signal = (features[:, 7, 3] + features[:, 8, 3]) / 2
            return self._count_cycles(elbow_signal, mode="valley")

        if action_type == "squat":
            knee_signal = (features[:, 13, 4] + features[:, 14, 4]) / 2
            return self._count_cycles(knee_signal, mode="valley")

        if action_type == "situp":
            shoulder_y = (seq[:, 5, 1] + seq[:, 6, 1]) / 2
            return self._count_cycles(shoulder_y, mode="peak")

        if action_type == "jump_rope":
            ankle_y = (seq[:, 15, 1] + seq[:, 16, 1]) / 2
            return self._count_cycles(ankle_y, mode="peak")

        if action_type == "long_jump":
            hip_x = (seq[:, 11, 0] + seq[:, 12, 0]) / 2
            return max(0, 1 if np.ptp(hip_x) > 40 else 0)

        return 0

    @staticmethod
    def _count_cycles(signal: np.ndarray, mode: str = "peak") -> int:
        """Count motion cycles via robust local extrema."""
        if len(signal) < 6:
            return 0

        arr = np.asarray(signal, dtype=float)
        smooth = np.convolve(arr, np.ones(5) / 5.0, mode="same")
        grad = np.diff(smooth)
        if len(grad) < 3:
            return 0

        scale = max(float(np.std(smooth)), 1e-6)
        threshold = 0.25 * scale

        count = 0
        for idx in range(1, len(grad)):
            left, right = grad[idx - 1], grad[idx]
            if mode == "peak":
                turning = left > 0 and right <= 0
                amplitude = smooth[idx] - min(
                    smooth[max(0, idx - 3) : min(len(smooth), idx + 3)]
                )
            else:
                turning = left < 0 and right >= 0
                amplitude = (
                    max(smooth[max(0, idx - 3) : min(len(smooth), idx + 3)])
                    - smooth[idx]
                )

            if turning and amplitude > threshold:
                count += 1

        return int(max(0, count))

    def _assess_quality(self, features: np.ndarray, action_type: str) -> Dict[str, Any]:
        try:
            metrics = self.processor.compute_quality_metrics(features, action_type)
            rule_score, details = compute_scores(
                metrics, self.config["actions"][action_type]
            )
            rule_errors = metrics.get("errors", [])
        except Exception as exc:
            metrics = {"errors": [], "runtime_warning": str(exc)}
            rule_score = 60.0
            details = {}
            rule_errors = []

        model_score = None
        model_errors = []
        if self.quality_model is not None:
            with torch.no_grad():
                x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                output = self.quality_model(x)
                model_score = float(output["overall"][0, 0].item() * 100.0)
                model_errors = [
                    self.error_types[i]
                    for i, prob in enumerate(output["errors"][0])
                    if float(prob.item()) > 0.5 and i < len(self.error_types)
                ]

        final_score = (
            float((model_score + rule_score) / 2)
            if model_score is not None
            else float(rule_score)
        )
        merged_errors = list(dict.fromkeys(model_errors + rule_errors))

        return {
            "overall_score": final_score,
            "model_score": model_score,
            "rule_score": float(rule_score),
            "is_standard": final_score >= 60.0,
            "errors": merged_errors,
            "details": details,
            "metrics": metrics,
        }

    def dump_session_report(self, report_path: Path, report: Dict[str, Any]) -> None:
        """Optional helper for storing session reports on disk."""
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
