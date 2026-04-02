"""Realtime single-user inference engine with smoothing and rep counting."""

from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from app.services.coach_feedback import build_tips
from app.services.model_runtime import ModelRuntime
from app.services.rep_counter import RepCounter


@dataclass
class RealtimeSession:
    """State container for one realtime practice session."""

    session_id: str
    runtime: ModelRuntime
    action_hint: Optional[str] = None
    target_reps: int = 20
    window_size: int = 60
    infer_interval: int = 3
    score_ema_alpha: float = 0.3
    error_hold_frames: int = 6
    action_lock_min_votes: int = 4

    skeleton_buffer: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=60))
    phase_history: Deque[int] = field(default_factory=lambda: deque(maxlen=5))
    action_vote_history: Deque[str] = field(default_factory=lambda: deque(maxlen=8))
    score_history: List[float] = field(default_factory=list)
    action_history: List[str] = field(default_factory=list)
    error_histogram: Counter = field(default_factory=Counter)
    error_streak: Dict[str, int] = field(default_factory=dict)

    frame_count: int = 0
    score_ema: Optional[float] = None
    latest_result: Optional[Dict[str, Any]] = None
    locked_action: Optional[str] = None
    started_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.skeleton_buffer = deque(maxlen=self.window_size)
        self.action_vote_history = deque(maxlen=8)
        initial_action = self.action_hint or "pushup"
        self.locked_action = self.action_hint
        self.rep_counter = RepCounter(action_type=initial_action)

    def process_frame(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """Process one frame and return realtime assessment result."""
        self.frame_count += 1

        skeleton = self.runtime.extract_skeleton_from_frame(frame_bgr)
        if skeleton is None:
            return self._status_payload("no_person", "未检测到人体，请进入画面中央。")

        self.skeleton_buffer.append(skeleton)

        if len(self.skeleton_buffer) < self.window_size:
            progress = len(self.skeleton_buffer) / self.window_size
            return self._status_payload(
                "warming_up",
                f"采集中 {len(self.skeleton_buffer)}/{self.window_size} 帧...",
                warmup_progress=round(progress, 3),
            )

        if (
            self.frame_count % self.infer_interval != 0
            and self.latest_result is not None
        ):
            cached = dict(self.latest_result)
            cached["status"] = "cached"
            return cached

        window = np.array(self.skeleton_buffer)
        runtime_action_hint = self.action_hint or self.locked_action
        raw = self.runtime.infer_window(window, runtime_action_hint)
        action_type = self._stable_action(raw["action_type"])
        if self.action_hint is None:
            self.rep_counter.set_action_type(action_type, reset=True)

        score = self._smooth_score(raw["overall_score"])
        phase = self._vote_phase(raw["phase"], action_type)
        stable_errors = self._stable_errors(raw["errors"])
        tips = build_tips(stable_errors)

        rep_count = self._update_rep_count(action_type, phase, skeleton)
        elapsed = max(time.time() - self.started_at, 1e-6)
        cadence = float(rep_count / elapsed * 60.0)

        self.score_history.append(score)
        self.action_history.append(action_type)
        for err in stable_errors:
            self.error_histogram[err] += 1

        action_source = raw.get("action_source", "unknown")
        if self.action_hint is None and self.locked_action is not None:
            action_source = "session_lock"

        result = {
            "status": "ok",
            "timestamp": int(time.time() * 1000),
            "action_type": action_type,
            "confidence": round(raw["confidence"], 4),
            "action_source": action_source,
            "phase": int(phase),
            "phase_name": self.runtime.config["actions"][action_type]["phases"][phase],
            "overall_score": round(score, 2),
            "is_standard": score >= 60,
            "errors": stable_errors,
            "tips": tips,
            "rep_count": rep_count,
            "target_reps": self.target_reps,
            "completion_rate": round(min(rep_count / max(self.target_reps, 1), 1.0), 3),
            "cadence": round(cadence, 2),
            "warmup_progress": 1.0,
        }

        self.latest_result = result
        return result

    def build_report(self) -> Dict[str, Any]:
        """Build report when session stops."""
        duration = max(time.time() - self.started_at, 0.0)
        avg_score = float(np.mean(self.score_history)) if self.score_history else 0.0
        best_score = float(np.max(self.score_history)) if self.score_history else 0.0
        main_action = self._most_common_action()

        return {
            "session_id": self.session_id,
            "duration_seconds": round(duration, 2),
            "action_type": main_action,
            "total_reps": self.rep_counter.total_reps,
            "target_reps": self.target_reps,
            "completion_rate": round(
                min(self.rep_counter.total_reps / max(self.target_reps, 1), 1.0),
                3,
            ),
            "avg_score": round(avg_score, 2),
            "best_score": round(best_score, 2),
            "error_histogram": dict(self.error_histogram),
            "score_series": [round(s, 2) for s in self.score_history[-120:]],
        }

    def snapshot(self) -> Dict[str, Any]:
        """Return lightweight runtime status for admin monitoring."""
        elapsed = max(time.time() - self.started_at, 0.0)
        latest = self.latest_result or {}
        return {
            "session_id": self.session_id,
            "action_hint": self.action_hint,
            "current_action": latest.get("action_type"),
            "action_source": latest.get("action_source"),
            "current_score": latest.get("overall_score"),
            "rep_count": self.rep_counter.total_reps,
            "target_reps": self.target_reps,
            "elapsed_seconds": round(elapsed, 2),
            "status": latest.get("status", "warming_up"),
            "last_message": latest.get("message"),
        }

    def _status_payload(
        self, status: str, message: str, **extra: Any
    ) -> Dict[str, Any]:
        payload = {
            "status": status,
            "message": message,
            "timestamp": int(time.time() * 1000),
            "rep_count": self.rep_counter.total_reps,
            "target_reps": self.target_reps,
            "completion_rate": round(
                min(self.rep_counter.total_reps / max(self.target_reps, 1), 1.0), 3
            ),
        }
        payload.update(extra)
        return payload

    def _smooth_score(self, score: float) -> float:
        if self.score_ema is None:
            self.score_ema = score
        else:
            self.score_ema = (
                self.score_ema_alpha * score
                + (1 - self.score_ema_alpha) * self.score_ema
            )
        return float(self.score_ema)

    def _vote_phase(self, phase: int, action_type: str) -> int:
        phase_count = len(self.runtime.config["actions"][action_type]["phases"])
        phase = max(0, min(int(phase), phase_count - 1))
        self.phase_history.append(phase)
        values = list(self.phase_history)
        return int(Counter(values).most_common(1)[0][0])

    def _stable_action(self, action_type: str) -> str:
        """Vote over a short history to reduce realtime action flicker."""
        if self.locked_action is not None:
            return self.locked_action

        self.action_vote_history.append(action_type)
        values = list(self.action_vote_history)
        winner, votes = Counter(values).most_common(1)[0]
        if len(values) >= self.action_lock_min_votes and votes >= self.action_lock_min_votes:
            self.locked_action = winner
            return winner
        return winner

    def _update_rep_count(
        self, action_type: str, phase: int, skeleton: np.ndarray
    ) -> int:
        """Prefer angle-based counting for cyclical strength actions."""
        signal = self._extract_rep_signal(action_type, skeleton)
        if signal is not None:
            recent_low = self._extract_recent_low_signal(action_type)
            return self.rep_counter.update_from_signal(signal, recent_low=recent_low)
        return self.rep_counter.update(phase)

    def _extract_rep_signal(
        self, action_type: str, skeleton: np.ndarray
    ) -> Optional[float]:
        angles = self.runtime.processor._compute_angles(np.asarray(skeleton, dtype=float))
        if action_type in {"pushup", "pullup"}:
            left = angles.get("left_elbow")
            right = angles.get("right_elbow")
            if left is None or right is None:
                return None
            return float((left + right) / 2.0)

        if action_type == "squat":
            left = angles.get("left_knee")
            right = angles.get("right_knee")
            if left is None or right is None:
                return None
            return float((left + right) / 2.0)

        return None

    def _extract_recent_low_signal(self, action_type: str) -> Optional[float]:
        if not self.skeleton_buffer:
            return None

        values = []
        for skeleton in self.skeleton_buffer:
            signal = self._extract_rep_signal(action_type, skeleton)
            if signal is not None:
                values.append(signal)

        if not values:
            return None

        return float(min(values))

    def _stable_errors(self, errors: List[str]) -> List[str]:
        active = set(errors)

        for err in list(self.error_streak.keys()):
            if err in active:
                self.error_streak[err] += 1
            else:
                self.error_streak[err] = 0

        for err in active:
            if err not in self.error_streak:
                self.error_streak[err] = 1

        stable = [
            err
            for err, streak in self.error_streak.items()
            if streak >= self.error_hold_frames
        ]
        return stable

    def _most_common_action(self) -> str:
        if not self.action_history:
            return self.action_hint or "pushup"
        return Counter(self.action_history).most_common(1)[0][0]
