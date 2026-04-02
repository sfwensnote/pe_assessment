"""Compact summary features for action recognition models."""

from __future__ import annotations

import numpy as np


def extract_action_summary_features(
    skeleton_window: np.ndarray, features: np.ndarray
) -> np.ndarray:
    """Build a compact per-video feature vector for classical action classifiers."""

    seq = np.asarray(skeleton_window, dtype=float)
    if seq.ndim != 3 or seq.shape[1] < 17:
        return np.zeros(16, dtype=float)

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
    hip_angle = (features[:, 11, 5] + features[:, 12, 5]) / 2

    elbow_range = float(np.percentile(elbow, 95) - np.percentile(elbow, 5))
    knee_range = float(np.percentile(knee, 95) - np.percentile(knee, 5))
    hip_range = float(np.percentile(hip_angle, 95) - np.percentile(hip_angle, 5))

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

    shoulder_x_move = float(np.ptp(shoulders[:, 0]))
    wrist_y_move = float(np.ptp(wrists[:, 1]))

    return np.asarray(
        [
            torso_horizontal,
            torso_vertical,
            elbow_range,
            knee_range,
            hip_range,
            shoulder_y_move,
            hip_y_move,
            hip_x_move,
            ankle_y_move,
            wrist_above_shoulder,
            repetition_density,
            shoulder_x_move,
            wrist_y_move,
            float(np.mean(elbow)),
            float(np.mean(knee)),
            float(np.mean(hip_angle)),
        ],
        dtype=float,
    )
