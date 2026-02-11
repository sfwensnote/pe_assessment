# @author Coder建设｜javpower
"""
骨骼数据处理模块
支持多种体育运动的骨骼特征提取
"""

import numpy as np
import cv2
from scipy import interpolate
from scipy.signal import savgol_filter
from typing import Dict, List, Tuple, Optional
import json


class SkeletonProcessor:
    """
    骨骼数据处理器
    支持：俯卧撑、深蹲、跳绳、跳远、引体向上、仰卧起坐等
    """
    
    # COCO格式17个关节
    JOINT_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    # 关节索引映射
    JOINT_IDX = {name: i for i, name in enumerate(JOINT_NAMES)}
    
    # 关节对（用于计算角度和绘制）
    SKELETON_PAIRS = [
        [0, 1], [0, 2], [1, 3], [2, 4],       # 头部
        [5, 6],                               # 肩部
        [5, 7], [7, 9], [6, 8], [8, 10],      # 手臂
        [5, 11], [6, 12], [11, 12],           # 躯干
        [11, 13], [13, 15], [12, 14], [14, 16] # 腿部
    ]
    
    # 角度计算定义 (joint1, center, joint2)
    ANGLE_JOINTS = {
        'left_elbow': (5, 7, 9),      # 左肩-肘-腕
        'right_elbow': (6, 8, 10),    # 右肩-肘-腕
        'left_knee': (11, 13, 15),    # 左髋-膝-踝
        'right_knee': (12, 14, 16),   # 右髋-膝-踝
        'left_hip': (5, 11, 13),      # 左肩-髋-膝
        'right_hip': (6, 12, 14),     # 右肩-髋-膝
        'left_shoulder': (7, 5, 11),  # 左肘-肩-髋
        'right_shoulder': (8, 6, 12), # 右肘-肩-髋
        'spine': (5, 11, 15),         # 躯干直线度（肩-髋-踝）
    }
    
    def __init__(self, target_frames: int = 60, smooth: bool = True):
        self.target_frames = target_frames
        self.smooth = smooth
        
    def process(self, keypoints_sequence: np.ndarray) -> np.ndarray:
        """
        处理原始骨骼序列
        
        Args:
            keypoints_sequence: [T, 17, 2] 或 [T, 17, 3] 像素坐标
        
        Returns:
            features: [target_frames, 17, 9] 归一化特征
                      通道: [x, y, confidence, angle1, angle2, angle3, vx, vy, speed]
        """
        sequence = np.array(keypoints_sequence)
        if len(sequence.shape) == 2:
            # 假设是 [T, 34]，重塑为 [T, 17, 2]
            T = sequence.shape[0]
            sequence = sequence.reshape(T, 17, 2)
        
        T, N, C = sequence.shape
        
        # 1. 坐标归一化
        sequence = self._normalize(sequence)
        
        # 2. 时序对齐（重采样）
        sequence = self._resample(sequence)
        
        # 3. 特征提取
        features = self._extract_features(sequence)
        
        return features
    
    def _normalize(self, sequence: np.ndarray) -> np.ndarray:
        """
        坐标归一化（针对固定摄像头优化）
        """
        T, N, C = sequence.shape
        
        # 使用髋中心作为原点
        left_hip = sequence[:, self.JOINT_IDX['left_hip']]
        right_hip = sequence[:, self.JOINT_IDX['right_hip']]
        hip_center = (left_hip + right_hip) / 2
        
        # 平移到原点
        normalized = sequence - hip_center[:, np.newaxis, :]
        
        # 缩放：以肩宽为单位长度
        left_shoulder = normalized[:, self.JOINT_IDX['left_shoulder']]
        right_shoulder = normalized[:, self.JOINT_IDX['right_shoulder']]
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder, axis=1)
        shoulder_width = np.maximum(shoulder_width, 1e-6)
        
        normalized = normalized / shoulder_width[:, np.newaxis, np.newaxis]
        
        # 旋转：使肩线水平
        angles = np.arctan2(
            right_shoulder[:, 1] - left_shoulder[:, 1],
            right_shoulder[:, 0] - left_shoulder[:, 0]
        )
        mean_angle = np.mean(angles)
        
        rot_matrix = np.array([
            [np.cos(-mean_angle), -np.sin(-mean_angle)],
            [np.sin(-mean_angle), np.cos(-mean_angle)]
        ])
        
        for t in range(T):
            normalized[t] = normalized[t] @ rot_matrix.T
        
        return normalized
    
    def _resample(self, sequence: np.ndarray) -> np.ndarray:
        """重采样到固定帧数"""
        T, N, C = sequence.shape
        
        if T == self.target_frames:
            return sequence
        
        # 线性插值
        old_times = np.linspace(0, 1, T)
        new_times = np.linspace(0, 1, self.target_frames)
        
        resampled = np.zeros((self.target_frames, N, C))
        for n in range(N):
            for c in range(C):
                f = interpolate.interp1d(
                    old_times, sequence[:, n, c],
                    kind='linear', fill_value='extrapolate'
                )
                resampled[:, n, c] = f(new_times)
        
        # Savitzky-Golay平滑
        if self.smooth and self.target_frames >= 5:
            for n in range(N):
                for c in range(C):
                    resampled[:, n, c] = savgol_filter(
                        resampled[:, n, c], window_length=5, polyorder=2
                    )
        
        return resampled
    
    def _extract_features(self, sequence: np.ndarray) -> np.ndarray:
        """
        提取9维特征
        
        Returns:
            features: [T, 17, 9]
                0-1: 归一化坐标 (x, y)
                2: 检测置信度
                3-5: 关键关节角度（存储在特定关节位置）
                6-7: 速度 (vx, vy)
                8: 速度大小
        """
        T, N, _ = sequence.shape
        features = np.zeros((T, N, 9))
        
        # 1-2维：归一化坐标
        features[:, :, :2] = sequence
        
        # 3维：检测置信度
        features[:, :, 2] = 1.0
        
        # 4-6维：关键关节角度
        for t in range(T):
            angles = self._compute_angles(sequence[t])
            # 将角度存储在对应关节
            features[t, 7, 3] = angles.get('left_elbow', 0)   # 左肘
            features[t, 8, 3] = angles.get('right_elbow', 0)  # 右肘
            features[t, 13, 4] = angles.get('left_knee', 0)   # 左膝
            features[t, 14, 4] = angles.get('right_knee', 0)  # 右膝
            features[t, 11, 5] = angles.get('left_hip', 0)    # 左髋
            features[t, 12, 5] = angles.get('right_hip', 0)   # 右髋
        
        # 7-9维：时序特征（速度）
        velocity = np.diff(features[:, :, :2], axis=0, prepend=features[:1, :, :2])
        features[:, :, 6:8] = velocity
        features[:, :, 8] = np.linalg.norm(velocity, axis=2)
        
        return features
    
    def _compute_angles(self, frame: np.ndarray) -> Dict[str, float]:
        """计算关节角度"""
        angles = {}
        
        for name, (i, j, k) in self.ANGLE_JOINTS.items():
            p1, p2, p3 = frame[i], frame[j], frame[k]
            
            v1 = p1 - p2
            v2 = p3 - p2
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
            angles[name] = angle
        
        return angles
    
    def compute_quality_metrics(self, features: np.ndarray, action_type: str) -> Dict:
        """
        计算质量指标（用于自动标注和评估）
        
        Args:
            features: [T, 17, 9] 特征序列
            action_type: 动作类型
        
        Returns:
            dict: 各项指标和检测到的错误
        """
        metrics = {'errors': []}
        
        if action_type == 'pushup':
            metrics.update(self._analyze_pushup(features))
        elif action_type == 'squat':
            metrics.update(self._analyze_squat(features))
        elif action_type == 'situp':
            metrics.update(self._analyze_situp(features))
        elif action_type == 'jump_rope':
            metrics.update(self._analyze_jump_rope(features))
        elif action_type == 'long_jump':
            metrics.update(self._analyze_long_jump(features))
        elif action_type == 'pullup':
            metrics.update(self._analyze_pullup(features))
        else:
            metrics['error'] = f"未知动作类型: {action_type}"
        
        return metrics
    
    def _analyze_pushup(self, features: np.ndarray) -> Dict:
        """俯卧撑分析"""
        metrics = {}
        
        # 肘关节角度序列
        left_elbow = features[:, 7, 3]
        right_elbow = features[:, 8, 3]
        elbow_angles = (left_elbow + right_elbow) / 2
        
        # 最低点检测
        bottom_idx = np.argmin(elbow_angles)
        metrics['bottom_elbow_angle'] = float(elbow_angles[bottom_idx])
        
        # 身体直线度（肩-髋-踝角度，应接近180）
        left_hip_angle = features[:, 11, 5]
        metrics['body_straightness'] = float(180 - abs(180 - np.mean(left_hip_angle)))
        
        # 髋部稳定性（检测塌腰/撅臀）
        hip_y = features[:, self.JOINT_IDX['left_hip'], 1]
        hip_range = np.max(hip_y) - np.min(hip_y)
        metrics['hip_stability'] = float(hip_range)
        
        # 速度分析
        metrics['speed_down'] = float(bottom_idx / 30.0)  # 假设30fps
        metrics['speed_up'] = float((self.target_frames - bottom_idx) / 30.0)
        
        # 错误检测
        if hip_range > 0.15:
            metrics['errors'].append('塌腰' if hip_y[bottom_idx] > np.mean(hip_y) else '撅臀')
        
        if elbow_angles[bottom_idx] > 90:
            metrics['errors'].append('未达深度')
        
        # 肘外扩检测（简化）
        left_wrist_x = features[:, self.JOINT_IDX['left_wrist'], 0]
        right_wrist_x = features[:, self.JOINT_IDX['right_wrist'], 0]
        elbow_spread = np.abs(left_wrist_x - right_wrist_x)
        if np.max(elbow_spread) > 1.5:
            metrics['errors'].append('肘外扩')
        
        return metrics
    
    def _analyze_squat(self, features: np.ndarray) -> Dict:
        """深蹲分析"""
        metrics = {}
        
        # 膝关节角度
        left_knee = features[:, 13, 4]
        right_knee = features[:, 14, 4]
        knee_angles = (left_knee + right_knee) / 2
        
        bottom_idx = np.argmin(knee_angles)
        metrics['depth_knee_angle'] = float(knee_angles[bottom_idx])
        
        # 膝盖内扣检测
        left_knee_x = features[:, self.JOINT_IDX['left_knee'], 0]
        left_ankle_x = features[:, self.JOINT_IDX['left_ankle'], 0]
        right_knee_x = features[:, self.JOINT_IDX['right_knee'], 0]
        right_ankle_x = features[:, self.JOINT_IDX['right_ankle'], 0]
        
        knee_collapse_left = np.abs(left_knee_x - left_ankle_x)
        knee_collapse_right = np.abs(right_knee_x - right_ankle_x)
        metrics['knee_collapse'] = float(np.max([knee_collapse_left.max(), knee_collapse_right.max()]))
        
        # 躯干前倾角度
        shoulder_y = features[:, self.JOINT_IDX['left_shoulder'], 1]
        hip_y = features[:, self.JOINT_IDX['left_hip'], 1]
        torso_angle = np.abs(np.degrees(np.arctan2(shoulder_y - hip_y, 
                                                    features[:, self.JOINT_IDX['left_shoulder'], 0] - 
                                                    features[:, self.JOINT_IDX['left_hip'], 0])))
        metrics['torso_angle'] = float(np.mean(torso_angle))
        
        # 错误检测
        if metrics['knee_collapse'] > 0.1:
            metrics['errors'].append('膝盖内扣')
        
        if knee_angles[bottom_idx] > 100:
            metrics['errors'].append('未达深度')
        
        return metrics
    
    def _analyze_situp(self, features: np.ndarray) -> Dict:
        """仰卧起坐分析"""
        metrics = {}
        
        # 髋-肩角度（起坐角度）
        hip_y = features[:, self.JOINT_IDX['left_hip'], 1]
        shoulder_y = features[:, self.JOINT_IDX['left_shoulder'], 1]
        
        # 检测最高点
        max_idx = np.argmin(shoulder_y)
        
        # 起坐角度估算
        vertical_dist = shoulder_y[0] - shoulder_y[max_idx]
        metrics['up_angle'] = float(np.degrees(np.arctan2(vertical_dist, 0.5)))
        
        # 肘膝距离
        left_wrist = features[:, self.JOINT_IDX['left_wrist'], :2]
        left_knee = features[:, self.JOINT_IDX['left_knee'], :2]
        wrist_knee_dist = np.linalg.norm(left_wrist - left_knee, axis=1)
        metrics['touch_distance'] = float(wrist_knee_dist[max_idx])
        
        # 臀部稳定性
        hip_variance = np.var(hip_y)
        metrics['hip_stability'] = float(hip_variance)
        
        # 错误检测
        if hip_variance > 0.05:
            metrics['errors'].append('臀部离地')
        
        if wrist_knee_dist[max_idx] > 0.15:
            metrics['errors'].append('未触膝')
        
        return metrics
    
    def _analyze_jump_rope(self, features: np.ndarray) -> Dict:
        """跳绳分析"""
        metrics = {}
        
        # 踝关节高度变化（跳跃高度）
        left_ankle_y = features[:, self.JOINT_IDX['left_ankle'], 1]
        jump_amplitude = np.max(left_ankle_y) - np.min(left_ankle_y)
        metrics['jump_height'] = float(jump_amplitude)
        
        # 膝关节角度变化（落地缓冲）
        left_knee = features[:, 13, 4]
        knee_flexion = np.max(left_knee) - np.min(left_knee)
        metrics['landing_softness'] = float(knee_flexion)
        
        # 节奏规律性（跳跃周期）
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(-left_ankle_y, distance=10)  # 找最低点（着地）
        if len(peaks) > 1:
            intervals = np.diff(peaks)
            metrics['rhythm_regularity'] = float(np.std(intervals) / (np.mean(intervals) + 1e-6))
        else:
            metrics['rhythm_regularity'] = 1.0
        
        # 错误检测
        if knee_flexion < 10:
            metrics['errors'].append('膝盖过直')
        
        if metrics['rhythm_regularity'] > 0.3:
            metrics['errors'].append('节奏不稳')
        
        return metrics
    
    def _analyze_long_jump(self, features: np.ndarray) -> Dict:
        """跳远分析"""
        metrics = {}
        
        # 起跳帧检测（髋部Y坐标最小点）
        hip_y = features[:, self.JOINT_IDX['left_hip'], 1]
        takeoff_idx = np.argmin(hip_y)
        
        # 起跳角度（髋部速度向量）
        if takeoff_idx > 0 and takeoff_idx < len(features) - 1:
            hip_velocity_y = hip_y[takeoff_idx-1] - hip_y[takeoff_idx+1]
            hip_velocity_x = features[takeoff_idx, self.JOINT_IDX['left_hip'], 0] - \
                           features[takeoff_idx-1, self.JOINT_IDX['left_hip'], 0]
            takeoff_angle = np.degrees(np.arctan2(hip_velocity_y, abs(hip_velocity_x) + 1e-6))
            metrics['takeoff_angle'] = float(takeoff_angle)
        else:
            metrics['takeoff_angle'] = 0.0
        
        # 起跳膝角
        left_knee = features[:, 13, 4]
        metrics['takeoff_knee_angle'] = float(left_knee[takeoff_idx])
        
        # 摆臂幅度
        left_wrist_x = features[:, self.JOINT_IDX['left_wrist'], 0]
        arm_swing = np.max(left_wrist_x) - np.min(left_wrist_x)
        metrics['arm_swing'] = float(arm_swing)
        
        # 落地稳定性（最后几帧髋部晃动）
        landing_hip_variance = np.var(hip_y[-10:])
        metrics['landing_stability'] = float(landing_hip_variance)
        
        # 错误检测
        if metrics['takeoff_angle'] > 30:
            metrics['errors'].append('起跳角度过大')
        elif metrics['takeoff_angle'] < 15:
            metrics['errors'].append('起跳角度过小')
        
        if arm_swing < 0.2:
            metrics['errors'].append('未充分摆臂')
        
        return metrics
    
    def _analyze_pullup(self, features: np.ndarray) -> Dict:
        """引体向上分析"""
        metrics = {}
        
        # 下巴高度（相对于起点）
        nose_y = features[:, self.JOINT_IDX['nose'], 1]
        start_y = nose_y[0]
        max_height = start_y - np.min(nose_y)
        metrics['chin_height'] = float(max_height)
        
        # 手臂伸直角度
        left_elbow = features[:, 7, 3]
        right_elbow = features[:, 8, 3]
        metrics['min_elbow_angle'] = float(np.min([left_elbow.min(), right_elbow.min()]))
        metrics['max_elbow_angle'] = float(np.max([left_elbow.max(), right_elbow.max()]))
        
        # 身体摆动（髋部水平位移）
        hip_x = features[:, self.JOINT_IDX['left_hip'], 0]
        hip_swing = np.max(hip_x) - np.min(hip_x)
        metrics['body_swing'] = float(hip_swing)
        
        # 上拉时间
        top_idx = np.argmin(nose_y)
        metrics['pull_time'] = float(top_idx / 30.0)
        
        # 错误检测
        if max_height < 0.05:
            metrics['errors'].append('未过杆')
        
        if metrics['max_elbow_angle'] < 150:
            metrics['errors'].append('未充分下放')
        
        if hip_swing > 0.15:
            metrics['errors'].append('身体摆动')
        
        return metrics
    
    @staticmethod
    def compute_score(metrics: Dict, action_config: Dict) -> Tuple[float, Dict]:
        """
        根据指标计算得分
        
        Returns:
            overall_score: 总分 (0-100)
            details: 各项子得分
        """
        standards = action_config['standard_params']
        scores = {}
        
        for param_name, standard in standards.items():
            if param_name not in metrics:
                continue
            
            value = metrics[param_name]
            min_val, max_val = standard['min'], standard['max']
            ideal = standard.get('ideal', (min_val + max_val) / 2)
            weight = standard.get('weight', 0.1)
            
            # 计算偏离理想值的程度
            if min_val <= value <= max_val:
                # 在范围内，计算与理想值的接近度
                deviation = abs(value - ideal) / (max_val - min_val + 1e-6)
                score = max(0, 100 - deviation * 50)
            else:
                # 超出范围，给低分
                score = max(0, 50 - abs(value - ideal) / (max_val - min_val + 1e-6) * 50)
            
            scores[param_name] = {
                'value': float(value),
                'score': float(score),
                'weight': float(weight)
            }
        
        # 加权计算总分
        if scores:
            total_weight = sum(s['weight'] for s in scores.values())
            overall = sum(s['score'] * s['weight'] for s in scores.values()) / total_weight
        else:
            overall = 50.0
        
        # 根据错误数量扣分
        error_count = len(metrics.get('errors', []))
        penalty = min(20, error_count * 5)
        overall = max(0, overall - penalty)
        
        return overall, scores


def load_skeleton_from_json(json_path: str) -> np.ndarray:
    """从JSON文件加载骨骼数据"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    sequence = []
    for frame in data['skeleton_sequence']:
        coords = [[kp['x'], kp['y']] for kp in frame['keypoints']]
        sequence.append(coords)
    
    return np.array(sequence)


def save_features_to_npy(features: np.ndarray, save_path: str):
    """保存特征到npy文件"""
    np.save(save_path, features)


def visualize_skeleton(features: np.ndarray, save_path: Optional[str] = None):
    """可视化骨骼序列（用于调试）"""
    import matplotlib.pyplot as plt
    
    T, N, C = features.shape
    fig, axes = plt.subplots(1, min(8, T), figsize=(20, 4))
    
    if T == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        frame_idx = i * T // min(8, T)
        coords = features[frame_idx, :, :2]
        
        # 绘制关节点
        ax.scatter(coords[:, 0], coords[:, 1], c='blue', s=50)
        
        # 绘制骨骼连接
        processor = SkeletonProcessor()
        for pair in processor.SKELETON_PAIRS:
            if pair[0] < N and pair[1] < N:
                ax.plot([coords[pair[0], 0], coords[pair[1], 0]],
                       [coords[pair[0], 1], coords[pair[1], 1]], 'g-', linewidth=2)
        
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title(f'Frame {frame_idx}')
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()
