# @author Coder建设｜javpower
"""
数据增强模块
针对骨骼数据的各种增强技术
"""

import numpy as np
import torch
from typing import Tuple, Optional


class SkeletonAugmentor:
    """骨骼数据增强器"""
    
    def __init__(self, 
                 rotation_range: float = 15.0,
                 scale_range: Tuple[float, float] = (0.9, 1.1),
                 translation_range: float = 0.1,
                 noise_std: float = 0.01,
                 time_warp_range: Tuple[float, float] = (0.85, 1.15),
                 flip_prob: float = 0.5,
                 mask_prob: float = 0.1):
        """
        Args:
            rotation_range: 旋转角度范围（度）
            scale_range: 缩放范围
            translation_range: 平移范围
            noise_std: 高斯噪声标准差
            time_warp_range: 时间扭曲范围
            flip_prob: 水平翻转概率
            mask_prob: 关节遮罩概率
        """
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.translation_range = translation_range
        self.noise_std = noise_std
        self.time_warp_range = time_warp_range
        self.flip_prob = flip_prob
        self.mask_prob = mask_prob
    
    def __call__(self, features: np.ndarray, 
                 labels: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        应用随机增强
        
        Args:
            features: [T, N, C] 特征序列
            labels: 可选的标签
        
        Returns:
            augmented_features, augmented_labels
        """
        features = features.copy()
        
        # 随机选择增强操作
        if np.random.rand() < 0.5:
            features = self.random_rotation(features)
        
        if np.random.rand() < 0.5:
            features = self.random_scale(features)
        
        if np.random.rand() < 0.3:
            features = self.random_translation(features)
        
        if np.random.rand() < 0.5:
            features = self.add_noise(features)
        
        if np.random.rand() < 0.3:
            features, labels = self.time_warp(features, labels)
        
        if np.random.rand() < self.flip_prob:
            features = self.horizontal_flip(features)
        
        if np.random.rand() < 0.2:
            features = self.joint_masking(features)
        
        return features, labels
    
    def random_rotation(self, features: np.ndarray) -> np.ndarray:
        """随机旋转"""
        angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        rad = np.radians(angle)
        
        # 旋转矩阵
        rot_matrix = np.array([
            [np.cos(rad), -np.sin(rad)],
            [np.sin(rad), np.cos(rad)]
        ])
        
        # 应用旋转到坐标（前2维）
        coords = features[:, :, :2]
        features[:, :, :2] = coords @ rot_matrix.T
        
        return features
    
    def random_scale(self, features: np.ndarray) -> np.ndarray:
        """随机缩放"""
        scale = np.random.uniform(*self.scale_range)
        features[:, :, :2] *= scale
        return features
    
    def random_translation(self, features: np.ndarray) -> np.ndarray:
        """随机平移"""
        translation = np.random.uniform(-self.translation_range, 
                                       self.translation_range, size=2)
        features[:, :, :2] += translation
        return features
    
    def add_noise(self, features: np.ndarray) -> np.ndarray:
        """添加高斯噪声"""
        noise = np.random.normal(0, self.noise_std, features.shape)
        features = features + noise
        return features
    
    def time_warp(self, features: np.ndarray, 
                  labels: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        时间扭曲（改变速度）
        
        Args:
            features: [T, N, C]
            labels: 可选的阶段标签 [T]
        """
        T = len(features)
        scale = np.random.uniform(*self.time_warp_range)
        new_T = int(T * scale)
        
        # 插值
        old_indices = np.linspace(0, T - 1, T)
        new_indices = np.linspace(0, T - 1, new_T)
        
        new_features = np.zeros((new_T, features.shape[1], features.shape[2]))
        for i in range(features.shape[1]):
            for j in range(features.shape[2]):
                new_features[:, i, j] = np.interp(
                    new_indices, old_indices, features[:, i, j]
                )
        
        # 截断或填充回原始长度
        if new_T >= T:
            features = new_features[:T]
            if labels is not None:
                labels = labels[:T]
        else:
            pad = T - new_T
            features = np.concatenate([
                new_features,
                np.tile(new_features[-1:], (pad, 1, 1))
            ])
            if labels is not None:
                labels = np.concatenate([
                    labels,
                    np.tile(labels[-1:], pad)
                ])
        
        return features, labels
    
    def horizontal_flip(self, features: np.ndarray) -> np.ndarray:
        """水平翻转（镜像）"""
        # 翻转X坐标
        features[:, :, 0] = -features[:, :, 0]
        
        # 交换左右关节
        left_right_pairs = [
            (1, 2), (3, 4), (5, 6), (7, 8), (9, 10),
            (11, 12), (13, 14), (15, 16)
        ]
        
        for left, right in left_right_pairs:
            features[:, [left, right]] = features[:, [right, left]]
        
        return features
    
    def joint_masking(self, features: np.ndarray, 
                      num_joints_to_mask: int = 1) -> np.ndarray:
        """
        随机遮罩关节点（模拟遮挡）
        """
        T, N, C = features.shape
        joints_to_mask = np.random.choice(N, num_joints_to_mask, replace=False)
        
        for joint in joints_to_mask:
            # 将该关节的特征置为0或插值
            features[:, joint, :] = 0
            # 置信度设为0
            features[:, joint, 2] = 0
        
        return features
    
    def temporal_crop(self, features: np.ndarray, 
                      labels: Optional[np.ndarray] = None,
                      crop_ratio: float = 0.8) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        时序裁剪
        """
        T = len(features)
        crop_len = int(T * crop_ratio)
        start_idx = np.random.randint(0, T - crop_len + 1)
        
        features = features[start_idx:start_idx + crop_len]
        
        # 插值回原始长度
        if crop_len != T:
            old_indices = np.linspace(0, crop_len - 1, crop_len)
            new_indices = np.linspace(0, crop_len - 1, T)
            
            resampled = np.zeros((T, features.shape[1], features.shape[2]))
            for i in range(features.shape[1]):
                for j in range(features.shape[2]):
                    resampled[:, i, j] = np.interp(
                        new_indices, old_indices, features[:, i, j]
                    )
            features = resampled
            
            if labels is not None:
                labels = labels[start_idx:start_idx + crop_len]
                labels = np.interp(new_indices, old_indices, labels).astype(int)
        
        return features, labels


class MixupAugmentation:
    """Mixup数据增强（用于动作识别）"""
    
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
    
    def __call__(self, features1: torch.Tensor, labels1: torch.Tensor,
                 features2: torch.Tensor, labels2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mixup两个样本
        
        Returns:
            mixed_features, mixed_labels
        """
        lam = np.random.beta(self.alpha, self.alpha)
        
        mixed_features = lam * features1 + (1 - lam) * features2
        mixed_labels = lam * labels1 + (1 - lam) * labels2
        
        return mixed_features, mixed_labels


class CutMixAugmentation:
    """CutMix数据增强（时序版本）"""
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
    
    def __call__(self, features1: torch.Tensor, labels1: torch.Tensor,
                 features2: torch.Tensor, labels2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        CutMix两个时序样本
        """
        B, T, N, C = features1.shape
        
        # 随机选择时序区域
        lam = np.random.beta(self.alpha, self.alpha)
        cut_len = int(T * lam)
        start_idx = np.random.randint(0, T - cut_len + 1)
        
        # 混合
        mixed_features = features1.clone()
        mixed_features[:, start_idx:start_idx + cut_len] = \
            features2[:, start_idx:start_idx + cut_len]
        
        # 标签混合（按时间比例）
        lam = 1 - (cut_len / T)
        mixed_labels = lam * labels1 + (1 - lam) * labels2
        
        return mixed_features, mixed_labels


def apply_augmentation_pipeline(features: np.ndarray,
                                augmentation_config: dict) -> np.ndarray:
    """
    根据配置应用增强管道
    
    Args:
        features: 输入特征
        augmentation_config: 增强配置字典
    
    Returns:
        增强后的特征
    """
    augmentor = SkeletonAugmentor(**augmentation_config)
    features, _ = augmentor(features)
    return features


if __name__ == '__main__':
    # 测试增强
    np.random.seed(42)
    
    # 生成测试数据
    T, N, C = 60, 17, 9
    features = np.random.randn(T, N, C)
    
    augmentor = SkeletonAugmentor()
    
    print("Original shape:", features.shape)
    
    # 测试各种增强
    aug_features, _ = augmentor(features)
    print("After augmentation:", aug_features.shape)
    
    # 测试特定增强
    rotated = augmentor.random_rotation(features.copy())
    print("After rotation:", rotated.shape)
    
    flipped = augmentor.horizontal_flip(features.copy())
    print("After flip:", flipped.shape)
    
    masked = augmentor.joint_masking(features.copy())
    print("After masking:", masked.shape)
