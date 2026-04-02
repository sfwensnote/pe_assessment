#!/usr/bin/env python3
# @author Coder建设｜javpower
"""
4_train_phase.py
训练阶段分割模型

用法:
    python 4_train_phase.py [--action pushup] [--epochs 80]
"""

import os
import sys
import json
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict

# 添加项目路径
sys.path.append(str(Path(__file__).parent))
from utils.annotation_io import load_json_any
from utils.skeleton import SkeletonProcessor
from utils.models import TemporalUNet
from utils.augmentation import SkeletonAugmentor
from utils.metrics import AssessmentMetrics, MetricsTracker
from utils.training_manifest import read_manifest, resolve_project_path

# 加载配置
CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH, encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)


class PhaseDataset(Dataset):
    """阶段分割数据集"""

    def __init__(self, samples: List[Dict], num_phases: int, augment: bool = False):
        self.samples = samples
        self.num_phases = num_phases
        self.augment = augment
        self.augmentor = SkeletonAugmentor() if augment else None

        # 计算类别权重
        self.class_weights = self._compute_class_weights()

    def _compute_class_weights(self):
        """计算阶段类别权重（处理类别不平衡）"""
        phase_counts = np.zeros(self.num_phases)

        for sample in self.samples:
            phases = np.array(sample["phases"])
            for i in range(self.num_phases):
                phase_counts[i] += np.sum(phases == i)

        # 反向频率加权
        total = np.sum(phase_counts)
        weights = total / (phase_counts + 1e-6)
        weights = weights / weights.sum() * self.num_phases

        return torch.FloatTensor(weights)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        features = sample["features"].copy()
        phases = np.array(sample["phases"])

        # 数据增强
        if self.augment and self.augmentor:
            features, phases = self.augmentor(features, phases)

        return {
            "features": torch.FloatTensor(features),
            "phases": torch.LongTensor(phases),
            "video_id": sample["video_id"],
        }


def _resample_phases(phases: list[int] | np.ndarray, target_len: int) -> np.ndarray:
    """Resample frame-wise phase labels to match feature length."""

    values = np.asarray(phases, dtype=int)
    if len(values) == target_len:
        return values
    if len(values) == 0:
        return np.zeros(target_len, dtype=int)
    if len(values) == 1:
        return np.full(target_len, values[0], dtype=int)

    indices = np.rint(np.linspace(0, len(values) - 1, target_len)).astype(int)
    indices = np.clip(indices, 0, len(values) - 1)
    return values[indices]


def load_action_data(action_type: str, manifest_file: str | None = None):
    """加载指定动作的数据"""
    print(f"\n加载 {action_type} 数据...")

    anno_dir = Path(CONFIG["paths"]["annotations"]) / action_type
    processor = SkeletonProcessor(target_frames=CONFIG["skeleton"]["target_frames"])

    samples = []

    annotation_jobs = []
    if manifest_file:
        manifest_path = resolve_project_path(manifest_file)
        manifest_records = read_manifest(manifest_path)
        print(f"使用清单文件: {manifest_path}")
        for record in manifest_records:
            if record["action_type"] != action_type:
                continue
            annotation_jobs.append(
                (
                    resolve_project_path(record["annotation_path"]),
                    resolve_project_path(record["skeleton_path"]),
                )
            )
    else:
        for json_path in anno_dir.glob("*.json"):
            skeleton_path = (
                Path(CONFIG["paths"]["skeletons"])
                / action_type
                / f"{json_path.stem}.json"
            )
            annotation_jobs.append((json_path, skeleton_path))

    for json_path, skeleton_path in annotation_jobs:
        try:
            anno, _ = load_json_any(json_path)

            if not skeleton_path.exists():
                continue

            skeleton_data, _ = load_json_any(skeleton_path)

            # 转换为numpy
            sequence = []
            for frame in skeleton_data["skeleton_sequence"]:
                coords = [[kp["x"], kp["y"]] for kp in frame["keypoints"]]
                sequence.append(coords)
            sequence = np.array(sequence)

            # 预处理
            features = processor.process(sequence)
            phases = _resample_phases(anno["phases"], len(features))

            samples.append(
                {
                    "features": features,
                    "phases": phases,
                    "video_id": anno["video_id"],
                }
            )

        except Exception as e:
            print(f"  错误加载 {json_path}: {e}")

    print(f"加载了 {len(samples)} 个样本")

    if len(samples) < 10:
        return [], []

    # 划分训练集/验证集
    np.random.seed(CONFIG["training"]["seed"])
    np.random.shuffle(samples)
    n_train = int(len(samples) * 0.8)

    return samples[:n_train], samples[n_train:]


def train_action_phase(action_type: str, args):
    """训练指定动作的阶段分割模型"""
    print(f"\n{'=' * 60}")
    print(f"训练动作阶段分割: {action_type} ({CONFIG['actions'][action_type]['name']})")
    print(f"{'=' * 60}")

    # 加载数据
    train_samples, val_samples = load_action_data(
        action_type, args.manifest_file or None
    )

    if len(train_samples) < 10:
        print(f"样本不足 ({len(train_samples)} 个)，跳过 {action_type}")
        return

    num_phases = len(CONFIG["actions"][action_type]["phases"])

    # 创建数据集
    train_dataset = PhaseDataset(train_samples, num_phases, augment=True)
    val_dataset = PhaseDataset(val_samples, num_phases, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 创建模型
    input_channels = CONFIG["skeleton"]["num_joints"] * 9
    model = TemporalUNet(in_channels=input_channels, num_phases=num_phases).to(device)

    # 损失函数（加权）
    criterion = nn.CrossEntropyLoss(weight=train_dataset.class_weights.to(device))

    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=CONFIG["training"]["phase_model"]["weight_decay"],
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 指标追踪
    tracker = MetricsTracker()

    # 训练循环
    best_f1 = 0.0
    save_dir = Path(args.checkpoint_dir or CONFIG["paths"]["checkpoints"])
    save_dir.mkdir(exist_ok=True)

    for epoch in range(args.epochs):
        # 训练
        model.train()
        train_loss = 0.0

        for batch in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]", leave=False
        ):
            features = batch["features"].to(device)
            phases = batch["phases"].to(device)

            optimizer.zero_grad()
            outputs = model(features)  # [B, T, num_phases]

            # 计算损失
            loss = criterion(outputs.reshape(-1, num_phases), phases.reshape(-1))

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # 验证
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device)
                phases = batch["phases"].to(device)

                outputs = model(features)
                loss = criterion(outputs.reshape(-1, num_phases), phases.reshape(-1))

                val_loss += loss.item()

                preds = outputs.argmax(2).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(phases.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)

        # 计算指标
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        metrics = AssessmentMetrics.compute_phase_metrics(
            all_preds.flatten(), all_targets.flatten(), num_phases
        )

        scheduler.step()

        # 记录
        tracker.update(
            {
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "frame_accuracy": metrics["frame_accuracy"],
                "boundary_f1": metrics["boundary_f1"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        print(
            f"Epoch {epoch + 1}: "
            f"Train Loss={avg_train_loss:.4f}, "
            f"Val Loss={avg_val_loss:.4f}, "
            f"Frame Acc={metrics['frame_accuracy']:.4f}, "
            f"Boundary F1={metrics['boundary_f1']:.4f}"
        )

        # 保存最优模型（基于边界F1）
        if metrics["boundary_f1"] > best_f1:
            best_f1 = metrics["boundary_f1"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_f1": best_f1,
                    "action_type": action_type,
                    "num_phases": num_phases,
                    "phase_names": CONFIG["actions"][action_type]["phases"],
                },
                save_dir / f"phase_model_{action_type}.pth",
            )
            print(f"  -> 保存最优模型 (F1={best_f1:.4f})")

    tracker.save(str(save_dir / f"phase_history_{action_type}.json"))

    print(f"\n{action_type} 训练完成！最优边界F1: {best_f1:.4f}")


def main():
    parser = argparse.ArgumentParser(description="训练阶段分割模型")
    parser.add_argument(
        "--action", type=str, default=None, help="指定训练的动作类型，默认训练所有"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=CONFIG["training"]["phase_model"]["epochs"],
        help="训练轮数",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=CONFIG["training"]["phase_model"]["batch_size"],
        help="批次大小",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=CONFIG["training"]["phase_model"]["lr"],
        help="学习率",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=CONFIG["training"].get("device", "cuda"),
        help="训练设备",
    )
    parser.add_argument(
        "--manifest_file",
        type=str,
        default="",
        help="可选训练清单文件（JSONL）",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="",
        help="可选输出目录，默认使用 checkpoints",
    )
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    args = parser.parse_args()

    if args.action:
        # 训练指定动作
        if args.action in CONFIG["actions"]:
            train_action_phase(args.action, args)
        else:
            print(f"错误: 未知的动作类型 '{args.action}'")
            print(f"可用动作: {list(CONFIG['actions'].keys())}")
    else:
        # 训练所有动作
        for action_type in CONFIG["actions"].keys():
            try:
                train_action_phase(action_type, args)
            except Exception as e:
                print(f"训练 {action_type} 失败: {e}")
                continue


if __name__ == "__main__":
    main()
