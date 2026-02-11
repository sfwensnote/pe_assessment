#!/usr/bin/env python3
# @author Coder建设｜javpower
"""
5_train_quality.py
训练质量评估模型

用法:
    python 5_train_quality.py [--epochs 60] [--batch_size 32]
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
from utils.skeleton import SkeletonProcessor
from utils.models import QualityNet
from utils.augmentation import SkeletonAugmentor
from utils.metrics import AssessmentMetrics, MetricsTracker

# 加载配置
CONFIG_PATH = Path(__file__).parent / 'config.yaml'
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

# 错误类型
ERROR_TYPES = CONFIG['error_types']


class QualityDataset(Dataset):
    """质量评估数据集"""
    
    def __init__(self, samples: List[Dict], augment: bool = False):
        self.samples = samples
        self.augment = augment
        self.augmentor = SkeletonAugmentor() if augment else None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        features = sample['features'].copy()
        quality = sample['quality']
        
        # 数据增强
        if self.augment and self.augmentor:
            features, _ = self.augmentor(features)
        
        # 解析质量标签
        overall_score = quality.get('overall_score', 75.0) / 100.0  # 归一化到0-1
        
        metric_scores = quality.get('metric_scores', {
            'accuracy': 75.0,
            'stability': 75.0,
            'standard': 75.0,
            'safety': 75.0
        })
        
        metrics = torch.FloatTensor([
            metric_scores.get('accuracy', 75.0) / 100.0,
            metric_scores.get('stability', 75.0) / 100.0,
            metric_scores.get('standard', 75.0) / 100.0,
            metric_scores.get('safety', 75.0) / 100.0
        ])
        
        # 错误标签（多标签）
        errors = quality.get('errors', [])
        error_label = np.zeros(len(ERROR_TYPES), dtype=np.float32)
        for error in errors:
            if error in ERROR_TYPES:
                error_label[ERROR_TYPES.index(error)] = 1.0
        
        is_standard = 1.0 if quality.get('is_standard', False) else 0.0
        
        return {
            'features': torch.FloatTensor(features),
            'overall_score': torch.FloatTensor([overall_score]),
            'metric_scores': metrics,
            'error_label': torch.FloatTensor(error_label),
            'is_standard': torch.FloatTensor([is_standard])
        }


def load_all_data():
    """加载所有动作的标注数据"""
    print("加载质量评估数据...")
    
    anno_dir = Path(CONFIG['paths']['annotations'])
    processor = SkeletonProcessor(target_frames=CONFIG['skeleton']['target_frames'])
    
    samples = []
    
    for action_type in CONFIG['actions'].keys():
        action_anno_dir = anno_dir / action_type
        
        if not action_anno_dir.exists():
            continue
        
        for json_path in action_anno_dir.glob('*.json'):
            try:
                with open(json_path) as f:
                    anno = json.load(f)
                
                # 加载骨骼
                skeleton_path = Path(CONFIG['paths']['skeletons']) / action_type / f"{anno['video_id']}.json"
                
                if not skeleton_path.exists():
                    continue
                
                with open(skeleton_path) as f:
                    skeleton_data = json.load(f)
                
                # 转换为numpy
                sequence = []
                for frame in skeleton_data['skeleton_sequence']:
                    coords = [[kp['x'], kp['y']] for kp in frame['keypoints']]
                    sequence.append(coords)
                sequence = np.array(sequence)
                
                # 预处理
                features = processor.process(sequence)
                
                # 解析质量标签
                quality = anno.get('quality', {})
                
                samples.append({
                    'features': features,
                    'quality': quality,
                    'action_type': action_type,
                    'video_id': anno['video_id']
                })
                
            except Exception as e:
                print(f"  错误加载 {json_path}: {e}")
    
    print(f"加载了 {len(samples)} 个样本")
    
    if len(samples) < 10:
        return [], [], []
    
    # 划分数据集 (75% / 15% / 10%)
    np.random.seed(CONFIG['training']['seed'])
    np.random.shuffle(samples)
    
    n_train = int(len(samples) * 0.75)
    n_val = int(len(samples) * 0.15)
    
    return samples[:n_train], samples[n_train:n_train+n_val], samples[n_train+n_val:]


def train_model(args):
    """训练质量评估模型"""
    # 加载数据
    train_samples, val_samples, test_samples = load_all_data()
    
    if not train_samples:
        return
    
    # 创建数据集
    train_dataset = QualityDataset(train_samples, augment=True)
    val_dataset = QualityDataset(val_samples, augment=False)
    test_dataset = QualityDataset(test_samples, augment=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = QualityNet(
        num_metrics=4,
        num_errors=len(ERROR_TYPES)
    ).to(device)
    
    # 损失函数
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()
    
    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=CONFIG['training']['quality_model']['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    
    # 指标追踪
    tracker = MetricsTracker()
    
    # 训练循环
    best_val_loss = float('inf')
    save_dir = Path(CONFIG['paths']['checkpoints'])
    save_dir.mkdir(exist_ok=True)
    
    print(f"\n开始训练，共 {args.epochs} 个epoch...")
    
    for epoch in range(args.epochs):
        # 训练
        model.train()
        train_loss = 0.0
        train_score_loss = 0.0
        train_error_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Train]', leave=False):
            features = batch['features'].to(device)
            overall_score = batch['overall_score'].to(device)
            metric_scores = batch['metric_scores'].to(device)
            error_label = batch['error_label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            
            # 计算损失
            loss_score = mse_loss(outputs['overall'], overall_score)
            loss_metrics = mse_loss(
                torch.stack([
                    outputs['metrics']['accuracy'],
                    outputs['metrics']['stability'],
                    outputs['metrics']['standard'],
                    outputs['metrics']['safety']
                ], dim=1),
                metric_scores
            )
            loss_error = bce_loss(outputs['errors'], error_label)
            
            loss = loss_score + 0.5 * loss_metrics + loss_error
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_score_loss += loss_score.item()
            train_error_loss += loss_error.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0.0
        val_score_mae = 0.0
        val_error_acc = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                overall_score = batch['overall_score'].to(device)
                error_label = batch['error_label'].to(device)
                
                outputs = model(features)
                
                loss = mse_loss(outputs['overall'], overall_score)
                val_loss += loss.item()
                
                # MAE (转回0-100分)
                val_score_mae += torch.abs(outputs['overall'] - overall_score).sum().item() * 100
                
                # 错误检测准确率
                pred_error = (outputs['errors'] > 0.5).float()
                val_error_acc += (pred_error == error_label).float().mean().item()
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_mae = val_score_mae / len(val_dataset)
        avg_val_error_acc = val_error_acc / len(val_loader)
        
        scheduler.step()
        
        # 记录
        tracker.update({
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_mae': avg_val_mae,
            'val_error_acc': avg_val_error_acc,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        print(f"Epoch {epoch+1}: "
              f"Train Loss={avg_train_loss:.4f}, "
              f"Val Loss={avg_val_loss:.4f}, "
              f"Val MAE={avg_val_mae:.2f}分, "
              f"Val Error Acc={avg_val_error_acc:.4f}")
        
        # 保存最优模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_val_loss': best_val_loss,
                'error_types': ERROR_TYPES,
                'config': CONFIG
            }, save_dir / 'quality_model_best.pth')
            print(f"  -> 保存最优模型")
    
    # 最终测试
    print("\n最终测试...")
    model.eval()
    test_mae = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            overall_score = batch['overall_score'].to(device)
            
            outputs = model(features)
            
            test_mae += torch.abs(outputs['overall'] - overall_score).sum().item()
            
            all_predictions.extend((outputs['overall'] * 100).cpu().numpy())
            all_targets.extend((overall_score * 100).cpu().numpy())
    
    test_mae = test_mae / len(test_dataset) * 100
    print(f"测试集 MAE: {test_mae:.2f}分")
    
    # 计算相关性
    from scipy.stats import pearsonr
    corr, _ = pearsonr(np.array(all_targets).flatten(), np.array(all_predictions).flatten())
    print(f"与人工评分相关性: {corr:.3f}")
    
    # 保存
    tracker.save(save_dir / 'quality_training_history.json')
    tracker.plot(save_dir / 'quality_training_curves.png')
    
    print(f"\n质量评估模型训练完成！")
    print(f"模型保存在: {save_dir / 'quality_model_best.pth'}")


def main():
    parser = argparse.ArgumentParser(description='训练质量评估模型')
    parser.add_argument('--epochs', type=int, default=CONFIG['training']['quality_model']['epochs'],
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=CONFIG['training']['quality_model']['batch_size'],
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=CONFIG['training']['quality_model']['lr'],
                       help='学习率')
    parser.add_argument('--device', type=str, default=CONFIG['training'].get('device', 'cuda'),
                       help='训练设备')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载线程数')
    args = parser.parse_args()
    
    train_model(args)


if __name__ == '__main__':
    main()
