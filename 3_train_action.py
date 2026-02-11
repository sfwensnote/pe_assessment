#!/usr/bin/env python3
# @author Coder建设｜javpower
"""
3_train_action.py
训练动作识别模型

用法:
    python 3_train_action.py [--epochs 100] [--batch_size 64]
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
from utils.skeleton import SkeletonProcessor, load_skeleton_from_json
from utils.models import STGCNAction, model_summary
from utils.augmentation import SkeletonAugmentor
from utils.metrics import AssessmentMetrics, MetricsTracker

# 加载配置
CONFIG_PATH = Path(__file__).parent / 'config.yaml'
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)


class ActionDataset(Dataset):
    """动作识别数据集"""
    
    def __init__(self, samples: List[Dict], augment: bool = False):
        self.samples = samples
        self.augment = augment
        self.augmentor = SkeletonAugmentor() if augment else None
        
        # 动作到ID的映射
        self.action_to_id = {
            name: idx for idx, name in enumerate(CONFIG['actions'].keys())
        }
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        features = sample['features'].copy()
        action_id = self.action_to_id[sample['action_type']]
        
        # 数据增强
        if self.augment and self.augmentor:
            features, _ = self.augmentor(features)
        
        return {
            'features': torch.FloatTensor(features),
            'label': torch.LongTensor([action_id])[0],
            'video_id': sample['video_id']
        }


def load_data():
    """加载训练数据"""
    print("加载训练数据...")
    
    anno_dir = Path(CONFIG['paths']['annotations'])
    processor = SkeletonProcessor(target_frames=CONFIG['skeleton']['target_frames'])
    
    samples = []
    
    for action_type in CONFIG['actions'].keys():
        action_anno_dir = anno_dir / action_type
        
        if not action_anno_dir.exists():
            print(f"警告: 标注目录不存在 {action_anno_dir}")
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
                
                samples.append({
                    'features': features,
                    'action_type': action_type,
                    'video_id': anno['video_id']
                })
                
            except Exception as e:
                print(f"  错误加载 {json_path}: {e}")
    
    print(f"总共加载 {len(samples)} 个样本")
    
    if len(samples) < 10:
        print("错误: 样本数量太少，无法训练")
        return [], []
    
    # 划分训练集/验证集
    np.random.seed(CONFIG['training']['seed'])
    np.random.shuffle(samples)
    
    n_train = int(len(samples) * 0.8)
    train_samples = samples[:n_train]
    val_samples = samples[n_train:]
    
    print(f"训练集: {len(train_samples)}, 验证集: {len(val_samples)}")
    
    return train_samples, val_samples


def train_model(args):
    """训练动作识别模型"""
    # 加载数据
    train_samples, val_samples = load_data()
    
    if not train_samples:
        return
    
    # 创建数据集
    train_dataset = ActionDataset(train_samples, augment=True)
    val_dataset = ActionDataset(val_samples, augment=False)
    
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
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    num_classes = len(CONFIG['actions'])
    model = STGCNAction(num_classes=num_classes).to(device)
    
    print("\n模型结构:")
    model_summary(model)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=CONFIG['training']['action_model']['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    
    # 指标追踪
    tracker = MetricsTracker()
    
    # 训练循环
    best_acc = 0.0
    save_dir = Path(CONFIG['paths']['checkpoints'])
    save_dir.mkdir(exist_ok=True)
    
    print(f"\n开始训练，共 {args.epochs} 个epoch...")
    
    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Train]')
        for batch in pbar:
            features = batch['features'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Val]', leave=False):
                features = batch['features'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
        
        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        # 更新学习率
        scheduler.step()
        
        # 记录指标
        tracker.update({
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        print(f'Epoch {epoch+1}: '
              f'Train Loss={avg_train_loss:.4f}, Acc={train_acc:.2f}%, '
              f'Val Loss={avg_val_loss:.4f}, Acc={val_acc:.2f}%')
        
        # 保存最优模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_acc': best_acc,
                'action_to_id': {name: idx for idx, name in enumerate(CONFIG['actions'].keys())},
                'config': CONFIG
            }, save_dir / 'action_model_best.pth')
            print(f'  -> 保存最优模型 (Val Acc: {best_acc:.2f}%)')
    
    # 保存训练历史
    tracker.save(save_dir / 'action_training_history.json')
    tracker.plot(save_dir / 'action_training_curves.png')
    
    print(f'\n训练完成！最优验证准确率: {best_acc:.2f}%')
    print(f'模型保存在: {save_dir / "action_model_best.pth"}')


def main():
    parser = argparse.ArgumentParser(description='训练动作识别模型')
    parser.add_argument('--epochs', type=int, default=CONFIG['training']['action_model']['epochs'],
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=CONFIG['training']['action_model']['batch_size'],
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=CONFIG['training']['action_model']['lr'],
                       help='学习率')
    parser.add_argument('--device', type=str, default=CONFIG['training'].get('device', 'cuda'),
                       help='训练设备')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载线程数')
    args = parser.parse_args()
    
    train_model(args)


if __name__ == '__main__':
    main()
