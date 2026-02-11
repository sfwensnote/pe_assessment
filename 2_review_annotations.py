#!/usr/bin/env python3
# @author Coder建设｜javpower
"""
2_review_annotations.py
人工复核工具 - 查看和修正自动标注结果

用法:
    python 2_review_annotations.py [--action pushup]

操作说明:
    - 左右方向键: 逐帧查看
    - 空格键: 播放/暂停
    - S键: 保存标注
    - N键: 下一个样本
    - Q键: 退出
"""

import os
import sys
import json
import yaml
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict

# 添加项目路径
sys.path.append(str(Path(__file__).parent))
from utils.skeleton import SkeletonProcessor

# 加载配置
CONFIG_PATH = Path(__file__).parent / 'config.yaml'
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

# 尝试导入GUI库
try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    print("警告: tkinter 不可用，将使用命令行模式")


def load_annotations(action_type: str = None) -> List[Dict]:
    """加载所有标注数据"""
    anno_dir = Path(CONFIG['paths']['annotations'])
    samples = []
    
    action_types = [action_type] if action_type else list(CONFIG['actions'].keys())
    
    for at in action_types:
        action_anno_dir = anno_dir / at
        
        if not action_anno_dir.exists():
            continue
        
        for json_path in action_anno_dir.glob('*.json'):
            with open(json_path) as f:
                anno = json.load(f)
            
            # 找到对应的骨骼文件
            skeleton_path = Path(CONFIG['paths']['skeletons']) / at / f"{anno['video_id']}.json"
            
            if skeleton_path.exists():
                samples.append({
                    'annotation_path': json_path,
                    'skeleton_path': skeleton_path,
                    'data': anno
                })
    
    return samples


class CLIReviewTool:
    """命令行复核工具"""
    
    def __init__(self, samples: List[Dict]):
        self.samples = samples
        self.current_idx = 0
    
    def display_sample(self, idx: int):
        """显示样本信息"""
        if idx >= len(self.samples):
            print("已到达最后一个样本")
            return False
        
        sample = self.samples[idx]
        anno = sample['data']
        
        print("\n" + "=" * 60)
        print(f"样本 {idx + 1} / {len(self.samples)}")
        print("=" * 60)
        print(f"视频ID: {anno['video_id']}")
        print(f"动作类型: {anno['action_type']} ({CONFIG['actions'][anno['action_type']]['name']})")
        print(f"自动标注: {'是' if anno.get('auto_annotated') else '否'}")
        print(f"已复核: {'是' if anno.get('reviewed') else '否'}")
        print(f"阶段数: {len(set(anno['phases']))}")
        
        quality = anno.get('quality', {})
        print(f"\n质量评估:")
        print(f"  总分: {quality.get('overall_score', 'N/A')}")
        print(f"  是否标准: {'是' if quality.get('is_standard') else '否'}")
        
        errors = quality.get('errors', [])
        if errors:
            print(f"  错误: {', '.join(errors)}")
        else:
            print(f"  错误: 无")
        
        print("\n操作: [s]保存 [n]下一个 [p]上一个 [q]退出")
        
        return True
    
    def mark_reviewed(self, idx: int):
        """标记为已复核"""
        sample = self.samples[idx]
        anno = sample['data']
        anno['reviewed'] = True
        
        with open(sample['annotation_path'], 'w') as f:
            json.dump(anno, f, indent=2, ensure_ascii=False)
        
        print(f"已保存: {sample['annotation_path']}")
    
    def run(self):
        """运行命令行界面"""
        print(f"\n加载了 {len(self.samples)} 个样本")
        
        while self.current_idx < len(self.samples):
            if not self.display_sample(self.current_idx):
                break
            
            cmd = input("\n选择操作: ").strip().lower()
            
            if cmd == 'q':
                print("退出复核工具")
                break
            elif cmd == 'n':
                self.mark_reviewed(self.current_idx)
                self.current_idx += 1
            elif cmd == 'p':
                if self.current_idx > 0:
                    self.current_idx -= 1
            elif cmd == 's':
                self.mark_reviewed(self.current_idx)
            else:
                print("无效命令")
        
        print("复核完成！")


def main():
    parser = argparse.ArgumentParser(description='人工复核标注数据')
    parser.add_argument('--action', type=str, default=None,
                       help='指定复核的动作类型')
    args = parser.parse_args()
    
    # 加载样本
    samples = load_annotations(args.action)
    
    if not samples:
        print("没有找到标注数据，请先运行自动标注")
        return
    
    # 使用命令行工具
    tool = CLIReviewTool(samples)
    tool.run()


if __name__ == '__main__':
    main()
