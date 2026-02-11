#!/usr/bin/env python3
# @author Coder建设｜javpower
"""
0_preprocess_videos.py
视频预处理：提取骨骼关键点

用法:
    python 0_preprocess_videos.py [--action pushup] [--input_dir path] [--output_dir path]

支持的动作:
    pushup - 俯卧撑
    squat - 深蹲
    situp - 仰卧起坐
    jump_rope - 跳绳
    long_jump - 跳远
    pullup - 引体向上
"""

import argparse
import json
from pathlib import Path

import yaml
from tqdm import tqdm

from ultralytics import YOLO

# 加载配置
CONFIG_PATH = Path(__file__).parent / 'config.yaml'
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)


def extract_skeleton_from_video(video_path: Path, model, conf_threshold: float = 0.3) -> dict:
    """
    从视频中提取骨骼关键点
    
    Args:
        video_path: 视频文件路径
        model: YOLO模型
        conf_threshold: 置信度阈值
    
    Returns:
        dict: 包含骨骼序列的字典
    """
    results = model(str(video_path), verbose=False, stream=True)
    
    skeleton_sequence = []
    
    for frame_idx, result in enumerate(results):
        if result.keypoints is None or len(result.keypoints) == 0:
            continue
        
        # 取最前面的人（假设单人）
        kpts = result.keypoints.xy[0].cpu().numpy()
        scores = result.keypoints.conf[0].cpu().numpy()
        
        # 过滤低置信度
        kpts[scores < conf_threshold] = [0, 0]
        
        frame_data = {
            'frame_id': frame_idx,
            'timestamp': frame_idx / CONFIG['camera']['fps'],
            'keypoints': [
                {
                    'id': i,
                    'name': name,
                    'x': float(x),
                    'y': float(y),
                    'score': float(s)
                }
                for i, (name, (x, y), s) in enumerate(
                    zip(CONFIG['skeleton']['joint_names'], kpts, scores)
                )
            ]
        }
        skeleton_sequence.append(frame_data)
    
    return {
        'video_id': video_path.stem,
        'source_path': str(video_path),
        'total_frames': len(skeleton_sequence),
        'fps': CONFIG['camera']['fps'],
        'skeleton_sequence': skeleton_sequence
    }


def process_action(action_type: str, input_dir: Path, output_dir: Path, model):
    """
    处理单个动作类型的所有视频
    """
    if action_type not in CONFIG['actions']:
        print(f"警告: 未知的动作类型 '{action_type}'，跳过")
        return 0
    
    action_input_dir = input_dir / action_type
    if not action_input_dir.exists():
        print(f"警告: 输入目录不存在 {action_input_dir}")
        return 0
    
    # 创建输出目录
    action_output_dir = output_dir / action_type
    action_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有视频文件
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm']
    video_files = []
    for ext in video_extensions:
        video_files.extend(action_input_dir.glob(ext))
    
    if not video_files:
        print(f"警告: 在 {action_input_dir} 中未找到视频文件")
        return 0
    
    print(f"\n处理动作: {action_type} ({CONFIG['actions'][action_type]['name']})")
    print(f"找到 {len(video_files)} 个视频文件")
    
    processed_count = 0
    error_count = 0
    
    for video_path in tqdm(video_files, desc=f"提取 {action_type}"):
        out_path = action_output_dir / f"{video_path.stem}.json"
        
        # 跳过已处理
        if out_path.exists():
            continue
        
        try:
            # 提取骨骼
            skeleton_data = extract_skeleton_from_video(video_path, model)
            
            # 过滤太短的视频
            if skeleton_data['total_frames'] < 10:
                print(f"  跳过 {video_path.name}: 帧数太少 ({skeleton_data['total_frames']})")
                continue
            
            # 保存
            with open(out_path, 'w') as f:
                json.dump(skeleton_data, f, indent=2)
            
            processed_count += 1
            
        except Exception as e:
            print(f"  错误处理 {video_path.name}: {e}")
            error_count += 1
    
    print(f"完成: 成功 {processed_count} 个, 失败 {error_count} 个")
    return processed_count


def main():
    parser = argparse.ArgumentParser(description='提取视频骨骼关键点')
    parser.add_argument('--action', type=str, default=None,
                       help='指定处理的动作类型，默认处理所有')
    parser.add_argument('--input_dir', type=str, default=CONFIG['paths']['raw_videos'],
                       help='输入视频目录')
    parser.add_argument('--output_dir', type=str, default=CONFIG['paths']['skeletons'],
                       help='输出骨骼目录')
    parser.add_argument('--model', type=str, default='yolov8x-pose.pt',
                       help='YOLO模型路径')
    parser.add_argument('--conf', type=float, default=0.3,
                       help='关键点置信度阈值')
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载YOLO模型
    print("加载YOLO姿态检测模型...")
    try:
        model = YOLO(args.model)
        print(f"模型加载成功: {args.model}")
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("尝试下载模型...")
        model = YOLO('yolov8x-pose.pt')
    
    # 处理动作
    if args.action:
        # 处理指定动作
        process_action(args.action, input_dir, output_dir, model)
    else:
        # 处理所有动作
        total_processed = 0
        for action_type in CONFIG['actions'].keys():
            count = process_action(action_type, input_dir, output_dir, model)
            total_processed += count
        
        print(f"\n{'='*50}")
        print(f"全部处理完成！共处理 {total_processed} 个视频")
    
    print(f"骨骼数据保存在: {output_dir}")


if __name__ == '__main__':
    main()
