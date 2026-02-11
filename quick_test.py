#!/usr/bin/env python3
# @author Coder建设｜javpower
"""
quick_test.py
快速测试脚本 - 验证环境是否正确配置

用法:
    python quick_test.py
"""

import sys
from pathlib import Path

def check_imports():
    """检查依赖导入"""
    print("检查依赖导入...")
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('yaml', 'PyYAML'),
        ('sklearn', 'scikit-learn'),
        ('scipy', 'SciPy'),
        ('tqdm', 'tqdm'),
        ('PIL', 'Pillow'),
    ]
    
    optional_packages = [
        ('ultralytics', 'Ultralytics (YOLO)'),
        ('matplotlib', 'Matplotlib'),
        ('pandas', 'Pandas'),
    ]
    
    all_ok = True
    
    for module, name in required_packages:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - 未安装")
            all_ok = False
    
    print("\n可选依赖:")
    for module, name in optional_packages:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ⚠ {name} - 未安装（可选）")
    
    return all_ok


def check_cuda():
    """检查CUDA可用性"""
    print("\n检查CUDA...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA 可用")
            print(f"    设备: {torch.cuda.get_device_name(0)}")
            print(f"    显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print(f"  ⚠ CUDA 不可用，将使用CPU训练（较慢）")
    except Exception as e:
        print(f"  ✗ 检查CUDA出错: {e}")


def check_project_structure():
    """检查项目结构"""
    print("\n检查项目结构...")
    
    required_dirs = [
        'utils',
        'data/raw_videos',
        'data/skeletons',
        'data/annotations',
        'checkpoints',
    ]
    
    required_files = [
        'config.yaml',
        '0_preprocess_videos.py',
        '1_auto_annotate.py',
        '3_train_action.py',
        '4_train_phase.py',
        '5_train_quality.py',
        '6_inference.py',
    ]
    
    all_ok = True
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"  ✓ {dir_path}/")
        else:
            print(f"  ⚠ {dir_path}/ - 不存在，将自动创建")
            path.mkdir(parents=True, exist_ok=True)
    
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} - 不存在")
            all_ok = False
    
    return all_ok


def test_models():
    """测试模型加载"""
    print("\n测试模型定义...")
    
    try:
        sys.path.append(str(Path(__file__).parent))
        from utils.models import STGCNAction, TemporalUNet, QualityNet
        from utils.skeleton import SkeletonProcessor
        from utils.metrics import AssessmentMetrics
        
        print("  ✓ 模型模块导入成功")
        
        # 测试模型创建
        import torch
        action_model = STGCNAction(num_classes=6)
        phase_model = TemporalUNet(num_phases=5)
        quality_model = QualityNet(num_errors=10)
        
        print("  ✓ 模型创建成功")
        
        # 测试前向传播
        dummy_input = torch.randn(2, 60, 17, 9)
        
        with torch.no_grad():
            action_out = action_model(dummy_input)
            phase_out = phase_model(dummy_input)
            quality_out = quality_model(dummy_input)
        
        print("  ✓ 前向传播测试通过")
        print(f"    动作识别输出: {action_out.shape}")
        print(f"    阶段分割输出: {phase_out.shape}")
        print(f"    质量评估输出: {quality_out['overall'].shape}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ 模型测试失败: {e}")
        return False


def main():
    print("=" * 60)
    print("体育动作评估系统 - 环境测试")
    print("=" * 60)
    
    results = []
    
    results.append(("依赖导入", check_imports()))
    check_cuda()
    results.append(("项目结构", check_project_structure()))
    results.append(("模型定义", test_models()))
    
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    for name, ok in results:
        status = "✓ 通过" if ok else "✗ 失败"
        print(f"{name}: {status}")
    
    all_passed = all(ok for _, ok in results)
    
    if all_passed:
        print("\n✓ 所有测试通过！可以开始使用了。")
        print("\n快速开始:")
        print("  1. 将视频放入 data/raw_videos/动作类型/ 目录")
        print("  2. 运行: python 0_preprocess_videos.py")
        print("  3. 运行: python 1_auto_annotate.py")
        print("  4. 运行: python 3_train_action.py")
        return 0
    else:
        print("\n✗ 部分测试失败，请检查错误信息。")
        return 1


if __name__ == '__main__':
    sys.exit(main())
