#!/usr/bin/env python3
# @author Coder建设｜javpower
"""
7_export_model.py
模型导出脚本 - 导出为ONNX/TorchScript格式用于部署

用法:
    python 7_export_model.py [--format onnx] [--output_dir deploy/models]
"""

import os
import sys
import json
import yaml
import argparse
import torch
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))
from utils.models import STGCNAction, TemporalUNet, QualityNet

# 加载配置
CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH, encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

DEFAULT_CHECKPOINT_DIR = Path(__file__).parent / "checkpoints" / "mixed_best_bundle"


class QualityNetExportWrapper(torch.nn.Module):
    """Flatten QualityNet dict output for ONNX/TorchScript export."""

    def __init__(self, model: QualityNet):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        metrics = torch.cat(
            [
                out["metrics"]["accuracy"],
                out["metrics"]["stability"],
                out["metrics"]["standard"],
                out["metrics"]["safety"],
            ],
            dim=1,
        )
        return out["overall"], metrics, out["errors"], out["is_standard"]


def export_action_model(format: str, output_dir: Path, checkpoint_dir: Path):
    """导出动作识别模型"""
    print("\n导出动作识别模型...")

    model_path = checkpoint_dir / "action_model_best.pth"
    if not model_path.exists():
        print(f"  错误: 模型不存在 {model_path}")
        return False

    # 加载模型
    device = torch.device("cpu")
    num_classes = len(CONFIG["actions"])
    model = STGCNAction(num_classes=num_classes).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 创建示例输入
    dummy_input = torch.randn(
        1, CONFIG["skeleton"]["target_frames"], CONFIG["skeleton"]["num_joints"], 9
    )

    if format == "onnx":
        # 导出ONNX
        output_path = output_dir / "action_model.onnx"
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=11,
            do_constant_folding=True,
        )
        print(f"  -> ONNX: {output_path}")

    elif format == "torchscript":
        # 导出TorchScript
        output_path = output_dir / "action_model.pt"
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(output_path)
        print(f"  -> TorchScript: {output_path}")

    # 保存标签映射
    label_map = {v: k for k, v in checkpoint["action_to_id"].items()}
    with open(output_dir / "action_labels.json", "w") as f:
        json.dump(label_map, f, indent=2)

    # 保存模型信息
    model_info = {
        "model_type": "action_recognition",
        "input_shape": [
            1,
            CONFIG["skeleton"]["target_frames"],
            CONFIG["skeleton"]["num_joints"],
            9,
        ],
        "output_shape": [1, num_classes],
        "num_classes": num_classes,
        "class_names": list(CONFIG["actions"].keys()),
    }
    with open(output_dir / "action_model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)

    return True


def export_phase_models(format: str, output_dir: Path, checkpoint_dir: Path):
    """导出阶段分割模型"""
    print("\n导出阶段分割模型...")

    exported = 0

    for action_type in CONFIG["actions"].keys():
        model_path = checkpoint_dir / f"phase_model_{action_type}.pth"

        if not model_path.exists():
            print(f"  跳过 {action_type}: 模型不存在")
            continue

        # 加载模型
        device = torch.device("cpu")
        num_phases = len(CONFIG["actions"][action_type]["phases"])
        model = TemporalUNet(num_phases=num_phases).to(device)

        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # 创建示例输入
        dummy_input = torch.randn(
            1, CONFIG["skeleton"]["target_frames"], CONFIG["skeleton"]["num_joints"], 9
        )

        if format == "onnx":
            output_path = output_dir / f"phase_model_{action_type}.onnx"
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                input_names=["input"],
                output_names=["phases"],
                dynamic_axes={"input": {0: "batch_size"}, "phases": {0: "batch_size"}},
                opset_version=11,
                do_constant_folding=True,
            )
            print(f"  -> {action_type}: {output_path}")

        elif format == "torchscript":
            output_path = output_dir / f"phase_model_{action_type}.pt"
            traced_model = torch.jit.trace(model, dummy_input)
            traced_model.save(output_path)
            print(f"  -> {action_type}: {output_path}")

        exported += 1

    print(f"  共导出 {exported} 个阶段分割模型")
    return True


def export_quality_model(format: str, output_dir: Path, checkpoint_dir: Path):
    """导出质量评估模型"""
    print("\n导出质量评估模型...")

    model_path = checkpoint_dir / "quality_model_best.pth"
    if not model_path.exists():
        print(f"  错误: 模型不存在 {model_path}")
        return False

    # 加载模型
    device = torch.device("cpu")
    num_errors = len(CONFIG["error_types"])
    model = QualityNet(num_errors=num_errors).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    export_model = QualityNetExportWrapper(model).eval()

    # 创建示例输入
    dummy_input = torch.randn(
        1, CONFIG["skeleton"]["target_frames"], CONFIG["skeleton"]["num_joints"], 9
    )

    if format == "onnx":
        output_path = output_dir / "quality_model.onnx"
        torch.onnx.export(
            export_model,
            dummy_input,
            output_path,
            input_names=["input"],
            output_names=["overall", "metrics", "errors", "is_standard"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "overall": {0: "batch_size"},
                "metrics": {0: "batch_size"},
                "errors": {0: "batch_size"},
                "is_standard": {0: "batch_size"},
            },
            opset_version=11,
            do_constant_folding=True,
        )
        print(f"  -> ONNX: {output_path}")

    elif format == "torchscript":
        output_path = output_dir / "quality_model.pt"
        traced_model = torch.jit.trace(export_model, dummy_input)
        traced_model.save(output_path)
        print(f"  -> TorchScript: {output_path}")

    # 保存错误类型
    with open(output_dir / "error_types.json", "w") as f:
        json.dump(CONFIG["error_types"], f, indent=2, ensure_ascii=False)

    # 保存模型信息
    model_info = {
        "model_type": "quality_assessment",
        "input_shape": [
            1,
            CONFIG["skeleton"]["target_frames"],
            CONFIG["skeleton"]["num_joints"],
            9,
        ],
        "output_shapes": {
            "overall": [1, 1],
            "metrics": [1, 4],
            "errors": [1, num_errors],
            "is_standard": [1, 1],
        },
    }
    with open(output_dir / "quality_model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)

    return True


def create_deploy_package(output_dir: Path, format: str, checkpoint_dir: Path):
    """创建部署包"""
    print("\n创建部署包...")

    # 复制配置文件
    import shutil

    deploy_config = output_dir / ".." / "config.yaml"
    shutil.copy(CONFIG_PATH, deploy_config)
    print(f"  -> 配置文件: {deploy_config}")

    bundle_summary = checkpoint_dir / "bundle_summary.json"
    if bundle_summary.exists():
        shutil.copy(bundle_summary, output_dir / "bundle_summary.json")
        print(f"  -> Bundle说明: {output_dir / 'bundle_summary.json'}")

    rf_model = checkpoint_dir / "action_model_rf.joblib"
    if rf_model.exists():
        shutil.copy(rf_model, output_dir / "action_model_rf.joblib")
        print(f"  -> RF动作模型: {output_dir / 'action_model_rf.joblib'}")

    rf_summary = checkpoint_dir / "action_model_rf_summary.json"
    if rf_summary.exists():
        shutil.copy(rf_summary, output_dir / "action_model_rf_summary.json")
        print(f"  -> RF模型说明: {output_dir / 'action_model_rf_summary.json'}")

    # 创建推理脚本
    if format == "onnx":
        inference_script = '''#!/usr/bin/env python3
"""
deploy/inference_onnx.py
ONNX推理脚本
"""

import json
import yaml
import numpy as np
import onnxruntime as ort
from pathlib import Path

class OnnxAssessor:
    def __init__(self, model_dir='models'):
        self.model_dir = Path(model_dir)
        
        # 加载配置
        with open(self.model_dir / '..' / 'config.yaml', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 加载ONNX模型
        self.action_session = ort.InferenceSession(
            self.model_dir / 'action_model.onnx',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        # 加载标签映射
        with open(self.model_dir / 'action_labels.json') as f:
            self.id_to_action = json.load(f)
        
        with open(self.model_dir / 'error_types.json') as f:
            self.error_types = json.load(f)
    
    def assess(self, features):
        """推理评估"""
        # 动作识别
        action_output = self.action_session.run(
            None, {'input': features.astype(np.float32)}
        )[0]
        action_id = np.argmax(action_output[0])
        action_type = self.id_to_action[str(action_id)]
        
        # 阶段分割
        phase_session = ort.InferenceSession(
            self.model_dir / f'phase_model_{action_type}.onnx'
        )
        phase_output = phase_session.run(
            None, {'input': features.astype(np.float32)}
        )[0]
        phases = np.argmax(phase_output[0], axis=1)
        
        # 质量评估
        quality_session = ort.InferenceSession(
            self.model_dir / 'quality_model.onnx'
        )
        quality_output = quality_session.run(
            None, {'input': features.astype(np.float32)}
        )
        
        overall_score = quality_output[0][0][0] * 100
        metrics = quality_output[1][0]
        errors = quality_output[2][0]
        
        return {
            'action_type': action_type,
            'confidence': float(action_output[0][action_id]),
            'phases': phases.tolist(),
            'overall_score': float(overall_score),
            'metric_scores': {
                'accuracy': float(metrics[0] * 100),
                'stability': float(metrics[1] * 100),
                'standard': float(metrics[2] * 100),
                'safety': float(metrics[3] * 100)
            },
            'detected_errors': [
                self.error_types[i] for i, p in enumerate(errors) if p > 0.5
            ]
        }

if __name__ == '__main__':
    # 示例用法
    assessor = OnnxAssessor()
    # features = ... # 加载预处理后的特征
    # result = assessor.assess(features)
    # print(result)
'''

        with open(output_dir / ".." / "inference_onnx.py", "w") as f:
            f.write(inference_script)

        print(f"  -> ONNX推理脚本: {output_dir / '..' / 'inference_onnx.py'}")

    # 创建部署依赖
    requirements = """onnxruntime-gpu>=1.16.0
numpy>=1.24.0
pyyaml>=6.0.0
opencv-python>=4.8.0
joblib>=1.3.0
scikit-learn>=1.3.0
"""

    with open(output_dir / ".." / "requirements_deploy.txt", "w") as f:
        f.write(requirements)

    print(f"  -> 部署依赖: {output_dir / '..' / 'requirements_deploy.txt'}")
    print("\n部署包创建完成！")


def main():
    parser = argparse.ArgumentParser(description="导出部署模型")
    parser.add_argument(
        "--format",
        type=str,
        default="onnx",
        choices=["onnx", "torchscript"],
        help="导出格式",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=str(
            DEFAULT_CHECKPOINT_DIR
            if DEFAULT_CHECKPOINT_DIR.exists()
            else CONFIG["paths"]["checkpoints"]
        ),
        help="检查点目录",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=CONFIG["paths"]["deploy"] + "/models",
        help="输出目录",
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)

    print(f"导出格式: {args.format.upper()}")
    print(f"检查点目录: {checkpoint_dir}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 导出模型
    export_action_model(args.format, output_dir, checkpoint_dir)
    export_phase_models(args.format, output_dir, checkpoint_dir)
    export_quality_model(args.format, output_dir, checkpoint_dir)

    # 创建部署包
    create_deploy_package(output_dir, args.format, checkpoint_dir)

    print("\n" + "=" * 60)
    print("模型导出完成！")
    print(f"部署文件位于: {output_dir}/")


if __name__ == "__main__":
    main()
