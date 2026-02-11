#!/bin/bash
# run_all.sh
# 一键运行完整训练和评估流程

set -e  # 遇到错误立即退出

echo "=========================================="
echo "体育动作智能评估系统 - 完整流程"
echo "=========================================="

# 检查依赖
echo -e "\n[1/8] 检查依赖..."
python -c "import torch, cv2, yaml, numpy" 2>/dev/null || {
    echo "错误: 缺少依赖，请先运行: pip install -r requirements.txt"
    exit 1
}
echo "依赖检查通过"

# 1. 提取骨骼
echo -e "\n[2/8] 提取骨骼关键点..."
python 0_preprocess_videos.py

# 2. 自动标注
echo -e "\n[3/8] 自动标注数据..."
python 1_auto_annotate.py

# 3. 人工复核（可选，跳过）
echo -e "\n[4/8] 人工复核（按回车跳过，输入 'review' 进行复核）"
read -t 5 -p "是否进行人工复核? (review/跳过): " choice || choice=""
if [ "$choice" = "review" ]; then
    python 2_review_annotations.py
fi

# 4. 训练动作识别模型
echo -e "\n[5/8] 训练动作识别模型..."
python 3_train_action.py --epochs 50

# 5. 训练阶段分割模型
echo -e "\n[6/8] 训练阶段分割模型..."
python 4_train_phase.py --epochs 40

# 6. 训练质量评估模型
echo -e "\n[7/8] 训练质量评估模型..."
python 5_train_quality.py --epochs 30

# 7. 导出模型
echo -e "\n[8/8] 导出部署模型..."
python 7_export_model.py --format onnx

echo -e "\n=========================================="
echo "全部完成！"
echo "模型文件位于: checkpoints/"
echo "部署文件位于: deploy/"
echo "=========================================="

# 提示如何进行推理
echo -e "\n使用示例:"
echo "  python 6_inference.py --video your_video.mp4"
