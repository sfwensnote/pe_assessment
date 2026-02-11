# 体育动作智能评估系统

基于深度学习的体育动作识别与质量评估系统，支持俯卧撑、深蹲、跳绳、跳远、引体向上、仰卧起坐等多种体育运动。

## 功能特性

- **动作识别**: 自动识别6种体育动作类型
- **阶段分割**: 精确划分动作的各个阶段
- **质量评估**: 多维度评估动作质量（0-100分）
- **错误检测**: 自动检测常见动作错误
- **实时评估**: 支持视频实时分析

## 支持的运动类型

| 动作 | 英文标识 | 支持功能 |
|------|---------|---------|
| 俯卧撑 | pushup | 识别/阶段/评估/错误检测 |
| 深蹲 | squat | 识别/阶段/评估/错误检测 |
| 仰卧起坐 | situp | 识别/阶段/评估/错误检测 |
| 跳绳 | jump_rope | 识别/阶段/评估/错误检测 |
| 跳远 | long_jump | 识别/阶段/评估/错误检测 |
| 引体向上 | pullup | 识别/阶段/评估/错误检测 |

## 项目结构

```
pe_assessment/
├── config.yaml                 # 全局配置文件
├── requirements.txt            # 依赖列表
├── README.md                   # 项目说明
│
├── utils/                      # 工具模块
│   ├── __init__.py
│   ├── skeleton.py             # 骨骼数据处理
│   ├── models.py               # 深度学习模型
│   ├── augmentation.py         # 数据增强
│   └── metrics.py              # 评估指标
│
├── data/                       # 数据目录
│   ├── raw_videos/             # 原始视频
│   ├── skeletons/              # 骨骼数据
│   ├── annotations/            # 标注数据
│   └── processed/              # 处理后数据
│
├── checkpoints/                # 模型检查点
│   ├── action_model_best.pth
│   ├── phase_model_*.pth
│   └── quality_model_best.pth
│
├── deploy/                     # 部署包
│   ├── models/                 # 导出模型
│   └── inference_onnx.py       # 部署推理脚本
│
└── 脚本文件
    ├── 0_preprocess_videos.py  # 视频预处理
    ├── 1_auto_annotate.py      # 自动标注
    ├── 2_review_annotations.py # 人工复核（GUI）
    ├── 3_train_action.py       # 训练动作识别模型
    ├── 4_train_phase.py        # 训练阶段分割模型
    ├── 5_train_quality.py      # 训练质量评估模型
    ├── 6_inference.py          # 推理测试
    └── 7_export_model.py       # 模型导出
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

将视频按动作类型放入对应目录：

```bash
mkdir -p data/raw_videos/{pushup,squat,situp,jump_rope,long_jump,pullup}

# 复制视频文件
cp pushup_001.mp4 data/raw_videos/pushup/
cp squat_001.mp4 data/raw_videos/squat/
# ... 其他动作
```

### 3. 运行完整流程

```bash
# 1. 提取骨骼关键点
python 0_preprocess_videos.py

# 2. 自动标注
python 1_auto_annotate.py

# 3. 人工复核（图形界面，可选但推荐）
python 2_review_annotations.py

# 4. 训练模型
python 3_train_action.py        # 动作识别
python 4_train_phase.py         # 阶段分割
python 5_train_quality.py       # 质量评估

# 5. 推理测试
python 6_inference.py --video test.mp4

# 6. 导出部署模型
python 7_export_model.py
```

## 使用说明

### 视频预处理

```bash
# 处理所有动作
python 0_preprocess_videos.py

# 处理指定动作
python 0_preprocess_videos.py --action pushup

# 指定输入输出目录
python 0_preprocess_videos.py --input_dir /path/to/videos --output_dir /path/to/skeletons
```

### 自动标注

```bash
# 标注所有动作
python 1_auto_annotate.py

# 标注指定动作
python 1_auto_annotate.py --action pushup
```

### 训练模型

```bash
# 动作识别模型
python 3_train_action.py --epochs 100 --batch_size 64

# 阶段分割模型（所有动作）
python 4_train_phase.py --epochs 80

# 阶段分割模型（指定动作）
python 4_train_phase.py --action pushup

# 质量评估模型
python 5_train_quality.py --epochs 60 --batch_size 32
```

### 推理测试

```bash
# 自动识别动作
python 6_inference.py --video test.mp4

# 指定动作类型
python 6_inference.py --video test.mp4 --action pushup

# 输出JSON格式
python 6_inference.py --video test.mp4 --format json

# 保存结果和可视化
python 6_inference.py --video test.mp4 --output result.json --visualize output.mp4
```

### 模型导出

```bash
# 导出为ONNX格式
python 7_export_model.py --format onnx

# 导出为TorchScript格式
python 7_export_model.py --format torchscript
```

## 评估指标

### 动作识别
- **准确率**: 动作分类准确率
- **每类准确率**: 各动作类型的识别准确率

### 阶段分割
- **帧级准确率**: 单帧阶段分类准确率
- **边界F1**: 阶段边界检测F1分数
- **编辑距离**: 阶段序列相似度

### 质量评估
- **MAE**: 与人工评分的平均绝对误差
- **相关性**: 与人工评分的皮尔逊相关系数
- **错误检测准确率**: 多标签分类准确率

## 可检测的错误类型

### 俯卧撑
- 塌腰、撅臀、肘外扩、未达深度、耸肩

### 深蹲
- 膝盖内扣、重心前移、未达深度、踮脚尖、圆背

### 仰卧起坐
- 借力拉头、臀部离地、未触膝、借助惯性

### 跳绳
- 全脚掌落地、膝盖过直、节奏不稳、跳跃过高、手臂外展

### 跳远
- 起跳角度过大/过小、未充分摆臂、落地不稳、身体后仰

### 引体向上
- 未过杆、未充分下放、身体摆动、蹬腿借力、耸肩

## 配置说明

编辑 `config.yaml` 可以自定义：

- **动作定义**: 阶段数、标准参数、错误类型
- **训练参数**: 学习率、批次大小、训练轮数
- **评估阈值**: 各等级分数阈值
- **路径配置**: 数据目录、输出目录

## 硬件要求

- **训练**: NVIDIA GPU (推荐 RTX 3060 或更高)
- **推理**: CPU 即可，GPU 加速可选
- **内存**: 至少 8GB RAM
- **存储**: 至少 10GB 可用空间

## 注意事项

1. **数据采集**: 建议固定摄像头位置，保持光线充足
2. **单人场景**: 当前版本针对单人动作设计
3. **视频质量**: 建议分辨率不低于 720p，帧率 30fps
4. **标注质量**: 自动标注后建议人工复核以提高模型性能

## 📄 许可证 (License)

本项目采用 **双重许可 (Dual License)** 模式：

### 开源许可：GPL-3.0

适用于 **个人学习、教育研究、非商业用途**

```
✅ 个人学习/研究：免费
✅ 学校教学：免费
✅ 学术论文引用：免费
✅ 开源项目贡献：免费
❌ 商业使用必须开源或购买商业授权
```

完整 GPL-3.0 协议见 [LICENSE-GPL](LICENSE-GPL) 或访问 https://www.gnu.org/licenses/gpl-3.0.html

### 商业授权

适用于 **企业商用、闭源集成、定制开发**

商业授权权益：
- 闭源使用，无需公开源代码


**授权方式：** 按年订阅或永久授权，根据使用规模和场景定价

**联系方式：**
- 微信：Coder建设
- 邮箱：javpower@163.com

---

##  贡献指南

欢迎提交 Issue 和 Pull Request！

**贡献者协议：** 向本项目提交代码即表示您同意将代码版权归属于本项目，并授权项目维护者在 GPL-3.0 及商业授权下使用您的代码。

---

**Copyright (c) 2026 Coder建设｜javpower**

*让 AI 助力体育教育，科技改变运动未来*
