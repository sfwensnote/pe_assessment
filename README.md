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

## 实时摄像头教学（单人版）

项目已提供单人实时训练的前后端骨架，适合本地课堂演示和学生自练。

### 1) 启动后端服务

```bash
python -m pip install -r requirements.txt
uvicorn app.main:app --reload --port 8001
```

可选：为管理员接口启用口令鉴权（建议）

```bash
export ADMIN_TOKEN="your-admin-token"
uvicorn app.main:app --reload --port 8001
```

### 2) 启动 React 前端

```bash
cd web
npm install
npm run dev
```

前端默认访问 `http://127.0.0.1:5173`。

说明：当前前端默认直连 `http://127.0.0.1:8001`，可通过 `VITE_API_BASE` 自定义后端地址。

### 3) 实时能力说明

- 浏览器调用摄像头并实时发送帧到后端
- 后端返回动作阶段、质量分、错误提示、纠正建议
- 支持镜像预览（更符合前置摄像头使用习惯）
- 内置简单计次（单人）
- 会话结束可生成训练报告 JSON
- 支持上传本地视频并做一次性评估
- 支持后台任务化视频评估（进度、任务列表、历史）
- 提供管理员观测接口（模型状态、实时会话、任务统计）
- 上传采用分块写入（默认单文件上限 300MB），超限会返回 413
- 上传接口按 IP 频率限制（默认 60 秒内最多 6 次），超限返回 429
- 上传视频结果支持动作计数（estimated_reps）

### 4) 模型部署说明

- 若只安装依赖不放训练权重，系统仍可运行，但会更多依赖规则评估，准确率有限。
- 若要完整能力，请将训练产物放在 `checkpoints/` 下：
  - `action_model_best.pth`
  - `phase_model_<action>.pth`（可按动作逐个放）
  - `quality_model_best.pth`
- `yolov8x-pose.pt` 若本地不存在，Ultralytics 会在首次运行时下载。
- 自动识别策略：优先动作模型；当模型缺失或置信度偏低时，系统会启用规则兜底识别。

### 5) 主要接口

- `GET /api/health`
- `GET /api/actions`
- `POST /api/realtime/session/start`
- `POST /api/realtime/session/{session_id}/stop`
- `GET /api/reports/{session_id}`
- `POST /api/inference/video`
- `POST /api/inference/video/tasks`（创建后台评估任务）
- `GET /api/inference/video/tasks`（任务列表 + 历史）
- `GET /api/inference/video/tasks/{task_id}`（任务详情）
- `DELETE /api/inference/video/tasks/{task_id}`（删除单条历史）
- `DELETE /api/inference/video/history`（清空已完成/失败历史）
- `GET /api/admin/overview`（管理员总览）
- `GET /api/admin/video_tasks`（管理员任务监控）
- `GET /api/admin/realtime_reports`（管理员会话报告）
- `WS /ws/realtime/{session_id}`

管理员接口在配置 `ADMIN_TOKEN` 后需要请求头：`X-Admin-Token: <token>`。

## 自动采集与实时监控（Pexels）

系统已提供“自动载入 -> 部署 -> 运行编排”的脚本，默认流程如下：

1. 采集 6 类动作视频到 `data/raw_videos/<action>/`
2. 生成部署与巡检脚本到 `deploy/ingest/`
3. 对新增动作自动执行预处理与自动标注

搜索关键词固定为中文：`俯卧撑 / 深蹲 / 仰卧起坐 / 跳绳 / 跳远 / 引体向上`。
并默认使用 Pexels `locale=zh-CN` 搜索，避免中文关键词命中偏离。
为降低“标题不相关”问题，采集脚本会先做链接关键词匹配，再做姿态快速质检。

### 1) 配置 API Key

```bash
export PEXELS_API_KEY="your_api_key"
```

### 2) 执行自动采集编排

```bash
# 全动作自动编排
python 8_ingest_pipeline.py

# 指定动作（示例：俯卧撑 + 深蹲）
python 8_ingest_pipeline.py --actions pushup,squat

# 仅采集，不执行预处理/标注
python 8_ingest_pipeline.py --no_pipeline

# 若环境证书链异常，可临时加上（不建议长期使用）
python 8_ingest_pipeline.py --insecure_ssl

# 若你只想调试下载流程（跳过预处理依赖检查）
python 8_ingest_pipeline.py --no_pipeline --skip_env_check

# 采集后自动打标签+清理垃圾视频
python 8_ingest_pipeline.py --auto_tag_cleanup
```

建议优先在虚拟环境里执行，避免系统 Python 依赖污染：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 3) 实时监控进展

```bash
# 单次查看
python 8_ingest_monitor.py

# 持续刷新看板
python 8_ingest_monitor.py --watch

# 一条命令同时启动编排 + 实时看板
bash deploy/ingest/run_with_live_monitor.sh
```

监控数据来源于：`data/processed/ingest/pipeline_state.json`

每次运行都会额外落地一份历史快照：

- `data/processed/ingest/pipeline_state_<run_id>.json`
- `data/processed/ingest/quality_rejected.jsonl`（被质检淘汰的视频记录）

### 4) 自动部署辅助脚本（由编排脚本自动生成）

- `deploy/ingest/run_ingest_pipeline.sh`
- `deploy/ingest/watch_ingest_progress.sh`
- `deploy/ingest/run_with_live_monitor.sh`
- `deploy/ingest/cron.example`

### 5) 视频打标签与垃圾清理

```bash
# 仅打标签
python 9_tag_and_cleanup_videos.py

# 打标签并清理
python 9_tag_and_cleanup_videos.py --cleanup

# 只预览将被清理的文件
python 9_tag_and_cleanup_videos.py --cleanup --dry_run
```

输出文件：

- `data/processed/ingest/video_tags.jsonl`
- `data/processed/ingest/cleanup_log.jsonl`
- `data/processed/ingest/quality_rejected.jsonl`

注意：如果你把 `data/raw_videos` 映射到外接硬盘，运行前请先确认硬盘已挂载；
否则采集和清理脚本会直接报错并停止。

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
