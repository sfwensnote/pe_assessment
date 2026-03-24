# 从 0 到拿到可用模型（实操清单）

本文档给出一套最短可执行路径，目标是产出可用于系统推理的模型权重文件。

## 目标产物

训练完成后，`checkpoints/` 目录下应至少有：

- `action_model_best.pth`
- `quality_model_best.pth`
- `phase_model_pushup.pth`
- `phase_model_squat.pth`
- `phase_model_situp.pth`
- `phase_model_jump_rope.pth`
- `phase_model_long_jump.pth`
- `phase_model_pullup.pth`

## Step 0：环境准备

```bash
cd /Users/wensifan/Desktop/pe_assessment
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python quick_test.py
```

说明：建议始终在 `.venv` 内运行，避免系统 Python 依赖冲突。

## Step 1：准备原始视频数据

将视频按动作放到：

- `data/raw_videos/pushup/`
- `data/raw_videos/squat/`
- `data/raw_videos/situp/`
- `data/raw_videos/jump_rope/`
- `data/raw_videos/long_jump/`
- `data/raw_videos/pullup/`

建议数据量：

- 起步：每类 50-100 条
- 可用：每类 100+ 条
- 稳定：每类 300+ 条（覆盖角度/光照/人群差异）

## Step 2：提取骨骼关键点

```bash
python 0_preprocess_videos.py
```

产物位置：`data/skeletons/<action>/*.json`

可选：只跑单动作

```bash
python 0_preprocess_videos.py --action pushup
```

## Step 3：自动标注 + 人工复核

先自动标注：

```bash
python 1_auto_annotate.py
```

再人工复核（强烈建议逐动作做）：

```bash
python 2_review_annotations.py --action pushup
```

产物位置：`data/annotations/<action>/*.json`

## Step 4：训练三个模型

建议先用中等参数跑通流程，再拉高 epoch。

```bash
# 动作识别模型
python 3_train_action.py --epochs 60 --batch_size 32

# 阶段模型（按动作训练）
python 4_train_phase.py --epochs 50

# 质量评估模型
python 5_train_quality.py --epochs 40 --batch_size 16
```

## Step 5：检查模型文件是否齐全

```bash
ls checkpoints
```

确保“目标产物”中的文件都存在。

## Step 6：后端加载验证

启动后端：

```bash
uvicorn app.main:app --reload --port 8001
```

新开终端检查：

```bash
curl -s http://127.0.0.1:8001/api/health
curl -s http://127.0.0.1:8001/api/system/overview
```

验收标准：

- `action_model_loaded = true`
- `phase_model_count_loaded >= 1`（理想是 6）
- `quality_model_loaded = true`
- `readiness.level = "full"`

## Step 7：前端联调

```bash
cd web
npm install
npm run dev
```

访问：`http://127.0.0.1:5173`

## 建议的提速策略（可选）

- 预处理阶段改用 `yolov8n-pose.pt` 提速：

```bash
python 0_preprocess_videos.py --model yolov8n-pose.pt
```

- 使用 GPU 机器执行 Step2 和训练（时间会显著缩短）
- 先小数据跑通全链路，再逐步扩容数据量

## 常见问题

- 只有姿态模型但三类训练权重缺失时，系统可运行但会依赖规则兜底，效果有限。
- `data/raw_videos` 若映射到外接硬盘，运行前先确认硬盘已挂载。
- 训练耗时较长，建议用 `nohup` 或 `tmux` 后台运行。
