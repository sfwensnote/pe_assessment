# 体育动作智能评估系统

`pe_assessment` 是一个面向体育动作教学、训练评估和视频分析的完整项目，覆盖数据采集、骨架提取、自动标注、人工复核、模型训练、离线推理、实时评估和前后端部署。当前支持 6 类动作：

- `pushup` 俯卧撑
- `squat` 深蹲
- `situp` 仰卧起坐
- `jump_rope` 跳绳
- `long_jump` 跳远
- `pullup` 引体向上

项目后端使用 `FastAPI + PyTorch`，前端使用 `React + TypeScript + Vite`，全局运行配置集中在 [`config.yaml`](./config.yaml)。

## 1. 当前仓库定位

本仓库保存的是源码、脚本、配置与文档，不包含大体积训练数据、原始视频、运行缓存和模型权重二进制。

如果你希望本地完整跑通，需要自行准备或放置：

- `checkpoints/` 下的模型权重
- `data/raw_videos/` 下的原始视频
- `data/annotations/` 下的标注文件

默认运行时会优先尝试加载：

- `checkpoints/mixed_best_bundle/action_model_best.pth`
- `checkpoints/mixed_best_bundle/action_model_rf.joblib`
- `checkpoints/mixed_best_bundle/phase_model_*.pth`
- `checkpoints/mixed_best_bundle/quality_model_best.pth`

如果 `mixed_best_bundle` 不存在，则回退到 `checkpoints/` 根目录。

## 2. 核心能力

- 动作识别：支持自动识别 6 类动作
- 阶段分割：按动作输出阶段标签
- 质量评估：输出动作质量分、错误项与纠正建议
- 离线推理：上传视频后返回整体评估结果
- 实时训练：浏览器摄像头逐帧发送，后端通过 WebSocket 返回实时反馈
- 运维观测：提供健康检查、系统总览、管理员概览、后台任务列表和实时报告
- 数据管线：内置预处理、自动标注、复核、训练、打包和验证脚本

## 3. 项目结构

```text
pe_assessment/
├── app/                          # FastAPI 服务与实时推理逻辑
├── checkpoints/                  # 本地模型权重目录（默认不入库）
├── data/                         # 原始数据、标注、处理产物（默认不入库）
├── deploy/                       # 启动脚本与部署相关文件
├── docs/                         # 中文专题文档
├── utils/                        # 骨架处理、模型、指标、增强等核心工具
├── web/                          # React 前端
├── 0_preprocess_videos.py        # 视频 -> 骨架关键点
├── 1_auto_annotate.py            # 自动标注
├── 2_review_annotations.py       # 人工复核
├── 3_train_action.py             # 动作识别训练
├── 4_train_phase.py              # 阶段模型训练
├── 5_train_quality.py            # 质量模型训练
├── 6_inference.py                # 离线推理
├── 7_export_model.py             # 模型导出
├── 8_ingest_pipeline.py          # 自动采集与入库
├── 8_ingest_monitor.py           # 入库监控
├── 9_tag_and_cleanup_videos.py   # 标签与清理
├── config.yaml                   # 全局配置
└── quick_test.py                 # 环境快速检查
```

## 4. 快速开始

### 4.1 准备环境

Windows：

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python quick_test.py
```

Linux / macOS：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python quick_test.py
```

### 4.2 启动后端

开发方式：

```powershell
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

或直接使用脚本：

```powershell
powershell -ExecutionPolicy Bypass -File .\deploy\start_backend.ps1
```

### 4.3 启动前端

```powershell
cd web
npm install
npm run dev
```

或使用项目根目录的一键脚本：

```powershell
powershell -ExecutionPolicy Bypass -File .\deploy\start_frontend.ps1
```

### 4.4 同时启动前后端

```powershell
powershell -ExecutionPolicy Bypass -File .\deploy\start_all.ps1
```

默认地址：

- 前端：`http://127.0.0.1:5173`
- 后端：`http://127.0.0.1:8001`

### 4.5 健康检查

```powershell
curl http://127.0.0.1:8001/api/health
curl http://127.0.0.1:8001/api/system/overview
```

## 5. 训练与推理主流程

### 5.1 数据处理

```powershell
python 0_preprocess_videos.py
python 1_auto_annotate.py
python 2_review_annotations.py --action pushup
```

### 5.2 模型训练

```powershell
python 3_train_action.py --epochs 100 --batch_size 64
python 4_train_phase.py --epochs 80
python 5_train_quality.py --epochs 60 --batch_size 32
```

### 5.3 模型验证与打包

```powershell
python 6_inference.py --video test.mp4 --format json
python 7_export_model.py --format onnx
```

如果仓库中已经有额外的清单构建和模型组合脚本，也可以继续使用：

- `build_training_manifests.py`
- `build_high_quality_manifests.py`
- `train_action_rf.py`
- `build_mixed_best_bundle.py`
- `validate_high_quality_models.py`

## 6. 当前部署状态说明

当前系统已经具备完整的本地部署能力，后端提供：

- 健康检查：`/api/health`
- 系统总览：`/api/system/overview`
- 离线视频评估：`/api/inference/video`
- 后台视频任务：`/api/inference/video/tasks`
- 实时训练会话：`/api/realtime/session/start`
- 实时 WebSocket：`/ws/realtime/{session_id}`
- 管理员接口：`/api/admin/*`

实时链路中已经加入：

- 动作识别短时投票与会话锁定
- 基于角度阈值的实时计数兜底
- 模型加载状态与部署健康检查

当前仍需注意：

- 阶段显示在实时场景里仍然不够稳定
- 自动识别与评分已经接上训练模型，但模型精度仍有继续提升空间
- 数据和模型质量是最终效果的主要瓶颈

模型质量提升建议见 [`docs/model_improvement_plan.md`](./docs/model_improvement_plan.md)。

## 7. 文档索引

- [`docs/project_overview.md`](./docs/project_overview.md)：项目总览、架构和模块说明
- [`docs/deployment_guide.md`](./docs/deployment_guide.md)：本地部署、启动方式、健康检查与常见问题
- [`docs/training_guide.md`](./docs/training_guide.md)：数据、标注、训练、验证、导出全流程
- [`docs/api_reference.md`](./docs/api_reference.md)：后端接口、WebSocket 协议和联调说明
- [`docs/model_improvement_plan.md`](./docs/model_improvement_plan.md)：模型质量问题分析与提升路线
- [`FROM_ZERO_TO_MODELS.md`](./FROM_ZERO_TO_MODELS.md)：从零到模型产出的简版实操清单
- [`CURRENT_RELEASE_NOTES.md`](./CURRENT_RELEASE_NOTES.md)：当前版本说明与运行建议

## 8. 仓库提交约定

以下内容默认不提交到 Git：

- `checkpoints/` 下的模型权重
- `data/raw_videos/` 下的原始视频
- `data/annotations/` 下的本地标注
- `data/processed/` 下的运行产物、报告和缓存
- `.ultralytics/`、`web/dist/`、`web/*.tsbuildinfo`

这样做的原因是：

- 避免仓库被大文件和生成物污染
- 降低推送失败和克隆缓慢的风险
- 让仓库聚焦源码、配置和文档

## 9. 许可证

项目中包含 `LICENSE` 与 `LICENSE-GPL`，使用前请结合具体代码与依赖链路确认适用范围。
