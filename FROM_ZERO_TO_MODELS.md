# 从零到拿到可用模型

这是一份最短可执行版本的实操清单，适合第一次把项目跑通。

更完整的说明请配合阅读：

- [`README.md`](./README.md)
- [`docs/training_guide.md`](./docs/training_guide.md)
- [`docs/deployment_guide.md`](./docs/deployment_guide.md)

## 1. 环境准备

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python quick_test.py
```

## 2. 准备原始视频

把视频分别放到：

- `data/raw_videos/pushup/`
- `data/raw_videos/squat/`
- `data/raw_videos/situp/`
- `data/raw_videos/jump_rope/`
- `data/raw_videos/long_jump/`
- `data/raw_videos/pullup/`

建议起步量：

- 每类 50 到 100 条：能跑通流程
- 每类 100 到 300 条：能得到初步可用模型
- 每类 300 条以上：更适合做稳定优化

## 3. 提取骨架

```powershell
python 0_preprocess_videos.py
```

产物默认写到：

- `data/skeletons/<action>/*.json`

## 4. 自动标注

```powershell
python 1_auto_annotate.py
```

产物默认写到：

- `data/annotations/<action>/*.json`

## 5. 人工复核

```powershell
python 2_review_annotations.py --action pushup
```

至少建议把每个动作都抽样复核一批，再进入训练。

## 6. 训练动作模型

```powershell
python 3_train_action.py --epochs 100 --batch_size 64
```

如需额外训练 RF 分类器：

```powershell
python train_action_rf.py
```

## 7. 训练阶段模型

```powershell
python 4_train_phase.py --epochs 80
```

按单动作训练：

```powershell
python 4_train_phase.py --action squat
```

## 8. 训练质量模型

```powershell
python 5_train_quality.py --epochs 60 --batch_size 32
```

## 9. 验证模型是否可用

```powershell
python 6_inference.py --video test.mp4 --format json
```

如果要直接验证后端：

```powershell
python smoke_test_backend.py
```

## 10. 导出与部署

```powershell
python 7_export_model.py --format onnx
```

然后启动后端与前端：

```powershell
powershell -ExecutionPolicy Bypass -File .\deploy\start_all.ps1
```

## 11. 最终检查

如果以下条件满足，说明链路已经打通：

- `checkpoints/` 下有动作、阶段、质量模型
- `GET /api/health` 返回 `ok = true`
- `GET /api/system/overview` 显示 `readiness.level = "full"`
- 前端能打开并正常连接后端

如果模型精度仍然不理想，请直接看：

- [`docs/model_improvement_plan.md`](./docs/model_improvement_plan.md)
