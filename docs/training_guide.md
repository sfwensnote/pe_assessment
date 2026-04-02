# 数据与训练手册

## 1. 目标

本文档描述从原始视频到可部署模型包的完整训练流程，适合以下场景：

- 从零开始搭训练数据
- 重新训练动作、阶段、质量模型
- 生成更稳定的组合模型包
- 对当前模型精度做系统性改进

## 2. 数据目录约定

原始数据与中间产物默认使用以下目录：

```text
data/
├── raw_videos/
├── skeletons/
├── annotations/
└── processed/
```

6 个动作建议各自使用独立子目录：

- `data/raw_videos/pushup/`
- `data/raw_videos/squat/`
- `data/raw_videos/situp/`
- `data/raw_videos/jump_rope/`
- `data/raw_videos/long_jump/`
- `data/raw_videos/pullup/`

## 3. Step 1：视频预处理

使用姿态模型把视频提取为每帧骨架关键点：

```powershell
python 0_preprocess_videos.py
```

只处理单个动作：

```powershell
python 0_preprocess_videos.py --action pushup
```

主要输出：

- `data/skeletons/<action>/*.json`
- `data/processed/preprocess/` 下的日志或统计文件

预处理阶段重点关注：

- 是否能稳定检测到单人
- 视频时长是否过短
- 关键点丢失比例是否过高

## 4. Step 2：自动标注

规则脚本会根据骨架和动作定义，生成阶段与质量标签：

```powershell
python 1_auto_annotate.py
```

如需只处理单动作：

```powershell
python 1_auto_annotate.py --action squat
```

主要输出：

- `data/annotations/<action>/*.json`
- `data/processed/annotate/` 下的统计或报告

自动标注只能作为起点，不能直接等价于高质量真值。

## 5. Step 3：人工复核

训练效果的上限，很大程度取决于这一步。

```powershell
python 2_review_annotations.py --action pushup
```

建议至少人工复核：

- 类间边界容易混淆的样本
- 起止帧不清晰的样本
- 摄像头角度偏、遮挡多的样本
- 规则标注明显不合理的样本

如果数据规模较大，可配合仓库中的辅助脚本：

- `prepare_review_queue.py`
- `auto_review_queue.py`
- `monitor_review_progress.py`
- `generate_step2_report.py`

## 6. Step 4：训练动作识别模型

```powershell
python 3_train_action.py --epochs 100 --batch_size 64
```

如果仓库保留了随机森林训练脚本，建议同步训练一个基于手工特征的动作分类器：

```powershell
python train_action_rf.py
```

动作识别建议关注：

- 类别均衡
- 视角多样性
- 误标样本比例
- 混淆矩阵

## 7. Step 5：训练阶段模型

```powershell
python 4_train_phase.py --epochs 80
```

按单动作训练：

```powershell
python 4_train_phase.py --action pushup
```

阶段模型对标签一致性高度敏感。推荐重点检查：

- 阶段定义是否统一
- 邻近阶段边界是否一致
- 时间重采样后标签是否错位

## 8. Step 6：训练质量模型

```powershell
python 5_train_quality.py --epochs 60 --batch_size 32
```

质量模型通常同时输出：

- 综合得分
- 错误标签
- 动作是否标准

这一块最容易受以下因素影响：

- 标注口径不一致
- 错误标签极度不平衡
- 训练集和真实使用场景差异过大

## 9. Step 7：构建更好的训练清单与模型包

如果当前仓库包含下列脚本，建议一并使用：

```powershell
python build_training_manifests.py
python build_high_quality_manifests.py
python build_mixed_best_bundle.py
python validate_high_quality_models.py
```

用途分别是：

- 构建标准训练清单
- 构建高质量子集训练清单
- 组装混合最优模型包
- 验证高质量重训模型

## 10. Step 8：离线验证

```powershell
python 6_inference.py --video test.mp4 --format json
```

建议至少覆盖：

- 每类动作的代表样本
- 易混淆样本
- 不同视角样本
- 噪声样本

如果你已经接通了后端服务，可额外跑：

```powershell
python smoke_test_backend.py
python smoke_test_live_server.py --base_url http://127.0.0.1:8001
```

## 11. Step 9：模型导出

```powershell
python 7_export_model.py --format onnx
```

或：

```powershell
python 7_export_model.py --format torchscript
```

导出完成后，应检查：

- 配置文件是否同步导出
- 模型文件命名是否与运行时一致
- 推理脚本是否仍能正确加载

## 12. 训练结果验收建议

不要只看单一准确率。建议同时评估：

- 动作分类准确率
- 动作分类混淆矩阵
- 阶段逐帧准确率或 F1
- 阶段序列编辑距离
- 质量分 MAE / RMSE
- 错误标签 Precision / Recall
- 离线端到端样本命中率
- 实时短窗口稳定性

## 13. 当前模型质量问题的现实判断

如果你已经观察到“能识别，但成功率不高”，优先排查下面几项：

1. 训练数据是否主要来自单一来源、单一视角。
2. 自动标注是否未经充分人工复核。
3. 动作类别之间是否存在明显混淆样本。
4. 阶段标签定义是否在不同动作之间不一致。
5. 实时窗口长度是否过短。

更详细的优化路线见 [`model_improvement_plan.md`](./model_improvement_plan.md)。
