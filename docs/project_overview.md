# 项目总览

## 1. 项目目标

本项目用于对体育动作视频做结构化评估，目标不是简单分类，而是形成一条完整链路：

1. 从视频中提取人体骨架关键点。
2. 对动作片段做阶段划分。
3. 评估动作完成质量并识别常见错误。
4. 以离线或实时方式把结果返回给前端或调用方。

项目当前适合以下场景：

- 体育课程演示
- 居家训练动作规范反馈
- 视频数据集清洗和弱监督标注
- 校内体测动作分析原型

## 2. 支持动作

系统当前支持 6 类动作：

| 动作键 | 中文名 | 类别 |
| --- | --- | --- |
| `pushup` | 俯卧撑 | 力量 |
| `squat` | 深蹲 | 力量 |
| `situp` | 仰卧起坐 | 力量 |
| `jump_rope` | 跳绳 | 有氧 |
| `long_jump` | 跳远 | 爆发力 |
| `pullup` | 引体向上 | 力量 |

每个动作在 [`config.yaml`](../config.yaml) 中定义了：

- 阶段名称
- 中文阶段名
- 标准动作参数
- 常见错误类型
- 重点关节
- 计数方式

## 3. 系统架构

系统可以拆成 4 层。

### 3.1 数据层

- `data/raw_videos/`：原始视频
- `data/skeletons/`：骨架 JSON
- `data/annotations/`：阶段与质量标注
- `data/processed/`：各阶段运行产物、验证报告、临时缓存

### 3.2 训练层

- `0_preprocess_videos.py`：视频转骨架
- `1_auto_annotate.py`：规则自动标注
- `2_review_annotations.py`：人工复核
- `3_train_action.py`：动作识别模型
- `4_train_phase.py`：阶段模型
- `5_train_quality.py`：质量模型
- `train_action_rf.py`：基于特征工程的随机森林动作分类器
- `build_mixed_best_bundle.py`：组合最佳模型包

### 3.3 推理层

- `6_inference.py`：命令行离线推理
- `app/services/model_runtime.py`：运行时模型装载与单窗口推理
- `app/services/realtime_engine.py`：实时会话逻辑
- `app/services/rep_counter.py`：实时计数

### 3.4 服务与交互层

- `app/main.py`：FastAPI 路由
- `web/`：React 前端
- `deploy/start_backend.ps1`：启动后端
- `deploy/start_frontend.ps1`：启动前端
- `deploy/start_all.ps1`：同时拉起前后端

## 4. 运行时推理链路

### 4.1 离线视频评估

1. 上传视频文件。
2. 使用 YOLO Pose 提取骨架序列。
3. 对骨架做标准化和时序切片。
4. 识别动作类型。
5. 预测阶段。
6. 计算质量分和错误项。
7. 返回 JSON 结果，必要时写入后台任务存储。

### 4.2 实时训练

1. 浏览器创建实时 session。
2. 摄像头逐帧采集图像并转为 `base64`。
3. 前端通过 WebSocket 发送帧。
4. 后端抽取骨架并累积到滑动窗口。
5. 达到窗口长度后执行动作识别、阶段预测和质量评估。
6. 返回动作、阶段、评分、错误提示、建议和计数。

## 5. 主要配置项

配置集中在 [`config.yaml`](../config.yaml)，重点关注：

- `camera`：摄像头位置、分辨率、FPS
- `skeleton.target_frames`：模型输入固定帧数
- `inference.pose_model`：姿态模型路径
- `actions.*`：动作定义与评估规则
- `training.*`：训练超参
- `paths.*`：目录映射
- `ingest.*`：自动采集与质量筛选
- `assessment.*`：评分阈值和权重

## 6. 当前模型与运行策略

默认运行时优先使用 `checkpoints/mixed_best_bundle`。这一策略的目的，是把不同训练轮次里表现更好的动作模型、阶段模型和质量模型拼成一个可运行的组合包。

运行时动作识别目前支持两条路径：

- 神经网络动作分类器
- 随机森林动作分类器

实时评估时会结合规则兜底、短时投票和会话锁定，尽量减少动作标签抖动。

## 7. 当前已知短板

当前系统“能跑”与“足够准”是两件不同的事。已知问题主要集中在：

- 数据分布不均衡
- 自动标注噪声较高
- 阶段模型在实时场景中的稳定性不足
- 自动识别短窗口下仍可能有误判
- 质量评分对数据质量高度敏感

这部分的细化分析见 [`model_improvement_plan.md`](./model_improvement_plan.md)。

## 8. 建议阅读顺序

第一次接手项目，建议按下面顺序阅读：

1. [`../README.md`](../README.md)
2. [`deployment_guide.md`](./deployment_guide.md)
3. [`training_guide.md`](./training_guide.md)
4. [`api_reference.md`](./api_reference.md)
5. [`model_improvement_plan.md`](./model_improvement_plan.md)
