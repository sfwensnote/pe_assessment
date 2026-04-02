# 当前版本说明

日期：2026-04-02

## 1. 默认模型包

当前运行时优先加载：

- `checkpoints/mixed_best_bundle/action_model_best.pth`
- `checkpoints/mixed_best_bundle/action_model_rf.joblib`
- `checkpoints/mixed_best_bundle/phase_model_*.pth`
- `checkpoints/mixed_best_bundle/quality_model_best.pth`

如果该目录不存在，则回退到 `checkpoints/` 根目录。

## 2. 当前可用能力

后端当前提供：

- 健康检查
- 系统总览
- 离线视频评估
- 异步视频任务
- 实时训练会话
- 管理员观测接口

前端当前提供：

- 用户训练界面
- 实时训练反馈
- 视频上传评估
- 管理员监控入口

## 3. 本轮已完成的运行修复

### 3.1 运行时与部署

- 将 Ultralytics 配置目录固定到仓库内 `.ultralytics/`
- 增加 `deploy/start_backend.ps1`
- 增加 `deploy/start_frontend.ps1`
- 增加 `deploy/start_all.ps1`

### 3.2 实时识别与计数

- 修复阶段预测 `argmax` 维度错误
- 增加实时动作短时投票
- 增加实时会话动作锁定
- 加入基于角度信号的实时计数兜底
- 降低规则与模型冲突时的误判风险

### 3.3 Windows 兼容性

- RF 模型加载后强制 `n_jobs=1`
- 修复部分 `joblib` 并行权限问题

## 4. 建议的本地启动方式

### 后端

```powershell
powershell -ExecutionPolicy Bypass -File .\deploy\start_backend.ps1
```

### 前端

```powershell
powershell -ExecutionPolicy Bypass -File .\deploy\start_frontend.ps1
```

### 同时启动

```powershell
powershell -ExecutionPolicy Bypass -File .\deploy\start_all.ps1
```

## 5. 建议的验证方式

### 健康检查

```powershell
curl http://127.0.0.1:8001/api/health
curl http://127.0.0.1:8001/api/system/overview
```

### 冒烟测试

```powershell
python smoke_test_backend.py
python smoke_test_live_server.py --base_url http://127.0.0.1:8001
```

### 前端构建

```powershell
cd web
npm run build
cd ..
```

## 6. 当前已知限制

- 模型已经接通训练好的权重，但总体精度仍需继续提升
- 阶段显示在实时场景里仍不够稳定
- 自动识别在短窗口下依然可能波动
- 当前更适合作为可运行原型和持续迭代基础，而不是最终稳定产品

## 7. 下一步优先事项

建议优先做下面几件事：

1. 清洗并复核高质量训练数据。
2. 固定一套可信验证集。
3. 重新评估动作、阶段、质量模型的单项指标。
4. 针对实时短窗口单独做回放测试和优化。

详细建议见 [`docs/model_improvement_plan.md`](./docs/model_improvement_plan.md)。
