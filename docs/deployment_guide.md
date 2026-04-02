# 本地部署指南

## 1. 适用范围

本文档用于把项目在本地 Windows 环境部署为可用状态，包括：

- 后端服务
- 前端页面
- 健康检查
- 实时训练链路
- 基础冒烟验证

如果你只是想看训练流程，请转到 [`training_guide.md`](./training_guide.md)。

## 2. 环境要求

建议环境：

- Python 3.10 或 3.11
- Node.js 18 及以上
- npm 9 及以上
- Windows PowerShell

可选但强烈建议：

- 独立虚拟环境 `.venv`
- NVIDIA GPU 用于训练

## 3. 安装依赖

### 3.1 Python 依赖

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

### 3.2 前端依赖

```powershell
cd web
npm install
cd ..
```

### 3.3 快速检查

```powershell
.venv\Scripts\python quick_test.py
```

这个脚本会检查：

- 依赖导入是否正常
- CUDA 是否可用
- 关键目录是否存在
- 模型定义是否能完成一次前向

## 4. 模型文件准备

后端运行不只依赖源码，还依赖权重文件。推荐把权重放到：

```text
checkpoints/mixed_best_bundle/
```

至少需要：

- `action_model_best.pth`
- `action_model_rf.joblib`
- `quality_model_best.pth`
- `phase_model_pushup.pth`
- `phase_model_squat.pth`
- `phase_model_situp.pth`
- `phase_model_jump_rope.pth`
- `phase_model_long_jump.pth`
- `phase_model_pullup.pth`

如果没有 `mixed_best_bundle`，运行时会回退到 `checkpoints/` 根目录。

## 5. 启动后端

### 5.1 直接命令启动

```powershell
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

### 5.2 使用 PowerShell 脚本启动

```powershell
powershell -ExecutionPolicy Bypass -File .\deploy\start_backend.ps1
```

如需管理员口令：

```powershell
powershell -ExecutionPolicy Bypass -File .\deploy\start_backend.ps1 -AdminToken your-token
```

脚本会自动：

- 使用 `.venv\Scripts\python.exe`
- 设置 `YOLO_CONFIG_DIR=.ultralytics`
- 设置 `PYTHONIOENCODING=utf-8`
- 按指定 host/port 启动 `uvicorn`

## 6. 启动前端

### 6.1 直接命令启动

```powershell
cd web
npm run dev
```

### 6.2 使用 PowerShell 脚本启动

```powershell
powershell -ExecutionPolicy Bypass -File .\deploy\start_frontend.ps1
```

如需指定后端地址：

```powershell
powershell -ExecutionPolicy Bypass -File .\deploy\start_frontend.ps1 -ApiBase http://127.0.0.1:8001
```

脚本会自动注入 `VITE_API_BASE`。

## 7. 同时启动前后端

```powershell
powershell -ExecutionPolicy Bypass -File .\deploy\start_all.ps1
```

这个脚本会：

- 新开一个 PowerShell 窗口启动后端
- 新开一个 PowerShell 窗口启动前端
- 输出访问地址

默认地址：

- 前端：`http://127.0.0.1:5173`
- 后端：`http://127.0.0.1:8001`

## 8. 验证部署是否成功

### 8.1 健康检查

```powershell
curl http://127.0.0.1:8001/api/health
```

期望至少看到：

- `ok = true`
- `models.yolo = true`
- `models.action = true`
- `models.phase_model_count >= 1`
- `models.quality = true`

### 8.2 系统总览

```powershell
curl http://127.0.0.1:8001/api/system/overview
```

重点关注：

- `readiness.level`
- `checkpoints.missing_files`
- `features`

### 8.3 后端冒烟测试

```powershell
.venv\Scripts\python smoke_test_backend.py
```

这个脚本会验证：

- 基础路由
- 健康检查
- 实时 session
- 离线上传推理
- 后台任务创建与查询
- 管理员接口

### 8.4 真实服务冒烟测试

```powershell
.venv\Scripts\python smoke_test_live_server.py --base_url http://127.0.0.1:8001
```

### 8.5 前端构建验证

```powershell
cd web
npm run build
cd ..
```

## 9. 部署常见问题

### 9.1 健康检查里模型为 false

通常是以下原因：

- 权重文件不在 `checkpoints/` 约定目录
- 文件名和代码期望不一致
- 权重加载失败
- 当前 Python 环境缺少依赖

### 9.2 Ultralytics 在用户目录写配置失败

当前运行时已把 Ultralytics 配置目录固定到仓库内的 `.ultralytics/`，正常情况下不会再写到用户目录。

### 9.3 Windows 下随机森林推理权限报错

当前运行时已经把 RF 模型的并行度收敛到 `n_jobs=1`，用于避免 `joblib` 在某些 Windows 场景下的权限问题。

### 9.4 前端打开但无法调用后端

检查：

- 后端是否真的监听在 `127.0.0.1:8001`
- `VITE_API_BASE` 是否正确
- 浏览器控制台是否有跨域或网络错误

### 9.5 实时训练能启动但识别不稳定

这通常不是部署问题，而是模型质量或数据质量问题。当前实时链路已经加入：

- 动作短时投票
- 动作锁定
- 角度计数兜底

但阶段显示和自动识别精度仍可能随数据波动。

## 10. 发布与仓库清理建议

建议不要把以下目录推到远端：

- `data/raw_videos/`
- `data/annotations/`
- `data/processed/`
- `checkpoints/`
- `.ultralytics/`
- `deploy/current_release/`

仓库应重点保留：

- 源码
- 配置
- 启动脚本
- 文档
- 轻量级样例和检查脚本
