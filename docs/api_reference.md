# API 与联调说明

## 1. 后端基础信息

默认后端地址：

- `http://127.0.0.1:8001`

主服务文件：

- `app/main.py`

服务类型：

- HTTP REST API
- WebSocket 实时流

## 2. 公开接口

### 2.1 `GET /`

用途：

- 服务根路由探活

返回：

- `message`

### 2.2 `GET /api/health`

用途：

- 检查依赖和模型加载状态

关键返回字段：

- `ok`
- `device`
- `models.yolo`
- `models.action`
- `models.phase_model_count`
- `models.quality`

### 2.3 `GET /api/system/overview`

用途：

- 给前端或部署人员查看系统可用能力和模型就绪程度

关键返回字段：

- `readiness.level`
- `models`
- `checkpoints.missing_files`
- `features`

### 2.4 `GET /api/actions`

用途：

- 返回配置里支持的动作列表

关键返回字段：

- `actions[].id`
- `actions[].name`
- `actions[].num_phases`

## 3. 离线视频评估接口

### 3.1 `POST /api/inference/video`

用途：

- 上传单个视频并同步返回评估结果

表单字段：

- `file`
- `action_type`，可选

典型返回字段：

- `ok`
- `action_type`
- `phase`
- `overall_score`
- `errors`
- `tips`

### 3.2 `POST /api/inference/video/tasks`

用途：

- 创建异步视频评估任务

表单字段：

- `file`
- `action_type`，可选

返回：

- `task`

### 3.3 `GET /api/inference/video/tasks`

用途：

- 列出所有后台视频任务

### 3.4 `GET /api/inference/video/tasks/{task_id}`

用途：

- 查看单个任务详情

### 3.5 `DELETE /api/inference/video/tasks/{task_id}`

用途：

- 删除单个已完成或失败的历史任务

### 3.6 `DELETE /api/inference/video/history`

用途：

- 清空已完成/失败任务历史

## 4. 实时训练接口

### 4.1 `POST /api/realtime/session/start`

用途：

- 创建一个实时训练会话

请求 JSON：

```json
{
  "action_type": "squat",
  "target_reps": 20,
  "window_size": 60,
  "infer_interval": 3
}
```

字段说明：

- `action_type`：可选，不传则自动识别
- `target_reps`：目标次数
- `window_size`：滑动窗口长度
- `infer_interval`：推理间隔

返回：

```json
{
  "session_id": "xxxx",
  "ws_url": "/ws/realtime/xxxx"
}
```

### 4.2 `POST /api/realtime/session/{session_id}/stop`

用途：

- 停止会话并返回报告

### 4.3 `GET /api/reports/{session_id}`

用途：

- 获取已保存的实时训练报告

## 5. WebSocket 协议

### 5.1 连接地址

```text
ws://127.0.0.1:8001/ws/realtime/{session_id}
```

### 5.2 前端发送示例

心跳：

```json
{
  "type": "ping"
}
```

图像帧：

```json
{
  "image_base64": "data:image/jpeg;base64,..."
}
```

### 5.3 后端返回示例

热身阶段：

```json
{
  "status": "warming_up",
  "message": "采集中 12/60 帧...",
  "rep_count": 0,
  "target_reps": 20,
  "completion_rate": 0.0,
  "warmup_progress": 0.2
}
```

实时推理结果：

```json
{
  "status": "ok",
  "timestamp": 1710000000000,
  "action_type": "squat",
  "confidence": 0.91,
  "action_source": "session_lock",
  "phase": 4,
  "phase_name": "lockout",
  "overall_score": 78.5,
  "is_standard": true,
  "errors": [],
  "tips": [],
  "rep_count": 3,
  "target_reps": 20,
  "completion_rate": 0.15,
  "cadence": 12.4,
  "warmup_progress": 1.0
}
```

### 5.4 当前实时链路特性

实时链路中已经加入：

- 热身窗口积累
- 短历史平滑
- 动作投票
- 动作锁定
- 基于角度信号的计数兜底

当前已知不足：

- 阶段名显示依然不够稳
- 自动识别精度受短窗口影响较大

## 6. 管理员接口

如果设置了 `ADMIN_TOKEN`，管理员接口需要带认证头。

可用接口：

- `GET /api/admin/overview`
- `GET /api/admin/video_tasks`
- `GET /api/admin/realtime_reports`
- `GET /api/admin/ingest/overview`

认证方式：

- `Authorization: Bearer <token>`
- 或 `X-Admin-Token: <token>`

## 7. 监控接口

### `GET /api/monitor/live`

用途：

- 获取轻量级实时监控快照

适合：

- 悬浮监控面板
- 后台轮询
- 运维态状态展示

## 8. 前端联调要点

前端位于 `web/`，默认使用 `VITE_API_BASE` 作为后端地址。

开发环境：

```powershell
cd web
npm install
npm run dev
```

生产构建：

```powershell
cd web
npm run build
```

如果使用脚本启动：

```powershell
powershell -ExecutionPolicy Bypass -File .\deploy\start_frontend.ps1 -ApiBase http://127.0.0.1:8001
```

## 9. 联调建议

建议按以下顺序联调：

1. `GET /api/health`
2. `GET /api/system/overview`
3. `POST /api/realtime/session/start`
4. WebSocket 连通性与热身消息
5. 离线视频上传
6. 后台任务接口
7. 管理员接口

## 10. 自动化验证

可直接运行：

```powershell
.venv\Scripts\python smoke_test_backend.py
.venv\Scripts\python smoke_test_live_server.py --base_url http://127.0.0.1:8001
```

它们覆盖了大部分核心接口和关键链路。
