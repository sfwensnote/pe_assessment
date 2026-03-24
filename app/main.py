"""FastAPI app for single-person realtime sports coaching."""

from __future__ import annotations

import base64
import json
import os
import time
import uuid
from collections import Counter, defaultdict, deque
from pathlib import Path
from threading import Lock
from typing import Optional

import numpy as np
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.services.model_runtime import ModelRuntime
from app.services.realtime_engine import RealtimeSession
from app.services.video_task_manager import VideoInferenceTaskManager
from app.storage.session_store import SessionStore


class StartSessionRequest(BaseModel):
    """Request body for creating a realtime session."""

    action_type: Optional[str] = Field(default=None, description="指定动作，不传则自动识别")
    target_reps: int = Field(default=20, ge=1, le=500)
    window_size: int = Field(default=60, ge=20, le=120)
    infer_interval: int = Field(default=3, ge=1, le=15)


class StopSessionResponse(BaseModel):
    """Response body for stopping a session."""

    session_id: str
    report_path: Optional[str] = None
    report: dict


app = FastAPI(title="体育动作实时教学服务", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = SessionStore()
runtime: Optional[ModelRuntime] = None
task_manager: Optional[VideoInferenceTaskManager] = None


def get_runtime() -> ModelRuntime:
    """Lazy singleton runtime."""
    global runtime
    if runtime is None:
        runtime = ModelRuntime()
    return runtime


def get_task_manager() -> VideoInferenceTaskManager:
    """Lazy singleton background task manager."""
    global task_manager
    if task_manager is None:
        rt = get_runtime()
        upload_dir = rt.project_root / rt.config["paths"]["processed"] / "temp_uploads"
        store_path = (
            rt.project_root
            / rt.config["paths"]["processed"]
            / "realtime_reports"
            / "video_tasks.json"
        )
        task_manager = VideoInferenceTaskManager(
            runtime_provider=get_runtime,
            upload_dir=upload_dir,
            store_path=store_path,
        )
    return task_manager


def _latest_realtime_reports(limit: int = 20) -> list[dict]:
    """Read latest persisted realtime session reports from disk."""
    rt = get_runtime()
    report_dir = rt.project_root / rt.config["paths"]["processed"] / "realtime_reports"
    if not report_dir.exists():
        return []

    rows = []
    for path in sorted(
        report_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
    ):
        if path.name == "video_tasks.json":
            continue
        try:
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
            payload["_report_file"] = path.name
            rows.append(payload)
        except Exception:
            continue
        if len(rows) >= limit:
            break

    return rows


def _build_system_overview_payload() -> dict:
    """Build user-facing system capability and readiness summary."""
    rt = get_runtime()
    status = rt.status()
    config = rt.config
    checkpoint_dir = rt.project_root / config["paths"]["checkpoints"]

    action_ckpt = checkpoint_dir / "action_model_best.pth"
    quality_ckpt = checkpoint_dir / "quality_model_best.pth"
    phase_files = {
        action: checkpoint_dir / f"phase_model_{action}.pth"
        for action in config["actions"].keys()
    }

    missing_phase = [
        action for action, path in phase_files.items() if not path.exists()
    ]
    missing_files = []
    if not action_ckpt.exists():
        missing_files.append(action_ckpt.name)
    if not quality_ckpt.exists():
        missing_files.append(quality_ckpt.name)
    missing_files.extend([f"phase_model_{action}.pth" for action in missing_phase])

    feature_flags = {
        "realtime_camera_assessment": status.yolo_ready,
        "video_upload_assessment": status.yolo_ready,
        "background_video_tasks": True,
        "admin_observability": True,
        "estimated_repetition_count": True,
        "auto_action_recognition_model": status.action_model_ready,
        "phase_segmentation_model": status.phase_models_ready > 0,
        "quality_scoring_model": status.quality_model_ready,
    }

    readiness_level = (
        "full"
        if all(
            [
                status.yolo_ready,
                status.action_model_ready,
                status.phase_models_ready > 0,
                status.quality_model_ready,
            ]
        )
        else "partial"
    )

    setup_steps = [
        "准备并清洗数据：raw_videos -> preprocess -> annotate -> review",
        "训练动作、阶段、质量模型并输出 checkpoints/*.pth",
        "重启后端服务并检查 /api/health 与 /api/system/overview",
        "先用 /upload 跑离线验证，再用 /realtime 做在线训练",
    ]

    return {
        "project": "体育动作智能评估系统",
        "device": status.device,
        "readiness": {
            "level": readiness_level,
            "summary": "完整模型链路已就绪" if readiness_level == "full" else "当前为部分就绪（规则+部分模型）",
        },
        "models": {
            "pose_model_loaded": status.yolo_ready,
            "action_model_loaded": status.action_model_ready,
            "phase_model_count_loaded": status.phase_models_ready,
            "quality_model_loaded": status.quality_model_ready,
        },
        "checkpoints": {
            "dir": str(checkpoint_dir),
            "action_model_file": action_ckpt.name,
            "quality_model_file": quality_ckpt.name,
            "phase_model_files": [path.name for path in phase_files.values()],
            "missing_files": missing_files,
            "missing_phase_actions": missing_phase,
        },
        "features": feature_flags,
        "recommended_setup_steps": setup_steps,
    }


def _load_jsonl(path: Path, limit: Optional[int] = None) -> list[dict]:
    if not path.exists():
        return []

    rows: list[dict] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
    except Exception:
        return []

    if limit and limit > 0:
        return rows[-limit:]
    return rows


def _count_json_files(path: Path) -> int:
    if not path.exists():
        return 0
    return len(list(path.glob("*.json")))


def _count_video_files(path: Path) -> int:
    if not path.exists():
        return 0
    values = 0
    for p in path.iterdir():
        if not p.is_file():
            continue
        if p.name.startswith(".") or p.name.startswith("._"):
            continue
        if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
            values += 1
    return values


def _build_ingest_overview_payload() -> dict:
    rt = get_runtime()
    config = rt.config
    processed_root = rt.project_root / config["paths"]["processed"]
    raw_root = rt.project_root / config["paths"]["raw_videos"]
    skeleton_root = rt.project_root / config["paths"]["skeletons"]
    annotation_root = rt.project_root / config["paths"]["annotations"]
    ingest_root = processed_root / "ingest"

    state_path = ingest_root / "pipeline_state.json"
    manifest_path = ingest_root / "pexels_manifest.jsonl"
    rejected_path = ingest_root / "quality_rejected.jsonl"
    tags_path = ingest_root / "video_tags.jsonl"
    cleanup_log_path = ingest_root / "cleanup_log.jsonl"

    run_state = {}
    if state_path.exists():
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                run_state = payload
        except Exception:
            run_state = {}

    manifest_rows = _load_jsonl(manifest_path)
    rejected_rows = _load_jsonl(rejected_path)
    tags_rows = _load_jsonl(tags_path)
    cleanup_rows = _load_jsonl(cleanup_log_path, limit=80)

    rejected_counter = Counter(
        [str(r.get("reason") or "unknown") for r in rejected_rows]
    )
    action_name_map = {
        action_id: info.get("name", action_id)
        for action_id, info in config.get("actions", {}).items()
    }

    conversion_rows = []
    for action_id, action_name in action_name_map.items():
        downloaded = _count_video_files(raw_root / action_id)
        skeleton = _count_json_files(skeleton_root / action_id)
        annotation = _count_json_files(annotation_root / action_id)
        skeleton_rate = round((skeleton / downloaded * 100), 1) if downloaded else 0.0
        annotation_rate = (
            round((annotation / downloaded * 100), 1) if downloaded else 0.0
        )
        conversion_rows.append(
            {
                "action_id": action_id,
                "action_name": action_name,
                "downloaded": downloaded,
                "skeleton": skeleton,
                "annotation": annotation,
                "skeleton_rate": skeleton_rate,
                "annotation_rate": annotation_rate,
            }
        )

    conversion_rows.sort(key=lambda row: row["action_id"])

    latest_events = list(run_state.get("events", [])[-12:]) if run_state else []
    return {
        "run_state": {
            "run_id": run_state.get("run_id", ""),
            "status": run_state.get("status", "idle"),
            "stage": run_state.get("stage", ""),
            "message": run_state.get("message", ""),
            "updated_at": run_state.get("updated_at"),
            "summary": run_state.get("summary", {}),
            "pipeline": run_state.get("pipeline", {}),
            "latest_events": latest_events,
        },
        "conversion": conversion_rows,
        "files": {
            "manifest_count": len(manifest_rows),
            "rejected_count": len(rejected_rows),
            "tagged_count": len(tags_rows),
            "cleanup_log_count": len(cleanup_rows),
        },
        "rejected_reason_stats": [
            {"reason": key, "count": count}
            for key, count in rejected_counter.most_common(8)
        ],
        "latest_cleanup_logs": cleanup_rows[-20:],
    }


def _finalize_session(session_id: str) -> Optional[dict]:
    """Finalize active session into report and persist it."""
    session = store.remove_session(session_id)
    if session is None:
        return store.get_report(session_id)

    report = session.build_report()
    store.set_report(session_id, report)

    rt = get_runtime()
    report_path = (
        rt.project_root
        / rt.config["paths"]["processed"]
        / "realtime_reports"
        / f"{session_id}.json"
    )
    rt.dump_session_report(report_path, report)
    return report


def decode_image(image_base64: str) -> Optional[np.ndarray]:
    """Decode base64 JPEG/PNG data URL to BGR frame."""
    try:
        import importlib

        cv2 = importlib.import_module("cv2")
    except ImportError:
        return None

    payload = image_base64
    if "," in payload:
        payload = payload.split(",", 1)[1]

    try:
        frame_bytes = base64.b64decode(payload)
        arr = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None

    return frame


ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
UPLOAD_CHUNK_SIZE = 1024 * 1024
MAX_VIDEO_UPLOAD_BYTES = 300 * 1024 * 1024
UPLOAD_RATE_LIMIT_WINDOW_SEC = 60
UPLOAD_RATE_LIMIT_MAX_REQUESTS = 6

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "").strip()

upload_rate_lock = Lock()
upload_rate_records: dict[str, deque[float]] = defaultdict(deque)


def _safe_video_suffix(filename: str) -> str:
    """Validate and return accepted video extension."""
    suffix = Path(filename or "").suffix.lower()
    if suffix and suffix not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="仅支持 mp4/mov/avi/mkv/webm 格式")
    return suffix or ".mp4"


async def _save_upload_file_stream(
    file: UploadFile,
    output_path: Path,
    max_bytes: int = MAX_VIDEO_UPLOAD_BYTES,
) -> int:
    """Save upload by stream chunks and enforce max size."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0

    try:
        with open(output_path, "wb") as out:
            while True:
                chunk = await file.read(UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"视频文件过大，请控制在 {max_bytes // (1024 * 1024)}MB 以内",
                    )
                out.write(chunk)
    except HTTPException:
        if output_path.exists():
            output_path.unlink()
        raise
    except Exception:
        if output_path.exists():
            output_path.unlink()
        raise HTTPException(status_code=500, detail="上传文件保存失败，请重试")

    if total <= 0:
        if output_path.exists():
            output_path.unlink()
        raise HTTPException(status_code=400, detail="上传文件为空")

    return total


def _client_ip(request: Request) -> str:
    """Get best-effort client ip for local rate limiting."""
    xff = request.headers.get("x-forwarded-for", "").strip()
    if xff:
        return xff.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _check_upload_rate_limit(request: Request) -> None:
    """Apply lightweight per-IP upload rate limit."""
    now = time.time()
    ip = _client_ip(request)

    with upload_rate_lock:
        records = upload_rate_records[ip]
        while records and now - records[0] > UPLOAD_RATE_LIMIT_WINDOW_SEC:
            records.popleft()

        if len(records) >= UPLOAD_RATE_LIMIT_MAX_REQUESTS:
            raise HTTPException(
                status_code=429,
                detail=(f"上传过于频繁，请在 {UPLOAD_RATE_LIMIT_WINDOW_SEC} 秒后重试"),
            )

        records.append(now)


def _require_admin_auth(
    authorization: Optional[str] = Header(default=None),
    x_admin_token: Optional[str] = Header(default=None),
) -> None:
    """Minimal token auth for admin endpoints."""
    if not ADMIN_TOKEN:
        return

    bearer = ""
    if authorization and authorization.lower().startswith("bearer "):
        bearer = authorization[7:].strip()

    provided = (x_admin_token or "").strip() or bearer
    if provided != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="管理员认证失败")


@app.get("/api/health")
def health() -> dict:
    """Return dependency and model readiness."""
    rt = get_runtime()
    status = rt.status()
    return {
        "ok": True,
        "device": status.device,
        "models": {
            "yolo": status.yolo_ready,
            "action": status.action_model_ready,
            "phase_model_count": status.phase_models_ready,
            "quality": status.quality_model_ready,
        },
    }


@app.get("/api/system/overview")
def system_overview() -> dict:
    """Return user-facing capability and model readiness overview."""
    return _build_system_overview_payload()


@app.get("/api/monitor/live")
def live_monitor_snapshot() -> dict:
    """Return lightweight live monitoring payload for popup dashboard."""
    rt = get_runtime()
    manager = get_task_manager()
    status = rt.status()
    ingest = _build_ingest_overview_payload()

    return {
        "time": time.time(),
        "health": {
            "device": status.device,
            "yolo": status.yolo_ready,
            "action": status.action_model_ready,
            "phase_model_count": status.phase_models_ready,
            "quality": status.quality_model_ready,
        },
        "realtime": {
            "active_sessions": len(store.list_sessions()),
        },
        "video_tasks": manager.get_stats(),
        "ingest": {
            "run_id": ingest.get("run_state", {}).get("run_id", ""),
            "status": ingest.get("run_state", {}).get("status", "idle"),
            "stage": ingest.get("run_state", {}).get("stage", ""),
            "message": ingest.get("run_state", {}).get("message", ""),
            "updated_at": ingest.get("run_state", {}).get("updated_at"),
            "summary": ingest.get("run_state", {}).get("summary", {}),
            "latest_events": ingest.get("run_state", {}).get("latest_events", []),
        },
    }


@app.get("/api/actions")
def list_actions() -> dict:
    """Return available action types from config."""
    rt = get_runtime()
    actions = [
        {
            "id": action_id,
            "name": action_info.get("name", action_id),
            "num_phases": len(action_info.get("phases", [])),
        }
        for action_id, action_info in rt.config["actions"].items()
    ]
    return {"actions": actions}


@app.post("/api/inference/video")
async def infer_uploaded_video(
    request: Request,
    file: UploadFile = File(...),
    action_type: Optional[str] = Form(default=None),
) -> dict:
    """Run one-shot inference for uploaded video."""
    rt = get_runtime()
    _check_upload_rate_limit(request)

    if action_type and action_type not in rt.config["actions"]:
        raise HTTPException(status_code=400, detail="未知动作类型")

    suffix = _safe_video_suffix(file.filename or "")

    upload_dir = rt.project_root / rt.config["paths"]["processed"] / "temp_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved_path = upload_dir / f"{uuid.uuid4().hex}{suffix}"
    await _save_upload_file_stream(file, saved_path)

    try:
        result = rt.assess_video_file(saved_path, action_type)
        if result.get("ok"):
            result["video_name"] = file.filename or saved_path.name
        return result
    finally:
        if saved_path.exists():
            saved_path.unlink()


@app.post("/api/inference/video/tasks")
async def create_video_inference_task(
    request: Request,
    file: UploadFile = File(...),
    action_type: Optional[str] = Form(default=None),
) -> dict:
    """Create an async task for uploaded video assessment."""
    rt = get_runtime()
    manager = get_task_manager()
    _check_upload_rate_limit(request)

    if action_type and action_type not in rt.config["actions"]:
        raise HTTPException(status_code=400, detail="未知动作类型")

    suffix = _safe_video_suffix(file.filename or "")
    upload_dir = rt.project_root / rt.config["paths"]["processed"] / "temp_uploads"
    staged_path = upload_dir / f"stage_{uuid.uuid4().hex}{suffix}"
    await _save_upload_file_stream(file, staged_path)

    try:
        task = manager.create_task_from_staged_file(
            staged_path=staged_path,
            filename=file.filename or "video.mp4",
            action_type=action_type,
        )
    except Exception:
        if staged_path.exists():
            staged_path.unlink()
        raise HTTPException(status_code=500, detail="创建评估任务失败，请重试")

    return {"task": task}


@app.get("/api/inference/video/tasks")
def list_video_inference_tasks() -> dict:
    """List video inference tasks (includes history)."""
    manager = get_task_manager()
    return {"tasks": manager.list_tasks()}


@app.get("/api/inference/video/tasks/{task_id}")
def get_video_inference_task(task_id: str) -> dict:
    """Get one task detail by id."""
    manager = get_task_manager()
    task = manager.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return {"task": task}


@app.delete("/api/inference/video/tasks/{task_id}")
def delete_video_inference_task(task_id: str) -> dict:
    """Delete one completed/failed video inference history task."""
    manager = get_task_manager()
    ok, message = manager.delete_task(task_id)
    if not ok:
        status = 404 if message == "任务不存在" else 409
        raise HTTPException(status_code=status, detail=message)
    return {"ok": True}


@app.delete("/api/inference/video/history")
def clear_video_inference_history() -> dict:
    """Delete all completed/failed task history items."""
    manager = get_task_manager()
    removed = manager.clear_history()
    return {"ok": True, "removed": removed}


@app.get("/api/admin/overview")
def admin_overview(_: None = Depends(_require_admin_auth)) -> dict:
    """Provide admin dashboard overview data."""
    rt = get_runtime()
    manager = get_task_manager()
    model_status = rt.status()
    task_stats = manager.get_stats()
    active_sessions = store.list_sessions()
    latest_tasks = manager.list_tasks()[:10]
    latest_reports = _latest_realtime_reports(limit=10)

    return {
        "system": {
            "device": model_status.device,
            "models": {
                "yolo": model_status.yolo_ready,
                "action": model_status.action_model_ready,
                "phase_model_count": model_status.phase_models_ready,
                "quality": model_status.quality_model_ready,
            },
        },
        "realtime": {
            "active_session_count": len(active_sessions),
            "active_sessions": active_sessions,
            "latest_reports": latest_reports,
        },
        "video_tasks": {
            "stats": task_stats,
            "latest": latest_tasks,
        },
    }


@app.get("/api/admin/video_tasks")
def admin_video_tasks(_: None = Depends(_require_admin_auth)) -> dict:
    """List all background video tasks for admin monitoring."""
    manager = get_task_manager()
    return {
        "stats": manager.get_stats(),
        "tasks": manager.list_tasks(),
    }


@app.get("/api/admin/realtime_reports")
def admin_realtime_reports(
    limit: int = 20,
    _: None = Depends(_require_admin_auth),
) -> dict:
    """List persisted realtime session reports for admin."""
    return {"reports": _latest_realtime_reports(limit=max(1, min(limit, 200)))}


@app.get("/api/admin/ingest/overview")
def admin_ingest_overview(_: None = Depends(_require_admin_auth)) -> dict:
    """Provide ingest pipeline and dataset conversion monitoring."""
    return _build_ingest_overview_payload()


@app.post("/api/realtime/session/start")
def start_session(payload: StartSessionRequest) -> dict:
    """Create one realtime session."""
    rt = get_runtime()

    if payload.action_type and payload.action_type not in rt.config["actions"]:
        raise HTTPException(status_code=400, detail="未知动作类型")

    session_id = uuid.uuid4().hex
    session = RealtimeSession(
        session_id=session_id,
        runtime=rt,
        action_hint=payload.action_type,
        target_reps=payload.target_reps,
        window_size=payload.window_size,
        infer_interval=payload.infer_interval,
    )
    store.set_session(session)

    return {
        "session_id": session_id,
        "ws_url": f"/ws/realtime/{session_id}",
    }


@app.post("/api/realtime/session/{session_id}/stop", response_model=StopSessionResponse)
def stop_session(session_id: str) -> StopSessionResponse:
    """Stop session and return report."""
    report = _finalize_session(session_id)
    if report is None:
        raise HTTPException(status_code=404, detail="会话不存在")

    rt = get_runtime()
    report_path = (
        rt.project_root
        / rt.config["paths"]["processed"]
        / "realtime_reports"
        / f"{session_id}.json"
    )

    return StopSessionResponse(
        session_id=session_id,
        report_path=str(report_path),
        report=report,
    )


@app.get("/api/reports/{session_id}")
def get_report(session_id: str) -> dict:
    """Fetch session report by id."""
    report = store.get_report(session_id)
    if report:
        return report

    rt = get_runtime()
    report_path = (
        rt.project_root
        / rt.config["paths"]["processed"]
        / "realtime_reports"
        / f"{session_id}.json"
    )
    if report_path.exists():
        import json

        with open(report_path, encoding="utf-8") as f:
            return json.load(f)

    raise HTTPException(status_code=404, detail="报告不存在")


@app.websocket("/ws/realtime/{session_id}")
async def ws_realtime(websocket: WebSocket, session_id: str) -> None:
    """Realtime websocket: receive frames and return inference."""
    await websocket.accept()

    session = store.get_session(session_id)
    if session is None:
        await websocket.send_json({"status": "error", "message": "会话不存在"})
        await websocket.close(code=1008)
        return

    try:
        while True:
            payload = await websocket.receive_json()
            if payload.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            image_base64 = payload.get("image_base64")
            if not image_base64:
                await websocket.send_json(
                    {"status": "error", "message": "缺少 image_base64 字段"}
                )
                continue

            frame = decode_image(image_base64)
            if frame is None:
                await websocket.send_json({"status": "error", "message": "图像解码失败"})
                continue

            result = session.process_frame(frame)
            await websocket.send_json(result)

    except WebSocketDisconnect:
        _finalize_session(session_id)
        return
    except Exception as exc:
        _finalize_session(session_id)
        await websocket.send_json({"status": "error", "message": f"服务异常: {exc}"})
        await websocket.close(code=1011)


@app.get("/")
def index() -> dict:
    """Simple root endpoint."""
    return {"message": "体育动作实时教学服务运行中"}
