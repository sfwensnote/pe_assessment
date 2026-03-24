"""Background task manager for uploaded video assessment."""

from __future__ import annotations

import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional

from app.services.model_runtime import ModelRuntime


@dataclass
class VideoInferenceTask:
    """Serializable task state for uploaded video assessment."""

    task_id: str
    status: str
    progress: float
    message: str
    filename: str
    action_type: Optional[str]
    created_at: float
    updated_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    duration_sec: Optional[float] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["progress"] = round(self.progress, 2)
        return payload


class VideoInferenceTaskManager:
    """Run upload evaluations asynchronously and track progress/history."""

    def __init__(
        self,
        runtime_provider: Callable[[], ModelRuntime],
        upload_dir: Path,
        store_path: Path,
        max_workers: int = 2,
    ) -> None:
        self.runtime_provider = runtime_provider
        self.upload_dir = upload_dir
        self.store_path = store_path
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.lock = Lock()
        self.tasks: Dict[str, VideoInferenceTask] = {}

        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_from_disk()

    def create_task(
        self,
        file_bytes: bytes,
        filename: str,
        action_type: Optional[str],
    ) -> Dict[str, Any]:
        """Create a task and enqueue async processing."""
        suffix = Path(filename or "").suffix.lower() or ".mp4"
        staged_path = self.upload_dir / f"stage_{uuid.uuid4().hex}{suffix}"
        with open(staged_path, "wb") as f:
            f.write(file_bytes)

        return self.create_task_from_staged_file(staged_path, filename, action_type)

    def create_task_from_staged_file(
        self,
        staged_path: Path,
        filename: str,
        action_type: Optional[str],
    ) -> Dict[str, Any]:
        """Create task from an existing staged upload file."""
        task_id = uuid.uuid4().hex
        suffix = staged_path.suffix.lower() or ".mp4"
        temp_path = self.upload_dir / f"{task_id}{suffix}"
        staged_path.replace(temp_path)

        now = time.time()
        task = VideoInferenceTask(
            task_id=task_id,
            status="queued",
            progress=0.0,
            message="排队中",
            filename=filename or temp_path.name,
            action_type=action_type,
            created_at=now,
            updated_at=now,
        )

        with self.lock:
            self.tasks[task_id] = task
            self._dump_to_disk_locked()

        self.executor.submit(self._run_task, task_id, temp_path, action_type, filename)
        return task.to_dict()

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Fetch one task by id."""
        with self.lock:
            task = self.tasks.get(task_id)
            return task.to_dict() if task else None

    def list_tasks(self) -> List[Dict[str, Any]]:
        """List all tasks in reverse chronological order."""
        with self.lock:
            ordered = sorted(
                self.tasks.values(),
                key=lambda t: t.created_at,
                reverse=True,
            )
            return [task.to_dict() for task in ordered]

    def get_stats(self) -> Dict[str, Any]:
        """Return aggregated task statistics for admin dashboard."""
        with self.lock:
            values = list(self.tasks.values())

        stats = {
            "total": len(values),
            "queued": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "avg_duration_sec": 0.0,
        }

        durations = []
        for task in values:
            if task.status in stats:
                stats[task.status] += 1
            if task.duration_sec is not None:
                durations.append(float(task.duration_sec))

        if durations:
            stats["avg_duration_sec"] = round(sum(durations) / len(durations), 2)

        return stats

    def delete_task(self, task_id: str) -> tuple[bool, str]:
        """Delete one history task if it is not running."""
        with self.lock:
            task = self.tasks.get(task_id)
            if task is None:
                return False, "任务不存在"

            if task.status in {"queued", "running"}:
                return False, "任务执行中，无法删除"

            del self.tasks[task_id]
            self._dump_to_disk_locked()

        return True, "ok"

    def clear_history(self) -> int:
        """Delete all completed/failed history tasks."""
        with self.lock:
            removable_ids = [
                task_id
                for task_id, task in self.tasks.items()
                if task.status in {"completed", "failed"}
            ]
            for task_id in removable_ids:
                del self.tasks[task_id]

            if removable_ids:
                self._dump_to_disk_locked()

        return len(removable_ids)

    def _run_task(
        self,
        task_id: str,
        temp_path: Path,
        action_type: Optional[str],
        original_filename: str,
    ) -> None:
        self._update_task(task_id, status="running", progress=2.0, message="开始处理")
        started_at = time.time()
        self._update_task(task_id, started_at=started_at)

        try:
            runtime = self.runtime_provider()

            def on_progress(progress: float, message: str) -> None:
                self._update_task(
                    task_id,
                    status="running",
                    progress=progress,
                    message=message,
                )

            result = runtime.assess_video_file(
                temp_path,
                action_hint=action_type,
                progress_callback=on_progress,
            )

            if result.get("ok"):
                result["video_name"] = original_filename

            finished_at = time.time()
            duration = max(0.0, finished_at - started_at)

            if result.get("ok"):
                self._update_task(
                    task_id,
                    status="completed",
                    progress=100.0,
                    message="评估完成",
                    result=result,
                    finished_at=finished_at,
                    duration_sec=round(duration, 2),
                )
            else:
                self._update_task(
                    task_id,
                    status="failed",
                    message=result.get("error", "评估失败"),
                    error=result.get("error", "评估失败"),
                    progress=max(5.0, self._current_progress(task_id)),
                    finished_at=finished_at,
                    duration_sec=round(duration, 2),
                )
        except Exception as exc:
            finished_at = time.time()
            duration = max(0.0, finished_at - started_at)
            self._update_task(
                task_id,
                status="failed",
                message="任务失败",
                error=str(exc),
                progress=max(5.0, self._current_progress(task_id)),
                finished_at=finished_at,
                duration_sec=round(duration, 2),
            )
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def _current_progress(self, task_id: str) -> float:
        with self.lock:
            task = self.tasks.get(task_id)
            return task.progress if task else 0.0

    def _update_task(self, task_id: str, **kwargs: Any) -> None:
        with self.lock:
            task = self.tasks.get(task_id)
            if task is None:
                return

            for key, value in kwargs.items():
                if key == "progress":
                    value = max(float(value), float(task.progress))
                setattr(task, key, value)
            task.updated_at = time.time()
            self._dump_to_disk_locked()

    def _load_from_disk(self) -> None:
        if not self.store_path.exists():
            return

        try:
            with open(self.store_path, "r", encoding="utf-8") as f:
                rows = json.load(f)
        except Exception:
            return

        if not isinstance(rows, list):
            return

        for row in rows:
            if not isinstance(row, dict) or "task_id" not in row:
                continue
            task = VideoInferenceTask(**row)
            if task.status in {"running", "queued"}:
                task.status = "failed"
                task.message = "服务重启导致任务中断"
                task.error = "任务中断"
            self.tasks[task.task_id] = task

    def _dump_to_disk_locked(self) -> None:
        rows = [task.to_dict() for task in self.tasks.values()]
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
