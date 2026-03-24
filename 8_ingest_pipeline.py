#!/usr/bin/env python3
"""
8_ingest_pipeline.py
自动采集 Pexels 视频，并编排执行：采集 -> 部署脚本生成 -> 预处理 -> 自动标注。

用法示例:
    export PEXELS_API_KEY="your_api_key"
    python 8_ingest_pipeline.py

    # 仅运行俯卧撑和深蹲，每个动作本轮最多新增 6 条
    python 8_ingest_pipeline.py --actions pushup,squat --max_new_per_action 6

    # 仅采集下载，不触发预处理与标注
    python 8_ingest_pipeline.py --no_pipeline
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import ssl
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import error, parse, request

import yaml


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
USER_AGENT = "pe-assessment-ingest/1.0"


class ProgressTracker:
    """JSON state writer for live monitor script."""

    def __init__(self, state_path: Path) -> None:
        self.state_path = state_path
        self.state: Dict[str, Any] = {}

    def start(
        self,
        *,
        run_id: str,
        actions_state: Dict[str, Dict[str, Any]],
        run_config: Dict[str, Any],
    ) -> None:
        now = time.time()
        self.state = {
            "run_id": run_id,
            "status": "running",
            "stage": "bootstrap",
            "message": "初始化中",
            "started_at": now,
            "updated_at": now,
            "finished_at": None,
            "run_config": run_config,
            "summary": {
                "existing_total": sum(
                    v["existing_local"] for v in actions_state.values()
                ),
                "scanned_total": 0,
                "added_total": 0,
                "skeleton_added_total": 0,
                "annotation_added_total": 0,
                "duplicate_total": 0,
                "quality_skipped_total": 0,
                "metadata_skipped_total": 0,
                "pose_skipped_total": 0,
                "failed_total": 0,
                "api_failure_total": 0,
            },
            "pipeline": {
                "deploy": "pending",
                "preprocess": "pending",
                "annotate": "pending",
            },
            "actions": actions_state,
            "events": [],
            "error": "",
        }
        self._flush()

    def set_stage(self, stage: str, message: str) -> None:
        self.state["stage"] = stage
        self.state["message"] = message
        self._flush()

    def set_pipeline_status(self, key: str, status: str) -> None:
        self.state.setdefault("pipeline", {})[key] = status
        self._flush()

    def update_action(self, action_id: str, **kwargs: Any) -> None:
        action = self.state.get("actions", {}).get(action_id)
        if not action:
            return
        action.update(kwargs)
        self._flush()

    def incr_action(self, action_id: str, key: str, delta: int = 1) -> None:
        action = self.state.get("actions", {}).get(action_id)
        if not action:
            return
        action[key] = int(action.get(key, 0)) + int(delta)
        self._flush()

    def incr_summary(self, key: str, delta: int = 1) -> None:
        summary = self.state.get("summary", {})
        summary[key] = int(summary.get(key, 0)) + int(delta)
        self._flush()

    def add_event(
        self,
        level: str,
        message: str,
        *,
        action_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "time": time.time(),
            "level": level,
            "message": message,
        }
        if action_id:
            payload["action_id"] = action_id
        if extra:
            payload["extra"] = extra

        events = self.state.setdefault("events", [])
        events.append(payload)
        if len(events) > 300:
            self.state["events"] = events[-300:]
        self._flush()

    def finish(self, status: str, message: str, error_message: str = "") -> None:
        self.state["status"] = status
        if status in {"completed", "partial", "cancelled"}:
            self.state["stage"] = "done"
        elif status == "failed":
            self.state["stage"] = "failed"
        self.state["message"] = message
        self.state["error"] = error_message
        self.state["finished_at"] = time.time()
        self._flush()

        run_id = str(self.state.get("run_id") or "").strip()
        if not run_id:
            return
        snapshot_path = self.state_path.parent / f"pipeline_state_{run_id}.json"
        try:
            with open(snapshot_path, "w", encoding="utf-8") as f:
                json.dump(self.state, f, ensure_ascii=False, indent=2)
        except Exception:
            return

    def _flush(self) -> None:
        self.state["updated_at"] = time.time()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)


class PexelsClient:
    """Minimal Pexels API client based on urllib."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout_sec: int,
        insecure_ssl: bool,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.timeout_sec = timeout_sec
        self.insecure_ssl = insecure_ssl

    def search_videos(
        self,
        *,
        query: str,
        page: int,
        per_page: int,
        orientation: str,
        locale: str,
    ) -> Dict[str, Any]:
        params = {
            "query": query,
            "page": page,
            "per_page": per_page,
        }
        if orientation:
            params["orientation"] = orientation
        if locale:
            params["locale"] = locale

        url = f"{self.base_url}?{parse.urlencode(params)}"
        req = request.Request(
            url,
            headers={
                "Authorization": self.api_key,
                "Accept": "application/json",
                "User-Agent": USER_AGENT,
            },
        )

        ssl_context = None
        if self.insecure_ssl:
            ssl_context = ssl._create_unverified_context()

        try:
            with request.urlopen(
                req,
                timeout=self.timeout_sec,
                context=ssl_context,
            ) as resp:
                body = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            text = ""
            try:
                text = exc.read().decode("utf-8", errors="ignore")
            except Exception:
                text = ""
            raise RuntimeError(
                f"Pexels API 请求失败 HTTP {exc.code}: {text[:180]}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(f"Pexels API 网络异常: {exc}") from exc

        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Pexels API 返回 JSON 解析失败") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="自动采集视频并执行预处理/自动标注")
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.getenv("PEXELS_API_KEY", ""),
        help="Pexels API Key；默认读取环境变量 PEXELS_API_KEY",
    )
    parser.add_argument(
        "--actions",
        type=str,
        default="all",
        help="动作列表，如 pushup,squat；默认 all",
    )
    parser.add_argument("--target_per_action", type=int, default=None, help="每个动作目标总量")
    parser.add_argument(
        "--max_new_per_action", type=int, default=None, help="每个动作本轮最多新增"
    )
    parser.add_argument("--max_new_total", type=int, default=None, help="本轮最多新增总量")
    parser.add_argument("--per_page", type=int, default=None, help="每页返回条数")
    parser.add_argument("--max_pages", type=int, default=None, help="每个关键词最多翻页数")
    parser.add_argument("--interval", type=float, default=None, help="请求间隔秒")
    parser.add_argument("--jitter", type=float, default=None, help="请求抖动秒")
    parser.add_argument("--timeout", type=int, default=None, help="API/下载超时秒")
    parser.add_argument(
        "--insecure_ssl",
        action="store_true",
        help="忽略 SSL 证书校验（仅在证书链异常时临时使用）",
    )
    parser.add_argument(
        "--no_pipeline",
        action="store_true",
        help="仅采集，不执行 0_preprocess_videos.py 与 1_auto_annotate.py",
    )
    parser.add_argument("--dry_run", action="store_true", help="仅演练，不落地下载")
    parser.add_argument(
        "--python_bin",
        type=str,
        default=sys.executable,
        help="执行后续脚本使用的 Python 解释器",
    )
    parser.add_argument(
        "--skip_env_check",
        action="store_true",
        help="跳过环境依赖检查（默认会检查预处理/标注依赖）",
    )
    parser.add_argument(
        "--disable_pose_probe",
        action="store_true",
        help="关闭下载后姿态快速质检（默认开启）",
    )
    parser.add_argument(
        "--pose_sample_frames",
        type=int,
        default=None,
        help="姿态快速质检抽帧数（默认读取配置）",
    )
    parser.add_argument(
        "--auto_tag_cleanup",
        action="store_true",
        help="采集后自动执行视频打标签与垃圾清理脚本",
    )
    parser.add_argument(
        "--auto_tag_cleanup_dry_run",
        action="store_true",
        help="与 --auto_tag_cleanup 搭配，仅预览清理结果",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_actions(actions_arg: str, config_actions: Dict[str, Any]) -> List[str]:
    if actions_arg.strip().lower() == "all":
        return list(config_actions.keys())

    values = [x.strip() for x in actions_arg.split(",") if x.strip()]
    unknown = [x for x in values if x not in config_actions]
    if unknown:
        raise ValueError(f"存在未知动作: {', '.join(unknown)}")
    return values


def count_local_videos(action_dir: Path) -> int:
    if not action_dir.exists():
        return 0
    count = 0
    for p in action_dir.iterdir():
        if (
            p.is_file()
            and p.suffix.lower() in VIDEO_EXTENSIONS
            and not p.name.startswith(".")
            and not p.name.startswith("._")
        ):
            count += 1
    return count


def validate_directory_path(path: Path, label: str) -> Tuple[bool, str]:
    if path.is_symlink() and not path.exists():
        return (
            False,
            f"{label} 路径不可用（符号链接目标不存在）: {path}",
        )

    if path.exists() and not path.is_dir():
        return False, f"{label} 不是目录: {path}"

    return True, ""


def _is_process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def acquire_run_lock(lock_path: Path) -> Tuple[bool, str]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        try:
            payload = json.loads(lock_path.read_text(encoding="utf-8"))
            old_pid = int(payload.get("pid", 0))
            old_run_id = str(payload.get("run_id", ""))
            if _is_process_alive(old_pid):
                return False, f"检测到已有运行中的采集任务(pid={old_pid}, run_id={old_run_id})"
        except Exception:
            pass

    lock_payload = {
        "pid": os.getpid(),
        "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "time": time.time(),
    }
    lock_path.write_text(json.dumps(lock_payload, ensure_ascii=False), encoding="utf-8")
    return True, ""


def release_run_lock(lock_path: Path) -> None:
    try:
        if lock_path.exists():
            lock_path.unlink()
    except Exception:
        return


def cleanup_sidecar_files(action_dir: Path) -> int:
    if not action_dir.exists():
        return 0

    removed = 0
    for path in action_dir.iterdir():
        if not path.is_file():
            continue
        if path.name == ".DS_Store" or path.name.startswith("._"):
            path.unlink(missing_ok=True)
            removed += 1
    return removed


def count_json_files(path: Path) -> int:
    if not path.exists():
        return 0
    return len(list(path.glob("*.json")))


def check_runtime_dependencies(python_bin: str) -> Tuple[bool, str]:
    check_code = (
        "import importlib.util as u,sys;"
        "mods=['numpy','cv2','yaml','ultralytics','matplotlib'];"
        "missing=[m for m in mods if u.find_spec(m) is None];"
        "print(','.join(missing));"
        "sys.exit(1 if missing else 0)"
    )

    process = subprocess.run(
        [python_bin, "-c", check_code],
        capture_output=True,
        text=True,
    )
    missing = (process.stdout or "").strip().strip(",")
    if process.returncode == 0:
        return True, ""
    if missing:
        return False, f"缺少依赖: {missing}"
    stderr = (process.stderr or "").strip()
    if stderr:
        return False, stderr
    return False, "依赖检查失败"


def load_video_ids_from_jsonl(path: Path) -> set[str]:
    if not path.exists():
        return set()

    values: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            raw_id = row.get("pexels_video_id")
            if raw_id is None:
                continue
            values.add(str(raw_id))
    return values


def normalize_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))


def is_informative_url_path(path: str) -> bool:
    tokens = normalize_text(path).split()
    generic = {"zh", "cn", "tw", "en", "video", "videos"}
    meaningful = [t for t in tokens if t not in generic and not t.isdigit()]
    return len(meaningful) >= 2


def keyword_hits(text: str, keywords: List[str]) -> List[str]:
    if not text:
        return []
    raw = text.lower()
    norm = normalize_text(text)
    hits: List[str] = []

    for item in keywords:
        keyword = (item or "").strip().lower()
        if not keyword:
            continue
        keyword_norm = normalize_text(keyword)
        if keyword in raw or (keyword_norm and keyword_norm in norm):
            hits.append(item)

    return hits


def action_relevance_keywords(
    config: Dict[str, Any], action_id: str, queries: List[str]
) -> List[str]:
    ingest_actions = config.get("ingest", {}).get("actions", {})
    custom = ingest_actions.get(action_id, {}).get("keywords", [])
    values = [str(x).strip() for x in custom if str(x).strip()]
    if values:
        return values

    auto: List[str] = []
    for query in queries:
        q = normalize_text(query)
        if q and q not in auto:
            auto.append(q)
    return auto


def even_sample_frame_indexes(total_frames: int, sample_count: int) -> List[int]:
    if total_frames <= 0:
        return []
    if sample_count <= 1 or total_frames == 1:
        return [0]
    if total_frames <= sample_count:
        return list(range(total_frames))

    step = (total_frames - 1) / float(sample_count - 1)
    values = sorted({int(round(i * step)) for i in range(sample_count)})
    return values


class PoseQualityGate:
    def __init__(
        self,
        *,
        model_path: str,
        sample_frames: int,
        keypoint_conf_threshold: float,
        min_person_frame_ratio: float,
        min_single_person_ratio: float,
        min_avg_visible_keypoints: float,
        min_primary_person_area_ratio: float,
    ) -> None:
        self.model_path = model_path
        self.sample_frames = max(4, int(sample_frames))
        self.keypoint_conf_threshold = float(keypoint_conf_threshold)
        self.min_person_frame_ratio = float(min_person_frame_ratio)
        self.min_single_person_ratio = float(min_single_person_ratio)
        self.min_avg_visible_keypoints = float(min_avg_visible_keypoints)
        self.min_primary_person_area_ratio = float(min_primary_person_area_ratio)

        self._cv2 = None
        self._model = None

    def initialize(self) -> Tuple[bool, str]:
        if self._model is not None and self._cv2 is not None:
            return True, ""

        try:
            import cv2
            from ultralytics import YOLO

            self._cv2 = cv2
            self._model = YOLO(self.model_path)
            return True, ""
        except Exception as exc:
            return False, str(exc)

    def evaluate(self, video_path: Path) -> Tuple[bool, Dict[str, float], str]:
        ok, reason = self.initialize()
        if not ok:
            return False, {}, f"姿态质检初始化失败: {reason}"

        assert self._cv2 is not None
        assert self._model is not None

        cap = self._cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False, {}, "视频无法打开"

        total_frames = int(cap.get(self._cv2.CAP_PROP_FRAME_COUNT) or 0)
        indexes = even_sample_frame_indexes(total_frames, self.sample_frames)
        if not indexes:
            cap.release()
            return False, {}, "视频帧数为 0"

        sampled = 0
        person_frames = 0
        single_person_frames = 0
        visible_keypoints_sum = 0.0
        area_ratio_sum = 0.0

        for idx in indexes:
            cap.set(self._cv2.CAP_PROP_POS_FRAMES, idx)
            ok_read, frame = cap.read()
            if not ok_read or frame is None:
                continue

            sampled += 1

            try:
                outputs = self._model(frame, verbose=False)
            except Exception:
                continue

            if not outputs:
                continue

            result = outputs[0]
            keypoints = getattr(result, "keypoints", None)
            if keypoints is None or len(keypoints) == 0:
                continue

            person_frames += 1
            if len(keypoints) == 1:
                single_person_frames += 1

            visible_count = 0
            conf = getattr(keypoints, "conf", None)
            if conf is not None and len(conf) > 0:
                try:
                    arr = conf[0].cpu().numpy()
                    visible_count = int((arr >= self.keypoint_conf_threshold).sum())
                except Exception:
                    visible_count = 0
            else:
                xy = getattr(keypoints, "xy", None)
                if xy is not None and len(xy) > 0:
                    try:
                        arr = xy[0].cpu().numpy()
                        visible_count = int(((arr[:, 0] > 0) & (arr[:, 1] > 0)).sum())
                    except Exception:
                        visible_count = 0

            visible_keypoints_sum += float(visible_count)

            area_ratio = 0.0
            boxes = getattr(result, "boxes", None)
            if boxes is not None and len(boxes) > 0:
                try:
                    box = boxes.xyxy[0].cpu().numpy().tolist()
                    x1, y1, x2, y2 = box[:4]
                    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                    denom = max(1.0, float(frame.shape[0] * frame.shape[1]))
                    area_ratio = area / denom
                except Exception:
                    area_ratio = 0.0

            area_ratio_sum += area_ratio

        cap.release()

        if sampled <= 0:
            return False, {}, "无法读取有效帧"

        person_ratio = person_frames / sampled
        single_ratio = (
            (single_person_frames / person_frames) if person_frames > 0 else 0.0
        )
        avg_visible_keypoints = (
            (visible_keypoints_sum / person_frames) if person_frames > 0 else 0.0
        )
        avg_area_ratio = (area_ratio_sum / person_frames) if person_frames > 0 else 0.0

        metrics = {
            "sampled_frames": float(sampled),
            "person_ratio": float(person_ratio),
            "single_person_ratio": float(single_ratio),
            "avg_visible_keypoints": float(avg_visible_keypoints),
            "avg_area_ratio": float(avg_area_ratio),
        }

        reasons: List[str] = []
        if person_ratio < self.min_person_frame_ratio:
            reasons.append(
                f"人体检测帧占比过低({person_ratio:.2f}<{self.min_person_frame_ratio:.2f})"
            )
        if person_frames > 0:
            if single_ratio < self.min_single_person_ratio:
                reasons.append(
                    f"单人帧占比过低({single_ratio:.2f}<{self.min_single_person_ratio:.2f})"
                )
            if avg_visible_keypoints < self.min_avg_visible_keypoints:
                reasons.append(
                    "关键点可见数偏低"
                    "({:.1f}<{:.1f})".format(
                        avg_visible_keypoints,
                        self.min_avg_visible_keypoints,
                    )
                )
            if avg_area_ratio < self.min_primary_person_area_ratio:
                reasons.append(
                    "人物主体面积占比过低"
                    "({:.3f}<{:.3f})".format(
                        avg_area_ratio,
                        self.min_primary_person_area_ratio,
                    )
                )

        if reasons:
            return False, metrics, "；".join(reasons)
        return True, metrics, "通过"


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def pick_video_candidate(
    video_payload: Dict[str, Any],
    *,
    min_width: int,
    min_height: int,
) -> Optional[Dict[str, Any]]:
    files = video_payload.get("video_files") or []
    candidates: List[Tuple[int, int, int, int, str, str]] = []

    for item in files:
        width = int(item.get("width") or 0)
        height = int(item.get("height") or 0)
        link = str(item.get("link") or "")
        file_type = str(item.get("file_type") or "")
        quality = str(item.get("quality") or "")

        if not link:
            continue

        is_mp4 = file_type.lower().endswith("mp4") or ".mp4" in link.lower()
        if not is_mp4:
            continue

        if width < min_width or height < min_height:
            continue

        area = width * height
        quality_score = 0 if quality.lower() in {"sd", ""} else 1
        candidates.append((area, quality_score, width, height, link, file_type))

    if not candidates:
        return None

    candidates.sort(key=lambda row: (-row[0], -row[1]))
    area, _, width, height, link, file_type = candidates[0]
    return {
        "width": width,
        "height": height,
        "link": link,
        "file_type": file_type,
        "area": area,
    }


def guess_suffix(file_type: str, link: str) -> str:
    lowered = file_type.lower()
    if lowered.endswith("mp4") or ".mp4" in link.lower():
        return ".mp4"
    parsed = parse.urlparse(link)
    ext = Path(parsed.path).suffix.lower()
    if ext in VIDEO_EXTENSIONS:
        return ext
    return ".mp4"


def download_file(
    url: str,
    output_path: Path,
    timeout_sec: int,
    insecure_ssl: bool,
) -> int:
    req = request.Request(url, headers={"User-Agent": USER_AGENT})
    ssl_context = None
    if insecure_ssl:
        ssl_context = ssl._create_unverified_context()
    with request.urlopen(req, timeout=timeout_sec, context=ssl_context) as resp:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            while True:
                chunk = resp.read(1024 * 512)
                if not chunk:
                    break
                f.write(chunk)

    return output_path.stat().st_size if output_path.exists() else 0


def sleep_with_jitter(base_sec: float, jitter_sec: float) -> None:
    interval = max(0.0, float(base_sec))
    jitter = max(0.0, float(jitter_sec))
    wait_sec = interval + random.uniform(0.0, jitter)
    time.sleep(wait_sec)


def write_if_changed(path: Path, content: str, executable: bool = False) -> None:
    existing = ""
    if path.exists():
        existing = path.read_text(encoding="utf-8")
    if existing != content:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    if executable:
        path.chmod(0o755)


def generate_deploy_assets(project_root: Path) -> None:
    deploy_dir = project_root / "deploy" / "ingest"

    run_script = """#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

PY_BIN="${PYTHON_BIN:-python3}"
"$PY_BIN" 8_ingest_pipeline.py "$@"
"""

    monitor_script = """#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

PY_BIN="${PYTHON_BIN:-python3}"
"$PY_BIN" 8_ingest_monitor.py --watch
"""

    run_with_monitor = """#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

PY_BIN="${PYTHON_BIN:-python3}"

"$PY_BIN" 8_ingest_pipeline.py "$@" &
PIPE_PID=$!

cleanup() {
  if kill -0 "$PIPE_PID" >/dev/null 2>&1; then
    kill "$PIPE_PID" >/dev/null 2>&1 || true
  fi
}

trap cleanup INT TERM

"$PY_BIN" 8_ingest_monitor.py --watch --stop_when_done
wait "$PIPE_PID"
"""

    cron_example = """# 每天 09:00 与 21:00 自动采集并编排处理
0 9,21 * * * cd /path/to/pe_assessment && \\
PEXELS_API_KEY=your_api_key python 8_ingest_pipeline.py >> logs/ingest_cron.log 2>&1

# 使用下面命令实时查看状态
# cd /path/to/pe_assessment && python 8_ingest_monitor.py --watch
"""

    write_if_changed(deploy_dir / "run_ingest_pipeline.sh", run_script, executable=True)
    write_if_changed(
        deploy_dir / "watch_ingest_progress.sh", monitor_script, executable=True
    )
    write_if_changed(
        deploy_dir / "run_with_live_monitor.sh",
        run_with_monitor,
        executable=True,
    )
    write_if_changed(deploy_dir / "cron.example", cron_example, executable=False)


def run_python_script(
    *,
    project_root: Path,
    python_bin: str,
    script_name: str,
    action_id: str,
    log_path: Path,
    extra_args: Optional[List[str]] = None,
) -> int:
    command = [python_bin, script_name, "--action", action_id]
    if extra_args:
        command.extend(extra_args)
    print(f"\n[执行] {' '.join(command)}")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "a", encoding="utf-8") as log:
        log.write(
            "\n\n===== "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
            f"{script_name} {action_id} =====\n"
        )
        process = subprocess.Popen(
            command,
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return process.wait()


def action_queries(config: Dict[str, Any], action_id: str) -> List[str]:
    ingest_actions = config.get("ingest", {}).get("actions", {})
    default_name = config.get("actions", {}).get(action_id, {}).get("name", action_id)
    queries = ingest_actions.get(action_id, {}).get("queries", [])
    values = [str(x).strip() for x in queries if str(x).strip()]
    if not values:
        values = [default_name]
    return values


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).parent
    config_path = project_root / "config.yaml"
    config = load_config(config_path)

    all_actions = config.get("actions", {})
    selected_actions = resolve_actions(args.actions, all_actions)

    ingest_cfg = config.get("ingest", {})
    pexels_cfg = ingest_cfg.get("pexels", {})
    run_policy = ingest_cfg.get("run_policy", {})
    quality_gate = ingest_cfg.get("quality_gate", {})

    api_key = (args.api_key or "").strip()
    if not api_key:
        print("错误: 未设置 PEXELS_API_KEY（或 --api_key）")
        print("示例: export PEXELS_API_KEY='your_api_key'")
        sys.exit(1)

    raw_videos_root = project_root / config["paths"]["raw_videos"]
    skeleton_root = project_root / config["paths"]["skeletons"]
    annotation_root = project_root / config["paths"]["annotations"]
    processed_root = project_root / config["paths"]["processed"]
    ingest_root = processed_root / "ingest"
    logs_root = project_root / "logs"

    state_path = ingest_root / "pipeline_state.json"
    lock_path = ingest_root / "pipeline.lock"
    manifest_path = ingest_root / "pexels_manifest.jsonl"
    failed_path = ingest_root / "failed_downloads.jsonl"
    rejected_path = ingest_root / "quality_rejected.jsonl"

    ok_raw, msg_raw = validate_directory_path(raw_videos_root, "原始视频目录")
    if not ok_raw:
        print(f"错误: {msg_raw}")
        print("请先确认外接硬盘已挂载，再重试。")
        sys.exit(1)

    ok_skeleton, msg_skeleton = validate_directory_path(skeleton_root, "骨骼目录")
    if not ok_skeleton:
        print(f"错误: {msg_skeleton}")
        sys.exit(1)

    ok_annotation, msg_annotation = validate_directory_path(annotation_root, "标注目录")
    if not ok_annotation:
        print(f"错误: {msg_annotation}")
        sys.exit(1)

    lock_ok, lock_message = acquire_run_lock(lock_path)
    if not lock_ok:
        print(f"错误: {lock_message}")
        print("请先结束已有任务，或等待其完成后重试。")
        sys.exit(1)

    raw_videos_root.mkdir(parents=True, exist_ok=True)
    skeleton_root.mkdir(parents=True, exist_ok=True)
    annotation_root.mkdir(parents=True, exist_ok=True)
    ingest_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    target_per_action = int(
        args.target_per_action or run_policy.get("target_per_action", 120)
    )
    max_new_per_action = int(
        args.max_new_per_action or run_policy.get("max_new_per_action_per_run", 12)
    )
    max_new_total = int(
        args.max_new_total or run_policy.get("max_new_total_per_run", 72)
    )

    per_page = int(args.per_page or pexels_cfg.get("per_page", 15))
    max_pages = int(args.max_pages or pexels_cfg.get("max_pages_per_query", 4))
    orientation = str(pexels_cfg.get("orientation", "landscape"))
    locale = str(pexels_cfg.get("locale", "zh-CN"))
    interval_sec = float(args.interval or pexels_cfg.get("request_interval_sec", 2.2))
    jitter_sec = float(args.jitter or pexels_cfg.get("request_jitter_sec", 0.6))
    timeout_sec = int(args.timeout or pexels_cfg.get("timeout_sec", 30))

    min_width = int(quality_gate.get("min_width", 720))
    min_height = int(quality_gate.get("min_height", 480))
    min_duration = float(quality_gate.get("min_duration_sec", 3))
    max_duration = float(quality_gate.get("max_duration_sec", 30))
    min_filesize = int(quality_gate.get("min_filesize_bytes", 200000))
    pose_probe_cfg = quality_gate.get("pose_probe", {})
    pose_probe_enabled = bool(pose_probe_cfg.get("enabled", True)) and not bool(
        args.disable_pose_probe
    )
    pose_probe_strict = bool(pose_probe_cfg.get("strict", False))

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    known_video_ids = load_video_ids_from_jsonl(manifest_path)
    known_video_ids.update(load_video_ids_from_jsonl(rejected_path))

    actions_state: Dict[str, Dict[str, Any]] = {}
    downloaded_files_by_action: Dict[str, List[str]] = {
        action_id: [] for action_id in selected_actions
    }
    for action_id in selected_actions:
        action_dir = raw_videos_root / action_id
        action_dir.mkdir(parents=True, exist_ok=True)
        removed_sidecars = cleanup_sidecar_files(action_dir)
        local_count = count_local_videos(action_dir)
        actions_state[action_id] = {
            "name": all_actions[action_id].get("name", action_id),
            "existing_local": local_count,
            "target_total": target_per_action,
            "remaining_target": max(0, target_per_action - local_count),
            "run_quota": max(
                0, min(max_new_per_action, target_per_action - local_count)
            ),
            "added_this_run": 0,
            "skeleton_added_this_run": 0,
            "annotation_added_this_run": 0,
            "scanned": 0,
            "duplicate": 0,
            "quality_skipped": 0,
            "metadata_skipped": 0,
            "pose_skipped": 0,
            "failed": 0,
            "api_failures": 0,
            "pages_fetched": 0,
            "latest_query": "",
            "latest_page": 0,
            "crawl_status": "pending",
            "preprocess_status": "pending",
            "annotate_status": "pending",
        }
        if removed_sidecars > 0:
            actions_state[action_id]["cleanup_removed"] = removed_sidecars

    tracker = ProgressTracker(state_path)
    tracker.start(
        run_id=run_id,
        actions_state=actions_state,
        run_config={
            "actions": selected_actions,
            "target_per_action": target_per_action,
            "max_new_per_action": max_new_per_action,
            "max_new_total": max_new_total,
            "per_page": per_page,
            "max_pages": max_pages,
            "orientation": orientation,
            "locale": locale,
            "interval_sec": interval_sec,
            "jitter_sec": jitter_sec,
            "timeout_sec": timeout_sec,
            "insecure_ssl": bool(args.insecure_ssl),
            "skip_env_check": bool(args.skip_env_check),
            "pose_probe_enabled": bool(pose_probe_enabled),
            "pose_probe_strict": bool(pose_probe_strict),
            "pose_probe_sample_frames": int(
                args.pose_sample_frames or pose_probe_cfg.get("sample_frames", 12)
            ),
            "dry_run": bool(args.dry_run),
            "no_pipeline": bool(args.no_pipeline),
        },
    )

    for action_id in selected_actions:
        removed_sidecars = int(actions_state[action_id].get("cleanup_removed", 0))
        if removed_sidecars > 0:
            tracker.add_event(
                "info",
                f"已清理隐藏无效视频文件 {removed_sidecars} 个",
                action_id=action_id,
            )

    if not args.no_pipeline and not args.skip_env_check:
        tracker.set_stage("env_check", "检查预处理与标注依赖")
        ok, check_message = check_runtime_dependencies(args.python_bin)
        if not ok:
            friendly = (
                "环境检查失败，请先安装依赖后再运行。"
                f" 详情: {check_message}。"
                " 建议先执行: python3 -m venv .venv && source .venv/bin/activate && "
                "python -m pip install -r requirements.txt"
            )
            tracker.add_event("error", friendly)
            tracker.finish("failed", "环境检查失败", error_message=friendly)
            print(friendly)
            release_run_lock(lock_path)
            return
        tracker.add_event("info", "环境检查通过")

    pose_gate: Optional[PoseQualityGate] = None
    if pose_probe_enabled and not args.dry_run:
        pose_gate = PoseQualityGate(
            model_path=str(pose_probe_cfg.get("model", "yolov8x-pose.pt")),
            sample_frames=int(
                args.pose_sample_frames or pose_probe_cfg.get("sample_frames", 12)
            ),
            keypoint_conf_threshold=float(
                pose_probe_cfg.get("keypoint_conf_threshold", 0.25)
            ),
            min_person_frame_ratio=float(
                pose_probe_cfg.get("min_person_frame_ratio", 0.55)
            ),
            min_single_person_ratio=float(
                pose_probe_cfg.get("min_single_person_ratio", 0.55)
            ),
            min_avg_visible_keypoints=float(
                pose_probe_cfg.get("min_avg_visible_keypoints", 8)
            ),
            min_primary_person_area_ratio=float(
                pose_probe_cfg.get("min_primary_person_area_ratio", 0.02)
            ),
        )
        gate_ok, gate_error = pose_gate.initialize()
        if not gate_ok:
            message = f"姿态快速质检初始化失败: {gate_error}"
            if pose_probe_strict:
                tracker.add_event("error", message)
                tracker.finish("failed", "姿态快速质检初始化失败", error_message=message)
                print(message)
                release_run_lock(lock_path)
                return
            tracker.add_event("warn", message + "，将跳过姿态快速质检")
            pose_gate = None

    client = PexelsClient(
        api_key=api_key,
        base_url=str(
            pexels_cfg.get("base_url", "https://api.pexels.com/videos/search")
        ),
        timeout_sec=timeout_sec,
        insecure_ssl=bool(args.insecure_ssl),
    )

    total_added = 0
    actions_with_new_data: List[str] = []

    try:
        tracker.set_stage("crawl", "开始按动作采集视频")

        for action_id in selected_actions:
            action_state = actions_state[action_id]
            action_name = action_state["name"]
            quota = int(action_state["run_quota"])
            action_dir = raw_videos_root / action_id
            queries = action_queries(config, action_id)
            relevance_keywords = action_relevance_keywords(config, action_id, queries)

            if quota <= 0:
                tracker.update_action(action_id, crawl_status="skipped")
                tracker.add_event(
                    "info", f"{action_name} 已达到目标总量，跳过采集", action_id=action_id
                )
                continue

            tracker.update_action(action_id, crawl_status="running")
            tracker.add_event(
                "info", f"开始采集 {action_name}，本轮上限 {quota}", action_id=action_id
            )

            stop_action = False
            for query in queries:
                if stop_action:
                    break

                for page in range(1, max_pages + 1):
                    if total_added >= max_new_total:
                        stop_action = True
                        break
                    if int(action_state["added_this_run"]) >= quota:
                        stop_action = True
                        break

                    tracker.update_action(
                        action_id, latest_query=query, latest_page=page
                    )

                    try:
                        payload = client.search_videos(
                            query=query,
                            page=page,
                            per_page=per_page,
                            orientation=orientation,
                            locale=locale,
                        )
                    except Exception as exc:
                        tracker.incr_action(action_id, "api_failures", 1)
                        tracker.incr_summary("api_failure_total", 1)
                        tracker.add_event(
                            "warn",
                            f"API 请求失败: {exc}",
                            action_id=action_id,
                            extra={"query": query, "page": page},
                        )
                        sleep_with_jitter(interval_sec, jitter_sec)
                        continue

                    videos = payload.get("videos") or []
                    tracker.incr_action(action_id, "pages_fetched", 1)

                    if not videos:
                        tracker.add_event(
                            "info",
                            "当前关键词无更多结果",
                            action_id=action_id,
                            extra={"query": query, "page": page},
                        )
                        sleep_with_jitter(interval_sec, jitter_sec)
                        break

                    for video in videos:
                        if total_added >= max_new_total:
                            stop_action = True
                            break
                        if int(action_state["added_this_run"]) >= quota:
                            stop_action = True
                            break

                        tracker.incr_action(action_id, "scanned", 1)
                        tracker.incr_summary("scanned_total", 1)

                        video_id = str(video.get("id") or "")
                        if not video_id:
                            tracker.incr_action(action_id, "quality_skipped", 1)
                            tracker.incr_summary("quality_skipped_total", 1)
                            continue

                        if video_id in known_video_ids:
                            tracker.incr_action(action_id, "duplicate", 1)
                            tracker.incr_summary("duplicate_total", 1)
                            continue

                        video_url = str(video.get("url") or "")
                        url_path = parse.urlparse(video_url).path
                        if relevance_keywords and is_informative_url_path(url_path):
                            hits = keyword_hits(url_path, relevance_keywords)
                            if not hits:
                                tracker.incr_action(action_id, "quality_skipped", 1)
                                tracker.incr_action(action_id, "metadata_skipped", 1)
                                tracker.incr_summary("quality_skipped_total", 1)
                                tracker.incr_summary("metadata_skipped_total", 1)
                                append_jsonl(
                                    rejected_path,
                                    {
                                        "provider": "pexels",
                                        "run_id": run_id,
                                        "action": action_id,
                                        "query": query,
                                        "page": page,
                                        "pexels_video_id": video_id,
                                        "video_url": video_url,
                                        "video_path": url_path,
                                        "reason": "metadata_mismatch",
                                        "created_at": time.time(),
                                    },
                                )
                                known_video_ids.add(video_id)
                                continue

                        duration = float(video.get("duration") or 0)
                        if duration < min_duration or duration > max_duration:
                            tracker.incr_action(action_id, "quality_skipped", 1)
                            tracker.incr_summary("quality_skipped_total", 1)
                            continue

                        candidate = pick_video_candidate(
                            video,
                            min_width=min_width,
                            min_height=min_height,
                        )
                        if not candidate:
                            tracker.incr_action(action_id, "quality_skipped", 1)
                            tracker.incr_summary("quality_skipped_total", 1)
                            continue

                        suffix = guess_suffix(candidate["file_type"], candidate["link"])
                        output_name = f"pexels_{video_id}{suffix}"
                        output_path = action_dir / output_name

                        if output_path.exists():
                            known_video_ids.add(video_id)
                            tracker.incr_action(action_id, "duplicate", 1)
                            tracker.incr_summary("duplicate_total", 1)
                            continue

                        if args.dry_run:
                            tracker.incr_action(action_id, "added_this_run", 1)
                            tracker.incr_summary("added_total", 1)
                            total_added += 1
                            tracker.add_event(
                                "info",
                                f"[dry-run] 计划下载 {output_name}",
                                action_id=action_id,
                                extra={"query": query, "page": page},
                            )
                            continue

                        try:
                            file_size = download_file(
                                candidate["link"],
                                output_path,
                                timeout_sec,
                                bool(args.insecure_ssl),
                            )
                            if file_size < min_filesize:
                                if output_path.exists():
                                    output_path.unlink()
                                raise RuntimeError(
                                    f"文件过小 ({file_size} bytes)，低于阈值 {min_filesize}"
                                )

                            if pose_gate is not None:
                                passed, pose_metrics, pose_reason = pose_gate.evaluate(
                                    output_path
                                )
                                if not passed:
                                    if output_path.exists():
                                        output_path.unlink()
                                    tracker.incr_action(action_id, "quality_skipped", 1)
                                    tracker.incr_action(action_id, "pose_skipped", 1)
                                    tracker.incr_summary("quality_skipped_total", 1)
                                    tracker.incr_summary("pose_skipped_total", 1)
                                    append_jsonl(
                                        rejected_path,
                                        {
                                            "provider": "pexels",
                                            "run_id": run_id,
                                            "action": action_id,
                                            "query": query,
                                            "page": page,
                                            "pexels_video_id": video_id,
                                            "video_url": video_url,
                                            "reason": "pose_probe_reject",
                                            "detail": pose_reason,
                                            "metrics": pose_metrics,
                                            "created_at": time.time(),
                                        },
                                    )
                                    tracker.add_event(
                                        "warn",
                                        f"姿态快速质检未通过: {pose_reason}",
                                        action_id=action_id,
                                        extra={"query": query, "page": page},
                                    )
                                    known_video_ids.add(video_id)
                                    continue

                            known_video_ids.add(video_id)
                            tracker.incr_action(action_id, "added_this_run", 1)
                            tracker.incr_summary("added_total", 1)
                            total_added += 1
                            downloaded_files_by_action[action_id].append(output_name)

                            append_jsonl(
                                manifest_path,
                                {
                                    "provider": "pexels",
                                    "run_id": run_id,
                                    "action": action_id,
                                    "action_name": action_name,
                                    "query": query,
                                    "page": page,
                                    "pexels_video_id": video_id,
                                    "duration": duration,
                                    "width": candidate["width"],
                                    "height": candidate["height"],
                                    "download_url": candidate["link"],
                                    "filename": output_name,
                                    "saved_path": str(
                                        output_path.relative_to(project_root)
                                    ),
                                    "file_size": file_size,
                                    "created_at": time.time(),
                                },
                            )

                            tracker.add_event(
                                "info",
                                f"下载完成 {output_name}",
                                action_id=action_id,
                                extra={
                                    "query": query,
                                    "page": page,
                                    "file_size": file_size,
                                },
                            )
                        except Exception as exc:
                            if output_path.exists():
                                output_path.unlink()
                            tracker.incr_action(action_id, "failed", 1)
                            tracker.incr_summary("failed_total", 1)
                            append_jsonl(
                                failed_path,
                                {
                                    "provider": "pexels",
                                    "run_id": run_id,
                                    "action": action_id,
                                    "query": query,
                                    "page": page,
                                    "pexels_video_id": video_id,
                                    "error": str(exc),
                                    "created_at": time.time(),
                                },
                            )
                            tracker.add_event(
                                "warn",
                                f"下载失败 video_id={video_id}: {exc}",
                                action_id=action_id,
                                extra={"query": query, "page": page},
                            )

                    sleep_with_jitter(interval_sec, jitter_sec)

            final_count = count_local_videos(action_dir)
            tracker.update_action(
                action_id,
                crawl_status="done",
                existing_local=final_count,
                remaining_target=max(0, target_per_action - final_count),
            )

            if int(actions_state[action_id]["added_this_run"]) > 0:
                actions_with_new_data.append(action_id)

        tracker.set_stage("deploy", "生成部署与巡检脚本")
        tracker.set_pipeline_status("deploy", "running")
        generate_deploy_assets(project_root)
        tracker.set_pipeline_status("deploy", "done")
        tracker.add_event("info", "已更新 deploy/ingest 下的运行与监控脚本")

        if args.auto_tag_cleanup:
            tracker.set_stage("tag_cleanup", "执行视频打标签与垃圾清理")
            cleanup_cmd = [
                args.python_bin,
                "9_tag_and_cleanup_videos.py",
                "--actions",
                ",".join(selected_actions),
                "--cleanup",
            ]
            if args.auto_tag_cleanup_dry_run:
                cleanup_cmd.append("--dry_run")

            cleanup_log = logs_root / "ingest_pipeline.log"
            with open(cleanup_log, "a", encoding="utf-8") as f:
                f.write(
                    "\n\n===== "
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
                    "9_tag_and_cleanup_videos.py =====\n"
                )
                proc = subprocess.run(
                    cleanup_cmd,
                    cwd=str(project_root),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                )
                if proc.stdout:
                    f.write(proc.stdout)
                if proc.stderr:
                    f.write(proc.stderr)

            if proc.returncode == 0:
                tracker.add_event(
                    "info",
                    "已执行视频打标签与垃圾清理",
                    extra={"dry_run": bool(args.auto_tag_cleanup_dry_run)},
                )
            else:
                tracker.add_event(
                    "warn",
                    f"视频打标签/清理执行失败，退出码 {proc.returncode}",
                )

        if args.no_pipeline:
            tracker.set_pipeline_status("preprocess", "skipped")
            tracker.set_pipeline_status("annotate", "skipped")
            for action_id in selected_actions:
                tracker.update_action(
                    action_id,
                    preprocess_status="skipped",
                    annotate_status="skipped",
                )
            tracker.finish("completed", "采集完成（已按参数跳过后续 pipeline）")
            print("\n完成: 仅采集模式已执行，后续 pipeline 已跳过")
            return

        if not actions_with_new_data:
            tracker.set_pipeline_status("preprocess", "skipped")
            tracker.set_pipeline_status("annotate", "skipped")
            for action_id in selected_actions:
                tracker.update_action(
                    action_id,
                    preprocess_status="skipped",
                    annotate_status="skipped",
                )
            tracker.finish("completed", "没有新增视频，已跳过预处理与标注")
            print("\n完成: 本轮没有新增视频，未触发预处理/自动标注")
            return

        tracker.set_stage("preprocess", "开始执行 0_preprocess_videos.py")
        tracker.set_pipeline_status("preprocess", "running")
        preprocess_failed: List[str] = []
        for action_id in actions_with_new_data:
            tracker.update_action(action_id, preprocess_status="running")
            tracker.add_event("info", "开始预处理", action_id=action_id)
            before_count = count_json_files(skeleton_root / action_id)
            only_files = downloaded_files_by_action.get(action_id, [])
            preprocess_args: List[str] = []
            if only_files:
                preprocess_args.extend(["--only_files", ",".join(only_files)])
            code = run_python_script(
                project_root=project_root,
                python_bin=args.python_bin,
                script_name="0_preprocess_videos.py",
                action_id=action_id,
                log_path=logs_root / "ingest_pipeline.log",
                extra_args=preprocess_args,
            )
            after_count = count_json_files(skeleton_root / action_id)
            produced = max(0, after_count - before_count)
            tracker.update_action(action_id, skeleton_added_this_run=produced)
            if produced > 0:
                tracker.incr_summary("skeleton_added_total", produced)

            if code == 0 and produced > 0:
                tracker.update_action(action_id, preprocess_status="done")
                tracker.add_event(
                    "info",
                    f"预处理完成，新增骨骼 {produced} 条",
                    action_id=action_id,
                )
            elif code == 0:
                preprocess_failed.append(action_id)
                tracker.update_action(action_id, preprocess_status="failed")
                tracker.add_event(
                    "error",
                    "预处理未产出骨骼文件，请提高采集质量门槛",
                    action_id=action_id,
                )
            else:
                preprocess_failed.append(action_id)
                tracker.update_action(action_id, preprocess_status="failed")
                tracker.add_event(
                    "error",
                    f"预处理失败，退出码 {code}",
                    action_id=action_id,
                )

        tracker.set_pipeline_status(
            "preprocess", "done" if not preprocess_failed else "partial"
        )

        tracker.set_stage("annotate", "开始执行 1_auto_annotate.py")
        annotate_failed: List[str] = []
        annotate_targets = [
            a for a in actions_with_new_data if a not in preprocess_failed
        ]

        for action_id in actions_with_new_data:
            if action_id in preprocess_failed:
                tracker.update_action(action_id, annotate_status="skipped")

        if not annotate_targets:
            tracker.set_pipeline_status("annotate", "skipped")
        else:
            tracker.set_pipeline_status("annotate", "running")
            for action_id in annotate_targets:
                tracker.update_action(action_id, annotate_status="running")
                tracker.add_event("info", "开始自动标注", action_id=action_id)
                before_count = count_json_files(annotation_root / action_id)
                annotate_files = [
                    Path(name).stem
                    for name in downloaded_files_by_action.get(action_id, [])
                    if name
                ]
                annotate_args: List[str] = []
                if annotate_files:
                    annotate_args.extend(["--only_files", ",".join(annotate_files)])
                code = run_python_script(
                    project_root=project_root,
                    python_bin=args.python_bin,
                    script_name="1_auto_annotate.py",
                    action_id=action_id,
                    log_path=logs_root / "ingest_pipeline.log",
                    extra_args=annotate_args,
                )
                after_count = count_json_files(annotation_root / action_id)
                produced = max(0, after_count - before_count)
                tracker.update_action(action_id, annotation_added_this_run=produced)
                if produced > 0:
                    tracker.incr_summary("annotation_added_total", produced)

                if code == 0 and produced > 0:
                    tracker.update_action(action_id, annotate_status="done")
                    tracker.add_event(
                        "info",
                        f"自动标注完成，新增标注 {produced} 条",
                        action_id=action_id,
                    )
                elif code == 0:
                    annotate_failed.append(action_id)
                    tracker.update_action(action_id, annotate_status="failed")
                    tracker.add_event(
                        "error",
                        "自动标注未产出新文件，请检查骨骼输入或重跑",
                        action_id=action_id,
                    )
                else:
                    annotate_failed.append(action_id)
                    tracker.update_action(action_id, annotate_status="failed")
                    tracker.add_event(
                        "error",
                        f"自动标注失败，退出码 {code}",
                        action_id=action_id,
                    )

            tracker.set_pipeline_status(
                "annotate", "done" if not annotate_failed else "partial"
            )

        if preprocess_failed or annotate_failed:
            parts: List[str] = []
            if preprocess_failed:
                parts.append(f"预处理失败: {', '.join(sorted(preprocess_failed))}")
            if annotate_failed:
                parts.append(f"自动标注失败: {', '.join(sorted(annotate_failed))}")
            message = "流程完成，但存在失败阶段；" + "；".join(parts)
            tracker.finish("partial", message, error_message=message)
            print(f"\n{message}")
        else:
            tracker.finish("completed", "采集与自动载入流水线全部完成")
            print("\n完成: 采集与自动载入流水线全部完成")

        print(f"状态文件: {state_path}")
        print("监控命令: python 8_ingest_monitor.py --watch")

    except KeyboardInterrupt:
        tracker.finish("cancelled", "用户中断执行", error_message="KeyboardInterrupt")
        print("\n已中断。")
        sys.exit(130)
    except Exception as exc:
        tracker.add_event("error", f"未处理异常: {exc}")
        tracker.finish("failed", "执行失败", error_message=str(exc))
        print(f"\n执行失败: {exc}")
        raise
    finally:
        release_run_lock(lock_path)


if __name__ == "__main__":
    main()
