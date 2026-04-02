#!/usr/bin/env python3
"""Run Step3 auto-annotation with adaptive parallelism and state tracking."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
STATE_PATH = PROJECT_ROOT / "data/processed/annotate/pipeline_state.json"
FAILED_PATH = PROJECT_ROOT / "data/processed/annotate/failed_files.jsonl"
LOG_DIR = PROJECT_ROOT / "logs"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)


@dataclass(frozen=True)
class ChunkTask:
    action: str
    files: list[str]
    index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Step3 auto-annotation")
    parser.add_argument(
        "--actions",
        type=str,
        default="all",
        help="Comma-separated actions, default: all",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel workers, 0 means auto",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=0,
        help="Files per subprocess chunk, 0 means auto",
    )
    parser.add_argument(
        "--python_bin",
        type=str,
        default=sys.executable,
        help="Python interpreter used for child jobs",
    )
    parser.add_argument(
        "--skeleton_dir",
        type=str,
        default=CONFIG["paths"]["skeletons"],
        help="Skeleton input directory",
    )
    parser.add_argument(
        "--anno_dir",
        type=str,
        default=CONFIG["paths"]["annotations"],
        help="Annotation output directory",
    )
    parser.add_argument(
        "--poll_interval",
        type=float,
        default=2.0,
        help="State refresh interval in seconds",
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="Plan only, do not run jobs"
    )
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {}
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def save_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def append_event(
    state: dict[str, Any], level: str, message: str, action: str | None = None
) -> None:
    events = state.setdefault("events", [])
    events.append(
        {
            "time": now_iso(),
            "level": level,
            "action": action,
            "message": message,
        }
    )
    if len(events) > 100:
        del events[:-100]


def append_failed(action: str, file_name: str, reason: str) -> None:
    FAILED_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "time": now_iso(),
        "action": action,
        "file": file_name,
        "reason": reason,
    }
    with open(FAILED_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def recommended_workers() -> int:
    cpu_count = os.cpu_count() or 4
    memory_gb = None
    try:
        import psutil  # type: ignore

        memory_gb = psutil.virtual_memory().total / (1024**3)
    except Exception:
        memory_gb = None

    workers = max(2, cpu_count // 2)
    workers = min(4, workers)
    if memory_gb is not None:
        if memory_gb < 12:
            workers = min(workers, 2)
        elif memory_gb < 24:
            workers = min(workers, 3)
    return workers


def recommended_chunk_size(workers: int) -> int:
    if workers >= 4:
        return 75
    if workers == 3:
        return 90
    return 120


def resolve_actions(raw: str) -> list[str]:
    all_actions = list(CONFIG["actions"].keys())
    if raw == "all":
        return all_actions
    values = [item.strip() for item in raw.split(",") if item.strip()]
    invalid = [item for item in values if item not in all_actions]
    if invalid:
        raise ValueError(f"Unknown actions: {', '.join(invalid)}")
    return values


def list_pending_files(
    action: str, skeleton_dir: Path, anno_dir: Path
) -> tuple[list[str], int, int]:
    action_skeleton_dir = skeleton_dir / action
    action_anno_dir = anno_dir / action
    skeleton_files = (
        sorted(p.name for p in action_skeleton_dir.glob("*.json"))
        if action_skeleton_dir.exists()
        else []
    )
    anno_files = (
        {p.name for p in action_anno_dir.glob("*.json")}
        if action_anno_dir.exists()
        else set()
    )
    pending = [name for name in skeleton_files if name not in anno_files]
    return pending, len(skeleton_files), len(anno_files)


def chunk_files(files: list[str], chunk_size: int) -> list[list[str]]:
    return [files[i : i + chunk_size] for i in range(0, len(files), chunk_size)]


def build_state(
    actions: list[str],
    skeleton_dir: Path,
    anno_dir: Path,
    workers: int,
    chunk_size: int,
) -> tuple[dict[str, Any], list[ChunkTask]]:
    state = {
        "run_id": f"step3-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "status": "pending",
        "stage": "step3_auto_annotate",
        "message": "Preparing tasks",
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "workers": workers,
        "chunk_size": chunk_size,
        "summary": {
            "skeleton_total": 0,
            "annotation_existing": 0,
            "annotation_added_this_run": 0,
            "remaining": 0,
            "chunks_total": 0,
            "chunks_done": 0,
            "failed_total": 0,
            "rate_files_per_min": 0.0,
        },
        "actions": {},
        "events": [],
    }
    tasks: list[ChunkTask] = []

    for action in actions:
        pending_files, skeleton_total, anno_existing = list_pending_files(
            action, skeleton_dir, anno_dir
        )
        chunks = chunk_files(pending_files, chunk_size)
        state["actions"][action] = {
            "name": CONFIG["actions"][action]["name"],
            "skeleton_total": skeleton_total,
            "annotation_existing": anno_existing,
            "annotation_current": anno_existing,
            "remaining": max(0, skeleton_total - anno_existing),
            "pending_files": len(pending_files),
            "chunks_total": len(chunks),
            "chunks_done": 0,
            "status": "completed"
            if not pending_files and skeleton_total > 0
            else "pending",
            "started_at": None,
            "completed_at": None,
            "last_error": "",
        }
        state["summary"]["skeleton_total"] += skeleton_total
        state["summary"]["annotation_existing"] += anno_existing
        state["summary"]["remaining"] += max(0, skeleton_total - anno_existing)
        state["summary"]["chunks_total"] += len(chunks)

        for index, files in enumerate(chunks, start=1):
            tasks.append(ChunkTask(action=action, files=files, index=index))

    append_event(state, "info", "Prepared Step3 task plan")
    return state, tasks


def refresh_counts(
    state: dict[str, Any], skeleton_dir: Path, anno_dir: Path, started_ts: float
) -> None:
    total_current = 0
    total_failed = 0
    for action, payload in state["actions"].items():
        _, skeleton_total, anno_current = list_pending_files(
            action, skeleton_dir, anno_dir
        )
        payload["annotation_current"] = anno_current
        payload["remaining"] = max(0, skeleton_total - anno_current)
        if (
            payload["remaining"] == 0
            and skeleton_total > 0
            and payload["chunks_total"] >= 0
        ):
            payload["status"] = "completed"
            payload["completed_at"] = payload.get("completed_at") or now_iso()
        total_current += anno_current
        if payload.get("last_error"):
            total_failed += 1

    state["summary"]["annotation_added_this_run"] = max(
        0, total_current - state["summary"]["annotation_existing"]
    )
    state["summary"]["remaining"] = max(
        0, state["summary"]["skeleton_total"] - total_current
    )
    state["summary"]["failed_total"] = total_failed
    elapsed_minutes = max((time.time() - started_ts) / 60.0, 1e-6)
    state["summary"]["rate_files_per_min"] = round(
        state["summary"]["annotation_added_this_run"] / elapsed_minutes, 2
    )
    state["updated_at"] = now_iso()


def run_chunk(
    task: ChunkTask,
    python_bin: str,
    skeleton_dir: Path,
    anno_dir: Path,
) -> dict[str, Any]:
    log_path = LOG_DIR / f"annotate_{task.action}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    only_files = ",".join(task.files)
    cmd = [
        python_bin,
        "1_auto_annotate.py",
        "--action",
        task.action,
        "--skeleton_dir",
        str(skeleton_dir),
        "--anno_dir",
        str(anno_dir),
        "--only_files",
        only_files,
    ]

    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(
            f"\n[{now_iso()}] chunk {task.index} start, files={len(task.files)}\n"
        )
        log_file.flush()
        completed = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
        )
        log_file.write(
            f"[{now_iso()}] chunk {task.index} end rc={completed.returncode}\n"
        )

    action_anno_dir = anno_dir / task.action
    missing_outputs = [
        name for name in task.files if not (action_anno_dir / name).exists()
    ]
    return {
        "action": task.action,
        "chunk_index": task.index,
        "returncode": completed.returncode,
        "file_count": len(task.files),
        "missing_outputs": missing_outputs,
    }


def main() -> int:
    args = parse_args()
    actions = resolve_actions(args.actions)
    skeleton_dir = PROJECT_ROOT / args.skeleton_dir
    anno_dir = PROJECT_ROOT / args.anno_dir
    workers = args.workers if args.workers > 0 else recommended_workers()
    chunk_size = (
        args.chunk_size if args.chunk_size > 0 else recommended_chunk_size(workers)
    )

    state, tasks = build_state(actions, skeleton_dir, anno_dir, workers, chunk_size)
    state["message"] = f"Planned {len(tasks)} chunk(s) across {len(actions)} action(s)"
    save_state(state)

    print("=" * 72)
    print("Step3 auto-annotation launcher")
    print("=" * 72)
    print(f"actions      : {', '.join(actions)}")
    print(f"workers      : {workers}")
    print(f"chunk_size   : {chunk_size}")
    print(f"skeleton_dir : {skeleton_dir}")
    print(f"anno_dir     : {anno_dir}")
    print(f"tasks        : {len(tasks)}")
    print(f"state_file   : {STATE_PATH}")
    print("monitor      : python monitor_step3.py --watch")

    if args.dry_run:
        state["status"] = "pending"
        state["message"] = "Dry run complete"
        save_state(state)
        print("Dry run only, no jobs started.")
        return 0

    if not tasks:
        state["status"] = "completed"
        state["message"] = "No pending files, Step3 already complete"
        state["updated_at"] = now_iso()
        save_state(state)
        print("No pending files found.")
        return 0

    started_ts = time.time()
    state["status"] = "running"
    state["message"] = "Running Step3 chunks"
    append_event(state, "info", "Step3 run started")
    save_state(state)

    future_map = {}
    pending_tasks = list(tasks)
    running_actions: set[str] = set()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        while pending_tasks or future_map:
            while pending_tasks and len(future_map) < workers:
                task = pending_tasks.pop(0)
                action_state = state["actions"][task.action]
                action_state["status"] = "running"
                action_state["started_at"] = action_state.get("started_at") or now_iso()
                running_actions.add(task.action)
                future = executor.submit(
                    run_chunk, task, args.python_bin, skeleton_dir, anno_dir
                )
                future_map[future] = task
                append_event(
                    state,
                    "info",
                    f"Started chunk {task.index} with {len(task.files)} file(s)",
                    task.action,
                )
                save_state(state)

            done, _ = wait(
                list(future_map.keys()),
                timeout=args.poll_interval,
                return_when=FIRST_COMPLETED,
            )

            for future in done:
                task = future_map.pop(future)
                action_state = state["actions"][task.action]
                try:
                    result = future.result()
                    action_state["chunks_done"] += 1
                    state["summary"]["chunks_done"] += 1
                    missing_outputs = result["missing_outputs"]
                    if result["returncode"] != 0:
                        action_state["last_error"] = (
                            f"chunk {task.index} rc={result['returncode']}"
                        )
                        append_event(
                            state,
                            "error",
                            f"Chunk {task.index} failed with rc={result['returncode']}",
                            task.action,
                        )
                    else:
                        append_event(
                            state,
                            "info",
                            f"Chunk {task.index} finished",
                            task.action,
                        )

                    for file_name in missing_outputs:
                        append_failed(
                            task.action,
                            file_name,
                            "missing annotation output after chunk run",
                        )

                    if not any(
                        item.action == task.action for item in pending_tasks
                    ) and not any(
                        item.action == task.action for item in future_map.values()
                    ):
                        running_actions.discard(task.action)
                        if action_state["remaining"] == 0:
                            action_state["status"] = "completed"
                            action_state["completed_at"] = now_iso()
                        elif action_state.get("last_error"):
                            action_state["status"] = "partial"
                        else:
                            action_state["status"] = "pending"
                except Exception as exc:
                    action_state["chunks_done"] += 1
                    state["summary"]["chunks_done"] += 1
                    action_state["last_error"] = str(exc)
                    append_event(
                        state,
                        "error",
                        f"Chunk {task.index} crashed: {exc}",
                        task.action,
                    )

            refresh_counts(state, skeleton_dir, anno_dir, started_ts)
            if state["summary"]["remaining"] == 0:
                state["status"] = "completed"
                state["message"] = "Step3 completed"
            else:
                state["status"] = "running"
                state["message"] = (
                    f"Running, remaining files: {state['summary']['remaining']}"
                )
            save_state(state)

    refresh_counts(state, skeleton_dir, anno_dir, started_ts)
    state["status"] = "completed" if state["summary"]["remaining"] == 0 else "partial"
    state["message"] = (
        "Step3 completed"
        if state["status"] == "completed"
        else "Step3 finished with remaining files"
    )
    append_event(state, "info", state["message"])
    save_state(state)

    print("\nRun finished.")
    print(f"status       : {state['status']}")
    print(f"added_this_run: {state['summary']['annotation_added_this_run']}")
    print(f"remaining    : {state['summary']['remaining']}")
    print(f"rate/min     : {state['summary']['rate_files_per_min']}")
    return 0 if state["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
