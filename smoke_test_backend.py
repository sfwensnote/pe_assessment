#!/usr/bin/env python3
"""Backend API smoke test for the current mixed-best model bundle."""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any

import cv2
from fastapi.testclient import TestClient

from app.main import app


PROJECT_ROOT = Path(__file__).parent
VIDEO_PATH = PROJECT_ROOT / "data/raw_videos/squat/pexels_6180021.mp4"
REPORT_PATH = PROJECT_ROOT / "data/processed/validation/backend_smoke.json"


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def to_data_url(frame_bgr) -> str:
    ok, encoded = cv2.imencode(".jpg", frame_bgr)
    ensure(ok, "Failed to encode websocket frame")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode(
        "ascii"
    )


def read_frames(video_path: Path, limit: int) -> list[str]:
    cap = cv2.VideoCapture(str(video_path))
    ensure(cap.isOpened(), f"Unable to open sample video: {video_path}")
    payloads: list[str] = []
    while len(payloads) < limit:
        ret, frame = cap.read()
        if not ret:
            break
        payloads.append(to_data_url(frame))
    cap.release()
    ensure(payloads, "No frames extracted from sample video")
    return payloads


def request_json(client: TestClient, method: str, url: str, **kwargs) -> dict[str, Any]:
    response = client.request(method, url, **kwargs)
    ensure(
        response.status_code == 200,
        f"{method} {url} failed: {response.status_code} {response.text}",
    )
    return response.json()


def main() -> int:
    ensure(VIDEO_PATH.exists(), f"Missing sample video: {VIDEO_PATH}")
    frame_payloads = read_frames(VIDEO_PATH, limit=25)
    report: dict[str, Any] = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "video": str(VIDEO_PATH),
    }

    with TestClient(app) as client:
        report["root"] = request_json(client, "GET", "/")
        report["health"] = request_json(client, "GET", "/api/health")
        report["system_overview"] = request_json(client, "GET", "/api/system/overview")
        report["monitor_live"] = request_json(client, "GET", "/api/monitor/live")
        report["actions"] = request_json(client, "GET", "/api/actions")
        report["admin_overview"] = request_json(client, "GET", "/api/admin/overview")
        report["admin_video_tasks_initial"] = request_json(
            client, "GET", "/api/admin/video_tasks"
        )
        report["admin_reports_initial"] = request_json(
            client, "GET", "/api/admin/realtime_reports?limit=5"
        )
        report["admin_ingest"] = request_json(
            client, "GET", "/api/admin/ingest/overview"
        )

        ensure(report["health"].get("ok") is True, "Health endpoint not ok")
        ensure(report["health"]["models"]["action"] is True, "Action model not ready")
        ensure(report["health"]["models"]["quality"] is True, "Quality model not ready")
        ensure(
            report["health"]["models"]["phase_model_count"] >= 1,
            "Phase models not ready",
        )

        session_payload = {
            "action_type": "squat",
            "target_reps": 3,
            "window_size": 20,
            "infer_interval": 1,
        }
        start_resp = client.post("/api/realtime/session/start", json=session_payload)
        ensure(
            start_resp.status_code == 200,
            f"Realtime session start failed: {start_resp.text}",
        )
        session_data = start_resp.json()
        session_id = session_data["session_id"]
        ws_url = session_data["ws_url"]
        report["realtime_start"] = session_data

        ws_results: list[dict[str, Any]] = []
        with client.websocket_connect(ws_url) as websocket:
            websocket.send_json({"type": "ping"})
            ws_results.append(websocket.receive_json())

            for payload in frame_payloads:
                websocket.send_json({"image_base64": payload})
                result = websocket.receive_json()
                ws_results.append(result)
                if result.get("status") in {"ok", "cached"}:
                    break

        report["realtime_messages"] = ws_results
        ensure(
            any(item.get("type") == "pong" for item in ws_results),
            "Realtime ping/pong failed",
        )
        ensure(
            any(item.get("status") in {"ok", "cached"} for item in ws_results),
            "Realtime inference never reached ok/cached",
        )

        stop_resp = client.post(f"/api/realtime/session/{session_id}/stop")
        ensure(stop_resp.status_code == 200, f"Realtime stop failed: {stop_resp.text}")
        report["realtime_stop"] = stop_resp.json()

        report["realtime_report_get"] = request_json(
            client, "GET", f"/api/reports/{session_id}"
        )

        with open(VIDEO_PATH, "rb") as f:
            files = {"file": (VIDEO_PATH.name, f, "video/mp4")}
            sync_resp = client.post("/api/inference/video", files=files)
        ensure(
            sync_resp.status_code == 200,
            f"Sync video inference failed: {sync_resp.text}",
        )
        report["sync_video_inference"] = sync_resp.json()
        ensure(
            report["sync_video_inference"].get("ok") is True,
            "Sync video inference did not return ok=true",
        )

        with open(VIDEO_PATH, "rb") as f:
            files = {"file": (VIDEO_PATH.name, f, "video/mp4")}
            async_resp = client.post("/api/inference/video/tasks", files=files)
        ensure(
            async_resp.status_code == 200,
            f"Async task creation failed: {async_resp.text}",
        )
        task_payload = async_resp.json()["task"]
        task_id = task_payload["task_id"]

        final_task = task_payload
        for _ in range(120):
            task_detail = request_json(
                client, "GET", f"/api/inference/video/tasks/{task_id}"
            )
            final_task = task_detail["task"]
            if final_task["status"] in {"completed", "failed"}:
                break
            time.sleep(1)
        report["async_task_final"] = final_task
        ensure(
            final_task["status"] == "completed",
            f"Async task not completed: {final_task}",
        )

        report["async_task_list"] = request_json(
            client, "GET", "/api/inference/video/tasks"
        )
        delete_resp = client.delete(f"/api/inference/video/tasks/{task_id}")
        ensure(
            delete_resp.status_code == 200,
            f"Async task delete failed: {delete_resp.text}",
        )
        report["async_task_delete"] = delete_resp.json()

        clear_resp = client.delete("/api/inference/video/history")
        ensure(
            clear_resp.status_code == 200,
            f"Clear video history failed: {clear_resp.text}",
        )
        report["async_history_clear"] = clear_resp.json()

        report["admin_video_tasks_final"] = request_json(
            client, "GET", "/api/admin/video_tasks"
        )
        report["admin_reports_final"] = request_json(
            client, "GET", "/api/admin/realtime_reports?limit=5"
        )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("=" * 72)
    print("Backend smoke test passed")
    print("=" * 72)
    print(f"report_file : {REPORT_PATH}")
    print(f"health_ok   : {report['health']['ok']}")
    print(f"sync_action : {report['sync_video_inference'].get('action_type')}")
    print(f"sync_score  : {report['sync_video_inference'].get('overall_score')}")
    print(f"async_task  : {report['async_task_final']['status']}")
    print(
        f"realtime_ok : {any(item.get('status') in {'ok', 'cached'} for item in report['realtime_messages'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
