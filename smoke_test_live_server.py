#!/usr/bin/env python3
"""Smoke-test a running backend server over real HTTP/WebSocket transport."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import time
from pathlib import Path
from typing import Any

import cv2
import httpx
import websockets


PROJECT_ROOT = Path(__file__).parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test a running backend server")
    parser.add_argument("--base_url", type=str, default="http://127.0.0.1:8001")
    parser.add_argument(
        "--video",
        type=str,
        default="data/raw_videos/squat/pexels_6180021.mp4",
        help="Sample video used for HTTP/WebSocket checks",
    )
    parser.add_argument(
        "--report_file",
        type=str,
        default="data/processed/validation/live_server_smoke.json",
        help="Output report path",
    )
    return parser.parse_args()


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path_str


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


async def run_websocket(
    base_url: str, session_id: str, frames: list[str]
) -> list[dict[str, Any]]:
    ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
    ws_url = f"{ws_url}/ws/realtime/{session_id}"
    messages: list[dict[str, Any]] = []
    async with websockets.connect(ws_url, max_size=8 * 1024 * 1024) as websocket:
        await websocket.send(json.dumps({"type": "ping"}))
        messages.append(json.loads(await websocket.recv()))

        for payload in frames:
            await websocket.send(json.dumps({"image_base64": payload}))
            result = json.loads(await websocket.recv())
            messages.append(result)
            if result.get("status") in {"ok", "cached"}:
                break
    return messages


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    video_path = resolve(args.video)
    report_path = resolve(args.report_file)
    ensure(video_path.exists(), f"Missing sample video: {video_path}")

    frame_payloads = read_frames(video_path, limit=25)
    report: dict[str, Any] = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_url": base_url,
        "video": str(video_path),
    }

    with httpx.Client(base_url=base_url, timeout=120.0) as client:
        for endpoint in [
            "/",
            "/api/health",
            "/api/system/overview",
            "/api/monitor/live",
            "/api/actions",
            "/api/admin/overview",
            "/api/admin/video_tasks",
            "/api/admin/realtime_reports?limit=5",
            "/api/admin/ingest/overview",
        ]:
            response = client.get(endpoint)
            ensure(
                response.status_code == 200,
                f"GET {endpoint} failed: {response.status_code} {response.text}",
            )
            report[endpoint] = response.json()

        start_resp = client.post(
            "/api/realtime/session/start",
            json={
                "action_type": "squat",
                "target_reps": 3,
                "window_size": 20,
                "infer_interval": 1,
            },
        )
        ensure(
            start_resp.status_code == 200,
            f"Realtime session start failed: {start_resp.text}",
        )
        session_data = start_resp.json()
        report["realtime_start"] = session_data

        ws_messages = asyncio.run(
            run_websocket(base_url, session_data["session_id"], frame_payloads)
        )
        report["realtime_messages"] = ws_messages
        ensure(
            any(msg.get("type") == "pong" for msg in ws_messages),
            "Realtime ping/pong failed",
        )
        ensure(
            any(msg.get("status") in {"ok", "cached"} for msg in ws_messages),
            "Realtime websocket never reached ok/cached",
        )

        stop_resp = client.post(
            f"/api/realtime/session/{session_data['session_id']}/stop"
        )
        ensure(stop_resp.status_code == 200, f"Realtime stop failed: {stop_resp.text}")
        report["realtime_stop"] = stop_resp.json()

        report["realtime_report_get"] = client.get(
            f"/api/reports/{session_data['session_id']}"
        ).json()

        with open(video_path, "rb") as f:
            sync_resp = client.post(
                "/api/inference/video",
                files={"file": (video_path.name, f, "video/mp4")},
            )
        ensure(
            sync_resp.status_code == 200,
            f"Sync video inference failed: {sync_resp.text}",
        )
        report["sync_video_inference"] = sync_resp.json()
        ensure(
            report["sync_video_inference"].get("ok") is True,
            "Sync inference did not return ok=true",
        )

        with open(video_path, "rb") as f:
            async_resp = client.post(
                "/api/inference/video/tasks",
                files={"file": (video_path.name, f, "video/mp4")},
            )
        ensure(
            async_resp.status_code == 200,
            f"Async task create failed: {async_resp.text}",
        )
        task = async_resp.json()["task"]
        report["async_task_created"] = task

        final_task = task
        for _ in range(120):
            poll_resp = client.get(f"/api/inference/video/tasks/{task['task_id']}")
            ensure(
                poll_resp.status_code == 200,
                f"Async task poll failed: {poll_resp.text}",
            )
            final_task = poll_resp.json()["task"]
            if final_task["status"] in {"completed", "failed"}:
                break
            time.sleep(1)
        report["async_task_final"] = final_task
        ensure(
            final_task["status"] == "completed",
            f"Async task ended in unexpected status: {final_task}",
        )

        delete_resp = client.delete(f"/api/inference/video/tasks/{task['task_id']}")
        ensure(
            delete_resp.status_code == 200,
            f"Async task delete failed: {delete_resp.text}",
        )
        report["async_task_delete"] = delete_resp.json()

        clear_resp = client.delete("/api/inference/video/history")
        ensure(
            clear_resp.status_code == 200, f"History clear failed: {clear_resp.text}"
        )
        report["async_history_clear"] = clear_resp.json()

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("=" * 72)
    print("Live backend smoke test passed")
    print("=" * 72)
    print(f"report_file : {report_path}")
    print(f"sync_action : {report['sync_video_inference'].get('action_type')}")
    print(f"sync_score  : {report['sync_video_inference'].get('overall_score')}")
    print(f"async_task  : {report['async_task_final']['status']}")
    print(
        f"realtime_ok : {any(msg.get('status') in {'ok', 'cached'} for msg in report['realtime_messages'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
