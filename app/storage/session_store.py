"""In-memory store for realtime sessions and reports."""

from __future__ import annotations

from threading import Lock
from typing import Any, Dict, Optional

from app.services.realtime_engine import RealtimeSession


class SessionStore:
    """Thread-safe in-memory store."""

    def __init__(self) -> None:
        self._sessions: Dict[str, RealtimeSession] = {}
        self._reports: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()

    def set_session(self, session: RealtimeSession) -> None:
        with self._lock:
            self._sessions[session.session_id] = session

    def get_session(self, session_id: str) -> Optional[RealtimeSession]:
        with self._lock:
            return self._sessions.get(session_id)

    def remove_session(self, session_id: str) -> Optional[RealtimeSession]:
        with self._lock:
            return self._sessions.pop(session_id, None)

    def set_report(self, session_id: str, report: Dict[str, Any]) -> None:
        with self._lock:
            self._reports[session_id] = report

    def get_report(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._reports.get(session_id)

    def list_sessions(self) -> list[Dict[str, Any]]:
        """List active realtime session snapshots."""
        with self._lock:
            return [session.snapshot() for session in self._sessions.values()]

    def list_reports(self) -> list[Dict[str, Any]]:
        """List cached report summaries in memory."""
        with self._lock:
            reports = list(self._reports.values())
        reports.sort(key=lambda item: item.get("duration_seconds", 0), reverse=True)
        return reports
