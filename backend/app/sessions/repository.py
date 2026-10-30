"""Session persistence: in-memory cache with JSON write-through, so sessions survive restarts."""

from __future__ import annotations

import threading
from pathlib import Path

from app.errors import NotFoundError
from app.sessions.models import Session


class SessionRepository:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, Session] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def _path(self, session_id: str) -> Path:
        if not session_id.isalnum():
            raise NotFoundError("Session not found.", detail=f"invalid id {session_id!r}")
        return self.root / f"{session_id}.json"

    def lock_for(self, session_id: str) -> threading.Lock:
        """Per-session lock so concurrent requests (e.g. double clicks) can't corrupt state."""
        with self._guard:
            return self._locks.setdefault(session_id, threading.Lock())

    def save(self, session: Session) -> None:
        with self._guard:
            self._sessions[session.session_id] = session
        self._path(session.session_id).write_text(session.model_dump_json(), encoding="utf-8")

    def get(self, session_id: str) -> Session:
        with self._guard:
            session = self._sessions.get(session_id)
        if session is not None:
            return session
        path = self._path(session_id)
        if not path.is_file():
            raise NotFoundError("Session not found. It may have expired; please start a new one.")
        session = Session.model_validate_json(path.read_text(encoding="utf-8"))
        with self._guard:
            self._sessions[session_id] = session
        return session
