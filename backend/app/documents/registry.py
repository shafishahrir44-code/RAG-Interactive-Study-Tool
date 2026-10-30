"""Stores document records (status, metadata, topics) as JSON files."""

from __future__ import annotations

import threading
from pathlib import Path

from app.documents.models import DocumentRecord
from app.errors import NotFoundError


class DocumentRegistry:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, DocumentRecord] = {}
        self._lock = threading.Lock()

    def _path(self, document_id: str) -> Path:
        if not document_id.isalnum():
            raise NotFoundError("Unknown document.", detail=f"invalid id {document_id!r}")
        return self.root / f"{document_id}.json"

    def save(self, record: DocumentRecord) -> None:
        with self._lock:
            self._records[record.document_id] = record
            self._path(record.document_id).write_text(record.model_dump_json(), encoding="utf-8")

    def find(self, document_id: str) -> DocumentRecord | None:
        with self._lock:
            record = self._records.get(document_id)
        if record is not None:
            return record
        path = self._path(document_id)
        if not path.is_file():
            return None
        record = DocumentRecord.model_validate_json(path.read_text(encoding="utf-8"))
        with self._lock:
            self._records[document_id] = record
        return record

    def get(self, document_id: str) -> DocumentRecord:
        record = self.find(document_id)
        if record is None:
            raise NotFoundError("Document not found. It may have been removed; please upload it again.")
        return record

    def update(self, document_id: str, **changes) -> DocumentRecord:
        record = self.get(document_id).model_copy(update=changes)
        self.save(record)
        return record
