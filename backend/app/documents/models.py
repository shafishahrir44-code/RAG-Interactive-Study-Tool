from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

from app.retrieval.topics import Topic


class DocumentStatus(str, Enum):
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class ProcessingStage(str, Enum):
    QUEUED = "queued"
    EXTRACTING = "extracting"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    DONE = "done"


class DocumentRecord(BaseModel):
    document_id: str
    filename: str
    size_bytes: int
    status: DocumentStatus = DocumentStatus.PROCESSING
    stage: ProcessingStage = ProcessingStage.QUEUED
    progress: float = 0.0  # 0..1
    title: str | None = None
    page_count: int = 0
    pages_with_text: int = 0
    empty_pages: list[int] = Field(default_factory=list)
    chunk_count: int = 0
    topics: list[Topic] = Field(default_factory=list)
    embedding_fingerprint: str | None = None
    warnings: list[str] = Field(default_factory=list)
    error_message: str | None = None
    error_detail: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
