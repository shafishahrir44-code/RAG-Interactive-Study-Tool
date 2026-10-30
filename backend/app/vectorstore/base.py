"""Vector store interface. Implementations must keep chunk metadata alongside vectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from app.ingestion.models import Chunk


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


class VectorStore(Protocol):
    def has_index(self, document_id: str, fingerprint: str) -> bool: ...

    def upsert(self, document_id: str, chunks: list[Chunk], vectors: np.ndarray, fingerprint: str) -> None: ...

    def get_chunks(self, document_id: str) -> list[Chunk]: ...

    def search(self, document_id: str, query_vector: np.ndarray, k: int) -> list[ScoredChunk]: ...

    def delete(self, document_id: str) -> None: ...
