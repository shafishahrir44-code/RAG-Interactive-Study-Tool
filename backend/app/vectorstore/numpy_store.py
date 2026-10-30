"""Persistent exact-search vector store backed by NumPy.

One directory per document: `vectors.npy`, `chunks.json`, `manifest.json`.
Exact cosine search over a single document's chunks is fast for documents up to
tens of thousands of chunks; swap in an ANN store (FAISS, Chroma, Qdrant) via the
`VectorStore` protocol for larger corpora.
"""

from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path

import numpy as np

from app.errors import NotFoundError, RetrievalError
from app.ingestion.models import Chunk
from app.vectorstore.base import ScoredChunk

MANIFEST_VERSION = 2  # bumped when chunk format changes, forcing a rebuild of older indexes


class NumpyVectorStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, tuple[list[Chunk], np.ndarray]] = {}
        self._lock = threading.Lock()

    def _dir(self, document_id: str) -> Path:
        if not document_id.isalnum():  # ids are hex digests; refuse anything path-like
            raise NotFoundError("Unknown document.", detail=f"invalid id {document_id!r}")
        return self.root / document_id

    def _manifest(self, document_id: str) -> dict | None:
        path = self._dir(document_id) / "manifest.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def has_index(self, document_id: str, fingerprint: str) -> bool:
        manifest = self._manifest(document_id)
        return bool(
            manifest
            and manifest.get("fingerprint") == fingerprint
            and manifest.get("version") == MANIFEST_VERSION
        )

    def upsert(self, document_id: str, chunks: list[Chunk], vectors: np.ndarray, fingerprint: str) -> None:
        if len(chunks) != len(vectors):
            raise RetrievalError("Index build failed.", detail="chunk/vector count mismatch")
        directory = self._dir(document_id)
        directory.mkdir(parents=True, exist_ok=True)
        vectors = np.asarray(vectors, dtype=np.float32)
        np.save(directory / "vectors.npy", vectors)
        (directory / "chunks.json").write_text(
            json.dumps([c.model_dump() for c in chunks]), encoding="utf-8"
        )
        # The manifest is written last so a crash never leaves a "valid-looking" partial index.
        manifest = {"version": MANIFEST_VERSION, "fingerprint": fingerprint, "count": len(chunks)}
        (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with self._lock:
            self._cache[document_id] = (chunks, vectors)

    def _load(self, document_id: str) -> tuple[list[Chunk], np.ndarray]:
        with self._lock:
            cached = self._cache.get(document_id)
        if cached:
            return cached
        directory = self._dir(document_id)
        if self._manifest(document_id) is None:
            raise NotFoundError("This document has not been indexed.", detail=document_id)
        try:
            chunks = [Chunk(**c) for c in json.loads((directory / "chunks.json").read_text(encoding="utf-8"))]
            vectors = np.load(directory / "vectors.npy")
        except (OSError, ValueError) as exc:
            raise RetrievalError("The document index is damaged; please re-upload.", detail=repr(exc)) from exc
        with self._lock:
            self._cache[document_id] = (chunks, vectors)
        return chunks, vectors

    def get_chunks(self, document_id: str) -> list[Chunk]:
        return list(self._load(document_id)[0])

    def get_vectors(self, document_id: str) -> np.ndarray:
        return self._load(document_id)[1]

    def search(self, document_id: str, query_vector: np.ndarray, k: int) -> list[ScoredChunk]:
        chunks, vectors = self._load(document_id)
        if not chunks or k <= 0:
            return []
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if query.shape[0] != vectors.shape[1]:
            raise RetrievalError(
                "The query embedding does not match the index. Re-upload the document.",
                detail=f"query dim {query.shape[0]} vs index dim {vectors.shape[1]}",
            )
        scores = vectors @ query
        k = min(k, len(chunks))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [ScoredChunk(chunks[i], float(scores[i])) for i in top]

    def delete(self, document_id: str) -> None:
        with self._lock:
            self._cache.pop(document_id, None)
        shutil.rmtree(self._dir(document_id), ignore_errors=True)
