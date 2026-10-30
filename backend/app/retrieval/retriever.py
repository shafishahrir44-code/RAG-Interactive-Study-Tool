"""Hybrid retriever: dense vector search followed by BM25-assisted reranking.

Pipeline: query -> embedding -> vector search (top_k * multiplier candidates)
-> rerank by alpha * cosine + (1 - alpha) * normalised BM25 -> top_k -> context budget.
"""

from __future__ import annotations

import threading

from app.config import RetrievalSettings
from app.embeddings.base import Embedder
from app.errors import AppError, RetrievalError
from app.ingestion.models import Chunk
from app.retrieval.lexical import BM25Index
from app.vectorstore.base import ScoredChunk, VectorStore


class Retriever:
    def __init__(self, store: VectorStore, embedder: Embedder, settings: RetrievalSettings) -> None:
        self.store = store
        self.embedder = embedder
        self.settings = settings
        self._bm25: dict[str, BM25Index] = {}
        self._lock = threading.Lock()

    def _bm25_index(self, document_id: str) -> BM25Index:
        with self._lock:
            index = self._bm25.get(document_id)
        if index is None:
            index = BM25Index(self.store.get_chunks(document_id))
            with self._lock:
                self._bm25[document_id] = index
        return index

    def invalidate(self, document_id: str) -> None:
        with self._lock:
            self._bm25.pop(document_id, None)

    def retrieve(self, document_id: str, query: str, k: int | None = None) -> list[ScoredChunk]:
        k = k or self.settings.top_k
        try:
            query_vector = self.embedder.embed_query(query)
            candidates = self.store.search(document_id, query_vector, k * self.settings.candidate_multiplier)
        except AppError:
            raise
        except Exception as exc:  # noqa: BLE001 - surface unexpected backend failures cleanly
            raise RetrievalError("Searching the document failed.", detail=repr(exc)) from exc
        return self._rerank(document_id, query, candidates)[:k]

    def _rerank(self, document_id: str, query: str, candidates: list[ScoredChunk]) -> list[ScoredChunk]:
        if not candidates:
            return []
        bm25 = self._bm25_index(document_id)
        lexical = [bm25.score(query, c.chunk.chunk_id) for c in candidates]
        max_lexical = max(lexical) or 1.0
        alpha = self.settings.hybrid_alpha
        rescored = [
            ScoredChunk(c.chunk, alpha * c.score + (1 - alpha) * (lex / max_lexical))
            for c, lex in zip(candidates, lexical)
        ]
        return sorted(rescored, key=lambda s: s.score, reverse=True)

    def context_for_seed(self, seed: Chunk) -> list[Chunk]:
        """Seed chunk plus its most related chunks, trimmed to the context budget.

        The seed anchors the question on one concept; neighbours supply supporting detail
        (and plausible distractors) from elsewhere in the document.
        """
        related = self.retrieve(seed.document_id, seed.text, self.settings.top_k + 1)
        ordered = [seed] + [r.chunk for r in related if r.chunk.chunk_id != seed.chunk_id]
        return self.fit_budget(ordered)

    def fit_budget(self, chunks: list[Chunk]) -> list[Chunk]:
        selected: list[Chunk] = []
        used = 0
        for chunk in chunks[: self.settings.top_k]:
            if selected and used + len(chunk.text) > self.settings.max_context_chars:
                break
            selected.append(chunk)
            used += len(chunk.text)
        return selected
