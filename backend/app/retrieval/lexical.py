"""BM25 lexical scoring used to rerank vector-search candidates (hybrid retrieval)."""

from __future__ import annotations

import math
from collections import Counter

from app.ingestion.models import Chunk
from app.text_utils import content_tokens

BM25_K1 = 1.5
BM25_B = 0.75


class BM25Index:
    def __init__(self, chunks: list[Chunk]) -> None:
        self._term_freqs: dict[str, Counter[str]] = {}
        self._lengths: dict[str, int] = {}
        doc_freq: Counter[str] = Counter()
        for chunk in chunks:
            tokens = content_tokens(chunk.text)
            self._term_freqs[chunk.chunk_id] = Counter(tokens)
            self._lengths[chunk.chunk_id] = len(tokens)
            doc_freq.update(set(tokens))
        total = max(len(chunks), 1)
        self._avg_length = sum(self._lengths.values()) / total or 1.0
        self.idf = {
            term: math.log(1 + (total - freq + 0.5) / (freq + 0.5)) for term, freq in doc_freq.items()
        }

    def score(self, query: str, chunk_id: str) -> float:
        term_freqs = self._term_freqs.get(chunk_id)
        if not term_freqs:
            return 0.0
        length_norm = 1 - BM25_B + BM25_B * self._lengths[chunk_id] / self._avg_length
        total = 0.0
        for term in set(content_tokens(query)):
            tf = term_freqs.get(term, 0)
            if tf:
                total += self.idf.get(term, 0.0) * tf * (BM25_K1 + 1) / (tf + BM25_K1 * length_norm)
        return total
