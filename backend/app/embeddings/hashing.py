"""Zero-download local embedder based on the hashing trick.

Maps unigrams and bigrams of content tokens into a fixed-size vector with sublinear TF
weighting. It is a lexical (not semantic) embedding: fast, deterministic and dependency-free,
which makes the app run out of the box. Use the Ollama or sentence-transformers providers
for semantic embeddings.
"""

from __future__ import annotations

import hashlib

import numpy as np

from app.embeddings.base import l2_normalize
from app.text_utils import content_tokens


def _bucket(feature: str, dimension: int) -> tuple[int, float]:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "little")
    sign = 1.0 if value & 1 else -1.0  # signed hashing reduces collision bias
    return (value >> 1) % dimension, sign


class HashingEmbedder:
    def __init__(self, dimension: int = 1024) -> None:
        self.dimension = dimension

    @property
    def fingerprint(self) -> str:
        return f"hash-v1-{self.dimension}"

    def _vector(self, text: str) -> np.ndarray:
        tokens = content_tokens(text)
        features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        counts: dict[str, int] = {}
        for feature in features:
            counts[feature] = counts.get(feature, 0) + 1
        vector = np.zeros(self.dimension, dtype=np.float32)
        for feature, count in counts.items():
            index, sign = _bucket(feature, self.dimension)
            vector[index] += sign * (1.0 + np.log(count))
        return vector

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return l2_normalize(np.stack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return l2_normalize(self._vector(text))
