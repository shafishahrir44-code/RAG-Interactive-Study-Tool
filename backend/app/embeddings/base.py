"""Embedding provider interface."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    @property
    def fingerprint(self) -> str:
        """Identifies the embedding space; indexes built with another fingerprint are rebuilt."""
        ...

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Return an (n, d) float32 matrix of L2-normalised vectors."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Return a (d,) float32 L2-normalised vector."""
        ...


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms
