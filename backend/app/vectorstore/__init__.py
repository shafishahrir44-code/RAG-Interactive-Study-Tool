from __future__ import annotations

from pathlib import Path

from app.errors import InvalidInputError
from app.vectorstore.base import ScoredChunk, VectorStore
from app.vectorstore.numpy_store import NumpyVectorStore

__all__ = ["ScoredChunk", "VectorStore", "NumpyVectorStore", "create_vector_store"]


def create_vector_store(name: str, root: Path) -> VectorStore:
    if name == "numpy":
        return NumpyVectorStore(root)
    raise InvalidInputError(f"Unknown VECTOR_STORE '{name}'.")
