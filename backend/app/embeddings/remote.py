"""HTTP embedding providers: Ollama (/api/embed) and any OpenAI-compatible /v1/embeddings server
(OpenAI, LM Studio, vLLM, llama.cpp server, LocalAI, ...)."""

from __future__ import annotations

import httpx
import numpy as np

from app.embeddings.base import l2_normalize
from app.errors import EmbeddingError
from app.http_utils import bearer_headers, post_json

# Some embedding models are trained with task prefixes; using them improves retrieval quality.
_TASK_PREFIXES = {
    "nomic-embed-text": ("search_document: ", "search_query: "),
}


def _prefixes(model: str) -> tuple[str, str]:
    for name, prefixes in _TASK_PREFIXES.items():
        if name in model:
            return prefixes
    return "", ""


class _BatchedHTTPEmbedder:
    service_name = "the embedding service"

    def __init__(self, model: str, base_url: str, batch_size: int, client: httpx.Client) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.batch_size = max(1, batch_size)
        self.client = client
        self.doc_prefix, self.query_prefix = _prefixes(model)

    @property
    def fingerprint(self) -> str:
        return f"{type(self).__name__}-{self.model}"

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover - abstract
        raise NotImplementedError

    def _embed(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            result = self._embed_batch(batch)
            if len(result) != len(batch):
                raise EmbeddingError(
                    "The embedding service returned an unexpected number of vectors.",
                    detail=f"expected {len(batch)}, got {len(result)}",
                )
            vectors.extend(result)
        return l2_normalize(np.asarray(vectors, dtype=np.float32))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return self._embed([self.doc_prefix + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([self.query_prefix + text])[0]


class OllamaEmbedder(_BatchedHTTPEmbedder):
    service_name = "Ollama (embeddings)"

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        data = post_json(
            self.client,
            f"{self.base_url}/api/embed",
            {"model": self.model, "input": texts},
            error_cls=EmbeddingError,
            service_name=self.service_name,
        )
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list):
            raise EmbeddingError("Ollama returned no embeddings.", detail=str(data)[:300])
        return embeddings


class OpenAICompatibleEmbedder(_BatchedHTTPEmbedder):
    service_name = "the embedding API"

    def __init__(self, model: str, base_url: str, batch_size: int, client: httpx.Client, api_key: str) -> None:
        super().__init__(model, base_url, batch_size, client)
        self.api_key = api_key

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        data = post_json(
            self.client,
            f"{self.base_url}/embeddings",
            {"model": self.model, "input": texts},
            error_cls=EmbeddingError,
            service_name=self.service_name,
            headers=bearer_headers(self.api_key),
        )
        items = data.get("data")
        if not isinstance(items, list):
            raise EmbeddingError("The embedding API returned no data.", detail=str(data)[:300])
        return [item["embedding"] for item in sorted(items, key=lambda item: item.get("index", 0))]


class SentenceTransformerEmbedder:
    """Local semantic embeddings via the optional `sentence-transformers` package."""

    def __init__(self, model: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingError(
                "EMBEDDING_PROVIDER=sentence_transformers requires `pip install sentence-transformers`.",
                detail=repr(exc),
            ) from exc
        self.model_name = model
        self._model = SentenceTransformer(model)

    @property
    def fingerprint(self) -> str:
        return f"st-{self.model_name}"

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return l2_normalize(self._model.encode(texts, convert_to_numpy=True))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]
