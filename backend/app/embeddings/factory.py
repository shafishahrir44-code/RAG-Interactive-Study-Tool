from __future__ import annotations

import httpx

from app.config import EmbeddingSettings
from app.embeddings.base import Embedder
from app.embeddings.hashing import HashingEmbedder
from app.embeddings.remote import OllamaEmbedder, OpenAICompatibleEmbedder, SentenceTransformerEmbedder
from app.errors import InvalidInputError

EMBEDDING_TIMEOUT_SECONDS = 120.0


def create_embedder(settings: EmbeddingSettings, client: httpx.Client | None = None) -> Embedder:
    provider = settings.provider
    if provider == "hash":
        return HashingEmbedder(settings.dimension)
    if provider == "sentence_transformers":
        return SentenceTransformerEmbedder(settings.model)
    client = client or httpx.Client(timeout=EMBEDDING_TIMEOUT_SECONDS)
    if provider == "ollama":
        return OllamaEmbedder(settings.model, settings.base_url, settings.batch_size, client)
    if provider == "openai_compatible":
        return OpenAICompatibleEmbedder(
            settings.model, settings.base_url, settings.batch_size, client, settings.api_key
        )
    raise InvalidInputError(f"Unknown EMBEDDING_PROVIDER '{provider}'.")
