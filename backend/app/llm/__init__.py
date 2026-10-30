from __future__ import annotations

import httpx

from app.config import LLMSettings
from app.errors import InvalidInputError
from app.llm.base import LLMClient, StructuredLLM
from app.llm.providers import GEMINI_DEFAULT_BASE_URL, GeminiClient, OllamaClient, OpenAICompatibleClient

__all__ = ["LLMClient", "StructuredLLM", "create_llm_client"]


def create_llm_client(settings: LLMSettings, client: httpx.Client | None = None) -> LLMClient | None:
    """Return the configured client, or None when LLM_PROVIDER=none (extractive-only mode)."""
    if settings.provider == "none":
        return None
    client = client or httpx.Client(timeout=settings.timeout_seconds)
    if settings.provider == "ollama":
        return OllamaClient(settings.model, settings.base_url, client, settings.reasoning_effort)
    if settings.provider == "openai_compatible":
        return OpenAICompatibleClient(
            settings.model, settings.base_url, settings.api_key, client, settings.reasoning_effort
        )
    if settings.provider == "gemini":
        return GeminiClient(
            settings.model,
            _gemini_base_url(settings.base_url),
            settings.api_key,
            client,
            settings.reasoning_effort,
            settings.fallback_models,
        )
    raise InvalidInputError(f"Unknown LLM_PROVIDER '{settings.provider}'.")


def _gemini_base_url(configured: str) -> str:
    """Use Google's native endpoint unless a Gemini URL was configured explicitly.
    Tolerates the OpenAI-compatible URL (".../openai") from earlier configurations."""
    if "googleapis.com" not in configured:
        return GEMINI_DEFAULT_BASE_URL
    return configured.rstrip("/").removesuffix("/openai")
