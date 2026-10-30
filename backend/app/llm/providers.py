"""Concrete LLM providers.

- OllamaClient: local models via Ollama, using native JSON-schema constrained decoding.
- OpenAICompatibleClient: any /v1/chat/completions server (OpenAI, Groq, LM Studio, vLLM,
  llama.cpp server, ...), using JSON mode plus the schema in the prompt.
- GeminiClient: Google's native Gemini API (generateContent) with JSON output. Works with the
  "AQ." keys AI Studio now issues, which the OpenAI-compatible Gemini endpoint often rejects.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from app.errors import LLMResponseError, LLMUnavailableError
from app.http_utils import bearer_headers, post_json

logger = logging.getLogger(__name__)

AVAILABILITY_TIMEOUT_SECONDS = 3.0


def _schema_instruction(schema: dict[str, Any]) -> str:
    return "Respond with ONLY a JSON object that conforms to this JSON schema:\n" + json.dumps(schema)


class OllamaClient:
    def __init__(self, model: str, base_url: str, client: httpx.Client, reasoning_effort: str = "") -> None:
        self.name = f"ollama:{model}"
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.client = client
        # Only "none" is mapped: it turns off thinking on reasoning models such as qwen3.
        self.extra = {"think": False} if reasoning_effort == "none" else {}

    def complete_json(self, system: str, user: str, schema: dict[str, Any], temperature: float) -> str:
        data = post_json(
            self.client,
            f"{self.base_url}/api/chat",
            {
                "model": self.model,
                "stream": False,
                "format": schema,
                "options": {"temperature": temperature},
                **self.extra,
                "messages": [
                    {"role": "system", "content": f"{system}\n\n{_schema_instruction(schema)}"},
                    {"role": "user", "content": user},
                ],
            },
            error_cls=LLMUnavailableError,
            service_name="Ollama",
        )
        content = (data.get("message") or {}).get("content")
        if not isinstance(content, str):
            raise LLMResponseError("Ollama returned an empty response.", detail=str(data)[:300])
        return content

    def is_available(self) -> bool:
        try:
            response = self.client.get(f"{self.base_url}/api/tags", timeout=AVAILABILITY_TIMEOUT_SECONDS)
            if response.status_code != 200:
                return False
            names = {m.get("name", "") for m in response.json().get("models", [])}
            return any(n == self.model or n.split(":")[0] == self.model for n in names)
        except (httpx.HTTPError, ValueError):
            return False


class OpenAICompatibleClient:
    def __init__(
        self, model: str, base_url: str, api_key: str, client: httpx.Client, reasoning_effort: str = ""
    ) -> None:
        self.name = f"openai_compatible:{model}"
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.headers = bearer_headers(api_key)
        self.client = client
        self.extra = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}

    def complete_json(self, system: str, user: str, schema: dict[str, Any], temperature: float) -> str:
        data = post_json(
            self.client,
            f"{self.base_url}/chat/completions",
            {
                "model": self.model,
                "temperature": temperature,
                "response_format": {"type": "json_object"},
                **self.extra,
                "messages": [
                    {"role": "system", "content": f"{system}\n\n{_schema_instruction(schema)}"},
                    {"role": "user", "content": user},
                ],
            },
            error_cls=LLMUnavailableError,
            service_name="the LLM API",
            headers=self.headers,
        )
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError("The LLM API returned an unexpected response.", detail=str(data)[:300]) from exc

    def is_available(self) -> bool:
        try:
            response = self.client.get(
                f"{self.base_url}/models", headers=self.headers, timeout=AVAILABILITY_TIMEOUT_SECONDS
            )
            return response.status_code == 200
        except httpx.HTTPError:
            return False


GEMINI_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_MAX_OUTPUT_TOKENS = 4096  # includes thinking tokens; a question needs well under 1,000


GEMINI_RETRYABLE_STATUSES = {408, 429, 500, 503, 504}  # timed out, overloaded, rate-limited or transient


class GeminiClient:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        client: httpx.Client,
        reasoning_effort: str = "",
        fallback_models: tuple[str, ...] = (),
    ) -> None:
        self.name = f"gemini:{model}"
        self.model = model.removeprefix("models/")
        self.fallback_models = tuple(m.removeprefix("models/") for m in fallback_models)
        self.base_url = base_url.rstrip("/")
        self.headers = {"x-goog-api-key": api_key}
        self.client = client
        # Gemini 3 models accept a thinking level ("low", "high", ...); lower is faster.
        self.thinking = {"thinkingConfig": {"thinkingLevel": reasoning_effort}} if reasoning_effort else {}

    def complete_json(self, system: str, user: str, schema: dict[str, Any], temperature: float) -> str:
        """Ask the main model; if it is overloaded or rate-limited, try each fallback model in turn."""
        models = (self.model, *self.fallback_models)
        for index, model in enumerate(models):
            try:
                return self._generate(model, system, user, schema)
            except LLMUnavailableError as exc:
                if index == len(models) - 1 or exc.upstream_status not in GEMINI_RETRYABLE_STATUSES:
                    raise
                logger.warning("%s returned HTTP %s; trying %s", model, exc.upstream_status, models[index + 1])
        raise AssertionError("unreachable")  # pragma: no cover

    def _generate(self, model: str, system: str, user: str, schema: dict[str, Any]) -> str:
        data = post_json(
            self.client,
            f"{self.base_url}/models/{model}:generateContent",
            {
                "systemInstruction": {"parts": [{"text": f"{system}\n\n{_schema_instruction(schema)}"}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                # No temperature: Google advises keeping Gemini 3 at its default, since lower
                # values can make it loop. The output cap stops any runaway generation early.
                "generationConfig": {
                    "maxOutputTokens": GEMINI_MAX_OUTPUT_TOKENS,
                    "responseMimeType": "application/json",
                    **self.thinking,
                },
            },
            error_cls=LLMUnavailableError,
            service_name="the Gemini API",
            headers=self.headers,
        )
        try:
            parts = data["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError, TypeError) as exc:
            reason = (data.get("promptFeedback") or {}).get("blockReason") or str(data)[:300]
            raise LLMResponseError("Gemini returned no answer.", detail=reason) from exc
        # Skip "thought" parts; only the final answer text is the JSON we asked for.
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        if data["candidates"][0].get("finishReason") == "MAX_TOKENS":
            raise LLMResponseError("Gemini's answer was cut off.", detail="finishReason=MAX_TOKENS")
        if not text:
            raise LLMResponseError("Gemini returned an empty answer.", detail=str(data)[:300])
        return text

    def is_available(self) -> bool:
        try:
            response = self.client.get(
                f"{self.base_url}/models/{self.model}", headers=self.headers, timeout=AVAILABILITY_TIMEOUT_SECONDS
            )
            return response.status_code == 200
        except httpx.HTTPError:
            return False
