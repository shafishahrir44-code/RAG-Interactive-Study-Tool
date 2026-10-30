"""LLM provider interface and structured-output helper."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from app.errors import LLMResponseError

logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class LLMClient(Protocol):
    name: str

    def complete_json(self, system: str, user: str, schema: dict[str, Any], temperature: float) -> str:
        """Return the raw model output, which should be a JSON object matching `schema`."""
        ...

    def is_available(self) -> bool: ...


def extract_json_object(raw: str) -> dict[str, Any]:
    """Parse a JSON object from model output, recovering from code fences or surrounding prose."""
    text = _FENCE_RE.sub("", raw.strip())
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise LLMResponseError("The AI model did not return JSON.", detail=raw[:500]) from None
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMResponseError("The AI model returned malformed JSON.", detail=raw[:500]) from exc
    if not isinstance(value, dict):
        raise LLMResponseError("The AI model returned JSON of the wrong shape.", detail=raw[:500])
    return value


class StructuredLLM:
    """Requests JSON matching a Pydantic model; validates it; retries once with the error fed back."""

    REPAIR_ATTEMPTS = 1

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    @property
    def name(self) -> str:
        return self.client.name

    def generate(self, model_cls: type[ModelT], system: str, user: str, temperature: float) -> ModelT:
        schema = model_cls.model_json_schema()
        prompt = user
        last_error = ""
        for attempt in range(self.REPAIR_ATTEMPTS + 1):
            raw = self.client.complete_json(system, prompt, schema, temperature)
            try:
                return model_cls.model_validate(extract_json_object(raw))
            except (LLMResponseError, ValidationError) as exc:
                last_error = exc.detail if isinstance(exc, LLMResponseError) else str(exc)
                logger.info("Invalid structured output (attempt %s): %s", attempt + 1, last_error[:300])
                prompt = (
                    f"{user}\n\nYour previous reply was invalid:\n{last_error[:800]}\n"
                    "Reply again with ONLY a JSON object that satisfies the schema."
                )
        raise LLMResponseError("The AI model kept returning invalid output.", detail=last_error[:1000])
