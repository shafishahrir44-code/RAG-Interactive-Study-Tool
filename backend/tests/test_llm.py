from __future__ import annotations

import json

import httpx
import pytest
from pydantic import BaseModel

from app.config import LLMSettings
from app.embeddings.remote import OllamaEmbedder, OpenAICompatibleEmbedder
from app.errors import EmbeddingError, LLMResponseError, LLMUnavailableError
from app.llm import create_llm_client
from app.llm.base import StructuredLLM, extract_json_object
from app.llm.providers import (
    GEMINI_DEFAULT_BASE_URL,
    GeminiClient,
    OllamaClient,
    OpenAICompatibleClient,
)


class Answer(BaseModel):
    value: int


class ScriptedLLM:
    name = "scripted"

    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.prompts: list[str] = []

    def complete_json(self, system, user, schema, temperature):
        self.prompts.append(user)
        return self.replies.pop(0)

    def is_available(self):
        return True


@pytest.mark.parametrize(
    "raw",
    ['{"value": 3}', '```json\n{"value": 3}\n```', 'Sure! Here it is: {"value": 3} Hope that helps.'],
)
def test_extract_json_recovers_from_common_wrappers(raw: str) -> None:
    assert extract_json_object(raw) == {"value": 3}


@pytest.mark.parametrize("raw", ["no json here", "[1, 2, 3]", '{"value": '])
def test_extract_json_rejects_unusable_output(raw: str) -> None:
    with pytest.raises(LLMResponseError):
        extract_json_object(raw)


def test_structured_llm_repairs_invalid_output_once() -> None:
    llm = ScriptedLLM(['{"value": "not a number"}', '{"value": 7}'])
    assert StructuredLLM(llm).generate(Answer, "sys", "user", 0.0).value == 7
    assert "previous reply was invalid" in llm.prompts[1]


def test_structured_llm_gives_up_after_repair_attempt() -> None:
    llm = ScriptedLLM(["garbage", "still garbage"])
    with pytest.raises(LLMResponseError):
        StructuredLLM(llm).generate(Answer, "sys", "user", 0.0)


def mock_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_ollama_client_sends_schema_and_returns_content() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": '{"value": 1}'}})

    client = OllamaClient("qwen", "http://ollama", mock_client(handler))
    assert client.complete_json("s", "u", {"type": "object"}, 0.2) == '{"value": 1}'
    assert seen["format"] == {"type": "object"} and seen["stream"] is False
    assert seen["options"]["temperature"] == 0.2


def test_openai_compatible_client_uses_json_mode_and_auth() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    client = OpenAICompatibleClient("m", "http://api/v1", "secret", mock_client(handler))
    client.complete_json("s", "u", {}, 0.0)
    assert seen["auth"] == "Bearer secret"
    assert seen["body"]["response_format"] == {"type": "json_object"}


def test_connection_errors_become_llm_unavailable() -> None:
    def handler(request):
        raise httpx.ConnectError("refused")

    client = OllamaClient("m", "http://ollama", mock_client(handler))
    with pytest.raises(LLMUnavailableError):
        client.complete_json("s", "u", {}, 0.0)
    assert client.is_available() is False


def test_http_errors_become_llm_unavailable_with_detail() -> None:
    client = OpenAICompatibleClient("m", "http://api", "", mock_client(lambda r: httpx.Response(429, text="rate limited")))
    with pytest.raises(LLMUnavailableError) as info:
        client.complete_json("s", "u", {}, 0.0)
    assert "429" in info.value.message and "rate limited" in info.value.detail


def test_ollama_availability_checks_model_is_pulled() -> None:
    tags = {"models": [{"name": "qwen2.5:7b-instruct"}]}
    client = OllamaClient("qwen2.5:7b-instruct", "http://o", mock_client(lambda r: httpx.Response(200, json=tags)))
    assert client.is_available()
    missing = OllamaClient("llama3", "http://o", mock_client(lambda r: httpx.Response(200, json=tags)))
    assert not missing.is_available()


def test_remote_embedders_batch_and_normalise() -> None:
    batches = []

    def ollama(request):
        texts = json.loads(request.content)["input"]
        batches.append(len(texts))
        return httpx.Response(200, json={"embeddings": [[3.0, 4.0] for _ in texts]})

    embedder = OllamaEmbedder("nomic-embed-text", "http://o", batch_size=2, client=mock_client(ollama))
    vectors = embedder.embed_documents(["a", "b", "c"])
    assert batches == [2, 1]
    assert vectors.shape == (3, 2) and abs(vectors[0][0] - 0.6) < 1e-6

    def openai(request):
        texts = json.loads(request.content)["input"]
        data = [{"index": i, "embedding": [float(i + 1), 0.0]} for i in reversed(range(len(texts)))]
        return httpx.Response(200, json={"data": data})

    embedder = OpenAICompatibleEmbedder("m", "http://api", 8, mock_client(openai), api_key="")
    assert embedder.embed_query("x").tolist() == [1.0, 0.0]


def test_embedding_service_down_raises_embedding_error() -> None:
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(EmbeddingError):
        OllamaEmbedder("m", "http://o", 4, mock_client(handler)).embed_documents(["a"])


def test_gemini_client_uses_native_endpoint_and_api_key_header() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("x-goog-api-key")
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        parts = [{"text": "thinking...", "thought": True}, {"text": '{"value": 5}'}]
        return httpx.Response(200, json={"candidates": [{"content": {"parts": parts}}]})

    client = GeminiClient("models/gemini-x-flash", GEMINI_DEFAULT_BASE_URL, "AQ.test", mock_client(handler), "low")
    assert client.complete_json("s", "u", {"type": "object"}, 0.1) == '{"value": 5}'  # thought parts skipped
    assert seen["url"].endswith("/v1beta/models/gemini-x-flash:generateContent")
    assert seen["key"] == "AQ.test" and seen["auth"] is None  # one credential only
    config = seen["body"]["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    assert config["thinkingConfig"] == {"thinkingLevel": "low"}


def test_gemini_blocked_prompt_is_a_response_error() -> None:
    blocked = {"promptFeedback": {"blockReason": "SAFETY"}}
    client = GeminiClient("m", GEMINI_DEFAULT_BASE_URL, "k", mock_client(lambda r: httpx.Response(200, json=blocked)))
    with pytest.raises(LLMResponseError) as info:
        client.complete_json("s", "u", {}, 0.0)
    assert info.value.detail == "SAFETY"


def test_gemini_factory_accepts_openai_style_url_and_default() -> None:
    settings = LLMSettings(provider="gemini", model="m", base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    assert create_llm_client(settings).base_url == GEMINI_DEFAULT_BASE_URL
    assert create_llm_client(LLMSettings(provider="gemini", model="m")).base_url == GEMINI_DEFAULT_BASE_URL


def test_reasoning_effort_is_sent_only_when_configured() -> None:
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    OpenAICompatibleClient("m", "http://api", "", mock_client(handler), "low").complete_json("s", "u", {}, 0)
    OpenAICompatibleClient("m", "http://api", "", mock_client(handler)).complete_json("s", "u", {}, 0)
    assert bodies[0]["reasoning_effort"] == "low" and "reasoning_effort" not in bodies[1]


def test_gemini_keeps_default_temperature_and_caps_output() -> None:
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content)["generationConfig"])
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "{}"}]}}]})

    GeminiClient("m", GEMINI_DEFAULT_BASE_URL, "k", mock_client(handler)).complete_json("s", "u", {}, 0.0)
    assert "temperature" not in seen and seen["maxOutputTokens"] > 0


def test_gemini_cut_off_answer_is_a_response_error() -> None:
    body = {"candidates": [{"finishReason": "MAX_TOKENS", "content": {"parts": [{"text": '{"a": '}]}}]}
    client = GeminiClient("m", GEMINI_DEFAULT_BASE_URL, "k", mock_client(lambda r: httpx.Response(200, json=body)))
    with pytest.raises(LLMResponseError):
        client.complete_json("s", "u", {}, 0.0)


def test_gemini_switches_to_fallback_model_when_overloaded() -> None:
    called = []

    def handler(request):
        model = request.url.path.split("/")[-1].split(":")[0]
        called.append(model)
        if model == "busy-flash":
            return httpx.Response(503, json={"error": {"message": "high demand"}})
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "{}"}]}}]})

    client = GeminiClient("busy-flash", GEMINI_DEFAULT_BASE_URL, "k", mock_client(handler), fallback_models=("spare-flash",))
    assert client.complete_json("s", "u", {}, 0.0) == "{}"
    assert called == ["busy-flash", "spare-flash"]


def test_gemini_does_not_fall_back_on_auth_errors() -> None:
    client = GeminiClient("m", GEMINI_DEFAULT_BASE_URL, "bad", mock_client(lambda r: httpx.Response(401, text="bad key")),
                          fallback_models=("other",))
    with pytest.raises(LLMUnavailableError) as info:
        client.complete_json("s", "u", {}, 0.0)
    assert info.value.upstream_status == 401


def test_gemini_switches_model_after_a_timeout() -> None:
    def handler(request):
        if "slow-flash" in request.url.path:
            raise httpx.ReadTimeout("timed out")
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "{}"}]}}]})

    client = GeminiClient("slow-flash", GEMINI_DEFAULT_BASE_URL, "k", mock_client(handler), fallback_models=("fast-flash",))
    assert client.complete_json("s", "u", {}, 0.0) == "{}"
