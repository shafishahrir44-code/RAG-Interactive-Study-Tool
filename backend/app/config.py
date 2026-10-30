"""Application configuration loaded from environment variables (and an optional .env file).

Every replaceable component (LLM, embeddings, vector store) is selected here by name,
so switching providers never requires code changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BACKEND_DIR.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (KEY=VALUE lines). Existing environment variables win."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class LLMSettings:
    provider: str = "ollama"  # ollama | gemini | openai_compatible | none
    model: str = "qwen2.5:7b-instruct"
    base_url: str = "http://localhost:11434"
    api_key: str = ""
    temperature: float = 0.3
    eval_temperature: float = 0.0
    timeout_seconds: float = 120.0
    # Lowers "thinking" time on reasoning models. openai_compatible: sent as reasoning_effort
    # (e.g. low); ollama: "none" disables thinking. Empty = provider default.
    reasoning_effort: str = ""
    # Gemini only: models to try, in order, when the main one is overloaded or rate-limited.
    fallback_models: tuple[str, ...] = ()
    # When the LLM is unreachable, fall back to the deterministic extractive generator
    # instead of failing. The UI always shows which engine produced a question.
    fallback_to_extractive: bool = True


@dataclass(frozen=True)
class EmbeddingSettings:
    provider: str = "hash"  # hash | ollama | openai_compatible | sentence_transformers
    model: str = "nomic-embed-text"
    base_url: str = "http://localhost:11434"
    api_key: str = ""
    dimension: int = 1024  # used by the hash embedder only
    batch_size: int = 32


@dataclass(frozen=True)
class ChunkingSettings:
    chunk_size: int = 900  # characters
    chunk_overlap: int = 150  # characters
    min_chunk_chars: int = 80


@dataclass(frozen=True)
class RetrievalSettings:
    vector_store: str = "numpy"
    top_k: int = 4
    candidate_multiplier: int = 3  # vector search fetches top_k * multiplier, then reranks
    hybrid_alpha: float = 0.7  # weight of vector similarity vs. BM25 in reranking
    max_context_chars: int = 6000


@dataclass(frozen=True)
class AdaptiveSettings:
    start_difficulty: str = "easy"
    window: int = 3  # recent answers considered at the current difficulty
    min_answers_before_change: int = 2
    promote_threshold: float = 0.8
    demote_threshold: float = 0.4
    weak_topic_threshold: float = 0.5
    strong_topic_threshold: float = 0.8


@dataclass(frozen=True)
class LimitSettings:
    max_upload_mb: int = 25
    max_pages: int = 500
    max_generation_attempts: int = 4
    duplicate_similarity: float = 0.9
    min_questions: int = 1
    max_questions: int = 50


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: BACKEND_DIR / "data")
    frontend_dir: Path = field(default_factory=lambda: PROJECT_DIR / "frontend")
    debug: bool = False
    llm: LLMSettings = field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    chunking: ChunkingSettings = field(default_factory=ChunkingSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    adaptive: AdaptiveSettings = field(default_factory=AdaptiveSettings)
    limits: LimitSettings = field(default_factory=LimitSettings)

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "indexes"

    @property
    def session_dir(self) -> Path:
        return self.data_dir / "sessions"


def load_settings(env_file: Path | None = None) -> Settings:
    _load_dotenv(env_file or PROJECT_DIR / ".env")
    llm = LLMSettings(
        provider=_env_str("LLM_PROVIDER", LLMSettings.provider).lower(),
        model=_env_str("LLM_MODEL", LLMSettings.model),
        base_url=_env_str("LLM_BASE_URL", LLMSettings.base_url),
        api_key=_env_str("LLM_API_KEY", ""),
        temperature=_env_float("LLM_TEMPERATURE", LLMSettings.temperature),
        eval_temperature=_env_float("LLM_EVAL_TEMPERATURE", LLMSettings.eval_temperature),
        timeout_seconds=_env_float("LLM_TIMEOUT_SECONDS", LLMSettings.timeout_seconds),
        reasoning_effort=_env_str("LLM_REASONING_EFFORT", "").lower(),
        fallback_models=tuple(m.strip() for m in _env_str("LLM_FALLBACK_MODELS", "").split(",") if m.strip()),
        fallback_to_extractive=_env_bool("LLM_FALLBACK_TO_EXTRACTIVE", True),
    )
    embedding = EmbeddingSettings(
        provider=_env_str("EMBEDDING_PROVIDER", EmbeddingSettings.provider).lower(),
        model=_env_str("EMBEDDING_MODEL", EmbeddingSettings.model),
        base_url=_env_str("EMBEDDING_BASE_URL", EmbeddingSettings.base_url),
        api_key=_env_str("EMBEDDING_API_KEY", ""),
        dimension=_env_int("EMBEDDING_DIMENSION", EmbeddingSettings.dimension),
        batch_size=_env_int("EMBEDDING_BATCH_SIZE", EmbeddingSettings.batch_size),
    )
    chunking = ChunkingSettings(
        chunk_size=_env_int("CHUNK_SIZE", ChunkingSettings.chunk_size),
        chunk_overlap=_env_int("CHUNK_OVERLAP", ChunkingSettings.chunk_overlap),
        min_chunk_chars=_env_int("MIN_CHUNK_CHARS", ChunkingSettings.min_chunk_chars),
    )
    retrieval = RetrievalSettings(
        vector_store=_env_str("VECTOR_STORE", RetrievalSettings.vector_store).lower(),
        top_k=_env_int("RETRIEVAL_TOP_K", RetrievalSettings.top_k),
        candidate_multiplier=_env_int("RETRIEVAL_CANDIDATE_MULTIPLIER", RetrievalSettings.candidate_multiplier),
        hybrid_alpha=_env_float("RETRIEVAL_HYBRID_ALPHA", RetrievalSettings.hybrid_alpha),
        max_context_chars=_env_int("MAX_CONTEXT_CHARS", RetrievalSettings.max_context_chars),
    )
    adaptive = AdaptiveSettings(
        start_difficulty=_env_str("START_DIFFICULTY", AdaptiveSettings.start_difficulty).lower(),
        window=_env_int("ADAPTIVE_WINDOW", AdaptiveSettings.window),
        min_answers_before_change=_env_int("ADAPTIVE_MIN_ANSWERS", AdaptiveSettings.min_answers_before_change),
        promote_threshold=_env_float("ADAPTIVE_PROMOTE_THRESHOLD", AdaptiveSettings.promote_threshold),
        demote_threshold=_env_float("ADAPTIVE_DEMOTE_THRESHOLD", AdaptiveSettings.demote_threshold),
    )
    limits = LimitSettings(
        max_upload_mb=_env_int("MAX_UPLOAD_MB", LimitSettings.max_upload_mb),
        max_pages=_env_int("MAX_PAGES", LimitSettings.max_pages),
        max_generation_attempts=_env_int("MAX_GENERATION_ATTEMPTS", LimitSettings.max_generation_attempts),
    )
    data_dir = Path(_env_str("DATA_DIR", str(BACKEND_DIR / "data"))).resolve()
    return Settings(
        data_dir=data_dir,
        debug=_env_bool("DEBUG", False),
        llm=llm,
        embedding=embedding,
        chunking=chunking,
        retrieval=retrieval,
        adaptive=adaptive,
        limits=limits,
    )
