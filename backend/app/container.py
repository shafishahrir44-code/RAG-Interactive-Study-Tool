"""Composition root: builds every component from Settings. Swap implementations here or via env vars."""

from __future__ import annotations

from dataclasses import dataclass

from app.adaptive.policy import AdaptivePolicy
from app.config import Settings
from app.documents.registry import DocumentRegistry
from app.embeddings.base import Embedder
from app.embeddings.factory import create_embedder
from app.evaluation.evaluator import AnswerEvaluator
from app.ingestion.chunker import Chunker
from app.ingestion.pdf_extractor import PypdfTextExtractor
from app.ingestion.pipeline import IngestionService
from app.llm import LLMClient, StructuredLLM, create_llm_client
from app.questions.extractive_engine import ExtractiveQuestionEngine
from app.questions.llm_engine import LLMQuestionEngine
from app.questions.service import QuestionService
from app.retrieval.retriever import Retriever
from app.sessions.repository import SessionRepository
from app.sessions.service import SessionService
from app.vectorstore import VectorStore, create_vector_store


@dataclass
class Container:
    settings: Settings
    llm_client: LLMClient | None
    embedder: Embedder
    store: VectorStore
    registry: DocumentRegistry
    retriever: Retriever
    ingestion: IngestionService
    questions: QuestionService
    sessions: SessionService


def build_container(
    settings: Settings, llm_client: LLMClient | None = None, embedder: Embedder | None = None, use_llm: bool = True
) -> Container:
    """`llm_client`/`embedder` overrides exist for tests and custom deployments."""
    embedder = embedder or create_embedder(settings.embedding)
    if llm_client is None and use_llm:
        llm_client = create_llm_client(settings.llm)
    store = create_vector_store(settings.retrieval.vector_store, settings.index_dir)
    registry = DocumentRegistry(settings.data_dir / "documents")
    retriever = Retriever(store, embedder, settings.retrieval)
    structured = StructuredLLM(llm_client) if llm_client else None

    ingestion = IngestionService(
        settings,
        registry,
        PypdfTextExtractor(settings.limits.max_pages),
        Chunker(settings.chunking),
        embedder,
        store,
    )
    questions = QuestionService(
        retriever=retriever,
        embedder=embedder,
        llm_engine=LLMQuestionEngine(structured, settings.llm.temperature) if structured else None,
        extractive_engine=ExtractiveQuestionEngine(),
        fallback_to_extractive=settings.llm.fallback_to_extractive,
        max_attempts=settings.limits.max_generation_attempts,
        duplicate_similarity=settings.limits.duplicate_similarity,
    )
    sessions = SessionService(
        settings,
        registry,
        store,
        SessionRepository(settings.session_dir),
        questions,
        AnswerEvaluator(structured, settings.llm.eval_temperature),
        AdaptivePolicy(settings.adaptive),
    )
    return Container(settings, llm_client, embedder, store, registry, retriever, ingestion, questions, sessions)
