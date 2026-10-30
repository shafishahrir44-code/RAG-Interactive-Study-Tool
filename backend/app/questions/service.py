"""Question orchestration: seed selection -> retrieval -> engine -> duplicate check, with retries."""

from __future__ import annotations

import logging
import time

import numpy as np

from app.adaptive.policy import QuestionPlan
from app.embeddings.base import Embedder
from app.errors import InsufficientContextError, LLMResponseError
from app.ingestion.models import Chunk
from app.questions.engine import GenerationRequest, QuestionEngine
from app.questions.models import Question
from app.retrieval.retriever import Retriever
from app.retrieval.topics import Topic
from app.sessions.models import Session
from app.text_utils import normalize_answer

logger = logging.getLogger(__name__)

LLM_RETRY_COOLDOWN_SECONDS = 60.0
FALLBACK_NOTICE = (
    "The AI model is unreachable, so questions are being built directly from document sentences "
    "(fill-in-the-blank, multiple choice and true/false)."
)


class QuestionService:
    def __init__(
        self,
        retriever: Retriever,
        embedder: Embedder,
        llm_engine: QuestionEngine | None,
        extractive_engine: QuestionEngine,
        fallback_to_extractive: bool,
        max_attempts: int,
        duplicate_similarity: float,
    ) -> None:
        self.retriever = retriever
        self.embedder = embedder
        self.llm_engine = llm_engine
        self.extractive_engine = extractive_engine
        self.fallback_to_extractive = fallback_to_extractive
        self.max_attempts = max_attempts
        self.duplicate_similarity = duplicate_similarity
        self._llm_down_until = 0.0

    # ----- engine selection ----------------------------------------------------------------

    def active_engine(self) -> QuestionEngine:
        if self.llm_engine is not None and (time.monotonic() >= self._llm_down_until or not self.fallback_to_extractive):
            return self.llm_engine
        return self.extractive_engine

    def mark_llm_unavailable(self) -> None:
        self._llm_down_until = time.monotonic() + LLM_RETRY_COOLDOWN_SECONDS

    def notice_for(self, engine: QuestionEngine) -> str | None:
        if engine is self.extractive_engine and self.llm_engine is not None:
            return FALLBACK_NOTICE
        return None

    # ----- generation ----------------------------------------------------------------------

    def generate(
        self, session: Session, plan: QuestionPlan, topics: list[Topic], chunks: list[Chunk], engine: QuestionEngine
    ) -> Question:
        """Generate one grounded, non-duplicate question; raises InsufficientContextError when
        no passage supports one. LLMUnavailableError propagates so the caller can fall back."""
        topic_labels = {t.topic_id: t.label for t in topics}
        previous = [a.question for a in session.answers]
        problems: list[str] = []
        started = time.monotonic()
        seeds = self._seed_order(session, plan, chunks)
        # Short documents have few passages: cycle through them again, and engines use the
        # attempt number (variant) to pick a different sentence or phrasing each time.
        for attempt in range(min(self.max_attempts, len(seeds) * self.max_attempts)):
            seed = seeds[attempt % len(seeds)]
            request = GenerationRequest(
                question_type=plan.question_type,
                difficulty=plan.difficulty,
                topic_id=seed.topic_id if seed.topic_id is not None else plan.topic_id,
                topic_label=topic_labels.get(seed.topic_id, "general"),
                seed=seed,
                context=self.retriever.context_for_seed(seed),
                avoid_questions=[q.question for q in previous],
                variant=attempt,
            )
            try:
                question = engine.generate(request)
            except (InsufficientContextError, LLMResponseError) as exc:
                problems.append(f"{seed.chunk_id}: {exc.message} {exc.detail or ''}".strip())
                logger.info("Attempt %d rejected after %.1fs: %s", attempt + 1, time.monotonic() - started, problems[-1][:200])
                continue
            if self._is_duplicate(question, previous, allow_same_evidence=plan.reinforce):
                problems.append(f"{seed.chunk_id}: duplicate of an earlier question")
                logger.info("Attempt %d rejected: duplicate question", attempt + 1)
                continue
            logger.info(
                "Question ready in %.1fs (attempt %d, %s, %s)",
                time.monotonic() - started, attempt + 1, engine.name, plan.question_type.value,
            )
            return question
        logger.info("Question generation exhausted attempts: %s", problems)
        raise InsufficientContextError(
            "Couldn't build another well-grounded question of this kind from the document. "
            "Try allowing more question types, or finish the session to see your summary.",
            detail=" | ".join(problems)[:2000],
        )

    @staticmethod
    def _seed_order(session: Session, plan: QuestionPlan, chunks: list[Chunk]) -> list[Chunk]:
        """Planned topic's passages first (least used first), then the rest of the document.
        When reinforcing, the passage behind the missed question is tried first."""

        def usage_key(chunk: Chunk) -> tuple[int, int]:
            return session.seed_usage.get(chunk.chunk_id, 0), chunk.chunk_index

        in_topic = sorted((c for c in chunks if c.topic_id == plan.topic_id), key=usage_key)
        others = sorted((c for c in chunks if c.topic_id != plan.topic_id), key=usage_key)
        ordered = in_topic + others
        if plan.reinforce and session.answers:
            missed_seed = session.answers[-1].question.seed_chunk_id
            ordered.sort(key=lambda c: c.chunk_id != missed_seed)  # stable: moves it to the front
        return ordered

    def _is_duplicate(self, question: Question, previous: list[Question], allow_same_evidence: bool) -> bool:
        if not previous:
            return False
        text = normalize_answer(question.question)
        if any(normalize_answer(p.question) == text for p in previous):
            return True
        if allow_same_evidence:
            # Reinforcement deliberately re-tests the same fact in a new form; only exact repeats count.
            return False
        if any(p.evidence == question.evidence for p in previous):
            return True
        vectors = self.embedder.embed_documents([question.question] + [p.question for p in previous])
        similarities = vectors[1:] @ vectors[0]
        return bool(np.max(similarities) >= self.duplicate_similarity)
