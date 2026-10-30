"""Learning-session orchestration: one question at a time, evaluation, adaptation, summary."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from app.adaptive.policy import AdaptivePolicy
from app.config import Settings
from app.documents.models import DocumentStatus
from app.documents.registry import DocumentRegistry
from app.errors import AppError, ConflictError, InvalidInputError, LLMUnavailableError
from app.evaluation.evaluator import AnswerEvaluator
from app.evaluation.models import Evaluation
from app.questions.models import Difficulty, Question, QuestionType
from app.questions.service import QuestionService
from app.sessions.models import AdaptationEvent, AnswerRecord, Session, SessionStatus
from app.sessions.repository import SessionRepository
from app.sessions.summary import SessionSummary, build_summary
from app.vectorstore.base import VectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnswerOutcome:
    session: Session
    record: AnswerRecord
    adaptation: AdaptationEvent | None
    next_step: str


class SessionService:
    def __init__(
        self,
        settings: Settings,
        registry: DocumentRegistry,
        store: VectorStore,
        repository: SessionRepository,
        questions: QuestionService,
        evaluator: AnswerEvaluator,
        policy: AdaptivePolicy,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.store = store
        self.repository = repository
        self.questions = questions
        self.evaluator = evaluator
        self.policy = policy

    def create(
        self,
        document_id: str,
        target_questions: int,
        question_types: list[QuestionType] | None,
        start_difficulty: Difficulty | None,
        adaptive: bool,
    ) -> Session:
        record = self.registry.get(document_id)
        if record.status != DocumentStatus.READY:
            raise ConflictError("This document is not ready yet. Wait for processing to finish.")
        limits = self.settings.limits
        if not limits.min_questions <= target_questions <= limits.max_questions:
            raise InvalidInputError(f"Choose between {limits.min_questions} and {limits.max_questions} questions.")
        session = Session(
            session_id=uuid.uuid4().hex,
            document_id=document_id,
            target_questions=target_questions,
            adaptive=adaptive,
            allowed_types=question_types or list(QuestionType),
            difficulty=start_difficulty or Difficulty(self.settings.adaptive.start_difficulty),
        )
        self.repository.save(session)
        return session

    def get(self, session_id: str) -> Session:
        return self.repository.get(session_id)

    # ----- questions -----------------------------------------------------------------------

    def next_question(self, session_id: str) -> Session:
        """Return the session with a current question. Idempotent: an unanswered question is
        returned again rather than replaced, so users always answer before moving on."""
        with self.repository.lock_for(session_id):
            session = self.repository.get(session_id)
            if session.status == SessionStatus.COMPLETED:
                raise ConflictError("This session is complete. View the summary or start a new session.")
            if session.current_question is not None:
                return session
            question, rationale = self._generate(session)
            session.current_question = question
            session.current_rationale = rationale
            session.seed_usage[question.seed_chunk_id] = session.seed_usage.get(question.seed_chunk_id, 0) + 1
            stats = session.stats_for(question.topic_id)
            stats.last_asked = session.answered
            self.repository.save(session)
            return session

    def prefetch_next(self, session_id: str) -> None:
        """Generate the next question in the background (while the learner reads feedback).
        Failures are only logged: the learner's own /next request retries and reports them."""
        try:
            self.next_question(session_id)
        except AppError as exc:
            logger.info("Prefetch skipped for %s: %s", session_id, exc.message)
        except Exception:  # noqa: BLE001 - a background task must never crash the worker
            logger.exception("Prefetch failed for %s", session_id)

    def _generate(self, session: Session) -> tuple[Question, str]:
        record = self.registry.get(session.document_id)
        chunks = self.store.get_chunks(session.document_id)
        engine = self.questions.active_engine()
        try:
            plan = self.policy.plan_next(session, record.topics, engine.supported_types)
            question = self.questions.generate(session, plan, record.topics, chunks, engine)
        except LLMUnavailableError as exc:
            if not self.questions.fallback_to_extractive:
                raise
            # Log the real cause (wrong model name, bad key, rate limit, timeout...) for debugging.
            logger.warning("LLM unavailable (%s | %s); falling back to extractive questions", exc.message, exc.detail)
            self.questions.mark_llm_unavailable()
            engine = self.questions.active_engine()
            plan = self.policy.plan_next(session, record.topics, engine.supported_types)
            question = self.questions.generate(session, plan, record.topics, chunks, engine)
        session.engine_notice = self.questions.notice_for(engine)
        return question, plan.rationale

    # ----- answers -------------------------------------------------------------------------

    def submit_answer(self, session_id: str, question_id: str, answer: str) -> AnswerOutcome:
        with self.repository.lock_for(session_id):
            session = self.repository.get(session_id)
            question = session.current_question
            if question is None or question.question_id != question_id:
                raise ConflictError("That question is no longer active. Load the current question and try again.")
            evaluation = self.evaluator.evaluate(question, answer)
            record = self._record(session, question, answer.strip(), evaluation)
            adaptation = self.policy.update_difficulty(session)
            if session.answered >= session.target_questions:
                session.status = SessionStatus.COMPLETED
            next_step = self._next_step(session, adaptation)
            self.repository.save(session)
            return AnswerOutcome(session=session, record=record, adaptation=adaptation, next_step=next_step)

    @staticmethod
    def _record(session: Session, question: Question, answer: str, evaluation: Evaluation) -> AnswerRecord:
        record = AnswerRecord(question=question, user_answer=answer, evaluation=evaluation)
        session.answers.append(record)
        session.current_question = None
        session.current_rationale = None
        stats = session.stats_for(question.topic_id)
        stats.attempts += 1
        stats.credit += evaluation.verdict.credit
        return record

    def _next_step(self, session: Session, adaptation: AdaptationEvent | None) -> str:
        if session.status == SessionStatus.COMPLETED:
            return "That was the last question. Open your session summary to review your results."
        record = self.registry.get(session.document_id)
        preview = self.policy.plan_next(session, record.topics, self.questions.active_engine().supported_types)
        prefix = f"{adaptation.reason} " if adaptation else ""
        return f"{prefix}Next: {preview.rationale}"

    def finish(self, session_id: str) -> Session:
        with self.repository.lock_for(session_id):
            session = self.repository.get(session_id)
            session.status = SessionStatus.COMPLETED
            session.current_question = None
            self.repository.save(session)
            return session

    def summary(self, session_id: str) -> SessionSummary:
        session = self.repository.get(session_id)
        record = self.registry.get(session.document_id)
        return build_summary(session, record.topics, self.settings.adaptive.weak_topic_threshold)
