"""LLM-backed question engine: retrieved context -> structured draft -> grounding validation."""

from __future__ import annotations

import logging

from app.errors import InsufficientContextError, LLMResponseError
from app.llm.base import StructuredLLM
from app.questions.engine import GenerationRequest, build_question
from app.questions.models import Question, QuestionDraft, QuestionType
from app.questions.prompts import QUESTION_SYSTEM_PROMPT, question_user_prompt
from app.questions.validator import DraftRejected, DraftValidator

logger = logging.getLogger(__name__)


class LLMQuestionEngine:
    supported_types = frozenset(QuestionType)

    def __init__(self, llm: StructuredLLM, temperature: float) -> None:
        self.llm = llm
        self.temperature = temperature
        self.validator = DraftValidator()
        self.name = f"llm:{llm.name}"

    def generate(self, request: GenerationRequest) -> Question:
        prompt = question_user_prompt(
            request.question_type, request.difficulty, request.topic_label, request.context, request.avoid_questions
        )
        draft = self.llm.generate(QuestionDraft, QUESTION_SYSTEM_PROMPT, prompt, self.temperature)
        if draft.insufficient_context:
            raise InsufficientContextError(
                "The selected passage does not support this kind of question.",
                detail=f"seed={request.seed.chunk_id} type={request.question_type.value}",
            )
        try:
            valid = self.validator.validate(draft, request.question_type, request.context)
        except DraftRejected as exc:
            logger.info("Rejected draft for %s: %s", request.seed.chunk_id, exc.problems)
            raise LLMResponseError("The generated question failed grounding checks.", detail=str(exc)) from exc

        return build_question(
            request,
            question=valid.question,
            expected_answer=valid.answer,
            explanation=valid.explanation,
            evidence=valid.evidence,
            source_chunks=valid.source_chunks,
            generated_by=self.name,
            options=valid.options,
            acceptable_answers=valid.acceptable_answers,
            key_points=valid.key_points,
        )
