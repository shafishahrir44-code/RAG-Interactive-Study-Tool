"""Answer evaluation.

Objective types are graded deterministically. Free-form answers are graded semantically by
the LLM against the reference answer, key points and the document evidence; if the LLM is
unavailable or misbehaves, a transparent heuristic grader is used instead.
"""

from __future__ import annotations

import logging

from app.errors import InvalidInputError, LLMResponseError, LLMUnavailableError
from app.evaluation.models import Evaluation, GradeDraft, Verdict
from app.evaluation.rules import (
    grade_fill_in_blank,
    grade_free_form_heuristic,
    grade_multiple_choice,
    grade_true_false,
)
from app.llm.base import StructuredLLM
from app.questions.models import Question, QuestionType

logger = logging.getLogger(__name__)

MAX_ANSWER_CHARS = 4000
# Score bands the LLM's verdict must agree with; inconsistent grades are re-derived from the score.
VERDICT_SCORE_BANDS = {
    Verdict.CORRECT: (0.75, 1.0),
    Verdict.PARTIALLY_CORRECT: (0.25, 0.85),
    Verdict.INCORRECT: (0.0, 0.4),
}

GRADER_SYSTEM_PROMPT = """You grade a learner's answer to a study question.
Judge ONLY against the reference answer, key points and document evidence provided.
Accept paraphrases and synonyms: grade meaning, not wording. Do not reward extra claims that are
not in the evidence, and mark claims that contradict the evidence as incorrect_points.
Verdicts: "correct" = all key points present and nothing wrong; "partially_correct" = some key
points present or minor errors; "incorrect" = key points missing or wrong.
Address the learner as "you" in the feedback and say concretely what was missing or wrong."""


def _grader_prompt(question: Question, answer: str) -> str:
    key_points = "\n".join(f"- {k}" for k in question.key_points) or f"- {question.expected_answer}"
    return (
        f"QUESTION ({question.question_type.label}):\n{question.question}\n\n"
        f"REFERENCE ANSWER:\n{question.expected_answer}\n\n"
        f"KEY POINTS:\n{key_points}\n\n"
        f"DOCUMENT EVIDENCE (pages {', '.join(map(str, question.source_pages))}):\n{question.evidence}\n\n"
        f"LEARNER ANSWER:\n{answer}"
    )


def _verdict_from_score(score: float) -> Verdict:
    if score >= VERDICT_SCORE_BANDS[Verdict.CORRECT][0]:
        return Verdict.CORRECT
    if score >= VERDICT_SCORE_BANDS[Verdict.PARTIALLY_CORRECT][0]:
        return Verdict.PARTIALLY_CORRECT
    return Verdict.INCORRECT


class AnswerEvaluator:
    def __init__(self, llm: StructuredLLM | None, temperature: float) -> None:
        self.llm = llm
        self.temperature = temperature

    def evaluate(self, question: Question, answer: str) -> Evaluation:
        answer = (answer or "").strip()
        if not answer:
            raise InvalidInputError("Please enter an answer before submitting.")
        if len(answer) > MAX_ANSWER_CHARS:
            raise InvalidInputError(f"Answers are limited to {MAX_ANSWER_CHARS} characters.")

        if question.question_type == QuestionType.MULTIPLE_CHOICE:
            return grade_multiple_choice(question, answer)
        if question.question_type == QuestionType.TRUE_FALSE:
            return grade_true_false(question, answer)
        if question.question_type == QuestionType.FILL_IN_BLANK:
            return grade_fill_in_blank(question, answer)
        return self._grade_free_form(question, answer)

    def _grade_free_form(self, question: Question, answer: str) -> Evaluation:
        if self.llm is None:
            return grade_free_form_heuristic(question, answer)
        try:
            draft = self.llm.generate(GradeDraft, GRADER_SYSTEM_PROMPT, _grader_prompt(question, answer), self.temperature)
        except (LLMUnavailableError, LLMResponseError) as exc:
            logger.warning("LLM grading failed, using heuristic grader: %s", exc.detail)
            return grade_free_form_heuristic(question, answer)

        low, high = VERDICT_SCORE_BANDS[draft.verdict]
        verdict = draft.verdict if low <= draft.score <= high else _verdict_from_score(draft.score)
        return Evaluation(
            verdict=verdict,
            score=round(draft.score, 2),
            feedback=draft.feedback.strip() or "See the explanation below.",
            correct_points=draft.correct_points,
            missing_points=draft.missing_points,
            incorrect_points=draft.incorrect_points,
            evaluated_by=f"llm:{self.llm.name}",
        )
