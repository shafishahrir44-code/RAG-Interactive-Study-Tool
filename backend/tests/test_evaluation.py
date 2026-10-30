from __future__ import annotations

import pytest

from app.errors import InvalidInputError, LLMUnavailableError
from app.evaluation.evaluator import AnswerEvaluator
from app.evaluation.models import Verdict
from app.llm.base import StructuredLLM
from app.questions.models import Difficulty, Question, QuestionType
from tests.fakes import FakeGroundedLLM


def question(qtype: QuestionType, expected: str, **extra) -> Question:
    return Question(
        question_id="q1",
        question="Q?",
        question_type=qtype,
        difficulty=Difficulty.EASY,
        expected_answer=expected,
        explanation="Because the document says so.",
        evidence="Glycolysis takes place in the cytoplasm.",
        source_chunks=["c1"],
        source_pages=[2],
        sources=[],
        topic="respiration",
        topic_id=0,
        seed_chunk_id="c1",
        generated_by="test",
        **extra,
    )


rules = AnswerEvaluator(llm=None, temperature=0.0)


def test_multiple_choice_accepts_text_or_letter() -> None:
    q = question(QuestionType.MULTIPLE_CHOICE, "Cytoplasm", options=["Nucleus", "Cytoplasm", "Stroma", "Golgi"])
    assert rules.evaluate(q, "cytoplasm").verdict == Verdict.CORRECT
    assert rules.evaluate(q, "B").verdict == Verdict.CORRECT
    wrong = rules.evaluate(q, "Nucleus")
    assert wrong.verdict == Verdict.INCORRECT and "Cytoplasm" in wrong.feedback


def test_true_false() -> None:
    q = question(QuestionType.TRUE_FALSE, "False")
    assert rules.evaluate(q, "false").verdict == Verdict.CORRECT
    assert rules.evaluate(q, "True").verdict == Verdict.INCORRECT


@pytest.mark.parametrize(
    ("answer", "verdict"),
    [
        ("cytoplasm", Verdict.CORRECT),
        ("The Cytoplasm.", Verdict.CORRECT),
        ("cytoplasn", Verdict.CORRECT),  # typo tolerance
        ("nucleus", Verdict.INCORRECT),
    ],
)
def test_fill_in_blank_is_tolerant_but_strict_on_meaning(answer, verdict) -> None:
    assert rules.evaluate(question(QuestionType.FILL_IN_BLANK, "cytoplasm"), answer).verdict == verdict


def test_fill_in_blank_partial_credit_for_multiword_terms() -> None:
    result = rules.evaluate(question(QuestionType.FILL_IN_BLANK, "electron transport chain"), "transport chain")
    assert result.verdict == Verdict.PARTIALLY_CORRECT


def test_heuristic_free_form_grading_uses_key_points() -> None:
    q = question(
        QuestionType.CONCEPTUAL,
        "Glycolysis splits glucose in the cytoplasm and yields 2 ATP.",
        key_points=["glycolysis happens in the cytoplasm", "glycolysis yields 2 ATP"],
    )
    full = rules.evaluate(q, "It happens in the cytoplasm and glycolysis yields a net 2 ATP.")
    half = rules.evaluate(q, "Glycolysis happens in the cytoplasm.")
    none = rules.evaluate(q, "It happens inside the nucleus.")
    assert (full.verdict, half.verdict, none.verdict) == (Verdict.CORRECT, Verdict.PARTIALLY_CORRECT, Verdict.INCORRECT)
    assert half.missing_points == ["glycolysis yields 2 ATP"]


def test_heuristic_calculation_checks_the_final_number() -> None:
    q = question(QuestionType.CALCULATION, "46 / 2 = 23 chromosomes", key_points=["halve 46", "23"])
    assert rules.evaluate(q, "23").verdict == Verdict.CORRECT
    assert rules.evaluate(q, "It would be 24").verdict != Verdict.CORRECT


def test_llm_grader_used_for_free_form() -> None:
    evaluator = AnswerEvaluator(StructuredLLM(FakeGroundedLLM()), 0.0)
    q = question(QuestionType.SHORT_ANSWER, "glycolysis occurs in the cytoplasm", key_points=["cytoplasm"])
    result = evaluator.evaluate(q, "glycolysis occurs in the cytoplasm of the cell")
    assert result.verdict == Verdict.CORRECT and result.evaluated_by.startswith("llm:")


def test_inconsistent_llm_verdict_is_rederived_from_score() -> None:
    fake = FakeGroundedLLM(override=lambda s, u: '{"verdict": "correct", "score": 0.1, "feedback": "ok"}')
    result = AnswerEvaluator(StructuredLLM(fake), 0.0).evaluate(question(QuestionType.SHORT_ANSWER, "x y z"), "something")
    assert result.verdict == Verdict.INCORRECT


def test_llm_outage_falls_back_to_heuristic_grader() -> None:
    def down(system, user):
        raise LLMUnavailableError("down")

    evaluator = AnswerEvaluator(StructuredLLM(FakeGroundedLLM(override=down)), 0.0)
    q = question(QuestionType.SHORT_ANSWER, "cytoplasm", key_points=["cytoplasm"])
    assert evaluator.evaluate(q, "in the cytoplasm").evaluated_by == "heuristic"


@pytest.mark.parametrize("answer", ["", "   ", "x" * 5000])
def test_invalid_answers_are_rejected(answer) -> None:
    with pytest.raises(InvalidInputError):
        rules.evaluate(question(QuestionType.SHORT_ANSWER, "x"), answer)
