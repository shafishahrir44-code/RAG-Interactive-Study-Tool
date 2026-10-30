"""Deterministic graders: objective types, fill-in-the-blank, and a heuristic free-form fallback."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from app.evaluation.models import Evaluation, Verdict
from app.questions.models import Question
from app.questions.validator import parse_bool
from app.text_utils import content_tokens, normalize_answer, token_overlap

RULE_GRADER = "rules"
HEURISTIC_GRADER = "heuristic"
TYPO_TOLERANCE = 0.85  # character similarity accepted as a misspelling of the right answer
PARTIAL_TOKEN_OVERLAP = 0.5
KEY_POINT_COVERAGE = 0.6  # token overlap needed to count a key point as covered
CORRECT_SCORE = 0.75
PARTIAL_SCORE = 0.35
NUMERIC_TOLERANCE = 0.01  # 1 % relative error
_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)*(?:\.\d+)?")


def _result(verdict: Verdict, score: float, feedback: str, grader: str = RULE_GRADER, **points) -> Evaluation:
    return Evaluation(verdict=verdict, score=score, feedback=feedback, evaluated_by=grader, **points)


def grade_multiple_choice(question: Question, answer: str) -> Evaluation:
    chosen = answer.strip()
    letter = chosen.upper().rstrip(").")
    if len(letter) == 1 and "A" <= letter < chr(ord("A") + len(question.options)):
        chosen = question.options[ord(letter) - ord("A")]
    if normalize_answer(chosen) == normalize_answer(question.expected_answer):
        return _result(Verdict.CORRECT, 1.0, "Correct — you picked the right option.")
    return _result(Verdict.INCORRECT, 0.0, f"You chose “{chosen}”, but the correct option is “{question.expected_answer}”.")


def grade_true_false(question: Question, answer: str) -> Evaluation:
    given, expected = parse_bool(answer), parse_bool(question.expected_answer)
    if given is None:
        return _result(Verdict.INCORRECT, 0.0, "Please answer True or False.")
    if given == expected:
        return _result(Verdict.CORRECT, 1.0, f"Correct — the statement is {question.expected_answer.lower()}.")
    return _result(Verdict.INCORRECT, 0.0, f"Not quite — the statement is {question.expected_answer.lower()}.")


def grade_fill_in_blank(question: Question, answer: str) -> Evaluation:
    given = normalize_answer(answer)
    accepted = [normalize_answer(a) for a in [question.expected_answer, *question.acceptable_answers]]
    if given in accepted:
        return _result(Verdict.CORRECT, 1.0, "Correct — that is the missing term.")
    best = max(SequenceMatcher(None, given, a).ratio() for a in accepted)
    if best >= TYPO_TOLERANCE:
        return _result(Verdict.CORRECT, 1.0, f"Correct (allowing for spelling) — the term is “{question.expected_answer}”.")
    if max(token_overlap(a, given) for a in accepted) >= PARTIAL_TOKEN_OVERLAP:
        return _result(
            Verdict.PARTIALLY_CORRECT,
            0.5,
            f"Partly right — the full term is “{question.expected_answer}”.",
            missing_points=[question.expected_answer],
        )
    return _result(Verdict.INCORRECT, 0.0, f"The missing term is “{question.expected_answer}”.")


def _numbers(text: str) -> list[float]:
    values = []
    for raw in _NUMBER_RE.findall(text):
        try:
            values.append(float(raw.replace(",", "")))
        except ValueError:
            continue
    return values


def _numbers_match(expected: str, answer: str) -> bool:
    targets, given = _numbers(expected), _numbers(answer)
    if not targets or not given:
        return False
    final = targets[-1]  # the final result is conventionally stated last
    return any(abs(g - final) <= NUMERIC_TOLERANCE * max(abs(final), 1e-9) for g in given)


def grade_free_form_heuristic(question: Question, answer: str) -> Evaluation:
    """Fallback when no LLM grader is available: key-point coverage plus numeric checking."""
    if question.question_type.value == "calculation" and _numbers_match(question.expected_answer, answer):
        return _result(Verdict.CORRECT, 1.0, "Your final value matches the expected result.", HEURISTIC_GRADER)

    key_points = question.key_points or [question.expected_answer]
    covered = [kp for kp in key_points if token_overlap(kp, answer) >= KEY_POINT_COVERAGE]
    missing = [kp for kp in key_points if kp not in covered]
    score = len(covered) / len(key_points)
    if not content_tokens(answer):
        score = 0.0
    if score >= CORRECT_SCORE:
        verdict, feedback = Verdict.CORRECT, "Your answer covers the key points from the document."
    elif score >= PARTIAL_SCORE:
        verdict, feedback = Verdict.PARTIALLY_CORRECT, "You covered some key points but missed others."
    else:
        verdict, feedback = Verdict.INCORRECT, "Your answer does not match the key points in the document."
    return _result(verdict, round(score, 2), feedback, HEURISTIC_GRADER, correct_points=covered, missing_points=missing)
