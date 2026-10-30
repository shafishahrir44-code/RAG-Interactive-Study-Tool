from __future__ import annotations

from app.adaptive.policy import TYPES_BY_DIFFICULTY, AdaptivePolicy
from app.config import AdaptiveSettings
from app.evaluation.models import Evaluation, Verdict
from app.questions.models import Difficulty, Question, QuestionType
from app.retrieval.topics import Topic
from app.sessions.models import AnswerRecord, Session

policy = AdaptivePolicy(AdaptiveSettings())
ALL_TYPES = frozenset(QuestionType)
TOPICS = [
    Topic(topic_id=i, label=f"topic {i}", keywords=[], chunk_ids=[f"c{i}"], pages=[i + 1], has_numeric_content=(i == 2))
    for i in range(3)
]


def new_session(difficulty=Difficulty.EASY, adaptive=True, types=None) -> Session:
    return Session(session_id="s", document_id="d", target_questions=20, adaptive=adaptive,
                   allowed_types=types or list(QuestionType), difficulty=difficulty)


def answer(session: Session, verdict: Verdict, topic_id=0, qtype=QuestionType.MULTIPLE_CHOICE) -> None:
    q = Question(question_id=str(len(session.answers)), question="q", question_type=qtype, difficulty=session.difficulty,
                 expected_answer="a", explanation="e", evidence="ev", source_chunks=[f"c{topic_id}"],
                 source_pages=[topic_id + 1], sources=[], topic=f"topic {topic_id}", topic_id=topic_id,
                 seed_chunk_id=f"c{topic_id}", generated_by="t")
    session.answers.append(AnswerRecord(question=q, user_answer="x",
                                        evaluation=Evaluation(verdict=verdict, score=verdict.credit, feedback="", evaluated_by="t")))
    stats = session.stats_for(topic_id)
    stats.attempts += 1
    stats.credit += verdict.credit
    stats.last_asked = len(session.answers) - 1
    policy.update_difficulty(session)


def test_strong_performance_raises_difficulty_with_explanation() -> None:
    session = new_session()
    answer(session, Verdict.CORRECT)
    assert session.difficulty == Difficulty.EASY  # needs two answers before changing
    answer(session, Verdict.CORRECT, topic_id=1)
    assert session.difficulty == Difficulty.MEDIUM
    assert session.adaptation_log[-1].change == "difficulty_up"
    assert "100%" in session.adaptation_log[-1].reason


def test_poor_performance_lowers_difficulty() -> None:
    session = new_session(Difficulty.HARD)
    answer(session, Verdict.INCORRECT)
    answer(session, Verdict.PARTIALLY_CORRECT, topic_id=1)
    assert session.difficulty == Difficulty.MEDIUM


def test_mixed_performance_keeps_difficulty_and_window_resets_after_change() -> None:
    session = new_session(Difficulty.MEDIUM)
    for verdict in (Verdict.CORRECT, Verdict.INCORRECT, Verdict.CORRECT):
        answer(session, verdict, topic_id=1)
    assert session.difficulty == Difficulty.MEDIUM
    answer(session, Verdict.CORRECT, topic_id=2)  # last 3 at medium: 1, 0, 1, 1 -> window = 0,1,1 = 0.67
    assert session.difficulty == Difficulty.MEDIUM


def test_fixed_difficulty_when_not_adaptive() -> None:
    session = new_session(adaptive=False)
    for _ in range(4):
        answer(session, Verdict.CORRECT)
    assert session.difficulty == Difficulty.EASY


def test_first_question_covers_first_topic() -> None:
    plan = policy.plan_next(new_session(), TOPICS, ALL_TYPES)
    assert plan.topic_id == 0 and not plan.reinforce
    assert plan.question_type in TYPES_BY_DIFFICULTY[Difficulty.EASY]


def test_wrong_answer_triggers_reinforcement_with_recognition_type() -> None:
    session = new_session()
    answer(session, Verdict.INCORRECT, topic_id=1, qtype=QuestionType.FILL_IN_BLANK)
    plan = policy.plan_next(session, TOPICS, ALL_TYPES)
    assert plan.topic_id == 1 and plan.reinforce
    assert plan.question_type == QuestionType.MULTIPLE_CHOICE
    assert "Reinforcing" in plan.rationale


def test_reinforcement_stops_after_two_questions_in_a_row() -> None:
    session = new_session()
    answer(session, Verdict.INCORRECT, topic_id=1)
    answer(session, Verdict.INCORRECT, topic_id=1)
    plan = policy.plan_next(session, TOPICS, ALL_TYPES)
    assert plan.topic_id != 1


def test_correct_answers_move_to_least_practised_topic() -> None:
    session = new_session()
    answer(session, Verdict.CORRECT, topic_id=0)
    answer(session, Verdict.CORRECT, topic_id=1)
    assert policy.plan_next(session, TOPICS, ALL_TYPES).topic_id == 2


def test_weak_topic_is_revisited_after_a_gap() -> None:
    session = new_session()
    answer(session, Verdict.INCORRECT, topic_id=0)
    answer(session, Verdict.INCORRECT, topic_id=0)
    for topic_id in (1, 2, 1):
        answer(session, Verdict.CORRECT, topic_id=topic_id)
    plan = policy.plan_next(session, TOPICS, ALL_TYPES)
    assert plan.topic_id == 0 and plan.reinforce and "weak area" in plan.rationale


def test_question_types_rotate_and_respect_user_choice() -> None:
    session = new_session(types=[QuestionType.TRUE_FALSE, QuestionType.FILL_IN_BLANK])
    first = policy.plan_next(session, TOPICS, ALL_TYPES).question_type
    answer(session, Verdict.CORRECT, qtype=first)
    second = policy.plan_next(session, TOPICS, ALL_TYPES).question_type
    assert {first, second} == {QuestionType.TRUE_FALSE, QuestionType.FILL_IN_BLANK}


def test_calculation_only_for_numeric_topics() -> None:
    session = new_session(Difficulty.HARD, types=[QuestionType.CALCULATION, QuestionType.CONCEPTUAL])
    session.topic_stats.clear()
    plan_text = policy.plan_next(session, TOPICS[:1], ALL_TYPES)
    plan_numeric = policy.plan_next(session, TOPICS[2:], ALL_TYPES)
    assert plan_text.question_type == QuestionType.CONCEPTUAL
    assert plan_numeric.question_type in {QuestionType.CALCULATION, QuestionType.CONCEPTUAL}


def test_engine_capabilities_limit_types() -> None:
    extractive = frozenset({QuestionType.MULTIPLE_CHOICE, QuestionType.TRUE_FALSE, QuestionType.FILL_IN_BLANK})
    plan = policy.plan_next(new_session(Difficulty.HARD, types=[QuestionType.CONCEPTUAL]), TOPICS, extractive)
    assert plan.question_type in extractive
