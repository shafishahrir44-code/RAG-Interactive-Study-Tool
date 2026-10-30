"""Session summary statistics."""

from __future__ import annotations

from collections import defaultdict

from pydantic import BaseModel

from app.evaluation.models import Verdict
from app.retrieval.topics import Topic
from app.sessions.models import AdaptationEvent, Session


class Breakdown(BaseModel):
    key: str
    label: str
    attempts: int
    score: float  # mean credit 0..1


class TopicSummary(BaseModel):
    topic_id: int
    label: str
    attempts: int
    score: float
    mastery: float
    pages: list[int]


class QuestionReview(BaseModel):
    number: int
    question: str
    question_type: str
    difficulty: str
    topic: str
    user_answer: str
    expected_answer: str
    verdict: Verdict
    source_pages: list[int]


class SessionSummary(BaseModel):
    session_id: str
    document_id: str
    status: str
    answered: int
    target_questions: int
    correct: int
    partially_correct: int
    incorrect: int
    score: float  # mean credit 0..1
    final_difficulty: str
    difficulty_path: list[str]
    by_topic: list[TopicSummary]
    by_type: list[Breakdown]
    by_difficulty: list[Breakdown]
    weak_topics: list[TopicSummary]
    pages_to_review: list[int]
    pages_covered: list[int]
    adaptation_log: list[AdaptationEvent]
    review: list[QuestionReview]


def _breakdown(groups: dict[str, list[float]], labels: dict[str, str]) -> list[Breakdown]:
    return [
        Breakdown(key=key, label=labels.get(key, key), attempts=len(credits), score=round(sum(credits) / len(credits), 3))
        for key, credits in groups.items()
    ]


def build_summary(session: Session, topics: list[Topic], weak_threshold: float) -> SessionSummary:
    answers = session.answers
    verdicts = [a.evaluation.verdict for a in answers]
    credits = [v.credit for v in verdicts]

    by_type: dict[str, list[float]] = defaultdict(list)
    by_difficulty: dict[str, list[float]] = defaultdict(list)
    topic_pages: dict[int, set[int]] = defaultdict(set)
    for answer in answers:
        q = answer.question
        by_type[q.question_type.value].append(answer.evaluation.verdict.credit)
        by_difficulty[q.difficulty.value].append(answer.evaluation.verdict.credit)
        topic_pages[q.topic_id].update(q.source_pages)

    labels = {t.topic_id: t.label for t in topics}
    by_topic = [
        TopicSummary(
            topic_id=topic_id,
            label=labels.get(topic_id, f"topic {topic_id}"),
            attempts=stats.attempts,
            score=round(stats.credit / stats.attempts, 3),
            mastery=round(stats.mastery, 3),
            pages=sorted(topic_pages[topic_id]),
        )
        for topic_id, stats in sorted(session.topic_stats.items())
        if stats.attempts
    ]
    weak = [t for t in by_topic if t.mastery < weak_threshold]
    missed_pages = {p for a in answers if a.evaluation.verdict != Verdict.CORRECT for p in a.question.source_pages}
    type_labels = {a.question.question_type.value: a.question.question_type.label for a in answers}

    return SessionSummary(
        session_id=session.session_id,
        document_id=session.document_id,
        status=session.status.value,
        answered=len(answers),
        target_questions=session.target_questions,
        correct=verdicts.count(Verdict.CORRECT),
        partially_correct=verdicts.count(Verdict.PARTIALLY_CORRECT),
        incorrect=verdicts.count(Verdict.INCORRECT),
        score=round(sum(credits) / len(credits), 3) if credits else 0.0,
        final_difficulty=session.difficulty.value,
        difficulty_path=[a.question.difficulty.value for a in answers],
        by_topic=by_topic,
        by_type=_breakdown(by_type, type_labels),
        by_difficulty=_breakdown(by_difficulty, {}),
        weak_topics=weak,
        pages_to_review=sorted(missed_pages),
        pages_covered=sorted({p for a in answers for p in a.question.source_pages}),
        adaptation_log=session.adaptation_log,
        review=[
            QuestionReview(
                number=i + 1,
                question=a.question.question,
                question_type=a.question.question_type.label,
                difficulty=a.question.difficulty.value,
                topic=a.question.topic,
                user_answer=a.user_answer,
                expected_answer=a.question.expected_answer,
                verdict=a.evaluation.verdict,
                source_pages=a.question.source_pages,
            )
            for i, a in enumerate(answers)
        ],
    )
