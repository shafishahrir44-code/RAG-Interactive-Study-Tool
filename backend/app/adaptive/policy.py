"""Explainable adaptive-learning policy.

Difficulty: after at least `min_answers_before_change` answers at the current level, the mean
credit (correct=1, partial=0.5, incorrect=0) over the last `window` answers at that level is
compared with thresholds: >= promote -> harder, <= demote -> easier.

Topic selection, in priority order:
  1. Reinforce: the last answer was not fully correct -> stay on that topic (different passage
     or question type) so the concept is revisited immediately.
  2. Remediate: a weak topic (mastery < weak threshold) not asked in the last 2 questions.
  3. Cover: the topic asked least often (ties -> lower mastery -> document order), skipping
     mastered topics while any unmastered topic remains.

Question type: the least recently used type allowed for the difficulty and the topic.
Reinforcement prefers recognition-style types (multiple choice / true-false) because they
re-expose the correct fact with lower cognitive load.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import AdaptiveSettings
from app.evaluation.models import Verdict
from app.questions.models import Difficulty, QuestionType
from app.retrieval.topics import Topic
from app.sessions.models import AdaptationEvent, Session, TopicStats

TYPES_BY_DIFFICULTY: dict[Difficulty, list[QuestionType]] = {
    Difficulty.EASY: [QuestionType.MULTIPLE_CHOICE, QuestionType.TRUE_FALSE, QuestionType.FILL_IN_BLANK],
    Difficulty.MEDIUM: [
        QuestionType.FILL_IN_BLANK,
        QuestionType.MULTIPLE_CHOICE,
        QuestionType.SHORT_ANSWER,
        QuestionType.TRUE_FALSE,
    ],
    Difficulty.HARD: [
        QuestionType.CONCEPTUAL,
        QuestionType.CALCULATION,
        QuestionType.SHORT_ANSWER,
        QuestionType.MULTIPLE_CHOICE,
        QuestionType.FILL_IN_BLANK,
    ],
}
REINFORCEMENT_TYPES = [QuestionType.MULTIPLE_CHOICE, QuestionType.TRUE_FALSE, QuestionType.FILL_IN_BLANK]
RECENT_TOPIC_GAP = 2
# After this many questions in a row on one topic, move on even if answers are still wrong
# (the topic returns later through the weak-topic rule), so a learner is never stuck.
MAX_CONSECUTIVE_QUESTIONS_PER_TOPIC = 2


@dataclass(frozen=True)
class QuestionPlan:
    topic_id: int
    question_type: QuestionType
    difficulty: Difficulty
    reinforce: bool
    rationale: str


class AdaptivePolicy:
    def __init__(self, settings: AdaptiveSettings) -> None:
        self.settings = settings

    # ----- difficulty ----------------------------------------------------------------------

    def update_difficulty(self, session: Session) -> AdaptationEvent | None:
        """Apply the difficulty rule after an answer. Mutates the session; returns the change."""
        if not session.adaptive:
            return None
        recent = session.answers[session.difficulty_since :][-self.settings.window :]
        if len(recent) < self.settings.min_answers_before_change:
            return None
        mean = sum(a.evaluation.verdict.credit for a in recent) / len(recent)
        step, change = 0, ""
        if mean >= self.settings.promote_threshold and session.difficulty != Difficulty.HARD:
            step, change = 1, "difficulty_up"
        elif mean <= self.settings.demote_threshold and session.difficulty != Difficulty.EASY:
            step, change = -1, "difficulty_down"
        if not step:
            return None
        previous = session.difficulty
        session.difficulty = previous.shifted(step)
        session.difficulty_since = session.answered
        direction = "harder" if step > 0 else "easier"
        event = AdaptationEvent(
            after_question=session.answered,
            change=change,
            reason=(
                f"You scored {mean:.0%} on your last {len(recent)} {previous.value} question(s), "
                f"so questions are now {direction} ({session.difficulty.value})."
            ),
        )
        session.adaptation_log.append(event)
        return event

    # ----- next question -------------------------------------------------------------------

    def plan_next(self, session: Session, topics: list[Topic], engine_types: frozenset[QuestionType]) -> QuestionPlan:
        if not topics:
            raise ValueError("document has no topics")
        topic, reinforce, reason = self._choose_topic(session, topics)
        question_type = self._choose_type(session, topic, reinforce, engine_types)
        return QuestionPlan(
            topic_id=topic.topic_id,
            question_type=question_type,
            difficulty=session.difficulty,
            reinforce=reinforce,
            rationale=f"{reason} Difficulty: {session.difficulty.value}.",
        )

    def _choose_topic(self, session: Session, topics: list[Topic]) -> tuple[Topic, bool, str]:
        by_id = {t.topic_id: t for t in topics}
        if session.answers:
            last = session.answers[-1]
            if (
                last.evaluation.verdict != Verdict.CORRECT
                and last.question.topic_id in by_id
                and self._consecutive_on_topic(session, last.question.topic_id) < MAX_CONSECUTIVE_QUESTIONS_PER_TOPIC
            ):
                return (
                    by_id[last.question.topic_id],
                    True,
                    f"Reinforcing “{last.question.topic}” because your last answer was not fully correct.",
                )

        index = session.answered
        weak = [
            t
            for t in topics
            if (stats := session.topic_stats.get(t.topic_id))
            and stats.attempts
            and stats.mastery < self.settings.weak_topic_threshold
            and index - stats.last_asked > RECENT_TOPIC_GAP
        ]
        if weak:
            topic = min(weak, key=lambda t: session.topic_stats[t.topic_id].mastery)
            mastery = session.topic_stats[topic.topic_id].mastery
            return topic, True, f"Revisiting “{topic.label}”, a weak area (mastery {mastery:.0%})."

        def mastered(t: Topic) -> bool:
            stats = session.topic_stats.get(t.topic_id)
            return bool(stats and stats.attempts >= 2 and stats.mastery >= self.settings.strong_topic_threshold)

        def coverage_key(t: Topic) -> tuple[int, float, int]:
            stats = session.topic_stats.get(t.topic_id) or TopicStats()
            return stats.attempts, stats.mastery, t.topic_id

        pool = [t for t in topics if not mastered(t)] or topics
        topic = min(pool, key=coverage_key)
        asked = session.topic_stats.get(topic.topic_id)
        if asked and asked.attempts:
            return topic, False, f"Continuing with “{topic.label}”, one of your least-practised topics."
        return topic, False, f"Covering a new topic: “{topic.label}”."

    @staticmethod
    def _consecutive_on_topic(session: Session, topic_id: int) -> int:
        count = 0
        for answer in reversed(session.answers):
            if answer.question.topic_id != topic_id:
                break
            count += 1
        return count

    def _choose_type(
        self, session: Session, topic: Topic, reinforce: bool, engine_types: frozenset[QuestionType]
    ) -> QuestionType:
        allowed = [t for t in TYPES_BY_DIFFICULTY[session.difficulty] if t in session.allowed_types and t in engine_types]
        if not topic.has_numeric_content:
            allowed = [t for t in allowed if t != QuestionType.CALCULATION]
        if not allowed:  # the user's selection doesn't intersect this level; respect the selection
            allowed = [t for t in session.allowed_types if t in engine_types] or sorted(engine_types, key=lambda t: t.value)
            if not topic.has_numeric_content and len(allowed) > 1:
                allowed = [t for t in allowed if t != QuestionType.CALCULATION] or allowed
        if reinforce:
            preferred = [t for t in REINFORCEMENT_TYPES if t in allowed]
            last_type = session.answers[-1].question.question_type if session.answers else None
            preferred = [t for t in preferred if t != last_type] or preferred
            if preferred:
                return preferred[0]

        last_used = {a.question.question_type: i for i, a in enumerate(session.answers)}
        return min(allowed, key=lambda t: (last_used.get(t, -1), allowed.index(t)))
