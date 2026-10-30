"""Learning-session state."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

from app.evaluation.models import Evaluation
from app.questions.models import Difficulty, Question, QuestionType


def _now() -> datetime:
    return datetime.now(timezone.utc)


class SessionStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"


class TopicStats(BaseModel):
    attempts: int = 0
    credit: float = 0.0  # sum of verdict credits (1 / 0.5 / 0)
    last_asked: int = -1  # index of the question that last used this topic

    @property
    def mastery(self) -> float:
        """Laplace-smoothed accuracy: starts at 0.5 and moves with evidence."""
        return (self.credit + 1) / (self.attempts + 2)


class AnswerRecord(BaseModel):
    question: Question
    user_answer: str
    evaluation: Evaluation
    answered_at: datetime = Field(default_factory=_now)


class AdaptationEvent(BaseModel):
    after_question: int
    change: str  # "difficulty_up" | "difficulty_down" | "reinforce" | ...
    reason: str


class Session(BaseModel):
    session_id: str
    document_id: str
    created_at: datetime = Field(default_factory=_now)
    status: SessionStatus = SessionStatus.ACTIVE
    target_questions: int
    adaptive: bool = True
    allowed_types: list[QuestionType]
    difficulty: Difficulty
    difficulty_since: int = 0  # number of answers when the current difficulty began
    current_question: Question | None = None
    current_rationale: str | None = None
    answers: list[AnswerRecord] = Field(default_factory=list)
    topic_stats: dict[int, TopicStats] = Field(default_factory=dict)
    seed_usage: dict[str, int] = Field(default_factory=dict)
    adaptation_log: list[AdaptationEvent] = Field(default_factory=list)
    engine_notice: str | None = None

    @property
    def answered(self) -> int:
        return len(self.answers)

    def stats_for(self, topic_id: int) -> TopicStats:
        return self.topic_stats.setdefault(topic_id, TopicStats())
