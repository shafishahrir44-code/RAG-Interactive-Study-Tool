from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Verdict(str, Enum):
    CORRECT = "correct"
    PARTIALLY_CORRECT = "partially_correct"
    INCORRECT = "incorrect"

    @property
    def credit(self) -> float:
        """Score contribution used by the session and adaptive policy."""
        return {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}[self.value]


class Evaluation(BaseModel):
    verdict: Verdict
    score: float = Field(ge=0.0, le=1.0)
    feedback: str
    correct_points: list[str] = Field(default_factory=list)
    missing_points: list[str] = Field(default_factory=list)
    incorrect_points: list[str] = Field(default_factory=list)
    evaluated_by: str


class GradeDraft(BaseModel):
    """Structured object the LLM grader must return."""

    verdict: Verdict
    score: float = Field(ge=0.0, le=1.0, description="0 = wrong, 1 = fully correct")
    correct_points: list[str] = Field(default_factory=list, description="key points the learner got right")
    missing_points: list[str] = Field(default_factory=list, description="key points the learner omitted")
    incorrect_points: list[str] = Field(default_factory=list, description="claims that contradict the reference")
    feedback: str = Field(description="2-3 sentences addressed to the learner")
