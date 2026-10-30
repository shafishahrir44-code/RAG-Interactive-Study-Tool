"""Question data models."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class QuestionType(str, Enum):
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    FILL_IN_BLANK = "fill_in_blank"
    SHORT_ANSWER = "short_answer"
    CONCEPTUAL = "conceptual"
    CALCULATION = "calculation"

    @property
    def label(self) -> str:
        return _TYPE_LABELS[self]


_TYPE_LABELS = {
    QuestionType.MULTIPLE_CHOICE: "Multiple choice",
    QuestionType.TRUE_FALSE: "True / false",
    QuestionType.FILL_IN_BLANK: "Fill in the blank",
    QuestionType.SHORT_ANSWER: "Short answer",
    QuestionType.CONCEPTUAL: "Conceptual",
    QuestionType.CALCULATION: "Calculation",
}


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

    @property
    def level(self) -> int:
        return _LEVELS.index(self)

    def shifted(self, steps: int) -> "Difficulty":
        return _LEVELS[max(0, min(len(_LEVELS) - 1, self.level + steps))]


_LEVELS = [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD]

BLANK = "_____"


class SourceRef(BaseModel):
    document_id: str
    chunk_id: str
    page_number: int
    excerpt: str


class Question(BaseModel):
    question_id: str
    question: str
    question_type: QuestionType
    difficulty: Difficulty
    options: list[str] = Field(default_factory=list)
    expected_answer: str
    acceptable_answers: list[str] = Field(default_factory=list)
    key_points: list[str] = Field(default_factory=list)
    explanation: str
    evidence: str  # verbatim supporting text from the document
    source_chunks: list[str]
    source_pages: list[int]
    sources: list[SourceRef]
    topic: str
    topic_id: int
    seed_chunk_id: str
    generated_by: str


class QuestionDraft(BaseModel):
    """The structured object an LLM must return when writing a question."""

    insufficient_context: bool = Field(
        default=False, description="true if the context cannot support a good question of this type"
    )
    question: str = ""
    options: list[str] = Field(default_factory=list, description="multiple_choice only: 4 options")
    answer: str = Field(
        default="",
        description="multiple_choice: the exact correct option text; true_false: 'true' or 'false'; "
        "fill_in_blank: the missing word(s); otherwise: a concise model answer",
    )
    acceptable_answers: list[str] = Field(default_factory=list, description="other valid phrasings")
    key_points: list[str] = Field(default_factory=list, description="facts a complete answer must contain")
    explanation: str = Field(default="", description="why the answer is correct, citing the context")
    evidence_quote: str = Field(default="", description="exact sentence(s) copied from the context")
    source_chunk_ids: list[str] = Field(default_factory=list)
