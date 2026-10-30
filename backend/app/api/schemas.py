"""Request/response models for the HTTP API.

Questions are exposed in two shapes: `PublicQuestion` (no answer, shown while answering)
and `RevealedQuestion` (answer, explanation and sources, shown after submission).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.documents.models import DocumentRecord
from app.evaluation.models import Evaluation
from app.questions.models import Difficulty, Question, QuestionType, SourceRef
from app.sessions.models import AdaptationEvent, Session


class ErrorBody(BaseModel):
    code: str
    message: str
    detail: str | None = None


class HealthResponse(BaseModel):
    status: str
    llm_provider: str
    llm_model: str | None
    llm_available: bool
    embedding: str
    vector_store: str
    question_types_available: list[QuestionType]
    max_upload_mb: int
    notice: str | None = None


class DocumentResponse(BaseModel):
    document: DocumentRecord
    reused: bool = False


class CreateSessionRequest(BaseModel):
    document_id: str = Field(min_length=8, max_length=128)
    num_questions: int = 10
    question_types: list[QuestionType] | None = None
    start_difficulty: Difficulty | None = None
    adaptive: bool = True


class PublicQuestion(BaseModel):
    question_id: str
    number: int
    question: str
    question_type: QuestionType
    type_label: str
    difficulty: Difficulty
    options: list[str]
    topic: str
    rationale: str | None
    generated_by: str

    @classmethod
    def from_session(cls, session: Session) -> "PublicQuestion":
        q = session.current_question
        assert q is not None
        return cls(
            question_id=q.question_id,
            number=session.answered + 1,
            question=q.question,
            question_type=q.question_type,
            type_label=q.question_type.label,
            difficulty=q.difficulty,
            options=q.options,
            topic=q.topic,
            rationale=session.current_rationale,
            generated_by=q.generated_by,
        )


class RevealedQuestion(BaseModel):
    question_id: str
    question: str
    question_type: QuestionType
    difficulty: Difficulty
    options: list[str]
    expected_answer: str
    acceptable_answers: list[str]
    key_points: list[str]
    explanation: str
    evidence: str
    source_pages: list[int]
    sources: list[SourceRef]
    topic: str

    @classmethod
    def from_question(cls, q: Question) -> "RevealedQuestion":
        return cls(**q.model_dump(include=set(cls.model_fields)))


class Progress(BaseModel):
    answered: int
    target: int
    correct: int
    partially_correct: int
    incorrect: int
    score: float
    difficulty: Difficulty
    status: str

    @classmethod
    def from_session(cls, session: Session) -> "Progress":
        verdicts = [a.evaluation.verdict.value for a in session.answers]
        credits = [a.evaluation.verdict.credit for a in session.answers]
        return cls(
            answered=session.answered,
            target=session.target_questions,
            correct=verdicts.count("correct"),
            partially_correct=verdicts.count("partially_correct"),
            incorrect=verdicts.count("incorrect"),
            score=round(sum(credits) / len(credits), 3) if credits else 0.0,
            difficulty=session.difficulty,
            status=session.status.value,
        )


class SessionResponse(BaseModel):
    session_id: str
    document_id: str
    adaptive: bool
    allowed_types: list[QuestionType]
    progress: Progress
    current_question: PublicQuestion | None
    engine_notice: str | None

    @classmethod
    def from_session(cls, session: Session) -> "SessionResponse":
        return cls(
            session_id=session.session_id,
            document_id=session.document_id,
            adaptive=session.adaptive,
            allowed_types=session.allowed_types,
            progress=Progress.from_session(session),
            current_question=PublicQuestion.from_session(session) if session.current_question else None,
            engine_notice=session.engine_notice,
        )


class SubmitAnswerRequest(BaseModel):
    question_id: str = Field(min_length=1, max_length=64)
    answer: str = Field(max_length=4000)


class AnswerResponse(BaseModel):
    user_answer: str
    evaluation: Evaluation
    question: RevealedQuestion
    adaptation: AdaptationEvent | None
    next_step: str
    progress: Progress
    session_completed: bool
