"""Question-engine interface and shared helpers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Protocol

from app.ingestion.models import Chunk
from app.questions.models import Difficulty, Question, QuestionType, SourceRef

EXCERPT_CHARS = 700
EXCERPT_LEAD_CHARS = 200  # context shown before the evidence sentence
EVIDENCE_ANCHOR_CHARS = 40


@dataclass(frozen=True)
class GenerationRequest:
    question_type: QuestionType
    difficulty: Difficulty
    topic_id: int
    topic_label: str
    seed: Chunk
    context: list[Chunk]  # seed first, then retrieved neighbours
    avoid_questions: list[str] = field(default_factory=list)
    variant: int = 0  # increments on retries so engines can vary their choice


class QuestionEngine(Protocol):
    name: str
    supported_types: frozenset[QuestionType]

    def generate(self, request: GenerationRequest) -> Question:
        """Return a validated, grounded question or raise InsufficientContextError / LLMResponseError."""
        ...


def excerpt_around(text: str, evidence: str) -> str:
    """Up to EXCERPT_CHARS of `text`, positioned so the evidence sentence stays visible."""
    if len(text) <= EXCERPT_CHARS:
        return text
    anchor = text.lower().find(evidence[:EVIDENCE_ANCHOR_CHARS].lower()) if evidence else -1
    start = max(0, min(anchor - EXCERPT_LEAD_CHARS, len(text) - EXCERPT_CHARS)) if anchor >= 0 else 0
    end = start + EXCERPT_CHARS
    return ("…" if start else "") + text[start:end].strip() + ("…" if end < len(text) else "")


def source_refs(chunks: list[Chunk], evidence: str = "") -> list[SourceRef]:
    return [
        SourceRef(
            document_id=c.document_id,
            chunk_id=c.chunk_id,
            page_number=c.page_number,
            excerpt=excerpt_around(c.text, evidence),
        )
        for c in chunks
    ]


def build_question(
    request: GenerationRequest,
    *,
    question: str,
    expected_answer: str,
    explanation: str,
    evidence: str,
    source_chunks: list[Chunk],
    generated_by: str,
    options: list[str] | None = None,
    acceptable_answers: list[str] | None = None,
    key_points: list[str] | None = None,
) -> Question:
    digest = hashlib.sha1(f"{request.seed.chunk_id}|{question}".encode()).hexdigest()[:16]
    return Question(
        question_id=digest,
        question=question.strip(),
        question_type=request.question_type,
        difficulty=request.difficulty,
        options=options or [],
        expected_answer=expected_answer.strip(),
        acceptable_answers=acceptable_answers or [],
        key_points=key_points or [],
        explanation=explanation.strip(),
        evidence=evidence.strip(),
        source_chunks=[c.chunk_id for c in source_chunks],
        source_pages=sorted({c.page_number for c in source_chunks}),
        sources=source_refs(source_chunks, evidence.strip()),
        topic=request.topic_label,
        topic_id=request.topic_id,
        seed_chunk_id=request.seed.chunk_id,
        generated_by=generated_by,
    )
