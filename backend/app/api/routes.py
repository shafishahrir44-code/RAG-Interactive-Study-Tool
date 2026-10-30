"""HTTP routes. Handlers stay thin: validation and translation only; logic lives in services."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, File, Request, UploadFile, status

from app.api.schemas import (
    AnswerResponse,
    CreateSessionRequest,
    DocumentResponse,
    HealthResponse,
    Progress,
    RevealedQuestion,
    SessionResponse,
    SubmitAnswerRequest,
)
from app.container import Container
from app.documents.models import DocumentRecord
from app.errors import DocumentTooLargeError, InvalidPDFError
from app.questions.service import FALLBACK_NOTICE
from app.sessions.models import SessionStatus
from app.sessions.summary import SessionSummary

router = APIRouter(prefix="/api")
UPLOAD_READ_CHUNK = 1024 * 1024
ALLOWED_CONTENT_TYPES = {"application/pdf", "application/x-pdf", "application/octet-stream", ""}


def _container(request: Request) -> Container:
    return request.app.state.container


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    c = _container(request)
    llm_available = bool(c.llm_client and c.llm_client.is_available())
    engine = c.questions.llm_engine if llm_available else c.questions.extractive_engine
    notice = None
    if c.llm_client and not llm_available:
        notice = f"{c.llm_client.name} is not reachable. {FALLBACK_NOTICE}"
    elif not c.llm_client:
        notice = "No LLM configured (LLM_PROVIDER=none). Questions are built directly from document sentences."
    return HealthResponse(
        status="ok",
        llm_provider=c.settings.llm.provider,
        llm_model=c.settings.llm.model if c.llm_client else None,
        llm_available=llm_available,
        embedding=c.embedder.fingerprint,
        vector_store=c.settings.retrieval.vector_store,
        question_types_available=sorted(engine.supported_types, key=lambda t: t.value),
        max_upload_mb=c.settings.limits.max_upload_mb,
        notice=notice,
    )


async def _read_upload(file: UploadFile, max_bytes: int) -> bytes:
    """Stream the upload with a hard size cap so oversized files are rejected early."""
    data = bytearray()
    while chunk := await file.read(UPLOAD_READ_CHUNK):
        data.extend(chunk)
        if len(data) > max_bytes:
            raise DocumentTooLargeError(f"The file is larger than the {max_bytes // (1024 * 1024)} MB limit.")
    return bytes(data)


@router.post("/documents", response_model=DocumentResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    request: Request, background: BackgroundTasks, file: UploadFile = File(...)
) -> DocumentResponse:
    c = _container(request)
    if (file.content_type or "") not in ALLOWED_CONTENT_TYPES:
        raise InvalidPDFError("Only PDF files are supported.", detail=f"content-type={file.content_type}")
    pdf_bytes = await _read_upload(file, c.settings.limits.max_upload_mb * 1024 * 1024)
    record, needs_processing = c.ingestion.register_upload(pdf_bytes, file.filename)
    if needs_processing:
        c.retriever.invalidate(record.document_id)
        background.add_task(c.ingestion.process, record.document_id, pdf_bytes)
    return DocumentResponse(document=record, reused=not needs_processing)


@router.get("/documents/{document_id}", response_model=DocumentRecord)
def get_document(request: Request, document_id: str) -> DocumentRecord:
    return _container(request).registry.get(document_id)


@router.post("/sessions", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
def create_session(request: Request, body: CreateSessionRequest, background: BackgroundTasks) -> SessionResponse:
    c = _container(request)
    session = c.sessions.create(
        body.document_id, body.num_questions, body.question_types, body.start_difficulty, body.adaptive
    )
    background.add_task(c.sessions.prefetch_next, session.session_id)
    return SessionResponse.from_session(session)


@router.get("/sessions/{session_id}", response_model=SessionResponse)
def get_session(request: Request, session_id: str) -> SessionResponse:
    return SessionResponse.from_session(_container(request).sessions.get(session_id))


@router.post("/sessions/{session_id}/next", response_model=SessionResponse)
def next_question(request: Request, session_id: str) -> SessionResponse:
    return SessionResponse.from_session(_container(request).sessions.next_question(session_id))


@router.post("/sessions/{session_id}/answer", response_model=AnswerResponse)
def submit_answer(
    request: Request, session_id: str, body: SubmitAnswerRequest, background: BackgroundTasks
) -> AnswerResponse:
    c = _container(request)
    outcome = c.sessions.submit_answer(session_id, body.question_id, body.answer)
    if outcome.session.status != SessionStatus.COMPLETED:
        # Write the next question while the learner reads this feedback.
        background.add_task(c.sessions.prefetch_next, session_id)
    return AnswerResponse(
        user_answer=outcome.record.user_answer,
        evaluation=outcome.record.evaluation,
        question=RevealedQuestion.from_question(outcome.record.question),
        adaptation=outcome.adaptation,
        next_step=outcome.next_step,
        progress=Progress.from_session(outcome.session),
        session_completed=outcome.session.status == SessionStatus.COMPLETED,
    )


@router.post("/sessions/{session_id}/finish", response_model=SessionSummary)
def finish_session(request: Request, session_id: str) -> SessionSummary:
    c = _container(request)
    c.sessions.finish(session_id)
    return c.sessions.summary(session_id)


@router.get("/sessions/{session_id}/summary", response_model=SessionSummary)
def session_summary(request: Request, session_id: str) -> SessionSummary:
    return _container(request).sessions.summary(session_id)
