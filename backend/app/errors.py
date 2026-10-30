"""Application error hierarchy.

Each error carries a user-facing `message` and an optional technical `detail` for debugging.
The API layer converts these into a consistent JSON shape.
"""

from __future__ import annotations


class AppError(Exception):
    status_code = 500
    code = "internal_error"

    def __init__(self, message: str, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail
        self.upstream_status: int | None = None  # HTTP status from an external service, if any


class InvalidInputError(AppError):
    status_code = 400
    code = "invalid_input"


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


class ConflictError(AppError):
    status_code = 409
    code = "conflict"


class DocumentTooLargeError(AppError):
    status_code = 413
    code = "document_too_large"


class InvalidPDFError(AppError):
    status_code = 422
    code = "invalid_pdf"


class NoExtractableTextError(AppError):
    status_code = 422
    code = "no_extractable_text"


class EmbeddingError(AppError):
    status_code = 503
    code = "embedding_failed"


class RetrievalError(AppError):
    status_code = 500
    code = "retrieval_failed"


class LLMUnavailableError(AppError):
    status_code = 503
    code = "llm_unavailable"


class LLMResponseError(AppError):
    """The LLM answered, but the output was malformed or failed validation."""

    status_code = 502
    code = "llm_bad_response"


class InsufficientContextError(AppError):
    """The document does not contain enough information to produce a grounded question."""

    status_code = 422
    code = "insufficient_context"
