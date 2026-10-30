"""Ingestion orchestration: validate -> extract -> chunk -> embed -> topics -> index.

Progress is written to the document registry so the UI can show each stage.
"""

from __future__ import annotations

import hashlib
import logging
import re

import numpy as np

from app.config import Settings
from app.documents.models import DocumentRecord, DocumentStatus, ProcessingStage
from app.documents.registry import DocumentRegistry
from app.embeddings.base import Embedder
from app.errors import AppError, EmbeddingError
from app.ingestion.chunker import Chunker
from app.ingestion.pdf_extractor import TextExtractor, validate_pdf_bytes
from app.retrieval.topics import build_topics
from app.vectorstore.base import VectorStore

logger = logging.getLogger(__name__)

MAX_FILENAME_LENGTH = 120
# Fraction of the progress bar allocated to each stage (embedding is usually the slowest).
_PROGRESS = {"extracted": 0.15, "chunked": 0.25, "embedded": 0.9}


def document_id_for(pdf_bytes: bytes) -> str:
    """Content-addressed id: identical uploads map to the same document and index."""
    return hashlib.sha256(pdf_bytes).hexdigest()


def sanitize_filename(raw_name: str | None) -> str:
    """Display-only filename. Paths are stripped and never used for storage."""
    name = re.split(r"[\\/]", raw_name or "")[-1]
    name = re.sub(r"[^\w.\- ()]", "_", name).strip(" .")
    return (name or "document.pdf")[:MAX_FILENAME_LENGTH]


class IngestionService:
    def __init__(
        self,
        settings: Settings,
        registry: DocumentRegistry,
        extractor: TextExtractor,
        chunker: Chunker,
        embedder: Embedder,
        store: VectorStore,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.extractor = extractor
        self.chunker = chunker
        self.embedder = embedder
        self.store = store

    def register_upload(self, pdf_bytes: bytes, filename: str | None) -> tuple[DocumentRecord, bool]:
        """Validate an upload and create (or reuse) its record.

        Returns (record, needs_processing). Re-uploading an already-indexed PDF reuses the
        existing index instead of re-embedding it.
        """
        validate_pdf_bytes(pdf_bytes, self.settings.limits.max_upload_mb * 1024 * 1024)
        document_id = document_id_for(pdf_bytes)
        existing = self.registry.find(document_id)
        if existing and existing.status == DocumentStatus.PROCESSING:
            return existing, False
        if (
            existing
            and existing.status == DocumentStatus.READY
            and self.store.has_index(document_id, self.embedder.fingerprint)
        ):
            return existing, False

        record = DocumentRecord(
            document_id=document_id, filename=sanitize_filename(filename), size_bytes=len(pdf_bytes)
        )
        self.registry.save(record)
        return record, True

    def process(self, document_id: str, pdf_bytes: bytes) -> DocumentRecord:
        """Run the full pipeline. Failures are recorded on the document, not raised."""
        try:
            return self._run(document_id, pdf_bytes)
        except AppError as exc:
            logger.warning("Ingestion failed for %s: %s (%s)", document_id, exc.message, exc.detail)
            return self._fail(document_id, exc.message, exc.detail)
        except Exception as exc:  # noqa: BLE001 - must never leave a document stuck "processing"
            logger.exception("Unexpected ingestion failure for %s", document_id)
            return self._fail(document_id, "Processing failed unexpectedly.", repr(exc))

    def _run(self, document_id: str, pdf_bytes: bytes) -> DocumentRecord:
        update = self.registry.update
        update(document_id, stage=ProcessingStage.EXTRACTING, progress=0.02)
        extracted = self.extractor.extract(pdf_bytes)
        warnings = []
        if extracted.empty_pages:
            warnings.append(
                f"{len(extracted.empty_pages)} page(s) had no extractable text and were skipped "
                f"(pages {', '.join(map(str, extracted.empty_pages[:10]))}"
                f"{'…' if len(extracted.empty_pages) > 10 else ''})."
            )
        update(
            document_id,
            stage=ProcessingStage.CHUNKING,
            progress=_PROGRESS["extracted"],
            title=extracted.title,
            page_count=extracted.page_count,
            pages_with_text=len(extracted.pages),
            empty_pages=extracted.empty_pages,
            warnings=warnings,
        )

        chunks = self.chunker.chunk_document(document_id, extracted.pages)
        update(document_id, stage=ProcessingStage.EMBEDDING, progress=_PROGRESS["chunked"], chunk_count=len(chunks))

        vectors = self._embed_with_progress(document_id, [c.text for c in chunks])
        update(document_id, stage=ProcessingStage.INDEXING, progress=_PROGRESS["embedded"])

        topics, labelled_chunks = build_topics(chunks, vectors)
        self.store.upsert(document_id, labelled_chunks, vectors, self.embedder.fingerprint)
        return update(
            document_id,
            status=DocumentStatus.READY,
            stage=ProcessingStage.DONE,
            progress=1.0,
            topics=topics,
            embedding_fingerprint=self.embedder.fingerprint,
            error_message=None,
            error_detail=None,
        )

    def _embed_with_progress(self, document_id: str, texts: list[str]) -> np.ndarray:
        batch = max(1, self.settings.embedding.batch_size)
        start_p, end_p = _PROGRESS["chunked"], _PROGRESS["embedded"]
        parts: list[np.ndarray] = []
        for start in range(0, len(texts), batch):
            parts.append(self.embedder.embed_documents(texts[start : start + batch]))
            done = min(start + batch, len(texts)) / len(texts)
            self.registry.update(document_id, progress=round(start_p + (end_p - start_p) * done, 3))
        vectors = np.concatenate(parts) if parts else np.zeros((0, 1), dtype=np.float32)
        if len(vectors) != len(texts) or not np.isfinite(vectors).all():
            raise EmbeddingError("Embedding produced invalid vectors.", detail=f"shape={vectors.shape}")
        return vectors

    def _fail(self, document_id: str, message: str, detail: str | None) -> DocumentRecord:
        return self.registry.update(
            document_id, status=DocumentStatus.FAILED, error_message=message, error_detail=detail
        )
