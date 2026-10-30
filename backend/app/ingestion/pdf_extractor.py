"""PDF validation and per-page text extraction.

`TextExtractor` is a protocol so an OCR-capable extractor (e.g. Tesseract for scanned pages)
can be dropped in later without touching chunking, embedding or the API.
"""

from __future__ import annotations

import io
import logging
from typing import Protocol

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.errors import DocumentTooLargeError, InvalidPDFError, NoExtractableTextError
from app.ingestion.models import ExtractedDocument, PageText
from app.text_utils import normalize_text

logger = logging.getLogger(__name__)

PDF_MAGIC = b"%PDF-"
# Pages with fewer meaningful characters than this are treated as empty (e.g. blank or image-only).
MIN_PAGE_CHARS = 20


class TextExtractor(Protocol):
    def extract(self, pdf_bytes: bytes) -> ExtractedDocument: ...


def validate_pdf_bytes(pdf_bytes: bytes, max_bytes: int) -> None:
    if not pdf_bytes:
        raise InvalidPDFError("The uploaded file is empty.")
    if len(pdf_bytes) > max_bytes:
        raise DocumentTooLargeError(
            f"The file is larger than the {max_bytes // (1024 * 1024)} MB limit.",
            detail=f"size={len(pdf_bytes)} bytes",
        )
    # Some generators prepend a few bytes before the header; the spec allows it within 1 KB.
    if PDF_MAGIC not in pdf_bytes[:1024]:
        raise InvalidPDFError("This file does not appear to be a PDF.", detail="missing %PDF- header")


class PypdfTextExtractor:
    """Extracts embedded text with pypdf. Does not perform OCR."""

    def __init__(self, max_pages: int) -> None:
        self.max_pages = max_pages

    def extract(self, pdf_bytes: bytes) -> ExtractedDocument:
        reader = self._open(pdf_bytes)
        page_count = len(reader.pages)
        if page_count == 0:
            raise InvalidPDFError("The PDF has no pages.")
        if page_count > self.max_pages:
            raise DocumentTooLargeError(
                f"The PDF has {page_count} pages; the limit is {self.max_pages}.",
                detail=f"pages={page_count}",
            )

        pages: list[PageText] = []
        empty_pages: list[int] = []
        for index, page in enumerate(reader.pages):
            page_number = index + 1
            text = self._extract_page(page, page_number)
            if len(text.replace(" ", "").replace("\n", "")) < MIN_PAGE_CHARS:
                empty_pages.append(page_number)
                continue
            pages.append(PageText(page_number=page_number, text=text))

        if not pages:
            raise NoExtractableTextError(
                "No readable text was found in this PDF. It may be a scanned/image-only document, "
                "which requires OCR (not enabled).",
                detail=f"pages={page_count}, all empty after extraction",
            )
        return ExtractedDocument(
            pages=pages, page_count=page_count, title=self._title(reader), empty_pages=empty_pages
        )

    @staticmethod
    def _open(pdf_bytes: bytes) -> PdfReader:
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            if reader.is_encrypted:
                try:
                    reader.decrypt("")  # many "encrypted" PDFs use an empty user password
                except Exception as exc:  # noqa: BLE001 - pypdf raises various types here
                    raise InvalidPDFError("This PDF is password-protected.", detail=str(exc)) from exc
            _ = len(reader.pages)  # forces parsing of the page tree
            return reader
        except InvalidPDFError:
            raise
        except (PdfReadError, ValueError, KeyError, TypeError, OSError) as exc:
            raise InvalidPDFError("The PDF could not be read; it may be corrupted.", detail=repr(exc)) from exc

    @staticmethod
    def _extract_page(page, page_number: int) -> str:
        try:
            return normalize_text(page.extract_text() or "")
        except Exception as exc:  # noqa: BLE001 - one bad page should not abort the document
            logger.warning("Text extraction failed on page %s: %r", page_number, exc)
            return ""

    @staticmethod
    def _title(reader: PdfReader) -> str | None:
        try:
            title = reader.metadata.title if reader.metadata else None
        except Exception:  # noqa: BLE001 - metadata is optional and frequently malformed
            return None
        return title.strip() if isinstance(title, str) and title.strip() else None
