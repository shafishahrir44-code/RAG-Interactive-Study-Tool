"""Page-aware, sentence-boundary chunking with overlap.

Chunks never cross page boundaries, so every chunk maps to exactly one page number.
Sentences are packed greedily up to `chunk_size` characters; the trailing sentences of a
chunk (up to `chunk_overlap` characters) are repeated at the start of the next chunk so
facts spanning a boundary remain retrievable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.config import ChunkingSettings
from app.ingestion.models import Chunk, PageText

_BOUNDARY_RE = re.compile(r"(?<=[.!?:;])\s+|\n\s*\n")


@dataclass(frozen=True)
class Span:
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


def sentence_spans(text: str) -> list[Span]:
    """Return character spans of sentences/paragraphs within `text`."""
    spans: list[Span] = []
    cursor = 0
    for match in _BOUNDARY_RE.finditer(text):
        if match.start() > cursor:
            spans.append(Span(cursor, match.start()))
        cursor = match.end()
    if cursor < len(text):
        spans.append(Span(cursor, len(text)))
    return [s for s in spans if text[s.start : s.end].strip()]


def _split_long_span(text: str, span: Span, max_chars: int) -> list[Span]:
    """Split a span longer than max_chars at whitespace (fallback for run-on text)."""
    pieces: list[Span] = []
    start = span.start
    while span.end - start > max_chars:
        cut = text.rfind(" ", start, start + max_chars)
        if cut <= start:
            cut = start + max_chars
        pieces.append(Span(start, cut))
        start = cut + 1 if text[cut : cut + 1] == " " else cut
    if start < span.end:
        pieces.append(Span(start, span.end))
    return pieces


def _clean(text: str) -> str:
    """Collapse spaces but keep single line breaks: slide bullets and headings stay separable."""
    lines = (re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


class Chunker:
    def __init__(self, settings: ChunkingSettings) -> None:
        if settings.chunk_overlap >= settings.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.settings = settings

    def chunk_document(self, document_id: str, pages: list[PageText]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for page in pages:
            for span in self._page_windows(page.text):
                chunks.append(
                    Chunk(
                        chunk_id=f"{document_id[:12]}-p{page.page_number}-c{len(chunks)}",
                        document_id=document_id,
                        page_number=page.page_number,
                        chunk_index=len(chunks),
                        text=_clean(page.text[span.start : span.end]),
                        char_start=span.start,
                        char_end=span.end,
                    )
                )
        return chunks

    def _page_windows(self, text: str) -> list[Span]:
        size, overlap = self.settings.chunk_size, self.settings.chunk_overlap
        units: list[Span] = []
        for span in sentence_spans(text):
            units.extend(_split_long_span(text, span, size) if span.length > size else [span])

        windows: list[Span] = []
        current: list[Span] = []
        for unit in units:
            if current and unit.end - current[0].start > size:
                windows.append(Span(current[0].start, current[-1].end))
                current = self._overlap_tail(current, overlap, unit, size)
            current.append(unit)
        if current:
            windows.append(Span(current[0].start, current[-1].end))
        return self._merge_small_tail(windows)

    @staticmethod
    def _overlap_tail(units: list[Span], overlap: int, next_unit: Span, size: int) -> list[Span]:
        """Trailing units of the previous window that fit the overlap budget and leave room
        for the next unit within the chunk size."""
        tail: list[Span] = []
        for unit in reversed(units):
            tail_chars = units[-1].end - unit.start
            if tail_chars > overlap or next_unit.end - unit.start > size:
                break
            tail.insert(0, unit)
        return tail

    def _merge_small_tail(self, windows: list[Span]) -> list[Span]:
        if len(windows) >= 2 and windows[-1].length < self.settings.min_chunk_chars:
            last = windows.pop()
            windows[-1] = Span(windows[-1].start, last.end)
        return windows
