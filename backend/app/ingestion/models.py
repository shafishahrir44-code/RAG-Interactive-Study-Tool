"""Data models produced by the ingestion pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PageText(BaseModel):
    page_number: int  # 1-based, as shown in PDF viewers
    text: str


class ExtractedDocument(BaseModel):
    pages: list[PageText]
    page_count: int
    title: str | None = None
    empty_pages: list[int] = Field(default_factory=list)


class Chunk(BaseModel):
    """A retrievable unit of text. Source metadata travels with the chunk everywhere."""

    chunk_id: str
    document_id: str
    page_number: int
    chunk_index: int  # position within the whole document
    text: str
    char_start: int  # offset within the page text
    char_end: int
    topic_id: int | None = None
