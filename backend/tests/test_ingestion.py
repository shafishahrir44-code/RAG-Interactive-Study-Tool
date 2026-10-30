from __future__ import annotations

import pytest

from app.config import ChunkingSettings
from app.documents.models import DocumentStatus
from app.errors import DocumentTooLargeError, InvalidPDFError, NoExtractableTextError
from app.ingestion.chunker import Chunker
from app.ingestion.pdf_extractor import PypdfTextExtractor, validate_pdf_bytes
from app.ingestion.pipeline import document_id_for, sanitize_filename
from tests.conftest import ingest
from tests.pdf_factory import make_pdf
from tests.sample_content import SAMPLE_PAGES, SAMPLE_TITLE


class TestExtraction:
    def test_extracts_pages_with_numbers_and_skips_blank_pages(self, sample_pdf: bytes) -> None:
        doc = PypdfTextExtractor(max_pages=100).extract(sample_pdf)
        assert doc.page_count == 5
        assert doc.empty_pages == [3]
        assert [p.page_number for p in doc.pages] == [1, 2, 4, 5]
        assert doc.title == SAMPLE_TITLE
        assert "Mitochondria" in doc.pages[1].text
        assert "Chloroplasts" in doc.pages[2].text  # page 4 keeps its real number

    def test_rejects_non_pdf_bytes(self) -> None:
        with pytest.raises(InvalidPDFError):
            validate_pdf_bytes(b"hello, I am a text file", max_bytes=1000)

    def test_rejects_empty_upload(self) -> None:
        with pytest.raises(InvalidPDFError):
            validate_pdf_bytes(b"", max_bytes=1000)

    def test_rejects_oversized_upload(self) -> None:
        with pytest.raises(DocumentTooLargeError):
            validate_pdf_bytes(b"%PDF-" + b"0" * 2000, max_bytes=1000)

    def test_rejects_corrupted_pdf(self) -> None:
        with pytest.raises(InvalidPDFError):
            PypdfTextExtractor(max_pages=10).extract(b"%PDF-1.4\nthis is not really a pdf")

    def test_rejects_too_many_pages(self) -> None:
        pdf = make_pdf(["Page text that is long enough to count as content."] * 3)
        with pytest.raises(DocumentTooLargeError):
            PypdfTextExtractor(max_pages=2).extract(pdf)

    def test_pdf_without_text_reports_ocr_requirement(self) -> None:
        with pytest.raises(NoExtractableTextError) as info:
            PypdfTextExtractor(max_pages=10).extract(make_pdf(["", ""]))
        assert "OCR" in info.value.message


class TestChunking:
    settings = ChunkingSettings(chunk_size=300, chunk_overlap=80, min_chunk_chars=40)

    def chunks(self):
        pages = PypdfTextExtractor(max_pages=100).extract(make_pdf(SAMPLE_PAGES)).pages
        return pages, Chunker(self.settings).chunk_document("abcdef123456789", pages)

    def test_chunks_respect_size_and_never_cross_pages(self) -> None:
        pages, chunks = self.chunks()
        page_text = {p.page_number: p.text for p in pages}
        assert len(chunks) > len(pages)
        for chunk in chunks:
            assert len(page_text[chunk.page_number][chunk.char_start : chunk.char_end]) <= self.settings.chunk_size
            assert chunk.page_number in page_text

    def test_metadata_maps_back_to_the_source_page(self) -> None:
        pages, chunks = self.chunks()
        page_text = {p.page_number: p.text for p in pages}
        for index, chunk in enumerate(chunks):
            assert chunk.chunk_index == index
            assert chunk.document_id == "abcdef123456789"
            assert chunk.chunk_id == f"abcdef123456-p{chunk.page_number}-c{index}"
            source = page_text[chunk.page_number][chunk.char_start : chunk.char_end]
            assert " ".join(source.split()) == " ".join(chunk.text.split())
            assert "  " not in chunk.text and "\n\n" not in chunk.text  # whitespace tidied, lines kept

    def test_consecutive_chunks_on_a_page_overlap(self) -> None:
        _, chunks = self.chunks()
        pairs = [(a, b) for a, b in zip(chunks, chunks[1:]) if a.page_number == b.page_number]
        assert pairs, "expected multiple chunks on at least one page"
        assert any(b.char_start < a.char_end for a, b in pairs)

    def test_all_text_is_covered(self) -> None:
        pages, chunks = self.chunks()
        for page in pages:
            covered = " ".join(c.text for c in chunks if c.page_number == page.page_number)
            for sentence_start in ("Cells are", "Mitochondria are", "Ribosomes are", "Meiosis is"):
                if sentence_start in " ".join(page.text.split()):
                    assert sentence_start in covered

    def test_overlap_must_be_smaller_than_size(self) -> None:
        with pytest.raises(ValueError):
            Chunker(ChunkingSettings(chunk_size=100, chunk_overlap=100))


class TestPipeline:
    def test_ingestion_builds_index_and_topics(self, extractive_container, sample_pdf) -> None:
        document_id = ingest(extractive_container, sample_pdf)
        record = extractive_container.registry.get(document_id)
        assert record.status == DocumentStatus.READY
        assert record.progress == 1.0
        assert record.chunk_count == len(extractive_container.store.get_chunks(document_id))
        assert record.topics and all(t.chunk_ids for t in record.topics)
        assert "pages 3" in record.warnings[0]

    def test_duplicate_upload_reuses_existing_index(self, extractive_container, sample_pdf) -> None:
        ingest(extractive_container, sample_pdf)
        record, needs_processing = extractive_container.ingestion.register_upload(sample_pdf, "copy.pdf")
        assert needs_processing is False
        assert record.document_id == document_id_for(sample_pdf)

    def test_failure_is_recorded_not_raised(self, extractive_container) -> None:
        pdf = make_pdf(["", ""])
        record, _ = extractive_container.ingestion.register_upload(pdf, "scan.pdf")
        result = extractive_container.ingestion.process(record.document_id, pdf)
        assert result.status == DocumentStatus.FAILED
        assert result.error_message and "OCR" in result.error_message


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("../../etc/passwd", "passwd"),
        ("C:\\Users\\me\\notes.pdf", "notes.pdf"),
        ("<script>.pdf", "_script_.pdf"),
        ("", "document.pdf"),
        (None, "document.pdf"),
    ],
)
def test_sanitize_filename_strips_paths_and_markup(raw, expected) -> None:
    assert sanitize_filename(raw) == expected
