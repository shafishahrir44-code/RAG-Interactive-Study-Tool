from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import LLMSettings, Settings, load_settings
from app.container import Container, build_container
from app.main import create_app
from tests.fakes import FakeGroundedLLM
from tests.pdf_factory import make_pdf
from tests.sample_content import SAMPLE_PAGES, SAMPLE_TITLE


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    for name in ("LLM_PROVIDER", "EMBEDDING_PROVIDER", "VECTOR_STORE", "DATA_DIR"):
        monkeypatch.delenv(name, raising=False)
    base = load_settings(env_file=tmp_path / "missing.env")
    return replace(base, data_dir=tmp_path / "data", llm=LLMSettings(provider="none"))


@pytest.fixture
def sample_pdf() -> bytes:
    return make_pdf(SAMPLE_PAGES, SAMPLE_TITLE)


@pytest.fixture
def fake_llm() -> FakeGroundedLLM:
    return FakeGroundedLLM()


@pytest.fixture
def llm_container(settings: Settings, fake_llm: FakeGroundedLLM) -> Container:
    return build_container(settings, llm_client=fake_llm)


@pytest.fixture
def extractive_container(settings: Settings) -> Container:
    return build_container(settings, use_llm=False)


@pytest.fixture
def client(settings: Settings, llm_container: Container) -> TestClient:
    return TestClient(create_app(settings, llm_container))


def ingest(container: Container, pdf_bytes: bytes) -> str:
    """Synchronously ingest a PDF and return its document id."""
    record, needs_processing = container.ingestion.register_upload(pdf_bytes, "sample.pdf")
    if needs_processing:
        container.ingestion.process(record.document_id, pdf_bytes)
    return record.document_id


def wait_until_ready(client: TestClient, document_id: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        doc = client.get(f"/api/documents/{document_id}").json()
        if doc["status"] != "processing":
            return doc
        time.sleep(0.05)
    raise AssertionError("document did not finish processing")
