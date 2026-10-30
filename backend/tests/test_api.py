from __future__ import annotations

from fastapi.testclient import TestClient

from app.container import build_container
from app.errors import LLMUnavailableError
from app.main import create_app
from tests.conftest import wait_until_ready
from tests.fakes import FakeGroundedLLM
from tests.pdf_factory import make_pdf


def upload(client: TestClient, pdf: bytes, name: str = "notes.pdf"):
    return client.post("/api/documents", files={"file": (name, pdf, "application/pdf")})


def ready_document(client: TestClient, pdf: bytes) -> dict:
    response = upload(client, pdf)
    assert response.status_code == 202
    return wait_until_ready(client, response.json()["document"]["document_id"])


def start(client: TestClient, document_id: str, **options) -> str:
    response = client.post("/api/sessions", json={"document_id": document_id, **options})
    assert response.status_code == 201, response.text
    return response.json()["session_id"]


def test_health_reports_configuration(client: TestClient) -> None:
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["llm_available"] is True
    assert "short_answer" in body["question_types_available"]


def test_full_learning_flow(client: TestClient, sample_pdf: bytes) -> None:
    doc = ready_document(client, sample_pdf)
    assert doc["status"] == "ready" and doc["chunk_count"] > 0 and doc["topics"]

    session_id = start(client, doc["document_id"], num_questions=4)
    for number in range(1, 5):
        session = client.post(f"/api/sessions/{session_id}/next").json()
        question = session["current_question"]
        assert question["number"] == number
        assert "expected_answer" not in question  # answers are hidden until submission
        # Asking again returns the same unanswered question (one question at a time).
        assert client.post(f"/api/sessions/{session_id}/next").json()["current_question"]["question_id"] == question["question_id"]

        answer = question["options"][0] if question["options"] else "True" if question["question_type"] == "true_false" else "cytoplasm"
        result = client.post(f"/api/sessions/{session_id}/answer", json={"question_id": question["question_id"], "answer": answer})
        assert result.status_code == 200, result.text
        body = result.json()
        assert body["evaluation"]["verdict"] in {"correct", "partially_correct", "incorrect"}
        assert body["question"]["expected_answer"] and body["question"]["explanation"]
        assert body["question"]["source_pages"] and body["question"]["sources"][0]["excerpt"]
        assert body["next_step"]
        assert body["progress"]["answered"] == number

    assert body["session_completed"] is True
    assert client.post(f"/api/sessions/{session_id}/next").status_code == 409

    summary = client.get(f"/api/sessions/{session_id}/summary").json()
    assert summary["answered"] == 4
    assert summary["correct"] + summary["partially_correct"] + summary["incorrect"] == 4
    assert len(summary["review"]) == 4 and summary["by_topic"] and summary["pages_covered"]


def test_duplicate_upload_is_reused(client: TestClient, sample_pdf: bytes) -> None:
    ready_document(client, sample_pdf)
    again = upload(client, sample_pdf, "renamed.pdf").json()
    assert again["reused"] is True and again["document"]["status"] == "ready"


def test_upload_validation_errors(client: TestClient, sample_pdf: bytes) -> None:
    not_pdf = client.post("/api/documents", files={"file": ("x.txt", b"plain text", "text/plain")})
    assert not_pdf.status_code == 422 and not_pdf.json()["error"]["code"] == "invalid_pdf"
    fake_pdf = upload(client, b"just some bytes")
    assert fake_pdf.status_code == 422 and "PDF" in fake_pdf.json()["error"]["message"]


def test_failed_processing_is_visible(client: TestClient) -> None:
    response = upload(client, make_pdf(["", ""]))
    doc = wait_until_ready(client, response.json()["document"]["document_id"])
    assert doc["status"] == "failed" and "OCR" in doc["error_message"]


def test_session_and_answer_errors(client: TestClient, sample_pdf: bytes) -> None:
    assert client.post("/api/sessions", json={"document_id": "deadbeef" * 8}).status_code == 404
    doc = ready_document(client, sample_pdf)
    bad_count = client.post("/api/sessions", json={"document_id": doc["document_id"], "num_questions": 0})
    assert bad_count.status_code == 400

    session_id = start(client, doc["document_id"], num_questions=3)
    question = client.post(f"/api/sessions/{session_id}/next").json()["current_question"]
    wrong_id = client.post(f"/api/sessions/{session_id}/answer", json={"question_id": "nope", "answer": "x"})
    assert wrong_id.status_code == 409
    empty = client.post(f"/api/sessions/{session_id}/answer", json={"question_id": question["question_id"], "answer": "  "})
    assert empty.status_code == 400 and empty.json()["error"]["message"].startswith("Please enter")
    unsupported = client.post("/api/sessions", json={"document_id": doc["document_id"], "question_types": ["essay"]})
    assert unsupported.status_code == 422
    assert client.get("/api/sessions/not.a.session").status_code == 404


def test_finish_early_returns_summary(client: TestClient, sample_pdf: bytes) -> None:
    doc = ready_document(client, sample_pdf)
    session_id = start(client, doc["document_id"], num_questions=10)
    client.post(f"/api/sessions/{session_id}/next")
    summary = client.post(f"/api/sessions/{session_id}/finish").json()
    assert summary["status"] == "completed" and summary["answered"] == 0


def test_falls_back_to_extractive_questions_when_llm_is_down(settings, sample_pdf: bytes) -> None:
    def down(system, user):
        raise LLMUnavailableError("Could not reach Ollama.")

    container = build_container(settings, llm_client=FakeGroundedLLM(override=down))
    client = TestClient(create_app(settings, container))
    doc = ready_document(client, sample_pdf)
    session_id = start(client, doc["document_id"], num_questions=2)
    session = client.post(f"/api/sessions/{session_id}/next").json()
    assert session["current_question"]["generated_by"] == "extractive"
    assert "unreachable" in session["engine_notice"]


def test_session_survives_restart(settings, llm_container, sample_pdf: bytes) -> None:
    client = TestClient(create_app(settings, llm_container))
    doc = ready_document(client, sample_pdf)
    session_id = start(client, doc["document_id"], num_questions=2)
    question_id = client.post(f"/api/sessions/{session_id}/next").json()["current_question"]["question_id"]

    restarted = TestClient(create_app(settings, build_container(settings, llm_client=FakeGroundedLLM())))
    resumed = restarted.get(f"/api/sessions/{session_id}").json()
    assert resumed["current_question"]["question_id"] == question_id


def test_next_question_is_prepared_while_feedback_is_read(client: TestClient, sample_pdf: bytes) -> None:
    doc = ready_document(client, sample_pdf)
    session_id = start(client, doc["document_id"], num_questions=3)
    # Creating the session already prepared question 1 in the background.
    first = client.get(f"/api/sessions/{session_id}").json()["current_question"]
    assert first is not None and first["number"] == 1

    client.post(f"/api/sessions/{session_id}/answer", json={"question_id": first["question_id"], "answer": "True"})
    prepared = client.get(f"/api/sessions/{session_id}").json()["current_question"]
    assert prepared is not None and prepared["number"] == 2
    # The learner's own request returns the prepared question instead of writing a new one.
    assert client.post(f"/api/sessions/{session_id}/next").json()["current_question"]["question_id"] == prepared["question_id"]
