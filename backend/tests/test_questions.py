from __future__ import annotations

import pytest

from app.adaptive.policy import QuestionPlan
from app.container import build_container
from app.errors import InsufficientContextError, LLMResponseError, LLMUnavailableError
from app.evaluation.models import Evaluation, Verdict
from app.ingestion.models import Chunk
from app.llm.base import StructuredLLM
from app.questions.engine import GenerationRequest
from app.questions.extractive_engine import ExtractiveQuestionEngine
from app.questions.llm_engine import LLMQuestionEngine
from app.questions.models import BLANK, Difficulty, QuestionDraft, QuestionType
from app.questions.validator import DraftRejected, DraftValidator
from app.sessions.models import AnswerRecord, Session
from tests.conftest import ingest
from tests.fakes import FakeGroundedLLM

TEXT = (
    "Mitochondria are the organelles where cellular respiration produces most of the cell's ATP. "
    "Glycolysis is the first stage of respiration and takes place in the cytoplasm. "
    "Fermentation regenerates NAD+ when oxygen is scarce and produces lactate in muscle cells."
)
CHUNK = Chunk(chunk_id="d-p2-c1", document_id="d", page_number=2, chunk_index=1, text=TEXT, char_start=0, char_end=len(TEXT))
EVIDENCE = "Glycolysis is the first stage of respiration and takes place in the cytoplasm."


def request(qtype: QuestionType, difficulty=Difficulty.EASY, variant=0) -> GenerationRequest:
    return GenerationRequest(qtype, difficulty, 0, "respiration", CHUNK, [CHUNK], variant=variant)


def draft(**fields) -> QuestionDraft:
    base = {"explanation": "Page 2 says so.", "evidence_quote": EVIDENCE, "source_chunk_ids": [CHUNK.chunk_id]}
    return QuestionDraft(**{**base, **fields})


class TestValidator:
    validator = DraftValidator()

    def test_accepts_grounded_multiple_choice_and_recovers_letter_answer(self) -> None:
        result = self.validator.validate(
            draft(question="Where does glycolysis occur?", options=["Cytoplasm", "Nucleus", "Stroma", "Ribosome"], answer="A"),
            QuestionType.MULTIPLE_CHOICE,
            [CHUNK],
        )
        assert result.answer == "Cytoplasm"
        assert sorted(result.options) == ["Cytoplasm", "Nucleus", "Ribosome", "Stroma"]
        assert [c.chunk_id for c in result.source_chunks] == [CHUNK.chunk_id]

    def test_rejects_evidence_not_in_context(self) -> None:
        with pytest.raises(DraftRejected) as info:
            self.validator.validate(
                draft(question="Is the sky blue on Mars?", answer="false", evidence_quote="Mars has a butterscotch sky at noon."),
                QuestionType.TRUE_FALSE,
                [CHUNK],
            )
        assert any("evidence" in p for p in info.value.problems)

    def test_rejects_mcq_whose_answer_is_not_an_option(self) -> None:
        with pytest.raises(DraftRejected):
            self.validator.validate(
                draft(question="Where does glycolysis occur?", options=["Nucleus", "Stroma", "Ribosome", "Golgi"], answer="Cytoplasm"),
                QuestionType.MULTIPLE_CHOICE,
                [CHUNK],
            )

    def test_fill_in_blank_needs_a_blank_and_must_not_leak_answer(self) -> None:
        with pytest.raises(DraftRejected):
            self.validator.validate(draft(question="Glycolysis happens where?", answer="cytoplasm"), QuestionType.FILL_IN_BLANK, [CHUNK])
        result = self.validator.validate(
            draft(question="Glycolysis takes place in the ___.", answer="cytoplasm"), QuestionType.FILL_IN_BLANK, [CHUNK]
        )
        assert BLANK in result.question

    def test_free_form_answers_must_be_supported_by_context(self) -> None:
        with pytest.raises(DraftRejected):
            self.validator.validate(
                draft(question="Why is glycolysis important?", answer="Quantum tunnelling powers starlight", key_points=["x"]),
                QuestionType.CONCEPTUAL,
                [CHUNK],
            )

    def test_true_false_answer_is_normalised(self) -> None:
        result = self.validator.validate(draft(question=EVIDENCE, answer="TRUE"), QuestionType.TRUE_FALSE, [CHUNK])
        assert result.answer == "True"


class TestLLMEngine:
    def test_produces_structured_grounded_question(self) -> None:
        engine = LLMQuestionEngine(StructuredLLM(FakeGroundedLLM()), temperature=0.3)
        question = engine.generate(request(QuestionType.FILL_IN_BLANK))
        for field in ("question", "question_type", "difficulty", "expected_answer", "explanation", "source_chunks", "source_pages", "topic"):
            assert getattr(question, field) not in (None, "", [])
        assert question.source_pages == [2]
        assert question.sources[0].chunk_id == CHUNK.chunk_id
        assert question.expected_answer.lower() in TEXT.lower()

    def test_prompt_contains_only_retrieved_context(self) -> None:
        fake = FakeGroundedLLM()
        LLMQuestionEngine(StructuredLLM(fake), 0.3).generate(request(QuestionType.SHORT_ANSWER))
        prompt = fake.calls[0]["user"]
        assert "[chunk_id: d-p2-c1 | page 2]" in prompt
        assert "ONLY facts stated in the CONTEXT" in fake.calls[0]["system"]

    def test_insufficient_context_is_reported(self) -> None:
        fake = FakeGroundedLLM(override=lambda s, u: '{"insufficient_context": true}')
        with pytest.raises(InsufficientContextError):
            LLMQuestionEngine(StructuredLLM(fake), 0.3).generate(request(QuestionType.CALCULATION))

    def test_hallucinated_question_is_rejected(self) -> None:
        bad = '{"question": "What colour is a Martian sky?", "answer": "true", "explanation": "x", "evidence_quote": "Mars has butterscotch skies."}'
        fake = FakeGroundedLLM(override=lambda s, u: bad)
        with pytest.raises(LLMResponseError):
            LLMQuestionEngine(StructuredLLM(fake), 0.3).generate(request(QuestionType.TRUE_FALSE))


class TestExtractiveEngine:
    engine = ExtractiveQuestionEngine()

    @pytest.mark.parametrize("qtype", [QuestionType.FILL_IN_BLANK, QuestionType.MULTIPLE_CHOICE, QuestionType.TRUE_FALSE])
    def test_questions_are_built_from_document_sentences(self, qtype) -> None:
        question = self.engine.generate(request(qtype))
        assert question.evidence in TEXT
        assert question.source_pages == [2]
        if qtype == QuestionType.MULTIPLE_CHOICE:
            assert question.expected_answer in question.options and len(set(question.options)) == 4
        if qtype == QuestionType.FILL_IN_BLANK:
            assert BLANK in question.question
            assert question.question.replace(BLANK, question.expected_answer) == question.evidence

    def test_unsupported_types_are_refused(self) -> None:
        with pytest.raises(InsufficientContextError):
            self.engine.generate(request(QuestionType.CONCEPTUAL))

    def test_variants_choose_different_sentences(self) -> None:
        first = self.engine.generate(request(QuestionType.FILL_IN_BLANK, variant=0))
        second = self.engine.generate(request(QuestionType.FILL_IN_BLANK, variant=1))
        assert first.evidence != second.evidence


class TestQuestionService:
    def plan(self, qtype=QuestionType.FILL_IN_BLANK, topic_id=0) -> QuestionPlan:
        return QuestionPlan(topic_id, qtype, Difficulty.EASY, reinforce=False, rationale="test")

    def test_generates_non_duplicate_questions(self, llm_container, sample_pdf) -> None:
        document_id = ingest(llm_container, sample_pdf)
        topics = llm_container.registry.get(document_id).topics
        chunks = llm_container.store.get_chunks(document_id)
        session = Session(session_id="s1", document_id=document_id, target_questions=5,
                          allowed_types=list(QuestionType), difficulty=Difficulty.EASY)
        service = llm_container.questions
        engine = service.active_engine()
        first = service.generate(session, self.plan(), topics, chunks, engine)
        session.answers.append(AnswerRecord(question=first, user_answer="x",
                                            evaluation=Evaluation(verdict=Verdict.CORRECT, score=1, feedback="", evaluated_by="t")))
        session.seed_usage[first.seed_chunk_id] = 1
        second = service.generate(session, self.plan(), topics, chunks, engine)
        assert second.question != first.question
        assert second.evidence != first.evidence

    def test_llm_outage_propagates_for_fallback(self, settings, sample_pdf) -> None:
        def down(system, user):
            raise LLMUnavailableError("down")

        container = build_container(settings, llm_client=FakeGroundedLLM(override=down))
        document_id = ingest(container, sample_pdf)
        session = Session(session_id="s2", document_id=document_id, target_questions=3,
                          allowed_types=list(QuestionType), difficulty=Difficulty.EASY)
        with pytest.raises(LLMUnavailableError):
            container.questions.generate(session, self.plan(), container.registry.get(document_id).topics,
                                         container.store.get_chunks(document_id), container.questions.active_engine())
        container.questions.mark_llm_unavailable()
        assert container.questions.active_engine().name == "extractive"


SLIDE_TEXT = (
    "Supervised Learning\n"
    "• Learns a mapping from inputs to labelled outputs\n"
    "• Common algorithms include linear regression and decision trees\n"
    "• Requires a large labelled training dataset\n"
    "Unsupervised Learning\n"
    "- Finds hidden structure in unlabelled data using clustering methods"
)


def test_extractive_engine_handles_slide_bullets() -> None:
    chunk = Chunk(chunk_id="s-p1-c0", document_id="s", page_number=1, chunk_index=0, text=SLIDE_TEXT,
                  char_start=0, char_end=len(SLIDE_TEXT))
    engine = ExtractiveQuestionEngine()
    for qtype in (QuestionType.FILL_IN_BLANK, QuestionType.MULTIPLE_CHOICE, QuestionType.TRUE_FALSE):
        question = engine.generate(GenerationRequest(qtype, Difficulty.EASY, 0, "learning", chunk, [chunk]))
        assert "•" not in question.question and "Unsupervised Learning" not in question.evidence
        assert question.evidence in SLIDE_TEXT
