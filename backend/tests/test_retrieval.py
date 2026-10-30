from __future__ import annotations

import numpy as np

from app.config import RetrievalSettings
from app.embeddings.hashing import HashingEmbedder
from app.ingestion.models import Chunk
from app.retrieval.lexical import BM25Index
from app.retrieval.retriever import Retriever
from app.retrieval.topics import build_topics, choose_topic_count, is_numeric_chunk
from app.text_utils import split_statements
from app.vectorstore.numpy_store import NumpyVectorStore
from tests.conftest import ingest

DOC = "feedc0ffee1234"


def make_chunks(texts: list[str]) -> list[Chunk]:
    return [
        Chunk(chunk_id=f"c{i}", document_id=DOC, page_number=i + 1, chunk_index=i, text=t, char_start=0, char_end=len(t))
        for i, t in enumerate(texts)
    ]


TEXTS = [
    "Photosynthesis in chloroplasts converts light energy into glucose.",
    "Mitochondria perform cellular respiration and produce ATP from glucose.",
    "The French Revolution began in 1789 and transformed European politics.",
]


def test_hash_embedder_is_deterministic_and_normalised() -> None:
    embedder = HashingEmbedder(256)
    a = embedder.embed_documents(TEXTS)
    b = embedder.embed_documents(TEXTS)
    assert np.allclose(a, b)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
    assert embedder.embed_query("mitochondria atp") @ a[1] > embedder.embed_query("mitochondria atp") @ a[2]


def test_vector_store_persists_search_and_checks_fingerprint(tmp_path) -> None:
    embedder = HashingEmbedder(256)
    chunks = make_chunks(TEXTS)
    store = NumpyVectorStore(tmp_path)
    store.upsert(DOC, chunks, embedder.embed_documents(TEXTS), embedder.fingerprint)

    reopened = NumpyVectorStore(tmp_path)  # fresh instance reads from disk
    assert reopened.has_index(DOC, embedder.fingerprint)
    assert not reopened.has_index(DOC, "another-model")
    hits = reopened.search(DOC, embedder.embed_query("French Revolution 1789"), k=2)
    assert hits[0].chunk.chunk_id == "c2"
    assert hits[0].chunk.page_number == 3  # metadata survives the round trip
    assert hits[0].score >= hits[1].score


def test_bm25_prefers_documents_with_rare_query_terms() -> None:
    index = BM25Index(make_chunks(TEXTS))
    assert index.score("chloroplasts", "c0") > index.score("chloroplasts", "c1") == 0.0


def test_retriever_returns_relevant_chunks_with_pages(extractive_container, sample_pdf) -> None:
    document_id = ingest(extractive_container, sample_pdf)
    results = extractive_container.retriever.retrieve(document_id, "Where does photosynthesis happen? chloroplasts", k=2)
    assert results[0].chunk.page_number == 4
    assert all(r.chunk.document_id == document_id for r in results)

    results = extractive_container.retriever.retrieve(document_id, "How many chromosomes in a human gamete", k=1)
    assert results[0].chunk.page_number == 5


def test_context_for_seed_starts_with_seed_and_respects_budget(tmp_path) -> None:
    embedder = HashingEmbedder(256)
    chunks = make_chunks(TEXTS)
    store = NumpyVectorStore(tmp_path)
    store.upsert(DOC, chunks, embedder.embed_documents(TEXTS), embedder.fingerprint)
    retriever = Retriever(store, embedder, RetrievalSettings(top_k=3, max_context_chars=150))
    context = retriever.context_for_seed(chunks[1])
    assert context[0].chunk_id == "c1"
    assert sum(len(c.text) for c in context) <= 150 or len(context) == 1
    assert len({c.chunk_id for c in context}) == len(context)


def test_topics_cover_every_chunk_deterministically() -> None:
    texts = TEXTS * 3
    chunks = [c.model_copy(update={"chunk_id": f"c{i}", "chunk_index": i}) for i, c in enumerate(make_chunks(texts))]
    vectors = HashingEmbedder(256).embed_documents(texts)
    topics_a, labelled = build_topics(chunks, vectors)
    topics_b, _ = build_topics(chunks, vectors)
    assert [t.model_dump() for t in topics_a] == [t.model_dump() for t in topics_b]
    assert sorted(cid for t in topics_a for cid in t.chunk_ids) == sorted(c.chunk_id for c in chunks)
    assert all(c.topic_id is not None for c in labelled)
    # identical texts end up in the same topic
    by_text: dict[str, set[int]] = {}
    for chunk in labelled:
        by_text.setdefault(chunk.text, set()).add(chunk.topic_id)
    assert all(len(ids) == 1 for ids in by_text.values())


def test_topic_helpers() -> None:
    assert choose_topic_count(1) == 1
    assert choose_topic_count(1000) == 8
    assert is_numeric_chunk("A body cell has 46 chromosomes and a gamete 23, about 2 sets.")
    assert not is_numeric_chunk("No numbers here at all.")


def test_split_statements_separates_bullets_but_joins_wrapped_prose() -> None:
    slide = "Key Ideas\n• Mitochondria produce ATP\n• Ribosomes build proteins"
    assert split_statements(slide) == ["Key Ideas", "Mitochondria produce ATP", "Ribosomes build proteins"]
    prose = "Cells are the basic structural and functional units of all\nliving organisms. New cells arise\nfrom existing cells."
    assert split_statements(prose) == [
        "Cells are the basic structural and functional units of all living organisms.",
        "New cells arise from existing cells.",
    ]
