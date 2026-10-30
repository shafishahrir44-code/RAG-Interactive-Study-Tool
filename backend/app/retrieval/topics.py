"""Deterministic topic discovery: k-means over chunk embeddings, labelled with TF-IDF keywords.

Topics give the adaptive engine stable units to track mastery and coverage against.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
from pydantic import BaseModel

from app.ingestion.models import Chunk
from app.text_utils import GENERIC_WORDS, content_tokens

MAX_TOPICS = 8
CHUNKS_PER_TOPIC = 2  # target average cluster size for small documents
KMEANS_ITERATIONS = 25
LABEL_KEYWORDS = 3
MIN_NUMBERS_FOR_CALCULATION = 3
_NUMBER_RE = re.compile(r"(?<![A-Za-z])\d+(?:[.,]\d+)?")


class Topic(BaseModel):
    topic_id: int
    label: str
    keywords: list[str]
    chunk_ids: list[str]
    pages: list[int]
    has_numeric_content: bool


def is_numeric_chunk(text: str) -> bool:
    return len(_NUMBER_RE.findall(text)) >= MIN_NUMBERS_FOR_CALCULATION


def choose_topic_count(chunk_count: int) -> int:
    return max(1, min(MAX_TOPICS, math.ceil(chunk_count / CHUNKS_PER_TOPIC)))


def kmeans(vectors: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Cosine k-means with k-means++ initialisation and a fixed seed (reproducible)."""
    n = len(vectors)
    k = min(k, n)
    rng = np.random.default_rng(seed)
    centroids = [vectors[0]]  # deterministic first centre: the opening chunk
    for _ in range(1, k):
        distances = np.min(1 - np.stack(centroids) @ vectors.T, axis=0).clip(min=0)
        total = distances.sum()
        if total <= 0:
            break
        centroids.append(vectors[rng.choice(n, p=distances / total)])
    centers = np.stack(centroids)
    labels = np.zeros(n, dtype=int)
    for iteration in range(KMEANS_ITERATIONS):
        new_labels = np.argmax(vectors @ centers.T, axis=1)
        if iteration > 0 and np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(len(centers)):
            members = vectors[labels == j]
            if len(members):
                center = members.mean(axis=0)
                centers[j] = center / (np.linalg.norm(center) or 1.0)
    return labels


def _label_terms(text: str) -> list[str]:
    return [
        t for t in content_tokens(text) if not t.isdigit() and len(t) > 2 and "'" not in t and t not in GENERIC_WORDS
    ]


def _keywords(
    cluster_chunks: list[Chunk], doc_freq: Counter[str], doc_counts: Counter[str], total_chunks: int
) -> list[str]:
    """Terms that are frequent in this cluster and comparatively rare elsewhere in the document."""
    term_counts: Counter[str] = Counter()
    for chunk in cluster_chunks:
        term_counts.update(_label_terms(chunk.text))
    scored = {
        term: count * math.log(1 + total_chunks / doc_freq[term]) * (count / doc_counts[term])
        for term, count in term_counts.items()
        if count > 1 or len(term_counts) < LABEL_KEYWORDS * 2  # one-off words make poor labels
    }
    chosen: list[str] = []
    for term, _ in sorted(scored.items(), key=lambda kv: (-kv[1], kv[0])):
        if term.rstrip("s") not in {c.rstrip("s") for c in chosen}:  # "cell" and "cells" are one keyword
            chosen.append(term)
        if len(chosen) == LABEL_KEYWORDS:
            break
    return chosen


def build_topics(chunks: list[Chunk], vectors: np.ndarray) -> tuple[list[Topic], list[Chunk]]:
    """Cluster chunks into topics. Returns topics (in document order) and chunks with topic_id set."""
    if not chunks:
        return [], []
    labels = kmeans(np.asarray(vectors, dtype=np.float32), choose_topic_count(len(chunks)))

    doc_freq: Counter[str] = Counter()
    doc_counts: Counter[str] = Counter()
    for chunk in chunks:
        terms = _label_terms(chunk.text)
        doc_freq.update(set(terms))
        doc_counts.update(terms)

    groups: dict[int, list[Chunk]] = {}
    for chunk, label in zip(chunks, labels):
        groups.setdefault(int(label), []).append(chunk)
    # Renumber clusters by their first appearance in the document for a natural order.
    ordered_groups = sorted(groups.values(), key=lambda members: members[0].chunk_index)

    topics: list[Topic] = []
    topic_of: dict[str, int] = {}
    for topic_id, members in enumerate(ordered_groups):
        keywords = _keywords(members, doc_freq, doc_counts, len(chunks)) or [f"section {topic_id + 1}"]
        topics.append(
            Topic(
                topic_id=topic_id,
                label=", ".join(keywords),
                keywords=keywords,
                chunk_ids=[c.chunk_id for c in members],
                pages=sorted({c.page_number for c in members}),
                has_numeric_content=any(is_numeric_chunk(c.text) for c in members),
            )
        )
        topic_of.update({c.chunk_id: topic_id for c in members})

    labelled = [chunk.model_copy(update={"topic_id": topic_of[chunk.chunk_id]}) for chunk in chunks]
    return topics, labelled
