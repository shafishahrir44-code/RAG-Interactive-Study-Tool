"""Test doubles for the LLM layer.

`FakeGroundedLLM` behaves like a well-behaved model: it only sees the prompt, parses the
retrieved context blocks out of it, and answers with JSON built from that context. It lets
tests exercise the real prompt -> structured output -> validation -> grading path offline.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from app.text_utils import split_sentences

_BLOCK_RE = re.compile(r"\[chunk_id: (?P<id>[^|\]]+) \| page (?P<page>\d+)\]\n(?P<text>.*?)(?=\n\n\[chunk_id:|\n\nTASK:)", re.S)
_TYPE_HINTS = {
    "multiple-choice": "multiple_choice",
    "true or false": "true_false",
    "replace a key term": "fill_in_blank",
    "one or two sentences": "short_answer",
    "'why' or 'how'": "conceptual",
    "requires a calculation": "calculation",
}


def parse_context(prompt: str) -> list[tuple[str, str]]:
    return [(m["id"].strip(), m["text"].strip()) for m in _BLOCK_RE.finditer(prompt)]


def _question_type(prompt: str) -> str:
    for hint, qtype in _TYPE_HINTS.items():
        if hint in prompt:
            return qtype
    raise AssertionError("unrecognised task in prompt")


def _key_word(sentence: str) -> str:
    words = [w.strip(".,;:") for w in sentence.split() if len(w.strip(".,;:")) >= 6 and w.isalpha()]
    return max(words, key=len) if words else sentence.split()[0]


class FakeGroundedLLM:
    name = "fake:grounded"

    def __init__(self, override: Callable[[str, str], str | None] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.override = override  # return a raw string to simulate misbehaviour

    def is_available(self) -> bool:
        return True

    def complete_json(self, system: str, user: str, schema: dict[str, Any], temperature: float) -> str:
        self.calls.append({"system": system, "user": user, "schema": schema})
        if self.override and (raw := self.override(system, user)) is not None:
            return raw
        if "LEARNER ANSWER" in user:
            return json.dumps(self._grade(user))
        return json.dumps(self._question(user))

    def _question(self, prompt: str) -> dict[str, Any]:
        qtype = _question_type(prompt)
        chunk_id, text = parse_context(prompt)[0]
        sentences = [s for s in split_sentences(text) if len(s.split()) >= 6]
        avoid = prompt.split("Do NOT repeat", 1)[1] if "Do NOT repeat" in prompt else ""
        sentence = next((s for s in sentences if s[:40] not in avoid), sentences[0])
        word = _key_word(sentence)
        base = {"evidence_quote": sentence, "source_chunk_ids": [chunk_id], "explanation": f"The document says: {sentence}"}
        if qtype == "multiple_choice":
            return {**base, "question": f"Which word completes: {sentence.replace(word, '_____')}",
                    "options": [word, "alpha", "beta", "gamma"], "answer": word}
        if qtype == "true_false":
            return {**base, "question": sentence, "answer": "true"}
        if qtype == "fill_in_blank":
            return {**base, "question": sentence.replace(word, "_____"), "answer": word}
        if qtype == "calculation":
            return {**base, "question": f"Using the document, compute the value described in: {sentence}",
                    "answer": "46", "key_points": ["double 23 to get 46"], "evidence_quote": sentence}
        return {**base, "question": f"Explain this idea from the document: {sentence[:60]}?",
                "answer": sentence, "key_points": [sentence]}

    @staticmethod
    def _grade(prompt: str) -> dict[str, Any]:
        reference = prompt.split("REFERENCE ANSWER:\n", 1)[1].split("\n\n", 1)[0]
        learner = prompt.split("LEARNER ANSWER:\n", 1)[1]
        ref_words = set(reference.lower().split())
        overlap = len(ref_words & set(learner.lower().split())) / max(len(ref_words), 1)
        if overlap >= 0.7:
            return {"verdict": "correct", "score": 0.95, "correct_points": [reference], "feedback": "You covered it."}
        if overlap >= 0.3:
            return {"verdict": "partially_correct", "score": 0.5, "missing_points": [reference],
                    "feedback": "You covered part of it."}
        return {"verdict": "incorrect", "score": 0.05, "missing_points": [reference], "feedback": "You missed the key idea."}
