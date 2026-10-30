"""Deterministic, LLM-free question engine.

Builds cloze-style questions (fill-in-the-blank, multiple choice, true/false) directly from
sentences in the retrieved context. Every question is a verbatim document sentence with one
key term removed or swapped, so it is grounded by construction. Used when no LLM is
configured or the LLM is unreachable.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from app.errors import InsufficientContextError
from app.ingestion.models import Chunk
from app.questions.engine import GenerationRequest, build_question
from app.questions.models import BLANK, Difficulty, Question, QuestionType
from app.text_utils import STOPWORDS, normalize_answer, split_statements, tokenize

ENGINE_NAME = "extractive"
SENTENCE_WORDS = (6, 45)  # slide bullets are often short statements
MIN_TERM_CHARS = 4
DISTRACTOR_COUNT = 3
DEFINITION_CUES = (" is ", " are ", " refers to ", " called ", " means ", " defined as ", " consists of ")
SALIENCE_CAP = 3
# Function-like words that pass the stopword filter but make obviously wrong options.
WEAK_DISTRACTORS = frozenset(
    "alone always another because called contains during either every first having including "
    "instead later least mainly most mostly never often only other place rather second several "
    "since still their then therefore these thing things those through together usually whereas "
    "which while within without".split()
)
_WORD_RE =re.compile(r"[A-Za-z][A-Za-z\-]+")


@dataclass(frozen=True)
class Cloze:
    sentence: str
    term: str  # surface form as it appears in the sentence
    chunk: Chunk

    @property
    def blanked(self) -> str:
        return re.sub(rf"\b{re.escape(self.term)}\b", BLANK, self.sentence, flags=re.IGNORECASE)


def _candidate_terms(text: str) -> list[str]:
    seen: dict[str, str] = {}
    for word in _WORD_RE.findall(text):
        key = word.lower()
        if len(key) >= MIN_TERM_CHARS and key not in STOPWORDS and key not in seen:
            seen[key] = word
    return list(seen.values())


def _stable_hash(text: str) -> int:
    return int(hashlib.sha1(text.encode()).hexdigest()[:8], 16)


class ExtractiveQuestionEngine:
    name = ENGINE_NAME
    supported_types = frozenset({QuestionType.FILL_IN_BLANK, QuestionType.MULTIPLE_CHOICE, QuestionType.TRUE_FALSE})

    def generate(self, request: GenerationRequest) -> Question:
        if request.question_type not in self.supported_types:
            raise InsufficientContextError(
                "This question type needs an LLM.", detail=f"type={request.question_type.value}"
            )
        context_terms = _candidate_terms(" ".join(c.text for c in request.context))
        clozes = self._clozes(request, context_terms)
        for offset in range(len(clozes)):
            cloze = clozes[(request.variant + offset) % len(clozes)]
            if request.question_type == QuestionType.FILL_IN_BLANK:
                return self._fill_in_blank(request, cloze)
            distractors = self._distractors(cloze, context_terms, request.context)
            if len(distractors) >= DISTRACTOR_COUNT and request.question_type == QuestionType.MULTIPLE_CHOICE:
                return self._multiple_choice(request, cloze, distractors)
            if distractors and request.question_type == QuestionType.TRUE_FALSE:
                return self._true_false(request, cloze, distractors[0])
        raise InsufficientContextError(
            "Not enough suitable sentences in this passage.", detail=f"seed={request.seed.chunk_id}"
        )

    # ----- candidate selection -------------------------------------------------------------

    def _clozes(self, request: GenerationRequest, context_terms: list[str]) -> list[Cloze]:
        context_tokens = tokenize(" ".join(c.text for c in request.context))
        frequency = {t.lower(): context_tokens.count(t.lower()) for t in context_terms}
        clozes: list[Cloze] = []
        for sentence in split_statements(request.seed.text):
            words = len(sentence.split())
            if not SENTENCE_WORDS[0] <= words <= SENTENCE_WORDS[1]:
                continue
            terms = _candidate_terms(sentence)
            if not terms:
                continue
            clozes.append(Cloze(sentence, self._pick_term(terms, frequency, request.difficulty), request.seed))
        return sorted(clozes, key=lambda c: self._sentence_rank(c, request.difficulty))

    @staticmethod
    def _pick_term(terms: list[str], frequency: dict[str, int], difficulty: Difficulty) -> str:
        """Easy: the most repeated (central) term. Hard: the rarest, longest term."""
        if difficulty == Difficulty.EASY:
            return max(terms, key=lambda t: (frequency.get(t.lower(), 0), len(t)))
        if difficulty == Difficulty.HARD:
            return max(terms, key=lambda t: (-frequency.get(t.lower(), 0), len(t)))
        return max(terms, key=len)

    @staticmethod
    def _sentence_rank(cloze: Cloze, difficulty: Difficulty) -> tuple:
        lowered = f" {cloze.sentence.lower()} "
        is_definition = any(cue in lowered for cue in DEFINITION_CUES)
        length = len(cloze.sentence.split())
        if difficulty == Difficulty.HARD:
            return (-length, cloze.sentence)
        return (not is_definition, length, cloze.sentence)

    @staticmethod
    def _distractors(cloze: Cloze, context_terms: list[str], context: list[Chunk]) -> list[str]:
        """Pick plausible wrong terms: salient context terms of the same surface form."""
        answer = cloze.term.lower()
        sentence_tokens = set(tokenize(cloze.sentence))
        context_tokens = tokenize(" ".join(c.text for c in context))

        def plausibility(term: str) -> tuple:
            lowered = term.lower()
            return (
                lowered.endswith("s") != answer.endswith("s"),  # plural matches plural
                lowered[-2:] != answer[-2:],  # same ending reads as the same kind of word
                -min(context_tokens.count(lowered), SALIENCE_CAP),  # repeated terms are key concepts
                abs(len(term) - len(answer)),
                lowered,
            )

        candidates = [
            t
            for t in context_terms
            if t.lower() not in sentence_tokens
            and t.lower()[:5] != answer[:5]  # skip variants of the answer
            and t.lower() not in WEAK_DISTRACTORS
        ]
        distractors = sorted(candidates, key=plausibility)
        # Swapped statements must not accidentally be true elsewhere in the context.
        full_context = normalize_answer(" ".join(c.text for c in context))
        return [
            d
            for d in distractors
            if normalize_answer(cloze.blanked.replace(BLANK, d)) not in full_context
        ][:DISTRACTOR_COUNT]

    # ----- question builders ---------------------------------------------------------------

    def _evidence_text(self, cloze: Cloze) -> str:
        return f"Page {cloze.chunk.page_number} of the document states: “{cloze.sentence}”"

    def _fill_in_blank(self, request: GenerationRequest, cloze: Cloze) -> Question:
        return build_question(
            request,
            question=cloze.blanked,
            expected_answer=cloze.term,
            explanation=f"{self._evidence_text(cloze)} The missing term is “{cloze.term}”.",
            evidence=cloze.sentence,
            source_chunks=[cloze.chunk],
            generated_by=self.name,
        )

    def _multiple_choice(self, request: GenerationRequest, cloze: Cloze, distractors: list[str]) -> Question:
        options = sorted([cloze.term, *distractors], key=lambda o: _stable_hash(cloze.sentence + o))
        return build_question(
            request,
            question=f"Which term correctly completes this statement from the document?\n“{cloze.blanked}”",
            expected_answer=cloze.term,
            options=options,
            explanation=f"{self._evidence_text(cloze)} So the correct term is “{cloze.term}”.",
            evidence=cloze.sentence,
            source_chunks=[cloze.chunk],
            generated_by=self.name,
        )

    def _true_false(self, request: GenerationRequest, cloze: Cloze, distractor: str) -> Question:
        is_true = _stable_hash(cloze.sentence) % 2 == 0
        statement = cloze.sentence if is_true else cloze.blanked.replace(BLANK, distractor)
        verdict = (
            "The statement matches the document, so it is true."
            if is_true
            else f"The document says “{cloze.term}”, not “{distractor}”, so the statement is false."
        )
        return build_question(
            request,
            question=f"True or false: {statement}",
            expected_answer="True" if is_true else "False",
            explanation=f"{self._evidence_text(cloze)} {verdict}",
            evidence=cloze.sentence,
            source_chunks=[cloze.chunk],
            generated_by=self.name,
        )
