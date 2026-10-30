"""Prompt templates for grounded question generation."""

from __future__ import annotations

from app.ingestion.models import Chunk
from app.questions.models import Difficulty, QuestionType

QUESTION_SYSTEM_PROMPT = """You are a careful tutor writing ONE study question from a document.

Grounding rules (mandatory):
- Use ONLY facts stated in the CONTEXT. Never add outside knowledge, even if you are sure it is true.
- The correct answer must be directly supported by the context; copy the supporting sentence(s)
  verbatim into "evidence_quote" and list the ids of the chunks you used in "source_chunk_ids".
- The explanation must justify the answer using the context and mention the page number.
- If the context cannot support a good question of the requested type, return
  {"insufficient_context": true} and nothing else.
- Ask about meaningful concepts, not trivia such as page numbers, figure labels or author names."""

TYPE_INSTRUCTIONS: dict[QuestionType, str] = {
    QuestionType.MULTIPLE_CHOICE: (
        "Write a multiple-choice question with exactly 4 distinct options. Exactly one option is correct; "
        "the 3 distractors must be plausible but clearly wrong according to the context. "
        'Set "answer" to the exact text of the correct option (not a letter).'
    ),
    QuestionType.TRUE_FALSE: (
        "Write a single declarative statement for the learner to judge true or false. "
        "For a false statement, alter one key fact from the context so the context clearly contradicts it. "
        'Set "answer" to "true" or "false". Do not phrase it as a question.'
    ),
    QuestionType.FILL_IN_BLANK: (
        "Take one important sentence from the context and replace a key term (1-3 words) with _____ . "
        'Set "answer" to the removed words exactly as they appear in the context. '
        "The blank must be inferable from understanding the material, not a trivial word."
    ),
    QuestionType.SHORT_ANSWER: (
        "Write a question answerable in one or two sentences. "
        'Set "answer" to a concise model answer and "key_points" to the 1-3 facts a correct answer needs.'
    ),
    QuestionType.CONCEPTUAL: (
        "Write a 'why' or 'how' question that tests understanding of a concept or relationship "
        "explained in the context (not mere recall). "
        'Set "answer" to a model answer and "key_points" to the 2-4 ideas a complete answer must include.'
    ),
    QuestionType.CALCULATION: (
        "Write a problem that requires a calculation using numbers, formulas or quantities given in the context. "
        "Every value needed must appear in the question or the context. "
        'Set "answer" to the final numeric result with units, "explanation" to the worked steps, '
        'and "key_points" to the method steps and the final value.'
    ),
}

DIFFICULTY_INSTRUCTIONS: dict[Difficulty, str] = {
    Difficulty.EASY: "Difficulty EASY: test recall of a single, clearly stated fact or definition.",
    Difficulty.MEDIUM: "Difficulty MEDIUM: require understanding or connecting two related facts.",
    Difficulty.HARD: (
        "Difficulty HARD: require applying, comparing or reasoning across several facts in the context."
    ),
}


def format_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[chunk_id: {c.chunk_id} | page {c.page_number}]\n{c.text}" for c in chunks)


def question_user_prompt(
    question_type: QuestionType,
    difficulty: Difficulty,
    topic_label: str,
    context: list[Chunk],
    avoid_questions: list[str],
) -> str:
    parts = [
        f"CONTEXT:\n{format_context(context)}",
        f"TASK: {TYPE_INSTRUCTIONS[question_type]}",
        DIFFICULTY_INSTRUCTIONS[difficulty],
        f"Focus on the topic of the first chunk (keywords: {topic_label}).",
    ]
    if avoid_questions:
        listed = "\n".join(f"- {q}" for q in avoid_questions[-8:])
        parts.append(f"Do NOT repeat or paraphrase these earlier questions:\n{listed}")
    return "\n\n".join(parts)
