"""Small, dependency-free text helpers shared by chunking, retrieval and evaluation."""

from __future__ import annotations

import re
import unicodedata

STOPWORDS = frozenset(
    """
    a about above after again against all also am an and any are as at be because been before being
    below between both but by can could did do does doing down during each either else etc few for
    from further had has have having he her here hers herself him himself his how however i if in into
    is it its itself just may me might more most must my myself no nor not now of off on once only or
    other our ours ourselves out over own same shall she should so some such than that the their theirs
    them themselves then there these they this those through thus to too under until up upon us very
    via was we were what when where whether which while who whom why will with within without would
    yet you your yours yourself yourselves one two use used using also many much often well like can
    cannot e.g i.e fig figure table page chapter section see
    """.split()
)

# Everyday verbs and filler adjectives: fine for retrieval, but poor quiz terms and topic labels.
GENERIC_WORDS = frozenset(
    """
    keeps keep holds hold provides provide stores store contains contain makes make gives give takes take
    uses runs moves returns becomes allows allow means mean called calls gets get goes puts sets shows
    describes described includes include provides needs need lets helps refers known based following
    several various certain another different same similar given simple main common important usually
    always often never instead whether while within without other others either each every
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_WHITESPACE_RE = re.compile(r"[ \t ]+")


def normalize_text(text: str) -> str:
    """Unicode-normalise and collapse horizontal whitespace while keeping paragraph breaks."""
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"-\n(?=[a-z])", "", text)  # re-join words hyphenated across line breaks
    lines = [_WHITESPACE_RE.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    """Tokens with stopwords and 1-character tokens removed."""
    return [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 1]


def split_sentences(text: str) -> list[str]:
    flat = re.sub(r"\s*\n\s*", " ", text).strip()
    if not flat:
        return []
    return [s.strip() for s in _SENTENCE_RE.split(flat) if s.strip()]


# Bullet markers, including the private-use Wingdings/Symbol glyphs that PowerPoint PDF exports
# produce (often with no space after them). Numbered items need a following space.
_BULLET_RE = re.compile(
    r"^\s*(?:[\u2022\u2023\u2043\u25a0\u25aa\u25ba\u25cf\u25e6\u2713\u2714\u2756\u27a2\uf0a7\uf0b7\uf0d8\uf0fc\uf076]\s*"
    r"|[\u2013\u2014\-*\u00b7>]\s+|\d{1,2}[.)]\s+)(?=\S)"
)
_LINE_END_RE = re.compile(r"[.!?:;]$")
SHORT_LINE_RATIO = 0.75  # a line this much shorter than the longest line likely ends a bullet or heading


def split_statements(text: str) -> list[str]:
    """Split text into sentences, treating bullet points and headings as separate statements.

    Line breaks inside a wrapped paragraph are joined; a line break ends a statement when the
    previous line ends with punctuation, the next line is a bullet, or a short line is followed
    by a capitalised one (typical of slides).
    """
    # Bare numbers on their own line are slide/page numbers, not content.
    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().isdigit()]
    if not lines:
        return []
    longest = max(len(line) for line in lines)
    blocks: list[str] = []
    current = ""
    previous_line = ""
    for line in lines:
        is_bullet = bool(_BULLET_RE.match(line))
        line = _BULLET_RE.sub("", line)
        starts_new = (
            not current
            or is_bullet
            or _LINE_END_RE.search(current)
            or (line[:1].isupper() and len(previous_line) < SHORT_LINE_RATIO * longest)
        )
        if starts_new and current:
            blocks.append(current)
            current = ""
        current = f"{current} {line}" if current else line
        previous_line = line
    blocks.append(current)
    statements: list[str] = []
    for block in blocks:
        statements.extend(split_sentences(block))
    return statements


def normalize_answer(text: str) -> str:
    """Canonical form for comparing short answers: lowercase, no punctuation or articles."""
    tokens = [t for t in tokenize(text) if t not in {"a", "an", "the"}]
    return " ".join(tokens)


def token_overlap(reference: str, candidate: str) -> float:
    """Fraction of the reference's content tokens that appear in the candidate."""
    ref = set(content_tokens(reference))
    if not ref:
        return 0.0
    return len(ref & set(content_tokens(candidate))) / len(ref)
