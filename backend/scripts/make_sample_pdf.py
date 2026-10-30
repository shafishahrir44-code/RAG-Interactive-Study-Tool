"""Regenerate samples/cell-biology.pdf (run from backend/: python -m scripts.make_sample_pdf)."""

from pathlib import Path

from tests.pdf_factory import make_pdf
from tests.sample_content import SAMPLE_PAGES, SAMPLE_TITLE

OUTPUT = Path(__file__).resolve().parents[2] / "samples" / "cell-biology.pdf"

if __name__ == "__main__":
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(make_pdf(SAMPLE_PAGES, SAMPLE_TITLE))
    print(f"Wrote {OUTPUT}")
