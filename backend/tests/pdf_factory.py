"""Builds small, valid text PDFs for tests without any extra dependency."""

from __future__ import annotations

import textwrap

LINE_WIDTH_CHARS = 90
LINE_HEIGHT = 14


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _content_stream(page_text: str) -> bytes:
    lines: list[str] = []
    for paragraph in page_text.split("\n"):
        lines.extend(textwrap.wrap(paragraph, LINE_WIDTH_CHARS) or [""])
    ops = ["BT", "/F1 10 Tf", f"{LINE_HEIGHT} TL", "50 780 Td"]
    ops += [f"({_escape(line)}) Tj T*" for line in lines]
    ops.append("ET")
    return "\n".join(ops).encode("latin-1", "replace")


def make_pdf(pages: list[str], title: str | None = None) -> bytes:
    """Return PDF bytes with one page per string ('' produces a blank page)."""
    objects: list[bytes] = []
    page_count = len(pages)
    font_id = 3
    first_page_id = 4
    kids = " ".join(f"{first_page_id + 2 * i} 0 R" for i in range(page_count))
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for i, text in enumerate(pages):
        content_id = first_page_id + 2 * i + 1
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>".encode()
        )
        stream = _content_stream(text)
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    info_id = None
    if title:
        objects.append(f"<< /Title ({_escape(title)}) >>".encode())
        info_id = len(objects)

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    trailer = f"<< /Size {len(objects) + 1} /Root 1 0 R"
    if info_id:
        trailer += f" /Info {info_id} 0 R"
    out += f"trailer\n{trailer} >>\nstartxref\n{xref_at}\n%%EOF\n".encode()
    return bytes(out)
