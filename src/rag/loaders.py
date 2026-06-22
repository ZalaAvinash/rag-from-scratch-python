"""Load text from PDF files.

Uses pypdf — pure Python, no system deps, works on any platform.
"""
from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


def load_pdf(path: str | Path) -> str:
    """Extract all text from a PDF file.

    Concatenates pages with double newlines as separators. Strips trailing
    whitespace on each page. Empty pages are skipped.

    Args:
        path: Path to the PDF file.

    Returns:
        Full extracted text.

    Raises:
        FileNotFoundError: If the PDF doesn't exist.
        ValueError: If the file isn't a valid PDF.
    """
    pdf_path = Path(path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    chunks: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            chunks.append(text)

    return "\n\n".join(chunks)
