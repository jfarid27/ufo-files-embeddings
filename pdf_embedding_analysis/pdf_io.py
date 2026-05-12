"""Discover PDFs and extract plain text with PyMuPDF."""

from __future__ import annotations

from pathlib import Path

import fitz


def list_pdf_paths(pdf_dir: Path, *, debug: bool = False) -> list[Path]:
    """Return sorted PDF paths under ``pdf_dir`` (non-recursive)."""
    if not pdf_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {pdf_dir}")
    paths = sorted(p for p in pdf_dir.iterdir() if p.suffix.lower() == ".pdf" and p.is_file())
    if debug:
        print(f"[list_pdf_paths] Found {len(paths)} PDF(s) in {pdf_dir}")
        for p in paths:
            print(f"  - {p.name}")
    return paths


def extract_text_from_pdf(pdf_path: Path, *, debug: bool = False) -> str:
    """Extract concatenated text from all pages of a PDF."""
    if debug:
        print(f"[extract_text_from_pdf] Opening {pdf_path}")
    doc = fitz.open(pdf_path)
    try:
        parts: list[str] = []
        for page_index, page in enumerate(doc):
            page_text = page.get_text("text") or ""
            parts.append(page_text)
            if debug:
                print(f"  page {page_index + 1}: {len(page_text)} chars")
        text = "\n".join(parts)
    finally:
        doc.close()
    if debug:
        print(f"[extract_text_from_pdf] Total length for {pdf_path.name}: {len(text)} chars")
    return text
