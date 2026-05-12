"""Compose PDF discovery, text extraction, and chunking into structured metadata."""

from __future__ import annotations

from pathlib import Path

from pdf_embedding_analysis.chunking import chunk_text
from pdf_embedding_analysis.pdf_io import extract_text_from_pdf, list_pdf_paths
from pdf_embedding_analysis.types import ChunkMetadata


def build_chunk_metadata_from_pdf_folder(
    pdf_dir: Path,
    max_chunk_chars: int = 256,
    *,
    debug: bool = False,
) -> list[ChunkMetadata]:
    """Walk ``pdf_dir`` for PDFs and produce one metadata row per text chunk."""
    rows: list[ChunkMetadata] = []
    for pdf_path in list_pdf_paths(pdf_dir, debug=debug):
        text = extract_text_from_pdf(pdf_path, debug=debug)
        chunks = chunk_text(text, max_chunk_chars, debug=debug)
        for i, chunk in enumerate(chunks):
            row: ChunkMetadata = {
                "source_pdf": str(pdf_path.resolve()),
                "chunk_index": i,
                "text": chunk,
                "char_length": len(chunk),
            }
            rows.append(row)
    if debug:
        print(f"[build_chunk_metadata_from_pdf_folder] Total chunks: {len(rows)}")
    return rows
